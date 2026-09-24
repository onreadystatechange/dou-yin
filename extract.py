"""
技术指标提取：让 DeepSeek 从逐字稿里找出博主讲的技术指标用法，并尽量翻译成可回测的规则 DSL（见 indicators.py）。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

import config
import db
import indicators

SYSTEM_PROMPT = f"""你是一名量化研究员，负责从股票博主的视频逐字稿中提取他讲到的"技术指标用法"，并翻译成可回测的交易规则。

逐字稿由语音识别生成，可能有同音错别字（如 MACD 被识别成 MICD、KDJ 被识别成 KDG），请结合上下文纠正。
每行格式为 [开始秒-结束秒] 文本。

只提取技术分析方法：指标（MACD、均线、KDJ、RSI、布林带、成交量等）、K 线形态、突破/回踩等价格行为的具体用法。
不要提取：对某只股票/板块/大盘的涨跌判断、消息面、基本面、个人经历、情绪表达。
博主只是顺口提到某指标、没有讲怎么用的，不要提取。

只输出一个 JSON 对象：
{{
  "rules": [
    {{
      "name": "简短规则名，如 MACD 红绿柱波段",
      "indicators": ["MACD"],
      "description": "用大白话完整复述博主的用法：看什么周期、什么参数、什么情况买、什么情况卖、有什么注意事项",
      "quote": "最能体现该用法的原话，可去掉口水词",
      "start_sec": 12.3,
      "end_sec": 45.6,
      "targets": ["科创50"],       // 博主用来演示或建议使用该方法的标的（指数/板块/个股/ETF 名称），没有就空数组
      "rule": 规则对象 或 null,
      "unsupported_reason": "rule 为 null 时说明为什么无法量化，否则为 null"
    }}
  ]
}}

规则对象格式：
{{
  "period": "daily|weekly",                    // 博主用的 K 线周期，没说默认 daily
  "params": {{"macd": {{"fast": 12, "slow": 26, "signal": 9}}}},   // 只写用到的指标；博主说用默认参数就写默认值
  "entry": 条件,                               // 买入条件
  "exit": 条件,                                // 卖出条件；博主没讲卖点时，用与买点对称的反向条件，并在 description 里注明"卖点为推断"
  "stop_loss_pct": null,                       // 博主讲了止损百分比才填，如 8
  "take_profit_pct": null,
  "max_hold_bars": null                        // 博主讲了持有多少根 K 线才填
}}
条件：{{"all": [...]}}（全部成立）/ {{"any": [...]}}（任一成立）/ 原子，可嵌套。
原子：{{"left": 序列, "op": "cross_above|cross_below|>|<|>=|<=", "right": 序列或数字,
       "left_shift": 0, "right_shift": 0, "right_mult": 1, "for_bars": 1}}
  - cross_above/cross_below：本根上穿/下穿（前一根不满足、本根满足）
  - left_shift/right_shift：取 N 根之前的值，如红柱放大：{{"left": "macd_hist", "op": ">", "right": "macd_hist", "right_shift": 1}}
  - right_mult：右值乘系数，如放量一倍：{{"left": "volume", "op": ">", "right": "vol_ma5", "right_mult": 2}}
  - for_bars：连续 N 根成立

{indicators.SERIES_DOC}

翻译示例：
- "MACD 由绿柱翻成红柱时买入，红柱变绿柱卖出" ->
  entry {{"left": "macd_hist", "op": "cross_above", "right": 0}}, exit {{"left": "macd_hist", "op": "cross_below", "right": 0}}
- "MACD 红柱期间持股，绿柱期间空仓" -> 与上面等价，用上穿/下穿 0 表示状态切换
- "股价站上 5 日线买，跌破 5 日线卖" ->
  entry {{"left": "close", "op": "cross_above", "right": "ma5"}}, exit {{"left": "close", "op": "cross_below", "right": "ma5"}}
- "放量突破前 20 日高点" ->
  {{"all": [{{"left": "close", "op": ">", "right": "hhv20"}}, {{"left": "volume", "op": ">", "right": "vol_ma5", "right_mult": 1.5}}]}}

规则要求：
1. 只用上面列出的序列和比较符；无法用它们表达的用法（如画线、波浪理论、筹码分布、主观形态判断），rule 填 null 并写 unsupported_reason。
2. 同一种用法在一段里只输出一条。
3. 没有任何技术指标用法时，rules 返回空数组。
"""


def _client() -> OpenAI:
    if not config.DEEPSEEK_API_KEY:
        raise SystemExit("未配置 DEEPSEEK_API_KEY，请在 .env.local 中填写")
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.LLM_BASE_URL)


def _chunk_sentences(sentences: list[dict]) -> list[str]:
    chunks, buf, size = [], [], 0
    for s in sentences:
        line = f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
        if buf and size + len(line) > config.LLM_CHUNK_CHARS:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _call_llm(client: OpenAI, user_msg: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  调用失败，重试：{e}")
            time.sleep(2 * (attempt + 1))
    return {}


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_rule(r: dict) -> Optional[dict]:
    name = (r.get("name") or "").strip()
    if not name:
        return None
    rule, reason = r.get("rule"), r.get("unsupported_reason")
    if rule is not None:
        err = indicators.validate(rule)
        if err:
            rule, reason = None, f"规则无法解析：{err}"
        else:
            rule = indicators.normalize(rule)
    targets = r.get("targets") or []
    return {
        "name": name,
        "indicators": ",".join(map(str, r.get("indicators") or [])),
        "description": r.get("description", ""),
        "quote": r.get("quote", ""),
        "start_sec": _num(r.get("start_sec")),
        "end_sec": _num(r.get("end_sec")),
        "targets": [str(t) for t in targets if t] if isinstance(targets, list) else [],
        "rule": rule,
        "rule_key": indicators.rule_key(rule) if rule else None,
        "unsupported_reason": None if rule else (reason or "未给出可量化规则"),
    }


def extract_one(client: OpenAI, title: str, sentences: list[dict]) -> list[dict]:
    chunks = _chunk_sentences(sentences)
    rules: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        header = f"视频标题：{title}\n"
        if len(chunks) > 1:
            header += f"（逐字稿第 {i}/{len(chunks)} 段）\n"
        data = _call_llm(client, header + "\n逐字稿：\n" + chunk)
        rules.extend(filter(None, (_normalize_rule(r) for r in data.get("rules") or [] if isinstance(r, dict))))
    # 同一视频里相同规则只保留一条
    seen, out = set(), []
    for r in rules:
        key = r["rule_key"] or r["name"]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def run(limit: Optional[int] = None, redo: bool = False) -> None:
    db.init()
    cond = "" if redo else "AND v.indicators_extracted_at IS NULL"
    with db.connect() as conn:
        rows = conn.execute(
            f"""SELECT v.aweme_id, v.title, t.sentences_json FROM videos v
                JOIN transcripts t ON t.aweme_id = v.aweme_id
                WHERE 1=1 {cond} ORDER BY v.create_time DESC"""
        ).fetchall()
    if limit:
        rows = rows[:limit]
    if not rows:
        print("没有待提取技术指标的视频")
        return

    client = _client()

    def work(row):
        sentences = json.loads(row["sentences_json"])
        return extract_one(client, row["title"], sentences) if sentences else []

    with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as pool:
        futures = {pool.submit(work, row): row for row in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="提取技术指标"):
            row = futures[fut]
            try:
                rules = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  提取失败 {row['aweme_id']}: {e}")
                continue
            _save(row["aweme_id"], rules)


def _save(aweme_id: str, rules: list[dict]) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM indicator_rules WHERE aweme_id=?", (aweme_id,))
        for r in rules:
            conn.execute(
                """INSERT INTO indicator_rules (aweme_id, name, indicators, description, quote, start_sec,
                       end_sec, targets_json, rule_json, rule_key, unsupported_reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    aweme_id, r["name"], r["indicators"], r["description"], r["quote"],
                    r["start_sec"], r["end_sec"], json.dumps(r["targets"], ensure_ascii=False),
                    json.dumps(r["rule"], ensure_ascii=False) if r["rule"] else None,
                    r["rule_key"], r["unsupported_reason"],
                ),
            )
        conn.execute(
            "UPDATE videos SET indicators_extracted_at=? WHERE aweme_id=?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), aweme_id),
        )
    if rules:
        tqdm.write(f"  {aweme_id}: {len(rules)} 条技术指标规则")
