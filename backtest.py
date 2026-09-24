"""
技术指标回测：把提取出的规则（同一规则跨视频合并）放到一组标的上逐根 K 线模拟，
与同期买入持有对比，输出规则库、逐标的结果、交易明细和 Markdown 报告。

- 信号在 K 线收盘时判定，下一根 K 线开盘成交（不偷看未来）
- 只做多，满仓进出，单边扣 COST_PER_SIDE
- 指标用全部历史预热，统计区间从 BACKTEST_START 开始
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import db
import indicators
import market
from targets import Resolver

_WARMUP_BARS = 60
_BARS_PER_YEAR = {"daily": 244, "weekly": 50}


def simulate(df: pd.DataFrame, rule: dict, start_date: str, cost: float) -> Optional[dict]:
    cache: dict = {}
    entry = indicators.evaluate(df, rule["entry"], rule["params"], cache).to_numpy()
    exit_ = indicators.evaluate(df, rule["exit"], rule["params"], cache).to_numpy()
    dates, o, c = df["date"].to_numpy(), df["open"].to_numpy(), df["close"].to_numpy()
    start = max(int(np.searchsorted(dates, np.datetime64(start_date))), _WARMUP_BARS)
    if start >= len(df) - 2:
        return None

    sl, tp, max_hold = rule.get("stop_loss_pct"), rule.get("take_profit_pct"), rule.get("max_hold_bars")
    equity, trades, in_pos = [], [], []
    eq, eq_at_entry, entry_price, entry_i, pos, pending = 1.0, 1.0, 0.0, 0, 0, None
    for i in range(start, len(df)):
        if pending == "buy":
            pos, entry_i, eq_at_entry = 1, i, eq
            entry_price = o[i] * (1 + cost)
        elif pending == "sell":
            ret = o[i] * (1 - cost) / entry_price - 1
            eq = eq_at_entry * (1 + ret)
            trades.append({"entry_date": dates[entry_i], "exit_date": dates[i], "ret": ret, "bars": i - entry_i})
            pos = 0
        pending = None
        equity.append(eq_at_entry * c[i] / entry_price if pos else eq)
        in_pos.append(pos)
        if pos == 0 and entry[i]:
            pending = "buy"
        elif pos == 1:
            r = c[i] / entry_price - 1
            if (
                exit_[i]
                or (sl and r <= -sl / 100)
                or (tp and r >= tp / 100)
                or (max_hold and i - entry_i + 1 >= max_hold)
            ):
                pending = "sell"
    open_trade = None
    if pos:
        open_trade = {"entry_date": dates[entry_i], "exit_date": None, "ret": c[-1] / entry_price - 1,
                      "bars": len(df) - 1 - entry_i}

    eq_s = pd.Series(equity)
    n_years = len(eq_s) / _BARS_PER_YEAR[rule["period"]]
    bh = pd.Series(c[start:] / o[start])
    total, bh_total = eq_s.iloc[-1] - 1, bh.iloc[-1] - 1
    closed = pd.DataFrame(trades)
    return {
        "start": pd.Timestamp(dates[start]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "trades": len(closed),
        "win_rate": float((closed["ret"] > 0).mean()) if len(closed) else np.nan,
        "avg_trade": float(closed["ret"].mean()) if len(closed) else np.nan,
        "avg_bars": float(closed["bars"].mean()) if len(closed) else np.nan,
        "total": total,
        "ann": (1 + total) ** (1 / n_years) - 1 if n_years > 0 and total > -1 else np.nan,
        "mdd": float((eq_s / eq_s.cummax() - 1).min()),
        "exposure": float(np.mean(in_pos)),
        "bh_total": bh_total,
        "bh_ann": (1 + bh_total) ** (1 / n_years) - 1 if n_years > 0 and bh_total > -1 else np.nan,
        "bh_mdd": float((bh / bh.cummax() - 1).min()),
        "holding": bool(pos),
        "pending": pending or "",
        "trade_list": trades + ([open_trade] if open_trade else []),
    }


def _load_rules() -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = db.query_df(
        """SELECT r.*, v.create_time, v.share_url, v.title FROM indicator_rules r
           JOIN videos v ON v.aweme_id = r.aweme_id ORDER BY v.create_time"""
    )
    if rules.empty:
        return rules, rules
    # 按当前规范化逻辑重算 key，规范化规则调整后无需重新调用 LLM
    has_rule = rules["rule_json"].notna()
    normalized = rules.loc[has_rule, "rule_json"].map(lambda s: indicators.normalize(json.loads(s)))
    rules.loc[has_rule, "rule_json"] = normalized.map(lambda r: json.dumps(r, ensure_ascii=False))
    rules.loc[has_rule, "rule_key"] = normalized.map(indicators.rule_key)
    grouped = []
    for key, g in rules[rules["rule_key"].notna()].groupby("rule_key", sort=False):
        targets = []
        for t in g["targets_json"]:
            targets.extend(json.loads(t or "[]"))
        grouped.append({
            "rule_key": key,
            "name": g["name"].mode().iloc[0] if g["name"].duplicated().any() else g["name"].iloc[-1],
            "indicators": g["indicators"].iloc[-1],
            "description": g["description"].iloc[-1],
            "rule": json.loads(g["rule_json"].iloc[0]),
            "targets": list(dict.fromkeys(targets)),
            "mentions": len(g),
            "first_date": g["create_time"].iloc[0][:10],
            "last_date": g["create_time"].iloc[-1][:10],
            "sources": g[["create_time", "share_url", "start_sec", "quote"]].to_dict("records"),
        })
    grouped = pd.DataFrame(grouped).sort_values("mentions", ascending=False, kind="stable").reset_index(drop=True)
    return rules, grouped


def _fmt_ts(sec) -> str:
    if sec is None or pd.isna(sec):
        return "--:--"
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def _pct(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "（无）\n"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in row) + " |")
    return "\n".join(lines) + "\n"


def run() -> None:
    db.init()
    all_rules, rules = _load_rules()
    if rules.empty:
        print("没有可回测的技术指标规则（先运行 extract）")
        return

    resolver = Resolver()
    resolved: dict[str, Optional[tuple]] = {}

    def resolve(name: str):
        if name not in resolved:
            resolved[name] = resolver.resolve(name)
            if resolved[name] is None:
                print(f"  标的无法解析，跳过：{name}")
        return resolved[name]

    results, trades = [], []
    for rule_row in tqdm(rules.itertuples(), total=len(rules), desc="回测规则"):
        rule = rule_row.rule
        demo_codes = {resolve(n)[:2] for n in rule_row.targets if resolve(n)}
        done_codes = set()
        for name in list(dict.fromkeys(rule_row.targets + config.UNIVERSE)):
            hit = resolve(name)
            if not hit or hit[:2] in done_codes:
                continue
            done_codes.add(hit[:2])
            kind, code, std_name = hit
            df = market.daily(kind, code)
            if df is None or df.empty:
                continue
            if rule["period"] == "weekly":
                df = market.to_weekly(df)
            res = simulate(df, rule, config.BACKTEST_START, config.COST_PER_SIDE)
            if not res:
                continue
            for t in res.pop("trade_list"):
                trades.append({"rule_key": rule_row.rule_key, "规则": rule_row.name, "标的": std_name, **t})
            results.append({
                "rule_key": rule_row.rule_key, "规则": rule_row.name, "标的": std_name, "类型": kind,
                "博主演示": (kind, code) in demo_codes, **res,
            })

    res_df = pd.DataFrame(results)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    all_rules.drop(columns=["id"]).to_csv(
        os.path.join(config.OUTPUT_DIR, "indicator_rules.csv"), index=False, encoding="utf-8-sig"
    )
    res_df.to_csv(os.path.join(config.OUTPUT_DIR, "indicator_backtest.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(trades).to_csv(os.path.join(config.OUTPUT_DIR, "indicator_trades.csv"), index=False,
                                encoding="utf-8-sig")
    path = os.path.join(config.OUTPUT_DIR, "indicator_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_report(all_rules, rules, res_df))
    print(f"回测完成：{len(rules)} 条规则 × 标的共 {len(res_df)} 组结果，报告：{path}")


def _report(all_rules: pd.DataFrame, rules: pd.DataFrame, res: pd.DataFrame) -> str:
    parts = ["# 博主技术指标回测报告\n"]
    parts.append(
        f"共提取 {len(all_rules)} 条技术指标用法，其中 {len(rules)} 条可量化回测。"
        f"回测区间自 {config.BACKTEST_START} 起，信号收盘判定、次日开盘成交，单边成本 {config.COST_PER_SIDE:.2%}，"
        "对比基准为同期买入持有。\n"
    )

    if not res.empty:
        parts.append("## 规则总览\n")
        summary = []
        for key, g in res.groupby("rule_key", sort=False):
            excess = g["ann"] - g["bh_ann"]
            summary.append({
                "规则": g["规则"].iloc[0],
                "标的数": len(g),
                "跑赢买入持有": f"{(excess > 0).sum()}/{len(g)}",
                "年化超额中位数": _pct(excess.median()),
                "回撤改善中位数": _pct((g["mdd"] - g["bh_mdd"]).median()),
                "胜率中位数": _pct(g["win_rate"].median()),
                "年均交易次数": f"{(g['trades'] / ((pd.to_datetime(g['end']) - pd.to_datetime(g['start'])).dt.days / 365)).median():.1f}",
                "持仓时间占比": _pct(g["exposure"].median()),
            })
        parts.append(_md_table(pd.DataFrame(summary)))
        parts.append("\n回撤改善为正表示最大回撤比买入持有更小。\n")

    for r in rules.itertuples():
        rule = r.rule
        parts.append(f"\n## {r.name}\n")
        parts.append(f"- 指标：{r.indicators}；周期：{'周线' if rule['period'] == 'weekly' else '日线'}；"
                     f"参数：{json.dumps(rule['params'], ensure_ascii=False) if rule['params'] else '无'}")
        parts.append(f"- 买入：{indicators.describe(rule['entry'])}")
        parts.append(f"- 卖出：{indicators.describe(rule['exit'])}")
        extra = [f"{label}{rule[k]}{unit}" for k, label, unit in
                 (("stop_loss_pct", "止损 ", "%"), ("take_profit_pct", "止盈 ", "%"), ("max_hold_bars", "最多持有 ", " 根"))
                 if rule.get(k)]
        if extra:
            parts.append(f"- 其他：{'；'.join(extra)}")
        parts.append(f"- 博主说法：{r.description}")
        parts.append(f"- 出现 {r.mentions} 次（{r.first_date} ~ {r.last_date}）：")
        for s in r.sources:
            parts.append(f"  - {s['create_time'][:10]} [{_fmt_ts(s['start_sec'])}]({s['share_url']}) {s['quote']}")
        g = res[res["rule_key"] == r.rule_key] if not res.empty else res
        if g.empty:
            parts.append("\n（无可用回测结果）\n")
            continue
        table = pd.DataFrame({
            "标的": g["标的"] + np.where(g["博主演示"], "（演示）", ""),
            "区间": g["start"].str[:7] + "~" + g["end"].str[:7],
            "交易次数": g["trades"],
            "胜率": g["win_rate"].map(_pct),
            "平均单笔": g["avg_trade"].map(_pct),
            "策略年化": g["ann"].map(_pct),
            "持有年化": g["bh_ann"].map(_pct),
            "策略回撤": g["mdd"].map(_pct),
            "持有回撤": g["bh_mdd"].map(_pct),
            "持仓占比": g["exposure"].map(_pct),
            "当前": np.where(g["pending"] == "buy", "明日买入", np.where(g["pending"] == "sell", "明日卖出",
                             np.where(g["holding"], "持有", "空仓"))),
        })
        parts.append("\n" + _md_table(table))

    unsupported = all_rules[all_rules["rule_json"].isna()]
    if not unsupported.empty:
        parts.append("\n## 无法量化的用法\n")
        parts.append(_md_table(pd.DataFrame({
            "日期": unsupported["create_time"].str[:10],
            "名称": unsupported["name"],
            "说法": unsupported["description"],
            "原因": unsupported["unsupported_reason"],
        })))
    return "\n".join(parts)
