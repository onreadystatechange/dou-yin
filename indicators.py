"""
技术指标规则 DSL：计算指标序列，并把规则里的买卖条件求值成布尔信号。

规则格式（extract.py 让大模型按此输出，backtest.py 执行）：
{
  "period": "daily" | "weekly",
  "params": {"macd": {"fast": 12, "slow": 26, "signal": 9}, "kdj": {"n": 9, "m1": 3, "m2": 3}, "boll": {"n": 20, "k": 2}},
  "entry": 条件,
  "exit": 条件,
  "stop_loss_pct": 8 | null, "take_profit_pct": 20 | null, "max_hold_bars": 10 | null
}
条件：{"all": [条件或原子, ...]} / {"any": [...]} / 原子
原子：{"left": 序列, "op": "cross_above|cross_below|>|<|>=|<=", "right": 序列或数字,
       "left_shift": 0, "right_shift": 0, "right_mult": 1, "for_bars": 1}
  - *_shift：取 N 根 K 线之前的值（如 macd_hist > 前一根 macd_hist 表示红柱放大）
  - right_mult：右侧乘以系数（如 volume > 2 * vol_ma5）
  - for_bars：条件需连续成立 N 根
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "kdj": {"n": 9, "m1": 3, "m2": 3},
    "boll": {"n": 20, "k": 2},
}
OPS = {"cross_above", "cross_below", ">", "<", ">=", "<="}
_FIXED = {
    "open", "high", "low", "close", "volume", "pct_chg",
    "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j",
    "boll_upper", "boll_mid", "boll_lower",
    "bull_engulf", "bear_engulf",
}
_PARAM_RE = re.compile(r"^(ma|ema|vol_ma|rsi|hhv|llv)(\d{1,3})$")
_FAMILY = {"macd_": "macd", "kdj_": "kdj", "boll_": "boll"}

SERIES_DOC = """可用序列：
- open / high / low / close / volume / pct_chg（涨跌幅 %）
- maN / emaN：N 周期简单 / 指数均线，如 ma5、ma20、ema12
- vol_maN：N 周期成交量均线
- rsiN：N 周期 RSI，如 rsi6、rsi14
- hhvN / llvN：前 N 根 K 线（不含当根）的最高价 / 最低价，用于突破判断
- macd_dif / macd_dea / macd_hist（MACD 柱，>0 为红柱，<0 为绿柱），参数见 params.macd
- kdj_k / kdj_d / kdj_j，参数见 params.kdj
- boll_upper / boll_mid / boll_lower，参数见 params.boll
- bull_engulf / bear_engulf：阳包阴 / 阴包阳形态，成立为 1，否则为 0"""


def _sma_cn(s: pd.Series, n: int) -> pd.Series:
    """通达信 SMA(X, N, 1)：alpha = 1/N 的递推平均。"""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _series(df: pd.DataFrame, name: str, params: dict, cache: dict) -> pd.Series:
    if name in cache:
        return cache[name]
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    m = _PARAM_RE.match(name)
    if name in ("open", "high", "low", "close", "volume"):
        s = df[name]
    elif name == "pct_chg":
        s = c.pct_change() * 100
    elif m:
        kind, n = m.group(1), int(m.group(2))
        if kind == "ma":
            s = c.rolling(n).mean()
        elif kind == "ema":
            s = c.ewm(span=n, adjust=False).mean()
        elif kind == "vol_ma":
            s = df["volume"].rolling(n).mean()
        elif kind == "rsi":
            diff = c.diff()
            s = _sma_cn(diff.clip(lower=0), n) / _sma_cn(diff.abs(), n) * 100
        elif kind == "hhv":
            s = h.shift(1).rolling(n).max()
        else:
            s = l.shift(1).rolling(n).min()
    elif name.startswith("macd_"):
        p = params["macd"]
        dif = c.ewm(span=p["fast"], adjust=False).mean() - c.ewm(span=p["slow"], adjust=False).mean()
        dea = dif.ewm(span=p["signal"], adjust=False).mean()
        cache.update({"macd_dif": dif, "macd_dea": dea, "macd_hist": 2 * (dif - dea)})
        return cache[name]
    elif name.startswith("kdj_"):
        p = params["kdj"]
        llv, hhv = l.rolling(p["n"]).min(), h.rolling(p["n"]).max()
        rsv = ((c - llv) / (hhv - llv).replace(0, np.nan) * 100).fillna(50)
        k = _sma_cn(rsv, p["m1"])
        d = _sma_cn(k, p["m2"])
        cache.update({"kdj_k": k, "kdj_d": d, "kdj_j": 3 * k - 2 * d})
        return cache[name]
    elif name.startswith("boll_"):
        p = params["boll"]
        mid = c.rolling(p["n"]).mean()
        std = c.rolling(p["n"]).std(ddof=0)
        cache.update({"boll_mid": mid, "boll_upper": mid + p["k"] * std, "boll_lower": mid - p["k"] * std})
        return cache[name]
    elif name == "bull_engulf":
        s = ((c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))).astype(float)
    elif name == "bear_engulf":
        s = ((c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1))).astype(float)
    else:
        raise ValueError(f"未知序列 {name}")
    cache[name] = s
    return s


def _is_series_name(x) -> bool:
    return isinstance(x, str) and (x in _FIXED or bool(_PARAM_RE.match(x)))


def _iter_atoms(cond):
    if isinstance(cond, dict) and ("all" in cond or "any" in cond):
        for sub in cond.get("all") or cond.get("any") or []:
            yield from _iter_atoms(sub)
    else:
        yield cond


def validate(rule: dict) -> Optional[str]:
    """合法返回 None，否则返回原因。"""
    if not isinstance(rule, dict):
        return "规则不是对象"
    if rule.get("period") not in ("daily", "weekly"):
        return "period 必须是 daily 或 weekly"
    for side in ("entry", "exit"):
        if not rule.get(side):
            return f"缺少 {side} 条件"
        for atom in _iter_atoms(rule[side]):
            if not isinstance(atom, dict):
                return f"{side} 条件格式错误"
            if atom.get("op") not in OPS:
                return f"不支持的比较符 {atom.get('op')}"
            if not _is_series_name(atom.get("left")):
                return f"不支持的序列 {atom.get('left')}"
            right = atom.get("right")
            if not (isinstance(right, (int, float)) and not isinstance(right, bool)) and not _is_series_name(right):
                return f"不支持的右值 {right}"
    return None


def normalize(rule: dict) -> dict:
    """补全用到的指标参数、去掉无关字段，保证同一规则的 JSON 一致。"""
    used = set()
    for side in ("entry", "exit"):
        for atom in _iter_atoms(rule[side]):
            for x in (atom.get("left"), atom.get("right")):
                if isinstance(x, str):
                    used.update(fam for prefix, fam in _FAMILY.items() if x.startswith(prefix))
    params = {}
    for fam in sorted(used):
        p = dict(DEFAULT_PARAMS[fam])
        p.update({k: v for k, v in ((rule.get("params") or {}).get(fam) or {}).items() if k in p and v})
        params[fam] = {k: (float(v) if fam == "boll" and k == "k" else int(v)) for k, v in p.items()}

    def clean(cond):
        if isinstance(cond, dict) and ("all" in cond or "any" in cond):
            key = "all" if "all" in cond else "any"
            return {key: [clean(c) for c in cond[key]]}
        atom = {"left": cond["left"], "op": cond["op"], "right": cond["right"]}
        for k, default in (("left_shift", 0), ("right_shift", 0), ("right_mult", 1), ("for_bars", 1)):
            v = cond.get(k)
            if v not in (None, default):
                atom[k] = v
        return atom

    entry, exit_ = clean(rule["entry"]), clean(rule["exit"])
    # “A>B 持有、A<B 离场”与“上穿买、下穿卖”在持有到反向信号的逻辑下等价，统一写法以便跨视频合并
    if (
        set(entry) == set(exit_) == {"left", "op", "right"}
        and (entry["left"], entry["right"]) == (exit_["left"], exit_["right"])
        and entry["op"] in (">", ">=", "cross_above")
        and exit_["op"] in ("<", "<=", "cross_below")
    ):
        entry, exit_ = {**entry, "op": "cross_above"}, {**exit_, "op": "cross_below"}

    out = {"period": rule["period"], "params": params, "entry": entry, "exit": exit_}
    for k in ("stop_loss_pct", "take_profit_pct", "max_hold_bars"):
        if rule.get(k):
            out[k] = rule[k]
    return out


def rule_key(rule: dict) -> str:
    return hashlib.md5(json.dumps(rule, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def evaluate(df: pd.DataFrame, cond, params: dict, cache: Optional[dict] = None) -> pd.Series:
    cache = {} if cache is None else cache
    if isinstance(cond, dict) and ("all" in cond or "any" in cond):
        parts = [evaluate(df, c, params, cache) for c in (cond.get("all") or cond.get("any"))]
        out = parts[0]
        for p in parts[1:]:
            out = (out & p) if "all" in cond else (out | p)
        return out

    left = _series(df, cond["left"], params, cache).shift(cond.get("left_shift", 0))
    right = cond["right"]
    right = (
        _series(df, right, params, cache).shift(cond.get("right_shift", 0))
        if isinstance(right, str)
        else pd.Series(float(right), index=df.index)
    ) * cond.get("right_mult", 1)
    op = cond["op"]
    if op == "cross_above":
        res = (left > right) & (left.shift(1) <= right.shift(1))
    elif op == "cross_below":
        res = (left < right) & (left.shift(1) >= right.shift(1))
    else:
        res = {">": left > right, "<": left < right, ">=": left >= right, "<=": left <= right}[op]
    res = res.fillna(False).astype(bool)
    n = int(cond.get("for_bars", 1) or 1)
    if n > 1:
        res = res.astype(int).rolling(n).sum().eq(n)
    return res


def describe(cond) -> str:
    """把条件翻译成可读文字，用于报告。"""
    if isinstance(cond, dict) and ("all" in cond or "any" in cond):
        key = "all" if "all" in cond else "any"
        joiner = " 且 " if key == "all" else " 或 "
        parts = [describe(c) for c in cond[key]]
        return parts[0] if len(parts) == 1 else "(" + joiner.join(parts) + ")"
    op = {"cross_above": "上穿", "cross_below": "下穿"}.get(cond["op"], cond["op"])
    left = cond["left"] + (f"[前{cond['left_shift']}]" if cond.get("left_shift") else "")
    right = cond["right"]
    right = f"{right}[前{cond['right_shift']}]" if cond.get("right_shift") else str(right)
    if cond.get("right_mult", 1) != 1:
        right = f"{cond['right_mult']}×{right}"
    text = f"{left} {op} {right}"
    if cond.get("for_bars", 1) > 1:
        text += f"（连续{cond['for_bars']}根）"
    return text
