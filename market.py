"""
行情：akshare 日线（按自然日缓存），以及日线合成周线。
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

import akshare as ak
import pandas as pd

import config
from utils import retry

_COLS = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_COLS)
    if "volume" not in df.columns:
        df["volume"] = float("nan")
    df = df[["date", "open", "close", "high", "low", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in ("open", "close", "high", "low", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "close", "high", "low"]).sort_values("date").reset_index(drop=True)


def _fetch(kind: str, code: str, start: str) -> pd.DataFrame:
    end = date.today().strftime("%Y%m%d")
    if kind == "个股":
        return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if kind == "行业板块":
        return ak.stock_board_industry_hist_em(symbol=code, start_date=start, end_date=end, period="日k", adjust="")
    if kind == "概念板块":
        return ak.stock_board_concept_hist_em(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
    if kind == "指数":
        return ak.stock_zh_index_daily(symbol=code)
    raise ValueError(kind)


_memo: dict[tuple[str, str], Optional[pd.DataFrame]] = {}


def daily(kind: str, code: str, start: str = "20100101") -> Optional[pd.DataFrame]:
    """日线（个股前复权），当天首次调用联网，之后读缓存；同一进程内失败的标的不再重试。"""
    if (kind, code) not in _memo:
        _memo[(kind, code)] = _daily(kind, code, start)
    return _memo[(kind, code)]


def _daily(kind: str, code: str, start: str) -> Optional[pd.DataFrame]:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = os.path.join(config.CACHE_DIR, f"daily_{kind}_{code.replace('/', '_')}.csv")
    if os.path.exists(path) and date.fromtimestamp(os.path.getmtime(path)) == date.today():
        return pd.read_csv(path, parse_dates=["date"])
    try:
        df = _normalize(retry(lambda: _fetch(kind, code, start)))
    except Exception as e:  # noqa: BLE001
        print(f"  行情获取失败 {kind} {code}: {e}")
        return pd.read_csv(path, parse_dates=["date"]) if os.path.exists(path) else None
    df.to_csv(path, index=False)
    return df


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线合成周线，日期取该周最后一个交易日。"""
    g = df.set_index("date").groupby(pd.Grouper(freq="W-FRI"))
    w = pd.DataFrame({
        "date": g["close"].apply(lambda s: s.index.max()),
        "open": g["open"].first(),
        "close": g["close"].last(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "volume": g["volume"].sum(min_count=1),
    })
    return w.dropna(subset=["close"]).reset_index(drop=True)
