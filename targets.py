"""
回测标的：把名称解析成可拉取行情的 (类型, 代码, 名称)。

- 指数：带市场前缀的代码，如 sh000300
- 行业/概念板块：东财板块名称（akshare 板块行情接口按名称查询）
- 个股：6 位代码
- ETF：按跟踪对象处理，名称里的 "ETF" 去掉后再解析（如 "半导体ETF" -> 半导体板块）

优先级：aliases.csv 人工别名 > 精确匹配 > 包含匹配 > 模糊匹配（difflib）。
"""

from __future__ import annotations

import difflib
import os
import re
import time
import unicodedata
from typing import Optional

import akshare as ak
import pandas as pd

import config
from utils import retry

INDEXES = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
    "沪深300": "sh000300",
    "上证50": "sh000016",
    "中证500": "sh000905",
    "中证1000": "sh000852",
    "科创50": "sh000688",
    "中证2000": "sh932000",
}
INDEX_ALIASES = {
    "大盘": "上证指数", "a股": "上证指数", "沪指": "上证指数", "上证": "上证指数", "上证综指": "上证指数",
    "深成指": "深证成指", "深指": "深证成指", "创业板": "创业板指", "创业板指数": "创业板指",
    "沪深三百": "沪深300", "上证五十": "上证50", "中证五百": "中证500", "中证一千": "中证1000",
    "科创五十": "科创50", "科创板指数": "科创50", "科创板": "科创50", "科创指数": "科创50",
}
KINDS = ("指数", "行业板块", "概念板块", "个股")
_CACHE_TTL_DAYS = 7


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKC", str(name))
    name = re.sub(r"\s+", "", name).lower()
    return re.sub(r"^\*?st", "", name)


def _cached(name: str, loader) -> pd.DataFrame:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = os.path.join(config.CACHE_DIR, f"names_{name}.csv")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < _CACHE_TTL_DAYS * 86400:
        return pd.read_csv(path, dtype=str)
    try:
        df = retry(loader)
        df.to_csv(path, index=False)
        return df
    except Exception as e:  # noqa: BLE001
        print(f"  获取{name}名称表失败：{e}")
        return pd.read_csv(path, dtype=str) if os.path.exists(path) else pd.DataFrame(columns=["code", "name"])


def _load_stocks() -> pd.DataFrame:
    df = ak.stock_info_a_code_name()
    return df[["code", "name"]].astype(str)


def _load_industry() -> pd.DataFrame:
    df = ak.stock_board_industry_name_em()
    return pd.DataFrame({"code": df["板块名称"], "name": df["板块名称"]})


def _load_concept() -> pd.DataFrame:
    df = ak.stock_board_concept_name_em()
    return pd.DataFrame({"code": df["板块名称"], "name": df["板块名称"]})


def _load_aliases() -> dict[str, str]:
    if not os.path.exists(config.ALIASES_PATH):
        return {}
    df = pd.read_csv(config.ALIASES_PATH, dtype=str).dropna()
    return {_norm(a): n.strip() for a, n in zip(df["alias"], df["name"])}


class Resolver:
    def __init__(self) -> None:
        self.aliases = _load_aliases()
        self.tables: dict[str, dict[str, tuple[str, str]]] = {"指数": {_norm(n): (c, n) for n, c in INDEXES.items()}}
        for kind, loader in (("个股", _load_stocks), ("行业板块", _load_industry), ("概念板块", _load_concept)):
            df = _cached(kind, loader)
            self.tables[kind] = {_norm(n): (str(c), str(n)) for c, n in zip(df["code"], df["name"])}

    def _match_in(self, kind: str, key: str, loose: bool) -> Optional[tuple[str, str]]:
        table = self.tables.get(kind) or {}
        if key in table:
            return table[key]
        if not loose:
            return None
        if kind in ("行业板块", "概念板块"):
            for suffix in ("概念", "板块", "行业"):
                k = key.removesuffix(suffix)
                for cand in (k, k + "概念", k + "行业"):
                    if cand in table:
                        return table[cand]
            # 板块名包含关系，如 "光伏" -> "光伏设备"，取最短的
            hits = [k for k in table if len(key) >= 2 and (key in k or k in key)]
            if hits:
                return table[min(hits, key=len)]
        close = difflib.get_close_matches(key, table.keys(), n=1, cutoff=config.RESOLVE_MIN_SIMILARITY)
        return table[close[0]] if close else None

    def resolve(self, name: str, kind_hint: Optional[str] = None) -> Optional[tuple[str, str, str]]:
        """返回 (类型, 代码, 标准名称) 或 None。"""
        key = _norm(self.aliases.get(_norm(name), name))
        key = re.sub(r"(etf|基金|指数基金)$", "", key) or key
        key = _norm(INDEX_ALIASES.get(key, key))
        order = ([kind_hint] if kind_hint in KINDS else []) + [k for k in KINDS if k != kind_hint]
        for kind in order:
            hit = self._match_in(kind, key, loose=False)
            if hit:
                return kind, hit[0], hit[1]
        # 宽松匹配只对板块做，避免个股被模糊错配
        for kind in ("行业板块", "概念板块"):
            hit = self._match_in(kind, key, loose=True)
            if hit:
                return kind, hit[0], hit[1]
        return None
