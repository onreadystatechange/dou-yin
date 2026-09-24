"""
SQLite 存储层。每一步按 aweme_id 记录进度，保证流水线可增量、可续跑。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    aweme_id     TEXT PRIMARY KEY,
    author       TEXT,
    title        TEXT,
    create_time  TEXT NOT NULL,          -- 本地时间 YYYY-MM-DD HH:MM:SS
    share_url    TEXT,
    duration_sec REAL,
    play_url     TEXT,
    video_path   TEXT,
    indicators_extracted_at TEXT         -- 技术指标提取完成时间，NULL 表示未提取
);

CREATE TABLE IF NOT EXISTS transcripts (
    aweme_id       TEXT PRIMARY KEY REFERENCES videos(aweme_id),
    text           TEXT NOT NULL,
    sentences_json TEXT NOT NULL,        -- [{start, end, text}]，秒
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indicator_rules (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    aweme_id     TEXT NOT NULL REFERENCES videos(aweme_id),
    name         TEXT,                   -- 规则名，如 "MACD 红绿柱波段"
    indicators   TEXT,                   -- 涉及的指标，逗号分隔
    description  TEXT,                   -- 博主讲的用法，大白话
    quote        TEXT,
    start_sec    REAL,
    end_sec      REAL,
    targets_json TEXT,                   -- 博主演示/推荐使用该指标的标的名称
    rule_json    TEXT,                   -- 可回测的规则 DSL；NULL 表示无法量化
    rule_key     TEXT,                   -- 规则 DSL 的规范化哈希，用于跨视频合并同一规则
    unsupported_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_rules_aweme ON indicator_rules(aweme_id);
CREATE INDEX IF NOT EXISTS idx_rules_key ON indicator_rules(rule_key);

DROP TABLE IF EXISTS view_returns;
DROP TABLE IF EXISTS views;
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(videos)")}
        if "indicators_extracted_at" not in cols:
            conn.execute("ALTER TABLE videos ADD COLUMN indicators_extracted_at TEXT")


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)
