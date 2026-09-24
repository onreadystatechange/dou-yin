"""
抓取：用 F2 的抖音接口拉取博主主页作品列表，元数据落库并下载视频文件。

只复用 F2 的签名与请求能力（DouyinCrawler），作品字段自己从原始 JSON 里取，
这样发布时间（回测锚点）等关键字段不依赖 F2 的格式化逻辑。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from datetime import datetime
from typing import Optional

import httpx
from f2.apps.douyin.api import DouyinAPIEndpoints  # noqa: F401  确保 F2 端点模块加载
from f2.apps.douyin.crawler import DouyinCrawler
from f2.exceptions.api_exceptions import APIError
from f2.apps.douyin.model import PostDetail, UserPost
from f2.apps.douyin.utils import AwemeIdFetcher, ClientConfManager, SecUserIdFetcher

import config
import db

# aweme_type 0 为普通视频；图文等其他类型没有可转写的语音
_VIDEO_AWEME_TYPES = {0, 4, 51, 55, 58, 61, 109, 201}
_STATE_PATH = os.path.join(config.DATA_DIR, "fetch_state.json")


def _headers() -> dict:
    headers = dict(ClientConfManager.headers())
    if config.DOUYIN_USER_AGENT:
        headers["User-Agent"] = config.DOUYIN_USER_AGENT
    return headers


def _crawler_kwargs() -> dict:
    if not config.DOUYIN_COOKIE:
        raise SystemExit("未配置 DOUYIN_COOKIE，请在 .env.local 中填写")
    fields = {kv.split("=", 1)[0].strip() for kv in config.DOUYIN_COOKIE.split(";") if "=" in kv}
    if not fields & {"sessionid", "sessionid_ss", "sid_guard"}:
        raise SystemExit(
            "DOUYIN_COOKIE 缺少登录字段（sessionid 等）。请在已登录的浏览器里 F12 -> Network，"
            "选一个 www.douyin.com 的请求，从 Request Headers 复制整串 Cookie；不要用 document.cookie"
        )
    return {
        "headers": _headers(),
        "proxies": {"http://": None, "https://": None},
        "cookie": config.DOUYIN_COOKIE,
        "timeout": 15,
        "max_retries": 5,
    }


def _parse_aweme(aweme: dict) -> Optional[dict]:
    if aweme.get("aweme_type") not in _VIDEO_AWEME_TYPES or aweme.get("images"):
        return None
    video = aweme.get("video") or {}
    urls = (video.get("play_addr") or {}).get("url_list") or []
    if not urls:
        bit_rate = video.get("bit_rate") or []
        if bit_rate:
            urls = (bit_rate[0].get("play_addr") or {}).get("url_list") or []
    aweme_id = str(aweme["aweme_id"])
    return {
        "aweme_id": aweme_id,
        "author": (aweme.get("author") or {}).get("nickname"),
        "title": aweme.get("desc") or "",
        "create_time": datetime.fromtimestamp(int(aweme["create_time"])).strftime("%Y-%m-%d %H:%M:%S"),
        "share_url": f"https://www.douyin.com/video/{aweme_id}",
        "duration_sec": (video.get("duration") or aweme.get("duration") or 0) / 1000,
        "play_url": urls[0] if urls else None,
    }


def _save_items(items: list[dict]) -> None:
    with db.connect() as conn:
        for it in items:
            conn.execute(
                """INSERT OR IGNORE INTO videos
                   (aweme_id, author, title, create_time, share_url, duration_sec, play_url)
                   VALUES (:aweme_id, :author, :title, :create_time, :share_url, :duration_sec, :play_url)""",
                it,
            )


def _load_cursor() -> int:
    if os.path.exists(_STATE_PATH):
        with open(_STATE_PATH, encoding="utf-8") as f:
            return int(json.load(f).get("max_cursor", 0))
    return 0


def _save_cursor(max_cursor: Optional[int]) -> None:
    if max_cursor:
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"max_cursor": max_cursor}, f)
    elif os.path.exists(_STATE_PATH):
        os.remove(_STATE_PATH)


async def _fetch_page(kwargs: dict, sec_user_id: str, max_cursor: int) -> Optional[dict]:
    """个别游标的请求会稳定失败（403/超时），把游标减几毫秒换个签名就能过，不会漏作品；
    仍失败则按 1/2/4 分钟退避，最终返回 None。"""
    kwargs = {**kwargs, "max_retries": 2}
    waits = [0, 5, 60, 120, 240]
    for attempt, wait in enumerate(waits):
        if wait:
            await asyncio.sleep(wait)
        cursor = max_cursor - attempt if max_cursor else 0
        try:
            async with DouyinCrawler(kwargs) as crawler:
                return await crawler.fetch_user_post(
                    UserPost(max_cursor=cursor, count=config.FETCH_PAGE_SIZE, sec_user_id=sec_user_id)
                )
        except APIError as e:
            print(f"  翻页失败（第 {attempt + 1} 次）：{str(e)[:60]}")
    return None


async def _list_posts(user_url: str, limit: Optional[int], full: bool) -> int:
    """按发布时间倒序翻页，每页立即入库。增量模式遇到已有作品即停；
    全量模式中断时记录游标，下次从断点继续。返回新增条数。"""
    kwargs = _crawler_kwargs()
    m = re.search(r"/user/([\w-]+)", user_url)
    sec_user_id = m.group(1) if m else await SecUserIdFetcher.get_sec_user_id(user_url)
    with db.connect() as conn:
        known = {r[0] for r in conn.execute("SELECT aweme_id FROM videos")}

    added = 0
    max_cursor = _load_cursor() if full else 0
    if max_cursor:
        print(f"  从上次中断处继续（{datetime.fromtimestamp(max_cursor / 1000):%Y-%m-%d}）")
    while True:
        resp = await _fetch_page(kwargs, sec_user_id, max_cursor)
        if resp is None:
            if full:
                _save_cursor(max_cursor)
                print("  已记录断点，稍后重新运行 fetch --full 即可继续")
            return added
        aweme_list = resp.get("aweme_list") or []
        if not aweme_list and resp.get("status_code") not in (0, None):
            raise RuntimeError(f"抖音接口返回异常：{resp.get('status_msg') or resp}，请检查 Cookie")

        hit_known = False
        items = []
        for aweme in aweme_list:
            # 置顶作品可能比后续作品旧，不能据此判断增量截止
            is_top = bool(aweme.get("is_top"))
            if str(aweme.get("aweme_id")) in known:
                if not is_top:
                    hit_known = True
                continue
            parsed = _parse_aweme(aweme)
            if parsed:
                items.append(parsed)
                known.add(parsed["aweme_id"])
        if limit:
            items = items[: max(limit - added, 0)]
        _save_items(items)
        added += len(items)

        last = aweme_list[-1]["create_time"] if aweme_list else None
        date = f"，已翻到 {datetime.fromtimestamp(int(last)):%Y-%m-%d}" if last else ""
        print(f"  新增 {added} 条{date}")
        if (limit and added >= limit) or (not full and hit_known) or not resp.get("has_more"):
            if full and not resp.get("has_more"):
                _save_cursor(None)
            return added
        max_cursor = resp.get("max_cursor", 0)
        if full:
            _save_cursor(max_cursor)
        await asyncio.sleep(random.uniform(4, 8))


async def _refresh_play_url(aweme_id: str) -> Optional[str]:
    """播放地址有时效，过期后通过作品详情接口重新获取。"""
    async with DouyinCrawler(_crawler_kwargs()) as crawler:
        resp = await crawler.fetch_post_detail(PostDetail(aweme_id=aweme_id))
    detail = resp.get("aweme_detail")
    parsed = _parse_aweme(detail) if detail else None
    return parsed["play_url"] if parsed else None


def _download(url: str, path: str) -> None:
    headers = {"User-Agent": _headers().get("User-Agent", ""), "Referer": "https://www.douyin.com/"}
    tmp = path + ".part"
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                f.write(chunk)
    os.replace(tmp, path)


async def _fetch_details(urls: list[str]) -> list[dict]:
    items = []
    for url in urls:
        aweme_id = await AwemeIdFetcher.get_aweme_id(url)
        async with DouyinCrawler(_crawler_kwargs()) as crawler:
            resp = await crawler.fetch_post_detail(PostDetail(aweme_id=aweme_id))
        detail = resp.get("aweme_detail")
        parsed = _parse_aweme(detail) if detail else None
        if parsed:
            items.append(parsed)
        else:
            print(f"  跳过（非视频或获取失败）：{url}")
    return items


def list_posts(limit: Optional[int] = None, full: bool = False, urls: Optional[list[str]] = None) -> None:
    """urls 不为空时只抓这几条视频；否则抓 DOUYIN_USER_URL 主页作品列表。只入库元数据，不下载。"""
    db.init()
    if urls:
        items = asyncio.run(_fetch_details(urls))
        _save_items(items)
        added = len(items)
    else:
        if not config.DOUYIN_USER_URL:
            raise SystemExit("未配置 DOUYIN_USER_URL，请在 .env.local 中填写博主主页链接")
        print(f"拉取作品列表：{config.DOUYIN_USER_URL}")
        added = asyncio.run(_list_posts(config.DOUYIN_USER_URL, limit, full))
    print(f"新增作品 {added} 条")


def pending_count() -> int:
    with db.connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM videos WHERE video_path IS NULL AND play_url IS NOT NULL"
        ).fetchone()[0]


def run(limit: Optional[int] = None, full: bool = False, urls: Optional[list[str]] = None) -> None:
    list_posts(limit, full, urls)
    download(limit)


def download(limit: Optional[int] = None) -> None:
    """下载尚未下载的视频，按发布时间倒序。刷新地址后仍失败的清空 play_url，避免批处理反复卡在同一条上。"""
    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    with db.connect() as conn:
        pending = conn.execute(
            "SELECT aweme_id, play_url FROM videos WHERE video_path IS NULL AND play_url IS NOT NULL "
            "ORDER BY create_time DESC"
        ).fetchall()
    if limit:
        pending = pending[:limit]
    for row in pending:
        path = os.path.join(config.VIDEO_DIR, f"{row['aweme_id']}.mp4")
        try:
            _download(row["play_url"], path)
        except Exception:  # noqa: BLE001
            try:
                url = asyncio.run(_refresh_play_url(row["aweme_id"]))
                if not url:
                    raise RuntimeError("作品详情中没有播放地址")
                _download(url, path)
            except Exception as e:  # noqa: BLE001
                print(f"  下载失败 {row['aweme_id']}: {e}")
                with db.connect() as conn:
                    conn.execute("UPDATE videos SET play_url=NULL WHERE aweme_id=?", (row["aweme_id"],))
                continue
        with db.connect() as conn:
            conn.execute("UPDATE videos SET video_path=? WHERE aweme_id=?", (path, row["aweme_id"]))
        print(f"  已下载 {row['aweme_id']}")
