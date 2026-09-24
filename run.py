"""
命令行入口。

    python run.py fetch [--limit 5] [--full] [--url 视频链接 ...]
    python run.py transcribe [--limit 5]
    python run.py extract [--limit 5] [--redo]
    python run.py backtest
    python run.py all [--limit 5] [--full]   # 拉列表后分批“下载 → 转写 → 删视频”，再提取、回测
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="抖音博主技术指标提取与回测")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="抓取作品列表并下载视频")
    f.add_argument("--limit", type=int)
    f.add_argument("--full", action="store_true", help="翻完全部历史作品（默认遇到已抓过的就停）")
    f.add_argument("--url", nargs="+", help="只抓指定的视频链接")

    t = sub.add_parser("transcribe", help="FunASR 转写")
    t.add_argument("--limit", type=int)

    e = sub.add_parser("extract", help="DeepSeek 提取技术指标规则")
    e.add_argument("--limit", type=int)
    e.add_argument("--redo", action="store_true", help="重新提取已提取过的视频（调整提示词后用）")

    sub.add_parser("backtest", help="回测技术指标规则并生成报告")

    a = sub.add_parser("all", help="依次执行全部步骤")
    a.add_argument("--limit", type=int)
    a.add_argument("--full", action="store_true")

    args = p.parse_args()

    if args.cmd == "fetch":
        import fetch

        fetch.run(limit=args.limit, full=args.full, urls=args.url)
    if args.cmd == "transcribe":
        import transcribe

        transcribe.run(limit=args.limit)
    if args.cmd == "all":
        import config
        import fetch
        import transcribe

        fetch.list_posts(limit=args.limit, full=args.full)
        transcribe.run()  # 先处理已下载但未转写的
        while fetch.pending_count():
            print(f"剩余待下载 {fetch.pending_count()} 条")
            fetch.download(limit=config.BATCH_SIZE)
            transcribe.run()
    if args.cmd in ("extract", "all"):
        import extract

        extract.run(limit=args.limit, redo=getattr(args, "redo", False))
    if args.cmd in ("backtest", "all"):
        import backtest

        backtest.run()


if __name__ == "__main__":
    main()
