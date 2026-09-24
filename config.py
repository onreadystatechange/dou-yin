"""
全局配置。敏感信息（Cookie、API Key）放 .env.local，不写进代码。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env.local"))

DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DB_PATH = os.path.join(DATA_DIR, "douyin.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ALIASES_PATH = os.path.join(BASE_DIR, "aliases.csv")

# ---- 抓取 ----
DOUYIN_USER_URL = os.getenv("DOUYIN_USER_URL", "")
DOUYIN_COOKIE = os.getenv("DOUYIN_COOKIE", "")
# 选填：与复制 Cookie 的浏览器一致的 User-Agent，不一致时抖音可能返回空内容
DOUYIN_USER_AGENT = os.getenv("DOUYIN_USER_AGENT", "")
FETCH_PAGE_SIZE = 20
# 转写完成后是否保留视频文件（博主作品多，全部保留很占磁盘）
KEEP_VIDEO = False
# all 命令每批下载并转写的视频数
BATCH_SIZE = 10

# ---- 转写（FunASR）----
# ModelScope 模型 ID，分别对应 FunASR 的 paraformer-zh / fsmn-vad / ct-punc
ASR_MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
ASR_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
ASR_PUNC_MODEL = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
MODELSCOPE_CACHE = os.getenv("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

# ---- 技术指标提取（DeepSeek，OpenAI 兼容接口）----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"
# 每块逐字稿的最大字符数，超出按句子切块
LLM_CHUNK_CHARS = 6000
LLM_CONCURRENCY = 8

# ---- 指标回测 ----
# 固定回测标的（名称，按 targets.py 规则解析）；博主视频里演示过的标的会自动追加
UNIVERSE = [
    "上证指数", "沪深300", "创业板指", "科创50", "中证500", "中证1000",
    "半导体", "证券", "白酒", "银行", "光伏设备", "电池", "医疗器械", "软件开发",
]
BACKTEST_START = "2016-01-01"
# 单边交易成本（佣金 + 滑点），ETF 实盘约万一到万五
COST_PER_SIDE = 0.0005
# 模糊匹配标的名称的最低相似度
RESOLVE_MIN_SIMILARITY = 0.75
