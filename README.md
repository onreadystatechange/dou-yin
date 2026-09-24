# 抖音博主技术指标提取与回测

抓取博主视频 → 语音转写 → LLM 提取视频中讲到的**技术指标用法**（如 MACD 红柱持有、阳包阴站上 5 日线）并翻译成可执行规则 → 在博主演示过的标的 + 一篮子宽基/行业上回测，与买入持有对比。

不关心博主对个股、大盘、板块的观点，只评估“方法”本身是否有效。

## 安装

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.local.example .env.local   # 填 DOUYIN_USER_URL / DOUYIN_COOKIE / DEEPSEEK_API_KEY
```

- Cookie 需在登录状态下从浏览器开发者工具 Network 面板复制完整请求头（含 HttpOnly 的 `sessionid` 等），`document.cookie` 拿到的不完整。
- FunASR 模型首次运行自动下载到 `~/.cache/modelscope`，之后离线加载。
- 系统无 ffmpeg 时自动使用 `imageio-ffmpeg` 自带的二进制。

## 使用

```bash
.venv/bin/python run.py fetch --limit 5    # 增量抓取最新视频；--full 抓全部历史；--url 抓指定视频
.venv/bin/python run.py transcribe         # 转写未处理的视频
.venv/bin/python run.py extract            # 提取技术指标规则；--redo 全部重新提取
.venv/bin/python run.py backtest           # 回测并生成报告
.venv/bin/python run.py all                # 以上四步依次执行
```

每一步都是增量的，已处理的视频不会重复处理。

- `all` 先拉作品列表，再按 `BATCH_SIZE` 分批“下载 → 转写”，转写完即删除视频（`KEEP_VIDEO=False`），磁盘只保留逐字稿。
- `fetch --full` 翻页被风控中断时会在 `data/fetch_state.json` 记录断点，重新运行即从断点继续。
- 提取按 `LLM_CONCURRENCY` 并发调用 DeepSeek。

## 输出（`output/`）

| 文件 | 内容 |
|---|---|
| `indicator_report.md` | 规则总览、每条规则的原话与视频时间点链接、各标的回测表、当前信号状态、无法量化的方法 |
| `indicator_rules.csv` | 每条提取记录（视频、原话、规则 JSON） |
| `indicator_backtest.csv` | 规则 × 标的的回测指标 |
| `indicator_trades.csv` | 逐笔交易明细 |

不同视频里讲的同一方法（规则 JSON 规范化后相同）会合并，并记录出现次数和来源。

## 回测口径

- 收盘判定信号，次日开盘成交；只做多；单边成本 `COST_PER_SIDE`（默认 0.05%）。
- 指标用全部历史预热，统计从 `BACKTEST_START`（默认 2016-01-01）开始。
- 标的 = 博主演示时提到的标的（报告中标“演示”）+ `config.UNIVERSE`。
- 周线由日线按周五重采样；个股前复权。
- 指标采用国内软件口径：MACD 柱 = 2×(DIF−DEA)，KDJ/RSI 用通达信 SMA，BOLL 标准差 ddof=0。
- 东方财富的行业板块指数多从 2021 年起才有数据，区间会比宽基短。

## 规则 DSL

LLM 把博主的说法翻译成如下 JSON（见 `indicators.py`）：

```json
{
  "period": "daily",
  "entry": {"left": "macd_hist", "op": "cross_above", "right": 0},
  "exit":  {"any": [{"left": "close", "op": "cross_below", "right": "ma5"},
                    {"left": "bear_engulf", "op": ">", "right": 0}]},
  "stop_loss_pct": null, "take_profit_pct": null, "max_hold_bars": null
}
```

- 条件可用 `all` / `any` 任意嵌套；原子条件为 `left op right`，可选 `left_shift`、`right_shift`、`right_mult`、`for_bars`（连续 N 根成立）。
- `op`：`>` `<` `>=` `<=` `cross_above` `cross_below`。
- 序列：`open/high/low/close/volume/pct_chg`、`maN`、`emaN`、`vol_maN`、`rsiN`、`hhvN`/`llvN`、`macd_dif/dea/hist`、`kdj_k/d/j`、`boll_upper/mid/lower`、`bull_engulf`/`bear_engulf`。
- 参数通过 `params` 指定（如 `{"macd": {"fast": 12, "slow": 26, "signal": 9}}`），缺省用常规参数。
- 无法用 DSL 表达的方法（如缠论笔段、主观画线）会记录 `unsupported_reason`，列在报告末尾。要支持新指标，在 `indicators.py` 的 `_series` 和 `SERIES_DOC` 中添加即可。

## 配置

- `config.py`：`UNIVERSE`、`BACKTEST_START`、`COST_PER_SIDE`、LLM 模型等。
- `aliases.csv`：标的别名（如 宁王→宁德时代、半导体ETF→半导体）。
- `hotwords.txt`：ASR 热词，提高 MACD、KDJ 等术语的识别准确率。
