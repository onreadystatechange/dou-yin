"""
转写：ffmpeg 抽 16kHz 单声道 wav，FunASR（paraformer-zh + VAD + 标点）生成带句子级时间戳的逐字稿。

可在项目根目录放 hotwords.txt（每行一个词，如股票名、博主口头禅），提升专有名词识别率。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Optional

from tqdm import tqdm

import config
import db

_HOTWORDS_PATH = os.path.join(config.BASE_DIR, "hotwords.txt")
_model = None


def _model_path(model_id: str) -> str:
    """已下载过的模型直接用本地目录；传 hub ID 时 ModelScope 每次都会联网校验，网络慢时加载要好几分钟。"""
    local = os.path.join(config.MODELSCOPE_CACHE, "models", model_id.replace("/", "--"), "snapshots", "master")
    return local if os.path.exists(os.path.join(local, "model.pt")) else model_id


def _load_model():
    global _model
    if _model is None:
        from funasr import AutoModel

        _model = AutoModel(
            model=_model_path(config.ASR_MODEL),
            vad_model=_model_path(config.ASR_VAD_MODEL),
            punc_model=_model_path(config.ASR_PUNC_MODEL),
            disable_update=True,
        )
    return _model


def _hotwords() -> str:
    if not os.path.exists(_HOTWORDS_PATH):
        return ""
    with open(_HOTWORDS_PATH, encoding="utf-8") as f:
        return " ".join(w.strip() for w in f if w.strip())


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _extract_audio(video_path: str, wav_path: str) -> None:
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
        check=True,
    )


def transcribe_file(wav_path: str) -> dict:
    kwargs = {"input": wav_path, "batch_size_s": 300, "sentence_timestamp": True}
    hot = _hotwords()
    if hot:
        kwargs["hotword"] = hot
    res = _load_model().generate(**kwargs)[0]
    sentences = [
        {"start": round(s["start"] / 1000, 2), "end": round(s["end"] / 1000, 2), "text": s["text"]}
        for s in res.get("sentence_info", [])
    ]
    return {"text": res.get("text", ""), "sentences": sentences}


def run(limit: Optional[int] = None) -> None:
    db.init()
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    os.makedirs(config.TRANSCRIPT_DIR, exist_ok=True)
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT v.aweme_id, v.video_path FROM videos v
               LEFT JOIN transcripts t ON t.aweme_id = v.aweme_id
               WHERE v.video_path IS NOT NULL AND t.aweme_id IS NULL
               ORDER BY v.create_time DESC"""
        ).fetchall()
    if limit:
        rows = rows[:limit]
    if not rows:
        print("没有待转写的视频")
        return

    for row in tqdm(rows, desc="转写"):
        aweme_id = row["aweme_id"]
        wav = os.path.join(config.AUDIO_DIR, f"{aweme_id}.wav")
        try:
            _extract_audio(row["video_path"], wav)
            result = transcribe_file(wav)
        except Exception as e:  # noqa: BLE001
            print(f"  转写失败 {aweme_id}: {e}")
            continue
        with open(os.path.join(config.TRANSCRIPT_DIR, f"{aweme_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        with db.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO transcripts (aweme_id, text, sentences_json, created_at) VALUES (?,?,?,?)",
                (
                    aweme_id,
                    result["text"],
                    json.dumps(result["sentences"], ensure_ascii=False),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        os.remove(wav)
        if not config.KEEP_VIDEO and os.path.exists(row["video_path"]):
            os.remove(row["video_path"])
