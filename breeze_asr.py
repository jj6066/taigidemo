#!/usr/bin/env python3
"""Shared MediaTek Breeze-ASR-26 transcription helpers.

This module deliberately keeps model loading lazy so importing it does not
load Torch or the ASR model. Callers can transcribe a local audio file or
audio bytes received from a service such as LINE.
"""

from __future__ import annotations

import os
import gc
import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any


MODEL_PATH = Path(__file__).resolve().parent / "models" / "Breeze-ASR-26-fp16"
_recognizer: Any = None
_force_cpu = False
_recognizer_lock = threading.Lock()
_transcription_lock = threading.Lock()
logger = logging.getLogger(__name__)


def decode_audio(audio_path: Path):
    """Decode audio to mono 16 kHz float32 samples with FFmpeg."""
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v", "error",
                "-i", str(audio_path),
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                "pipe:1",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("找不到 FFmpeg，請先安裝 ffmpeg。") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("FFmpeg 音檔解碼逾時。") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg 無法讀取音檔：{detail}") from exc

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("缺少 NumPy，請安裝 taigi/requirements.txt 內的依賴。") from exc

    samples = np.frombuffer(completed.stdout, dtype=np.int16)
    if samples.size == 0:
        raise RuntimeError("音檔解碼後沒有任何聲音資料。")
    return samples.astype(np.float32) / 32768.0


def _get_recognizer() -> Any:
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    with _recognizer_lock:
        if _recognizer is not None:
            return _recognizer
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "目前執行程式的 Python 找不到 Torch 或 Transformers；"
                "請安裝 taigi/requirements.txt 內的依賴。"
            ) from exc

        use_cuda = torch.cuda.is_available()
        use_cuda = use_cuda and not _force_cpu
        device = 0 if use_cuda else -1
        if not MODEL_PATH.is_dir():
            raise RuntimeError(f"找不到本機模型資料夾：{MODEL_PATH}")
        _recognizer = pipeline(
            "automatic-speech-recognition",
            model=str(MODEL_PATH),
            device=device,
            dtype=torch.float16 if use_cuda else torch.float32,
            # Use half precision on GPU to reduce model memory use. Keep CPU
            # inference in float32 because CPU float16 support varies by op.
        )
        return _recognizer


def transcribe_audio_file(audio_path: str | os.PathLike[str]) -> str:
    """Transcribe an audio file and return its recognized text."""
    global _recognizer, _force_cpu
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到音檔：{path}")

    audio_samples = decode_audio(path)
    # Serialize inference so concurrent webhook requests do not contend for
    # the same model/GPU memory.
    with _transcription_lock:
        inputs = {"array": audio_samples, "sampling_rate": 16000}
        try:
            result = _get_recognizer()(inputs, return_timestamps=False)
        except RuntimeError as exc:
            if _force_cpu or "cuda out of memory" not in str(exc).casefold():
                raise
            logger.warning("ASR CUDA memory exhausted; retrying on CPU")
            _force_cpu = True
            _recognizer = None
            gc.collect()
            import torch

            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                logger.warning("Could not clear CUDA cache before ASR CPU retry")
            result = _get_recognizer()(inputs, return_timestamps=False)
    text = str(result.get("text", "")).strip()
    if not text:
        raise RuntimeError("語音辨識沒有產生文字。")
    return text


def transcribe_audio_bytes(audio_bytes: bytes, content_type: str = "") -> str:
    """Transcribe audio bytes, such as audio content downloaded from LINE."""
    if not audio_bytes:
        raise ValueError("收到的語音檔案是空的。")

    normalized_type = content_type.split(";", 1)[0].strip().lower()
    suffix = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/aac": ".aac",
        "audio/mp4": ".m4a",
        "audio/m4a": ".m4a",
        "video/mp4": ".m4a",
    }.get(normalized_type, ".m4a")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="line-audio-", suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)
        return transcribe_audio_file(temp_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
