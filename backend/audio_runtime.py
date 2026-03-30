from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("audio.runtime")

MODEL_ID      = "Stacking-5xWav2Vec2"
MODEL_DISPLAY = "5× Wav2Vec2 + LogisticRegression Stacking"

SUPPORTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/ogg",
    "audio/x-flac",
    "audio/mp4",
    "audio/m4a",
    "video/mp4",          # some m4a files come as video/mp4
    "application/octet-stream",
}

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _resolve_device() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return -1


class AudioDetectorService:
    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._loaded        = False
        self._load_error: str | None = None
        self._stacking      = None
        self._scaler        = None
        self._meta: dict    = {}
        self._base_models   = None
        self._device        = _resolve_device()

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._load()
                self._loaded = True
                LOGGER.info("Audio detector loaded on device=%s", self._device)
            except Exception as exc:
                self._load_error = str(exc)
                LOGGER.exception("Failed to load audio detector")
                raise

    def _load(self) -> None:
        from ai.audio_detector.inference import (
            load_stacking_model,
            load_base_models,
        )
        self._stacking, self._scaler, _, self._meta = load_stacking_model()
        self._base_models = load_base_models(self._device)

    def health_snapshot(self) -> dict[str, Any]:
        device_name = "cuda" if self._device == 0 else "cpu"
        return {
            "device":            device_name,
            "configured_models": [MODEL_ID],
            "loaded_models":     [MODEL_ID] if self._loaded else [],
            "model_status":      "ready" if self._loaded else "lazy",
            "base_models":       [n for n, _ in (self._base_models or [])],
            "meta_accuracy":     self._meta.get("accuracy"),
            "load_errors":       {MODEL_ID: self._load_error} if self._load_error else {},
        }

    def predict(self, audio_bytes: bytes, filename: str) -> dict[str, Any]:
        self.ensure_loaded()

        from ai.audio_detector.inference import predict_audio

        suffix = Path(filename).suffix.lower() or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        wav_path: str | None = None
        try:
            # wav2vec2 pipelines cannot read M4A/MP4/OGG — convert to WAV via ffmpeg
            if suffix in {".m4a", ".mp4", ".ogg"}:
                import subprocess
                wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wav_tmp.close()
                wav_path = wav_tmp.name
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                    check=True,
                    capture_output=True,
                )
                infer_path = wav_path
            else:
                infer_path = tmp_path

            result = predict_audio(infer_path, self._base_models, self._stacking, self._scaler)
        finally:
            os.unlink(tmp_path)
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        device_name = "cuda" if self._device == 0 else "cpu"
        result["device"]       = device_name
        result["primary_model"] = MODEL_ID
        return result
