from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("video.runtime")

MODEL_ID = "VideoMAE-FFC23"
MODEL_DISPLAY = "VideoMAE (FaceForensics++ C23)"

SUPPORTED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
    "application/octet-stream",
}

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class VideoDetectorService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._model = None
        self._processor = None
        self._device = self._resolve_device()

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._load()
                self._loaded = True
                LOGGER.info("Video detector loaded on device=%s", self._device)
            except Exception as exc:
                self._load_error = str(exc)
                LOGGER.exception("Failed to load video detector")
                raise

    def _load(self) -> None:
        from ai.video_detector.inference import load_model
        self._model, self._processor = load_model(self._device)

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "device": self._device,
            "configured_models": [MODEL_ID],
            "loaded_models": [MODEL_ID] if self._loaded else [],
            "model_status": "ready" if self._loaded else "lazy",
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "supported_content_types": sorted(SUPPORTED_CONTENT_TYPES),
            "load_errors": {MODEL_ID: self._load_error} if self._load_error else {},
        }

    def predict(self, video_bytes: bytes, filename: str) -> dict[str, Any]:
        self.ensure_loaded()

        from ai.video_detector.inference import predict_video

        suffix = Path(filename).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            result = predict_video(
                tmp_path, self._model, self._processor, self._device,
            )
        finally:
            os.unlink(tmp_path)

        result["device"] = self._device
        result["primary_model"] = MODEL_ID
        return result
