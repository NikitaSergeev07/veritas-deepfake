from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

LOGGER = logging.getLogger("deepfake.runtime")

PROJECT_ROOT = Path(__file__).parent.parent  # deepfake_detector/
CHECKPOINT_PATH = PROJECT_ROOT / "ai" / "image_detector" / "checkpoints" / "best_model.pth"

MODEL_ID = "CLIP-ViT-B16-MLP"
MODEL_DISPLAY = "CLIP ViT-B/16 + MLP Head"


def resolve_device(preferred: str | None = None) -> str:
    target = (preferred or "auto").strip().lower()
    if target and target != "auto":
        return target
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def build_views(image: Image.Image) -> list[tuple[str, Image.Image]]:
    base = _normalize_image(image)
    width, height = base.size
    short_side = min(width, height)
    views: list[tuple[str, Image.Image]] = [("full_frame", base)]

    if width != height:
        left = (width - short_side) // 2
        top = (height - short_side) // 2
        square = base.crop((left, top, left + short_side, top + short_side))
        views.append(("center_square", square))

    focus_side = int(short_side * 0.82)
    if focus_side >= 160 and focus_side < short_side:
        left = (width - focus_side) // 2
        top = (height - focus_side) // 2
        focus = base.crop((left, top, left + focus_side, top + focus_side))
        views.append(("focus_crop", focus))

    return views


class DeepfakeDetectorService:
    def __init__(self) -> None:
        self.device_str = resolve_device(os.environ.get("DEEPFAKE_DEVICE", "auto"))
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._clip_visual = None
        self._mlp = None
        self._threshold = 0.5
        self._device: torch.device | None = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._load()
                self._loaded = True
                LOGGER.info("Model loaded on %s", self.device_str)
            except Exception as exc:
                self._load_error = str(exc)
                LOGGER.exception("Failed to load model")
                raise

    def _load(self) -> None:
        from ai.image_detector.inference import get_device, load_model

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {CHECKPOINT_PATH}. "
                "Run: python -m ai.image_detector.train"
            )

        device = get_device()
        self._device = device
        self.device_str = str(device)

        clip_visual, mlp, threshold = load_model(str(CHECKPOINT_PATH), device)
        self._clip_visual = clip_visual
        self._mlp = mlp
        self._threshold = threshold

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "device": self.device_str,
            "configured_models": [MODEL_ID],
            "loaded_models": [MODEL_ID] if self._loaded else [],
            "model_status": "ready" if self._loaded else "lazy",
            "view_strategy": ["full_frame", "center_square", "focus_crop"],
            "load_errors": {MODEL_ID: self._load_error} if self._load_error else {},
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        self.ensure_loaded()

        from ai.image_detector.dataset import get_val_transform

        transform = get_val_transform()
        views = build_views(image)

        view_scores = []
        with torch.no_grad():
            for name, view_img in views:
                tensor = transform(view_img.convert("RGB")).unsqueeze(0).to(self._device)
                feats = self._clip_visual(tensor)
                prob = float(self._mlp.predict_proba(feats).cpu())
                view_scores.append({
                    "name": name,
                    "fake_probability": prob,
                    "real_probability": 1.0 - prob,
                })

        fake_probability = sum(v["fake_probability"] for v in view_scores) / len(view_scores)
        real_probability = 1.0 - fake_probability
        label = "fake" if fake_probability >= self._threshold else "real"
        confidence = fake_probability if label == "fake" else real_probability

        summary = (
            f"Likely {'deepfake' if label == 'fake' else 'real image'} "
            f"with {confidence * 100:.1f}% confidence "
            f"(threshold={self._threshold:.2f}, {len(views)} view(s) analysed)."
        )

        return {
            "label": label,
            "confidence": float(confidence),
            "fake_probability": float(fake_probability),
            "real_probability": float(real_probability),
            "summary": summary,
            "device": self.device_str,
            "primary_model": MODEL_ID,
            "fallback_model": None,
            "models": [
                {
                    "model_id": MODEL_ID,
                    "display_name": MODEL_DISPLAY,
                    "backend": "clip-mlp",
                    "device": self.device_str,
                    "weight": 1.0,
                    "fake_probability": float(fake_probability),
                    "real_probability": float(real_probability),
                    "views": view_scores,
                }
            ],
            "view_names": [name for name, _ in views],
        }
