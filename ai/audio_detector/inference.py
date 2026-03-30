from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("audio_detector")

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

BASE_MODELS = [
    ("model1", "garystafford/wav2vec2-deepfake-voice-detector"),
    ("model2", "MelodyMachine/Deepfake-audio-detection"),
    ("model3", "Hemgg/Deepfake-audio-detection"),
    ("model4", "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"),
    ("model5", "mo-thecreator/Deepfake-audio-detection"),
]

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _resolve_device() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return -1


def _extract_fake_probability(result: list[dict]) -> float:
    if not result:
        return 0.5
    score = float(result[0]["score"])
    for item in result:
        if "real" in item["label"].lower():
            score = 1.0 - float(item["score"])
            break
    return score


def load_stacking_model():
    import joblib
    model   = joblib.load(CHECKPOINTS_DIR / "logistic_regression_model.joblib")
    scaler  = joblib.load(CHECKPOINTS_DIR / "scaler.joblib")
    names   = joblib.load(CHECKPOINTS_DIR / "feature_names.joblib")
    with open(CHECKPOINTS_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    return model, scaler, names, meta


def load_base_models(device: int = -1) -> list[tuple[str, Any]]:
    from transformers import pipeline as hf_pipeline

    loaded = []
    for name, model_id in BASE_MODELS:
        LOGGER.info("Loading base model %s (%s) ...", name, model_id)
        pipe = hf_pipeline("audio-classification", model=model_id, device=device)
        loaded.append((name, pipe))
        LOGGER.info("  %s loaded", name)
    return loaded


def predict_audio(
    audio_path: str | Path,
    base_models: list[tuple[str, Any]],
    stacking_model,
    scaler,
) -> dict[str, Any]:
    audio_path = str(audio_path)
    base_scores: dict[str, float] = {}

    for name, pipe in base_models:
        try:
            result = pipe(audio_path)
            score = _extract_fake_probability(result)
        except Exception as exc:
            LOGGER.warning("Base model %s failed on %s: %s", name, audio_path, exc)
            score = 0.5
        base_scores[name] = score
        LOGGER.debug("  %s → %.4f", name, score)

    scores_array = np.array([base_scores[n] for n, _ in BASE_MODELS]).reshape(1, -1)
    scores_scaled = scaler.transform(scores_array)

    probabilities = stacking_model.predict_proba(scores_scaled)[0]

    fake_probability = float(probabilities[1])
    real_probability = float(probabilities[0])

    FAKE_THRESHOLD = 0.70
    REAL_THRESHOLD = 0.35

    if fake_probability >= FAKE_THRESHOLD:
        label      = "fake"
        confidence = fake_probability
    elif fake_probability <= REAL_THRESHOLD:
        label      = "real"
        confidence = real_probability
    else:
        label      = "uncertain"
        confidence = max(fake_probability, real_probability)

    _summary_map = {
        "fake":      f"Likely deepfake audio with {confidence * 100:.1f}% confidence",
        "real":      f"Likely real audio with {confidence * 100:.1f}% confidence",
        "uncertain": f"Cannot determine authenticity (fake score {fake_probability * 100:.1f}%)",
    }
    summary = f"{_summary_map[label]} ({len(base_models)} base model(s) used)."

    return {
        "label":            label,
        "confidence":       confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "base_scores":      base_scores,
        "summary":          summary,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m ai.audio_detector.inference <audio_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    device     = _resolve_device()

    print("Loading stacking model...")
    stacking, sc, names, meta = load_stacking_model()
    print(f"Meta-learner: accuracy={meta['accuracy']:.4f}")

    print("Loading base models (first run downloads weights, ~few minutes)...")
    models = load_base_models(device)

    result = predict_audio(audio_file, models, stacking, sc)

    print(f"\n{'='*40}")
    print(f"Verdict:        {result['label'].upper()}")
    print(f"Confidence:     {result['confidence']*100:.1f}%")
    print(f"Fake prob:      {result['fake_probability']*100:.1f}%")
    print(f"Base scores:")
    for name, score in result["base_scores"].items():
        print(f"  {name}: {score:.4f}")
