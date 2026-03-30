from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("video_detector")

MODEL_ID = "Vansh180/VideoMae-ffc23-deepfake-detector"
NUM_FRAMES = 16
EXTRACT_FPS = 2

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.35


def _resolve_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def extract_frames(video_path: str, fps: int = EXTRACT_FPS) -> list[Image.Image]:
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = os.path.join(tmpdir, "frame_%06d.png")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", pattern],
            check=True, capture_output=True,
        )
        frame_files = sorted(Path(tmpdir).glob("frame_*.png"))
        frames = [Image.open(f).convert("RGB") for f in frame_files]
    return frames


def _crop_face(image: Image.Image, expand: float = 1.4) -> Image.Image:
    try:
        import mediapipe as mp
        arr = np.array(image)
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4,
        ) as detector:
            results = detector.process(arr)
            if results.detections:
                bb = results.detections[0].location_data.relative_bounding_box
                h, w = arr.shape[:2]
                cx = int((bb.xmin + bb.width / 2) * w)
                cy = int((bb.ymin + bb.height / 2) * h)
                size = int(max(bb.width * w, bb.height * h) * expand)
                half = size // 2
                x1 = max(cx - half, 0)
                y1 = max(cy - half, 0)
                x2 = min(cx + half, w)
                y2 = min(cy + half, h)
                return image.crop((x1, y1, x2, y2))
    except Exception:
        pass
    return _center_crop(image)


def _center_crop(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def crop_faces(frames: list[Image.Image]) -> list[Image.Image]:
    if not frames:
        return frames
    ref_frame = frames[len(frames) // 2]
    try:
        import mediapipe as mp
        arr = np.array(ref_frame)
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4,
        ) as detector:
            results = detector.process(arr)
            if results.detections:
                bb = results.detections[0].location_data.relative_bounding_box
                h, w = arr.shape[:2]
                cx_rel = bb.xmin + bb.width / 2
                cy_rel = bb.ymin + bb.height / 2
                size_rel = max(bb.width, bb.height) * 1.4
                cropped = []
                for frame in frames:
                    fw, fh = frame.size
                    cx = int(cx_rel * fw)
                    cy = int(cy_rel * fh)
                    size = int(size_rel * max(fw, fh))
                    half = size // 2
                    x1 = max(cx - half, 0)
                    y1 = max(cy - half, 0)
                    x2 = min(cx + half, fw)
                    y2 = min(cy + half, fh)
                    cropped.append(frame.crop((x1, y1, x2, y2)))
                return cropped
    except Exception:
        pass
    return [_center_crop(f) for f in frames]


def load_model(device: str = "cpu"):
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    LOGGER.info("Loading VideoMAE from %s ...", MODEL_ID)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
    model.eval()
    if device != "cpu":
        model.to(device)
    LOGGER.info("VideoMAE loaded on %s", device)
    return model, processor


def _sample(frames: list[Image.Image], n: int = NUM_FRAMES) -> list[Image.Image]:
    if len(frames) <= n:
        out = list(frames)
        while len(out) < n:
            out.append(out[-1].copy())
        return out
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def _classify(
    frames: list[Image.Image],
    model,
    processor,
    device: str = "cpu",
) -> float:
    import torch

    sampled = _sample(frames, NUM_FRAMES)
    inputs = processor(sampled, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    return float(probs[1])


def predict_video(
    video_path: str,
    model,
    processor,
    device: str = "cpu",
) -> dict[str, Any]:
    video_path = str(video_path)
    duration = get_video_duration(video_path)

    LOGGER.info("Extracting frames from %s (%.1fs) ...", video_path, duration)
    raw_frames = extract_frames(video_path, fps=EXTRACT_FPS)

    if not raw_frames:
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "duration": duration,
            "frames_analyzed": 0,
            "segments": [],
            "summary": "Could not extract frames from video.",
        }

    frames = crop_faces(raw_frames)
    total_frames = len(frames)
    LOGGER.info("Extracted %d frames, running VideoMAE ...", total_frames)

    fake_prob = _classify(frames, model, processor, device)
    real_prob = 1.0 - fake_prob

    if fake_prob >= FAKE_THRESHOLD:
        label = "fake"
        confidence = fake_prob
    elif fake_prob <= REAL_THRESHOLD:
        label = "real"
        confidence = real_prob
    else:
        label = "uncertain"
        confidence = max(fake_prob, real_prob)

    segments: list[dict[str, Any]] = []
    if total_frames >= NUM_FRAMES * 2:
        n_segments = min(total_frames // NUM_FRAMES, 6)
        seg_size = total_frames // n_segments

        for i in range(n_segments):
            s = i * seg_size
            e = min(s + seg_size, total_frames)
            seg_frames = frames[s:e]
            if len(seg_frames) < 4:
                continue

            seg_fake = _classify(seg_frames, model, processor, device)
            start_sec = round(s / EXTRACT_FPS, 1)
            end_sec = round(e / EXTRACT_FPS, 1)

            seg_label = (
                "fake" if seg_fake >= FAKE_THRESHOLD
                else ("real" if seg_fake <= REAL_THRESHOLD else "uncertain")
            )
            segments.append({
                "start": start_sec,
                "end": end_sec,
                "fake_probability": round(seg_fake, 4),
                "label": seg_label,
            })

    _summary_map = {
        "fake":      f"Likely deepfake video with {confidence * 100:.1f}% confidence",
        "real":      f"Likely authentic video with {confidence * 100:.1f}% confidence",
        "uncertain": f"Cannot determine authenticity (fake score {fake_prob * 100:.1f}%)",
    }
    summary = f"{_summary_map[label]} ({total_frames} frames analyzed)."

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "duration": round(duration, 1),
        "frames_analyzed": total_frames,
        "segments": segments,
        "summary": summary,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m ai.video_detector.inference <video_path>")
        sys.exit(1)

    video_file = sys.argv[1]
    device = _resolve_device()

    print(f"Loading VideoMAE ({MODEL_ID}) ...")
    mdl, proc = load_model(device)

    result = predict_video(video_file, mdl, proc, device)

    print(f"\n{'='*40}")
    print(f"Verdict:    {result['label'].upper()}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Fake prob:  {result['fake_probability']*100:.1f}%")
    print(f"Duration:   {result['duration']}s")
    print(f"Frames:     {result['frames_analyzed']}")

    if result["segments"]:
        print(f"\nSegments:")
        for seg in result["segments"]:
            print(f"  {seg['start']:.1f}s – {seg['end']:.1f}s: "
                  f"{seg['label']} ({seg['fake_probability']*100:.1f}%)")
