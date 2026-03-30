import os
from pathlib import Path

import torch
import open_clip
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    from .model import MLPHead
    from .dataset import get_val_transform
except ImportError:
    from model import MLPHead
    from dataset import get_val_transform


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feature_dim = ckpt.get("feature_dim", 512)
    threshold   = ckpt.get("threshold", 0.5)
    clip_name   = ckpt.get("clip_model", "ViT-B-16")

    clip_model, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained="openai")
    clip_visual = clip_model.visual.eval().to(device)
    clip_visual.requires_grad_(False)

    mlp = MLPHead(feature_dim=feature_dim)
    mlp.load_state_dict(ckpt["state_dict"])
    mlp = mlp.eval().to(device)

    val_auc = ckpt.get("val_auc", None)
    print(f"Model loaded: {clip_name} | threshold={threshold:.2f}" +
          (f" | val_auc={val_auc:.4f}" if val_auc else ""))
    return clip_visual, mlp, threshold


def detect_face(image: Image.Image, margin: float = 0.30):
    try:
        import mediapipe as mp
        import numpy as np

        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        rgb = np.array(image.convert("RGB"))
        results = detector.process(rgb)
        detector.close()

        if not results.detections:
            return None

        det = max(results.detections, key=lambda d: d.score[0])
        bb  = det.location_data.relative_bounding_box
        w, h = image.size
        x1 = max(0, int((bb.xmin - margin * bb.width) * w))
        y1 = max(0, int((bb.ymin - margin * bb.height) * h))
        x2 = min(w, int((bb.xmin + (1 + margin) * bb.width) * w))
        y2 = min(h, int((bb.ymin + (1 + margin) * bb.height) * h))
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))
    except Exception:
        return None


@torch.no_grad()
def predict(image: Image.Image, clip_visual, mlp, threshold: float, device: torch.device):
    transform = get_val_transform()

    face_crop = detect_face(image)
    face_detected = face_crop is not None
    input_img = face_crop if face_detected else image

    tensor = transform(input_img.convert("RGB")).unsqueeze(0).to(device)
    feats  = clip_visual(tensor)
    prob   = float(mlp.predict_proba(feats).cpu())

    return {
        "verdict":       "FAKE" if prob >= threshold else "REAL",
        "confidence":    prob if prob >= threshold else 1 - prob,
        "fake_prob":     prob,
        "face_detected": face_detected,
        "face_crop":     face_crop,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m ai.image_detector.inference <image_path> <checkpoint_path>")
        sys.exit(1)

    img_path  = sys.argv[1]
    ckpt_path = sys.argv[2]
    device    = get_device()

    clip_visual, mlp, threshold = load_model(ckpt_path, device)
    image  = Image.open(img_path).convert("RGB")
    result = predict(image, clip_visual, mlp, threshold, device)

    print(f"\n{'='*40}")
    print(f"Verdict:        {result['verdict']}")
    print(f"Confidence:     {result['confidence']*100:.1f}%")
    print(f"Fake prob:      {result['fake_prob']*100:.1f}%")
    print(f"Face detected:  {result['face_detected']}")
