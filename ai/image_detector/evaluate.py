import argparse
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/test")
    parser.add_argument("--checkpoint", default="ai/image_detector/checkpoints/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
        from .inference import get_device, load_model
        from .dataset import DeepfakeImageDataset, get_val_transform
    except ImportError:
        from inference import get_device, load_model
        from dataset import DeepfakeImageDataset, get_val_transform

    device = get_device()
    clip_visual, mlp, threshold = load_model(args.checkpoint, device)

    ds = DeepfakeImageDataset(args.data_dir, transform=get_val_transform())
    if len(ds) == 0:
        print("No test images found in", args.data_dir)
        return

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            feats = clip_visual(imgs.to(device))
            probs = mlp.predict_proba(feats).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    cm  = confusion_matrix(labels, preds)

    print(f"\n{'='*50}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold:  {threshold:.2f}")
    print(f"Samples:    {len(labels)}")
    print(f"AUC-ROC:    {auc:.4f}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  [real→real={cm[0,0]}, real→fake={cm[0,1]}]")
    print(f"  [fake→real={cm[1,0]}, fake→fake={cm[1,1]}]")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=["real", "fake"]))


if __name__ == "__main__":
    main()
