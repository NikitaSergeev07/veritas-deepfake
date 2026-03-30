import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from .dataset import CachedFeatureDataset
    from .model import MLPHead
except ImportError:
    from dataset import CachedFeatureDataset
    from model import MLPHead

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_features(feat_dir: Path, device: torch.device):
    print(f"Loading features from {feat_dir} ...")
    train_feats = torch.load(feat_dir / "train_features.pt", map_location=device, weights_only=True)
    train_lbls  = torch.load(feat_dir / "train_labels.pt",   map_location=device, weights_only=True)
    val_feats   = torch.load(feat_dir / "val_features.pt",   map_location=device, weights_only=True)
    val_lbls    = torch.load(feat_dir / "val_labels.pt",     map_location=device, weights_only=True)
    print(f"  Train: {train_feats.shape} | Val: {val_feats.shape}")
    print(f"  Train real={int((train_lbls==0).sum())}, fake={int((train_lbls==1).sum())}")
    return train_feats, train_lbls, val_feats, val_lbls


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict]:
    best_thr, best_ba = 0.5, 0.0
    for thr in np.arange(0.01, 1.0, 0.01):
        preds = (probs >= thr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tpr = tp / (tp + fn + 1e-9)
        tnr = tn / (tn + fp + 1e-9)
        ba  = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    preds = (probs >= best_thr).astype(int)
    acc   = (preds == labels).mean()
    return best_thr, {"balanced_acc": best_ba, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir",    default="data/image/features")
    parser.add_argument("--ckpt-dir",    default="ai/image_detector/checkpoints")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--weight-decay",type=float, default=0.01)
    parser.add_argument("--patience",    type=int,   default=10)
    args = parser.parse_args()

    device   = get_device()
    feat_dir = Path(args.feat_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_feats, train_lbls, val_feats, val_lbls = load_features(feat_dir, device)

    feature_dim = train_feats.shape[1]
    print(f"Feature dim: {feature_dim}")

    n_fake = int((train_lbls == 1).sum())
    n_real = int((train_lbls == 0).sum())
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    print(f"pos_weight (real/fake): {pos_weight.item():.3f}")

    train_ds = CachedFeatureDataset(train_feats, train_lbls)
    val_ds   = CachedFeatureDataset(val_feats,   val_lbls)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = MLPHead(feature_dim=feature_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc    = 0.0
    best_state  = None
    no_improve  = 0

    print(f"\nTraining MLP head for up to {args.epochs} epochs (patience={args.patience}) ...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for feats, labels in train_loader:
            optimizer.zero_grad()
            logits = model(feats).squeeze(-1)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(feats)
        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        model.eval()
        all_probs, all_lbls = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                probs = model.predict_proba(feats)
                all_probs.extend(probs.cpu().tolist())
                all_lbls.extend(labels.cpu().tolist())

        auc = roc_auc_score(all_lbls, all_probs)
        acc = ((np.array(all_probs) >= 0.5) == np.array(all_lbls)).mean()

        marker = ""
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker     = " ← best"
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | val_auc={auc:.4f} | val_acc={acc:.4f}{marker}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest val AUC: {best_auc:.4f}")
    model.load_state_dict(best_state)
    model.eval()

    all_probs, all_lbls = [], []
    with torch.no_grad():
        for feats, labels in val_loader:
            probs = model.predict_proba(feats)
            all_probs.extend(probs.cpu().tolist())
            all_lbls.extend(labels.cpu().tolist())

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_lbls)
    opt_thr, metrics = find_optimal_threshold(labels_arr, probs_arr)

    print(f"Optimal threshold: {opt_thr:.2f}")
    print(f"  balanced_acc: {metrics['balanced_acc']:.4f}")
    print(f"  accuracy:     {metrics['accuracy']:.4f}")

    ckpt_path = ckpt_dir / "best_model.pth"
    torch.save({
        "state_dict":  best_state,
        "threshold":   opt_thr,
        "clip_model":  "ViT-B-16",
        "feature_dim": feature_dim,
        "val_auc":     best_auc,
        "val_acc":     float(metrics["accuracy"]),
    }, ckpt_path)
    print(f"\nSaved: {ckpt_path}")


if __name__ == "__main__":
    main()
