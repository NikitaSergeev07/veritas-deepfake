import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import DeepfakeImageDataset, get_train_transform, get_val_transform
except ImportError:
    from dataset import DeepfakeImageDataset, get_train_transform, get_val_transform

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip(device):
    print("Loading CLIP ViT-B/16 ...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.visual
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    print("  CLIP loaded (86M params, 512-dim features)")
    return model


@torch.no_grad()
def extract(clip_model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = clip_model(imgs)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/image/train")
    parser.add_argument("--feat-dir",   default="data/image/features")
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument("--aug-copies", type=int,   default=2)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device   = get_device()
    feat_dir = Path(args.feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    full_ds = DeepfakeImageDataset(args.data_dir, transform=None)
    n       = len(full_ds)
    indices = list(range(n))
    random.shuffle(indices)
    val_n   = int(n * args.val_frac)
    val_idx = indices[:val_n]
    trn_idx = indices[val_n:]
    print(f"Split: {len(trn_idx)} train / {val_n} val")

    clip_model = load_clip(device)

    print(f"\nExtracting val features ({val_n} images) ...")
    val_transform = get_val_transform()
    val_files  = [full_ds.files[i]  for i in val_idx]
    val_labels = [full_ds.labels[i] for i in val_idx]

    class _SubsetDs(torch.utils.data.Dataset):
        def __init__(self, files, labels, transform):
            self.files, self.labels, self.transform = files, labels, transform
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            from PIL import Image as _I
            try: img = _I.open(self.files[i]).convert("RGB")
            except: img = _I.new("RGB", (224, 224))
            return self.transform(img), self.labels[i]

    val_loader = DataLoader(
        _SubsetDs(val_files, val_labels, val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    val_feats, val_lbls = extract(clip_model, val_loader, device)
    torch.save(val_feats, feat_dir / "val_features.pt")
    torch.save(val_lbls,  feat_dir / "val_labels.pt")
    print(f"  Saved val: {val_feats.shape}")

    trn_files  = [full_ds.files[i]  for i in trn_idx]
    trn_labels = [full_ds.labels[i] for i in trn_idx]

    all_train_feats, all_train_lbls = [], []
    for copy_idx in range(args.aug_copies):
        print(f"\nExtracting train features — aug copy {copy_idx+1}/{args.aug_copies} ...")
        trn_loader = DataLoader(
            _SubsetDs(trn_files, trn_labels, get_train_transform()),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        feats, lbls = extract(clip_model, tqdm(trn_loader, desc=f"copy {copy_idx+1}"), device)
        all_train_feats.append(feats)
        all_train_lbls.append(lbls)

    train_feats = torch.cat(all_train_feats)
    train_lbls  = torch.cat(all_train_lbls)
    torch.save(train_feats, feat_dir / "train_features.pt")
    torch.save(train_lbls,  feat_dir / "train_labels.pt")
    print(f"  Saved train: {train_feats.shape}  (real={int((train_lbls==0).sum())}, fake={int((train_lbls==1).sum())})")

    del clip_model
    if device.type == "mps":
        torch.mps.empty_cache()
    print("\nCLIP unloaded. Features ready in", feat_dir)


if __name__ == "__main__":
    main()
