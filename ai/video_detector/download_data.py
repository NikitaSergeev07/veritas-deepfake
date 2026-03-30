"""
Download video deepfake test clips from HuggingFace.

Source: faridlab/deepaction_v1  (real Pexels videos vs AI-generated, open access)

Usage:
  python -m ai.video_detector.download_data
  python -m ai.video_detector.download_data --max-per-class 30
"""

import argparse
import hashlib
import io
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/video")
    parser.add_argument("--max-per-class", type=int, default=20,
                        help="Max videos per class (default 20)")
    args = parser.parse_args()

    real_dir = Path(args.data_dir) / "real"
    fake_dir = Path(args.data_dir) / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target: {args.data_dir}")
    print(f"Existing: real={len(list(real_dir.glob('*.mp4')))}, fake={len(list(fake_dir.glob('*.mp4')))}")

    from datasets import load_dataset
    from tqdm import tqdm

    # deepaction_v1: real (Pexels) vs AI-generated video clips
    print("\nLoading faridlab/deepaction_v1 ...")
    try:
        ds = load_dataset(
            "faridlab/deepaction_v1",
            split="test",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Could not load split='test', trying split='train': {e}")
        try:
            ds = load_dataset(
                "faridlab/deepaction_v1",
                split="train",
                trust_remote_code=True,
            )
        except Exception as e2:
            print(f"  SKIP faridlab/deepaction_v1: {e2}")
            print("\nFallback: trying ductai199x/video-std-manip ...")
            _download_video_std_manip(real_dir, fake_dir, args.max_per_class)
            return

    n_real = n_fake = 0
    max_per = args.max_per_class

    # Inspect columns
    print(f"  Columns: {ds.column_names}")
    print(f"  Size: {len(ds)} samples")

    # Determine which column has the video and label
    video_col = None
    label_col = None
    for c in ds.column_names:
        cl = c.lower()
        if "video" in cl or "file" in cl or "path" in cl:
            video_col = c
        if "label" in cl or "type" in cl or "class" in cl or "source" in cl:
            label_col = c

    if not video_col or not label_col:
        print(f"  Cannot determine video/label columns from: {ds.column_names}")
        return

    print(f"  Using video_col={video_col}, label_col={label_col}")

    # Check label values
    sample_labels = set()
    for i in range(min(50, len(ds))):
        sample_labels.add(str(ds[i][label_col]))
    print(f"  Sample labels: {sample_labels}")

    # Determine which labels are real vs fake
    real_labels = set()
    fake_labels = set()
    real_kw = ("real", "pexels", "authentic", "original", "genuine")
    fake_kw = ("fake", "generated", "ai", "deepfake", "synthetic", "sora", "kling", "runway", "pika", "luma")

    for lbl in sample_labels:
        ll = lbl.lower()
        if any(k in ll for k in real_kw):
            real_labels.add(lbl)
        elif any(k in ll for k in fake_kw):
            fake_labels.add(lbl)

    # If no matches, try numeric: 0=real, 1=fake
    if not real_labels and not fake_labels:
        if "0" in sample_labels:
            real_labels.add("0")
        if "1" in sample_labels:
            fake_labels.add("1")

    print(f"  real_labels={real_labels}, fake_labels={fake_labels}")

    if not real_labels and not fake_labels:
        print("  Cannot determine real/fake labels. Aborting.")
        return

    pbar = tqdm(ds, desc="deepaction_v1", leave=False)
    for sample in pbar:
        if n_real >= max_per and n_fake >= max_per:
            break

        lbl = str(sample[label_col])
        video_val = sample[video_col]

        raw = _extract_video_bytes(video_val)
        if not raw:
            continue

        h = hashlib.md5(raw[:4096]).hexdigest()[:10]

        if lbl in real_labels and n_real < max_per:
            path = real_dir / f"da_{h}.mp4"
            if not path.exists():
                path.write_bytes(raw)
            n_real += 1
        elif lbl in fake_labels and n_fake < max_per:
            path = fake_dir / f"da_{h}.mp4"
            if not path.exists():
                path.write_bytes(raw)
            n_fake += 1

        pbar.set_postfix(real=n_real, fake=n_fake)

    print(f"\nDone. Saved: {n_real} real, {n_fake} fake")
    print(f"Total: real={len(list(real_dir.glob('*.mp4')))}, fake={len(list(fake_dir.glob('*.mp4')))}")


def _extract_video_bytes(video_val) -> bytes:
    """Extract raw bytes from a HuggingFace video column value."""
    if isinstance(video_val, bytes):
        return video_val

    if isinstance(video_val, str):
        p = Path(video_val)
        if p.exists():
            return p.read_bytes()
        return b""

    if isinstance(video_val, dict):
        raw = video_val.get("bytes")
        if raw:
            return raw
        path = video_val.get("path", "")
        if path and Path(path).exists():
            return Path(path).read_bytes()

    return b""


def _download_video_std_manip(real_dir: Path, fake_dir: Path, max_per: int):
    """Fallback: download from ductai199x/video-std-manip."""
    from datasets import load_dataset
    from tqdm import tqdm

    print("Loading ductai199x/video-std-manip (vcms) ...")
    try:
        ds = load_dataset("ductai199x/video-std-manip", "vcms", split="test", streaming=True)
    except Exception as e:
        print(f"  SKIP: {e}")
        return

    n_real = n_fake = 0
    for sample in tqdm(ds, desc="video-std-manip", leave=False):
        if n_real >= max_per and n_fake >= max_per:
            break

        label = sample.get("label", -1)
        video_val = sample.get("video") or sample.get("vid_path")
        raw = _extract_video_bytes(video_val)
        if not raw:
            continue

        import hashlib
        h = hashlib.md5(raw[:4096]).hexdigest()[:10]

        if label == 0 and n_real < max_per:
            (real_dir / f"vsm_{h}.mp4").write_bytes(raw)
            n_real += 1
        elif label == 1 and n_fake < max_per:
            (fake_dir / f"vsm_{h}.mp4").write_bytes(raw)
            n_fake += 1

    print(f"  Saved: {n_real} real, {n_fake} fake")


if __name__ == "__main__":
    main()
