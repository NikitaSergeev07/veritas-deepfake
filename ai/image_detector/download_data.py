import argparse
import hashlib
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageFile
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SOURCES = [
    {
        "name": "Hemg/deepfake-and-real-images",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_values": [0, "0", "fake", "Fake", "FAKE"],
        "real_values": [1, "1", "real", "Real", "REAL"],
    },
    {
        "name": "itsLeen/deepfake_vs_real_image_detection",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_values": None,
        "real_values": None,
    },
    {
        "name": "date3k2/raw_real_fake_images",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_values": None,
        "real_values": None,
    },
]


def img_hash(img: Image.Image) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()[:12]


def download_source(cfg: dict, real_dir: Path, fake_dir: Path, max_per: int):
    name = cfg["name"]
    print(f"\nLoading {name} ...")
    try:
        ds = load_dataset(name, split=cfg["split"], streaming=False)
    except Exception as e:
        print(f"  SKIP: {e}")
        return 0, 0

    fake_vals = list(cfg["fake_values"]) if cfg["fake_values"] else []
    real_vals = list(cfg["real_values"]) if cfg["real_values"] else []

    label_col = cfg["label_col"]
    if label_col in ds.features:
        feat = ds.features[label_col]
        if hasattr(feat, "names"):
            names = feat.names
            print(f"  ClassLabel names: {names}")
            fake_kw = ("fake", "ai", "generated", "synth", "deepfake")
            real_kw = ("real", "authentic", "genuine", "original")
            fi = [i for i, n in enumerate(names) if any(k in n.lower() for k in fake_kw)]
            ri = [i for i, n in enumerate(names) if any(k in n.lower() for k in real_kw)]
            if fi and ri:
                fake_vals = fi + [str(i) for i in fi] + [names[i] for i in fi]
                real_vals = ri + [str(i) for i in ri] + [names[i] for i in ri]
                print(f"  → fake_idx={fi}, real_idx={ri}")
            else:
                print(f"  WARNING: cannot auto-detect labels from {names}")

    n_real = n_fake = 0
    pbar = tqdm(ds, desc=name.split("/")[-1], leave=False)
    for sample in pbar:
        if n_real >= max_per and n_fake >= max_per:
            break
        lbl = sample[label_col]
        img = sample[cfg["image_col"]]
        if not isinstance(img, Image.Image):
            continue
        img = img.convert("RGB")
        h = img_hash(img)

        if lbl in fake_vals and n_fake < max_per:
            path = fake_dir / f"hf_{h}.jpg"
            if not path.exists():
                img.save(path, "JPEG", quality=95)
            n_fake += 1
        elif lbl in real_vals and n_real < max_per:
            path = real_dir / f"hf_{h}.jpg"
            if not path.exists():
                img.save(path, "JPEG", quality=95)
            n_real += 1

        pbar.set_postfix(real=n_real, fake=n_fake)

    print(f"  Saved: {n_real} real, {n_fake} fake")
    return n_real, n_fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/image/train")
    parser.add_argument("--max-per-source", type=int, default=8000)
    args = parser.parse_args()

    real_dir = Path(args.data_dir) / "real"
    fake_dir = Path(args.data_dir) / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target: {args.data_dir}")
    print(f"Existing: real={len(list(real_dir.glob('*.jpg')))}, fake={len(list(fake_dir.glob('*.jpg')))}")

    total_r = total_f = 0
    for src in SOURCES:
        r, f = download_source(src, real_dir, fake_dir, args.max_per_source)
        total_r += r
        total_f += f

    print(f"\nDone. Added: {total_r} real, {total_f} fake")
    print(f"Total: real={len(list(real_dir.glob('*.jpg')))}, fake={len(list(fake_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
