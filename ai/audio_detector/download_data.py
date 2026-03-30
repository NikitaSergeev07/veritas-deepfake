"""
Download audio deepfake datasets from HuggingFace.

Sources:
  - garystafford/deepfake-audio-detection  (the original training dataset,
    ElevenLabs/Kokoro/Polly/Hume fake voices vs real YouTube clips)
    label: "fake" / "real" string column

Usage:
  python -m ai.audio_detector.download_data
  python -m ai.audio_detector.download_data --max-per-source 500
"""

import argparse
import io
from pathlib import Path

SOURCES = [
    {
        "name": "garystafford/deepfake-audio-detection",
        "split": "train",
        "audio_col": "audio",
        "label_col": "label",
        "fake_values": ["fake", "FAKE", 0, "0"],
        "real_values": ["real", "REAL", 1, "1"],
    },
]


def audio_to_bytes(audio_val) -> tuple[bytes, str]:
    """Convert HuggingFace audio value to (bytes, extension)."""
    if isinstance(audio_val, bytes):
        return audio_val, ".flac"

    if isinstance(audio_val, str):
        p = Path(audio_val)
        if p.exists():
            return p.read_bytes(), p.suffix or ".flac"
        return b"", ".flac"

    if isinstance(audio_val, dict):
        # Try raw bytes first
        raw = audio_val.get("bytes")
        if raw:
            path = audio_val.get("path", "")
            ext = Path(path).suffix if path else ".flac"
            return raw, ext or ".flac"

        # Try path
        path = audio_val.get("path", "")
        if path and Path(path).exists():
            return Path(path).read_bytes(), Path(path).suffix or ".flac"

        # Convert numpy array → WAV bytes
        arr = audio_val.get("array")
        if arr is not None:
            try:
                import numpy as np
                import soundfile as sf
                arr = np.array(arr, dtype=np.float32)
                sr  = audio_val.get("sampling_rate", 16000)
                buf = io.BytesIO()
                sf.write(buf, arr, sr, format="WAV")
                return buf.getvalue(), ".wav"
            except Exception as e:
                print(f"    array→wav failed: {e}")

    return b"", ".flac"


def detect_labels(ds, label_col: str, fake_vals, real_vals):
    feat = ds.features.get(label_col)
    if feat is None:
        return fake_vals or [], real_vals or []
    if hasattr(feat, "names"):
        names = feat.names
        print(f"  ClassLabel names: {names}")
        fake_kw = ("fake", "spoof", "generated", "synth", "deepfake")
        real_kw = ("real", "authentic", "genuine", "bonafide", "original")
        fi = [i for i, n in enumerate(names) if any(k in n.lower() for k in fake_kw)]
        ri = [i for i, n in enumerate(names) if any(k in n.lower() for k in real_kw)]
        if fi and ri:
            f_vals = fi + [str(i) for i in fi] + [names[i] for i in fi]
            r_vals = ri + [str(i) for i in ri] + [names[i] for i in ri]
            print(f"  → fake={fi}, real={ri}")
            return f_vals, r_vals
        print(f"  WARNING: cannot auto-detect labels from {names}")
    return fake_vals or [], real_vals or []


def download_source(cfg: dict, real_dir: Path, fake_dir: Path, max_per: int) -> tuple[int, int]:
    from datasets import load_dataset
    from tqdm import tqdm

    name = cfg["name"]
    print(f"\nLoading {name} ...")
    try:
        ds = load_dataset(name, streaming=False)
        # Merge all splits into one iterable
        from datasets import concatenate_datasets
        splits = list(ds.values())
        ds_all = concatenate_datasets(splits)
        # Disable audio decoding — we want raw bytes
        import datasets as hf_datasets
        ds_all = ds_all.cast_column(cfg["audio_col"], hf_datasets.Audio(decode=False))
    except Exception as e:
        print(f"  SKIP: {e}")
        return 0, 0

    fake_vals, real_vals = detect_labels(ds_all, cfg["label_col"], cfg["fake_values"], cfg["real_values"])

    n_real = n_fake = 0
    pbar = tqdm(ds_all, desc=name.split("/")[-1], leave=False)

    for sample in pbar:
        if n_real >= max_per and n_fake >= max_per:
            break

        lbl       = sample[cfg["label_col"]]
        audio_val = sample[cfg["audio_col"]]
        raw, ext  = audio_to_bytes(audio_val)

        if not raw:
            continue

        # Use original filename if available, else generate one
        orig_path = audio_val.get("path", "") if isinstance(audio_val, dict) else ""
        if orig_path:
            stem = Path(orig_path).stem
            fname = f"{stem}{ext}"
        else:
            import hashlib
            h = hashlib.md5(raw).hexdigest()[:12]
            fname = f"gs_{h}{ext}"

        if lbl in fake_vals and n_fake < max_per:
            path = fake_dir / fname
            if not path.exists():
                path.write_bytes(raw)
            n_fake += 1
        elif lbl in real_vals and n_real < max_per:
            path = real_dir / fname
            if not path.exists():
                path.write_bytes(raw)
            n_real += 1

        pbar.set_postfix(real=n_real, fake=n_fake)

    print(f"  Saved: {n_real} real, {n_fake} fake")
    return n_real, n_fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",        default="data/audio")
    parser.add_argument("--max-per-source",  type=int, default=500,
                        help="Max clips per class per source (default 500)")
    args = parser.parse_args()

    real_dir = Path(args.data_dir) / "real"
    fake_dir = Path(args.data_dir) / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target: {args.data_dir}")
    print(f"Existing: real={len(list(real_dir.glob('*')))}, fake={len(list(fake_dir.glob('*')))}")

    total_r = total_f = 0
    for src in SOURCES:
        r, f = download_source(src, real_dir, fake_dir, args.max_per_source)
        total_r += r
        total_f += f

    print(f"\nDone. Added: {total_r} real, {total_f} fake")
    print(f"Total: real={len(list(real_dir.glob('*')))}, fake={len(list(fake_dir.glob('*')))}")
    print(f"\nTest: upload any file from {args.data_dir}/fake/ to the audio detector.")


if __name__ == "__main__":
    main()
