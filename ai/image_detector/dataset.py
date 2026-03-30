import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class RandomJPEGCompression:

    def __init__(self, quality_range=(30, 95), p=0.4):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


class RandomGaussianNoise:

    def __init__(self, std_range=(0.005, 0.03), p=0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32) / 255.0
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


def _to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    import torchvision.transforms.functional as F
    t = F.to_tensor(img)
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std  = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (t - mean) / std


def get_train_transform():
    import torchvision.transforms as T

    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        RandomJPEGCompression(quality_range=(30, 95), p=0.4),
        RandomGaussianNoise(std_range=(0.005, 0.03), p=0.3),
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_val_transform():
    import torchvision.transforms as T

    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class DeepfakeImageDataset(Dataset):

    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        real_dir = Path(data_dir) / "real"
        fake_dir = Path(data_dir) / "fake"

        real_files = sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.png"))
        fake_files = sorted(fake_dir.glob("*.jpg")) + sorted(fake_dir.glob("*.png"))

        self.files  = real_files + fake_files
        self.labels = [0] * len(real_files) + [1] * len(fake_files)

        print(f"Dataset: {len(real_files)} real + {len(fake_files)} fake = {len(self.files)} total")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label


class CachedFeatureDataset(Dataset):

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert len(features) == len(labels)
        self.features = features
        self.labels   = labels.float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
