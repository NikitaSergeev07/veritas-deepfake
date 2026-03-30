"""
DeepfakeDetector: CLIP ViT-B/16 (frozen) + MLP classification head.

Based on UniversalFakeDetect (CVPR 2023) — frozen CLIP features + lightweight classifier.
Architecture: LayerNorm(512) → Linear(512,256) → GELU → Dropout(0.3) → Linear(256,1)
"""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """132K-parameter classification head on top of 512-dim CLIP features."""

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, feature_dim) → logits (B, 1)"""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probability (B,)"""
        return torch.sigmoid(self.forward(x)).squeeze(-1)
