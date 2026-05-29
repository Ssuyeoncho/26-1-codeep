"""Utilities, data loading, and neural network models for Phase 2."""
from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig


# ── Utilities ─────────────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> str:
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def vp_alpha_sigma(lam: Tensor) -> tuple[Tensor, Tensor]:
    return torch.sqrt(torch.sigmoid(lam)), torch.sqrt(torch.sigmoid(-lam))


def vp_alpha_sigma_np(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha_sq = 1.0 / (1.0 + np.exp(-lam))
    sigma_sq = 1.0 / (1.0 + np.exp(lam))
    return np.sqrt(alpha_sq), np.sqrt(sigma_sq)


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_mnist_two_class(config: ExperimentConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return train_imgs, train_labels, test_imgs, test_labels in [-1, 1]."""
    try:
        import torchvision.datasets as dsets
    except ImportError:
        raise ImportError("torchvision is required. Install with: pip install torchvision")

    d0, d1 = config.digits

    def extract(dataset) -> tuple[Tensor, Tensor]:
        imgs = dataset.data.float().unsqueeze(1) / 255.0 * 2.0 - 1.0  # [-1, 1]
        labels = dataset.targets
        mask = (labels == d0) | (labels == d1)
        imgs = imgs[mask]
        orig = labels[mask]
        remapped = torch.where(orig == d0, torch.zeros_like(orig), torch.ones_like(orig))
        return imgs, remapped

    def _load(train: bool) -> dsets.MNIST:
        # Try local disk first to avoid network instability.
        try:
            return dsets.MNIST(config.data_root, train=train, download=False)
        except RuntimeError:
            pass
        try:
            return dsets.MNIST(config.data_root, train=train, download=True)
        except Exception as e:
            split = "train" if train else "test"
            raise RuntimeError(
                f"Failed to load MNIST ({split}) from '{config.data_root}'.\n"
                "If the download keeps failing, manually place the four .gz files in "
                f"'{config.data_root}/MNIST/raw/' and re-run.\n"
                f"Original error: {e}"
            ) from e

    tr_imgs, tr_labels = extract(_load(train=True))
    te_imgs, te_labels = extract(_load(train=False))
    print(f"[Phase 2] MNIST {d0} vs {d1}: train={len(tr_imgs)}, test={len(te_imgs)}")
    return tr_imgs, tr_labels, te_imgs, te_labels


# ── Neural Network Models ──────────────────────────────────────────────────────

def sinusoidal_embedding(lam: Tensor, dim: int = 128) -> Tensor:
    """lam: (B,) → (B, dim)."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=lam.device) / half
    )
    x = lam.float().view(-1, 1) * freqs.view(1, -1)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


def _groupnorm(num_channels: int) -> nn.GroupNorm:
    num_groups = 8
    while num_channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = _groupnorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = _groupnorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t).view(-1, h.shape[1], 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class MiniUNet(nn.Module):
    """2-resolution-level U-Net for 28×28 MNIST. Pads input to 32×32 internally."""

    def __init__(self, base_ch: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        ch = base_ch
        self.time_dim = time_dim
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.in_conv = nn.Conv2d(1, ch, 3, padding=1)
        self.enc0    = ResBlock(ch,     ch,     time_dim)           # (B, ch,  32,32) → skip
        self.down0   = nn.Conv2d(ch, ch, 3, stride=2, padding=1)   # (B, ch,  16,16)
        self.enc1    = ResBlock(ch,     ch * 2, time_dim)           # (B, 2ch, 16,16)
        self.mid     = ResBlock(ch * 2, ch * 2, time_dim)           # (B, 2ch, 16,16)
        self.up1     = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)  # (B, ch,  32,32)
        self.dec1    = ResBlock(ch * 2, ch,     time_dim)           # concat skip → ch
        self.out_norm = _groupnorm(ch)
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x: Tensor, lam: Tensor) -> Tensor:
        lam_flat = lam.view(-1)
        t = self.time_emb(sinusoidal_embedding(lam_flat, self.time_dim))

        x  = F.pad(x, (2, 2, 2, 2))                         # (B,1,28,28) → (B,1,32,32)
        h  = self.in_conv(x)                                 # (B, ch, 32,32)
        h0 = self.enc0(h, t)                                 # (B, ch, 32,32) — skip
        h  = self.enc1(self.down0(h0), t)                    # (B, 2ch, 16,16)
        h  = self.mid(h, t)                                  # (B, 2ch, 16,16)
        h  = self.dec1(torch.cat([self.up1(h), h0], 1), t)   # (B, ch,  32,32)
        h  = self.out_conv(F.silu(self.out_norm(h)))         # (B, 1,   32,32)
        return h[:, :, 2:-2, 2:-2]                           # (B, 1,   28,28)


class FeatureClassifier(nn.Module):
    """Two-class CNN classifier. Penultimate layer used as φ(x)."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 14×14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 7×7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, feature_dim), nn.ReLU(),
        )
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.feature_net(x))

    @torch.no_grad()
    def get_features(self, x: Tensor) -> Tensor:
        return self.feature_net(x)


def train_feature_classifier(
    train_imgs: Tensor,
    train_labels: Tensor,
    test_imgs: Tensor,
    test_labels: Tensor,
    config: ExperimentConfig,
) -> FeatureClassifier:
    clf = FeatureClassifier(config.clf_feature_dim).to(config.device)
    opt = torch.optim.Adam(clf.parameters(), lr=config.clf_lr)
    loader = DataLoader(
        TensorDataset(train_imgs, train_labels),
        batch_size=config.clf_batch_size, shuffle=True,
    )
    clf.train()
    for epoch in range(config.clf_epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(config.device), labels.to(config.device)
            loss = F.cross_entropy(clf(imgs), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        logits = clf(test_imgs.to(config.device))
        acc = (logits.argmax(1) == test_labels.to(config.device)).float().mean().item()
    print(f"[Phase 2] Classifier test accuracy: {acc:.4f}")
    return clf
