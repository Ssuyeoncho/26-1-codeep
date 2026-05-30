"""Utilities, data loading, and neural network models for Phase 3."""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import CLASS_PAIRS, ExperimentConfig


# ── Device resolution ─────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available accelerator; validate explicit choices."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_str == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            print("WARNING: CUDA not available — using MPS (Apple Silicon) instead.")
            return "mps"
        raise RuntimeError(
            "GPU requested (--device cuda) but neither CUDA nor MPS is available."
        )
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine.")
        return "mps"
    if device_str == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device: {device_str!r}. Choose from: auto, cuda, mps, cpu.")


# ── Utilities ─────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pin_memory_for(device: str) -> bool:
    """pin_memory is only valid for CUDA; MPS and CPU don't support it."""
    return device == "cuda"


# ── VP forward process ────────────────────────────────────────────────────────

def vp_alpha_sigma(lam: Tensor) -> tuple[Tensor, Tensor]:
    return torch.sqrt(torch.sigmoid(lam)), torch.sqrt(torch.sigmoid(-lam))


def vp_alpha_sigma_np(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha_sq = 1.0 / (1.0 + np.exp(-lam))
    return np.sqrt(alpha_sq), np.sqrt(1.0 - alpha_sq)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TwoClassCIFAR(Dataset):
    def __init__(self, full_dataset: datasets.CIFAR10, class_a: int, class_b: int) -> None:
        self.dataset = full_dataset
        self.label_map = {class_a: 0, class_b: 1}
        # Use .targets directly to avoid iterating all images with transforms
        targets = torch.tensor(full_dataset.targets)
        self.indices = torch.where((targets == class_a) | (targets == class_b))[0].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img, label = self.dataset[self.indices[idx]]
        return img, self.label_map[label]


def get_two_class_dataset(config: ExperimentConfig, train: bool) -> TwoClassCIFAR:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    class_a, class_b = CLASS_PAIRS[config.class_pair]
    full = datasets.CIFAR10(config.data_root, train=train, download=True, transform=transform)
    return TwoClassCIFAR(full, class_a, class_b)


# ── CNN Classifier (phi features) ─────────────────────────────────────────────

class FeatureClassifier(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.penultimate = nn.Sequential(nn.Linear(128 * 16, feature_dim), nn.ReLU())
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.penultimate(self.backbone(x)))

    @torch.no_grad()
    def features(self, x: Tensor) -> Tensor:
        return self.penultimate(self.backbone(x))


def train_classifier(config: ExperimentConfig, device: str) -> FeatureClassifier:
    ds = get_two_class_dataset(config, train=True)
    loader = DataLoader(
        ds, batch_size=config.clf_batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=pin_memory_for(device),
    )
    model = FeatureClassifier(config.clf_feature_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.clf_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.clf_epochs)

    for epoch in range(config.clf_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)          # single forward pass — reused for loss and accuracy
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(imgs)
        sched.step()
        print(f"  [Classifier] epoch {epoch+1}/{config.clf_epochs}  "
              f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    model.eval()
    return model


# ── U-Net building blocks ─────────────────────────────────────────────────────

def _groupnorm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Find the largest valid group count ≤ max_groups that divides num_channels."""
    g = min(max_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


def sinusoidal_embedding(x: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=x.device) / half
    )
    args = x.view(-1, 1) * freqs.view(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = _groupnorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.norm2 = _groupnorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = _groupnorm(ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = nn.Conv1d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2))
        return x + self.proj(h).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ── U-Net (base_ch=64, ch_mult=[1,2,2,2], attn at 16×16) ────────────────────

class UNet(nn.Module):
    """DDPM-style U-Net for 32×32 CIFAR. Epsilon-prediction."""

    def __init__(self, in_ch: int = 3, base_ch: int = 64, num_res_blocks: int = 2) -> None:
        super().__init__()
        emb_dim = base_ch * 4
        c0, c1, c2, c3 = base_ch, base_ch * 2, base_ch * 2, base_ch * 2

        self.lam_embed = nn.Sequential(
            nn.Linear(base_ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.init_conv = nn.Conv2d(in_ch, c0, 3, padding=1)

        self.enc0 = self._level(c0, c0, emb_dim, num_res_blocks, attn=False)
        self.down0 = Downsample(c0)
        self.enc1 = self._level(c0, c1, emb_dim, num_res_blocks, attn=True)   # 16×16
        self.down1 = Downsample(c1)
        self.enc2 = self._level(c1, c2, emb_dim, num_res_blocks, attn=False)  # 8×8
        self.down2 = Downsample(c2)

        self.mid1 = ResBlock(c2, c3, emb_dim)
        self.mid_attn = SelfAttention(c3)
        self.mid2 = ResBlock(c3, c3, emb_dim)

        self.up2 = Upsample(c3)
        self.dec2 = self._level(c3 + c2, c2, emb_dim, num_res_blocks + 1, attn=False)
        self.up1 = Upsample(c2)
        self.dec1 = self._level(c2 + c1, c1, emb_dim, num_res_blocks + 1, attn=True)
        self.up0 = Upsample(c1)
        self.dec0 = self._level(c1 + c0, c0, emb_dim, num_res_blocks + 1, attn=False)

        self.out_norm = _groupnorm(c0)
        self.out_conv = nn.Conv2d(c0, in_ch, 3, padding=1)

    @staticmethod
    def _level(in_ch: int, out_ch: int, emb_dim: int, n: int, attn: bool) -> nn.ModuleList:
        blocks: list[nn.Module] = []
        for i in range(n):
            blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim))
            if attn:
                blocks.append(SelfAttention(out_ch))
        return nn.ModuleList(blocks)

    def _run(self, h: Tensor, blocks: nn.ModuleList, emb: Tensor) -> Tensor:
        for b in blocks:
            h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        return h

    def forward(self, x: Tensor, lam: Tensor) -> Tensor:
        sin_emb = sinusoidal_embedding(lam.view(-1), self.lam_embed[0].in_features)
        emb = self.lam_embed(sin_emb)

        h = self.init_conv(x)
        h = self._run(h, self.enc0, emb);      s32 = h
        h = self._run(self.down0(h), self.enc1, emb); s16 = h
        h = self._run(self.down1(h), self.enc2, emb); s8 = h
        h = self.down2(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        h = self._run(torch.cat([self.up2(h), s8], 1), self.dec2, emb)
        h = self._run(torch.cat([self.up1(h), s16], 1), self.dec1, emb)
        h = self._run(torch.cat([self.up0(h), s32], 1), self.dec0, emb)

        return self.out_conv(F.silu(self.out_norm(h)))


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        dtype = next(model.parameters()).dtype
        model.load_state_dict({k: v.to(dtype) for k, v in self.shadow.items()})
