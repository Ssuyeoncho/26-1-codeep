#!/usr/bin/env python3
"""Phase 3: CIFAR-10 Two-Class — DMSR-guided noise schedule experiment.

Self-contained pipeline:
  1. Load two-class CIFAR-10 (airplane vs automobile by default).
  2. Train two-class CNN classifier; fix penultimate-layer features phi.
  3. Compute empirical DMSR_phi(lambda) on lambda grid -> lambda_R* and T_R.
  4. Train small DDPM U-Net under each p_train(lambda) distribution.
  5. Generate with DDIM 50-step VP sampler (fixed for all runs).
  6. Compute FID, classifier confidence, per-lambda MSE, M/S coverage.
  7. Save CSV, JSON, Markdown, and plot artefacts.

Usage:
    python phase3/phase3_cifar_experiment.py --preset smoke   # sanity check
    python phase3/phase3_cifar_experiment.py --preset fast    # 50k steps
    python phase3/phase3_cifar_experiment.py --preset full    # 100k steps
    python phase3/phase3_cifar_experiment.py --device cuda    # GPU
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "results" / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

Tensor = torch.Tensor
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_DIR / "results"

CIFAR_CLASS_IDX = {
    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
    "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9,
}
CLASS_PAIRS: dict[str, tuple[int, int]] = {
    "airplane_vs_automobile": (0, 1),
    "cat_vs_dog": (3, 5),
    "deer_vs_horse": (4, 7),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    class_pair: str = "airplane_vs_automobile"
    image_size: int = 32
    lambda_min: float = -15.0
    lambda_max: float = 15.0
    rho: float = 0.5
    # Classifier
    clf_epochs: int = 20
    clf_batch_size: int = 256
    clf_lr: float = 1e-3
    clf_feature_dim: int = 128
    # DMSR
    dmsr_grid_size: int = 50
    dmsr_n_samples: int = 1000
    # Diffusion training
    train_steps: int = 100_000
    batch_size: int = 128
    lr: float = 2e-4
    ema_decay: float = 0.9999
    base_ch: int = 64
    num_res_blocks: int = 2
    # Eval
    eval_grid_size: int = 40
    eval_n_samples: int = 512
    n_gen_samples: int = 10_000
    ddim_steps: int = 50
    # Schedule sweep
    s_values: tuple[float, ...] = (1.5, 0.8, 0.3)
    laplace_b: float = 0.5
    # Misc
    seed: int = 20260526
    num_seeds: int = 1
    device: str = "cuda"
    num_workers: int = 0
    data_root: str = str(PROJECT_DIR / "data")
    run_name: str = "phase3_cifar"
    compute_fid: bool = True


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    kind: str
    center_lambda: float | None = None
    scale: float | None = None
    note: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_mpl_config_dir(out_dir: Path) -> None:
    (out_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# VP forward process
# ---------------------------------------------------------------------------

def vp_alpha_sigma(lam: Tensor) -> tuple[Tensor, Tensor]:
    return torch.sqrt(torch.sigmoid(lam)), torch.sqrt(torch.sigmoid(-lam))


def vp_alpha_sigma_np(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha_sq = 1.0 / (1.0 + np.exp(-lam))
    return np.sqrt(alpha_sq), np.sqrt(1.0 - alpha_sq)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TwoClassCIFAR(Dataset):
    def __init__(self, full_dataset: datasets.CIFAR10, class_a: int, class_b: int) -> None:
        self.dataset = full_dataset
        self.label_map = {class_a: 0, class_b: 1}
        self.indices = [i for i, (_, y) in enumerate(full_dataset) if y in self.label_map]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img, label = self.dataset[self.indices[idx]]
        return img, self.label_map[label]


def get_two_class_dataset(
    config: ExperimentConfig, train: bool
) -> TwoClassCIFAR:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    class_a, class_b = CLASS_PAIRS[config.class_pair]
    full = datasets.CIFAR10(config.data_root, train=train, download=True, transform=transform)
    return TwoClassCIFAR(full, class_a, class_b)


# ---------------------------------------------------------------------------
# CNN Classifier (phi features)
# ---------------------------------------------------------------------------

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
    loader = DataLoader(ds, batch_size=config.clf_batch_size, shuffle=True,
                        num_workers=config.num_workers, pin_memory=(device != "cpu"))
    model = FeatureClassifier(config.clf_feature_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.clf_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.clf_epochs)

    for epoch in range(config.clf_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = F.cross_entropy(model(imgs), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(imgs)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += len(imgs)
        sched.step()
        print(f"  [Classifier] epoch {epoch+1}/{config.clf_epochs}  "
              f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

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
        g1 = min(32, in_ch)
        g2 = min(32, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(g2, out_ch)
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
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = nn.Conv1d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h).chunk(3, dim=1)
        q, k, v = qkv
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


# ---------------------------------------------------------------------------
# U-Net  (base_ch=64, ch_mult=[1,2,2,2], attn at 16x16)
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Small DDPM-style U-Net for 32x32 CIFAR. Epsilon-prediction."""

    def __init__(self, in_ch: int = 3, base_ch: int = 64, num_res_blocks: int = 2) -> None:
        super().__init__()
        emb_dim = base_ch * 4
        c0 = base_ch
        c1 = base_ch * 2
        c2 = base_ch * 2
        c3 = base_ch * 2

        self.lam_embed = nn.Sequential(
            nn.Linear(base_ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.init_conv = nn.Conv2d(in_ch, c0, 3, padding=1)

        # Encoder: 32x32 -> 16x16 -> 8x8 -> (bottleneck 4x4)
        self.enc0 = self._level(c0, c0, emb_dim, num_res_blocks, attn=False)
        self.down0 = Downsample(c0)
        self.enc1 = self._level(c0, c1, emb_dim, num_res_blocks, attn=True)   # 16x16
        self.down1 = Downsample(c1)
        self.enc2 = self._level(c1, c2, emb_dim, num_res_blocks, attn=False)  # 8x8
        self.down2 = Downsample(c2)

        # Bottleneck 4x4
        self.mid1 = ResBlock(c2, c3, emb_dim)
        self.mid_attn = SelfAttention(c3)
        self.mid2 = ResBlock(c3, c3, emb_dim)

        # Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.up2 = Upsample(c3)
        self.dec2 = self._level(c3 + c2, c2, emb_dim, num_res_blocks + 1, attn=False)
        self.up1 = Upsample(c2)
        self.dec1 = self._level(c2 + c1, c1, emb_dim, num_res_blocks + 1, attn=True)
        self.up0 = Upsample(c1)
        self.dec0 = self._level(c1 + c0, c0, emb_dim, num_res_blocks + 1, attn=False)

        self.out_norm = nn.GroupNorm(min(32, c0), c0)
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
        h = self._run(h, self.enc0, emb); s32 = h         # 32x32, c0
        h = self._run(self.down0(h), self.enc1, emb); s16 = h  # 16x16, c1
        h = self._run(self.down1(h), self.enc2, emb); s8 = h   # 8x8, c2
        h = self.down2(h)                                   # 4x4, c2

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)                               # 4x4, c3

        h = self._run(torch.cat([self.up2(h), s8], 1), self.dec2, emb)   # 8x8
        h = self._run(torch.cat([self.up1(h), s16], 1), self.dec1, emb)  # 16x16
        h = self._run(torch.cat([self.up0(h), s32], 1), self.dec0, emb)  # 32x32

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Schedule samplers
# ---------------------------------------------------------------------------

def clamp_lambda(lam: Tensor, cfg: ExperimentConfig) -> Tensor:
    return lam.clamp(cfg.lambda_min, cfg.lambda_max)


def sample_schedule(spec: ScheduleSpec, n: int, cfg: ExperimentConfig, device: str) -> Tensor:
    eps = 1e-5
    if spec.kind == "dmsr_normal":
        lam = spec.center_lambda + spec.scale * torch.randn(n, 1, device=device)
        return clamp_lambda(lam, cfg)
    if spec.kind == "dmsr_laplace":
        d = torch.distributions.Laplace(spec.center_lambda, spec.scale)
        return clamp_lambda(d.sample((n, 1)).to(device), cfg)
    if spec.kind == "hang_laplace":
        d = torch.distributions.Laplace(0.0, spec.scale)
        return clamp_lambda(d.sample((n, 1)).to(device), cfg)
    if spec.kind == "cosine_vp":
        t = torch.rand(n, 1, device=device).clamp(eps, 1.0 - eps)
        return clamp_lambda(-2.0 * torch.log(torch.tan(0.5 * math.pi * t)), cfg)
    raise ValueError(f"Unknown schedule kind: {spec.kind!r}")


def build_schedules(lambda_r_star: float, cfg: ExperimentConfig) -> list[ScheduleSpec]:
    specs = [
        ScheduleSpec("cosine_vp", "cosine_vp",
                     note="VP cosine schedule induced density."),
        ScheduleSpec("hang_laplace_b0.5", "hang_laplace", scale=0.5,
                     note="Laplace(0, 0.5) — Hang et al. baseline."),
    ]
    for s in cfg.s_values:
        specs.append(ScheduleSpec(
            f"dmsr_normal_s{s}", "dmsr_normal",
            center_lambda=lambda_r_star, scale=s,
            note=f"N(lambda_R*={lambda_r_star:.2f}, s={s})."))
    specs.append(ScheduleSpec(
        f"dmsr_laplace_b{cfg.laplace_b}", "dmsr_laplace",
        center_lambda=lambda_r_star, scale=cfg.laplace_b,
        note=f"Laplace(lambda_R*={lambda_r_star:.2f}, b={cfg.laplace_b})."))
    return specs


# ---------------------------------------------------------------------------
# DMSR_phi computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dmsr_phi(
    classifier: FeatureClassifier,
    cfg: ExperimentConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Compute empirical DMSR_phi(lambda) and return (grid, dmsr, slope, low, center, high)."""
    ds = get_two_class_dataset(cfg, train=True)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=cfg.num_workers)

    imgs_a_list, imgs_b_list = [], []
    for imgs, labels in loader:
        imgs_a_list.append(imgs[labels == 0])
        imgs_b_list.append(imgs[labels == 1])
    imgs_a = torch.cat(imgs_a_list)[:cfg.dmsr_n_samples]
    imgs_b = torch.cat(imgs_b_list)[:cfg.dmsr_n_samples]

    lambda_grid = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.dmsr_grid_size)
    dmsr_vals: list[float] = []
    classifier.eval()

    for lam_val in lambda_grid:
        alpha_np, sigma_np = vp_alpha_sigma_np(np.array([lam_val]))
        alpha_v = float(alpha_np[0])
        sigma_v = float(sigma_np[0])

        noisy_a = alpha_v * imgs_a + sigma_v * torch.randn_like(imgs_a)
        noisy_b = alpha_v * imgs_b + sigma_v * torch.randn_like(imgs_b)

        feats_a = _extract_features(classifier, noisy_a, device)
        feats_b = _extract_features(classifier, noisy_b, device)

        mu_a = feats_a.mean(0)
        mu_b = feats_b.mean(0)
        dist = torch.norm(mu_a - mu_b).item()
        tr_a = ((feats_a - mu_a) ** 2).sum(1).mean().item()
        tr_b = ((feats_b - mu_b) ** 2).sum(1).mean().item()
        denom = math.sqrt((tr_a + tr_b) / 2.0 + 1e-12)
        dmsr_vals.append(dist / denom)

    dmsr_arr = np.array(dmsr_vals)
    slope = np.abs(np.gradient(dmsr_arr, lambda_grid))

    mask = slope >= cfg.rho * slope.max()
    low = float(lambda_grid[mask][0]) if mask.any() else float(lambda_grid[0])
    high = float(lambda_grid[mask][-1]) if mask.any() else float(lambda_grid[-1])
    center = float(lambda_grid[np.argmax(slope)])

    return lambda_grid, dmsr_arr, slope, low, center, high


def _extract_features(model: FeatureClassifier, imgs: Tensor, device: str, bs: int = 256) -> Tensor:
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            out.append(model.features(imgs[i:i + bs].to(device)).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def train_one_schedule(
    spec: ScheduleSpec,
    cfg: ExperimentConfig,
    device: str,
    run_seed: int,
) -> tuple[UNet, EMA, list[dict[str, float]]]:
    seed_everything(run_seed)
    ds = get_two_class_dataset(cfg, train=True)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=(device != "cpu"), drop_last=True)

    model = UNet(base_ch=cfg.base_ch, num_res_blocks=cfg.num_res_blocks).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []
    report_every = max(1, cfg.train_steps // 20)

    model.train()
    for step, (imgs, _) in enumerate(
        (x for x in _infinite(loader)), start=1
    ):
        if step > cfg.train_steps:
            break
        imgs = imgs.to(device)
        lam = sample_schedule(spec, len(imgs), cfg, device)          # (B, 1)
        alpha, sigma = vp_alpha_sigma(lam)                           # (B, 1)
        eps = torch.randn_like(imgs)
        x_lam = alpha[:, :, None, None] * imgs + sigma[:, :, None, None] * eps

        pred_eps = model(x_lam, lam)
        loss = F.mse_loss(pred_eps, eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if step == 1 or step % report_every == 0 or step == cfg.train_steps:
            history.append({"step": float(step), "train_mse": float(loss.detach().cpu())})
            print(f"  [{spec.name}] step {step}/{cfg.train_steps}  loss={loss.item():.5f}")

    return model, ema, history


# ---------------------------------------------------------------------------
# DDIM VP sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    model: UNet,
    n: int,
    cfg: ExperimentConfig,
    device: str,
) -> Tensor:
    """Deterministic DDIM sampler in VP lambda space."""
    # lambda goes from lambda_min (pure noise) to lambda_max (clean)
    lam_seq = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.ddim_steps + 1)

    model.eval()
    x = torch.randn(n, 3, cfg.image_size, cfg.image_size, device=device)

    for i in range(cfg.ddim_steps):
        lam_s = float(lam_seq[i])
        lam_t = float(lam_seq[i + 1])
        alpha_s, sigma_s = vp_alpha_sigma_np(np.array([lam_s]))
        alpha_t, sigma_t = vp_alpha_sigma_np(np.array([lam_t]))
        a_s, s_s = float(alpha_s[0]), float(sigma_s[0])
        a_t, s_t = float(alpha_t[0]), float(sigma_t[0])

        lam_in = torch.full((n, 1), lam_s, device=device)
        eps_hat = model(x, lam_in)
        x0_hat = (x - s_s * eps_hat) / (a_s + 1e-8)
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        x = a_t * x0_hat + s_t * eps_hat

    return x.clamp(-1.0, 1.0)


def generate_samples_batched(
    model: UNet, n: int, cfg: ExperimentConfig, device: str, bs: int = 256
) -> Tensor:
    out = []
    for start in range(0, n, bs):
        b = min(bs, n - start)
        out.append(ddim_sample(model, b, cfg, device).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_per_lambda_mse(
    model: UNet,
    cfg: ExperimentConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    ds = get_two_class_dataset(cfg, train=False)
    loader = DataLoader(ds, batch_size=cfg.eval_n_samples, shuffle=True,
                        num_workers=cfg.num_workers)
    imgs, _ = next(iter(loader))
    imgs = imgs[:cfg.eval_n_samples].to(device)

    lambda_grid = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.eval_grid_size)
    mse_vals: list[float] = []
    model.eval()
    for lam_val in lambda_grid:
        lam = torch.full((len(imgs), 1), lam_val, device=device)
        alpha, sigma = vp_alpha_sigma(lam)
        eps = torch.randn_like(imgs)
        x_lam = alpha[:, :, None, None] * imgs + sigma[:, :, None, None] * eps
        pred = model(x_lam, lam)
        mse_vals.append(F.mse_loss(pred, eps).item())
    return lambda_grid, np.array(mse_vals)


@torch.no_grad()
def eval_classifier_confidence(
    classifier: FeatureClassifier,
    gen_imgs: Tensor,
    device: str,
    bs: int = 256,
) -> tuple[float, float]:
    classifier.eval()
    probs_list = []
    for i in range(0, len(gen_imgs), bs):
        logits = classifier(gen_imgs[i:i + bs].to(device))
        probs_list.append(F.softmax(logits, dim=1).cpu())
    probs = torch.cat(probs_list)
    confidence = probs.max(1).values.mean().item()
    balance_err = abs(float((probs.argmax(1) == 0).float().mean()) - 0.5)
    return confidence, balance_err


def compute_coverage_metrics(
    spec: ScheduleSpec,
    lambda_grid: np.ndarray,
    slope: np.ndarray,
    t_low: float,
    t_high: float,
    cfg: ExperimentConfig,
    device: str,
) -> tuple[float, float]:
    sampled = sample_schedule(spec, 200_000, cfg, device).cpu().numpy().reshape(-1)
    m = float(np.mean((sampled >= t_low) & (sampled <= t_high)))
    s_interp = np.interp(sampled, lambda_grid, slope)
    s = float(np.mean(s_interp))
    return m, s


def compute_fid(
    model: UNet,
    cfg: ExperimentConfig,
    device: str,
    n_fake: int,
) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  [FID] torchmetrics not installed — skipping FID. pip install torchmetrics[image]")
        return float("nan")

    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    ds = get_two_class_dataset(cfg, train=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=cfg.num_workers)

    for imgs, _ in loader:
        imgs_01 = (imgs.clamp(-1, 1) + 1) / 2
        fid_metric.update(imgs_01.to(device), real=True)

    model.eval()
    bs = 256
    for start in range(0, n_fake, bs):
        b = min(bs, n_fake - start)
        with torch.no_grad():
            fake = ddim_sample(model, b, cfg, device)
        fid_metric.update(((fake + 1) / 2).to(device), real=False)

    return float(fid_metric.compute())


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_metrics_csv(rows: list[dict], path: Path) -> None:
    keys = ["schedule", "seed", "fid", "classifier_confidence", "balance_error",
            "coverage_m", "expected_s", "mean_mse", "transition_mse",
            "transition_low", "lambda_r_star", "transition_high"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def save_per_lambda_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "seed", "lambda", "mse"])
        for rec in records:
            for lam, mse in zip(rec["lambda_grid"], rec["per_lambda_mse"]):
                w.writerow([rec["schedule"], rec["seed"], lam, mse])


def save_summary_md(
    cfg: ExperimentConfig,
    specs: list[ScheduleSpec],
    rows: list[dict],
    lambda_r_star: float,
    t_low: float,
    t_high: float,
    out_path: Path,
) -> None:
    sorted_rows = sorted(rows, key=lambda r: float(r.get("fid") or 1e9))
    lines = [
        "# Phase 3 CIFAR-10 Two-Class Experiment Summary",
        "",
        f"## Config",
        f"- class pair: {cfg.class_pair}",
        f"- lambda range: [{cfg.lambda_min}, {cfg.lambda_max}]",
        f"- rho: {cfg.rho}",
        f"- lambda_R*: {lambda_r_star:.4f}",
        f"- T_R: [{t_low:.4f}, {t_high:.4f}]",
        f"- train steps: {cfg.train_steps}",
        f"- batch size: {cfg.batch_size}",
        "",
        "## Schedules",
        "",
    ]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.note}")
    lines += [
        "",
        "## Main Results (sorted by FID)",
        "",
        "| rank | schedule | seed | FID | M | S | mean MSE | transition MSE | clf conf | balance err |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(sorted_rows, 1):
        lines.append(
            f"| {rank} | {r['schedule']} | {r['seed']} | "
            f"{float(r.get('fid') or float('nan')):.2f} | "
            f"{float(r['coverage_m']):.4f} | {float(r['expected_s']):.4f} | "
            f"{float(r['mean_mse']):.5f} | {float(r['transition_mse']):.5f} | "
            f"{float(r['classifier_confidence']):.4f} | {float(r['balance_error']):.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_dmsr_profile(
    lambda_grid: np.ndarray,
    dmsr_vals: np.ndarray,
    slope: np.ndarray,
    t_low: float,
    center: float,
    t_high: float,
    out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(lambda_grid, dmsr_vals, color="#2457A6", linewidth=2.2, label="DMSR_phi(lambda)")
    ax1.axvspan(t_low, t_high, color="#F2B84B", alpha=0.25, label="T_R")
    ax1.axvline(center, color="#C23B22", linestyle="--", linewidth=1.8, label="lambda_R*")
    ax1.set_xlabel("lambda = log SNR")
    ax1.set_ylabel("DMSR separability", color="#2457A6")
    ax1.tick_params(axis="y", labelcolor="#2457A6")
    ax2 = ax1.twinx()
    ax2.plot(lambda_grid, slope, color="#268C6C", linewidth=2.0, label="|dDMSR/dlambda|")
    ax2.set_ylabel("|dDMSR/dlambda|", color="#268C6C")
    ax2.tick_params(axis="y", labelcolor="#268C6C")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right", frameon=False)
    ax1.set_title("Empirical DMSR_phi profile (CIFAR-10)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_schedule_densities(
    specs: list[ScheduleSpec],
    cfg: ExperimentConfig,
    t_low: float,
    center: float,
    t_high: float,
    device: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bins = np.linspace(cfg.lambda_min, cfg.lambda_max, 90)
    for spec in specs:
        lam = sample_schedule(spec, 80_000, cfg, device).cpu().numpy().reshape(-1)
        ax.hist(lam, bins=bins, density=True, histtype="step", linewidth=1.7, label=spec.name)
    ax.axvspan(t_low, t_high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("density")
    ax.set_title("Training noise distributions + T_R")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_lambda_mse(
    records: list[dict],
    lambda_grid: np.ndarray,
    t_low: float,
    center: float,
    t_high: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for rec in records:
        ax.plot(rec["lambda_grid"], rec["per_lambda_mse"],
                linewidth=1.9, label=str(rec["schedule"]))
    ax.axvspan(t_low, t_high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("epsilon-prediction MSE")
    ax.set_title("Per-lambda denoising MSE by schedule")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fid_vs_s(rows: list[dict], out_path: Path) -> None:
    dmsr_rows = [r for r in rows if "dmsr_normal" in str(r["schedule"])]
    if not dmsr_rows:
        return
    s_vals = [float(str(r["schedule"]).split("_s")[-1]) for r in dmsr_rows]
    fid_vals = [float(r.get("fid") or float("nan")) for r in dmsr_rows]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(s_vals, fid_vals, s=80, zorder=3)
    ax.plot(s_vals, fid_vals, linewidth=1.5, linestyle="--")
    for s, f, r in zip(s_vals, fid_vals, dmsr_rows):
        ax.annotate(r["schedule"], (s, f), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("s (schedule width)")
    ax.set_ylabel("FID")
    ax.set_title("FID vs s (DMSR-Normal schedules)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fid_vs_m(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for r in rows:
        fid = float(r.get("fid") or float("nan"))
        m = float(r["coverage_m"])
        ax.scatter(m, fid, s=70, zorder=3)
        ax.annotate(str(r["schedule"]), (m, fid),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("M (transition mass)")
    ax.set_ylabel("FID")
    ax.set_title("FID vs T_R coverage M")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_sample_grid(imgs: Tensor, out_path: Path, nrow: int = 8, title: str = "") -> None:
    try:
        from torchvision.utils import make_grid
    except ImportError:
        return
    grid = make_grid(imgs[:nrow * nrow].clamp(-1, 1), nrow=nrow, normalize=True, value_range=(-1, 1))
    npgrid = grid.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(nrow * 1.2, nrow * 1.2))
    ax.imshow(npgrid)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(cfg: ExperimentConfig, out_root: Path) -> Path:
    seed_everything(cfg.seed)
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = "cpu"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / "phase3" / f"{run_id}_{cfg.run_name}_{cfg.class_pair}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ensure_mpl_config_dir(out_dir)

    print(f"\n=== Phase 3: {cfg.class_pair} | device={device} ===")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    # --- Step 1: Train classifier ---
    print("\n[1/5] Training feature classifier...")
    classifier = train_classifier(cfg, device)
    torch.save(classifier.state_dict(), out_dir / "classifier.pt")

    # --- Step 2: Compute DMSR_phi ---
    print("\n[2/5] Computing empirical DMSR_phi(lambda)...")
    lambda_grid, dmsr_vals, slope, t_low, lambda_r_star, t_high = compute_dmsr_phi(
        classifier, cfg, device
    )
    print(f"  lambda_R* = {lambda_r_star:.4f}  T_R = [{t_low:.4f}, {t_high:.4f}]")
    np.save(out_dir / "dmsr_grid.npy", lambda_grid)
    np.save(out_dir / "dmsr_vals.npy", dmsr_vals)
    np.save(out_dir / "dmsr_slope.npy", slope)
    plot_dmsr_profile(lambda_grid, dmsr_vals, slope, t_low, lambda_r_star, t_high,
                      plots_dir / "dmsr_profile.png")

    # --- Step 3: Build schedules ---
    specs = build_schedules(lambda_r_star, cfg)
    (out_dir / "schedules.json").write_text(
        json.dumps([asdict(s) for s in specs], indent=2), encoding="utf-8"
    )
    plot_schedule_densities(specs, cfg, t_low, lambda_r_star, t_high, device,
                             plots_dir / "schedule_densities.png")

    # --- Step 4: Train + evaluate each schedule ---
    print("\n[3/5] Training diffusion models...")
    all_per_lambda: list[dict] = []
    summary_rows: list[dict] = []
    histories: dict[str, list[dict]] = {}

    for seed_idx in range(cfg.num_seeds):
        run_seed = cfg.seed + seed_idx
        for spec in specs:
            print(f"\n  Schedule: {spec.name}  seed={run_seed}")
            model, ema, history = train_one_schedule(spec, cfg, device, run_seed)
            histories[f"{spec.name}_seed{run_seed}"] = history

            # Switch to EMA weights for eval
            ema.copy_to(model)
            model.eval()

            print(f"  Evaluating per-lambda MSE...")
            lam_g, per_mse = eval_per_lambda_mse(model, cfg, device)
            transition_mask = (lam_g >= t_low) & (lam_g <= t_high)
            mean_mse = float(np.mean(per_mse))
            transition_mse = float(np.mean(per_mse[transition_mask])) if transition_mask.any() else float("nan")

            print(f"  Generating {cfg.n_gen_samples} samples...")
            gen_imgs = generate_samples_batched(model, cfg.n_gen_samples, cfg, device)
            sample_path = plots_dir / f"samples_{spec.name}_seed{run_seed}.png"
            save_sample_grid(gen_imgs, sample_path, title=spec.name)

            print(f"  Classifier confidence...")
            clf_conf, bal_err = eval_classifier_confidence(classifier, gen_imgs, device)

            fid_val = float("nan")
            if cfg.compute_fid:
                print(f"  FID...")
                fid_val = compute_fid(model, cfg, device, cfg.n_gen_samples)

            print(f"  Coverage metrics...")
            m, s = compute_coverage_metrics(
                spec, lambda_grid, slope, t_low, t_high, cfg, device
            )

            rec = {
                "schedule": spec.name,
                "seed": run_seed,
                "fid": fid_val,
                "classifier_confidence": clf_conf,
                "balance_error": bal_err,
                "coverage_m": m,
                "expected_s": s,
                "mean_mse": mean_mse,
                "transition_mse": transition_mse,
                "transition_low": t_low,
                "lambda_r_star": lambda_r_star,
                "transition_high": t_high,
                "lambda_grid": lam_g.tolist(),
                "per_lambda_mse": per_mse.tolist(),
            }
            all_per_lambda.append(rec)
            summary_rows.append({k: v for k, v in rec.items() if not isinstance(v, list)})

            print(f"  [{spec.name}] FID={fid_val:.2f}  M={m:.4f}  S={s:.4f}  "
                  f"mean_mse={mean_mse:.5f}")

            torch.save(model.state_dict(), out_dir / f"model_{spec.name}_seed{run_seed}.pt")

    # --- Step 5: Save all results ---
    print("\n[5/5] Saving results...")
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_metrics_csv(summary_rows, out_dir / "metrics_summary.csv")
    save_per_lambda_csv(all_per_lambda, out_dir / "per_lambda_metrics.csv")
    save_summary_md(cfg, specs, summary_rows, lambda_r_star, t_low, t_high,
                    out_dir / "summary.md")
    plot_per_lambda_mse(all_per_lambda, lambda_grid, t_low, lambda_r_star, t_high,
                        plots_dir / "per_lambda_mse.png")
    plot_fid_vs_s(summary_rows, plots_dir / "fid_vs_s.png")
    plot_fid_vs_m(summary_rows, plots_dir / "fid_vs_m.png")

    print(f"\nDone. Results saved to: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Preset + CLI
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "smoke": dict(clf_epochs=2, train_steps=50, n_gen_samples=64,
                  dmsr_n_samples=100, dmsr_grid_size=10, eval_n_samples=64,
                  compute_fid=False, num_seeds=1),
    "fast":  dict(clf_epochs=10, train_steps=50_000, n_gen_samples=5_000,
                  dmsr_n_samples=500, dmsr_grid_size=30, eval_n_samples=256,
                  compute_fid=True, num_seeds=1),
    "full":  dict(clf_epochs=20, train_steps=100_000, n_gen_samples=10_000,
                  dmsr_n_samples=1000, dmsr_grid_size=50, eval_n_samples=512,
                  compute_fid=True, num_seeds=1),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 CIFAR-10 DMSR experiment")
    p.add_argument("--preset", choices=list(PRESETS), default="full")
    p.add_argument("--class-pair", choices=list(CLASS_PAIRS), default="airplane_vs_automobile")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--num-seeds", type=int, default=None)
    p.add_argument("--s-values", type=float, nargs="+", default=None)
    p.add_argument("--no-fid", action="store_true")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]

    cfg = ExperimentConfig(
        class_pair=args.class_pair,
        clf_epochs=preset["clf_epochs"],
        train_steps=args.train_steps if args.train_steps is not None else preset["train_steps"],
        batch_size=args.batch_size if args.batch_size is not None else 128,
        n_gen_samples=preset["n_gen_samples"],
        dmsr_n_samples=preset["dmsr_n_samples"],
        dmsr_grid_size=preset["dmsr_grid_size"],
        eval_n_samples=preset["eval_n_samples"],
        compute_fid=(not args.no_fid) and preset["compute_fid"],
        seed=args.seed,
        num_seeds=args.num_seeds if args.num_seeds is not None else preset["num_seeds"],
        s_values=tuple(args.s_values) if args.s_values else (1.5, 0.8, 0.3),
        device=args.device,
        num_workers=args.num_workers,
    )
    run(cfg, args.out_root)


if __name__ == "__main__":
    main()
