#!/usr/bin/env python3
"""Phase 2: MNIST Pilot — DMSR-based Noise Scheduling Pipeline Validation.

This file is intentionally self-contained. It implements the full Phase 2
pipeline from the experiment plan:

1. Load MNIST digit 0 vs 1, normalize to [-1, 1].
2. Train a small CNN classifier φ; use penultimate features to compute
   empirical DMSR_φ(λ) on a VP λ grid.
3. Estimate λ_R* and transition region T_R from the empirical slope.
4. Train the same Mini U-Net denoiser under each p_train(λ) schedule
   (only the schedule changes; loss w(λ)=1, model/optimizer/steps identical).
5. Generate samples via DDIM 50 steps (VP cosine λ schedule, fixed).
6. Evaluate FID (using φ features), classifier confidence, balance error,
   per-λ MSE, and DMSR-transition coverage metrics M and S.
7. Save CSV, Markdown, and PNG artifacts to results/phase2/<run_id>/.

Run (full):
    python3 phase2/phase2_mnist_experiment.py

Smoke test:
    python3 phase2/phase2_mnist_experiment.py \\
        --train-steps 500 --n-generate 50 --ddim-steps 5 \\
        --dmsr-grid-size 12 --eval-grid-size 12

Other digits:
    python3 phase2/phase2_mnist_experiment.py --digits 3 8
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

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "results" / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

Tensor = torch.Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentConfig:
    # Data
    digits: tuple = (0, 1)
    img_size: int = 28

    # VP noise range
    lambda_min: float = -10.0
    lambda_max: float = 10.0
    rho: float = 0.5

    # Feature classifier φ
    clf_epochs: int = 10
    clf_lr: float = 1e-3
    clf_batch_size: int = 256
    clf_feature_dim: int = 64

    # Empirical DMSR computation
    dmsr_grid_size: int = 40
    dmsr_n_samples: int = 512

    # Mini U-Net
    base_ch: int = 32
    time_emb_dim: int = 128

    # Denoiser training
    train_steps: int = 20000
    batch_size: int = 128
    lr: float = 2e-4
    eval_batch_size: int = 256
    eval_grid_size: int = 40
    seed: int = 20260526
    num_seeds: int = 1
    device: str = "cpu"

    # DDIM generation
    ddim_steps: int = 50
    n_generate: int = 5000
    gen_batch_size: int = 100
    ddim_lambda_min: float = -8.0
    ddim_lambda_max: float = 6.0

    # Output
    run_name: str = "phase2_mnist"
    data_root: str = "./data"


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    kind: str
    center_lambda: float | None = None
    scale: float | None = None
    note: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

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


def ensure_mpl_config_dir(out_dir: Path) -> None:
    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def vp_alpha_sigma(lam: Tensor) -> tuple[Tensor, Tensor]:
    return torch.sqrt(torch.sigmoid(lam)), torch.sqrt(torch.sigmoid(-lam))


def vp_alpha_sigma_np(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha_sq = 1.0 / (1.0 + np.exp(-lam))
    sigma_sq = 1.0 / (1.0 + np.exp(lam))
    return np.sqrt(alpha_sq), np.sqrt(sigma_sq)


# ──────────────────────────────────────────────────────────────────────────────
# MNIST Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_mnist_two_class(config: ExperimentConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return train_imgs, train_labels, test_imgs, test_labels in [-1, 1]."""
    try:
        import torchvision.datasets as dsets
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("torchvision is required. Install with: pip install torchvision")

    d0, d1 = config.digits

    def extract(dataset) -> tuple[Tensor, Tensor]:
        imgs = dataset.data.float().unsqueeze(1) / 255.0 * 2.0 - 1.0  # [-1,1]
        labels = dataset.targets
        mask = (labels == d0) | (labels == d1)
        imgs = imgs[mask]
        orig = labels[mask]
        remapped = torch.where(orig == d0, torch.zeros_like(orig), torch.ones_like(orig))
        return imgs, remapped

    transform = T.ToTensor()
    train_raw = dsets.MNIST(config.data_root, train=True,  download=True, transform=transform)
    test_raw  = dsets.MNIST(config.data_root, train=False, download=True, transform=transform)
    tr_imgs, tr_labels = extract(train_raw)
    te_imgs, te_labels = extract(test_raw)
    print(f"[Phase 2] MNIST {d0} vs {d1}: train={len(tr_imgs)}, test={len(te_imgs)}")
    return tr_imgs, tr_labels, te_imgs, te_labels


# ──────────────────────────────────────────────────────────────────────────────
# Feature Classifier φ
# ──────────────────────────────────────────────────────────────────────────────

class FeatureClassifier(nn.Module):
    """Two-class CNN classifier. Penultimate layer used as φ(x)."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                               # 14×14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                               # 7×7
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
    train_imgs: Tensor, train_labels: Tensor,
    test_imgs:  Tensor, test_labels:  Tensor,
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


# ──────────────────────────────────────────────────────────────────────────────
# Empirical DMSR_φ(λ)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_empirical_dmsr(
    clf: FeatureClassifier,
    imgs_a: Tensor,
    imgs_b: Tensor,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DMSR_φ(λ) on a VP λ grid and return (lambda_grid, dmsr_values)."""
    clf.eval()
    lambda_grid = np.linspace(config.lambda_min, config.lambda_max, config.dmsr_grid_size)

    n_a = min(config.dmsr_n_samples, len(imgs_a))
    n_b = min(config.dmsr_n_samples, len(imgs_b))
    imgs_a = imgs_a[:n_a].to(config.device)
    imgs_b = imgs_b[:n_b].to(config.device)

    dmsr_values: list[float] = []
    for lam_val in lambda_grid:
        lam_t = torch.tensor(lam_val, dtype=torch.float32, device=config.device)
        alpha, sigma = vp_alpha_sigma(lam_t)

        x_a = alpha * imgs_a + sigma * torch.randn_like(imgs_a)
        x_b = alpha * imgs_b + sigma * torch.randn_like(imgs_b)

        phi_a = clf.get_features(x_a)  # (n_a, feat_dim)
        phi_b = clf.get_features(x_b)  # (n_b, feat_dim)

        mu_a, mu_b = phi_a.mean(0), phi_b.mean(0)
        mean_dist = (mu_a - mu_b).norm().item()

        tr_a = phi_a.var(0, unbiased=True).sum().item()
        tr_b = phi_b.var(0, unbiased=True).sum().item()
        within_std = math.sqrt((tr_a + tr_b) / 2.0 + 1e-10)

        dmsr_values.append(mean_dist / within_std)

    return lambda_grid, np.array(dmsr_values)


def compute_empirical_transition(
    lambda_grid: np.ndarray,
    dmsr_values: np.ndarray,
    rho: float,
) -> tuple[float, float, float]:
    """Return (transition_low, lambda_r_star, transition_high)."""
    slope_abs = np.abs(np.gradient(dmsr_values, lambda_grid))
    if slope_abs.max() < 1e-10:
        mid = float(lambda_grid[len(lambda_grid) // 2])
        return mid, mid, mid
    mask = slope_abs >= rho * slope_abs.max()
    r_star_idx = int(np.argmax(slope_abs))
    in_region = lambda_grid[mask]
    return float(in_region[0]), float(lambda_grid[r_star_idx]), float(in_region[-1])


# ──────────────────────────────────────────────────────────────────────────────
# Mini U-Net Denoiser (2 resolution levels, base_ch=32)
# ──────────────────────────────────────────────────────────────────────────────

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
        # Encoder
        self.in_conv = nn.Conv2d(1, ch, 3, padding=1)
        self.enc0    = ResBlock(ch,      ch,      time_dim)           # (B, ch,  32,32) → skip
        self.down0   = nn.Conv2d(ch, ch, 3, stride=2, padding=1)     # (B, ch,  16,16)
        self.enc1    = ResBlock(ch,      ch * 2,  time_dim)           # (B, 2ch, 16,16)
        # Bottleneck
        self.mid     = ResBlock(ch * 2,  ch * 2,  time_dim)           # (B, 2ch, 16,16)
        # Decoder
        self.up1     = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)   # (B, ch,  32,32)
        self.dec1    = ResBlock(ch * 2,  ch,      time_dim)           # concat skip → ch
        # Output
        self.out_norm = _groupnorm(ch)
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x: Tensor, lam: Tensor) -> Tensor:
        lam_flat = lam.view(-1)
        t = self.time_emb(sinusoidal_embedding(lam_flat, self.time_dim))

        x = F.pad(x, (2, 2, 2, 2))                        # (B,1,28,28) → (B,1,32,32)
        h  = self.in_conv(x)                               # (B, ch, 32,32)
        h0 = self.enc0(h, t)                               # (B, ch, 32,32) — skip
        h  = self.enc1(self.down0(h0), t)                  # (B, 2ch, 16,16)
        h  = self.mid(h, t)                                # (B, 2ch, 16,16)
        h  = self.dec1(torch.cat([self.up1(h), h0], 1), t) # (B, ch,  32,32)
        h  = self.out_conv(F.silu(self.out_norm(h)))       # (B, 1,   32,32)
        return h[:, :, 2:-2, 2:-2]                         # (B,1,28,28)


# ──────────────────────────────────────────────────────────────────────────────
# Schedule Sampler
# ──────────────────────────────────────────────────────────────────────────────

def clamp_lambda(lam: Tensor, config: ExperimentConfig) -> Tensor:
    return lam.clamp(config.lambda_min, config.lambda_max)


def sample_schedule(spec: ScheduleSpec, n: int, config: ExperimentConfig, device: str) -> Tensor:
    eps = 1e-5
    if spec.kind == "dmsr_normal":
        assert spec.center_lambda is not None and spec.scale is not None
        lam = spec.center_lambda + spec.scale * torch.randn(n, 1, device=device)
        return clamp_lambda(lam, config)

    if spec.kind == "dmsr_laplace":
        assert spec.center_lambda is not None and spec.scale is not None
        dist = torch.distributions.Laplace(spec.center_lambda, spec.scale)
        return clamp_lambda(dist.sample((n, 1)).to(device), config)

    if spec.kind == "hang_laplace_lambda":
        b = spec.scale if spec.scale is not None else 0.5
        lam = torch.distributions.Laplace(0.0, b).sample((n, 1)).to(device)
        return clamp_lambda(lam, config)

    if spec.kind == "cosine_vp":
        t = torch.rand(n, 1, device=device).clamp(eps, 1.0 - eps)
        lam = -2.0 * torch.log(torch.tan(0.5 * math.pi * t))
        return clamp_lambda(lam, config)

    raise ValueError(f"Unknown schedule kind: {spec.kind}")


def build_schedules(config: ExperimentConfig, lambda_r_star: float) -> list[ScheduleSpec]:
    return [
        ScheduleSpec("cosine_vp",              "cosine_vp",        note="VP cosine schedule induced density."),
        ScheduleSpec("hang_laplace_b0.5",       "hang_laplace_lambda", scale=0.5,
                     note="Hang-style Laplace centered at λ=0."),
        ScheduleSpec("dmsr_normal_wide_s1.5",  "dmsr_normal",      center_lambda=lambda_r_star, scale=1.5,
                     note=f"DMSR-centered Normal, s=1.5, center={lambda_r_star:.3f}."),
        ScheduleSpec("dmsr_normal_mid_s0.8",   "dmsr_normal",      center_lambda=lambda_r_star, scale=0.8,
                     note=f"DMSR-centered Normal, s=0.8, center={lambda_r_star:.3f}."),
        ScheduleSpec("dmsr_normal_narrow_s0.3","dmsr_normal",      center_lambda=lambda_r_star, scale=0.3,
                     note=f"DMSR-centered Normal, s=0.3, center={lambda_r_star:.3f}."),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# DDIM Sampler (VP, deterministic)
# ──────────────────────────────────────────────────────────────────────────────

def make_ddim_lambda_schedule(
    n_steps: int,
    lambda_min: float,
    lambda_max: float,
) -> np.ndarray:
    """VP cosine λ schedule for DDIM: noisy (small λ) → clean (large λ)."""
    t = np.linspace(1.0, 0.0, n_steps + 1)
    t = np.clip(t, 1e-5, 1.0 - 1e-5)
    lam = -2.0 * np.log(np.tan(0.5 * np.pi * t))
    return np.clip(lam, lambda_min, lambda_max)  # (n_steps+1,)


@torch.no_grad()
def ddim_generate(
    model: MiniUNet,
    n_samples: int,
    config: ExperimentConfig,
) -> Tensor:
    """Generate n_samples images with DDIM (deterministic, VP cosine schedule)."""
    model.eval()
    lam_sched = make_ddim_lambda_schedule(
        config.ddim_steps, config.ddim_lambda_min, config.ddim_lambda_max
    )
    sigma_init = float(np.sqrt(1.0 / (1.0 + np.exp(lam_sched[0]))))

    all_imgs: list[Tensor] = []
    remaining = n_samples
    while remaining > 0:
        bs = min(config.gen_batch_size, remaining)
        remaining -= bs

        x = sigma_init * torch.randn(bs, 1, config.img_size, config.img_size, device=config.device)

        for i in range(len(lam_sched) - 1):
            lc, ln = float(lam_sched[i]), float(lam_sched[i + 1])

            lam_t = torch.full((bs,), lc, device=config.device)
            alpha_c = torch.sqrt(torch.sigmoid(lam_t)).view(bs, 1, 1, 1)
            sigma_c = torch.sqrt(torch.sigmoid(-lam_t)).view(bs, 1, 1, 1)

            a_n = float(np.sqrt(1.0 / (1.0 + np.exp(-ln))))
            s_n = float(np.sqrt(1.0 / (1.0 + np.exp(ln))))

            pred_eps = model(x, lam_t)
            pred_x0  = ((x - sigma_c * pred_eps) / alpha_c.clamp_min(1e-8)).clamp(-1, 1)
            x = a_n * pred_x0 + s_n * pred_eps

        all_imgs.append(x.clamp(-1, 1).cpu())

    return torch.cat(all_imgs, 0)[:n_samples]


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_one_schedule(
    spec: ScheduleSpec,
    train_imgs: Tensor,
    config: ExperimentConfig,
    run_seed: int,
) -> tuple[MiniUNet, list[dict[str, float]]]:
    seed_everything(run_seed)
    model = MiniUNet(config.base_ch, config.time_emb_dim).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []
    report_every = max(1, config.train_steps // 10)
    n_train = len(train_imgs)

    model.train()
    for step in range(1, config.train_steps + 1):
        idx = torch.randint(0, n_train, (config.batch_size,))
        x0  = train_imgs[idx].to(config.device)
        lam = sample_schedule(spec, config.batch_size, config, config.device)  # (B,1)
        alpha, sigma = vp_alpha_sigma(lam.view(-1, 1, 1, 1))
        eps  = torch.randn_like(x0)
        x_lam = alpha * x0 + sigma * eps
        pred_eps = model(x_lam, lam.view(-1))
        loss = F.mse_loss(pred_eps, eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % report_every == 0 or step == config.train_steps:
            history.append({"step": float(step), "train_mse": float(loss.detach().cpu())})

    model.eval()
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_lambda(
    model: MiniUNet,
    train_imgs: Tensor,
    config: ExperimentConfig,
    transition_low: float,
    transition_high: float,
) -> dict:
    model.eval()
    lambda_grid = np.linspace(config.lambda_min, config.lambda_max, config.eval_grid_size)
    per_lambda_mse: list[float] = []
    n = len(train_imgs)

    for lam_val in lambda_grid:
        idx = torch.randint(0, n, (config.eval_batch_size,))
        x0  = train_imgs[idx].to(config.device)
        lam_t = torch.full((config.eval_batch_size,), lam_val, device=config.device)
        alpha = torch.sqrt(torch.sigmoid(lam_t)).view(-1, 1, 1, 1)
        sigma = torch.sqrt(torch.sigmoid(-lam_t)).view(-1, 1, 1, 1)
        eps   = torch.randn_like(x0)
        pred_eps = model(alpha * x0 + sigma * eps, lam_t)
        per_lambda_mse.append(F.mse_loss(pred_eps, eps).item())

    arr = np.array(per_lambda_mse)
    lam_arr = lambda_grid
    t_mask = (lam_arr >= transition_low) & (lam_arr <= transition_high)

    def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if mask.any() else float("nan")

    return {
        "mean_mse":        float(arr.mean()),
        "transition_mse":  _safe_mean(arr, t_mask),
        "low_noise_mse":   _safe_mean(arr, lam_arr > transition_high),
        "high_noise_mse":  _safe_mean(arr, lam_arr < transition_low),
        "lambda_grid":     lambda_grid.tolist(),
        "per_lambda_mse":  per_lambda_mse,
    }


def compute_coverage_metrics(
    spec: ScheduleSpec,
    config: ExperimentConfig,
    transition_low: float,
    transition_high: float,
    dmsr_lambda_grid: np.ndarray,
    dmsr_slope_abs: np.ndarray,
) -> dict[str, float]:
    sampled = (
        sample_schedule(spec, 200_000, config, config.device)
        .detach().cpu().numpy().reshape(-1)
    )
    coverage_m = float(np.mean((sampled >= transition_low) & (sampled <= transition_high)))
    slopes_at = np.interp(sampled, dmsr_lambda_grid, dmsr_slope_abs)
    expected_s = float(np.mean(slopes_at))
    max_slope = float(dmsr_slope_abs.max()) if dmsr_slope_abs.max() > 0 else 1.0
    return {
        "coverage_m":      coverage_m,
        "expected_s":      expected_s,
        "expected_s_norm": expected_s / max_slope,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FID and Classifier Metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_features(clf: FeatureClassifier, imgs: Tensor, batch_size: int, device: str) -> np.ndarray:
    clf.eval()
    feats: list[np.ndarray] = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size].to(device)
        feats.append(clf.get_features(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid_phi(
    clf: FeatureClassifier,
    real_imgs: Tensor,
    gen_imgs: Tensor,
    batch_size: int,
    device: str,
) -> float:
    """FID computed in φ-feature space (no InceptionV3 download needed)."""
    from scipy.linalg import sqrtm

    real_f = _extract_features(clf, real_imgs, batch_size, device)
    gen_f  = _extract_features(clf, gen_imgs,  batch_size, device)

    mu1, mu2 = real_f.mean(0), gen_f.mean(0)
    s1 = np.cov(real_f, rowvar=False) + np.eye(real_f.shape[1]) * 1e-6
    s2 = np.cov(gen_f,  rowvar=False) + np.eye(gen_f.shape[1])  * 1e-6

    diff = mu1 - mu2
    prod, _ = sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(prod):
        prod = prod.real

    fid = float(diff @ diff + np.trace(s1 + s2 - 2.0 * prod))
    return max(fid, 0.0)


@torch.no_grad()
def compute_classifier_metrics(
    clf: FeatureClassifier,
    gen_imgs: Tensor,
    config: ExperimentConfig,
) -> dict[str, float]:
    clf.eval()
    probs_list: list[Tensor] = []
    for i in range(0, len(gen_imgs), config.eval_batch_size):
        batch = gen_imgs[i : i + config.eval_batch_size].to(config.device)
        probs_list.append(F.softmax(clf(batch), dim=1).cpu())
    probs = torch.cat(probs_list, 0)
    preds = probs.argmax(1)
    conf  = probs.max(1).values.mean().item()
    frac0 = (preds == 0).float().mean().item()
    return {
        "classifier_confidence": conf,
        "balance_error":         abs(frac0 - 0.5),
        "frac_class_0":          frac0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CSV / Markdown Saving
# ──────────────────────────────────────────────────────────────────────────────

_SUMMARY_KEYS = [
    "schedule", "seed",
    "coverage_m", "expected_s_norm",
    "fid_phi",
    "classifier_confidence", "balance_error",
    "mean_mse", "transition_mse", "low_noise_mse", "high_noise_mse",
]


def save_metrics_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_KEYS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _SUMMARY_KEYS})


def save_per_lambda_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "seed", "lambda", "mse"])
        for r in results:
            sched, seed = str(r["schedule"]), int(r["seed"])
            for lam, mse in zip(r["lambda_grid"], r["per_lambda_mse"]):
                writer.writerow([sched, seed, lam, mse])


def save_summary_md(
    config: ExperimentConfig,
    specs: list[ScheduleSpec],
    summary_rows: list[dict],
    dmsr_info: dict,
    out_path: Path,
) -> None:
    d0, d1 = config.digits
    low   = dmsr_info["transition_low"]
    high  = dmsr_info["transition_high"]
    r_star = dmsr_info["lambda_r_star"]

    sorted_rows = sorted(summary_rows, key=lambda r: float(r.get("fid_phi", float("inf"))))

    lines = [
        "# Phase 2 MNIST Pilot Summary",
        "",
        "## Purpose",
        "",
        "Validate the full VP diffusion pipeline, empirical DMSR_φ(λ) computation,",
        "DDIM sampler, FID measurement, and schedule comparison on MNIST.",
        "",
        "## Configuration",
        "",
        f"- digits: {d0} vs {d1}",
        f"- image size: 1 × {config.img_size} × {config.img_size}",
        f"- λ range: [{config.lambda_min}, {config.lambda_max}]",
        f"- ρ threshold: {config.rho}",
        f"- empirical λ_R*: {r_star:.4f}",
        f"- transition region: [{low:.4f}, {high:.4f}]",
        f"- train steps per schedule: {config.train_steps}",
        f"- batch size: {config.batch_size}",
        f"- DDIM steps: {config.ddim_steps}",
        f"- n_generate (FID): {config.n_generate}",
        "",
        "## Schedules",
        "",
    ]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.note}")
    lines += [
        "",
        "## Main Results",
        "",
        "| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(sorted_rows, 1):
        lines.append(
            f"| {rank} | {row['schedule']} | {int(row['seed'])} "
            f"| {float(row['coverage_m']):.4f} | {float(row['expected_s_norm']):.4f} "
            f"| {float(row.get('fid_phi', float('nan'))):.2f} "
            f"| {float(row.get('classifier_confidence', float('nan'))):.4f} "
            f"| {float(row.get('balance_error', float('nan'))):.4f} "
            f"| {float(row['mean_mse']):.4f} | {float(row['transition_mse']):.4f} |"
        )
    lines += [
        "",
        "## Interpretation Guide",
        "",
        "- FID is computed in φ-feature space (not InceptionV3). Lower is better.",
        "- `coverage_m` alone is not sufficient; the balance with full-range support matters.",
        "- Schedule differences may be small on MNIST (easy dataset). This stage validates the pipeline.",
        "- λ_R* is estimated empirically from the numerical slope of DMSR_φ(λ).",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _span_kwargs(low: float, high: float) -> dict:
    return dict(color="#F2B84B", alpha=0.22)


def plot_dmsr_profile(
    lambda_grid: np.ndarray,
    dmsr_values: np.ndarray,
    dmsr_slope: np.ndarray,
    transition_low: float,
    transition_high: float,
    lambda_r_star: float,
    config: ExperimentConfig,
    out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(lambda_grid, dmsr_values, label="DMSR_φ(λ)", color="#2457A6", linewidth=2.2)
    ax1.set_xlabel("λ = log SNR")
    ax1.set_ylabel("DMSR_φ separability", color="#2457A6")
    ax1.tick_params(axis="y", labelcolor="#2457A6")
    ax1.axvspan(transition_low, transition_high, **_span_kwargs(transition_low, transition_high),
                label="DMSR-transition region")
    ax1.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.8, label="λ_R*")
    ax2 = ax1.twinx()
    ax2.plot(lambda_grid, dmsr_slope, label="|dDMSR_φ/dλ|", color="#268C6C", linewidth=2.0)
    ax2.set_ylabel("|dDMSR_φ/dλ|", color="#268C6C")
    ax2.tick_params(axis="y", labelcolor="#268C6C")
    lines, labels = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lb2, loc="upper right", frameon=False)
    d0, d1 = config.digits
    ax1.set_title(f"Empirical VP DMSR_φ(λ): MNIST {d0} vs {d1}, ρ={config.rho}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_schedule_densities(
    specs: list[ScheduleSpec],
    config: ExperimentConfig,
    transition_low: float,
    transition_high: float,
    lambda_r_star: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bins = np.linspace(config.lambda_min, config.lambda_max, 90)
    for spec in specs:
        lam = sample_schedule(spec, 80_000, config, config.device).detach().cpu().numpy().reshape(-1)
        ax.hist(lam, bins=bins, density=True, histtype="step", linewidth=1.7, label=spec.name)
    ax.axvspan(transition_low, transition_high, **_span_kwargs(transition_low, transition_high))
    ax.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("λ = log SNR")
    ax.set_ylabel("density")
    ax.set_title("Training noise distributions (shading = DMSR transition region)")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_lambda_mse(
    results: list[dict],
    config: ExperimentConfig,
    transition_low: float,
    transition_high: float,
    lambda_r_star: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for r in results:
        ax.plot(r["lambda_grid"], r["per_lambda_mse"], linewidth=1.9, label=str(r["schedule"]))
    ax.axvspan(transition_low, transition_high, **_span_kwargs(transition_low, transition_high))
    ax.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("λ = log SNR")
    ax.set_ylabel("Epsilon-prediction MSE")
    ax.set_title("Per-λ denoising MSE by schedule")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fid_summary(summary_rows: list[dict], out_path: Path) -> None:
    names = [r["schedule"] for r in summary_rows]
    fids  = [float(r.get("fid_phi", float("nan"))) for r in summary_rows]
    order = np.argsort(fids)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.barh(
        [names[i] for i in order],
        [fids[i] for i in order],
        color="#2457A6", alpha=0.8,
    )
    ax.set_xlabel("FID (φ-feature space)")
    ax.set_title("FID comparison by training schedule")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_coverage_tradeoff(summary_rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    fids = [float(r.get("fid_phi", 0.0)) for r in summary_rows]
    sc = ax.scatter(
        [float(r["coverage_m"]) for r in summary_rows],
        [float(r.get("fid_phi", float("nan"))) for r in summary_rows],
        c=fids, cmap="viridis", s=70,
    )
    for r in summary_rows:
        ax.annotate(
            str(r["schedule"]),
            (float(r["coverage_m"]), float(r.get("fid_phi", 0))),
            xytext=(4, 3), textcoords="offset points", fontsize=7,
        )
    fig.colorbar(sc, ax=ax, label="FID (φ)")
    ax.set_xlabel("DMSR-transition mass M")
    ax.set_ylabel("FID (φ-feature space)")
    ax.set_title("Transition coverage vs FID")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_noisy_images(
    imgs_a: Tensor,
    imgs_b: Tensor,
    config: ExperimentConfig,
    lambda_r_star: float,
    out_path: Path,
) -> None:
    """Show noisy images at a few λ levels around λ_R*."""
    lam_vals = [lambda_r_star - 2.0, lambda_r_star - 1.0, lambda_r_star, lambda_r_star + 1.0, lambda_r_star + 2.0]
    n_show = 3
    fig, axes = plt.subplots(2, len(lam_vals), figsize=(len(lam_vals) * 2.2, 5.0))
    for col, lv in enumerate(lam_vals):
        lv = float(np.clip(lv, config.lambda_min, config.lambda_max))
        lam_t = torch.tensor(lv)
        alpha, sigma = vp_alpha_sigma(lam_t)
        for row, imgs in enumerate([imgs_a, imgs_b]):
            x0 = imgs[:n_show]
            x_lam = alpha * x0 + sigma * torch.randn_like(x0)
            grid = x_lam[0, 0].clamp(-1, 1).numpy() * 0.5 + 0.5
            axes[row, col].imshow(grid, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"λ={lv:.1f}", fontsize=8)
    axes[0, 0].set_ylabel(f"digit {config.digits[0]}", fontsize=8)
    axes[1, 0].set_ylabel(f"digit {config.digits[1]}", fontsize=8)
    fig.suptitle(f"Noisy images around λ_R*={lambda_r_star:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_grid(schedule_name: str, gen_imgs: Tensor, out_path: Path, n_rows: int = 4, n_cols: int = 8) -> None:
    n = min(n_rows * n_cols, len(gen_imgs))
    imgs = gen_imgs[:n].clamp(-1, 1) * 0.5 + 0.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.1))
    for idx, ax in enumerate(axes.reshape(-1)):
        if idx < n:
            ax.imshow(imgs[idx, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"Generated samples: {schedule_name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main Run
# ──────────────────────────────────────────────────────────────────────────────

def run(config: ExperimentConfig, out_root: Path) -> Path:
    seed_everything(config.seed)
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    d0, d1  = config.digits
    out_dir = out_root / "phase2" / f"{run_id}_{config.run_name}_d{d0}v{d1}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ensure_mpl_config_dir(out_dir)

    # ── 1. Load data ────────────────────────────────────────────────────────
    tr_imgs, tr_labels, te_imgs, te_labels = load_mnist_two_class(config)
    imgs_a = tr_imgs[tr_labels == 0]
    imgs_b = tr_imgs[tr_labels == 1]

    # ── 2. Train classifier φ ───────────────────────────────────────────────
    print("[Phase 2] Training feature classifier φ...")
    clf = train_feature_classifier(tr_imgs, tr_labels, te_imgs, te_labels, config)
    torch.save(clf.state_dict(), out_dir / "classifier_phi.pt")

    # ── 3. Compute empirical DMSR_φ(λ) ─────────────────────────────────────
    print("[Phase 2] Computing empirical DMSR_φ(λ)...")
    dmsr_grid, dmsr_vals = compute_empirical_dmsr(clf, imgs_a, imgs_b, config)
    dmsr_slope_abs = np.abs(np.gradient(dmsr_vals, dmsr_grid))
    trans_low, lambda_r_star, trans_high = compute_empirical_transition(
        dmsr_grid, dmsr_vals, config.rho
    )
    print(f"[Phase 2] λ_R* = {lambda_r_star:.4f}  T_R = [{trans_low:.4f}, {trans_high:.4f}]")

    dmsr_info = {
        "transition_low": trans_low,
        "lambda_r_star":  lambda_r_star,
        "transition_high": trans_high,
    }
    (out_dir / "dmsr_info.json").write_text(json.dumps({
        "lambda_grid": dmsr_grid.tolist(),
        "dmsr_values": dmsr_vals.tolist(),
        "dmsr_slope_abs": dmsr_slope_abs.tolist(),
        **dmsr_info,
    }, indent=2), encoding="utf-8")

    # ── 4. Plots: DMSR profile & noisy images ───────────────────────────────
    plot_dmsr_profile(
        dmsr_grid, dmsr_vals, dmsr_slope_abs,
        trans_low, trans_high, lambda_r_star, config,
        plots_dir / "dmsr_profile.png",
    )
    plot_noisy_images(imgs_a, imgs_b, config, lambda_r_star, plots_dir / "noisy_images_at_transition.png")

    # ── 5. Build schedules ──────────────────────────────────────────────────
    specs = build_schedules(config, lambda_r_star)
    (out_dir / "schedules.json").write_text(
        json.dumps([asdict(s) for s in specs], indent=2), encoding="utf-8"
    )
    (out_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    plot_schedule_densities(specs, config, trans_low, trans_high, lambda_r_star, plots_dir / "schedule_densities.png")

    # ── 6. Pre-compute real features for FID ────────────────────────────────
    print("[Phase 2] Extracting real features for FID...")
    real_imgs_fid = tr_imgs[:config.n_generate]
    real_feats = _extract_features(clf, real_imgs_fid, config.eval_batch_size, config.device)

    # ── 7. Train + evaluate each schedule ───────────────────────────────────
    all_per_lambda: list[dict] = []
    summary_rows: list[dict]   = []
    histories: dict             = {}

    for seed_idx in range(config.num_seeds):
        run_seed = config.seed + seed_idx
        for spec in specs:
            print(f"[Phase 2] Training {spec.name} (seed={run_seed})...")
            model, history = train_one_schedule(spec, tr_imgs, config, run_seed)
            histories[f"{spec.name}_seed{run_seed}"] = history

            # Per-λ MSE
            per_lam = evaluate_per_lambda(model, tr_imgs, config, trans_low, trans_high)
            per_lam.update({"schedule": spec.name, "seed": run_seed})
            all_per_lambda.append(per_lam)

            # Coverage M, S
            cov = compute_coverage_metrics(
                spec, config, trans_low, trans_high, dmsr_grid, dmsr_slope_abs
            )

            # Generate & FID
            if config.n_generate > 0:
                print(f"[Phase 2]   Generating {config.n_generate} samples...")
                gen_imgs = ddim_generate(model, config.n_generate, config)
                fid = compute_fid_phi(clf, real_imgs_fid, gen_imgs, config.eval_batch_size, config.device)
                clf_metrics = compute_classifier_metrics(clf, gen_imgs, config)
                plot_sample_grid(spec.name, gen_imgs, plots_dir / f"samples_{spec.name}.png")
            else:
                fid = float("nan")
                clf_metrics = {"classifier_confidence": float("nan"), "balance_error": float("nan"), "frac_class_0": float("nan")}
                gen_imgs = torch.empty(0)

            row: dict = {
                "schedule": spec.name,
                "seed":     run_seed,
                **cov,
                "fid_phi":  fid,
                **clf_metrics,
                "mean_mse":       per_lam["mean_mse"],
                "transition_mse": per_lam["transition_mse"],
                "low_noise_mse":  per_lam["low_noise_mse"],
                "high_noise_mse": per_lam["high_noise_mse"],
            }
            summary_rows.append(row)
            print(
                f"[Phase 2]   FID={fid:.2f}  M={cov['coverage_m']:.4f}"
                f"  mean_mse={per_lam['mean_mse']:.4f}"
            )

    # ── 8. Save artifacts ───────────────────────────────────────────────────
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_metrics_csv(summary_rows, out_dir / "metrics_summary.csv")
    save_per_lambda_csv(all_per_lambda, out_dir / "per_lambda_metrics.csv")
    save_summary_md(config, specs, summary_rows, dmsr_info, out_dir / "summary.md")

    plot_per_lambda_mse(all_per_lambda, config, trans_low, trans_high, lambda_r_star, plots_dir / "per_lambda_mse.png")
    if config.n_generate > 0 and not any(math.isnan(r.get("fid_phi", float("nan"))) for r in summary_rows):
        plot_fid_summary(summary_rows, plots_dir / "fid_summary.png")
        plot_coverage_tradeoff(summary_rows, plots_dir / "coverage_vs_fid.png")

    print(f"[Phase 2] Results saved to {out_dir}")
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 2 MNIST DMSR noise schedule experiment.")
    p.add_argument("--digits",          type=int, nargs=2, default=[0, 1])
    p.add_argument("--lambda-min",      type=float, default=-10.0)
    p.add_argument("--lambda-max",      type=float, default=10.0)
    p.add_argument("--rho",             type=float, default=0.5)
    p.add_argument("--clf-epochs",      type=int,   default=10)
    p.add_argument("--dmsr-grid-size",  type=int,   default=40)
    p.add_argument("--dmsr-n-samples",  type=int,   default=512)
    p.add_argument("--base-ch",         type=int,   default=32)
    p.add_argument("--train-steps",     type=int,   default=20000)
    p.add_argument("--batch-size",      type=int,   default=128)
    p.add_argument("--eval-grid-size",  type=int,   default=40)
    p.add_argument("--seed",            type=int,   default=20260526)
    p.add_argument("--num-seeds",       type=int,   default=1)
    p.add_argument("--device",          type=str,   default="auto",
                   help="Device to use: 'auto' (default), 'cuda', 'mps', or 'cpu'.")
    p.add_argument("--ddim-steps",      type=int,   default=50)
    p.add_argument("--n-generate",      type=int,   default=5000)
    p.add_argument("--gen-batch-size",  type=int,   default=100)
    p.add_argument("--data-root",       type=str,   default="./data")
    p.add_argument("--out-root",        type=Path,  default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[Phase 2] Using device: {device}")
    config = ExperimentConfig(
        digits=tuple(args.digits),
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        rho=args.rho,
        clf_epochs=args.clf_epochs,
        dmsr_grid_size=args.dmsr_grid_size,
        dmsr_n_samples=args.dmsr_n_samples,
        base_ch=args.base_ch,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_grid_size=args.eval_grid_size,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=device,
        ddim_steps=args.ddim_steps,
        n_generate=args.n_generate,
        gen_batch_size=args.gen_batch_size,
        data_root=args.data_root,
    )
    run(config, args.out_root)


if __name__ == "__main__":
    main()