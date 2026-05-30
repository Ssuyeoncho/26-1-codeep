"""DMSR computation, noise schedules, training, DDIM sampling, and evaluation."""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .config import ExperimentConfig, ScheduleSpec
from .models import (
    EMA, FeatureClassifier, UNet,
    get_two_class_dataset, pin_memory_for,
    seed_everything, vp_alpha_sigma, vp_alpha_sigma_np,
)


# ── DMSR_phi computation ──────────────────────────────────────────────────────

@torch.no_grad()
def _extract_features(
    model: FeatureClassifier, imgs: Tensor, device: str, bs: int = 256
) -> Tensor:
    out: list[Tensor] = []
    model.eval()
    for i in range(0, len(imgs), bs):
        out.append(model.features(imgs[i:i + bs].to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def compute_dmsr_phi(
    classifier: FeatureClassifier,
    cfg: ExperimentConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Compute empirical DMSR_phi(lambda).

    Returns (lambda_grid, dmsr_vals, slope, transition_low, lambda_r_star, transition_high).
    """
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
        a, s = float(alpha_np[0]), float(sigma_np[0])

        noisy_a = a * imgs_a + s * torch.randn_like(imgs_a)
        noisy_b = a * imgs_b + s * torch.randn_like(imgs_b)

        feats_a = _extract_features(classifier, noisy_a, device)
        feats_b = _extract_features(classifier, noisy_b, device)

        mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
        dist = torch.norm(mu_a - mu_b).item()
        tr_a = ((feats_a - mu_a) ** 2).sum(1).mean().item()
        tr_b = ((feats_b - mu_b) ** 2).sum(1).mean().item()
        denom = math.sqrt((tr_a + tr_b) / 2.0 + 1e-12)
        dmsr_vals.append(dist / denom)

    dmsr_arr = np.array(dmsr_vals)
    slope = np.abs(np.gradient(dmsr_arr, lambda_grid))

    mask = slope >= cfg.rho * slope.max()
    t_low = float(lambda_grid[mask][0]) if mask.any() else float(lambda_grid[0])
    t_high = float(lambda_grid[mask][-1]) if mask.any() else float(lambda_grid[-1])
    lambda_r_star = float(lambda_grid[np.argmax(slope)])

    return lambda_grid, dmsr_arr, slope, t_low, lambda_r_star, t_high


# ── Schedule samplers ─────────────────────────────────────────────────────────

def _clamp_lambda(lam: Tensor, cfg: ExperimentConfig) -> Tensor:
    return lam.clamp(cfg.lambda_min, cfg.lambda_max)


def sample_schedule(spec: ScheduleSpec, n: int, cfg: ExperimentConfig, device: str) -> Tensor:
    eps = 1e-5
    if spec.kind == "dmsr_normal":
        lam = spec.center_lambda + spec.scale * torch.randn(n, 1, device=device)
        return _clamp_lambda(lam, cfg)
    if spec.kind == "dmsr_laplace":
        d = torch.distributions.Laplace(spec.center_lambda, spec.scale)
        return _clamp_lambda(d.sample((n, 1)).to(device), cfg)
    if spec.kind == "hang_laplace":
        d = torch.distributions.Laplace(0.0, spec.scale)
        return _clamp_lambda(d.sample((n, 1)).to(device), cfg)
    if spec.kind == "cosine_vp":
        t = torch.rand(n, 1, device=device).clamp(eps, 1.0 - eps)
        return _clamp_lambda(-2.0 * torch.log(torch.tan(0.5 * math.pi * t)), cfg)
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


# ── Training ──────────────────────────────────────────────────────────────────

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
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory_for(device), drop_last=True,
    )

    model = UNet(base_ch=cfg.base_ch, num_res_blocks=cfg.num_res_blocks).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []
    report_every = max(1, cfg.train_steps // 20)

    model.train()
    for step, (imgs, _) in enumerate(_infinite(loader), start=1):
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


# ── DDIM VP sampler ───────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model: UNet, n: int, cfg: ExperimentConfig, device: str) -> Tensor:
    """Deterministic DDIM in VP lambda space (lambda_min → lambda_max)."""
    lam_seq = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.ddim_steps + 1)
    model.eval()
    x = torch.randn(n, 3, cfg.image_size, cfg.image_size, device=device)

    for i in range(cfg.ddim_steps):
        lam_s, lam_t = float(lam_seq[i]), float(lam_seq[i + 1])
        a_s = float(vp_alpha_sigma_np(np.array([lam_s]))[0][0])
        s_s = float(vp_alpha_sigma_np(np.array([lam_s]))[1][0])
        a_t = float(vp_alpha_sigma_np(np.array([lam_t]))[0][0])
        s_t = float(vp_alpha_sigma_np(np.array([lam_t]))[1][0])

        lam_in = torch.full((n, 1), lam_s, device=device)
        eps_hat = model(x, lam_in)
        x0_hat = (x - s_s * eps_hat) / (a_s + 1e-8)
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        x = a_t * x0_hat + s_t * eps_hat

    return x.clamp(-1.0, 1.0)


def generate_samples_batched(
    model: UNet, n: int, cfg: ExperimentConfig, device: str, bs: int = 256
) -> Tensor:
    out: list[Tensor] = []
    for start in range(0, n, bs):
        b = min(bs, n - start)
        out.append(ddim_sample(model, b, cfg, device).cpu())
    return torch.cat(out)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_per_lambda_mse(
    model: UNet, cfg: ExperimentConfig, device: str
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
    classifier: FeatureClassifier, gen_imgs: Tensor, device: str, bs: int = 256
) -> tuple[float, float]:
    classifier.eval()
    probs_list: list[Tensor] = []
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
    model: UNet, cfg: ExperimentConfig, device: str, n_fake: int
) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  [FID] torchmetrics not installed — skipping. pip install torchmetrics[image]")
        return float("nan")

    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    ds = get_two_class_dataset(cfg, train=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=pin_memory_for(device))

    for imgs, _ in loader:
        fid_metric.update(((imgs.clamp(-1, 1) + 1) / 2).to(device), real=True)

    model.eval()
    bs = 256
    for start in range(0, n_fake, bs):
        b = min(bs, n_fake - start)
        with torch.no_grad():
            fake = ddim_sample(model, b, cfg, device)
        fid_metric.update(((fake + 1) / 2).to(device), real=False)

    return float(fid_metric.compute())
