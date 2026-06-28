"""DMSR computation, noise schedules, training, DDIM sampling, and evaluation."""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from gpu_perf import autocast_ctx, make_grad_scaler, maybe_compile, resolve_amp

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
    loader_kwargs: dict = dict(
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory_for(device), drop_last=True,
    )
    if cfg.num_workers > 0:
        # 워커를 살려두면(persistent) epoch마다 재생성 비용이 없어 GPU가 안 굶는다.
        # prefetch_factor는 워커가 있을 때만 유효(구버전 torch 호환을 위해 조건부 전달).
        loader_kwargs.update(persistent_workers=True, prefetch_factor=cfg.prefetch_factor)
    loader = DataLoader(ds, **loader_kwargs)

    model = UNet(base_ch=cfg.base_ch, num_res_blocks=cfg.num_res_blocks).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []
    report_every = max(1, cfg.train_steps // 20)

    # ── GPU 가속: 혼합정밀(AMP) + torch.compile ───────────────────────────────
    amp_on, amp_dtype, use_scaler = resolve_amp(device, cfg.amp)
    scaler = make_grad_scaler(use_scaler)
    # forward 만 컴파일 객체로 호출하고, opt/ema/state_dict 은 원본 model 사용
    # (둘은 같은 파라미터를 공유하므로 갱신이 그대로 반영됨).
    fwd = maybe_compile(model, cfg.compile_model)
    micro_bs = cfg.micro_batch_size or cfg.batch_size
    micro_bs = max(1, min(micro_bs, cfg.batch_size))

    model.train()
    for step, (imgs, _) in enumerate(_infinite(loader), start=1):
        if step > cfg.train_steps:
            break
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        chunks = list(imgs.split(micro_bs))
        batch_n = len(imgs)
        for chunk in chunks:
            chunk = chunk.to(device, non_blocking=True)
            lam = sample_schedule(spec, len(chunk), cfg, device)     # (B, 1)
            alpha, sigma = vp_alpha_sigma(lam)                       # (B, 1)
            eps = torch.randn_like(chunk)
            x_lam = alpha[:, :, None, None] * chunk + sigma[:, :, None, None] * eps

            with autocast_ctx(device, amp_on, amp_dtype):
                pred_eps = fwd(x_lam, lam)
                raw_loss = F.mse_loss(pred_eps, eps)
                loss = raw_loss * (len(chunk) / batch_n)
            loss_accum += float(raw_loss.detach().cpu()) * (len(chunk) / batch_n)

            if use_scaler:                              # fp16 경로: scaler로 underflow 방지
                scaler.scale(loss).backward()
            else:                                       # bf16/fp32 경로
                loss.backward()

        if use_scaler:
            scaler.unscale_(opt)                        # clip 전에 gradient 원복
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ema.update(model)

        if step == 1 or step % report_every == 0 or step == cfg.train_steps:
            history.append({"step": float(step), "train_mse": loss_accum})
            print(f"  [{spec.name}] step {step}/{cfg.train_steps}  loss={loss_accum:.5f}")

    return model, ema, history


# ── DDIM VP sampler ───────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model: UNet, n: int, cfg: ExperimentConfig, device: str) -> Tensor:
    """Deterministic DDIM in VP lambda space (lambda_min → lambda_max).

    생성도 학습과 동일한 AMP(autocast)로 forward 하여 처리량을 높인다(추론 전용이라
    GradScaler는 불필요). 모든 schedule에 똑같이 적용되므로 비교 공정성에는 영향 없다.
    """
    # Phase 2와 동일한 공용 cosine 격자(중앙 집중). DMSR 분석 범위가 아니라
    # ddim_lambda_min/max 를 쓴다.
    from ddim_grid import cosine_lambda_grid
    lam_seq = cosine_lambda_grid(cfg.ddim_steps, cfg.ddim_lambda_min, cfg.ddim_lambda_max)
    model.eval()
    amp_on, amp_dtype, _ = resolve_amp(device, cfg.amp)
    x = torch.randn(n, 3, cfg.image_size, cfg.image_size, device=device)

    for i in range(cfg.ddim_steps):
        lam_s, lam_t = float(lam_seq[i]), float(lam_seq[i + 1])
        a_s = float(vp_alpha_sigma_np(np.array([lam_s]))[0][0])
        s_s = float(vp_alpha_sigma_np(np.array([lam_s]))[1][0])
        a_t = float(vp_alpha_sigma_np(np.array([lam_t]))[0][0])
        s_t = float(vp_alpha_sigma_np(np.array([lam_t]))[1][0])

        lam_in = torch.full((n, 1), lam_s, device=device)
        with autocast_ctx(device, amp_on, amp_dtype):
            eps_hat = model(x, lam_in)
        eps_hat = eps_hat.float()                    # 이후 산술은 fp32로 안정적으로
        x0_hat = (x - s_s * eps_hat) / (a_s + 1e-8)
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        x = a_t * x0_hat + s_t * eps_hat

    return x.clamp(-1.0, 1.0)


def generate_samples_batched(
    model: UNet, n: int, cfg: ExperimentConfig, device: str, bs: int | None = None
) -> Tensor:
    bs = bs if bs is not None else cfg.gen_batch_size
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


# ── φ-feature 공간 생성 품질 지표 (Phase 2와 동일한 정의) ───────────────────────

@torch.no_grad()
def gather_real_images(cfg: ExperimentConfig, device: str, n_max: int) -> Tensor:
    """test split에서 진짜 이미지를 최대 n_max장 모아 하나의 텐서로 반환한다.

    φ-feature 기반 지표(FID/KID/PRDC)의 '진짜 분포' 쪽 표본으로 쓴다.
    """
    ds = get_two_class_dataset(cfg, train=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=cfg.num_workers)
    imgs: list[Tensor] = []
    total = 0
    for x, _ in loader:
        imgs.append(x)
        total += len(x)
        if total >= n_max:
            break
    return torch.cat(imgs)[:n_max]


def compute_phi_metrics(
    classifier: FeatureClassifier,
    real_imgs: Tensor,
    gen_imgs: Tensor,
    device: str,
) -> dict[str, float]:
    """φ-feature 공간에서 FID·KID·Precision/Recall/Density/Coverage를 계산한다.

    Phase 2와 **완전히 동일한** 공용 함수(gen_metrics.compute_feature_metrics)를 쓴다.
    Inception-FID(compute_fid)는 CIFAR 표준 헤드라인 지표로 따로 두고, 이 φ 지표들은
    두 phase를 일관되게 잇는 비교 축으로 사용한다.
    반환 키: fid_phi, kid_phi, kid_phi_std, precision_phi, recall_phi,
             density_phi, coverage_phi.
    """
    from gen_metrics import compute_feature_metrics

    real_f = _extract_features(classifier, real_imgs, device).numpy()
    gen_f  = _extract_features(classifier, gen_imgs,  device).numpy()
    return compute_feature_metrics(real_f, gen_f, suffix="_phi")


def compute_fid(
    gen_imgs: Tensor, cfg: ExperimentConfig, device: str
) -> float:
    """Compute FID using pre-generated images against the full training set.

    Uses train=True (10k images for two-class CIFAR) so the real covariance
    estimate is well-conditioned relative to InceptionV3's 2048-dim features.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  [FID] torchmetrics not installed — skipping. pip install torchmetrics[image]")
        return float("nan")

    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    ds = get_two_class_dataset(cfg, train=True)
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=pin_memory_for(device))

    for imgs, _ in loader:
        fid_metric.update(((imgs.clamp(-1, 1) + 1) / 2).to(device), real=True)

    bs = 256
    for start in range(0, len(gen_imgs), bs):
        batch = ((gen_imgs[start : start + bs].clamp(-1, 1) + 1) / 2).to(device)
        fid_metric.update(batch, real=False)

    return float(fid_metric.compute())
