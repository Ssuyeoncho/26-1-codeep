"""DMSR computation, noise schedules, training, DDIM sampling, evaluation,
그리고 seed 간 통계 집계·유의성 검정 프레임워크."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gpu_perf import autocast_ctx, make_grad_scaler, maybe_compile, resolve_amp

from .config import ExperimentConfig, ScheduleSpec
from .models import FeatureClassifier, MiniUNet, vp_alpha_sigma, vp_alpha_sigma_np, seed_everything


# ── Empirical DMSR_φ(λ) ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_empirical_dmsr(
    clf: FeatureClassifier,
    imgs_a: Tensor,
    imgs_b: Tensor,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DMSR_φ(λ) on a VP λ grid. Returns (lambda_grid, dmsr_values)."""
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

        phi_a = clf.get_features(alpha * imgs_a + sigma * torch.randn_like(imgs_a))
        phi_b = clf.get_features(alpha * imgs_b + sigma * torch.randn_like(imgs_b))

        mean_dist  = (phi_a.mean(0) - phi_b.mean(0)).norm().item()
        tr_a       = phi_a.var(0, unbiased=True).sum().item()
        tr_b       = phi_b.var(0, unbiased=True).sum().item()
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
    mask      = slope_abs >= rho * slope_abs.max()
    r_star_idx = int(np.argmax(slope_abs))
    in_region  = lambda_grid[mask]
    return float(in_region[0]), float(lambda_grid[r_star_idx]), float(in_region[-1])


# ── Noise Schedule Sampling ────────────────────────────────────────────────────

def _clamp_lambda(lam: Tensor, config: ExperimentConfig) -> Tensor:
    return lam.clamp(config.lambda_min, config.lambda_max)


def _sample_truncated_laplace(
    center: float, b: float, lo: float, hi: float, n: int, device: str
) -> Tensor:
    """Sample true TruncatedLaplace_[lo,hi](center,b) by rejection, not clamp."""
    dist = torch.distributions.Laplace(float(center), float(b))
    out = torch.empty(n, device=device)
    filled = 0
    while filled < n:
        need = n - filled
        cand = dist.sample((int(need * 1.6) + 8,)).to(device)
        cand = cand[(cand >= lo) & (cand <= hi)]
        take = min(cand.numel(), need)
        if take > 0:
            out[filled:filled + take] = cand[:take]
            filled += take
    return out.view(n, 1)


def sample_schedule(spec: ScheduleSpec, n: int, config: ExperimentConfig, device: str) -> Tensor:
    eps = 1e-5
    if spec.kind == "mixture":
        assert spec.center_lambda is not None and spec.scale is not None and spec.eta is not None
        lo, hi = config.lambda_min, config.lambda_max
        use_peak = torch.rand(n, 1, device=device) < float(spec.eta)
        unif = lo + (hi - lo) * torch.rand(n, 1, device=device)
        peak = _sample_truncated_laplace(float(spec.center_lambda), float(spec.scale),
                                         lo, hi, n, device)
        return torch.where(use_peak, peak, unif)

    if spec.kind == "dmsr_normal":
        assert spec.center_lambda is not None and spec.scale is not None
        return _clamp_lambda(spec.center_lambda + spec.scale * torch.randn(n, 1, device=device), config)

    if spec.kind == "dmsr_laplace":
        assert spec.center_lambda is not None and spec.scale is not None
        dist = torch.distributions.Laplace(spec.center_lambda, spec.scale)
        return _clamp_lambda(dist.sample((n, 1)).to(device), config)

    if spec.kind == "hang_laplace_lambda":
        b = spec.scale if spec.scale is not None else 0.5
        lam = torch.distributions.Laplace(0.0, b).sample((n, 1)).to(device)
        return _clamp_lambda(lam, config)

    if spec.kind == "cosine_vp":
        t = torch.rand(n, 1, device=device).clamp(eps, 1.0 - eps)
        lam = -2.0 * torch.log(torch.tan(0.5 * math.pi * t))
        return _clamp_lambda(lam, config)

    if spec.kind == "uniform":
        # λ 범위 전체에 균일. 어떤 noise level도 선호하지 않는 '무정보' 기준.
        lam = config.lambda_min + (config.lambda_max - config.lambda_min) * torch.rand(n, 1, device=device)
        return _clamp_lambda(lam, config)

    if spec.kind == "linear_vp":
        # VP linear-β schedule(DDPM/Song VP-SDE): β(t)=β_min+t(β_max−β_min), t~U(0,1).
        # ᾱ(t)=exp(−(β_min t + ½(β_max−β_min)t²)), λ=log(ᾱ/(1−ᾱ)).
        t = torch.rand(n, 1, device=device)
        bmin, bmax = config.linear_beta_min, config.linear_beta_max
        log_abar = -(bmin * t + 0.5 * (bmax - bmin) * t * t)
        abar = torch.exp(log_abar)
        lam = torch.log(abar / (1.0 - abar).clamp_min(1e-8))
        return _clamp_lambda(lam, config)

    raise ValueError(f"Unknown schedule kind: {spec.kind}")


# ── Analytic densities (for smooth plotting of the *designed* p_train) ──────────
#
# schedule은 우리가 해석적으로 설계한 분포이므로, 그림을 그릴 때 굳이 샘플을 뽑아
# 히스토그램(계단형 + 표본잡음)으로 그릴 필요가 없다. 아래 함수가 각 schedule의 진짜
# 확률밀도 p_train(λ)를 닫힌 형태(또는 linear는 수치 변수변환)로 돌려준다 → 매끄럽고
# 정확히 대칭인 곡선. (clamp는 ±λ_max 경계의 미세한 질량 이동일 뿐이라 설계 분포 자체는
# 아래 밀도가 맞다.)

def _normal_pdf(lam: np.ndarray, mu: float, s: float) -> np.ndarray:
    return np.exp(-0.5 * ((lam - mu) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))


def _laplace_pdf(lam: np.ndarray, c: float, b: float) -> np.ndarray:
    return np.exp(-np.abs(lam - c) / b) / (2.0 * b)


def _laplace_cdf(x: float, c: float, b: float) -> float:
    if x < c:
        return 0.5 * math.exp((x - c) / b)
    return 1.0 - 0.5 * math.exp(-(x - c) / b)


def _mixture_pdf(
    lam: np.ndarray, eta: float, c: float, b: float, lo: float, hi: float
) -> np.ndarray:
    inside = (lam >= lo) & (lam <= hi)
    z = _laplace_cdf(hi, c, b) - _laplace_cdf(lo, c, b)
    lap_trunc = np.where(inside, _laplace_pdf(lam, c, b) / max(z, 1e-12), 0.0)
    unif = np.where(inside, 1.0 / (hi - lo), 0.0)
    return (1.0 - eta) * unif + eta * lap_trunc


def _cosine_vp_pdf(lam: np.ndarray) -> np.ndarray:
    """cosine VP schedule이 유도하는 λ 밀도 = (1/2π)·sech(λ/2). 0 중심 대칭, 두꺼운 꼬리."""
    return 1.0 / (2.0 * math.pi * np.cosh(0.5 * lam))


def _linear_vp_pdf(lam: np.ndarray, config: ExperimentConfig, n: int = 8000) -> np.ndarray:
    """VP linear-β schedule의 λ 밀도. 닫힌형이 없어 변수변환을 수치적으로 계산한다.

    t~U(0,1)에서 λ(t)가 정해지므로 p(λ)=|dt/dλ|. dense t 격자에서 λ(t)와 dλ/dt를 구해
    밀도를 만든 뒤 질의 λ 격자에 보간한다.
    """
    t = np.linspace(1e-6, 1.0 - 1e-6, n)
    bmin, bmax = config.linear_beta_min, config.linear_beta_max
    log_abar = -(bmin * t + 0.5 * (bmax - bmin) * t * t)
    abar = np.exp(log_abar)
    lam_t = np.log(abar / (1.0 - abar))
    dens_t = 1.0 / np.abs(np.gradient(lam_t, t))      # p(λ)=|dt/dλ|
    order = np.argsort(lam_t)
    return np.interp(lam, lam_t[order], dens_t[order], left=0.0, right=0.0)


def schedule_density(spec: ScheduleSpec, lam: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """schedule의 해석적(설계된) 확률밀도 p_train(λ)를 λ 격자 위에서 반환한다."""
    lam = np.asarray(lam, dtype=float)
    k = spec.kind
    if k == "mixture":
        return _mixture_pdf(lam, float(spec.eta), float(spec.center_lambda),
                            float(spec.scale), config.lambda_min, config.lambda_max)
    if k == "dmsr_normal":
        return _normal_pdf(lam, float(spec.center_lambda), float(spec.scale))
    if k == "dmsr_laplace":
        return _laplace_pdf(lam, float(spec.center_lambda), float(spec.scale))
    if k == "hang_laplace_lambda":
        return _laplace_pdf(lam, 0.0, float(spec.scale if spec.scale is not None else 0.5))
    if k == "cosine_vp":
        return _cosine_vp_pdf(lam)
    if k == "uniform":
        d = config.lambda_max - config.lambda_min
        return np.where((lam >= config.lambda_min) & (lam <= config.lambda_max), 1.0 / d, 0.0)
    if k == "linear_vp":
        return _linear_vp_pdf(lam, config)
    raise ValueError(f"Unknown schedule kind for density: {k}")


def build_schedules(config: ExperimentConfig, lambda_r_star: float) -> list[ScheduleSpec]:
    """Phase 2.2 pilot schedule set.

    성능 비교용 factorial sweep이 아니라 Phase 3.2 파이프라인 검증용 4-config 구성이다.
    변수는 p_train(lambda)뿐이며 mixture의 (eta,beta)는 고정하고 중심만 0과 lambda_R*
    로 바꾼다.
    """
    specs: list[ScheduleSpec] = []
    if config.include_linear:
        specs.append(ScheduleSpec("linear_vp", "linear_vp",
                     note="VP linear-β (DDPM) induced λ density. 관행 baseline."))
    if config.include_uniform:
        specs.append(ScheduleSpec("uniform", "uniform",
                     note=f"Uniform(lambda in [{config.lambda_min},{config.lambda_max}]). 무정보 baseline."))
    specs.append(ScheduleSpec("cosine_vp", "cosine_vp",
                 note="VP cosine-induced lambda density truncated to the pilot range."))

    centers = [("dmsr", lambda_r_star)]
    if config.include_center0:
        centers.append(("at0", 0.0))
    for cname, cval in centers:
        ctxt = f"lambda_R*={lambda_r_star:.3f}" if cname == "dmsr" else "0"
        tag = "DMSR center" if cname == "dmsr" else "Hang-style center"
        for eta in config.mixture_etas:
            for b in config.mixture_bs:
                eta_s = f"{eta:g}"
                b_s = f"{b:g}"
                specs.append(ScheduleSpec(
                    f"mix_{cname}_eta{eta_s}_b{b_s}", "mixture",
                    center_lambda=cval, scale=b, eta=eta,
                    note=f"(1-{eta_s})*U[{config.lambda_min},{config.lambda_max}] + "
                         f"{eta_s}*TruncLaplace(center={ctxt}, b={b_s}). [{tag}]"))
    # Put the Hang-style center before DMSR in plots/tables for the baseline-to-proposal read.
    base = [s for s in specs if s.kind in {"linear_vp", "uniform", "cosine_vp"}]
    mixes = [s for s in specs if s.kind == "mixture"]
    mixes.sort(key=lambda s: (0 if "_at0_" in s.name else 1, s.name))
    return base + mixes


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_schedule(
    spec: ScheduleSpec,
    train_imgs: Tensor,
    train_labels: Tensor,
    config: ExperimentConfig,
    run_seed: int,
) -> tuple[MiniUNet, list[dict[str, float]]]:
    seed_everything(run_seed)
    device = config.device
    n_cls = config.num_classes if config.class_cond else 0
    model = MiniUNet(config.base_ch, config.time_emb_dim, num_classes=n_cls).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []
    report_every = max(1, config.train_steps // 10)

    # MNIST two-class는 작아서 전체를 GPU에 한 번만 올려두면 매 스텝 host→device 복사가
    # 사라진다(GPU가 굶지 않음). 인덱스도 같은 device에서 뽑는다.
    train_imgs = train_imgs.to(device)
    train_labels = train_labels.to(device)
    n_train = len(train_imgs)

    # ── GPU 가속: 혼합정밀(AMP) + torch.compile ───────────────────────────────
    amp_on, amp_dtype, use_scaler = resolve_amp(device, config.amp)
    scaler = make_grad_scaler(use_scaler)
    fwd = maybe_compile(model, config.compile_model)   # forward만 컴파일 객체로 호출
    micro_bs = config.micro_batch_size or config.batch_size
    micro_bs = max(1, min(micro_bs, config.batch_size))

    model.train()
    for step in range(1, config.train_steps + 1):
        idx   = torch.randint(0, n_train, (config.batch_size,), device=device)
        x0    = train_imgs[idx]
        y_all = train_labels[idx]
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        chunks = list(x0.split(micro_bs))
        y_chunks = list(y_all.split(micro_bs))
        batch_n = len(x0)
        for chunk, lbl in zip(chunks, y_chunks):
            # class-conditional: 라벨을 주되 cond_dropout_prob 만큼 null로 떨군다(CFG 학습).
            if n_cls > 0:
                y = lbl.clone()
                drop = torch.rand(len(y), device=device) < config.cond_dropout_prob
                y[drop] = n_cls                              # null 토큰
            else:
                y = None
            lam = sample_schedule(spec, len(chunk), config, device)  # (B, 1)
            alpha, sigma = vp_alpha_sigma(lam.view(-1, 1, 1, 1))
            eps = torch.randn_like(chunk)
            with autocast_ctx(device, amp_on, amp_dtype):
                pred_eps = fwd(alpha * chunk + sigma * eps, lam.view(-1), y)
                raw_loss = F.mse_loss(pred_eps, eps)
                loss = raw_loss * (len(chunk) / batch_n)
            loss_accum += float(raw_loss.detach().cpu()) * (len(chunk) / batch_n)

            if use_scaler:                              # fp16 경로
                scaler.scale(loss).backward()
            else:                                       # bf16/fp32 경로
                loss.backward()

        if use_scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        if step == 1 or step % report_every == 0 or step == config.train_steps:
            history.append({"step": float(step), "train_mse": loss_accum})

    model.eval()
    return model, history


# ── DDIM Sampling ──────────────────────────────────────────────────────────────

def _make_ddim_lambda_schedule(n_steps: int, lambda_min: float, lambda_max: float) -> np.ndarray:
    """VP cosine λ schedule for DDIM: noisy (small λ) → clean (large λ).

    Phase 3와 동일한 격자가 되도록 공용 함수(ddim_grid.cosine_lambda_grid)에 위임한다.
    """
    from ddim_grid import cosine_lambda_grid
    return cosine_lambda_grid(n_steps, lambda_min, lambda_max)


@torch.no_grad()
def ddim_generate(model: MiniUNet, n_samples: int, config: ExperimentConfig) -> Tensor:
    model.eval()
    amp_on, amp_dtype, _ = resolve_amp(config.device, config.amp)  # 생성도 AMP로 가속(추론)
    lam_sched = _make_ddim_lambda_schedule(
        config.ddim_steps, config.ddim_lambda_min, config.ddim_lambda_max
    )
    # Use vp_alpha_sigma_np for consistency with the rest of the codebase.
    _, sigma_init_arr = vp_alpha_sigma_np(lam_sched[:1])
    sigma_init = float(sigma_init_arr[0])

    use_cfg = config.class_cond and config.num_classes > 0

    all_imgs: list[Tensor] = []
    remaining = n_samples
    done = 0
    while remaining > 0:
        bs = min(config.gen_batch_size, remaining)
        remaining -= bs

        # class_cond이면 두 클래스를 균형 있게(번갈아) 생성 → FID 표본 분포를 맞춘다.
        if use_cfg:
            y = (torch.arange(done, done + bs, device=config.device) % config.num_classes)
            null = torch.full_like(y, config.num_classes)
        else:
            y = None

        x = sigma_init * torch.randn(bs, 1, config.img_size, config.img_size, device=config.device)

        for i in range(len(lam_sched) - 1):
            lc, ln = float(lam_sched[i]), float(lam_sched[i + 1])

            lam_t   = torch.full((bs,), lc, device=config.device)
            alpha_c = torch.sqrt(torch.sigmoid(lam_t)).view(bs, 1, 1, 1)
            sigma_c = torch.sqrt(torch.sigmoid(-lam_t)).view(bs, 1, 1, 1)

            a_n_arr, s_n_arr = vp_alpha_sigma_np(np.array([ln]))
            a_n, s_n = float(a_n_arr[0]), float(s_n_arr[0])

            with autocast_ctx(config.device, amp_on, amp_dtype):
                if use_cfg:
                    # CFG: eps = eps_uncond + w·(eps_cond − eps_uncond)
                    eps_c = model(x, lam_t, y).float()
                    eps_u = model(x, lam_t, null).float()
                    pred_eps = eps_u + config.cfg_scale * (eps_c - eps_u)
                else:
                    pred_eps = model(x, lam_t).float()
            pred_eps = pred_eps.float()                  # 이후 산술은 fp32로 안정적으로
            pred_x0  = ((x - sigma_c * pred_eps) / alpha_c.clamp_min(1e-8)).clamp(-1, 1)
            x = a_n * pred_x0 + s_n * pred_eps

        all_imgs.append(x.clamp(-1, 1).cpu())
        done += bs

    return torch.cat(all_imgs, 0)[:n_samples]


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_lambda(
    model: MiniUNet,
    train_imgs: Tensor,
    train_labels: Tensor,
    config: ExperimentConfig,
    transition_low: float,
    transition_high: float,
) -> dict:
    model.eval()
    lambda_grid = np.linspace(config.lambda_min, config.lambda_max, config.eval_grid_size)
    per_lambda_mse: list[float] = []
    n = len(train_imgs)
    n_cls = config.num_classes if config.class_cond else 0

    for lam_val in lambda_grid:
        idx   = torch.randint(0, n, (config.eval_batch_size,))
        x0    = train_imgs[idx].to(config.device)
        # 학습이 conditional이면 per-λ MSE도 진짜 라벨 조건으로 잰다(학습 task와 동일).
        y = train_labels[idx].to(config.device) if n_cls > 0 else None
        lam_t = torch.full((config.eval_batch_size,), lam_val, device=config.device)
        alpha = torch.sqrt(torch.sigmoid(lam_t)).view(-1, 1, 1, 1)
        sigma = torch.sqrt(torch.sigmoid(-lam_t)).view(-1, 1, 1, 1)
        eps   = torch.randn_like(x0)
        pred_eps = model(alpha * x0 + sigma * eps, lam_t, y)
        per_lambda_mse.append(F.mse_loss(pred_eps, eps).item())

    arr    = np.array(per_lambda_mse)
    t_mask = (lambda_grid >= transition_low) & (lambda_grid <= transition_high)

    def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if mask.any() else float("nan")

    return {
        "mean_mse":       float(arr.mean()),
        "transition_mse": _safe_mean(arr, t_mask),
        "low_noise_mse":  _safe_mean(arr, lambda_grid > transition_high),
        "high_noise_mse": _safe_mean(arr, lambda_grid < transition_low),
        "lambda_grid":    lambda_grid.tolist(),
        "per_lambda_mse": per_lambda_mse,
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
    slopes_at  = np.interp(sampled, dmsr_lambda_grid, dmsr_slope_abs)
    expected_s = float(np.mean(slopes_at))
    max_slope  = float(dmsr_slope_abs.max()) if dmsr_slope_abs.max() > 0 else 1.0
    return {
        "coverage_m":      coverage_m,
        "expected_s":      expected_s,
        "expected_s_norm": expected_s / max_slope,
    }


@torch.no_grad()
def extract_features(clf: FeatureClassifier, imgs: Tensor, batch_size: int, device: str) -> np.ndarray:
    clf.eval()
    feats: list[np.ndarray] = []
    for i in range(0, len(imgs), batch_size):
        feats.append(clf.get_features(imgs[i : i + batch_size].to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_phi_metrics(
    clf: FeatureClassifier,
    real_imgs: Tensor,
    gen_imgs: Tensor,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """φ-feature 공간에서 생성 품질 지표 일습을 계산한다.

    InceptionV3 다운로드 없이, 학습해 둔 분류기 φ의 penultimate feature 위에서
    공용 모듈 gen_metrics 로 FID·KID·Precision/Recall/Density/Coverage 를 모두 구한다.
    이 지표 정의는 Phase 3와 **완전히 동일**하다(같은 함수 사용).
    반환 키: fid_phi, kid_phi, kid_phi_std, precision_phi, recall_phi,
             density_phi, coverage_phi.
    """
    from gen_metrics import compute_feature_metrics

    real_f = extract_features(clf, real_imgs, batch_size, device)
    gen_f  = extract_features(clf, gen_imgs,  batch_size, device)
    return compute_feature_metrics(real_f, gen_f, suffix="_phi")


@torch.no_grad()
def evaluate_classifier_predictions(
    clf: FeatureClassifier,
    test_imgs: Tensor,
    test_labels: Tensor,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """φ 분류기를 테스트셋에서 평가해 (y_true, y_pred) 정수 레이블 배열을 반환한다.

    정답 레이블이 있는 유일한 곳이라, 여기서만 Accuracy/Precision/Recall/F1/Confusion
    같은 분류 지표가 정의된다(생성 샘플은 unconditional이라 정답이 없다).
    """
    clf.eval()
    preds: list[Tensor] = []
    for i in range(0, len(test_imgs), config.eval_batch_size):
        batch = test_imgs[i : i + config.eval_batch_size].to(config.device)
        preds.append(clf(batch).argmax(1).cpu())
    y_pred = torch.cat(preds).numpy().astype(int)
    y_true = test_labels.detach().cpu().numpy().astype(int)
    return y_true, y_pred


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

# ── 통계 분석 프레임워크 (공용 모듈 재사용) ─────────────────────────────────────
#
# seed 간 집계·유의성 검정 로직은 Phase 2·3가 **동일한 코드**를 쓰도록 최상위
# stats_analysis.py 로 분리했다. 여기서는 그대로 가져와 재노출(re-export)만 한다.
# (run.py 가 phase2.experiment 에서 이 이름들을 import 하므로 호환성 유지)
from stats_analysis import (  # noqa: E402,F401
    LOWER_IS_BETTER,
    aggregate_over_seeds,
    aggregate_per_lambda,
    aggregated_csv_fieldnames,
    paired_differences_vs_baseline,
    per_lambda_excess_and_skill,
    region_curve_mean,
    significance_tests,
)
