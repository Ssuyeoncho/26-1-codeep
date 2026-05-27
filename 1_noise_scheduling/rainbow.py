"""
====================================================================
VP-DDPM Toy Implementation for Beta Scheduling Study
====================================================================

목적
----
1) VP(Variance Preserving) 형식의 DDPM을 2D toy 데이터(Gaussian Mixture)로 구현
2) 다양한 beta schedule을 갈아끼우며 학습
3) 학습이 끝난 모델에 대해 SNR, logSNR, alpha_bar, p(lambda) 등 지표 그래프를 비교
4) 향후: "최적 SNR 곡선이 나오면 그게 어떤 beta schedule(linear/cosine/...)에 해당하는지" 역추적

이 코드는 4090 1장이면 1~2분 안에 모든 schedule 학습이 끝나는 규모로 맞췄음.
=====================================================================

용어 빠른 정리
----------------------------------------------------------------------
- DDPM       : Denoising Diffusion Probabilistic Model. 이미지를 점점 노이즈로 만들었다가
               그 반대를 학습해서 노이즈 -> 이미지를 만들어내는 모델.
- VP-SDE     : "Variance Preserving" 스타일. 매 step마다 분산이 1 근처로 유지됨.
               x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
               여기서 alpha_t = 1 - beta_t, alpha_bar_t = prod(alpha_1..alpha_t).
- beta_t     : t번째 step에서 "얼마나 노이즈를 섞을지" 정하는 값 (0~1).
- alpha_bar_t: 시점 t까지 살아남은 "원본 신호의 비율" 같은 양 (sqrt 하면 신호 계수).
- SNR(t)     : alpha_bar_t / (1 - alpha_bar_t). 신호/노이즈 비율.
- logSNR(t)  : log(SNR(t)). 줄여서 λ(lambda)라고 부름.
               Hang et al. 논문에서는 이 λ가 핵심 변수임.
- p(λ)       : 학습 중 timestep t를 uniform하게 뽑을 때, λ가 따르는 분포.
               결과적으로 "어느 노이즈 레벨에 학습을 얼마나 집중하느냐"를 의미.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

# GPU 전역 설정: cuDNN이 입력 크기에 맞는 가장 빠른 알고리즘을 자동 탐색
torch.backends.cudnn.benchmark = True


# =====================================================================
# 1. Beta Schedules
# ---------------------------------------------------------------------
# 다양한 beta_t 스케줄을 한 곳에 모아둠.
# 모두 길이 T짜리 1D 텐서를 반환. 각 원소는 (0, 1) 안에 있어야 함.
# 새 schedule을 실험하고 싶으면 여기에 함수를 하나 추가하고
# SCHEDULES dict에 등록하면 끝.
# =====================================================================

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """가장 원조 DDPM(Ho et al., 2020)이 쓴 linear schedule.
    beta_1=1e-4 에서 beta_T=0.02 까지 선형 증가.
    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Improved DDPM (Nichol & Dhariwal, 2021)의 cosine schedule.
    alpha_bar_t = cos^2((t/T + s)/(1+s) * pi/2) / cos^2(s/(1+s)*pi/2)
    로 정의한 뒤 beta_t = 1 - alpha_bar_t / alpha_bar_{t-1} 로 역산.
    초반에 노이즈가 너무 빨리 끼는 것을 막아줌.
    """
    t = torch.linspace(0, T, T + 1) / T  # 길이 T+1
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]                  # alpha_bar_0 = 1 로 정규화
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(min=1e-8, max=0.999)


def quadratic_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """sqrt를 linear로 잡고 제곱. 즉 beta_t가 t의 2차식처럼 증가."""
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2


def sigmoid_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """sigmoid를 0..1로 정규화해서 beta_start~beta_end에 매핑."""
    x = torch.linspace(-6, 6, T)
    s = torch.sigmoid(x)
    s = (s - s.min()) / (s.max() - s.min())
    return beta_start + (beta_end - beta_start) * s


def laplace_alpha_bar_schedule(T: int, mu: float = 0.0, b: float = 0.5) -> torch.Tensor:
    """
    Hang et al. (ICCV 2025)의 Laplace schedule.
    p(λ) = (1 / 2b) * exp(-|λ - μ| / b) (Laplace 분포)로 두면
    inverse-CDF로부터 t↔λ 관계가:
        λ(u) = μ - b * sign(u - 0.5) * log(1 - 2|u - 0.5|),    u ∈ (0, 1)
    이 식에서 u=0일 때 λ=+∞, u=1일 때 λ=-∞ 라서
    "u가 작을수록 신호가 강하다(λ↑)" = "u가 작을수록 t가 작다"의 컨벤션을 맞추려면
    u = t/T 로 두면 끝. (Hang 논문 Table 1의 표기는 t↑ → λ↓.)
    얻은 λ(t)로부터 VP 관계 alpha_bar_t = sigmoid(λ) 를 쓰면
    alpha_bar는 t↑일 때 단조감소 → 정상적인 forward process가 된다.
    """
    # u = t/T 를 (1e-5, 1-1e-5)로 가져와서 양끝 발산 방지
    u = torch.linspace(1e-5, 1 - 1e-5, T)
    # u→0 (t=0, 신호 강함) 에서 λ → +∞,
    # u→1 (t=T, 노이즈) 에서 λ → -∞ 가 되도록 부호 선택.
    # 일반 Laplace inverse-CDF는 λ(u) = μ - b·sign(u-0.5)·log(1-2|u-0.5|) 인데
    # 그 식은 "u가 클수록 λ가 커짐"이라 우리 컨벤션과 반대. 부호 통째로 뒤집어 사용:
    sign = torch.sign(0.5 - u)
    lam = mu + b * sign * torch.log(1 - 2 * torch.abs(u - 0.5)) * (-1)
    #  ── 풀어쓰면:  λ(u) = μ - b·sign(0.5-u)·log(1-2|u-0.5|)
    #  ── u=0 근처: sign=+1, log(1-2·0.5)=log(0)=-∞ → λ = μ - b·(+1)·(-∞) = +∞  ✓
    # VP 관계: alpha_bar_t = sigmoid(λ_t).  λ↓ 이면 alpha_bar↓ (신호 ↓, 노이즈 ↑)
    alpha_bar = torch.sigmoid(lam).clamp(min=1e-6, max=1 - 1e-6)
    # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}.  alpha_bar_0 의 직전 값은 1로 가정.
    alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
    betas = 1.0 - alpha_bar / alpha_bar_prev
    return betas.clamp(min=1e-8, max=0.999)


def logistic_alpha_bar_schedule(T: int, k: float = 0.015, t0_ratio: float = 0.6) -> torch.Tensor:
    """
    Lin et al. (NeurIPS 2024)의 Logistic schedule.
    alpha_bar_t = 1 / (1 + exp(-k(t - t0)))  형태에서 시작.
    원 논문은 inversion 안정성(t=0에서 미분이 발산하지 않게)을 위해 도입했음.
    여기서는 SNR 곡선 비교용으로 그대로 가져옴.
    """
    t = torch.arange(T, dtype=torch.float32)
    t0 = t0_ratio * T
    # 논문 정의는 noise scale(1-alpha_bar)에 가까운데, 우리는 t↑일수록 신호↓ 되도록 뒤집어 사용
    a = 1.0 / (1.0 + torch.exp(-k * (t - t0)))
    # 위 식은 t↑일 때 a↑(신호↑). 우리 컨벤션은 반대니까 1 - a 로 사용
    alpha_bar = 1.0 - a
    alpha_bar = alpha_bar.clamp(min=1e-6, max=1 - 1e-6)
    alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
    betas = 1.0 - alpha_bar / alpha_bar_prev
    return betas.clamp(min=1e-8, max=0.999)


# 새 schedule 추가하려면 여기 등록만 하면 됨
SCHEDULES: Dict[str, Callable[[int], torch.Tensor]] = {
    "linear":    lambda T: linear_beta_schedule(T),
    "cosine":    lambda T: cosine_beta_schedule(T),
    "quadratic": lambda T: quadratic_beta_schedule(T),
    "sigmoid":   lambda T: sigmoid_beta_schedule(T),
    "laplace":   lambda T: laplace_alpha_bar_schedule(T, mu=0.0, b=0.5),
    "logistic":  lambda T: logistic_alpha_bar_schedule(T, k=0.015, t0_ratio=0.6),
}


# =====================================================================
# 2. Diffusion utilities
# ---------------------------------------------------------------------
# beta가 정해지면 그로부터 alpha, alpha_bar, SNR, logSNR을 미리 계산해두는 헬퍼 클래스.
# 학습/샘플링/그래프 그리기 모두 여기에 있는 텐서를 참조함.
# =====================================================================

class VPDiffusion:
    """
    VP-DDPM의 전/역 과정을 담당.

    forward (학습 시 noise 추가):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps~N(0,I)

    reverse (샘플링):
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * eps_theta) + sigma_t * z
        (Ho et al. 2020 식 그대로)
    """

    def __init__(self, betas: torch.Tensor, device: str = "cuda"):
        self.device = device
        self.betas = betas.to(device)                              # (T,)
        self.T = len(betas)
        self.alphas = 1.0 - self.betas                             # alpha_t = 1 - beta_t
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)        # alpha_bar_t = ∏ alpha_i
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)         # 신호 계수
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)  # 노이즈 계수
        # SNR, logSNR. 0/0 방지를 위해 클램프.
        self.snr = self.alpha_bars / (1 - self.alpha_bars).clamp(min=1e-20)
        self.log_snr = torch.log(self.snr.clamp(min=1e-20))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """주어진 깨끗한 x0와 timestep t로부터 노이즈 낀 x_t를 한 방에 샘플.
        t는 정수 텐서 (B,), x0/noise는 (B, D)."""
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_omab * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, int]) -> torch.Tensor:
        """순수 노이즈에서 시작해 t=T-1 -> 0 까지 한 step씩 디노이즈해서 샘플 생성."""
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps_theta = model(x, t_batch)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            # 평균: 1/sqrt(alpha_t) * (x - (beta_t / sqrt(1-alpha_bar_t)) * eps_theta)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
            )
            if t > 0:
                # 마지막 step만 noise 안 더함 (DDPM 표준)
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * z
            else:
                x = mean
        return x


# =====================================================================
# 3. Toy dataset: 2D Gaussian Mixture
# ---------------------------------------------------------------------
# 8개의 점이 원형으로 배치된 Gaussian mixture. 2D라 그래프 그리기 쉽고,
# 4090 위에서 거의 즉시 학습됨. "DDPM이 분포를 잘 복원했는지" 시각화 용도.
# =====================================================================

class GMM2D(Dataset):
    def __init__(self, n_samples: int = 20000, n_modes: int = 8,
                 radius: float = 4.0, std: float = 0.2, seed: int = 0):
        rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
        centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        which = rng.integers(0, n_modes, size=n_samples)
        noise = rng.normal(0, std, size=(n_samples, 2))
        self.data = centers[which] + noise
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =====================================================================
# 4. Tiny denoising MLP
# ---------------------------------------------------------------------
# 2D 데이터용이니까 진짜 작은 MLP면 충분.
# 입력: (x_t, t)  →  출력: 예측한 노이즈 eps_theta
# t는 sinusoidal embedding을 거쳐 hidden vector로 변환.
# =====================================================================

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer에서 쓰는 positional encoding과 같은 방식.
    timestep t (정수, shape (B,))를 (B, dim) 벡터로 변환."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TinyDenoiser(nn.Module):
    def __init__(self, data_dim: int = 2, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),            nn.SiLU(),
            nn.Linear(hidden, hidden),            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_time_embedding(t, self.time_dim)
        emb = self.time_mlp(emb)
        return self.net(torch.cat([x, emb], dim=-1))


# =====================================================================
# 5. Training loop
# ---------------------------------------------------------------------
# 표준 epsilon-prediction loss:
#   1) 배치 x_0 가져옴
#   2) 각 샘플마다 t ~ Uniform{0,...,T-1} 뽑음
#   3) eps ~ N(0,I)
#   4) x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps
#   5) MSE( eps_theta(x_t, t),  eps )  를 최소화
# =====================================================================

@dataclass
class TrainConfig:
    T: int = 1000
    batch_size: int = 4096     # GPU 메모리 여유 있으면 8192까지 올려도 됨
    n_steps: int = 5000
    lr: float = 2e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 500
    num_workers: int = 4       # CPU 코어 수에 맞게 조절 (보통 4~8)
    use_amp: bool = True       # Automatic Mixed Precision (CUDA 전용)


def train_one_schedule(schedule_name: str, cfg: TrainConfig) -> Tuple[nn.Module, VPDiffusion, List[float]]:
    """주어진 schedule로 모델 하나 학습. (model, diffusion_helper, loss_history) 반환."""
    is_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    amp_enabled = cfg.use_amp and is_cuda

    print(f"\n[Train] schedule={schedule_name}  device={cfg.device}  AMP={amp_enabled}")

    betas = SCHEDULES[schedule_name](cfg.T)
    diff = VPDiffusion(betas, device=cfg.device)

    dataset = GMM2D(n_samples=20000)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=is_cuda,          # GPU 전송 전 메모리를 page-locked으로 고정
        num_workers=cfg.num_workers if is_cuda else 0,
        persistent_workers=is_cuda and cfg.num_workers > 0,
    )
    loader_iter = iter(loader)

    model = TinyDenoiser().to(cfg.device)

    # torch.compile: PyTorch 2.0+에서 커널 fusion / 최적화 그래프 생성
    if is_cuda and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    losses: List[float] = []
    for step in range(cfg.n_steps):
        try:
            x0 = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x0 = next(loader_iter)

        # non_blocking: CPU→GPU 전송을 다음 CUDA 연산과 겹쳐서 대기시간 제거
        x0 = x0.to(cfg.device, non_blocking=True)

        t = torch.randint(0, cfg.T, (x0.size(0),), device=cfg.device)
        noise = torch.randn_like(x0)
        x_t = diff.q_sample(x0, t, noise)

        # autocast: forward pass를 float16으로 실행해 Tensor Core 활용
        with torch.amp.autocast(device_type="cuda" if is_cuda else "cpu", enabled=amp_enabled):
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        losses.append(loss.item())
        if (step + 1) % cfg.log_every == 0:
            recent = np.mean(losses[-cfg.log_every:])
            print(f"  step {step+1:5d} / {cfg.n_steps}  loss(recent)={recent:.4f}")

    return model, diff, losses


# =====================================================================
# 6. Analysis: SNR / logSNR / alpha_bar / p(lambda) 그래프
# ---------------------------------------------------------------------
# 학습 자체에는 관여하지 않지만, 팀 연구 주제(어떤 SNR 곡선이 가장 좋은가)의
# 핵심 분석 도구. p(λ)는 t가 uniform일 때 λ=logSNR이 따르는 분포로,
# logSNR(t)의 미분 |dλ/dt|의 역수를 정규화해서 추정한다.
# =====================================================================

def compute_p_lambda(log_snr: torch.Tensor,
                     n_bins: int = 60,
                     lam_range: Tuple[float, float] = (-15, 15)) -> Tuple[np.ndarray, np.ndarray]:
    """
    p(λ) 추정.
    t ~ Uniform(0, T-1) 일 때 λ(t) = log SNR(t) 의 히스토그램.
    이론적으로는 p(λ) = |dt/dλ| 이지만(논문 식 3), 토이에서는 그냥 히스토그램으로 충분.
    """
    lam = log_snr.detach().cpu().numpy()
    # 학습용 범위로 클립 (양끝의 극단값이 히스토그램을 짓누르는 것 방지)
    lam = lam[(lam > lam_range[0]) & (lam < lam_range[1])]
    hist, edges = np.histogram(lam, bins=n_bins, range=lam_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def plot_schedule_diagnostics(diffusions: Dict[str, VPDiffusion],
                              out_path: str = "schedule_diagnostics.png"):
    """4개 subplot: beta, alpha_bar, logSNR, p(lambda) 한 번에 그림."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for name, diff in diffusions.items():
        t = np.arange(diff.T)

        axes[0, 0].plot(t, diff.betas.cpu().numpy(), label=name)
        axes[0, 1].plot(t, diff.alpha_bars.cpu().numpy(), label=name)
        axes[1, 0].plot(t, diff.log_snr.cpu().numpy(), label=name)

        centers, p = compute_p_lambda(diff.log_snr)
        axes[1, 1].plot(centers, p, label=name)

    axes[0, 0].set_title(r"$\beta_t$")
    axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel(r"$\beta_t$")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_title(r"$\bar{\alpha}_t$ (signal coefficient squared)")
    axes[0, 1].set_xlabel("t"); axes[0, 1].set_ylabel(r"$\bar{\alpha}_t$")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_title(r"$\log \mathrm{SNR}(t) = \lambda(t)$")
    axes[1, 0].set_xlabel("t"); axes[1, 0].set_ylabel(r"$\lambda$")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_title(r"$p(\lambda)$  (induced by t ~ Uniform)")
    axes[1, 1].set_xlabel(r"$\lambda$"); axes[1, 1].set_ylabel(r"$p(\lambda)$")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[Saved] {out_path}")
    plt.close(fig)


def plot_samples(samples_per_schedule: Dict[str, np.ndarray],
                 real: np.ndarray,
                 out_path: str = "samples.png"):
    """각 schedule로 학습한 모델이 만든 샘플들을 ground truth와 함께 시각화."""
    n = len(samples_per_schedule) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    axes[0].scatter(real[:, 0], real[:, 1], s=4, alpha=0.4)
    axes[0].set_title("real data"); axes[0].set_aspect("equal")
    axes[0].set_xlim(-6, 6); axes[0].set_ylim(-6, 6); axes[0].grid(alpha=0.3)

    for ax, (name, s) in zip(axes[1:], samples_per_schedule.items()):
        ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.4)
        ax.set_title(name); ax.set_aspect("equal")
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[Saved] {out_path}")
    plt.close(fig)


def plot_losses(loss_histories: Dict[str, List[float]],
                out_path: str = "loss_curves.png",
                smooth: int = 50):
    """학습 loss 곡선 비교. 이동평균으로 살짝 부드럽게."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, hist in loss_histories.items():
        h = np.array(hist)
        if len(h) >= smooth:
            kernel = np.ones(smooth) / smooth
            h_s = np.convolve(h, kernel, mode="valid")
            ax.plot(h_s, label=name)
        else:
            ax.plot(h, label=name)
    ax.set_xlabel("step"); ax.set_ylabel("MSE loss (smoothed)")
    ax.set_title("Training loss per schedule")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[Saved] {out_path}")
    plt.close(fig)


# =====================================================================
# 7. Main entry
# =====================================================================

def _next_run_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(base, exist_ok=True)
    existing = [
        int(d.split("_")[0])
        for d in os.listdir(base)
        if d.split("_")[0].isdigit()
    ]
    n = max(existing, default=0) + 1
    return os.path.join(base, f"{n}_")


def main():
    out_dir = _next_run_dir()
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Run] 결과 저장 위치: {out_dir}")

    cfg = TrainConfig(
        T=1000,
        batch_size=512,
        n_steps=5000,
        lr=2e-3,
    )

    # 비교하고 싶은 schedule들 선택. 새로운 거 추가하려면 SCHEDULES dict에 등록 후 여기에 추가.
    schedule_names = ["linear", "cosine", "laplace", "logistic"]

    # 학습 결과 저장용
    diffusions: Dict[str, VPDiffusion] = {}
    loss_histories: Dict[str, List[float]] = {}
    samples_per_schedule: Dict[str, np.ndarray] = {}

    for name in schedule_names:
        model, diff, losses = train_one_schedule(name, cfg)
        diffusions[name] = diff
        loss_histories[name] = losses

        # 1024개 샘플 생성해서 분포 시각화용으로 저장
        model.eval()
        with torch.no_grad():
            samples = diff.p_sample_loop(model, shape=(1024, 2)).cpu().numpy()
        samples_per_schedule[name] = samples

    # --- 분석 그래프들 ---
    plot_schedule_diagnostics(diffusions, out_path=os.path.join(out_dir, "schedule_diagnostics.png"))
    plot_losses(loss_histories, out_path=os.path.join(out_dir, "loss_curves.png"))

    real = GMM2D(n_samples=2000, seed=123).data.numpy()
    plot_samples(samples_per_schedule, real, out_path=os.path.join(out_dir, "samples.png"))

    print(f"\nDone. 그림은 {out_dir} 에 저장됨.")
    print("다음 단계 아이디어:")
    print("  - schedule_diagnostics.png 의 p(lambda)/logSNR 모양과 samples.png의 분포 품질을 짝지어 보기")
    print("  - laplace의 b, logistic의 (k, t0)를 바꿔가며 'p(lambda) 봉우리 위치 vs 샘플 품질' 곡선 그리기")
    print("  - 반대로 '내가 원하는 p(lambda) 모양'을 하나 정해두고 거기서 beta를 역산하는 함수 만들기")


if __name__ == "__main__":
    main()