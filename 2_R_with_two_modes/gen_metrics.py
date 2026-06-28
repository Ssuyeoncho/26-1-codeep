"""Phase 2·3 공용 생성 품질 지표 (feature 벡터 위에서 계산).

FID 하나만으로는 '품질'과 '다양성'을 구분하지 못하기 때문에, 이미지 생성 연구에서
표준으로 쓰이는 보조 지표들을 함께 제공한다. 모든 함수는 **feature 행렬**
(real_features, gen_features; shape = (N, D))을 입력으로 받는다. 이미지를 feature로
바꾸는 부분(어떤 encoder φ를 쓸지)은 각 phase가 책임지고, 여기서는 순수하게 수치
계산만 한다. 덕분에 Phase 2(φ = MNIST 분류기)와 Phase 3(φ = CIFAR 분류기, 또는
Inception)이 **완전히 동일한 지표 정의**를 공유할 수 있다.

제공 지표:
  - FID  : 두 분포의 평균·공분산 차이 (Fréchet distance). 작을수록 좋음.
  - KID  : 다항 커널 MMD². FID보다 표본이 적을 때 편향이 작아 신뢰성이 높음. 작을수록 좋음.
  - Precision / Recall (Kynkäänniemi et al. 2019) : 품질(사실성) / 다양성(coverage) 분리.
  - Density / Coverage (Naeem et al. 2020)        : Precision/Recall의 robust 개선판.
"""
from __future__ import annotations

import numpy as np


# ── 거리 계산 유틸 ──────────────────────────────────────────────────────────────

def _pairwise_sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """행별 제곱 유클리드 거리 행렬 (|x_i - y_j|²). shape = (len(x), len(y))."""
    x_sq = (x ** 2).sum(1, keepdims=True)          # (Nx, 1)
    y_sq = (y ** 2).sum(1, keepdims=True).T         # (1, Ny)
    d = x_sq + y_sq - 2.0 * x @ y.T
    return np.maximum(d, 0.0)                        # 수치 오차로 음수가 되는 것 방지


# ── FID ─────────────────────────────────────────────────────────────────────────

def compute_fid_from_features(real_f: np.ndarray, gen_f: np.ndarray) -> float:
    """Fréchet distance: ||μ_r - μ_g||² + tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^½)."""
    from scipy.linalg import sqrtm

    mu_r, mu_g = real_f.mean(0), gen_f.mean(0)
    eps_eye_r = np.eye(real_f.shape[1]) * 1e-6
    eps_eye_g = np.eye(gen_f.shape[1]) * 1e-6
    s_r = np.cov(real_f, rowvar=False) + eps_eye_r
    s_g = np.cov(gen_f, rowvar=False) + eps_eye_g

    diff = mu_r - mu_g
    # scipy 버전에 따라 sqrtm 시그니처가 다르다(>=1.18 은 disp 인자 제거).
    try:
        covmean = sqrtm(s_r @ s_g, disp=False)
    except TypeError:
        covmean = sqrtm(s_r @ s_g)
    if isinstance(covmean, tuple):       # 구버전은 (행렬, errest) 튜플 반환
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return max(float(diff @ diff + np.trace(s_r + s_g - 2.0 * covmean)), 0.0)


# ── KID (다항 커널 MMD²) ────────────────────────────────────────────────────────

def _poly_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """KID 표준 다항 커널 k(x, y) = ((x·y)/d + 1)³, d = feature 차원."""
    d = x.shape[1]
    return (x @ y.T / d + 1.0) ** 3


def compute_kid_from_features(
    real_f: np.ndarray,
    gen_f: np.ndarray,
    subset_size: int = 100,
    num_subsets: int = 100,
    seed: int = 0,
) -> tuple[float, float]:
    """불편(unbiased) MMD² 추정량을 여러 부분표본에서 평균낸 KID.

    표본이 적을 때 FID보다 신뢰성이 높고, 부분표본 간 표준편차를 함께 주므로
    seed 통계 틀과 잘 맞는다. 반환: (kid_mean, kid_std).
    """
    rng = np.random.default_rng(seed)
    n = min(subset_size, len(real_f), len(gen_f))
    if n < 2:
        return float("nan"), float("nan")

    estimates: list[float] = []
    for _ in range(num_subsets):
        x = real_f[rng.choice(len(real_f), n, replace=False)]
        y = gen_f[rng.choice(len(gen_f), n, replace=False)]
        k_xx, k_yy, k_xy = _poly_kernel(x, x), _poly_kernel(y, y), _poly_kernel(x, y)
        # 대각(자기 자신) 제외한 unbiased 항
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)
        mmd2 = (k_xx.sum() / (n * (n - 1))
                + k_yy.sum() / (n * (n - 1))
                - 2.0 * k_xy.mean())
        estimates.append(float(mmd2))
    return float(np.mean(estimates)), float(np.std(estimates))


# ── Precision / Recall / Density / Coverage ─────────────────────────────────────

def _knn_radii(features: np.ndarray, nearest_k: int) -> np.ndarray:
    """각 점에서 k번째 최근접 이웃까지의 거리(=manifold 반경 추정)."""
    sq = _pairwise_sq_dist(features, features)
    dist = np.sqrt(sq)
    dist.sort(axis=1)               # 0번째 열은 자기 자신(거리 0)
    k = min(nearest_k, dist.shape[1] - 1)
    return dist[:, k]               # k번째 이웃까지 거리


def compute_prdc_from_features(
    real_f: np.ndarray,
    gen_f: np.ndarray,
    nearest_k: int = 5,
    max_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Precision/Recall/Density/Coverage (Naeem et al. 2020).

    직관:
      - precision : 생성 샘플이 '진짜 데이터 manifold' 안에 든 비율 (품질·사실성).
      - recall    : 진짜 샘플이 '생성 manifold' 안에 든 비율 (다양성·coverage).
      - density   : precision의 robust 버전 (이웃 수로 가중).
      - coverage  : 각 진짜 샘플의 이웃 안에 생성 샘플이 있는 비율.

    N×N 거리 행렬을 쓰므로 메모리 보호를 위해 max_samples로 잘라 쓴다.
    """
    rng = np.random.default_rng(seed)

    def _subsample(arr: np.ndarray) -> np.ndarray:
        if len(arr) <= max_samples:
            return arr
        return arr[rng.choice(len(arr), max_samples, replace=False)]

    real_f, gen_f = _subsample(real_f), _subsample(gen_f)

    real_radii = _knn_radii(real_f, nearest_k)      # (Nr,)
    gen_radii  = _knn_radii(gen_f, nearest_k)        # (Ng,)
    dist_rg = np.sqrt(_pairwise_sq_dist(real_f, gen_f))   # (Nr, Ng)

    # precision: 각 생성 샘플 j가 어떤 진짜 샘플 i의 반경 안에 들어오는가
    in_real_ball = dist_rg <= real_radii[:, None]    # (Nr, Ng)
    precision = float(in_real_ball.any(axis=0).mean())

    # recall: 각 진짜 샘플 i가 어떤 생성 샘플 j의 반경 안에 들어오는가
    in_gen_ball = dist_rg <= gen_radii[None, :]      # (Nr, Ng)
    recall = float(in_gen_ball.any(axis=1).mean())

    # density: 생성 샘플당 '포함된 진짜 반경 수'의 평균을 k로 정규화
    density = float((1.0 / nearest_k) * in_real_ball.sum(axis=0).mean())

    # coverage: 진짜 샘플 i의 반경 안에 생성 샘플이 하나라도 있는 비율
    coverage = float(in_real_ball.any(axis=1).mean())

    return {"precision": precision, "recall": recall,
            "density": density, "coverage": coverage}


# ── 한 번에 모든 feature 기반 지표 계산 ─────────────────────────────────────────

def compute_feature_metrics(
    real_f: np.ndarray,
    gen_f: np.ndarray,
    suffix: str = "_phi",
    nearest_k: int = 5,
    kid_subset_size: int = 100,
    kid_num_subsets: int = 100,
) -> dict[str, float]:
    """FID·KID·PRDC를 한꺼번에 계산해 dict로 반환한다.

    키 이름에 suffix를 붙여(기본 '_phi') 어떤 feature 공간에서 잰 값인지 표시한다.
    예: fid_phi, kid_phi, precision_phi, recall_phi, density_phi, coverage_phi.
    """
    kid_mean, kid_std = compute_kid_from_features(
        real_f, gen_f, kid_subset_size, kid_num_subsets
    )
    prdc = compute_prdc_from_features(real_f, gen_f, nearest_k)
    out = {
        f"fid{suffix}": compute_fid_from_features(real_f, gen_f),
        f"kid{suffix}": kid_mean,
        f"kid{suffix}_std": kid_std,
    }
    for k, v in prdc.items():
        out[f"{k}{suffix}"] = v
    return out
