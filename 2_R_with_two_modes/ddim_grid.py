"""Phase 2·3 공용 DDIM λ 격자 (sampling 시 어느 noise level들을 거칠지).

DDIM 생성은 순수 noise(λ 작음) → 깨끗한 이미지(λ 큼)로 여러 step에 나눠 내려온다.
이때 '어느 λ 값들에 step을 놓을지'가 sampling 품질을 좌우한다. 등간격(linspace)으로
넓은 구간을 훑으면 정작 신호가 형성되는 중앙(λ≈0) 부근이 성기게 풀려 손해다.

cosine 간격은 중앙(λ≈0)에 step을 집중시키고 극단(거의 순수 noise / 거의 클린)에는
적게 두어, 같은 step 수로 더 나은 discretization을 준다. 두 phase가 **이 함수 하나**를
공유하므로 sampler가 서로 어긋날 일이 없다(범위만 phase별 config로 지정).
"""
from __future__ import annotations

import numpy as np


def cosine_lambda_grid(n_steps: int, lambda_min: float, lambda_max: float) -> np.ndarray:
    """noisy(작은 λ) → clean(큰 λ) 방향으로 증가하는 길이 (n_steps+1) 의 λ 격자.

    표준 VP cosine schedule λ(t) = -2·log(tan(π t / 2)) 를 사용한다. t를 1→0으로
    균일하게 두면 λ는 -∞→+∞로 증가하며, 중앙(λ≈0, t≈0.5) 부근이 촘촘해진다.
    양 끝은 [lambda_min, lambda_max]로 clip 한다(거의 순수 noise / 거의 클린 상태).
    """
    t = np.linspace(1.0, 0.0, n_steps + 1)
    t = np.clip(t, 1e-5, 1.0 - 1e-5)
    lam = -2.0 * np.log(np.tan(0.5 * np.pi * t))
    return np.clip(lam, lambda_min, lambda_max)
