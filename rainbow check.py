# 새 schedule을 만들고 학습하기 전에 sanity check - 추가한 schedule이 정상인지 확인하는 과정

import torch, numpy as np
from vp_ddpm_toy import SCHEDULES, VPDiffusion, compute_p_lambda

for name, fn in SCHEDULES.items():
    betas = fn(1000)
    d = VPDiffusion(betas, device='cpu')
    
    # 체크 1: β가 (0, 1) 범위 안에 있나
    ok_range = (betas > 0).all() and (betas < 1).all()
    # 체크 2: ᾱ가 단조감소하나 (forward process가 정상)
    monotone = (d.alpha_bars[1:] <= d.alpha_bars[:-1]).all()
    # 체크 3: ᾱ_0 ≈ 1, ᾱ_T ≈ 0 (신호 있음 → 노이즈)
    endpoints = (d.alpha_bars[0] > 0.99) and (d.alpha_bars[-1] < 0.01)
    # 체크 4: p(λ) 봉우리 위치
    centers, p = compute_p_lambda(d.log_snr)
    peak = centers[np.argmax(p)]
    
    print(f"{name:15s}  range={ok_range}  monotone={monotone}  endpoints={endpoints}  peak λ={peak:+.2f}")