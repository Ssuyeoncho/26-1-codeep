# Phase 2 MNIST Pilot Summary

## Configuration

- digits: 4 vs 9
- λ range: [-10.0, 10.0]
- ρ threshold: 0.5
- empirical λ_R*: 0.7692
- transition region: [-1.2821, 1.2821]
- train steps per schedule: 20000
- DDIM steps: 50
- n_generate (FID): 5000

## φ Classifier Evaluation (test set)

DMSR_φ·FID-φ의 기반인 분류기 φ의 테스트셋 분류 성능 (자세한 표·confusion은 `classifier_report.md` / `classifier_report.json`, 그림은 plots 참고).
- accuracy: **99.50%** (n=1991), macro-F1: 0.9950
  digit 4: P=0.994/R=0.996/F1=0.995, digit 9: P=0.996/R=0.994/F1=0.995

## Schedules

- **cosine_vp**: VP cosine-β induced λ density (0-centered, heavy sech tails). 관행 baseline.
- **linear_vp**: VP linear-β (DDPM) induced λ density. 관행 baseline.
- **uniform**: Uniform(λ∈[-10.0,10.0]). 무정보 baseline.
- **dmsr_normal_s0.5**: Normal(center=λ_R*=0.769, s=0.5).
- **dmsr_normal_s1.5**: Normal(center=λ_R*=0.769, s=1.5).
- **dmsr_normal_s4.0**: Normal(center=λ_R*=0.769, s=4.0).
- **dmsr_laplace_b0.5**: Laplace(center=λ_R*=0.769, b=0.5).
- **dmsr_laplace_b1.5**: Laplace(center=λ_R*=0.769, b=1.5).
- **dmsr_laplace_b4.0**: Laplace(center=λ_R*=0.769, b=4.0).
- **at0_normal_s0.5**: Normal(center=0, s=0.5).
- **at0_normal_s1.5**: Normal(center=0, s=1.5).
- **at0_normal_s4.0**: Normal(center=0, s=4.0).
- **at0_laplace_b0.5**: Laplace(center=0, b=0.5).  [Hang et al. baseline]
- **at0_laplace_b1.5**: Laplace(center=0, b=1.5).  [Hang et al. baseline]
- **at0_laplace_b4.0**: Laplace(center=0, b=4.0).  [Hang et al. baseline]

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | linear_vp | 20260526 | 0.2345 | 0.2914 | 1.60 | 0.9795 | 0.0164 | 0.0596 | 0.0302 |
| 2 | at0_laplace_b1.5 | 20260526 | 0.5739 | 0.5648 | 1.67 | 0.9733 | 0.0192 | 0.0622 | 0.0294 |
| 3 | dmsr_normal_s4.0 | 20260526 | 0.2482 | 0.3074 | 2.63 | 0.9698 | 0.0306 | 0.0552 | 0.0304 |
| 4 | at0_normal_s1.5 | 20260526 | 0.6063 | 0.6085 | 2.74 | 0.9662 | 0.0342 | 0.1588 | 0.0290 |
| 5 | dmsr_laplace_b1.5 | 20260526 | 0.5178 | 0.5573 | 2.93 | 0.9654 | 0.0358 | 0.0655 | 0.0300 |
| 6 | at0_normal_s4.0 | 20260526 | 0.2511 | 0.3124 | 3.39 | 0.9676 | 0.0402 | 0.0560 | 0.0304 |
| 7 | uniform | 20260526 | 0.1278 | 0.1796 | 3.78 | 0.9647 | 0.0190 | 0.0548 | 0.0321 |
| 8 | cosine_vp | 20260526 | 0.3811 | 0.4282 | 3.78 | 0.9670 | 0.0386 | 0.0581 | 0.0299 |
| 9 | dmsr_laplace_b4.0 | 20260526 | 0.2608 | 0.3184 | 5.09 | 0.9597 | 0.0634 | 0.0551 | 0.0310 |
| 10 | at0_laplace_b4.0 | 20260526 | 0.2747 | 0.3254 | 5.14 | 0.9607 | 0.0528 | 0.0550 | 0.0309 |
| 11 | at0_laplace_b0.5 | 20260526 | 0.9232 | 0.7718 | 7.23 | 0.9502 | 0.1004 | 0.1980 | 0.0288 |
| 12 | dmsr_normal_s1.5 | 20260526 | 0.5488 | 0.5872 | 11.82 | 0.9414 | 0.1290 | 0.1402 | 0.0292 |
| 13 | dmsr_laplace_b0.5 | 20260526 | 0.8114 | 0.8114 | 22.00 | 0.9326 | 0.1848 | 0.1786 | 0.0292 |
| 14 | dmsr_normal_s0.5 | 20260526 | 0.8472 | 0.8470 | 30.98 | 0.9428 | 0.2968 | 0.3221 | 0.0296 |
| 15 | at0_normal_s0.5 | 20260526 | 0.9897 | 0.8009 | 57.21 | 0.9641 | 0.3826 | 0.3019 | 0.0290 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | linear_vp | 1 | 1.60 | 0.6384 | 0.860 | 0.815 | 0.0596 |
| 2 | at0_laplace_b1.5 | 1 | 1.67 | 0.5631 | 0.809 | 0.712 | 0.0622 |
| 3 | dmsr_normal_s4.0 | 1 | 2.63 | 0.8628 | 0.830 | 0.722 | 0.0552 |
| 4 | at0_normal_s1.5 | 1 | 2.74 | 0.7353 | 0.778 | 0.609 | 0.1588 |
| 5 | dmsr_laplace_b1.5 | 1 | 2.93 | 0.9011 | 0.792 | 0.623 | 0.0655 |
| 6 | at0_normal_s4.0 | 1 | 3.39 | 1.0441 | 0.818 | 0.708 | 0.0560 |
| 7 | uniform | 1 | 3.78 | 1.2697 | 0.801 | 0.650 | 0.0548 |
| 8 | cosine_vp | 1 | 3.78 | 1.1945 | 0.817 | 0.694 | 0.0581 |
| 9 | dmsr_laplace_b4.0 | 1 | 5.09 | 1.4810 | 0.768 | 0.603 | 0.0551 |
| 10 | at0_laplace_b4.0 | 1 | 5.14 | 1.4886 | 0.799 | 0.621 | 0.0550 |
| 11 | at0_laplace_b0.5 | 1 | 7.23 | 1.6783 | 0.699 | 0.356 | 0.1980 |
| 12 | dmsr_normal_s1.5 | 1 | 11.82 | 2.2862 | 0.628 | 0.181 | 0.1402 |
| 13 | dmsr_laplace_b0.5 | 1 | 22.00 | 3.7249 | 0.354 | 0.036 | 0.1786 |
| 14 | dmsr_normal_s0.5 | 1 | 30.98 | 5.0729 | 0.213 | 0.027 | 0.3221 |
| 15 | at0_normal_s0.5 | 1 | 57.21 | 17.7760 | 0.092 | 0.008 | 0.3019 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| at0_laplace_b0.5 | 1 | 3.447 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | -2.112 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | 1.360 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 53.427 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | -1.042 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | -0.391 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 18.223 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | -0.847 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | 1.310 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 27.201 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 8.039 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | -1.146 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -2.175 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -0.002 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| at0_laplace_b0.5 | -0.0011 | 0.9712 | 0.8020 |
| at0_normal_s0.5 | -0.0009 | 0.9710 | 0.6981 |
| at0_normal_s1.5 | -0.0009 | 0.9710 | 0.8412 |
| dmsr_laplace_b0.5 | -0.0007 | 0.9708 | 0.8214 |
| dmsr_normal_s1.5 | -0.0007 | 0.9708 | 0.8598 |
| at0_laplace_b1.5 | -0.0005 | 0.9706 | 0.9378 |
| dmsr_normal_s0.5 | -0.0003 | 0.9704 | 0.6779 |
| cosine_vp | 0 (baseline) | 0.9701 | 0.9419 |
| dmsr_laplace_b1.5 | +0.0001 | 0.9700 | 0.9345 |
| linear_vp | +0.0003 | 0.9698 | 0.9404 |
| at0_normal_s4.0 | +0.0005 | 0.9696 | 0.9440 |
| dmsr_normal_s4.0 | +0.0005 | 0.9696 | 0.9448 |
| at0_laplace_b4.0 | +0.0010 | 0.9691 | 0.9450 |
| dmsr_laplace_b4.0 | +0.0011 | 0.9690 | 0.9449 |
| uniform | +0.0022 | 0.9679 | 0.9452 |

## Interpretation Guide

- 이 단계의 목적은 생성 성능 주장이 아니라 **파이프라인·통계 틀 검증**이다.
  MNIST는 쉬운 데이터라 schedule 간 FID 차이가 작아도 실패가 아니다.
- 생성 품질은 φ-feature space에서 FID·KID·Precision/Recall/Density/Coverage로 잰다.
  FID는 품질·다양성을 뭉뚱그리므로, Precision(품질)과 Recall/Coverage(다양성)를 함께 보면 mode collapse 같은 실패를 분리해 볼 수 있다(Phase 3와 동일한 지표).
- KID는 표본이 적을 때 FID보다 신뢰성이 높고 부분표본 분산을 함께 준다.
- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며, full-range support 와의 균형이 중요하다(Phase 1 결론).
- `*_normal_s*`/`*_laplace_b*`는 폭을 좁게→넓게 sweep한다. narrow는 clean끝을 못 배워 붕괴하기 쉽고, 넓힐수록 회복하는지를 본다. 같은 폭에서 dmsr(중심 λ_R*)과 at0(중심 0)을 비교해 중심 위치 효과를 본다. Precision–Recall 그림으로 붕괴(좌하단) 여부를 확인하라.
- λ_R*는 DMSR_φ(λ)의 수치 미분 peak에서 경험적으로 추정한다. 이 값은 Phase 3로 넘어가지 않으며 CIFAR에서 독립적으로 재추정한다.
- 유의성 검정은 seed가 부족하면(현재 1개) 건너뛴다. Phase 3에서 `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.
