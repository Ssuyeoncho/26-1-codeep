# Phase 2 MNIST Pilot Summary

## Configuration

- digits: 1 vs 8
- λ range: [-10.0, 10.0]
- ρ threshold: 0.5
- empirical λ_R*: 0.2564
- transition region: [-1.2821, 2.3077]
- train steps per schedule: 20000
- DDIM steps: 50
- n_generate (FID): 5000

## φ Classifier Evaluation (test set)

DMSR_φ·FID-φ의 기반인 분류기 φ의 테스트셋 분류 성능 (자세한 표·confusion은 `classifier_report.md` / `classifier_report.json`, 그림은 plots 참고).
- accuracy: **99.91%** (n=2109), macro-F1: 0.9990
  digit 1: P=1.000/R=0.998/F1=0.999, digit 8: P=0.998/R=1.000/F1=0.999

## Schedules

- **cosine_vp**: VP cosine-β induced λ density (0-centered, heavy sech tails). 관행 baseline.
- **linear_vp**: VP linear-β (DDPM) induced λ density. 관행 baseline.
- **uniform**: Uniform(λ∈[-10.0,10.0]). 무정보 baseline.
- **dmsr_normal_s0.5**: Normal(center=λ_R*=0.256, s=0.5).
- **dmsr_normal_s1.5**: Normal(center=λ_R*=0.256, s=1.5).
- **dmsr_normal_s4.0**: Normal(center=λ_R*=0.256, s=4.0).
- **dmsr_laplace_b0.5**: Laplace(center=λ_R*=0.256, b=0.5).
- **dmsr_laplace_b1.5**: Laplace(center=λ_R*=0.256, b=1.5).
- **dmsr_laplace_b4.0**: Laplace(center=λ_R*=0.256, b=4.0).
- **at0_normal_s0.5**: Normal(center=0, s=0.5).
- **at0_normal_s1.5**: Normal(center=0, s=1.5).
- **at0_normal_s4.0**: Normal(center=0, s=4.0).
- **at0_laplace_b0.5**: Laplace(center=0, b=0.5).  [Hang et al. baseline]
- **at0_laplace_b1.5**: Laplace(center=0, b=1.5).  [Hang et al. baseline]
- **at0_laplace_b4.0**: Laplace(center=0, b=4.0).  [Hang et al. baseline]

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | at0_laplace_b1.5 | 20260526 | 0.6798 | 0.6871 | 15.51 | 0.9717 | 0.1408 | 0.0552 | 0.0308 |
| 2 | dmsr_laplace_b4.0 | 20260526 | 0.3599 | 0.4033 | 18.77 | 0.9691 | 0.1588 | 0.0487 | 0.0321 |
| 3 | linear_vp | 20260526 | 0.2943 | 0.3526 | 19.90 | 0.9686 | 0.1656 | 0.0537 | 0.0316 |
| 4 | at0_laplace_b4.0 | 20260526 | 0.3557 | 0.4009 | 26.86 | 0.9646 | 0.1978 | 0.0485 | 0.0322 |
| 5 | dmsr_normal_s4.0 | 20260526 | 0.3439 | 0.4054 | 27.48 | 0.9619 | 0.2000 | 0.0488 | 0.0315 |
| 6 | dmsr_laplace_b1.5 | 20260526 | 0.6925 | 0.6951 | 29.00 | 0.9682 | 0.2156 | 0.0600 | 0.0308 |
| 7 | uniform | 20260526 | 0.1797 | 0.2333 | 32.38 | 0.9609 | 0.2272 | 0.0483 | 0.0331 |
| 8 | cosine_vp | 20260526 | 0.4955 | 0.5367 | 42.14 | 0.9636 | 0.2760 | 0.0506 | 0.0311 |
| 9 | at0_normal_s4.0 | 20260526 | 0.3414 | 0.4039 | 43.14 | 0.9686 | 0.2836 | 0.0498 | 0.0314 |
| 10 | dmsr_laplace_b0.5 | 20260526 | 0.9686 | 0.9252 | 46.06 | 0.9520 | 0.2502 | 0.1800 | 0.0309 |
| 11 | at0_laplace_b0.5 | 20260526 | 0.9565 | 0.9087 | 47.63 | 0.9588 | 0.2970 | 0.1877 | 0.0311 |
| 12 | dmsr_normal_s1.5 | 20260526 | 0.7600 | 0.7456 | 85.38 | 0.9717 | 0.4010 | 0.1389 | 0.0301 |
| 13 | at0_normal_s1.5 | 20260526 | 0.7403 | 0.7362 | 91.27 | 0.9747 | 0.4074 | 0.1477 | 0.0301 |
| 14 | dmsr_normal_s0.5 | 20260526 | 0.9988 | 0.9544 | 127.59 | 0.9424 | 0.3490 | 0.3036 | 0.0314 |
| 15 | at0_normal_s0.5 | 20260526 | 0.9949 | 0.9385 | 130.51 | 0.9058 | 0.1006 | 0.3039 | 0.0326 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | at0_laplace_b1.5 | 1 | 15.51 | 23.6946 | 0.826 | 0.562 | 0.0552 |
| 2 | dmsr_laplace_b4.0 | 1 | 18.77 | 22.4074 | 0.799 | 0.443 | 0.0487 |
| 3 | linear_vp | 1 | 19.90 | 21.9942 | 0.890 | 0.572 | 0.0537 |
| 4 | at0_laplace_b4.0 | 1 | 26.86 | 25.0867 | 0.813 | 0.419 | 0.0485 |
| 5 | dmsr_normal_s4.0 | 1 | 27.48 | 42.3565 | 0.856 | 0.408 | 0.0488 |
| 6 | dmsr_laplace_b1.5 | 1 | 29.00 | 28.9054 | 0.839 | 0.444 | 0.0600 |
| 7 | uniform | 1 | 32.38 | 33.3774 | 0.837 | 0.349 | 0.0483 |
| 8 | cosine_vp | 1 | 42.14 | 37.7789 | 0.886 | 0.441 | 0.0506 |
| 9 | at0_normal_s4.0 | 1 | 43.14 | 33.5890 | 0.870 | 0.395 | 0.0498 |
| 10 | dmsr_laplace_b0.5 | 1 | 46.06 | 106.8195 | 0.717 | 0.137 | 0.1800 |
| 11 | at0_laplace_b0.5 | 1 | 47.63 | 69.9885 | 0.740 | 0.194 | 0.1877 |
| 12 | dmsr_normal_s1.5 | 1 | 85.38 | 56.6655 | 0.764 | 0.252 | 0.1389 |
| 13 | at0_normal_s1.5 | 1 | 91.27 | 55.8317 | 0.804 | 0.320 | 0.1477 |
| 14 | dmsr_normal_s0.5 | 1 | 127.59 | 334.9559 | 0.013 | 0.002 | 0.3036 |
| 15 | at0_normal_s0.5 | 1 | 130.51 | 322.6412 | 0.182 | 0.009 | 0.3039 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| at0_laplace_b0.5 | 1 | 5.488 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | -26.635 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | -15.279 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 88.368 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 49.129 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | 1.002 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 3.917 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | -13.141 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | -23.368 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 85.446 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 43.242 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | -14.657 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -22.242 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -9.759 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| dmsr_normal_s1.5 | -0.0011 | 0.9699 | 0.8611 |
| at0_normal_s1.5 | -0.0010 | 0.9699 | 0.8523 |
| dmsr_laplace_b1.5 | -0.0004 | 0.9692 | 0.9400 |
| at0_laplace_b1.5 | -0.0003 | 0.9692 | 0.9448 |
| dmsr_laplace_b0.5 | -0.0002 | 0.9691 | 0.8200 |
| at0_laplace_b0.5 | -0.0001 | 0.9689 | 0.8123 |
| cosine_vp | 0 (baseline) | 0.9689 | 0.9494 |
| dmsr_normal_s0.5 | +0.0003 | 0.9686 | 0.6964 |
| at0_normal_s4.0 | +0.0003 | 0.9686 | 0.9502 |
| dmsr_normal_s4.0 | +0.0003 | 0.9685 | 0.9512 |
| linear_vp | +0.0004 | 0.9684 | 0.9463 |
| dmsr_laplace_b4.0 | +0.0010 | 0.9679 | 0.9513 |
| at0_laplace_b4.0 | +0.0010 | 0.9678 | 0.9515 |
| at0_normal_s0.5 | +0.0015 | 0.9674 | 0.6961 |
| uniform | +0.0019 | 0.9669 | 0.9517 |

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
