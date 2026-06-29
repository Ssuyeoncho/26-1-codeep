# Phase 2 MNIST Pilot Summary

## Configuration

- digits: 0 vs 1
- λ range: [-10.0, 10.0]
- ρ threshold: 0.5
- empirical λ_R*: -0.2564
- transition region: [-2.3077, 1.2821]
- train steps per schedule: 20000
- DDIM steps: 50
- n_generate (FID): 5000

## φ Classifier Evaluation (test set)

DMSR_φ·FID-φ의 기반인 분류기 φ의 테스트셋 분류 성능 (자세한 표·confusion은 `classifier_report.md` / `classifier_report.json`, 그림은 plots 참고).
- accuracy: **100.00%** (n=2115), macro-F1: 1.0000
  digit 0: P=1.000/R=1.000/F1=1.000, digit 1: P=1.000/R=1.000/F1=1.000

## Schedules

- **cosine_vp**: VP cosine-β induced λ density (0-centered, heavy sech tails). 관행 baseline.
- **linear_vp**: VP linear-β (DDPM) induced λ density. 관행 baseline.
- **uniform**: Uniform(λ∈[-10.0,10.0]). 무정보 baseline.
- **hang_laplace_b0.5**: Laplace(0, b=0.5) — Hang et al. baseline.
- **dmsr_normal_s0.3**: N(λ_R*=-0.256, s=0.3).
- **dmsr_normal_s0.8**: N(λ_R*=-0.256, s=0.8).
- **dmsr_normal_s1.5**: N(λ_R*=-0.256, s=1.5).
- **dmsr_normal_s2.5**: N(λ_R*=-0.256, s=2.5).
- **dmsr_normal_s4.0**: N(λ_R*=-0.256, s=4.0).
- **dmsr_normal_s6.0**: N(λ_R*=-0.256, s=6.0).
- **dmsr_laplace_b0.5**: Laplace(λ_R*=-0.256, b=0.5).
- **dmsr_cosmix_w0.5**: (1-0.5)·cosine + 0.5·N(λ_R*=-0.256, s=1.0).
- **dmsr_cosmix_w0.8**: (1-0.8)·cosine + 0.8·N(λ_R*=-0.256, s=1.0).

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | linear_vp | 20260526 | 0.3344 | 0.3855 | 27.12 | 0.9689 | 0.0864 | 0.0551 | 0.0222 |
| 2 | dmsr_cosmix_w0.5 | 20260526 | 0.7074 | 0.6570 | 41.52 | 0.9549 | 0.0728 | 0.0521 | 0.0217 |
| 3 | dmsr_cosmix_w0.8 | 20260526 | 0.8323 | 0.7402 | 44.18 | 0.9522 | 0.0514 | 0.0580 | 0.0215 |
| 4 | dmsr_normal_s6.0 | 20260526 | 0.2343 | 0.2892 | 44.41 | 0.9590 | 0.1486 | 0.0482 | 0.0232 |
| 5 | uniform | 20260526 | 0.1798 | 0.2354 | 50.17 | 0.9599 | 0.1720 | 0.0484 | 0.0234 |
| 6 | cosine_vp | 20260526 | 0.4972 | 0.5176 | 53.35 | 0.9699 | 0.2044 | 0.0548 | 0.0218 |
| 7 | dmsr_normal_s2.5 | 20260526 | 0.5246 | 0.5500 | 55.23 | 0.9740 | 0.2154 | 0.0601 | 0.0215 |
| 8 | dmsr_normal_s4.0 | 20260526 | 0.3442 | 0.4007 | 93.28 | 0.9744 | 0.2850 | 0.0495 | 0.0225 |
| 9 | dmsr_normal_s1.5 | 20260526 | 0.7617 | 0.7021 | 108.52 | 0.9710 | 0.3104 | 0.1555 | 0.0208 |
| 10 | dmsr_laplace_b0.5 | 20260526 | 0.9687 | 0.8772 | 108.89 | 0.9531 | 0.2808 | 0.2057 | 0.0208 |
| 11 | hang_laplace_b0.5 | 20260526 | 0.9569 | 0.8606 | 110.03 | 0.9326 | 0.1816 | 0.1864 | 0.0210 |
| 12 | dmsr_normal_s0.8 | 20260526 | 0.9684 | 0.8382 | 148.10 | 0.9084 | 0.0204 | 0.2555 | 0.0208 |
| 13 | dmsr_normal_s0.3 | 20260526 | 1.0000 | 0.9489 | 483.98 | 0.9986 | 0.4992 | 0.3400 | 0.0223 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | linear_vp | 1 | 27.12 | 97.2583 | 0.856 | 0.553 | 0.0551 |
| 2 | dmsr_cosmix_w0.5 | 1 | 41.52 | 166.5486 | 0.803 | 0.373 | 0.0521 |
| 3 | dmsr_cosmix_w0.8 | 1 | 44.18 | 185.6127 | 0.802 | 0.352 | 0.0580 |
| 4 | dmsr_normal_s6.0 | 1 | 44.41 | 118.2068 | 0.809 | 0.347 | 0.0482 |
| 5 | uniform | 1 | 50.17 | 103.3865 | 0.812 | 0.354 | 0.0484 |
| 6 | cosine_vp | 1 | 53.35 | 53.6571 | 0.845 | 0.483 | 0.0548 |
| 7 | dmsr_normal_s2.5 | 1 | 55.23 | 46.3637 | 0.848 | 0.519 | 0.0601 |
| 8 | dmsr_normal_s4.0 | 1 | 93.28 | 62.7489 | 0.848 | 0.430 | 0.0495 |
| 9 | dmsr_normal_s1.5 | 1 | 108.52 | 73.1276 | 0.817 | 0.386 | 0.1555 |
| 10 | dmsr_laplace_b0.5 | 1 | 108.89 | 130.7850 | 0.763 | 0.211 | 0.2057 |
| 11 | hang_laplace_b0.5 | 1 | 110.03 | 232.8599 | 0.610 | 0.108 | 0.1864 |
| 12 | dmsr_normal_s0.8 | 1 | 148.10 | 431.8026 | 0.274 | 0.015 | 0.2555 |
| 13 | dmsr_normal_s0.3 | 1 | 483.98 | 2209.0855 | 0.000 | 0.000 | 0.3400 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| dmsr_cosmix_w0.5 | 1 | -11.829 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_cosmix_w0.8 | 1 | -9.166 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 55.541 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.3 | 1 | 430.633 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.8 | 1 | 94.754 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 55.173 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s2.5 | 1 | 1.880 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | 39.932 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s6.0 | 1 | -8.932 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| hang_laplace_b0.5 | 1 | 56.685 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -26.228 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -3.176 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| dmsr_normal_s0.8 | -0.0010 | 0.9792 | 0.7445 |
| dmsr_laplace_b0.5 | -0.0010 | 0.9792 | 0.7943 |
| dmsr_normal_s1.5 | -0.0009 | 0.9792 | 0.8445 |
| hang_laplace_b0.5 | -0.0008 | 0.9790 | 0.8136 |
| dmsr_cosmix_w0.8 | -0.0003 | 0.9785 | 0.9420 |
| dmsr_normal_s2.5 | -0.0003 | 0.9785 | 0.9399 |
| dmsr_cosmix_w0.5 | -0.0001 | 0.9783 | 0.9479 |
| cosine_vp | 0 (baseline) | 0.9782 | 0.9452 |
| linear_vp | +0.0004 | 0.9778 | 0.9449 |
| dmsr_normal_s0.3 | +0.0006 | 0.9777 | 0.6600 |
| dmsr_normal_s4.0 | +0.0007 | 0.9775 | 0.9505 |
| dmsr_normal_s6.0 | +0.0014 | 0.9768 | 0.9518 |
| uniform | +0.0016 | 0.9766 | 0.9516 |

## Interpretation Guide

- 이 단계의 목적은 생성 성능 주장이 아니라 **파이프라인·통계 틀 검증**이다.
  MNIST는 쉬운 데이터라 schedule 간 FID 차이가 작아도 실패가 아니다.
- 생성 품질은 φ-feature space에서 FID·KID·Precision/Recall/Density/Coverage로 잰다.
  FID는 품질·다양성을 뭉뚱그리므로, Precision(품질)과 Recall/Coverage(다양성)를 함께 보면 mode collapse 같은 실패를 분리해 볼 수 있다(Phase 3와 동일한 지표).
- KID는 표본이 적을 때 FID보다 신뢰성이 높고 부분표본 분산을 함께 준다.
- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며, full-range support 와의 균형이 중요하다(Phase 1 결론).
- `dmsr_normal_s*`는 폭을 좁게→넓게 sweep한다. narrow는 clean끝을 못 배워 붕괴하기 쉽고, 넓힐수록(또는 `dmsr_cosmix_w*`처럼 cosine을 섞어 full-range support를 확보할수록) 회복하는지를 본다. Precision–Recall 그림으로 붕괴(좌하단) 여부를 확인하라.
- λ_R*는 DMSR_φ(λ)의 수치 미분 peak에서 경험적으로 추정한다. 이 값은 Phase 3로 넘어가지 않으며 CIFAR에서 독립적으로 재추정한다.
- 유의성 검정은 seed가 부족하면(현재 1개) 건너뛴다. Phase 3에서 `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.
