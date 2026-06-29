# Phase 2 MNIST Pilot Summary

## Configuration

- digits: 3 vs 8
- λ range: [-10.0, 10.0]
- ρ threshold: 0.5
- empirical λ_R*: 0.7692
- transition region: [-1.2821, 1.7949]
- train steps per schedule: 20000
- DDIM steps: 50
- n_generate (FID): 5000

## φ Classifier Evaluation (test set)

DMSR_φ·FID-φ의 기반인 분류기 φ의 테스트셋 분류 성능 (자세한 표·confusion은 `classifier_report.md` / `classifier_report.json`, 그림은 plots 참고).
- accuracy: **99.90%** (n=1984), macro-F1: 0.9990
  digit 3: P=0.999/R=0.999/F1=0.999, digit 8: P=0.999/R=0.999/F1=0.999

## Schedules

- **cosine_vp**: VP cosine-β induced λ density (0-centered, heavy sech tails). 관행 baseline.
- **linear_vp**: VP linear-β (DDPM) induced λ density. 관행 baseline.
- **uniform**: Uniform(λ∈[-10.0,10.0]). 무정보 baseline.
- **hang_laplace_b0.5**: Laplace(0, b=0.5) — Hang et al. baseline.
- **dmsr_normal_s0.3**: N(λ_R*=0.769, s=0.3).
- **dmsr_normal_s0.8**: N(λ_R*=0.769, s=0.8).
- **dmsr_normal_s1.5**: N(λ_R*=0.769, s=1.5).
- **dmsr_normal_s2.5**: N(λ_R*=0.769, s=2.5).
- **dmsr_normal_s4.0**: N(λ_R*=0.769, s=4.0).
- **dmsr_normal_s6.0**: N(λ_R*=0.769, s=6.0).
- **dmsr_laplace_b0.5**: Laplace(λ_R*=0.769, b=0.5).
- **dmsr_cosmix_w0.5**: (1-0.5)·cosine + 0.5·N(λ_R*=0.769, s=1.0).
- **dmsr_cosmix_w0.8**: (1-0.8)·cosine + 0.8·N(λ_R*=0.769, s=1.0).

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dmsr_normal_s2.5 | 20260526 | 0.4525 | 0.4898 | 12.53 | 0.9451 | 0.0864 | 0.0695 | 0.0397 |
| 2 | linear_vp | 20260526 | 0.2670 | 0.3218 | 12.80 | 0.9478 | 0.0942 | 0.0705 | 0.0403 |
| 3 | dmsr_cosmix_w0.5 | 20260526 | 0.6362 | 0.6204 | 13.05 | 0.9447 | 0.0982 | 0.0674 | 0.0392 |
| 4 | dmsr_cosmix_w0.8 | 20260526 | 0.7496 | 0.7038 | 14.61 | 0.9410 | 0.0384 | 0.0747 | 0.0392 |
| 5 | hang_laplace_b0.5 | 20260526 | 0.9477 | 0.8344 | 14.67 | 0.9369 | 0.0024 | 0.2053 | 0.0388 |
| 6 | dmsr_normal_s4.0 | 20260526 | 0.2959 | 0.3468 | 15.12 | 0.9372 | 0.0432 | 0.0658 | 0.0413 |
| 7 | dmsr_normal_s1.5 | 20260526 | 0.6666 | 0.6522 | 16.16 | 0.9360 | 0.0082 | 0.1438 | 0.0388 |
| 8 | cosine_vp | 20260526 | 0.4461 | 0.4801 | 16.87 | 0.9426 | 0.1492 | 0.0682 | 0.0401 |
| 9 | dmsr_normal_s6.0 | 20260526 | 0.2009 | 0.2464 | 18.09 | 0.9305 | 0.0586 | 0.0636 | 0.0425 |
| 10 | uniform | 20260526 | 0.1545 | 0.1997 | 20.38 | 0.9317 | 0.0752 | 0.0644 | 0.0428 |
| 11 | dmsr_laplace_b0.5 | 20260526 | 0.9277 | 0.8517 | 25.18 | 0.9249 | 0.0386 | 0.1800 | 0.0390 |
| 12 | dmsr_normal_s0.3 | 20260526 | 0.9997 | 0.9318 | 38.02 | 0.9196 | 0.1400 | 0.3536 | 0.0404 |
| 13 | dmsr_normal_s0.8 | 20260526 | 0.8955 | 0.8083 | 43.90 | 0.9451 | 0.3460 | 0.2502 | 0.0386 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | dmsr_normal_s2.5 | 1 | 12.53 | 7.2101 | 0.908 | 0.638 | 0.0695 |
| 2 | linear_vp | 1 | 12.80 | 7.2453 | 0.924 | 0.658 | 0.0705 |
| 3 | dmsr_cosmix_w0.5 | 1 | 13.05 | 6.7575 | 0.897 | 0.588 | 0.0674 |
| 4 | dmsr_cosmix_w0.8 | 1 | 14.61 | 9.0638 | 0.905 | 0.545 | 0.0747 |
| 5 | hang_laplace_b0.5 | 1 | 14.67 | 8.9924 | 0.831 | 0.354 | 0.2053 |
| 6 | dmsr_normal_s4.0 | 1 | 15.12 | 9.1155 | 0.922 | 0.567 | 0.0658 |
| 7 | dmsr_normal_s1.5 | 1 | 16.16 | 8.7668 | 0.728 | 0.204 | 0.1438 |
| 8 | cosine_vp | 1 | 16.87 | 7.9365 | 0.925 | 0.584 | 0.0682 |
| 9 | dmsr_normal_s6.0 | 1 | 18.09 | 10.4415 | 0.920 | 0.508 | 0.0636 |
| 10 | uniform | 1 | 20.38 | 11.3824 | 0.920 | 0.496 | 0.0644 |
| 11 | dmsr_laplace_b0.5 | 1 | 25.18 | 12.5654 | 0.552 | 0.059 | 0.1800 |
| 12 | dmsr_normal_s0.3 | 1 | 38.02 | 17.4801 | 0.415 | 0.026 | 0.3536 |
| 13 | dmsr_normal_s0.8 | 1 | 43.90 | 8.9131 | 0.453 | 0.021 | 0.2502 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| dmsr_cosmix_w0.5 | 1 | -3.820 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_cosmix_w0.8 | 1 | -2.256 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 8.316 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.3 | 1 | 21.149 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.8 | 1 | 27.037 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | -0.707 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s2.5 | 1 | -4.340 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | -1.748 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s6.0 | 1 | 1.218 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| hang_laplace_b0.5 | 1 | -2.202 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -4.072 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | 3.508 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| dmsr_normal_s0.8 | -0.0015 | 0.9614 | 0.7498 |
| hang_laplace_b0.5 | -0.0013 | 0.9612 | 0.7947 |
| dmsr_normal_s1.5 | -0.0012 | 0.9612 | 0.8562 |
| dmsr_laplace_b0.5 | -0.0011 | 0.9610 | 0.8200 |
| dmsr_cosmix_w0.8 | -0.0009 | 0.9608 | 0.9253 |
| dmsr_cosmix_w0.5 | -0.0008 | 0.9608 | 0.9326 |
| dmsr_normal_s2.5 | -0.0004 | 0.9603 | 0.9305 |
| cosine_vp | 0 (baseline) | 0.9599 | 0.9318 |
| linear_vp | +0.0002 | 0.9597 | 0.9295 |
| dmsr_normal_s0.3 | +0.0003 | 0.9596 | 0.6464 |
| dmsr_normal_s4.0 | +0.0013 | 0.9587 | 0.9342 |
| dmsr_normal_s6.0 | +0.0024 | 0.9575 | 0.9364 |
| uniform | +0.0028 | 0.9572 | 0.9356 |

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
