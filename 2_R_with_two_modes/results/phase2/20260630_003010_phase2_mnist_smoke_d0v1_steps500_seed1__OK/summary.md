# Phase 2 MNIST Pilot Summary

## Configuration

- digits: 0 vs 1
- λ range: [-10.0, 10.0]
- ρ threshold: 0.5
- empirical λ_R*: -0.9091
- transition region: [-2.7273, 0.9091]
- train steps per schedule: 500
- DDIM steps: 5
- n_generate (FID): 100

## φ Classifier Evaluation (test set)

DMSR_φ·FID-φ의 기반인 분류기 φ의 테스트셋 분류 성능 (자세한 표·confusion은 `classifier_report.md` / `classifier_report.json`, 그림은 plots 참고).
- accuracy: **100.00%** (n=2115), macro-F1: 1.0000
  digit 0: P=1.000/R=1.000/F1=1.000, digit 1: P=1.000/R=1.000/F1=1.000

## Schedules

- **cosine_vp**: VP cosine-β induced λ density (0-centered, heavy sech tails). 관행 baseline.
- **linear_vp**: VP linear-β (DDPM) induced λ density. 관행 baseline.
- **uniform**: Uniform(λ∈[-10.0,10.0]). 무정보 baseline.
- **hang_laplace_b0.5**: Laplace(0, b=0.5) — Hang et al. baseline.
- **dmsr_normal_s0.3**: N(λ_R*=-0.909, s=0.3).
- **dmsr_normal_s0.8**: N(λ_R*=-0.909, s=0.8).
- **dmsr_normal_s1.5**: N(λ_R*=-0.909, s=1.5).
- **dmsr_normal_s2.5**: N(λ_R*=-0.909, s=2.5).
- **dmsr_normal_s4.0**: N(λ_R*=-0.909, s=4.0).
- **dmsr_normal_s6.0**: N(λ_R*=-0.909, s=6.0).
- **dmsr_laplace_b0.5**: Laplace(λ_R*=-0.909, b=0.5).
- **dmsr_cosmix_w0.5**: (1-0.5)·cosine + 0.5·N(λ_R*=-0.909, s=1.0).
- **dmsr_cosmix_w0.8**: (1-0.8)·cosine + 0.8·N(λ_R*=-0.909, s=1.0).

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dmsr_normal_s1.5 | 20260526 | 0.7727 | 0.7937 | 102.06 | 0.9204 | 0.1900 | 0.2892 | 0.0319 |
| 2 | dmsr_normal_s2.5 | 20260526 | 0.5325 | 0.6484 | 121.41 | 0.9094 | 0.2500 | 0.1963 | 0.0330 |
| 3 | dmsr_normal_s4.0 | 20260526 | 0.3503 | 0.4879 | 125.64 | 0.8723 | 0.1300 | 0.1281 | 0.0374 |
| 4 | dmsr_cosmix_w0.8 | 20260526 | 0.8412 | 0.8161 | 125.99 | 0.8720 | 0.1500 | 0.2007 | 0.0328 |
| 5 | dmsr_cosmix_w0.5 | 20260526 | 0.7051 | 0.7369 | 127.80 | 0.8863 | 0.1500 | 0.1644 | 0.0336 |
| 6 | dmsr_laplace_b0.5 | 20260526 | 0.9735 | 0.9177 | 130.95 | 0.8991 | 0.2200 | 0.3529 | 0.0352 |
| 7 | dmsr_normal_s0.8 | 20260526 | 0.9771 | 0.8955 | 132.14 | 0.8901 | 0.2500 | 0.3843 | 0.0339 |
| 8 | linear_vp | 20260526 | 0.3462 | 0.4643 | 132.51 | 0.8964 | 0.2100 | 0.1686 | 0.0352 |
| 9 | dmsr_normal_s6.0 | 20260526 | 0.2378 | 0.3587 | 134.12 | 0.8635 | 0.0500 | 0.1094 | 0.0422 |
| 10 | dmsr_normal_s0.3 | 20260526 | 1.0000 | 0.9610 | 138.91 | 0.8842 | 0.2400 | 0.4358 | 0.0437 |
| 11 | hang_laplace_b0.5 | 20260526 | 0.9160 | 0.8732 | 142.34 | 0.8469 | 0.2000 | 0.3153 | 0.0317 |
| 12 | cosine_vp | 20260526 | 0.4779 | 0.6036 | 142.71 | 0.8805 | 0.2300 | 0.1452 | 0.0358 |
| 13 | uniform | 20260526 | 0.1817 | 0.2954 | 153.69 | 0.8391 | 0.0200 | 0.1067 | 0.0451 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | dmsr_normal_s1.5 | 1 | 102.06 | 239.2887 | 0.980 | 0.140 | 0.2892 |
| 2 | dmsr_normal_s2.5 | 1 | 121.41 | 249.7825 | 0.990 | 0.160 | 0.1963 |
| 3 | dmsr_normal_s4.0 | 1 | 125.64 | 315.6473 | 1.000 | 0.130 | 0.1281 |
| 4 | dmsr_cosmix_w0.8 | 1 | 125.99 | 282.2084 | 1.000 | 0.130 | 0.2007 |
| 5 | dmsr_cosmix_w0.5 | 1 | 127.80 | 296.9534 | 1.000 | 0.130 | 0.1644 |
| 6 | dmsr_laplace_b0.5 | 1 | 130.95 | 277.6494 | 1.000 | 0.130 | 0.3529 |
| 7 | dmsr_normal_s0.8 | 1 | 132.14 | 282.9311 | 1.000 | 0.130 | 0.3843 |
| 8 | linear_vp | 1 | 132.51 | 291.8234 | 0.980 | 0.130 | 0.1686 |
| 9 | dmsr_normal_s6.0 | 1 | 134.12 | 352.8821 | 0.990 | 0.110 | 0.1094 |
| 10 | dmsr_normal_s0.3 | 1 | 138.91 | 291.2191 | 1.000 | 0.120 | 0.4358 |
| 11 | hang_laplace_b0.5 | 1 | 142.34 | 311.5773 | 0.990 | 0.120 | 0.3153 |
| 12 | cosine_vp | 1 | 142.71 | 305.3527 | 1.000 | 0.130 | 0.1452 |
| 13 | uniform | 1 | 153.69 | 385.7207 | 0.990 | 0.090 | 0.1067 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| dmsr_cosmix_w0.5 | 1 | -14.911 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_cosmix_w0.8 | 1 | -16.718 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | -11.760 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.3 | 1 | -3.794 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.8 | 1 | -10.571 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | -40.645 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s2.5 | 1 | -21.301 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | -17.065 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s6.0 | 1 | -8.588 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| hang_laplace_b0.5 | 1 | -0.368 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -10.194 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | 10.979 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| hang_laplace_b0.5 | -0.0041 | 0.9683 | 0.6847 |
| dmsr_normal_s1.5 | -0.0038 | 0.9681 | 0.7108 |
| dmsr_cosmix_w0.8 | -0.0029 | 0.9672 | 0.7993 |
| dmsr_normal_s2.5 | -0.0027 | 0.9670 | 0.8037 |
| dmsr_cosmix_w0.5 | -0.0022 | 0.9664 | 0.8356 |
| dmsr_normal_s0.8 | -0.0018 | 0.9661 | 0.6157 |
| linear_vp | -0.0006 | 0.9648 | 0.8314 |
| dmsr_laplace_b0.5 | -0.0005 | 0.9648 | 0.6471 |
| cosine_vp | 0 (baseline) | 0.9642 | 0.8548 |
| dmsr_normal_s4.0 | +0.0016 | 0.9626 | 0.8719 |
| dmsr_normal_s6.0 | +0.0065 | 0.9578 | 0.8906 |
| dmsr_normal_s0.3 | +0.0080 | 0.9563 | 0.5642 |
| uniform | +0.0093 | 0.9549 | 0.8933 |

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
