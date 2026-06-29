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
- **hang_laplace_b0.5**: Laplace(0, b=0.5) — Hang et al. baseline.
- **dmsr_normal_s0.3**: N(λ_R*=0.256, s=0.3).
- **dmsr_normal_s0.8**: N(λ_R*=0.256, s=0.8).
- **dmsr_normal_s1.5**: N(λ_R*=0.256, s=1.5).
- **dmsr_normal_s2.5**: N(λ_R*=0.256, s=2.5).
- **dmsr_normal_s4.0**: N(λ_R*=0.256, s=4.0).
- **dmsr_normal_s6.0**: N(λ_R*=0.256, s=6.0).
- **dmsr_laplace_b0.5**: Laplace(λ_R*=0.256, b=0.5).
- **dmsr_cosmix_w0.5**: (1-0.5)·cosine + 0.5·N(λ_R*=0.256, s=1.0).
- **dmsr_cosmix_w0.8**: (1-0.8)·cosine + 0.8·N(λ_R*=0.256, s=1.0).

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | linear_vp | 20260526 | 0.2935 | 0.3530 | 13.15 | 0.9705 | 0.0796 | 0.0527 | 0.0316 |
| 2 | dmsr_cosmix_w0.5 | 20260526 | 0.7073 | 0.6988 | 16.01 | 0.9725 | 0.1274 | 0.0522 | 0.0309 |
| 3 | dmsr_cosmix_w0.8 | 20260526 | 0.8327 | 0.7945 | 22.60 | 0.9738 | 0.1918 | 0.0573 | 0.0307 |
| 4 | dmsr_normal_s2.5 | 20260526 | 0.5242 | 0.5682 | 27.84 | 0.9726 | 0.2202 | 0.0582 | 0.0308 |
| 5 | cosine_vp | 20260526 | 0.4972 | 0.5386 | 30.65 | 0.9746 | 0.2382 | 0.0503 | 0.0312 |
| 6 | uniform | 20260526 | 0.1808 | 0.2344 | 34.36 | 0.9548 | 0.0658 | 0.0480 | 0.0336 |
| 7 | dmsr_normal_s4.0 | 20260526 | 0.3452 | 0.4062 | 60.92 | 0.9764 | 0.3314 | 0.0483 | 0.0318 |
| 8 | dmsr_normal_s6.0 | 20260526 | 0.2341 | 0.2897 | 62.26 | 0.9721 | 0.3466 | 0.0486 | 0.0336 |
| 9 | dmsr_normal_s0.3 | 20260526 | 1.0000 | 0.9786 | 84.35 | 0.9529 | 0.3214 | 0.3287 | 0.0366 |
| 10 | dmsr_normal_s0.8 | 20260526 | 0.9679 | 0.9030 | 119.53 | 0.9246 | 0.0304 | 0.2395 | 0.0302 |
| 11 | dmsr_laplace_b0.5 | 20260526 | 0.9686 | 0.9259 | 128.60 | 0.9846 | 0.4650 | 0.1795 | 0.0313 |
| 12 | dmsr_normal_s1.5 | 20260526 | 0.7623 | 0.7475 | 134.92 | 0.9858 | 0.4588 | 0.1389 | 0.0304 |
| 13 | hang_laplace_b0.5 | 20260526 | 0.9565 | 0.9097 | 136.67 | 0.9866 | 0.4664 | 0.1831 | 0.0315 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | linear_vp | 1 | 13.15 | 52.7244 | 0.833 | 0.611 | 0.0527 |
| 2 | dmsr_cosmix_w0.5 | 1 | 16.01 | 52.0606 | 0.819 | 0.536 | 0.0522 |
| 3 | dmsr_cosmix_w0.8 | 1 | 22.60 | 33.5359 | 0.820 | 0.494 | 0.0573 |
| 4 | dmsr_normal_s2.5 | 1 | 27.84 | 28.4797 | 0.844 | 0.527 | 0.0582 |
| 5 | cosine_vp | 1 | 30.65 | 25.3568 | 0.846 | 0.520 | 0.0503 |
| 6 | uniform | 1 | 34.36 | 132.1507 | 0.792 | 0.334 | 0.0480 |
| 7 | dmsr_normal_s4.0 | 1 | 60.92 | 37.6358 | 0.863 | 0.423 | 0.0483 |
| 8 | dmsr_normal_s6.0 | 1 | 62.26 | 51.3975 | 0.846 | 0.312 | 0.0486 |
| 9 | dmsr_normal_s0.3 | 1 | 84.35 | 190.0943 | 0.202 | 0.018 | 0.3287 |
| 10 | dmsr_normal_s0.8 | 1 | 119.53 | 321.3629 | 0.289 | 0.012 | 0.2395 |
| 11 | dmsr_laplace_b0.5 | 1 | 128.60 | 120.7566 | 0.407 | 0.054 | 0.1795 |
| 12 | dmsr_normal_s1.5 | 1 | 134.92 | 83.9842 | 0.753 | 0.223 | 0.1389 |
| 13 | hang_laplace_b0.5 | 1 | 136.67 | 95.2043 | 0.638 | 0.137 | 0.1831 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| dmsr_cosmix_w0.5 | 1 | -14.641 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_cosmix_w0.8 | 1 | -8.050 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 97.947 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.3 | 1 | 53.700 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.8 | 1 | 88.882 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 104.267 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s2.5 | 1 | -2.811 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | 30.271 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s6.0 | 1 | 31.612 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| hang_laplace_b0.5 | 1 | 106.023 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -17.503 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | 3.710 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| dmsr_normal_s0.8 | -0.0010 | 0.9698 | 0.7605 |
| dmsr_normal_s1.5 | -0.0008 | 0.9696 | 0.8611 |
| dmsr_cosmix_w0.8 | -0.0005 | 0.9693 | 0.9427 |
| dmsr_normal_s2.5 | -0.0005 | 0.9692 | 0.9418 |
| dmsr_cosmix_w0.5 | -0.0003 | 0.9691 | 0.9478 |
| cosine_vp | 0 (baseline) | 0.9688 | 0.9497 |
| dmsr_laplace_b0.5 | +0.0000 | 0.9687 | 0.8205 |
| hang_laplace_b0.5 | +0.0003 | 0.9685 | 0.8169 |
| linear_vp | +0.0004 | 0.9684 | 0.9473 |
| dmsr_normal_s4.0 | +0.0005 | 0.9682 | 0.9517 |
| uniform | +0.0023 | 0.9664 | 0.9520 |
| dmsr_normal_s6.0 | +0.0024 | 0.9664 | 0.9514 |
| dmsr_normal_s0.3 | +0.0053 | 0.9634 | 0.6713 |

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
