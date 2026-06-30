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
- **dmsr_normal_s0.5**: Normal(center=λ_R*=-0.256, s=0.5).
- **dmsr_normal_s1.5**: Normal(center=λ_R*=-0.256, s=1.5).
- **dmsr_normal_s4.0**: Normal(center=λ_R*=-0.256, s=4.0).
- **dmsr_laplace_b0.5**: Laplace(center=λ_R*=-0.256, b=0.5).
- **dmsr_laplace_b1.5**: Laplace(center=λ_R*=-0.256, b=1.5).
- **dmsr_laplace_b4.0**: Laplace(center=λ_R*=-0.256, b=4.0).
- **at0_normal_s0.5**: Normal(center=0, s=0.5).
- **at0_normal_s1.5**: Normal(center=0, s=1.5).
- **at0_normal_s4.0**: Normal(center=0, s=4.0).
- **at0_laplace_b0.5**: Laplace(center=0, b=0.5).  [Hang et al. baseline]
- **at0_laplace_b1.5**: Laplace(center=0, b=1.5).  [Hang et al. baseline]
- **at0_laplace_b4.0**: Laplace(center=0, b=4.0).  [Hang et al. baseline]

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dmsr_laplace_b1.5 | 20260526 | 0.6925 | 0.6608 | 51.45 | 0.9732 | 0.1814 | 0.0606 | 0.0211 |
| 2 | at0_laplace_b1.5 | 20260526 | 0.6792 | 0.6512 | 54.73 | 0.9663 | 0.1772 | 0.0589 | 0.0211 |
| 3 | uniform | 20260526 | 0.1780 | 0.2347 | 68.26 | 0.9553 | 0.2116 | 0.0486 | 0.0232 |
| 4 | at0_laplace_b0.5 | 20260526 | 0.9569 | 0.8607 | 90.89 | 0.9388 | 0.1984 | 0.1918 | 0.0208 |
| 5 | at0_normal_s4.0 | 20260526 | 0.3442 | 0.3987 | 96.61 | 0.9638 | 0.2734 | 0.0505 | 0.0226 |
| 6 | dmsr_laplace_b0.5 | 20260526 | 0.9687 | 0.8771 | 98.53 | 0.9497 | 0.2616 | 0.2014 | 0.0207 |
| 7 | linear_vp | 20260526 | 0.3329 | 0.3851 | 98.82 | 0.9692 | 0.2912 | 0.0536 | 0.0216 |
| 8 | at0_laplace_b4.0 | 20260526 | 0.3573 | 0.3912 | 99.57 | 0.9614 | 0.2784 | 0.0499 | 0.0222 |
| 9 | dmsr_laplace_b4.0 | 20260526 | 0.3610 | 0.3947 | 102.89 | 0.9617 | 0.2870 | 0.0496 | 0.0221 |
| 10 | cosine_vp | 20260526 | 0.4963 | 0.5170 | 128.05 | 0.9718 | 0.3366 | 0.0525 | 0.0216 |
| 11 | dmsr_normal_s4.0 | 20260526 | 0.3461 | 0.4012 | 154.11 | 0.9707 | 0.3692 | 0.0506 | 0.0223 |
| 12 | dmsr_normal_s1.5 | 20260526 | 0.7607 | 0.7015 | 223.87 | 0.9812 | 0.4368 | 0.1539 | 0.0210 |
| 13 | dmsr_normal_s0.5 | 20260526 | 0.9989 | 0.9046 | 242.66 | 0.9478 | 0.3892 | 0.3164 | 0.0214 |
| 14 | at0_normal_s1.5 | 20260526 | 0.7409 | 0.6911 | 261.28 | 0.9854 | 0.4586 | 0.1450 | 0.0211 |
| 15 | at0_normal_s0.5 | 20260526 | 0.9949 | 0.8898 | 277.52 | 0.9594 | 0.4272 | 0.3025 | 0.0215 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | dmsr_laplace_b1.5 | 1 | 51.45 | 40.5040 | 0.868 | 0.539 | 0.0606 |
| 2 | at0_laplace_b1.5 | 1 | 54.73 | 42.5397 | 0.870 | 0.445 | 0.0589 |
| 3 | uniform | 1 | 68.26 | 78.5985 | 0.806 | 0.333 | 0.0486 |
| 4 | at0_laplace_b0.5 | 1 | 90.89 | 189.4944 | 0.761 | 0.158 | 0.1918 |
| 5 | at0_normal_s4.0 | 1 | 96.61 | 64.4226 | 0.849 | 0.367 | 0.0505 |
| 6 | dmsr_laplace_b0.5 | 1 | 98.53 | 138.3116 | 0.788 | 0.228 | 0.2014 |
| 7 | linear_vp | 1 | 98.82 | 63.1157 | 0.867 | 0.465 | 0.0536 |
| 8 | at0_laplace_b4.0 | 1 | 99.57 | 71.4940 | 0.833 | 0.355 | 0.0499 |
| 9 | dmsr_laplace_b4.0 | 1 | 102.89 | 67.6416 | 0.843 | 0.387 | 0.0496 |
| 10 | cosine_vp | 1 | 128.05 | 74.0617 | 0.847 | 0.416 | 0.0525 |
| 11 | dmsr_normal_s4.0 | 1 | 154.11 | 87.8567 | 0.842 | 0.375 | 0.0506 |
| 12 | dmsr_normal_s1.5 | 1 | 223.87 | 145.5185 | 0.831 | 0.391 | 0.1539 |
| 13 | dmsr_normal_s0.5 | 1 | 242.66 | 849.8890 | 0.080 | 0.004 | 0.3164 |
| 14 | at0_normal_s1.5 | 1 | 261.28 | 194.4843 | 0.788 | 0.343 | 0.1450 |
| 15 | at0_normal_s0.5 | 1 | 277.52 | 1048.4193 | 0.021 | 0.002 | 0.3025 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| at0_laplace_b0.5 | 1 | -37.163 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | -73.322 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | -28.477 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 149.467 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 133.228 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | -31.436 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | -29.520 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | -76.597 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | -25.160 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 114.613 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 95.823 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | 26.057 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -29.226 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -59.786 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| dmsr_laplace_b0.5 | -0.0009 | 0.9793 | 0.7986 |
| at0_laplace_b0.5 | -0.0008 | 0.9792 | 0.8082 |
| dmsr_normal_s1.5 | -0.0006 | 0.9790 | 0.8461 |
| dmsr_laplace_b1.5 | -0.0006 | 0.9789 | 0.9394 |
| at0_laplace_b1.5 | -0.0005 | 0.9789 | 0.9411 |
| at0_normal_s1.5 | -0.0005 | 0.9789 | 0.8550 |
| dmsr_normal_s0.5 | -0.0002 | 0.9786 | 0.6836 |
| at0_normal_s0.5 | -0.0001 | 0.9785 | 0.6975 |
| cosine_vp | 0 (baseline) | 0.9784 | 0.9475 |
| linear_vp | +0.0000 | 0.9784 | 0.9464 |
| dmsr_laplace_b4.0 | +0.0005 | 0.9779 | 0.9504 |
| at0_laplace_b4.0 | +0.0005 | 0.9778 | 0.9501 |
| dmsr_normal_s4.0 | +0.0007 | 0.9777 | 0.9494 |
| at0_normal_s4.0 | +0.0010 | 0.9774 | 0.9495 |
| uniform | +0.0016 | 0.9768 | 0.9514 |

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
