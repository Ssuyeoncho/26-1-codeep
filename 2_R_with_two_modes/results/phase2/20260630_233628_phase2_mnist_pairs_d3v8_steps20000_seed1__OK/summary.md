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
- accuracy: **99.85%** (n=1984), macro-F1: 0.9985
  digit 3: P=0.999/R=0.998/F1=0.999, digit 8: P=0.998/R=0.999/F1=0.998

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
| 1 | linear_vp | 20260526 | 0.2677 | 0.3216 | 2.06 | 0.9827 | 0.0066 | 0.0699 | 0.0397 |
| 2 | cosine_vp | 20260526 | 0.4439 | 0.4802 | 2.91 | 0.9803 | 0.0004 | 0.0665 | 0.0393 |
| 3 | at0_normal_s4.0 | 20260526 | 0.2978 | 0.3537 | 4.07 | 0.9735 | 0.0084 | 0.0648 | 0.0398 |
| 4 | at0_laplace_b1.5 | 20260526 | 0.6353 | 0.6240 | 4.68 | 0.9737 | 0.0228 | 0.0720 | 0.0387 |
| 5 | dmsr_laplace_b1.5 | 20260526 | 0.6194 | 0.6160 | 5.19 | 0.9735 | 0.0248 | 0.0707 | 0.0389 |
| 6 | at0_laplace_b4.0 | 20260526 | 0.3181 | 0.3550 | 6.40 | 0.9718 | 0.0178 | 0.0641 | 0.0405 |
| 7 | dmsr_normal_s4.0 | 20260526 | 0.2980 | 0.3497 | 6.77 | 0.9696 | 0.0244 | 0.0646 | 0.0401 |
| 8 | at0_normal_s1.5 | 20260526 | 0.6866 | 0.6732 | 7.12 | 0.9605 | 0.0004 | 0.1697 | 0.0384 |
| 9 | dmsr_laplace_b4.0 | 20260526 | 0.3140 | 0.3503 | 7.62 | 0.9689 | 0.0280 | 0.0637 | 0.0405 |
| 10 | uniform | 20260526 | 0.1531 | 0.1995 | 8.69 | 0.9638 | 0.0052 | 0.0644 | 0.0420 |
| 11 | at0_laplace_b0.5 | 20260526 | 0.9476 | 0.8328 | 19.32 | 0.9388 | 0.0648 | 0.2085 | 0.0383 |
| 12 | dmsr_normal_s1.5 | 20260526 | 0.6662 | 0.6559 | 24.05 | 0.9317 | 0.0916 | 0.1427 | 0.0383 |
| 13 | dmsr_laplace_b0.5 | 20260526 | 0.9276 | 0.8557 | 26.53 | 0.9152 | 0.0066 | 0.1870 | 0.0385 |
| 14 | dmsr_normal_s0.5 | 20260526 | 0.9797 | 0.8867 | 27.56 | 0.9198 | 0.1254 | 0.3246 | 0.0387 |
| 15 | at0_normal_s0.5 | 20260526 | 0.9947 | 0.8614 | 62.97 | 0.9690 | 0.4266 | 0.3138 | 0.0389 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | linear_vp | 1 | 2.06 | 1.3793 | 0.876 | 0.776 | 0.0699 |
| 2 | cosine_vp | 1 | 2.91 | 2.1714 | 0.845 | 0.726 | 0.0665 |
| 3 | at0_normal_s4.0 | 1 | 4.07 | 2.8525 | 0.851 | 0.720 | 0.0648 |
| 4 | at0_laplace_b1.5 | 1 | 4.68 | 3.5819 | 0.848 | 0.690 | 0.0720 |
| 5 | dmsr_laplace_b1.5 | 1 | 5.19 | 4.0452 | 0.839 | 0.625 | 0.0707 |
| 6 | at0_laplace_b4.0 | 1 | 6.40 | 5.0778 | 0.849 | 0.653 | 0.0641 |
| 7 | dmsr_normal_s4.0 | 1 | 6.77 | 4.8299 | 0.867 | 0.652 | 0.0646 |
| 8 | at0_normal_s1.5 | 1 | 7.12 | 4.8960 | 0.820 | 0.525 | 0.1697 |
| 9 | dmsr_laplace_b4.0 | 1 | 7.62 | 5.7905 | 0.856 | 0.633 | 0.0637 |
| 10 | uniform | 1 | 8.69 | 6.1199 | 0.860 | 0.622 | 0.0644 |
| 11 | at0_laplace_b0.5 | 1 | 19.32 | 11.5214 | 0.781 | 0.269 | 0.2085 |
| 12 | dmsr_normal_s1.5 | 1 | 24.05 | 12.4109 | 0.683 | 0.146 | 0.1427 |
| 13 | dmsr_laplace_b0.5 | 1 | 26.53 | 12.2640 | 0.301 | 0.029 | 0.1870 |
| 14 | dmsr_normal_s0.5 | 1 | 27.56 | 10.5262 | 0.133 | 0.009 | 0.3246 |
| 15 | at0_normal_s0.5 | 1 | 62.97 | 11.2148 | 0.262 | 0.017 | 0.3138 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| at0_laplace_b0.5 | 1 | 16.411 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | 1.769 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | 3.491 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 60.059 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 4.212 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | 1.164 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 23.627 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | 2.281 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | 4.711 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 24.650 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 21.147 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | 3.866 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -0.851 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | 5.783 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| dmsr_normal_s1.5 | -0.0010 | 0.9617 | 0.8573 |
| at0_laplace_b0.5 | -0.0010 | 0.9617 | 0.7915 |
| at0_normal_s1.5 | -0.0010 | 0.9616 | 0.8303 |
| dmsr_laplace_b0.5 | -0.0009 | 0.9615 | 0.8130 |
| at0_laplace_b1.5 | -0.0007 | 0.9613 | 0.9280 |
| dmsr_normal_s0.5 | -0.0006 | 0.9613 | 0.6754 |
| at0_normal_s0.5 | -0.0005 | 0.9611 | 0.6862 |
| dmsr_laplace_b1.5 | -0.0005 | 0.9611 | 0.9293 |
| cosine_vp | 0 (baseline) | 0.9607 | 0.9335 |
| linear_vp | +0.0003 | 0.9603 | 0.9301 |
| at0_normal_s4.0 | +0.0004 | 0.9602 | 0.9352 |
| dmsr_normal_s4.0 | +0.0008 | 0.9599 | 0.9354 |
| at0_laplace_b4.0 | +0.0011 | 0.9595 | 0.9359 |
| dmsr_laplace_b4.0 | +0.0011 | 0.9595 | 0.9363 |
| uniform | +0.0026 | 0.9580 | 0.9356 |

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
