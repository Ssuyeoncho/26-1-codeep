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

## Schedules

- **cosine_vp**: VP cosine schedule induced density (관행 baseline).
- **hang_laplace_b0.5**: Laplace(0, 0.5) — Hang et al. baseline.
- **dmsr_normal_s1.5**: N(λ_R*=-0.256, s=1.5).
- **dmsr_normal_s0.8**: N(λ_R*=-0.256, s=0.8).
- **dmsr_normal_s0.3**: N(λ_R*=-0.256, s=0.3).
- **dmsr_laplace_b0.5**: Laplace(λ_R*=-0.256, b=0.5).

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cosine_vp | 20260526 | 0.4972 | 0.5186 | 36.77 | 0.9668 | 0.1406 | 0.0530 | 0.0217 |
| 2 | dmsr_laplace_b0.5 | 20260526 | 0.9687 | 0.8776 | 101.48 | 0.9497 | 0.2554 | 0.2086 | 0.0208 |
| 3 | dmsr_normal_s1.5 | 20260526 | 0.7617 | 0.7032 | 103.67 | 0.9716 | 0.3008 | 0.1554 | 0.0208 |
| 4 | hang_laplace_b0.5 | 20260526 | 0.9569 | 0.8615 | 112.32 | 0.9309 | 0.1914 | 0.1896 | 0.0209 |
| 5 | dmsr_normal_s0.8 | 20260526 | 0.9684 | 0.8389 | 149.49 | 0.9087 | 0.0222 | 0.2566 | 0.0208 |
| 6 | dmsr_normal_s0.3 | 20260526 | 1.0000 | 0.9488 | 478.34 | 0.9983 | 0.4992 | 0.3396 | 0.0223 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | cosine_vp | 1 | 36.77 | 71.0683 | 0.847 | 0.482 | 0.0530 |
| 2 | dmsr_laplace_b0.5 | 1 | 101.48 | 138.1761 | 0.737 | 0.190 | 0.2086 |
| 3 | dmsr_normal_s1.5 | 1 | 103.67 | 70.6872 | 0.814 | 0.397 | 0.1554 |
| 4 | hang_laplace_b0.5 | 1 | 112.32 | 236.5965 | 0.577 | 0.092 | 0.1896 |
| 5 | dmsr_normal_s0.8 | 1 | 149.49 | 435.0726 | 0.271 | 0.015 | 0.2566 |
| 6 | dmsr_normal_s0.3 | 1 | 478.34 | 2140.1582 | 0.000 | 0.000 | 0.3396 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| dmsr_laplace_b0.5 | 1 | 64.709 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.3 | 1 | 441.575 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.8 | 1 | 112.723 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 66.896 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| hang_laplace_b0.5 | 1 | 75.548 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Interpretation Guide

- 이 단계의 목적은 생성 성능 주장이 아니라 **파이프라인·통계 틀 검증**이다.
  MNIST는 쉬운 데이터라 schedule 간 FID 차이가 작아도 실패가 아니다.
- 생성 품질은 φ-feature space에서 FID·KID·Precision/Recall/Density/Coverage로 잰다.
  FID는 품질·다양성을 뭉뚱그리므로, Precision(품질)과 Recall/Coverage(다양성)를 함께 보면 mode collapse 같은 실패를 분리해 볼 수 있다(Phase 3와 동일한 지표).
- KID는 표본이 적을 때 FID보다 신뢰성이 높고 부분표본 분산을 함께 준다.
- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며, full-range support 와의 균형이 중요하다(Phase 1 결론).
- λ_R*는 DMSR_φ(λ)의 수치 미분 peak에서 경험적으로 추정한다. 이 값은 Phase 3로 넘어가지 않으며 CIFAR에서 독립적으로 재추정한다.
- 유의성 검정은 seed가 부족하면(현재 1개) 건너뛴다. Phase 3에서 `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.
