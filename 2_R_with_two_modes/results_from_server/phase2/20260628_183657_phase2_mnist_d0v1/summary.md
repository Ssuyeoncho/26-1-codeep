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

## Schedules

- **cosine_vp**: VP cosine schedule induced density (관행 baseline).
- **hang_laplace_b0.5**: Laplace(0, 0.5) — Hang et al. baseline.
- **dmsr_normal_s1.5**: N(λ_R*=-0.909, s=1.5).
- **dmsr_normal_s0.8**: N(λ_R*=-0.909, s=0.8).
- **dmsr_normal_s0.3**: N(λ_R*=-0.909, s=0.3).
- **dmsr_laplace_b0.5**: Laplace(λ_R*=-0.909, b=0.5).

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dmsr_normal_s1.5 | 20260526 | 0.7727 | 0.7938 | 105.30 | 0.9232 | 0.2200 | 0.2890 | 0.0320 |
| 2 | cosine_vp | 20260526 | 0.4779 | 0.6039 | 113.32 | 0.8927 | 0.1700 | 0.1397 | 0.0364 |
| 3 | dmsr_normal_s0.8 | 20260526 | 0.9771 | 0.8956 | 129.95 | 0.8932 | 0.2500 | 0.3909 | 0.0344 |
| 4 | dmsr_laplace_b0.5 | 20260526 | 0.9735 | 0.9177 | 132.63 | 0.8951 | 0.2300 | 0.3525 | 0.0352 |
| 5 | dmsr_normal_s0.3 | 20260526 | 1.0000 | 0.9610 | 138.73 | 0.8850 | 0.2600 | 0.4348 | 0.0438 |
| 6 | hang_laplace_b0.5 | 20260526 | 0.9160 | 0.8740 | 141.06 | 0.8468 | 0.1500 | 0.3151 | 0.0318 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | dmsr_normal_s1.5 | 1 | 105.30 | 237.3952 | 0.990 | 0.140 | 0.2890 |
| 2 | cosine_vp | 1 | 113.32 | 294.3676 | 0.990 | 0.130 | 0.1397 |
| 3 | dmsr_normal_s0.8 | 1 | 129.95 | 285.0802 | 1.000 | 0.130 | 0.3909 |
| 4 | dmsr_laplace_b0.5 | 1 | 132.63 | 278.1871 | 1.000 | 0.130 | 0.3525 |
| 5 | dmsr_normal_s0.3 | 1 | 138.73 | 294.8703 | 1.000 | 0.120 | 0.4348 |
| 6 | hang_laplace_b0.5 | 1 | 141.06 | 321.8049 | 0.990 | 0.120 | 0.3151 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| dmsr_laplace_b0.5 | 1 | 19.311 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.3 | 1 | 25.411 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.8 | 1 | 16.629 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | -8.023 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| hang_laplace_b0.5 | 1 | 27.739 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Interpretation Guide

- 이 단계의 목적은 생성 성능 주장이 아니라 **파이프라인·통계 틀 검증**이다.
  MNIST는 쉬운 데이터라 schedule 간 FID 차이가 작아도 실패가 아니다.
- 생성 품질은 φ-feature space에서 FID·KID·Precision/Recall/Density/Coverage로 잰다.
  FID는 품질·다양성을 뭉뚱그리므로, Precision(품질)과 Recall/Coverage(다양성)를 함께 보면 mode collapse 같은 실패를 분리해 볼 수 있다(Phase 3와 동일한 지표).
- KID는 표본이 적을 때 FID보다 신뢰성이 높고 부분표본 분산을 함께 준다.
- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며, full-range support 와의 균형이 중요하다(Phase 1 결론).
- λ_R*는 DMSR_φ(λ)의 수치 미분 peak에서 경험적으로 추정한다. 이 값은 Phase 3로 넘어가지 않으며 CIFAR에서 독립적으로 재추정한다.
- 유의성 검정은 seed가 부족하면(현재 1개) 건너뛴다. Phase 3에서 `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.
