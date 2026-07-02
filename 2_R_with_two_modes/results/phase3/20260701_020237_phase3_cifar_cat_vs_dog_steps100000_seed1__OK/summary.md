# Phase 3 CIFAR-10 Two-Class Experiment Summary

## Config

- class pair: cat_vs_dog
- lambda range: [-15.0, 15.0]
- rho: 0.5
- lambda_R*: 3.9796
- T_R: [1.5306, 5.2041]
- train steps: 100000
- batch size: 128
- num_seeds: 1

## Schedules

- **cosine_vp**: VP cosine-β induced λ density (0-centered, heavy sech tails). 관행 baseline.
- **linear_vp**: VP linear-β (DDPM) induced λ density. 관행 baseline.
- **uniform**: Uniform(λ∈[-15.0,15.0]). 무정보 baseline.
- **dmsr_normal_s0.5**: Normal(center=lambda_R*=3.98, s=0.5).
- **dmsr_normal_s1.5**: Normal(center=lambda_R*=3.98, s=1.5).
- **dmsr_normal_s4.0**: Normal(center=lambda_R*=3.98, s=4.0).
- **dmsr_laplace_b0.5**: Laplace(center=lambda_R*=3.98, b=0.5).
- **dmsr_laplace_b1.5**: Laplace(center=lambda_R*=3.98, b=1.5).
- **dmsr_laplace_b4.0**: Laplace(center=lambda_R*=3.98, b=4.0).
- **at0_normal_s0.5**: Normal(center=0, s=0.5).
- **at0_normal_s1.5**: Normal(center=0, s=1.5).
- **at0_normal_s4.0**: Normal(center=0, s=4.0).
- **at0_laplace_b0.5**: Laplace(center=0, b=0.5).  [Hang et al. baseline]
- **at0_laplace_b1.5**: Laplace(center=0, b=1.5).  [Hang et al. baseline]
- **at0_laplace_b4.0**: Laplace(center=0, b=4.0).  [Hang et al. baseline]

## Per-run Results (sorted by Inception-FID)

| rank | schedule | seed | FID(Incep) | FID(φ) | KID(φ) | M | mean MSE | clf conf |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | dmsr_normal_s4.0 | 20260526 | 48.92 | 9.19 | 1.1433 | 0.3508 | 0.24808 | 0.8568 |
| 2 | at0_normal_s4.0 | 20260526 | 52.93 | 2.31 | 0.1370 | 0.2536 | 0.24754 | 0.8881 |
| 3 | at0_laplace_b4.0 | 20260526 | 53.41 | 3.18 | 0.2499 | 0.2052 | 0.23970 | 0.8866 |
| 4 | dmsr_laplace_b4.0 | 20260526 | 53.59 | 10.31 | 1.1259 | 0.3607 | 0.24473 | 0.8501 |
| 5 | uniform | 20260526 | 54.13 | 2.61 | 0.3297 | 0.1242 | 0.23596 | 0.8805 |
| 6 | linear_vp | 20260526 | 62.08 | 5.14 | 0.6518 | 0.1161 | 0.24769 | 0.8936 |
| 7 | cosine_vp | 20260526 | 67.76 | 6.53 | 0.5976 | 0.2315 | 0.24815 | 0.8842 |
| 8 | dmsr_laplace_b1.5 | 20260526 | 77.27 | 25.13 | 3.2241 | 0.6822 | 0.26161 | 0.8358 |
| 9 | at0_laplace_b1.5 | 20260526 | 86.57 | 14.25 | 1.3772 | 0.1655 | 0.25919 | 0.8753 |
| 10 | at0_normal_s1.5 | 20260526 | 113.52 | 16.67 | 1.6522 | 0.1532 | 0.35388 | 0.8662 |
| 11 | at0_laplace_b0.5 | 20260526 | 155.99 | 22.23 | 2.1644 | 0.0235 | 0.40077 | 0.8532 |
| 12 | at0_normal_s0.5 | 20260526 | 273.46 | 202.31 | 74.0552 | 0.0012 | 0.44156 | 0.9862 |
| 13 | dmsr_normal_s0.5 | 20260526 | 446.99 | 1194.59 | 4015.7387 | 0.9928 | 0.84993 | 1.0000 |
| 14 | dmsr_laplace_b0.5 | 20260526 | 459.87 | 816.21 | 1595.7905 | 0.9532 | 0.52141 | 1.0000 |
| 15 | dmsr_normal_s1.5 | 20260526 | 460.14 | 662.44 | 997.3182 | 0.7420 | 0.40545 | 1.0000 |

## Aggregated over seeds (n_seeds = 1)

FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다. seed 1개면 std 미표시.

| rank | schedule | n | FID(Incep) | FID(φ) | KID(φ) | Precision(φ) | Coverage(φ) |
|---:|---|---:|---|---|---|---|---|
| 1 | dmsr_normal_s4.0 | 1 | 48.92 | 9.19 | 1.1433 | 0.972 | 0.896 |
| 2 | at0_normal_s4.0 | 1 | 52.93 | 2.31 | 0.1370 | 0.982 | 0.940 |
| 3 | at0_laplace_b4.0 | 1 | 53.41 | 3.18 | 0.2499 | 0.975 | 0.927 |
| 4 | dmsr_laplace_b4.0 | 1 | 53.59 | 10.31 | 1.1259 | 0.970 | 0.887 |
| 5 | uniform | 1 | 54.13 | 2.61 | 0.3297 | 0.965 | 0.951 |
| 6 | linear_vp | 1 | 62.08 | 5.14 | 0.6518 | 0.981 | 0.918 |
| 7 | cosine_vp | 1 | 67.76 | 6.53 | 0.5976 | 0.980 | 0.898 |
| 8 | dmsr_laplace_b1.5 | 1 | 77.27 | 25.13 | 3.2241 | 0.932 | 0.749 |
| 9 | at0_laplace_b1.5 | 1 | 86.57 | 14.25 | 1.3772 | 0.980 | 0.781 |
| 10 | at0_normal_s1.5 | 1 | 113.52 | 16.67 | 1.6522 | 0.970 | 0.710 |
| 11 | at0_laplace_b0.5 | 1 | 155.99 | 22.23 | 2.1644 | 0.984 | 0.640 |
| 12 | at0_normal_s0.5 | 1 | 273.46 | 202.31 | 74.0552 | 0.927 | 0.122 |
| 13 | dmsr_normal_s0.5 | 1 | 446.99 | 1194.59 | 4015.7387 | 0.001 | 0.001 |
| 14 | dmsr_laplace_b0.5 | 1 | 459.87 | 816.21 | 1595.7905 | 0.135 | 0.004 |
| 15 | dmsr_normal_s1.5 | 1 | 460.14 | 662.44 | 997.3182 | 0.144 | 0.004 |

## Significance vs baseline (cosine_vp, metric = Inception-FID)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| at0_laplace_b0.5 | 1 | 88.221 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | 18.810 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | -14.355 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 205.691 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 45.755 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | -14.837 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 392.101 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | 9.509 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | -14.179 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 379.224 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 392.376 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | -18.845 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -5.684 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -13.633 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Interpretation Guide

- 헤드라인은 InceptionV3 기반 FID(CIFAR 표준). φ 공간 FID/KID/PRDC는 Phase 2와 동일한 정의로, 두 phase를 잇는 비교축이다.
- FID는 품질·다양성을 뭉뚱그리므로 Precision(품질)·Recall/Coverage(다양성)를 함께 본다.
- KID는 표본이 적을 때 FID보다 신뢰성이 높다.
- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며 full-range support 와의 균형이 중요하다(Phase 1 결론).
- 유의성 검정은 seed가 부족하면(현재 1개) 건너뛴다. `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.
