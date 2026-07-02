# Phase 3 CIFAR-10 Two-Class Experiment Summary

## Config

- class pair: airplane_vs_frog
- lambda range: [-15.0, 15.0]
- rho: 0.5
- lambda_R*: 3.3673
- T_R: [-1.5306, 4.5918]
- train steps: 100000
- batch size: 128
- num_seeds: 1

## Schedules

- **cosine_vp**: VP cosine-β induced λ density (0-centered, heavy sech tails). 관행 baseline.
- **linear_vp**: VP linear-β (DDPM) induced λ density. 관행 baseline.
- **uniform**: Uniform(λ∈[-15.0,15.0]). 무정보 baseline.
- **dmsr_normal_s0.5**: Normal(center=lambda_R*=3.37, s=0.5).
- **dmsr_normal_s1.5**: Normal(center=lambda_R*=3.37, s=1.5).
- **dmsr_normal_s4.0**: Normal(center=lambda_R*=3.37, s=4.0).
- **dmsr_laplace_b0.5**: Laplace(center=lambda_R*=3.37, b=0.5).
- **dmsr_laplace_b1.5**: Laplace(center=lambda_R*=3.37, b=1.5).
- **dmsr_laplace_b4.0**: Laplace(center=lambda_R*=3.37, b=4.0).
- **at0_normal_s0.5**: Normal(center=0, s=0.5).
- **at0_normal_s1.5**: Normal(center=0, s=1.5).
- **at0_normal_s4.0**: Normal(center=0, s=4.0).
- **at0_laplace_b0.5**: Laplace(center=0, b=0.5).  [Hang et al. baseline]
- **at0_laplace_b1.5**: Laplace(center=0, b=1.5).  [Hang et al. baseline]
- **at0_laplace_b4.0**: Laplace(center=0, b=4.0).  [Hang et al. baseline]

## Per-run Results (sorted by Inception-FID)

| rank | schedule | seed | FID(Incep) | FID(φ) | KID(φ) | M | mean MSE | clf conf |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | uniform | 20260526 | 43.36 | 3.50 | 4.4438 | 0.2051 | 0.22899 | 0.9907 |
| 2 | dmsr_laplace_b4.0 | 20260526 | 47.87 | 10.01 | 10.8469 | 0.4845 | 0.23515 | 0.9822 |
| 3 | dmsr_normal_s4.0 | 20260526 | 48.08 | 26.47 | 23.8081 | 0.5115 | 0.24149 | 0.9698 |
| 4 | at0_normal_s4.0 | 20260526 | 50.57 | 8.00 | 10.0833 | 0.5242 | 0.23729 | 0.9877 |
| 5 | at0_laplace_b4.0 | 20260526 | 52.02 | 9.46 | 11.6995 | 0.4980 | 0.23533 | 0.9879 |
| 6 | cosine_vp | 20260526 | 64.96 | 12.81 | 12.9008 | 0.6587 | 0.24181 | 0.9871 |
| 7 | linear_vp | 20260526 | 68.22 | 13.74 | 17.8576 | 0.3871 | 0.24195 | 0.9905 |
| 8 | dmsr_laplace_b1.5 | 20260526 | 72.51 | 61.65 | 47.3125 | 0.7603 | 0.25614 | 0.9543 |
| 9 | at0_laplace_b1.5 | 20260526 | 87.84 | 23.96 | 21.8240 | 0.7969 | 0.24948 | 0.9840 |
| 10 | at0_normal_s1.5 | 20260526 | 106.78 | 28.65 | 23.9321 | 0.8447 | 0.34821 | 0.9857 |
| 11 | at0_laplace_b0.5 | 20260526 | 146.67 | 38.05 | 32.4619 | 0.9767 | 0.38480 | 0.9815 |
| 12 | at0_normal_s0.5 | 20260526 | 270.67 | 349.38 | 94.3816 | 0.9990 | 0.42741 | 0.9713 |
| 13 | dmsr_normal_s0.5 | 20260526 | 412.99 | 587.63 | 107.4123 | 0.9928 | 0.67501 | 1.0000 |
| 14 | dmsr_normal_s1.5 | 20260526 | 414.47 | 475.70 | 102.9376 | 0.7931 | 0.34573 | 0.9981 |
| 15 | dmsr_laplace_b0.5 | 20260526 | 421.16 | 529.15 | 97.7328 | 0.9569 | 0.42378 | 0.9997 |

## Aggregated over seeds (n_seeds = 1)

FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다. seed 1개면 std 미표시.

| rank | schedule | n | FID(Incep) | FID(φ) | KID(φ) | Precision(φ) | Coverage(φ) |
|---:|---|---:|---|---|---|---|---|
| 1 | uniform | 1 | 43.36 | 3.50 | 4.4438 | 0.970 | 0.952 |
| 2 | dmsr_laplace_b4.0 | 1 | 47.87 | 10.01 | 10.8469 | 0.964 | 0.928 |
| 3 | dmsr_normal_s4.0 | 1 | 48.08 | 26.47 | 23.8081 | 0.946 | 0.913 |
| 4 | at0_normal_s4.0 | 1 | 50.57 | 8.00 | 10.0833 | 0.972 | 0.939 |
| 5 | at0_laplace_b4.0 | 1 | 52.02 | 9.46 | 11.6995 | 0.966 | 0.920 |
| 6 | cosine_vp | 1 | 64.96 | 12.81 | 12.9008 | 0.967 | 0.885 |
| 7 | linear_vp | 1 | 68.22 | 13.74 | 17.8576 | 0.980 | 0.910 |
| 8 | dmsr_laplace_b1.5 | 1 | 72.51 | 61.65 | 47.3125 | 0.935 | 0.748 |
| 9 | at0_laplace_b1.5 | 1 | 87.84 | 23.96 | 21.8240 | 0.971 | 0.837 |
| 10 | at0_normal_s1.5 | 1 | 106.78 | 28.65 | 23.9321 | 0.973 | 0.738 |
| 11 | at0_laplace_b0.5 | 1 | 146.67 | 38.05 | 32.4619 | 0.972 | 0.697 |
| 12 | at0_normal_s0.5 | 1 | 270.67 | 349.38 | 94.3816 | 0.997 | 0.230 |
| 13 | dmsr_normal_s0.5 | 1 | 412.99 | 587.63 | 107.4123 | 0.573 | 0.013 |
| 14 | dmsr_normal_s1.5 | 1 | 414.47 | 475.70 | 102.9376 | 0.519 | 0.013 |
| 15 | dmsr_laplace_b0.5 | 1 | 421.16 | 529.15 | 97.7328 | 0.690 | 0.009 |

## Significance vs baseline (cosine_vp, metric = Inception-FID)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| at0_laplace_b0.5 | 1 | 81.706 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | 22.883 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | -12.936 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 205.708 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 41.821 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | -14.393 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 356.199 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | 7.551 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | -17.084 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 348.027 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 349.514 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | -16.882 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | 3.260 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -21.600 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Interpretation Guide

- 헤드라인은 InceptionV3 기반 FID(CIFAR 표준). φ 공간 FID/KID/PRDC는 Phase 2와 동일한 정의로, 두 phase를 잇는 비교축이다.
- FID는 품질·다양성을 뭉뚱그리므로 Precision(품질)·Recall/Coverage(다양성)를 함께 본다.
- KID는 표본이 적을 때 FID보다 신뢰성이 높다.
- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며 full-range support 와의 균형이 중요하다(Phase 1 결론).
- 유의성 검정은 seed가 부족하면(현재 1개) 건너뛴다. `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.
