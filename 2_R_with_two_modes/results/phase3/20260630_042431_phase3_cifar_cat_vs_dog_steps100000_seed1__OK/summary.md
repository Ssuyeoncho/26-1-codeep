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
| 1 | uniform | 20260526 | 52.74 | 6.73 | 0.4669 | 0.1242 | 0.23606 | 0.8463 |
| 2 | at0_normal_s4.0 | 20260526 | 53.05 | 7.43 | 0.5323 | 0.2536 | 0.24752 | 0.8550 |
| 3 | dmsr_laplace_b4.0 | 20260526 | 53.61 | 11.81 | 0.8754 | 0.3607 | 0.24696 | 0.8373 |
| 4 | at0_laplace_b4.0 | 20260526 | 53.82 | 8.37 | 0.5400 | 0.2052 | 0.23959 | 0.8531 |
| 5 | dmsr_normal_s4.0 | 20260526 | 54.82 | 14.34 | 1.3356 | 0.3508 | 0.24934 | 0.8325 |
| 6 | linear_vp | 20260526 | 65.71 | 9.47 | 0.7741 | 0.1161 | 0.24813 | 0.8592 |
| 7 | cosine_vp | 20260526 | 71.12 | 11.90 | 0.8865 | 0.2315 | 0.24875 | 0.8503 |
| 8 | dmsr_laplace_b1.5 | 20260526 | 86.67 | 30.35 | 3.7756 | 0.6822 | 0.26134 | 0.8200 |
| 9 | at0_laplace_b1.5 | 20260526 | 94.29 | 20.79 | 1.5251 | 0.1655 | 0.25895 | 0.8578 |
| 10 | at0_normal_s1.5 | 20260526 | 127.39 | 24.72 | 1.8474 | 0.1532 | 0.35910 | 0.8544 |
| 11 | at0_laplace_b0.5 | 20260526 | 165.51 | 30.79 | 2.1404 | 0.0235 | 0.40209 | 0.8518 |
| 12 | at0_normal_s0.5 | 20260526 | 308.54 | 330.75 | 210.3304 | 0.0012 | 0.43584 | 0.9964 |
| 13 | dmsr_normal_s0.5 | 20260526 | 441.50 | 1194.44 | 4014.1136 | 0.9928 | 0.77562 | 1.0000 |
| 14 | dmsr_laplace_b0.5 | 20260526 | 464.26 | 598.59 | 805.8571 | 0.9532 | 0.46342 | 0.9999 |
| 15 | dmsr_normal_s1.5 | 20260526 | 473.08 | 291.01 | 155.4660 | 0.7420 | 0.39266 | 0.9988 |

## Aggregated over seeds (n_seeds = 1)

FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다. seed 1개면 std 미표시.

| rank | schedule | n | FID(Incep) | FID(φ) | KID(φ) | Precision(φ) | Coverage(φ) |
|---:|---|---:|---|---|---|---|---|
| 1 | uniform | 1 | 52.74 | 6.73 | 0.4669 | 0.960 | 0.914 |
| 2 | at0_normal_s4.0 | 1 | 53.05 | 7.43 | 0.5323 | 0.965 | 0.902 |
| 3 | dmsr_laplace_b4.0 | 1 | 53.61 | 11.81 | 0.8754 | 0.960 | 0.853 |
| 4 | at0_laplace_b4.0 | 1 | 53.82 | 8.37 | 0.5400 | 0.976 | 0.874 |
| 5 | dmsr_normal_s4.0 | 1 | 54.82 | 14.34 | 1.3356 | 0.952 | 0.836 |
| 6 | linear_vp | 1 | 65.71 | 9.47 | 0.7741 | 0.977 | 0.874 |
| 7 | cosine_vp | 1 | 71.12 | 11.90 | 0.8865 | 0.977 | 0.842 |
| 8 | dmsr_laplace_b1.5 | 1 | 86.67 | 30.35 | 3.7756 | 0.910 | 0.690 |
| 9 | at0_laplace_b1.5 | 1 | 94.29 | 20.79 | 1.5251 | 0.972 | 0.755 |
| 10 | at0_normal_s1.5 | 1 | 127.39 | 24.72 | 1.8474 | 0.975 | 0.689 |
| 11 | at0_laplace_b0.5 | 1 | 165.51 | 30.79 | 2.1404 | 0.974 | 0.601 |
| 12 | at0_normal_s0.5 | 1 | 308.54 | 330.75 | 210.3304 | 0.981 | 0.043 |
| 13 | dmsr_normal_s0.5 | 1 | 441.50 | 1194.44 | 4014.1136 | 0.009 | 0.002 |
| 14 | dmsr_laplace_b0.5 | 1 | 464.26 | 598.59 | 805.8571 | 0.813 | 0.006 |
| 15 | dmsr_normal_s1.5 | 1 | 473.08 | 291.01 | 155.4660 | 0.998 | 0.056 |

## Significance vs baseline (cosine_vp, metric = Inception-FID)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| at0_laplace_b0.5 | 1 | 94.394 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | 23.173 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | -17.304 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 237.417 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 56.271 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | -18.067 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 393.136 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | 15.545 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | -17.514 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 370.376 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 401.959 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | -16.298 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -5.410 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -18.380 | ✓ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Interpretation Guide

- 헤드라인은 InceptionV3 기반 FID(CIFAR 표준). φ 공간 FID/KID/PRDC는 Phase 2와 동일한 정의로, 두 phase를 잇는 비교축이다.
- FID는 품질·다양성을 뭉뚱그리므로 Precision(품질)·Recall/Coverage(다양성)를 함께 본다.
- KID는 표본이 적을 때 FID보다 신뢰성이 높다.
- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며 full-range support 와의 균형이 중요하다(Phase 1 결론).
- 유의성 검정은 seed가 부족하면(현재 1개) 건너뛴다. `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.
