# Phase 3 유의성 검정 (fid_inception, baseline = cosine_vp)

- 지표 해석: **fid_inception** 은 낮을수록 좋다.
- 설계: 동일 seed에서 측정한 값끼리 짝지은 **paired** 비교 (n_seeds=1).
- mean_diff = (해당 schedule − baseline)의 seed 평균. 낮을수록 좋은 지표면 음수가 개선을 뜻한다.
- p-value 는 paired t-test 기준 (seed ≥ 5 이면 Wilcoxon 보조 제시).

| schedule | n_pairs | mean_diff | improved | t p-value | wilcoxon p-value | status |
|---|---:|---:|:---:|---:|---:|---|
| at0_laplace_b0.5 | 1 | 94.39 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | 23.17 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | -17.3 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 237.4 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 56.27 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | -18.07 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 393.1 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | 15.54 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | -17.51 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 370.4 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 402 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | -16.3 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -5.41 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -18.38 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

> ⚠️ seed가 1개뿐이라 유의성 검정을 수행할 수 없습니다. `--num-seeds 3` 이상으로 재실행해 이 표를 채웁니다.
