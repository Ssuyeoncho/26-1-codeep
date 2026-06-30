# Phase 2 유의성 검정 (fid_phi, baseline = cosine_vp)

- 지표 해석: **fid_phi** 은 낮을수록 좋다.
- 설계: 동일 seed에서 측정한 값끼리 짝지은 **paired** 비교 (n_seeds=1).
- mean_diff = (해당 schedule − baseline)의 seed 평균. 낮을수록 좋은 지표면 음수가 개선을 뜻한다.
- p-value 는 paired t-test 기준 (seed ≥ 5 이면 Wilcoxon 보조 제시).

| schedule | n_pairs | mean_diff | improved | t p-value | wilcoxon p-value | status |
|---|---:|---:|:---:|---:|---:|---|
| at0_laplace_b0.5 | 1 | -37.16 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | -73.32 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | -28.48 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 149.5 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 133.2 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | -31.44 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | -29.52 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | -76.6 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | -25.16 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 114.6 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 95.82 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | 26.06 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -29.23 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | -59.79 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

> ⚠️ seed가 1개뿐이라 유의성 검정을 수행할 수 없습니다. Phase 3 본 실험에서는 `--num-seeds 3` 이상으로 재실행해 이 표를 채웁니다.
