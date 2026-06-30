# Phase 2 유의성 검정 (fid_phi, baseline = cosine_vp)

- 지표 해석: **fid_phi** 은 낮을수록 좋다.
- 설계: 동일 seed에서 측정한 값끼리 짝지은 **paired** 비교 (n_seeds=1).
- mean_diff = (해당 schedule − baseline)의 seed 평균. 낮을수록 좋은 지표면 음수가 개선을 뜻한다.
- p-value 는 paired t-test 기준 (seed ≥ 5 이면 Wilcoxon 보조 제시).

| schedule | n_pairs | mean_diff | improved | t p-value | wilcoxon p-value | status |
|---|---:|---:|:---:|---:|---:|---|
| at0_laplace_b0.5 | 1 | 16.41 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b1.5 | 1 | 1.769 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_laplace_b4.0 | 1 | 3.491 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s0.5 | 1 | 60.06 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s1.5 | 1 | 4.212 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| at0_normal_s4.0 | 1 | 1.164 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 23.63 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b1.5 | 1 | 2.281 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b4.0 | 1 | 4.711 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.5 | 1 | 24.65 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 21.15 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | 3.866 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | -0.8511 | ✓ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | 5.783 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

> ⚠️ seed가 1개뿐이라 유의성 검정을 수행할 수 없습니다. Phase 3 본 실험에서는 `--num-seeds 3` 이상으로 재실행해 이 표를 채웁니다.
