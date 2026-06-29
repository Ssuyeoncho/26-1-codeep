# Phase 2 유의성 검정 (fid_phi, baseline = cosine_vp)

- 지표 해석: **fid_phi** 은 낮을수록 좋다.
- 설계: 동일 seed에서 측정한 값끼리 짝지은 **paired** 비교 (n_seeds=1).
- mean_diff = (해당 schedule − baseline)의 seed 평균. 낮을수록 좋은 지표면 음수가 개선을 뜻한다.
- p-value 는 paired t-test 기준 (seed ≥ 5 이면 Wilcoxon 보조 제시).

| schedule | n_pairs | mean_diff | improved | t p-value | wilcoxon p-value | status |
|---|---:|---:|:---:|---:|---:|---|
| dmsr_cosmix_w0.5 | 1 | 0.7791 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_cosmix_w0.8 | 1 | 0.0291 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 16.96 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.3 | 1 | 31.12 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.8 | 1 | 7.484 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 13.19 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s2.5 | 1 | 3.276 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | 6.189 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s6.0 | 1 | 5.436 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| hang_laplace_b0.5 | 1 | 5.229 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | 2.312 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | 4.606 | ✗ | nan | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

> ⚠️ seed가 1개뿐이라 유의성 검정을 수행할 수 없습니다. Phase 3 본 실험에서는 `--num-seeds 3` 이상으로 재실행해 이 표를 채웁니다.
