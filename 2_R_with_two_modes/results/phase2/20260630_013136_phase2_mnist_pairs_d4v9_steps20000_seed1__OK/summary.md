# Phase 2 MNIST Pilot Summary

## Configuration

- digits: 4 vs 9
- λ range: [-10.0, 10.0]
- ρ threshold: 0.5
- empirical λ_R*: 0.7692
- transition region: [-1.2821, 1.2821]
- train steps per schedule: 20000
- DDIM steps: 50
- n_generate (FID): 5000

## φ Classifier Evaluation (test set)

DMSR_φ·FID-φ의 기반인 분류기 φ의 테스트셋 분류 성능 (자세한 표·confusion은 `classifier_report.md` / `classifier_report.json`, 그림은 plots 참고).
- accuracy: **99.50%** (n=1991), macro-F1: 0.9950
  digit 4: P=0.994/R=0.996/F1=0.995, digit 9: P=0.996/R=0.994/F1=0.995

## Schedules

- **cosine_vp**: VP cosine-β induced λ density (0-centered, heavy sech tails). 관행 baseline.
- **linear_vp**: VP linear-β (DDPM) induced λ density. 관행 baseline.
- **uniform**: Uniform(λ∈[-10.0,10.0]). 무정보 baseline.
- **hang_laplace_b0.5**: Laplace(0, b=0.5) — Hang et al. baseline.
- **dmsr_normal_s0.3**: N(λ_R*=0.769, s=0.3).
- **dmsr_normal_s0.8**: N(λ_R*=0.769, s=0.8).
- **dmsr_normal_s1.5**: N(λ_R*=0.769, s=1.5).
- **dmsr_normal_s2.5**: N(λ_R*=0.769, s=2.5).
- **dmsr_normal_s4.0**: N(λ_R*=0.769, s=4.0).
- **dmsr_normal_s6.0**: N(λ_R*=0.769, s=6.0).
- **dmsr_laplace_b0.5**: Laplace(λ_R*=0.769, b=0.5).
- **dmsr_cosmix_w0.5**: (1-0.5)·cosine + 0.5·N(λ_R*=0.769, s=1.0).
- **dmsr_cosmix_w0.8**: (1-0.8)·cosine + 0.8·N(λ_R*=0.769, s=1.0).

## Per-run Results (schedule × seed)

| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cosine_vp | 20260526 | 0.3830 | 0.4368 | 5.52 | 0.9574 | 0.0536 | 0.0583 | 0.0305 |
| 2 | dmsr_cosmix_w0.8 | 20260526 | 0.6169 | 0.6530 | 5.55 | 0.9570 | 0.0332 | 0.0635 | 0.0303 |
| 3 | dmsr_cosmix_w0.5 | 20260526 | 0.5300 | 0.5724 | 6.30 | 0.9538 | 0.0170 | 0.0583 | 0.0302 |
| 4 | linear_vp | 20260526 | 0.2346 | 0.2967 | 7.84 | 0.9537 | 0.1052 | 0.0597 | 0.0312 |
| 5 | dmsr_normal_s2.5 | 20260526 | 0.3746 | 0.4411 | 8.80 | 0.9524 | 0.1050 | 0.0675 | 0.0305 |
| 6 | uniform | 20260526 | 0.1286 | 0.1836 | 10.13 | 0.9487 | 0.0856 | 0.0555 | 0.0332 |
| 7 | hang_laplace_b0.5 | 20260526 | 0.9233 | 0.7851 | 10.75 | 0.9519 | 0.1598 | 0.1923 | 0.0298 |
| 8 | dmsr_normal_s6.0 | 20260526 | 0.1670 | 0.2297 | 10.96 | 0.9463 | 0.1278 | 0.0546 | 0.0330 |
| 9 | dmsr_normal_s4.0 | 20260526 | 0.2454 | 0.3110 | 11.71 | 0.9463 | 0.1052 | 0.0558 | 0.0317 |
| 10 | dmsr_normal_s0.8 | 20260526 | 0.7343 | 0.7614 | 13.01 | 0.9372 | 0.1182 | 0.2506 | 0.0298 |
| 11 | dmsr_normal_s1.5 | 20260526 | 0.5472 | 0.5963 | 18.72 | 0.9495 | 0.2150 | 0.1361 | 0.0300 |
| 12 | dmsr_laplace_b0.5 | 20260526 | 0.8114 | 0.8188 | 22.48 | 0.9460 | 0.2142 | 0.1747 | 0.0302 |
| 13 | dmsr_normal_s0.3 | 20260526 | 0.9567 | 0.9192 | 36.64 | 0.9479 | 0.3294 | 0.3626 | 0.0318 |

## Aggregated over seeds (n_seeds = 1)

생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.
FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.

| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |
|---:|---|---:|---|---|---|---|---|
| 1 | cosine_vp | 1 | 5.52 | 1.7587 | 0.886 | 0.705 | 0.0583 |
| 2 | dmsr_cosmix_w0.8 | 1 | 5.55 | 1.7269 | 0.889 | 0.675 | 0.0635 |
| 3 | dmsr_cosmix_w0.5 | 1 | 6.30 | 2.0276 | 0.910 | 0.721 | 0.0583 |
| 4 | linear_vp | 1 | 7.84 | 2.0102 | 0.899 | 0.709 | 0.0597 |
| 5 | dmsr_normal_s2.5 | 1 | 8.80 | 2.2997 | 0.880 | 0.647 | 0.0675 |
| 6 | uniform | 1 | 10.13 | 2.9417 | 0.873 | 0.599 | 0.0555 |
| 7 | hang_laplace_b0.5 | 1 | 10.75 | 1.7730 | 0.788 | 0.420 | 0.1923 |
| 8 | dmsr_normal_s6.0 | 1 | 10.96 | 2.7336 | 0.874 | 0.573 | 0.0546 |
| 9 | dmsr_normal_s4.0 | 1 | 11.71 | 3.2604 | 0.861 | 0.531 | 0.0558 |
| 10 | dmsr_normal_s0.8 | 1 | 13.01 | 2.1286 | 0.206 | 0.030 | 0.2506 |
| 11 | dmsr_normal_s1.5 | 1 | 18.72 | 3.1447 | 0.586 | 0.144 | 0.1361 |
| 12 | dmsr_laplace_b0.5 | 1 | 22.48 | 3.6050 | 0.371 | 0.039 | 0.1747 |
| 13 | dmsr_normal_s0.3 | 1 | 36.64 | 5.3462 | 0.526 | 0.077 | 0.3626 |

## Significance vs baseline (cosine_vp, metric = FID φ)

동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).

| schedule | n_pairs | mean_diff | improved | t p-value | status |
|---|---:|---:|:---:|---:|---|
| dmsr_cosmix_w0.5 | 1 | 0.779 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_cosmix_w0.8 | 1 | 0.029 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_laplace_b0.5 | 1 | 16.958 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.3 | 1 | 31.119 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s0.8 | 1 | 7.484 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s1.5 | 1 | 13.195 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s2.5 | 1 | 3.276 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s4.0 | 1 | 6.189 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| dmsr_normal_s6.0 | 1 | 5.436 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| hang_laplace_b0.5 | 1 | 5.229 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| linear_vp | 1 | 2.312 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |
| uniform | 1 | 4.606 | ✗ | nan | seed 부족 (paired 검정에는 num_seeds≥2 필요) |

## Per-λ MSE 분해 (transition 구간)

per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. **excess** = (해당 schedule − cosine_vp)의 같은 λ MSE 차이(공통 Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). **skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 `plots/per_lambda_decomposition.png`.

| schedule | transition excess (vs baseline) | transition skill | mean skill |
|---|---:|---:|---:|
| dmsr_normal_s0.8 | -0.0008 | 0.9702 | 0.7494 |
| hang_laplace_b0.5 | -0.0007 | 0.9702 | 0.8077 |
| dmsr_normal_s1.5 | -0.0005 | 0.9700 | 0.8639 |
| dmsr_laplace_b0.5 | -0.0003 | 0.9698 | 0.8253 |
| dmsr_cosmix_w0.5 | -0.0003 | 0.9698 | 0.9417 |
| dmsr_cosmix_w0.8 | -0.0003 | 0.9697 | 0.9365 |
| dmsr_normal_s2.5 | -0.0000 | 0.9695 | 0.9325 |
| cosine_vp | 0 (baseline) | 0.9695 | 0.9417 |
| linear_vp | +0.0007 | 0.9688 | 0.9403 |
| dmsr_normal_s4.0 | +0.0012 | 0.9683 | 0.9442 |
| dmsr_normal_s0.3 | +0.0012 | 0.9682 | 0.6374 |
| dmsr_normal_s6.0 | +0.0025 | 0.9670 | 0.9454 |
| uniform | +0.0027 | 0.9668 | 0.9445 |

## Interpretation Guide

- 이 단계의 목적은 생성 성능 주장이 아니라 **파이프라인·통계 틀 검증**이다.
  MNIST는 쉬운 데이터라 schedule 간 FID 차이가 작아도 실패가 아니다.
- 생성 품질은 φ-feature space에서 FID·KID·Precision/Recall/Density/Coverage로 잰다.
  FID는 품질·다양성을 뭉뚱그리므로, Precision(품질)과 Recall/Coverage(다양성)를 함께 보면 mode collapse 같은 실패를 분리해 볼 수 있다(Phase 3와 동일한 지표).
- KID는 표본이 적을 때 FID보다 신뢰성이 높고 부분표본 분산을 함께 준다.
- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며, full-range support 와의 균형이 중요하다(Phase 1 결론).
- `dmsr_normal_s*`는 폭을 좁게→넓게 sweep한다. narrow는 clean끝을 못 배워 붕괴하기 쉽고, 넓힐수록(또는 `dmsr_cosmix_w*`처럼 cosine을 섞어 full-range support를 확보할수록) 회복하는지를 본다. Precision–Recall 그림으로 붕괴(좌하단) 여부를 확인하라.
- λ_R*는 DMSR_φ(λ)의 수치 미분 peak에서 경험적으로 추정한다. 이 값은 Phase 3로 넘어가지 않으며 CIFAR에서 독립적으로 재추정한다.
- 유의성 검정은 seed가 부족하면(현재 1개) 건너뛴다. Phase 3에서 `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.
