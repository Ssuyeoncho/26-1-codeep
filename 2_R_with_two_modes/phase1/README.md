# Phase 1: VP DMSR Toy Experiment

이 단계의 목적은 이미지 생성 성능이나 FID를 비교하는 것이 아니라, 1D two-mode Gaussian mixture에서 DMSR 지표가 noise level에 따른 class/mode separability 변화를 올바르게 잡는지 확인하는 것입니다.

계획서 기준에 맞춰 Phase 1은 VP forward process와 epsilon-prediction을 사용합니다.

```text
p(x0) = 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2)
x_lambda = alpha_lambda x0 + sigma_lambda epsilon
alpha_lambda^2 = exp(lambda) / (1 + exp(lambda))
sigma_lambda^2 = 1 / (1 + exp(lambda))
```

해석적 DMSR은 다음과 같습니다.

```text
DMSR_VP(lambda) = 2 alpha_lambda d / sqrt(alpha_lambda^2 sigma0^2 + sigma_lambda^2)
lambda_R* = -log(2 sigma0^2)
```

## 실행

빠른 smoke test:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --preset smoke
```

계획서 기본 반복 수:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --preset full
```

`full` preset은 기본적으로 3 seeds를 실행해 schedule ranking의 안정성을 확인합니다.

문서의 세 toy 파라미터를 모두 실행:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --preset full --toy-params plan
```

cosine VP와 DMSR-wide의 차이를 더 엄밀히 보려면 seed 수를 늘립니다.

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --preset full --toy-params plan --num-seeds 5
```

기본 결과 위치는 실행 위치와 무관하게 `2_R_with_two_modes/results/phase1/`입니다.

## 저장 구조

실행 결과는 `2_R_with_two_modes/results/phase1/<run_id>/` 아래에 저장됩니다.

- `config.json`: 실험 하이퍼파라미터
- `schedules.json`: 비교한 training noise distribution 정의
- `metrics_summary.csv`: schedule별 요약 지표
- `metrics_aggregate.csv`: seed 평균/표준편차 집계 지표
- `per_lambda_metrics.csv`: lambda grid별 epsilon MSE, Bayes-optimal epsilon MSE, excess MSE, mode error
- `summary.md`: 결과를 바로 읽을 수 있는 Markdown 보고서
- `plots/dmsr_profile.png`: analytic `DMSR_VP(lambda)`와 transition region
- `plots/schedule_densities.png`: schedule별 `lambda` density
- `plots/per_lambda_mse.png`: lambda별 epsilon-prediction MSE
- `plots/per_lambda_bayes_mse.png`: lambda별 Bayes-optimal epsilon MSE
- `plots/per_lambda_excess_mse.png`: 학습된 MLP와 Bayes-optimal epsilon predictor 사이의 excess MSE
- `plots/per_lambda_mode_error.png`: predicted x0 기준 mode classification error
- `plots/coverage_vs_transition_mse.png`: transition mass `M`과 transition excess MSE 관계. 색은 전체 lambda 범위의 mean MSE

## 비교 Schedule

- `cosine_vp`: VP cosine schedule이 유도하는 lambda 분포
- `uniform_lambda`: 전체 lambda 범위를 균등하게 덮는 broad baseline
- `hang_laplace_lambda_b0.5`: Hang-style `lambda ~ Laplace(0, 0.5)`
- `dmsr_normal_wide_s1.5`: `lambda_R*` 중심 normal, s=1.5
- `dmsr_normal_mid_s0.8`: `lambda_R*` 중심 normal, s=0.8
- `dmsr_normal_narrow_s0.3`: `lambda_R*` 중심 normal, s=0.3
- `dmsr_laplace_b0.5`: Hang의 Laplace 형태를 `lambda_R*` 중심으로 이동

`linear_gamma`는 문서의 핵심 비교군은 아니므로 기본 실행에서는 제외합니다. 필요하면 `--include-linear-gamma`로 추가합니다.

## 해석

Phase 1에서 보고 싶은 신호는 `DMSR` 변화율이 큰 transition region을 적절히 덮는 schedule이 그 구간의 excess MSE를 낮추는지입니다. 여기서 excess MSE는 학습된 MLP의 epsilon 예측이 analytic Bayes-optimal epsilon predictor에서 얼마나 떨어져 있는지를 뜻합니다.

`coverage_m`이 높을수록 무조건 좋다는 주장은 아닙니다. 너무 좁게 transition region에만 몰린 schedule은 전체 denoising trajectory를 못 배워 transition 밖의 MSE가 커질 수 있습니다. 따라서 핵심 해석은 “충분한 transition coverage와 full-range support 사이의 균형”입니다.

현재 Phase 1에서 안전하게 주장할 수 있는 것은 다음입니다.

- `dmsr_normal_narrow_s0.3`처럼 T_R에 과도하게 집중하면 transition 또는 전체 denoising 성능이 불안정해질 수 있습니다.
- `uniform_lambda`는 broad support만으로 충분한지 확인하는 반례 baseline입니다. 일부 toy 세팅에서 DMSR-targeted schedule이 uniform보다 낮은 transition excess MSE를 보이면, T_R 근방 density 자체가 도움이 된다는 partial evidence로 해석할 수 있습니다.
- `cosine_vp`와 `dmsr_normal_wide_s1.5`의 차이는 seed 표준편차가 겹치면 우열을 주장하지 않습니다. 이 경우 “비슷하다”로 정리하고 Phase 2에서 더 큰 데이터/모델로 재검증합니다.

이 단계는 FID를 주장하지 않으며, Phase 2 MNIST와 Phase 3 CIFAR에서 사용할 empirical DMSR 계산 및 schedule 설계 근거를 만드는 역할입니다.
