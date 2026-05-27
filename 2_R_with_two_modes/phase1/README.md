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
lambda_R* = -log(sigma0^2)
```

## 실행

계획서 기본 반복 수:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py
```

빠른 smoke test:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --train-steps 20 --batch-size 64 --eval-batch-size 128 --eval-grid-size 12
```

문서의 다른 toy 파라미터:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --d 1.5 --sigma0 0.7
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --d 1.0 --sigma0 0.8
```

## 저장 구조

실행 결과는 `results/phase1/<run_id>/` 아래에 저장됩니다.

- `config.json`: 실험 하이퍼파라미터
- `schedules.json`: 비교한 training noise distribution 정의
- `metrics_summary.csv`: schedule별 요약 지표
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
- `linear_gamma`: Chen의 `gamma(t)=1-t` baseline
- `hang_laplace_lambda_b0.5`: Hang-style `lambda ~ Laplace(0, 0.5)`
- `dmsr_normal_wide_s1.5`: `lambda_R*` 중심 normal, s=1.5
- `dmsr_normal_mid_s0.8`: `lambda_R*` 중심 normal, s=0.8
- `dmsr_normal_narrow_s0.3`: `lambda_R*` 중심 normal, s=0.3
- `dmsr_laplace_b0.5`: Hang의 Laplace 형태를 `lambda_R*` 중심으로 이동

## 해석

Phase 1에서 보고 싶은 신호는 `DMSR` 변화율이 큰 transition region을 적절히 덮는 schedule이 그 구간의 excess MSE를 낮추는지입니다. 여기서 excess MSE는 학습된 MLP의 epsilon 예측이 analytic Bayes-optimal epsilon predictor에서 얼마나 떨어져 있는지를 뜻합니다.

`coverage_m`이 높을수록 무조건 좋다는 주장은 아닙니다. 너무 좁게 transition region에만 몰린 schedule은 전체 denoising trajectory를 못 배워 transition 밖의 MSE가 커질 수 있습니다. 따라서 핵심 해석은 “충분한 transition coverage와 full-range support 사이의 균형”입니다.

이 단계는 FID를 주장하지 않으며, Phase 2 MNIST와 Phase 3 CIFAR에서 사용할 empirical DMSR 계산 및 schedule 설계 근거를 만드는 역할입니다.
