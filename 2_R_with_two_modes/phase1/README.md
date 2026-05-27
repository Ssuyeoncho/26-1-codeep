# Phase 1: 1D Gaussian Mixture Toy Experiment

이 단계의 목적은 이미지 생성 성능을 바로 비교하는 것이 아니라, `R(sigma)`가 mode separability를 설명하고 `R`이 빠르게 변하는 transition region이 denoising 난이도와 연결되는지 확인하는 것입니다.

## 실행

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py
```

빠른 확인용:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --train-steps 100 --batch-size 128 --eval-batch-size 512
```

문서의 다른 toy 파라미터는 다음처럼 실행합니다.

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --d 1.5 --sigma0 0.7
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --d 1.0 --sigma0 0.8
```

## 저장 구조

실행 결과는 `results/phase1/<run_id>/` 아래에 저장됩니다.

- `config.json`: 실험 하이퍼파라미터
- `schedules.json`: 비교한 training noise distribution 정의
- `metrics_summary.csv`: schedule별 요약 지표
- `per_sigma_metrics.csv`: sigma grid별 denoising MSE, Bayes-optimal MSE, excess MSE, mode error
- `summary.md`: 결과를 바로 읽을 수 있는 Markdown 보고서
- `plots/r_profile.png`: analytic `R(sigma)`와 transition region
- `plots/schedule_densities.png`: schedule별 `log sigma` density
- `plots/per_sigma_mse.png`: sigma별 denoising MSE
- `plots/per_sigma_bayes_mse.png`: sigma별 Bayes-optimal denoising MSE
- `plots/per_sigma_excess_mse.png`: 학습된 MLP와 Bayes-optimal denoiser 사이의 excess MSE
- `plots/per_sigma_mode_error.png`: sigma별 mode classification error
- `plots/coverage_vs_transition_mse.png`: transition mass `M`과 transition excess MSE 관계. 색은 전체 sigma 범위의 mean MSE

## 비교 Schedule

- `cosine_vp_as_ve`: cosine VP schedule을 `sigma ~= exp(-lambda/2)`로 VE/EDM 관점에 매핑
- `linear_gamma_as_ve`: Chen의 `gamma(t)=1-t` baseline을 VE sigma로 매핑
- `hang_laplace_lambda_b0.5`: Hang-style `lambda ~ Laplace(0, 0.5)`
- `edm_lognormal`: EDM baseline `log sigma ~ N(-1.2, 1.2^2)`
- `r_normal_wide/mid/narrow`: `log sigma_R*` 중심의 normal 분포, 폭만 변경. 여기서 analytic toy 기준 `sigma_R* = sqrt(2) * sigma0`
- `r_laplace_mid`: `log sigma_R*` 중심의 Laplace 분포

## 해석

Phase 1에서 가장 보고 싶은 신호는 `R`-transition region을 적절히 덮는 schedule이 transition excess MSE를 낮추는지입니다. 여기서 excess MSE는 학습된 MLP가 analytic Bayes-optimal denoiser에서 얼마나 떨어져 있는지를 뜻합니다.

주의할 점은 `coverage_m`이 높을수록 무조건 좋다는 주장이 아니라는 것입니다. 너무 좁게 transition region에만 몰린 schedule은 전체 denoising range를 못 배워 mean MSE가 나빠질 수 있습니다. 따라서 핵심 해석은 “충분한 transition coverage와 full-range support 사이의 균형”입니다.

공정한 비교를 위해 같은 seed index에서는 각 schedule 훈련 직전에 동일 seed로 모델을 초기화합니다. 여러 seed를 돌리고 싶으면 `--num-seeds`를 사용합니다.

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --num-seeds 3
```

이 단계는 FID를 주장하지 않으며, Phase 2 MNIST와 Phase 3 CIFAR에서 사용할 schedule 설계 근거를 만드는 역할입니다.
