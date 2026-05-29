# Phase 1 Agent Brief: VP DMSR-Transition Toy Experiment

## 1. Overall Research Goal

This project studies diffusion training noise schedules from the perspective of data-dependent mode/class separability.

The intended question is not:

> Is training near logSNR = 0 always best?

The intended question is:

> Does performance improve when the training noise distribution covers the region where the data structure actually becomes ambiguous, as measured by DMSR?

Phase 1 is a mechanism check. It does not evaluate image generation quality or FID.

## 2. Phase 1 Purpose

Phase 1 uses a 1D two-mode Gaussian mixture:

```text
p(x0) = 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2)
```

The forward process follows the experiment plan's VP setting:

```text
x_lambda = alpha_lambda x0 + sigma_lambda epsilon
epsilon ~ N(0, 1)
alpha_lambda^2 = exp(lambda) / (1 + exp(lambda))
sigma_lambda^2 = 1 / (1 + exp(lambda))
lambda = log SNR
```

For this toy distribution:

```text
DMSR_VP(lambda)
  = 2 alpha_lambda d / sqrt(alpha_lambda^2 sigma0^2 + sigma_lambda^2)
  = 2d exp(lambda / 2) / sqrt(1 + sigma0^2 exp(lambda))

|dDMSR / dlambda|
  = d exp(lambda / 2) / (1 + sigma0^2 exp(lambda))^(3/2)

lambda_R* = -log(2 sigma0^2)
```

The transition region is:

```text
T_R = {lambda : |dDMSR/dlambda| >= rho * max_lambda |dDMSR/dlambda|}
```

The default threshold is `rho = 0.5`.

## 3. What The Code Implements

Main file:

```text
phase1/phase1_toy_experiment.py
```

It implements:

1. sampling from the 1D Gaussian mixture,
2. analytic `DMSR_VP(lambda)` and transition-region computation,
3. schedule samplers in lambda space,
4. a small MLP epsilon-prediction denoiser,
5. per-lambda epsilon MSE evaluation,
6. Bayes-optimal epsilon prediction for reference,
7. mode classification error from reconstructed `pred_x0`,
8. transition coverage metrics `M` and `S`,
9. CSV, Markdown, and PNG result outputs.

The model input is:

```text
(x_lambda, lambda)
```

The prediction target is:

```text
epsilon
```

The training loss is:

```text
MSE(pred_epsilon, epsilon)
```

Loss weighting is fixed across schedules. The independent variable is only `p_train(lambda)`.

## 4. Compared Training Noise Distributions

- `cosine_vp`: VP cosine schedule induced distribution over lambda.
- `hang_laplace_lambda_b0.5`: Hang-style `lambda ~ Laplace(0, 0.5)`.
- `dmsr_normal_wide_s1.5`: DMSR-centered `N(lambda_R*, 1.5^2)`.
- `dmsr_normal_mid_s0.8`: DMSR-centered `N(lambda_R*, 0.8^2)`.
- `dmsr_normal_narrow_s0.3`: DMSR-centered `N(lambda_R*, 0.3^2)`.
- `dmsr_laplace_b0.5`: Laplace distribution centered at `lambda_R*`.

`linear_gamma` is available as an optional diagnostic baseline through
`--include-linear-gamma`, but it is not part of the default plan comparison.

The key comparison is whether schedules that cover `T_R` improve denoising in that region without becoming too narrow to learn the full denoising range.

## 5. Evaluation Metrics

Saved metrics include:

- `mean_mse`: mean epsilon-prediction MSE over the full lambda grid.
- `transition_mse`: epsilon-prediction MSE restricted to `T_R`.
- `transition_bayes_mse`: Bayes-optimal epsilon-prediction MSE in `T_R`.
- `transition_excess_mse`: MSE between learned epsilon prediction and Bayes-optimal epsilon prediction in `T_R`.
- `low_noise_mse`: mean MSE where lambda is above the transition region.
- `high_noise_mse`: mean MSE where lambda is below the transition region.
- `mean_mode_error`: mode classification error using reconstructed `pred_x0 >= 0`.
- `transition_mode_error`: mode error inside `T_R`.
- `coverage_m`: probability that `p_train(lambda)` samples inside `T_R`.
- `expected_s_norm`: expected DMSR slope under `p_train(lambda)`, normalized by the maximum slope.

Important interpretation note:

`coverage_m` is not expected to be monotonically correlated with better performance. A schedule can put nearly all mass inside the transition region and still perform poorly if it neglects the broader denoising range.

## 6. Output Structure

Each run writes results to:

```text
2_R_with_two_modes/results/phase1/<timestamp>_phase1_toy_d<d>_s0<sigma0>/
```

Important files:

```text
config.json
schedules.json
train_history.json
metrics_summary.csv
per_lambda_metrics.csv
summary.md
plots/dmsr_profile.png
plots/schedule_densities.png
plots/per_lambda_mse.png
plots/per_lambda_bayes_mse.png
plots/per_lambda_excess_mse.png
plots/per_lambda_mode_error.png
plots/coverage_vs_transition_mse.png
```

## 7. Recommended Commands

Fast smoke test:

```text
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --preset smoke
```

Full default run:

```text
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --preset full
```

Run all planned toy parameter settings:

```text
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --preset full --toy-params plan
```

## 8. Sanity Checklist

- Is the forward process VP, not VE additive noise?
- Is the model predicting epsilon, not x0?
- Is `lambda_R* = -log(2 sigma0^2)` used as the analytic toy center?
- Is `T_R` computed from `|dDMSR/dlambda|`, not from DMSR itself?
- Are schedule densities plotted in lambda space?
- Are all schedules compared with the same model, optimizer, steps, batch size, and loss weighting?
- Are results interpreted as a mechanism check, not a FID claim?
