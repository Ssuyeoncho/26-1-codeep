# Phase 1 Agent Brief: R-Transition Toy Experiment

## 1. Overall Research Goal

This project studies diffusion training noise schedules from the perspective of mode/class separability.

The central question is not simply:

> Is training near `logSNR = 0` useful?

The intended question is:

> Does performance improve when the training noise distribution covers the region where the data structure actually becomes ambiguous, as measured by a separability indicator `R`?

The project combines three viewpoints:

- Hang et al.: a training noise schedule can be interpreted as choosing a probability density over `lambda = log SNR`; concentrating probability near intermediate noise levels can improve diffusion training.
- Chen: the best noise schedule is data/task/resolution dependent, so a fixed `lambda = 0` center should not be treated as universally optimal.
- EDM: diffusion design choices should be separated. In later image experiments, the goal is to change the training noise distribution while keeping architecture, loss weighting, preconditioning, and sampler as controlled as possible.

Phase 1 is the first sanity-check stage. It does not evaluate image generation quality or FID. It tests the mechanism in the simplest possible setting.

## 2. Phase 1 Purpose

Phase 1 uses a 1D two-mode Gaussian mixture:

```text
p(x0) = 0.5 * N(-d, sigma0^2) + 0.5 * N(d, sigma0^2)
```

The purpose is to check whether the analytic separability indicator `R(sigma)` explains where denoising becomes difficult.

In the EDM/VE additive-noise view:

```text
x_sigma = x0 + sigma * epsilon
epsilon ~ N(0, 1)
```

For the 1D Gaussian mixture:

```text
R_EDM(sigma) = 2d / sqrt(sigma0^2 + sigma^2)
```

The transition region is defined by the slope magnitude with respect to `log sigma`:

```text
|dR / dlog(sigma)| = 2d * sigma^2 / (sigma0^2 + sigma^2)^(3/2)
```

The transition center is the maximum of this slope:

```text
sigma_R* = sqrt(2) * sigma0
```

The transition region is:

```text
T_R = {sigma : |dR/dlog(sigma)| >= rho * max_sigma |dR/dlog(sigma)|}
```

The default threshold is:

```text
rho = 0.5
```

## 3. What The Code Implements

Main file:

```text
phase1/phase1_toy_experiment.py
```

This is intentionally self-contained. It implements:

1. sampling from the 1D Gaussian mixture,
2. analytic `R_EDM(sigma)` and transition-region computation,
3. several training noise distributions,
4. a small MLP denoiser,
5. per-sigma denoising evaluation,
6. mode classification error from predicted clean signal,
7. transition coverage metrics,
8. CSV, Markdown, and PNG result outputs.

The model input is:

```text
(x_sigma, sigma)
```

The prediction target is:

```text
x0
```

The training loss is:

```text
MSE(pred_x0, x0)
```

This choice is acceptable for Phase 1 because the purpose is not to reproduce full EDM preconditioning, but to compare which training noise distributions help a fixed denoising model at different noise levels.

For fairer schedule comparison, the current implementation resets the random seed before training each schedule for a given seed index. This makes the MLP initialization identical across schedules for that seed. Multiple seeds can be requested with `--num-seeds`.

## 4. Bayes-Optimal Reference Denoiser

Because this is a 1D Gaussian mixture, the Bayes-optimal denoiser is analytic. The code uses it as a reference line to separate irreducible denoising difficulty from learned-model suboptimality.

For `sigma0 > 0`, the posterior mean includes both mode posterior mixing and within-mode Gaussian shrinkage:

```text
v = sigma0^2 + sigma^2
E[mode_mean | x_sigma, sigma] = d * tanh(d * x_sigma / v)
E[x0 | x_sigma, sigma]
  = (sigma0^2 / v) * x_sigma
    + (1 - sigma0^2 / v) * d * tanh(d * x_sigma / v)
```

This is more complete than the simplified expression:

```text
d * tanh(d * x_sigma / (sigma0^2 + sigma^2))
```

The simplified expression estimates the posterior mode center; the experiment target is the original clean sample `x0`, so the shrinkage term matters when `sigma0 > 0`.

## 5. Compared Training Noise Distributions

The code compares the following schedule families through one shared interface:

```text
sample_schedule(spec, n, config, device)
```

Schedules:

- `cosine_vp_as_ve`
  - Cosine VP-style schedule mapped into VE sigma using `sigma ~= exp(-lambda/2)`.
  - This is an approximate bridge between VP logSNR and EDM/VE sigma space.

- `linear_gamma_as_ve`
  - Chen-style simple linear signal schedule `gamma(t) = 1 - t`.
  - Converted to VE-like sigma by `sigma = sqrt((1 - gamma) / gamma)`.

- `hang_laplace_lambda_b0.5`
  - Hang-style midpoint-focused schedule.
  - Samples `lambda ~ Laplace(0, 0.5)` and maps by `sigma ~= exp(-lambda/2)`.

- `edm_lognormal`
  - EDM baseline.
  - Samples `log sigma ~ N(-1.2, 1.2^2)`.

- `r_normal_wide`
  - R-matched normal distribution centered at `log(sigma_R*)`, with wide scale.

- `r_normal_mid`
  - R-matched normal distribution centered at `log(sigma_R*)`, with moderate scale.

- `r_normal_narrow`
  - R-matched normal distribution centered at `log(sigma_R*)`, with narrow scale.

- `r_laplace_mid`
  - R-matched Laplace distribution centered at `log(sigma_R*)`.

The important conceptual comparison is not just which schedule has the lowest average MSE. The key comparison is whether schedules that cover the `R`-transition region improve denoising in that region without becoming too narrow to learn the full denoising range.

## 6. Evaluation Metrics

For each trained schedule, the code evaluates on a fixed log-spaced sigma grid.

Saved metrics include:

- `mean_mse`
  - Mean denoising MSE over the full sigma grid.

- `transition_mse`
  - Mean denoising MSE restricted to the `R`-transition region.

- `transition_bayes_mse`
  - Bayes-optimal denoising MSE restricted to the `R`-transition region.
  - This is the irreducible error for this toy distribution.

- `transition_excess_mse`
  - MSE between the learned MLP prediction and the Bayes-optimal denoiser inside the transition region.
  - This is the most important Phase 1 metric after adding the Bayes reference.

- `low_noise_mse`
  - Mean MSE below the transition region.

- `high_noise_mse`
  - Mean MSE above the transition region.

- `mean_mode_error`
  - Mode classification error using `pred_x0 >= 0`.

- `transition_mode_error`
  - Mode classification error inside the transition region.

- `coverage_m`
  - The probability that the training schedule samples sigma inside `T_R`.

```text
M = P_{sigma ~ p_train}(sigma in T_R)
```

- `expected_s_norm`
  - Expected R-slope under the training distribution, normalized by the maximum slope.

```text
S = E_{sigma ~ p_train}[|dR/dlog(sigma)|]
```

Important interpretation note:

`coverage_m` is not expected to be monotonically correlated with better performance. A schedule can put nearly all mass inside the transition region and still perform poorly if it neglects the broader denoising range. The intended claim is about adequate transition coverage with enough full-range support, not “more transition mass is always better.”

## 7. Output Structure

Each run writes results to:

```text
results/phase1/<timestamp>_phase1_toy_d<d>_s0<sigma0>/
```

Important files:

```text
config.json
schedules.json
train_history.json
metrics_summary.csv
per_sigma_metrics.csv
summary.md
plots/r_profile.png
plots/schedule_densities.png
plots/per_sigma_mse.png
plots/per_sigma_bayes_mse.png
plots/per_sigma_excess_mse.png
plots/per_sigma_mode_error.png
plots/coverage_vs_transition_mse.png
```

The most useful first files to inspect are:

```text
summary.md
metrics_summary.csv
plots/r_profile.png
plots/per_sigma_mse.png
plots/per_sigma_excess_mse.png
plots/coverage_vs_transition_mse.png
```

## 8. How To Run

Default run:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py
```

Fast smoke test:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --train-steps 100 --batch-size 128 --eval-batch-size 512
```

Other toy settings from the experiment plan:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --d 1.5 --sigma0 0.7
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --d 1.0 --sigma0 0.8
```

Multiple seeds:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 phase1/phase1_toy_experiment.py --num-seeds 3
```

## 9. What To Check In A Code Review

Please check the code against the Phase 1 purpose:

- Does the experiment truly isolate training noise distribution as the changed variable?
- Are all schedules trained with the same MLP, optimizer, batch size, training steps, and evaluation grid?
- For a fixed seed index, are all schedule models initialized from the same seed?
- Is the R-transition region computed from `|dR/dlog(sigma)|`, not from `R(sigma)` itself?
- Is `sigma_R* = sqrt(2) * sigma0` used as the analytic center for this toy EDM/VE setting?
- Is the Bayes-optimal denoiser used to report `transition_excess_mse`?
- Are schedule densities plotted in `log sigma` space?
- Is `transition_excess_mse` emphasized as the key Phase 1 result rather than FID?
- Are too-narrow schedules interpreted carefully as possible failures of full trajectory coverage?
- Are VP/logSNR schedules clearly marked as approximate mappings into VE sigma space?

## 10. Known Scope Limits

This Phase 1 code intentionally does not implement:

- image datasets,
- MNIST or CIFAR,
- FID,
- a U-Net,
- EDM preconditioning,
- EDM Heun sampling,
- feature-space empirical `R_phi(sigma)`.

Those belong to later phases.

Phase 1 is a toy mechanism test. Its job is to validate whether the `R`-transition idea is meaningful enough to motivate Phase 2 and Phase 3.
