# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic R-transition region improve denoising exactly where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=2.0, sigma0=0.5
- sigma range: [0.01, 20.0]
- rho threshold: 0.5
- sigma_R*: 0.707107
- transition region: [0.263906, 2.439167]
- train steps per schedule: 30
- batch size: 96
- seeds per schedule: 1

## Schedules

- **cosine_vp_as_ve**: Cosine VP density mapped with sigma ~= exp(-lambda/2).
- **linear_gamma_as_ve**: Chen-style gamma(t)=1-t mapped to VE sigma.
- **hang_laplace_lambda_b0.5**: Hang-style Laplace around lambda=0.
- **edm_lognormal**: EDM baseline log sigma ~ N(-1.2, 1.2^2).
- **r_normal_wide**: R-matched normal, broad coverage.
- **r_normal_mid**: R-matched normal, moderate focus.
- **r_normal_narrow**: R-matched normal, narrow focus.
- **r_laplace_mid**: R-matched Laplace in log sigma.

## Main Results

| rank | schedule | seed | M coverage | S norm | transition MSE | Bayes transition MSE | transition excess MSE | mean MSE | transition mode error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cosine_vp_as_ve | 20260526 | 0.5869 | 0.5742 | 0.746230 | 0.653442 | 0.091885 | 1.497768 | 0.0533 |
| 2 | linear_gamma_as_ve | 20260526 | 0.7904 | 0.7242 | 0.754312 | 0.653442 | 0.100338 | 1.552671 | 0.0520 |
| 3 | r_normal_wide | 20260526 | 0.7788 | 0.7182 | 0.832289 | 0.653442 | 0.184159 | 2.346311 | 0.0481 |
| 4 | edm_lognormal | 20260526 | 0.5041 | 0.5081 | 0.887786 | 0.653442 | 0.239322 | 3.428967 | 0.0508 |
| 5 | r_normal_mid | 20260526 | 0.9830 | 0.8924 | 0.902013 | 0.653442 | 0.246384 | 6.575655 | 0.0508 |
| 6 | r_laplace_mid | 20260526 | 0.9123 | 0.8451 | 0.906399 | 0.653442 | 0.252599 | 3.676509 | 0.0510 |
| 7 | r_normal_narrow | 20260526 | 1.0000 | 0.9747 | 0.922577 | 0.653442 | 0.269680 | 8.270108 | 0.0510 |
| 8 | hang_laplace_lambda_b0.5 | 20260526 | 0.9833 | 0.8863 | 0.924911 | 0.653442 | 0.270746 | 5.870581 | 0.0510 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible denoising error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal denoiser.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the R/mode-separability mechanism before MNIST/CIFAR phases.
