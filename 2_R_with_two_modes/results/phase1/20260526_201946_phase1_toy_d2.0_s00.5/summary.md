# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic R-transition region improve denoising exactly where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=2.0, sigma0=0.5
- sigma range: [0.01, 20.0]
- rho threshold: 0.5
- sigma_R*: 0.353553
- transition region: [0.263906, 2.439167]
- train steps per schedule: 1000
- batch size: 512

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

| rank | schedule | M coverage | S norm | transition MSE | mean MSE | transition mode error |
|---:|---|---:|---:|---:|---:|---:|
| 1 | hang_laplace_lambda_b0.5 | 0.9833 | 1.2535 | 0.675636 | 2.097746 | 0.0551 |
| 2 | linear_gamma_as_ve | 0.7920 | 1.0262 | 0.678817 | 1.260698 | 0.0534 |
| 3 | cosine_vp_as_ve | 0.5884 | 0.8142 | 0.685157 | 1.228719 | 0.0540 |
| 4 | edm_lognormal | 0.5054 | 0.7191 | 0.692758 | 1.265692 | 0.0548 |
| 5 | r_laplace_mid | 0.7322 | 0.9239 | 0.693605 | 1.440794 | 0.0529 |
| 6 | r_normal_wide | 0.6127 | 0.8466 | 0.695815 | 1.747593 | 0.0554 |
| 7 | r_normal_mid | 0.7426 | 0.9590 | 0.737285 | 8.476515 | 0.0548 |
| 8 | r_normal_narrow | 0.9277 | 0.9931 | 0.893878 | 13.011081 | 0.0546 |

## Interpretation Guide

- If a schedule has high M but worse transition MSE, it may be too narrow or may under-cover the full denoising range.
- If R-matched mid/wide schedules improve transition MSE, Phase 1 supports the claim that R-transition is a useful schedule-design target.
- This is not an image-generation or FID experiment; it validates the R/mode-separability mechanism before MNIST/CIFAR phases.
