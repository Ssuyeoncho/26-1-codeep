# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic R-transition region improve denoising exactly where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=2.0, sigma0=0.5
- sigma range: [0.01, 20.0]
- rho threshold: 0.5
- sigma_R*: 0.353553
- transition region: [0.263906, 2.439167]
- train steps per schedule: 20
- batch size: 64

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
| 1 | linear_gamma_as_ve | 0.7895 | 1.0239 | 0.929365 | 2.966482 | 0.0564 |
| 2 | cosine_vp_as_ve | 0.5886 | 0.8129 | 0.972825 | 1.698575 | 0.0543 |
| 3 | hang_laplace_lambda_b0.5 | 0.9830 | 1.2535 | 0.987389 | 13.838995 | 0.0521 |
| 4 | r_normal_narrow | 0.9283 | 0.9926 | 1.099096 | 18.764241 | 0.0562 |
| 5 | edm_lognormal | 0.5033 | 0.7174 | 1.102685 | 16.751487 | 0.0525 |
| 6 | r_laplace_mid | 0.7328 | 0.9246 | 1.279412 | 20.146569 | 0.0645 |
| 7 | r_normal_wide | 0.6100 | 0.8448 | 1.310565 | 21.382302 | 0.0515 |
| 8 | r_normal_mid | 0.7414 | 0.9591 | 1.565721 | 27.410140 | 0.0615 |

## Interpretation Guide

- If a schedule has high M but worse transition MSE, it may be too narrow or may under-cover the full denoising range.
- If R-matched mid/wide schedules improve transition MSE, Phase 1 supports the claim that R-transition is a useful schedule-design target.
- This is not an image-generation or FID experiment; it validates the R/mode-separability mechanism before MNIST/CIFAR phases.
