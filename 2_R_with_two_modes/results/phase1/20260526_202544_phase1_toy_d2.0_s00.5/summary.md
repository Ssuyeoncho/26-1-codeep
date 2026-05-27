# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic R-transition region improve denoising exactly where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=2.0, sigma0=0.5
- sigma range: [0.01, 20.0]
- rho threshold: 0.5
- sigma_R*: 0.707107
- transition region: [0.263906, 2.439167]
- train steps per schedule: 40
- batch size: 128

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
| 1 | linear_gamma_as_ve | 0.7908 | 0.7245 | 0.752603 | 1.477950 | 0.0565 |
| 2 | r_laplace_mid | 0.9121 | 0.8457 | 0.779100 | 2.991689 | 0.0562 |
| 3 | r_normal_wide | 0.7799 | 0.7181 | 0.782824 | 2.173774 | 0.0559 |
| 4 | edm_lognormal | 0.5006 | 0.5057 | 0.783191 | 1.773528 | 0.0581 |
| 5 | cosine_vp_as_ve | 0.5868 | 0.5744 | 0.790845 | 1.396418 | 0.0532 |
| 6 | hang_laplace_lambda_b0.5 | 0.9830 | 0.8864 | 0.836207 | 6.639981 | 0.0554 |
| 7 | r_normal_mid | 0.9820 | 0.8922 | 0.954606 | 11.698295 | 0.0573 |
| 8 | r_normal_narrow | 1.0000 | 0.9747 | 0.976565 | 11.462525 | 0.0509 |

## Interpretation Guide

- If a schedule has high M but worse transition MSE, it may be too narrow or may under-cover the full denoising range.
- If R-matched mid/wide schedules improve transition MSE, Phase 1 supports the claim that R-transition is a useful schedule-design target.
- This is not an image-generation or FID experiment; it validates the R/mode-separability mechanism before MNIST/CIFAR phases.
