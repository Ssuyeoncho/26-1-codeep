# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic R-transition region improve denoising exactly where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=2.0, sigma0=0.5
- sigma range: [0.01, 20.0]
- rho threshold: 0.5
- sigma_R*: 0.707107
- transition region: [0.263906, 2.439167]
- train steps per schedule: 2500
- batch size: 512
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
| 1 | r_normal_mid | 20260526 | 0.9826 | 0.8928 | 0.680657 | 0.678558 | 0.002396 | 3.844756 | 0.0542 |
| 2 | r_laplace_mid | 20260526 | 0.9122 | 0.8456 | 0.681835 | 0.678558 | 0.003564 | 1.261089 | 0.0544 |
| 3 | r_normal_wide | 20260526 | 0.7798 | 0.7190 | 0.682128 | 0.678558 | 0.003620 | 1.290503 | 0.0543 |
| 4 | edm_lognormal | 20260526 | 0.5038 | 0.5073 | 0.681951 | 0.678558 | 0.003712 | 1.256612 | 0.0543 |
| 5 | linear_gamma_as_ve | 20260526 | 0.7912 | 0.7250 | 0.686573 | 0.678558 | 0.008576 | 1.252211 | 0.0544 |
| 6 | cosine_vp_as_ve | 20260526 | 0.5874 | 0.5754 | 0.687016 | 0.678558 | 0.009425 | 1.242028 | 0.0546 |
| 7 | hang_laplace_lambda_b0.5 | 20260526 | 0.9836 | 0.8864 | 0.692595 | 0.678558 | 0.014861 | 1.510579 | 0.0547 |
| 8 | r_normal_narrow | 20260526 | 1.0000 | 0.9748 | 0.734734 | 0.678558 | 0.057826 | 5.900909 | 0.0544 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible denoising error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal denoiser.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the R/mode-separability mechanism before MNIST/CIFAR phases.
