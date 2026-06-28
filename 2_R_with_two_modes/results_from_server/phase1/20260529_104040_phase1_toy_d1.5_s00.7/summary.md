# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic VP DMSR-transition region improve epsilon prediction where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=1.5, sigma0=0.7
- lambda range: [-10.0, 10.0]
- rho threshold: 0.5
- lambda_R*: 0.020203
- transition region: [-2.458115, 1.992998]
- train steps per schedule: 10000
- batch size: 512
- seeds per schedule: 1
- optional linear gamma baseline: False

## Schedules

- **cosine_vp**: Cosine VP schedule induced density over lambda.
- **hang_laplace_lambda_b0.5**: Hang-style Laplace around lambda=0.
- **dmsr_normal_wide_s1.5**: DMSR-centered normal, wide s=1.5.
- **dmsr_normal_mid_s0.8**: DMSR-centered normal, middle s=0.8.
- **dmsr_normal_narrow_s0.3**: DMSR-centered normal, narrow s=0.3.
- **dmsr_laplace_b0.5**: Hang-style Laplace shifted to lambda_R*.

## Main Results

| rank | schedule | seed | M coverage | S norm | transition MSE | Bayes transition MSE | transition excess MSE | mean MSE | transition mode error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cosine_vp | 20260526 | 0.5940 | 0.5770 | 0.577623 | 0.576468 | 0.000954 | 0.533976 | 0.1522 |
| 2 | dmsr_normal_wide_s1.5 | 20260526 | 0.8569 | 0.7748 | 0.579143 | 0.576468 | 0.002206 | 0.557141 | 0.1519 |
| 3 | dmsr_normal_mid_s0.8 | 20260526 | 0.9925 | 0.9118 | 0.579455 | 0.576468 | 0.002771 | 0.617221 | 0.1519 |
| 4 | dmsr_laplace_b0.5 | 20260526 | 0.9871 | 0.9350 | 0.580335 | 0.576468 | 0.003748 | 0.613316 | 0.1521 |
| 5 | hang_laplace_lambda_b0.5 | 20260526 | 0.9873 | 0.9352 | 0.580388 | 0.576468 | 0.003824 | 0.612592 | 0.1521 |
| 6 | dmsr_normal_narrow_s0.3 | 20260526 | 1.0000 | 0.9855 | 0.600594 | 0.576468 | 0.024423 | 0.644360 | 0.1520 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible epsilon-prediction error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal epsilon predictor.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the DMSR/mode-separability mechanism before MNIST/CIFAR phases.
