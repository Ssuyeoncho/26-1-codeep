# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic VP DMSR-transition region improve epsilon prediction where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=1.0, sigma0=0.8
- lambda range: [-10.0, 10.0]
- rho threshold: 0.5
- lambda_R*: -0.246860
- transition region: [-2.723181, 1.722931]
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
| 1 | cosine_vp | 20260526 | 0.5863 | 0.5708 | 0.500224 | 0.499021 | 0.000992 | 0.524117 | 0.2604 |
| 2 | dmsr_normal_wide_s1.5 | 20260526 | 0.8565 | 0.7748 | 0.501231 | 0.499021 | 0.002133 | 0.580888 | 0.2609 |
| 3 | hang_laplace_lambda_b0.5 | 20260526 | 0.9823 | 0.9238 | 0.501723 | 0.499021 | 0.002466 | 0.665566 | 0.2611 |
| 4 | dmsr_normal_mid_s0.8 | 20260526 | 0.9924 | 0.9118 | 0.502245 | 0.499021 | 0.002879 | 0.654933 | 0.2610 |
| 5 | dmsr_laplace_b0.5 | 20260526 | 0.9871 | 0.9350 | 0.502280 | 0.499021 | 0.002959 | 0.623144 | 0.2602 |
| 6 | dmsr_normal_narrow_s0.3 | 20260526 | 1.0000 | 0.9855 | 0.525692 | 0.499021 | 0.026050 | 0.631635 | 0.2615 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible epsilon-prediction error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal epsilon predictor.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the DMSR/mode-separability mechanism before MNIST/CIFAR phases.
