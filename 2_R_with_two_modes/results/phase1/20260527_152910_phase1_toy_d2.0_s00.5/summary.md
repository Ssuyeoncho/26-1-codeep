# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic VP DMSR-transition region improve epsilon prediction where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=2.0, sigma0=0.5
- lambda range: [-10.0, 10.0]
- rho threshold: 0.5
- lambda_R*: 1.386294
- transition region: [-1.782946, 2.663166]
- train steps per schedule: 10000
- batch size: 512
- seeds per schedule: 1

## Schedules

- **cosine_vp**: Cosine VP schedule induced density over lambda.
- **linear_gamma**: Chen-style gamma(t)=1-t baseline in VP lambda space.
- **hang_laplace_lambda_b0.5**: Hang-style Laplace around lambda=0.
- **dmsr_normal_wide_s1.5**: DMSR-centered normal, wide s=1.5.
- **dmsr_normal_mid_s0.8**: DMSR-centered normal, middle s=0.8.
- **dmsr_normal_narrow_s0.3**: DMSR-centered normal, narrow s=0.3.
- **dmsr_laplace_b0.5**: Hang-style Laplace shifted to lambda_R*.

## Main Results

| rank | schedule | seed | M coverage | S norm | transition MSE | Bayes transition MSE | transition excess MSE | mean MSE | transition mode error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cosine_vp | 20260526 | 0.5873 | 0.6253 | 0.550640 | 0.548257 | 0.002153 | 0.504716 | 0.0507 |
| 2 | linear_gamma | 20260526 | 0.7911 | 0.7892 | 0.551296 | 0.548257 | 0.003310 | 0.511654 | 0.0507 |
| 3 | dmsr_normal_wide_s1.5 | 20260526 | 0.7872 | 0.7918 | 0.552393 | 0.548257 | 0.003891 | 0.627568 | 0.0508 |
| 4 | dmsr_normal_mid_s0.8 | 20260526 | 0.9451 | 0.9186 | 0.553071 | 0.548257 | 0.004283 | 1.818648 | 0.0515 |
| 5 | hang_laplace_lambda_b0.5 | 20260526 | 0.9836 | 0.9650 | 0.554361 | 0.548257 | 0.005681 | 0.544735 | 0.0510 |
| 6 | dmsr_laplace_b0.5 | 20260526 | 0.9605 | 0.9401 | 0.554833 | 0.548257 | 0.006051 | 0.688307 | 0.0508 |
| 7 | dmsr_normal_narrow_s0.3 | 20260526 | 1.0000 | 0.9866 | 0.573438 | 0.548257 | 0.023314 | 0.575708 | 0.0506 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible epsilon-prediction error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal epsilon predictor.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the DMSR/mode-separability mechanism before MNIST/CIFAR phases.
