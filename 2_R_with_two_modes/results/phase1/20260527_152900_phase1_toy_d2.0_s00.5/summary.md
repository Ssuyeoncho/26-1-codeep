# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic VP DMSR-transition region improve epsilon prediction where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=2.0, sigma0=0.5
- lambda range: [-10.0, 10.0]
- rho threshold: 0.5
- lambda_R*: 1.386294
- transition region: [-1.782946, 2.663166]
- train steps per schedule: 20
- batch size: 64
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
| 1 | cosine_vp | 20260526 | 0.5862 | 0.6241 | 0.836111 | 0.586434 | 0.335638 | 0.689022 | 0.0664 |
| 2 | linear_gamma | 20260526 | 0.7904 | 0.7883 | 0.865736 | 0.586434 | 0.353927 | 0.759191 | 0.0625 |
| 3 | hang_laplace_lambda_b0.5 | 20260526 | 0.9834 | 0.9653 | 0.865783 | 0.586434 | 0.361421 | 0.952440 | 0.0547 |
| 4 | dmsr_normal_wide_s1.5 | 20260526 | 0.7859 | 0.7903 | 0.932566 | 0.586434 | 0.412682 | 0.906899 | 0.0586 |
| 5 | dmsr_normal_mid_s0.8 | 20260526 | 0.9445 | 0.9178 | 0.951139 | 0.586434 | 0.430755 | 0.962810 | 0.0586 |
| 6 | dmsr_laplace_b0.5 | 20260526 | 0.9605 | 0.9392 | 0.954862 | 0.586434 | 0.434511 | 0.969078 | 0.0586 |
| 7 | dmsr_normal_narrow_s0.3 | 20260526 | 1.0000 | 0.9864 | 0.958255 | 0.586434 | 0.439583 | 0.983039 | 0.0586 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible epsilon-prediction error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal epsilon predictor.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the DMSR/mode-separability mechanism before MNIST/CIFAR phases.
