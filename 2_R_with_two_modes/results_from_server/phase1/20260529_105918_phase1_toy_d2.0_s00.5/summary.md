# Phase 1 Toy Experiment Summary

## Purpose

This run tests whether training noise distributions that cover the analytic VP DMSR-transition region improve epsilon prediction where two Gaussian mixture modes become hard to separate.

## Configuration

- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d=2.0, sigma0=0.5
- lambda range: [-10.0, 10.0]
- rho threshold: 0.5
- lambda_R*: 0.693147
- transition region: [-1.782946, 2.663166]
- train steps per schedule: 10000
- batch size: 512
- seeds per schedule: 3
- optional linear gamma baseline: False

## Schedules

- **cosine_vp**: Cosine VP schedule induced density over lambda.
- **uniform_lambda**: Flat broad baseline over the full lambda range.
- **hang_laplace_lambda_b0.5**: Hang-style Laplace around lambda=0.
- **dmsr_normal_wide_s1.5**: DMSR-centered normal, wide s=1.5.
- **dmsr_normal_mid_s0.8**: DMSR-centered normal, middle s=0.8.
- **dmsr_normal_narrow_s0.3**: DMSR-centered normal, narrow s=0.3.
- **dmsr_laplace_b0.5**: Hang-style Laplace shifted to lambda_R*.

## Main Results

Aggregated across seeds. Standard deviations are sample std when `num_seeds > 1`.

| rank | schedule | seeds | M coverage | S norm | transition excess MSE | mean MSE | transition mode error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | dmsr_normal_wide_s1.5 | 3 | 0.8564 +/- 0.0006 | 0.7741 +/- 0.0006 | 0.003214 +/- 0.000108 | 0.532056 +/- 0.022382 | 0.0499 +/- 0.0008 |
| 2 | cosine_vp | 3 | 0.5884 +/- 0.0013 | 0.5746 +/- 0.0009 | 0.003289 +/- 0.001382 | 0.503317 +/- 0.002338 | 0.0498 +/- 0.0009 |
| 3 | uniform_lambda | 3 | 0.2233 +/- 0.0007 | 0.2595 +/- 0.0004 | 0.003633 +/- 0.000701 | 0.501810 +/- 0.001768 | 0.0499 +/- 0.0010 |
| 4 | dmsr_normal_mid_s0.8 | 3 | 0.9923 +/- 0.0002 | 0.9115 +/- 0.0003 | 0.003690 +/- 0.001449 | 0.850808 +/- 0.296224 | 0.0500 +/- 0.0010 |
| 5 | dmsr_laplace_b0.5 | 3 | 0.9868 +/- 0.0003 | 0.9349 +/- 0.0002 | 0.004079 +/- 0.001158 | 0.673593 +/- 0.159866 | 0.0500 +/- 0.0011 |
| 6 | hang_laplace_lambda_b0.5 | 3 | 0.9835 +/- 0.0002 | 0.8864 +/- 0.0001 | 0.005843 +/- 0.002039 | 0.590945 +/- 0.073130 | 0.0501 +/- 0.0009 |
| 7 | dmsr_normal_narrow_s0.3 | 3 | 1.0000 +/- 0.0000 | 0.9855 +/- 0.0001 | 0.054526 +/- 0.013414 | 0.675633 +/- 0.074662 | 0.0499 +/- 0.0007 |

## Per-Seed Results

| rank | schedule | seed | M coverage | S norm | transition MSE | Bayes transition MSE | transition excess MSE | mean MSE | transition mode error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cosine_vp | 20260526 | 0.5873 | 0.5744 | 0.550640 | 0.548257 | 0.002153 | 0.504716 | 0.0507 |
| 2 | dmsr_normal_mid_s0.8 | 20260528 | 0.9921 | 0.9111 | 0.543254 | 0.541355 | 0.002316 | 0.543714 | 0.0498 |
| 3 | uniform_lambda | 20260527 | 0.2237 | 0.2599 | 0.545472 | 0.542834 | 0.002840 | 0.502816 | 0.0489 |
| 4 | cosine_vp | 20260527 | 0.5882 | 0.5738 | 0.545496 | 0.542834 | 0.002887 | 0.504618 | 0.0490 |
| 5 | dmsr_normal_wide_s1.5 | 20260528 | 0.8557 | 0.7735 | 0.543940 | 0.541355 | 0.003120 | 0.514700 | 0.0498 |
| 6 | dmsr_normal_wide_s1.5 | 20260527 | 0.8569 | 0.7742 | 0.546534 | 0.542834 | 0.003190 | 0.557318 | 0.0492 |
| 7 | dmsr_laplace_b0.5 | 20260528 | 0.9865 | 0.9349 | 0.544359 | 0.541355 | 0.003256 | 0.842556 | 0.0500 |
| 8 | dmsr_normal_wide_s1.5 | 20260526 | 0.8565 | 0.7748 | 0.551579 | 0.548257 | 0.003332 | 0.524150 | 0.0508 |
| 9 | dmsr_normal_mid_s0.8 | 20260526 | 0.9924 | 0.9118 | 0.551665 | 0.548257 | 0.003549 | 1.134812 | 0.0511 |
| 10 | dmsr_laplace_b0.5 | 20260527 | 0.9869 | 0.9347 | 0.545693 | 0.542834 | 0.003579 | 0.653499 | 0.0490 |
| 11 | uniform_lambda | 20260526 | 0.2237 | 0.2594 | 0.552606 | 0.548257 | 0.003886 | 0.502847 | 0.0508 |
| 12 | hang_laplace_lambda_b0.5 | 20260528 | 0.9836 | 0.8865 | 0.544668 | 0.541355 | 0.003890 | 0.675259 | 0.0501 |
| 13 | uniform_lambda | 20260528 | 0.2225 | 0.2593 | 0.544504 | 0.541355 | 0.004173 | 0.499769 | 0.0500 |
| 14 | cosine_vp | 20260528 | 0.5898 | 0.5755 | 0.545860 | 0.541355 | 0.004828 | 0.500618 | 0.0498 |
| 15 | dmsr_normal_mid_s0.8 | 20260527 | 0.9925 | 0.9115 | 0.548260 | 0.542834 | 0.005204 | 0.873898 | 0.0492 |
| 16 | dmsr_laplace_b0.5 | 20260526 | 0.9871 | 0.9350 | 0.554223 | 0.548257 | 0.005402 | 0.524723 | 0.0512 |
| 17 | hang_laplace_lambda_b0.5 | 20260526 | 0.9836 | 0.8864 | 0.554361 | 0.548257 | 0.005681 | 0.544735 | 0.0510 |
| 18 | hang_laplace_lambda_b0.5 | 20260527 | 0.9833 | 0.8862 | 0.550675 | 0.542834 | 0.007958 | 0.552841 | 0.0492 |
| 19 | dmsr_normal_narrow_s0.3 | 20260528 | 1.0000 | 0.9854 | 0.583802 | 0.541355 | 0.042972 | 0.606886 | 0.0500 |
| 20 | dmsr_normal_narrow_s0.3 | 20260526 | 1.0000 | 0.9855 | 0.600209 | 0.548257 | 0.051371 | 0.755059 | 0.0504 |
| 21 | dmsr_normal_narrow_s0.3 | 20260527 | 1.0000 | 0.9855 | 0.610682 | 0.542834 | 0.069237 | 0.664955 | 0.0491 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible epsilon-prediction error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal epsilon predictor.
- `uniform_lambda` is a broad full-range baseline used to separate broad support from cosine-specific density effects.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the DMSR/mode-separability mechanism before MNIST/CIFAR phases.
