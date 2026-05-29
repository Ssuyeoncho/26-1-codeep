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
| 1 | cosine_vp | 3 | 0.5934 +/- 0.0010 | 0.5770 +/- 0.0005 | 0.001540 +/- 0.000813 | 0.531179 +/- 0.003474 | 0.1518 +/- 0.0009 |
| 2 | dmsr_normal_wide_s1.5 | 3 | 0.8568 +/- 0.0007 | 0.7741 +/- 0.0006 | 0.001988 +/- 0.000867 | 0.572305 +/- 0.022720 | 0.1515 +/- 0.0005 |
| 3 | dmsr_normal_mid_s0.8 | 3 | 0.9924 +/- 0.0002 | 0.9115 +/- 0.0003 | 0.002030 +/- 0.000771 | 0.632582 +/- 0.040074 | 0.1514 +/- 0.0004 |
| 4 | dmsr_laplace_b0.5 | 3 | 0.9869 +/- 0.0003 | 0.9349 +/- 0.0002 | 0.002291 +/- 0.001322 | 0.610770 +/- 0.028213 | 0.1516 +/- 0.0005 |
| 5 | hang_laplace_lambda_b0.5 | 3 | 0.9871 +/- 0.0003 | 0.9351 +/- 0.0002 | 0.002330 +/- 0.001367 | 0.612240 +/- 0.037394 | 0.1515 +/- 0.0005 |
| 6 | uniform_lambda | 3 | 0.2240 +/- 0.0006 | 0.2594 +/- 0.0001 | 0.002875 +/- 0.001138 | 0.530388 +/- 0.002959 | 0.1519 +/- 0.0007 |
| 7 | dmsr_normal_narrow_s0.3 | 3 | 1.0000 +/- 0.0000 | 0.9855 +/- 0.0001 | 0.027199 +/- 0.003088 | 0.701234 +/- 0.078274 | 0.1516 +/- 0.0008 |

## Per-Seed Results

| rank | schedule | seed | M coverage | S norm | transition MSE | Bayes transition MSE | transition excess MSE | mean MSE | transition mode error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cosine_vp | 20260526 | 0.5940 | 0.5770 | 0.577623 | 0.576468 | 0.000954 | 0.533976 | 0.1522 |
| 2 | dmsr_normal_wide_s1.5 | 20260528 | 0.8561 | 0.7735 | 0.563547 | 0.562510 | 0.001034 | 0.561347 | 0.1515 |
| 3 | hang_laplace_lambda_b0.5 | 20260528 | 0.9868 | 0.9351 | 0.563674 | 0.562510 | 0.001141 | 0.574672 | 0.1513 |
| 4 | dmsr_laplace_b0.5 | 20260528 | 0.9866 | 0.9349 | 0.563696 | 0.562510 | 0.001169 | 0.581371 | 0.1514 |
| 5 | cosine_vp | 20260527 | 0.5921 | 0.5765 | 0.570794 | 0.569385 | 0.001198 | 0.532270 | 0.1508 |
| 6 | dmsr_normal_mid_s0.8 | 20260528 | 0.9921 | 0.9111 | 0.563564 | 0.562510 | 0.001232 | 0.602461 | 0.1513 |
| 7 | uniform_lambda | 20260527 | 0.2246 | 0.2595 | 0.571187 | 0.569385 | 0.001654 | 0.532469 | 0.1512 |
| 8 | dmsr_laplace_b0.5 | 20260527 | 0.9870 | 0.9347 | 0.571898 | 0.569385 | 0.001957 | 0.637624 | 0.1512 |
| 9 | hang_laplace_lambda_b0.5 | 20260527 | 0.9872 | 0.9349 | 0.571943 | 0.569385 | 0.002026 | 0.649458 | 0.1512 |
| 10 | dmsr_normal_mid_s0.8 | 20260527 | 0.9926 | 0.9115 | 0.571766 | 0.569385 | 0.002088 | 0.678064 | 0.1510 |
| 11 | dmsr_normal_wide_s1.5 | 20260526 | 0.8569 | 0.7748 | 0.579143 | 0.576468 | 0.002206 | 0.557141 | 0.1519 |
| 12 | cosine_vp | 20260528 | 0.5940 | 0.5776 | 0.565021 | 0.562510 | 0.002468 | 0.527290 | 0.1524 |
| 13 | dmsr_normal_wide_s1.5 | 20260527 | 0.8574 | 0.7742 | 0.572344 | 0.569385 | 0.002726 | 0.598428 | 0.1510 |
| 14 | dmsr_normal_mid_s0.8 | 20260526 | 0.9925 | 0.9118 | 0.579455 | 0.576468 | 0.002771 | 0.617221 | 0.1519 |
| 15 | uniform_lambda | 20260526 | 0.2237 | 0.2594 | 0.579980 | 0.576468 | 0.003063 | 0.531695 | 0.1526 |
| 16 | dmsr_laplace_b0.5 | 20260526 | 0.9871 | 0.9350 | 0.580335 | 0.576468 | 0.003748 | 0.613316 | 0.1521 |
| 17 | hang_laplace_lambda_b0.5 | 20260526 | 0.9873 | 0.9352 | 0.580388 | 0.576468 | 0.003824 | 0.612592 | 0.1521 |
| 18 | uniform_lambda | 20260528 | 0.2236 | 0.2593 | 0.565975 | 0.562510 | 0.003908 | 0.527001 | 0.1517 |
| 19 | dmsr_normal_narrow_s0.3 | 20260526 | 1.0000 | 0.9855 | 0.600594 | 0.576468 | 0.024423 | 0.644360 | 0.1520 |
| 20 | dmsr_normal_narrow_s0.3 | 20260527 | 1.0000 | 0.9855 | 0.596744 | 0.569385 | 0.026649 | 0.790505 | 0.1507 |
| 21 | dmsr_normal_narrow_s0.3 | 20260528 | 1.0000 | 0.9854 | 0.592962 | 0.562510 | 0.030525 | 0.668836 | 0.1522 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible epsilon-prediction error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal epsilon predictor.
- `uniform_lambda` is a broad full-range baseline used to separate broad support from cosine-specific density effects.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the DMSR/mode-separability mechanism before MNIST/CIFAR phases.
