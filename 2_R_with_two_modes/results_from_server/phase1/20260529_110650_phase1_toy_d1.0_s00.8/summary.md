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
| 1 | cosine_vp | 3 | 0.5852 +/- 0.0012 | 0.5708 +/- 0.0004 | 0.001195 +/- 0.000747 | 0.522824 +/- 0.002747 | 0.2610 +/- 0.0017 |
| 2 | dmsr_normal_wide_s1.5 | 3 | 0.8563 +/- 0.0006 | 0.7741 +/- 0.0006 | 0.001352 +/- 0.000785 | 0.574744 +/- 0.006250 | 0.2607 +/- 0.0009 |
| 3 | dmsr_normal_mid_s0.8 | 3 | 0.9923 +/- 0.0002 | 0.9115 +/- 0.0003 | 0.001516 +/- 0.001210 | 0.691715 +/- 0.032117 | 0.2606 +/- 0.0009 |
| 4 | uniform_lambda | 3 | 0.2234 +/- 0.0007 | 0.2593 +/- 0.0000 | 0.001743 +/- 0.000991 | 0.522469 +/- 0.002923 | 0.2603 +/- 0.0007 |
| 5 | hang_laplace_lambda_b0.5 | 3 | 0.9821 +/- 0.0005 | 0.9237 +/- 0.0002 | 0.002009 +/- 0.000425 | 0.668749 +/- 0.021493 | 0.2612 +/- 0.0012 |
| 6 | dmsr_laplace_b0.5 | 3 | 0.9868 +/- 0.0003 | 0.9349 +/- 0.0002 | 0.002072 +/- 0.000770 | 0.631804 +/- 0.029065 | 0.2609 +/- 0.0015 |
| 7 | dmsr_normal_narrow_s0.3 | 3 | 1.0000 +/- 0.0000 | 0.9855 +/- 0.0001 | 0.026411 +/- 0.001674 | 0.662251 +/- 0.032556 | 0.2627 +/- 0.0012 |

## Per-Seed Results

| rank | schedule | seed | M coverage | S norm | transition MSE | Bayes transition MSE | transition excess MSE | mean MSE | transition mode error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dmsr_normal_wide_s1.5 | 20260528 | 0.8556 | 0.7735 | 0.484854 | 0.484319 | 0.000562 | 0.568392 | 0.2614 |
| 2 | dmsr_normal_mid_s0.8 | 20260528 | 0.9921 | 0.9111 | 0.484903 | 0.484319 | 0.000567 | 0.706003 | 0.2611 |
| 3 | cosine_vp | 20260527 | 0.5839 | 0.5704 | 0.496206 | 0.495662 | 0.000571 | 0.524686 | 0.2597 |
| 4 | uniform_lambda | 20260527 | 0.2240 | 0.2592 | 0.496237 | 0.495662 | 0.000685 | 0.524879 | 0.2594 |
| 5 | cosine_vp | 20260526 | 0.5863 | 0.5708 | 0.500224 | 0.499021 | 0.000992 | 0.524117 | 0.2604 |
| 6 | dmsr_normal_mid_s0.8 | 20260527 | 0.9925 | 0.9115 | 0.496866 | 0.495662 | 0.001102 | 0.714208 | 0.2596 |
| 7 | dmsr_normal_wide_s1.5 | 20260527 | 0.8569 | 0.7742 | 0.497113 | 0.495662 | 0.001360 | 0.574952 | 0.2597 |
| 8 | dmsr_laplace_b0.5 | 20260528 | 0.9865 | 0.9349 | 0.486039 | 0.484319 | 0.001581 | 0.608053 | 0.2626 |
| 9 | hang_laplace_lambda_b0.5 | 20260528 | 0.9815 | 0.9237 | 0.486154 | 0.484319 | 0.001626 | 0.649025 | 0.2625 |
| 10 | dmsr_laplace_b0.5 | 20260527 | 0.9869 | 0.9347 | 0.497197 | 0.495662 | 0.001674 | 0.664215 | 0.2599 |
| 11 | uniform_lambda | 20260526 | 0.2227 | 0.2593 | 0.501048 | 0.499021 | 0.001895 | 0.523310 | 0.2606 |
| 12 | hang_laplace_lambda_b0.5 | 20260527 | 0.9824 | 0.9235 | 0.497679 | 0.495662 | 0.001934 | 0.691655 | 0.2600 |
| 13 | cosine_vp | 20260528 | 0.5853 | 0.5712 | 0.486107 | 0.484319 | 0.002023 | 0.519670 | 0.2629 |
| 14 | dmsr_normal_wide_s1.5 | 20260526 | 0.8565 | 0.7748 | 0.501231 | 0.499021 | 0.002133 | 0.580888 | 0.2609 |
| 15 | hang_laplace_lambda_b0.5 | 20260526 | 0.9823 | 0.9238 | 0.501723 | 0.499021 | 0.002466 | 0.665566 | 0.2611 |
| 16 | uniform_lambda | 20260528 | 0.2234 | 0.2593 | 0.486580 | 0.484319 | 0.002650 | 0.519217 | 0.2608 |
| 17 | dmsr_normal_mid_s0.8 | 20260526 | 0.9924 | 0.9118 | 0.502245 | 0.499021 | 0.002879 | 0.654933 | 0.2610 |
| 18 | dmsr_laplace_b0.5 | 20260526 | 0.9871 | 0.9350 | 0.502280 | 0.499021 | 0.002959 | 0.623144 | 0.2602 |
| 19 | dmsr_normal_narrow_s0.3 | 20260528 | 1.0000 | 0.9854 | 0.509286 | 0.484319 | 0.024947 | 0.696450 | 0.2628 |
| 20 | dmsr_normal_narrow_s0.3 | 20260526 | 1.0000 | 0.9855 | 0.525692 | 0.499021 | 0.026050 | 0.631635 | 0.2615 |
| 21 | dmsr_normal_narrow_s0.3 | 20260527 | 1.0000 | 0.9855 | 0.523982 | 0.495662 | 0.028235 | 0.658668 | 0.2639 |

## Interpretation Guide

- `transition_bayes_mse` is the irreducible epsilon-prediction error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal epsilon predictor.
- `uniform_lambda` is a broad full-range baseline used to separate broad support from cosine-specific density effects.
- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.
- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.
- This is not an image-generation or FID experiment; it validates the DMSR/mode-separability mechanism before MNIST/CIFAR phases.
