# Toy Log-SNR Noise-Schedule Experiment

Generated: 2026-05-19 23:18:49

This run analyzes a 1D Gaussian mixture:

```text
p(x0) = 0.5 N(-d, sigma0^2) + 0.5 N(+d, sigma0^2)
d = 3.0, sigma0 = 1.0
```

This is a toy-model analysis, not a reproduction of a large diffusion paper.
There is no ImageNet training, DiT backbone, FID table, or full schedule search.
The contribution here is the mode-separability indicator `R` and the question:

```text
Can R give an intuitive explanation of which log-SNR regions are structurally
important during diffusion training?
```

## Conceptual Motivation From The Paper

Hang et al., *Improved Noise Schedule for Diffusion Training* (ICCV 2025), are
used only for high-level motivation.  The borrowed perspective is:

```text
lambda = log SNR
p(lambda) = |dt / d lambda|
```

If training samples timesteps uniformly, different schedules still visit
log-SNR values with different frequencies.  The paper motivates paying attention
to intermediate noise levels around `lambda = 0`, where denoising transitions
from signal-dominant to noise-dominant.  This script asks a smaller question:
does our toy `R(lambda)` identify a structurally meaningful intermediate region?

## Toy Separability Index

For the noised 1D mixture, the two mode means move toward zero and the component
variance changes.  We summarize mode visibility with:

```text
R(lambda) = 2 d exp(lambda/2) / sqrt(exp(lambda) sigma0^2 + 1)
```

Interpretation:

- large `R`: the two modes are clearly separated, so the denoising decision is easier
- small `R`: mode information is mostly destroyed
- intermediate `R`: the sample is structurally ambiguous, so the reverse process must make a meaningful decision

This run defines the ambiguous region as:

```text
1.5 <= R(lambda) <= 4.5
lambda interval approximately [-2.7058, 0.2473]
```

The exact thresholds are a toy analysis choice, not a theorem.  They are chosen
so that the transition includes `lambda = 0` for the default mixture.

## Schedule Comparison

| schedule | ambiguous mass | first t where R < 4.5 | first t where R < 1.5 |
|---|---:|---:|---:|
| cosine | 0.378500 | 0.460250 | 0.839250 |
| linear_ddpm | 0.286000 | 0.235250 | 0.521750 |
| laplace_centered | 0.693000 | 0.302750 | 0.998250 |

`ambiguous mass` is the fraction of uniform timesteps whose `lambda(t)` falls in
the ambiguous region.  It measures how much each schedule overlaps the region
where the toy modes are neither clearly separated nor fully destroyed.

## Output Files

- `mode_separability_lambda.svg`: schedule-independent `R(lambda)`.  This shows
  how the mixture structure collapses as log SNR decreases.
- `schedule_R_t.svg`: schedule-dependent `R(t) = R(lambda(t))`.  The same
  structural curve is traversed at different speeds by different schedules.
- `p_lambda_allocation.svg`: induced `p(lambda)` for each schedule, with the
  ambiguous region shaded.
- `gmm_noised_densities.svg`: analytic density of the noised mixture at several
  representative log-SNR values.
- `schedule_metrics.csv`: scalar summary table.
- `curves/*.csv`: raw data behind each plot.

## Reading The Results

The most important distinction is:

```text
R(lambda) is schedule-independent.
R(t) and p(lambda) are schedule-dependent.
```

The log-SNR-centered schedule is deliberately built from a distribution peaked
near `lambda = 0`, so it should assign more mass to the ambiguous transition than
a schedule whose probability is spread broadly across log-SNR.  This is not a
claim that the toy schedule is generally better.  The useful research-style
argument is narrower: `R` gives a concrete toy-model reason to look at
intermediate log-SNR values, and `p(lambda)` tells us whether a schedule actually
samples those values often.
