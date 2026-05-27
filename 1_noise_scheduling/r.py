"""
Toy log-SNR noise-schedule analysis for a 1D Gaussian mixture.

This file is intentionally self-contained.  It uses only the Python standard
library so that the experiment can run even in a minimal environment without
numpy/matplotlib/torch installed.

Research goal
-------------
Hang et al., "Improved Noise Schedule for Diffusion Training" (ICCV 2025)
is used here as conceptual motivation, not as a reproduction target.  We borrow
one high-level lens: a noise schedule can be studied as a probability
distribution over log SNR values, lambda = log SNR.  With timesteps sampled
uniformly, a schedule lambda(t) induces

    p(lambda) = |dt / d lambda|.

This project makes a smaller, separate toy contribution: define mode
separability R for a 1D Gaussian mixture and ask whether R gives an intuitive
explanation of which log-SNR regions are structurally important.  The script
does not implement ImageNet training, DiT, FID evaluation, or the full method of
the paper.  It records where each schedule spends probability relative to the
region where the two toy modes become ambiguous.

Run:
    python r.py

Outputs:
    outputs/run_XXX_logsnr_toy/
        EXPERIMENT_SUMMARY.md
        run_log.txt
        schedule_metrics.csv
        mode_separability_lambda.svg
        schedule_R_t.svg
        p_lambda_allocation.svg
        gmm_noised_densities.svg
        curves/*.csv
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyConfig:
    # Data distribution: 0.5 N(-d, sigma0^2) + 0.5 N(+d, sigma0^2)
    d: float = 3.0
    sigma0: float = 1.0

    # Ambiguous region in terms of mode-separability R(lambda).
    # For the default d=3, sigma0=1 this gives a transition interval roughly
    # lambda in [-2.71, 0.25], i.e. it includes lambda=0 but excludes both the
    # trivially separated and nearly destroyed regimes.
    ambiguous_r_low: float = 1.5
    ambiguous_r_high: float = 4.5

    # Numerical grids.
    lambda_min: float = -12.0
    lambda_max: float = 12.0
    n_lambda: int = 1201
    n_t: int = 2000
    histogram_bins: int = 120

    # Discrete DDPM linear schedule parameters.
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # A simple log-SNR-centered reference schedule.  It uses a Laplace-shaped
    # p(lambda) because that is a convenient way to put probability near
    # lambda=0, but here it is only a toy comparison curve.
    laplace_mu: float = 0.0
    laplace_b: float = 0.5

    # Toy density visualization.
    density_lambda_values: Tuple[float, ...] = (4.0, 0.0, -2.5, -6.0)


CFG = ToyConfig()


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid_stable(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def alpha_bar_from_lambda(lam: float) -> float:
    """VP relation: alpha_bar = exp(lambda) / (1 + exp(lambda))."""
    return sigmoid_stable(lam)


def lambda_from_alpha_bar(alpha_bar: float) -> float:
    alpha_bar = clamp(alpha_bar, 1e-300, 1.0 - 1e-15)
    return math.log(alpha_bar / (1.0 - alpha_bar))


def mode_separability_from_lambda(
    lam: float,
    d: float = CFG.d,
    sigma0: float = CFG.sigma0,
) -> float:
    """
    R(lambda) = 2 d exp(lambda/2) / sqrt(exp(lambda) sigma0^2 + 1).

    This is schedule-independent.  A schedule only decides how often training
    visits each lambda value.
    """
    if lam > 80:
        # Limit as lambda -> +infinity.
        return 2.0 * d / sigma0
    exp_lam = math.exp(lam)
    return (2.0 * d * math.exp(0.5 * lam)) / math.sqrt(exp_lam * sigma0 * sigma0 + 1.0)


def mode_separability_from_alpha_bar(
    alpha_bar: float,
    d: float = CFG.d,
    sigma0: float = CFG.sigma0,
) -> float:
    numerator = 2.0 * math.sqrt(alpha_bar) * d
    denominator = math.sqrt(alpha_bar * sigma0 * sigma0 + (1.0 - alpha_bar))
    return numerator / denominator


def gaussian_pdf(x: float, mean: float, std: float) -> float:
    z = (x - mean) / std
    return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def noised_gmm_density(x: float, lam: float, d: float = CFG.d, sigma0: float = CFG.sigma0) -> float:
    """Density of x_lambda after the VP forward process for the 1D mixture."""
    alpha_bar = alpha_bar_from_lambda(lam)
    mean_scale = math.sqrt(alpha_bar)
    var = alpha_bar * sigma0 * sigma0 + (1.0 - alpha_bar)
    std = math.sqrt(var)
    return 0.5 * gaussian_pdf(x, -mean_scale * d, std) + 0.5 * gaussian_pdf(x, mean_scale * d, std)


# ---------------------------------------------------------------------------
# Schedules in log-SNR space
# ---------------------------------------------------------------------------


def lambda_cosine(t: float) -> float:
    """
    Continuous cosine schedule: alpha_bar(t) = cos^2(pi t / 2).

    Then lambda(t) = log(alpha_bar / (1 - alpha_bar))
                   = -2 log tan(pi t / 2).
    """
    eps = 1e-6
    t = clamp(t, eps, 1.0 - eps)
    return -2.0 * math.log(math.tan(0.5 * math.pi * t))


def lambda_laplace_centered(t: float, mu: float = CFG.laplace_mu, b: float = CFG.laplace_b) -> float:
    """
    Log-SNR-centered toy schedule from a Laplace-shaped p(lambda), using the
    convention t up means lambda down:

        lambda(t) = mu - b * sign(0.5 - t) * log(1 - 2 |t - 0.5|).

    In this project it is not treated as "the paper's method"; it is just a
    compact reference curve that spends many timesteps near lambda=0.
    """
    eps = 1e-6
    t = clamp(t, eps, 1.0 - eps)
    sign = 1.0 if (0.5 - t) >= 0.0 else -1.0
    return mu - b * sign * math.log(max(1e-300, 1.0 - 2.0 * abs(t - 0.5)))


def make_linear_ddpm_lambdas(cfg: ToyConfig = CFG) -> List[float]:
    alpha_bar = 1.0
    vals: List[float] = []
    for i in range(cfg.T):
        if cfg.T == 1:
            beta = cfg.beta_end
        else:
            frac = i / (cfg.T - 1)
            beta = cfg.beta_start + frac * (cfg.beta_end - cfg.beta_start)
        alpha_bar *= 1.0 - beta
        vals.append(lambda_from_alpha_bar(alpha_bar))
    return vals


LINEAR_DDPM_LAMBDAS = make_linear_ddpm_lambdas()


def lambda_linear_ddpm(t: float) -> float:
    """Interpolate the discrete Ho et al. linear beta schedule in lambda space."""
    t = clamp(t, 0.0, 1.0)
    pos = t * (len(LINEAR_DDPM_LAMBDAS) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(LINEAR_DDPM_LAMBDAS) - 1)
    w = pos - lo
    return (1.0 - w) * LINEAR_DDPM_LAMBDAS[lo] + w * LINEAR_DDPM_LAMBDAS[hi]


SCHEDULES: Dict[str, Callable[[float], float]] = {
    "cosine": lambda_cosine,
    "linear_ddpm": lambda_linear_ddpm,
    "laplace_centered": lambda_laplace_centered,
}


# ---------------------------------------------------------------------------
# Numerics and summaries
# ---------------------------------------------------------------------------


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def sample_t_grid(n: int) -> List[float]:
    # Midpoints avoid infinite endpoints for schedules with lambda -> +/- infinity.
    return [(i + 0.5) / n for i in range(n)]


def find_ambiguous_lambda_interval(cfg: ToyConfig = CFG) -> Tuple[float, float]:
    grid = linspace(cfg.lambda_min, cfg.lambda_max, cfg.n_lambda * 4)
    inside = [
        lam
        for lam in grid
        if cfg.ambiguous_r_low <= mode_separability_from_lambda(lam, cfg.d, cfg.sigma0) <= cfg.ambiguous_r_high
    ]
    if not inside:
        raise RuntimeError("No ambiguous region found. Adjust R thresholds.")
    return inside[0], inside[-1]


def histogram_density(values: Sequence[float], lo: float, hi: float, bins: int) -> Tuple[List[float], List[float]]:
    width = (hi - lo) / bins
    counts = [0 for _ in range(bins)]
    kept = 0
    for v in values:
        if lo <= v <= hi:
            idx = min(bins - 1, int((v - lo) / width))
            counts[idx] += 1
            kept += 1
    centers = [lo + (i + 0.5) * width for i in range(bins)]
    if kept == 0:
        return centers, [0.0 for _ in counts]
    density = [c / (kept * width) for c in counts]
    return centers, density


def curve_for_schedule(name: str, fn: Callable[[float], float], cfg: ToyConfig = CFG) -> Dict[str, List[float]]:
    t_vals = sample_t_grid(cfg.n_t)
    lam_vals = [fn(t) for t in t_vals]
    r_vals = [mode_separability_from_lambda(lam, cfg.d, cfg.sigma0) for lam in lam_vals]
    return {"t": t_vals, "lambda": lam_vals, "R": r_vals}


def approximate_ambiguous_mass(lambdas: Sequence[float], amb_lo: float, amb_hi: float) -> float:
    return sum(1 for lam in lambdas if amb_lo <= lam <= amb_hi) / len(lambdas)


def first_crossing_t(t_vals: Sequence[float], y_vals: Sequence[float], threshold: float) -> float | None:
    for t, y in zip(t_vals, y_vals):
        if y <= threshold:
            return t
    return None


def median(values: Sequence[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        raise ValueError("quantile of empty sequence")
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    w = pos - lo
    return (1.0 - w) * vals[lo] + w * vals[hi]


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def next_run_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(base, exist_ok=True)
    max_seen = 0
    for name in os.listdir(base):
        if not name.startswith("run_"):
            continue
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            max_seen = max(max_seen, int(parts[1]))
        elif len(parts) >= 2 and parts[1][:3].isdigit():
            max_seen = max(max_seen, int(parts[1][:3]))
    run_num = max_seen + 1
    out_dir = os.path.join(base, f"run_{run_num:03d}_logsnr_toy")
    os.makedirs(out_dir, exist_ok=False)
    os.makedirs(os.path.join(out_dir, "curves"), exist_ok=True)
    return out_dir


def write_csv(path: str, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def fmt(x: float | None, digits: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Tiny SVG plotting utilities
# ---------------------------------------------------------------------------


COLORS = {
    "green": "#1D9E75",
    "orange": "#D85A30",
    "blue": "#378ADD",
    "purple": "#534AB7",
    "gray": "#6B7280",
    "red": "#B91C1C",
    "teal": "#0F766E",
    "gold": "#A16207",
}


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass
class Series:
    x: Sequence[float]
    y: Sequence[float]
    label: str
    color: str
    width: float = 2.4


def make_svg_line_plot(
    path: str,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Series],
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    vlines: Sequence[Tuple[float, str, str]] = (),
    hlines: Sequence[Tuple[float, str, str]] = (),
    shade_x: Tuple[float, float] | None = None,
    width: int = 920,
    height: int = 560,
) -> None:
    margin_l, margin_r, margin_t, margin_b = 76, 26, 58, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    xs = [v for s in series for v in s.x if math.isfinite(v)]
    ys = [v for s in series for v in s.y if math.isfinite(v)]
    xmin, xmax = xlim if xlim else (min(xs), max(xs))
    ymin, ymax = ylim if ylim else (min(ys), max(ys))
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    ypad = 0.05 * (ymax - ymin)
    if ylim is None:
        ymin -= ypad
        ymax += ypad

    def sx(x: float) -> float:
        return margin_l + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return margin_t + (ymax - y) / (ymax - ymin) * plot_h

    def tick_values(lo: float, hi: float, n: int = 6) -> List[float]:
        return [lo + i * (hi - lo) / (n - 1) for i in range(n)]

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="600">{svg_escape(title)}</text>')

    # Grid and ticks.
    for tv in tick_values(xmin, xmax):
        x = sx(tv)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_t}" x2="{x:.2f}" y2="{margin_t+plot_h}" stroke="#E5E7EB" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{margin_t+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{tv:.2g}</text>')
    for tv in tick_values(ymin, ymax):
        y = sy(tv)
        parts.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l+plot_w}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{tv:.2g}</text>')

    if shade_x is not None:
        a, b = shade_x
        left = max(margin_l, min(margin_l + plot_w, sx(a)))
        right = max(margin_l, min(margin_l + plot_w, sx(b)))
        if right > left:
            parts.append(f'<rect x="{left:.2f}" y="{margin_t}" width="{right-left:.2f}" height="{plot_h}" fill="#F97316" opacity="0.10"/>')

    for xval, color, label in vlines:
        if xmin <= xval <= xmax:
            x = sx(xval)
            parts.append(f'<line x1="{x:.2f}" y1="{margin_t}" x2="{x:.2f}" y2="{margin_t+plot_h}" stroke="{color}" stroke-width="1.7" stroke-dasharray="6 5"/>')
            if label:
                parts.append(f'<text x="{x+6:.2f}" y="{margin_t+18}" font-family="Arial" font-size="12" fill="{color}">{svg_escape(label)}</text>')
    for yval, color, label in hlines:
        if ymin <= yval <= ymax:
            y = sy(yval)
            parts.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l+plot_w}" y2="{y:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="5 5"/>')
            if label:
                parts.append(f'<text x="{margin_l+8}" y="{y-6:.2f}" font-family="Arial" font-size="12" fill="{color}">{svg_escape(label)}</text>')

    # Axes.
    parts.append(f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" stroke="#111827" stroke-width="1.2"/>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#111827" stroke-width="1.2"/>')

    # Lines.
    for s in series:
        pts = []
        last_inside = False
        segments: List[List[str]] = []
        current: List[str] = []
        for xval, yval in zip(s.x, s.y):
            finite = math.isfinite(xval) and math.isfinite(yval)
            inside = finite and (xmin <= xval <= xmax) and (ymin <= yval <= ymax)
            if inside:
                current.append(f"{sx(xval):.2f},{sy(yval):.2f}")
            elif current:
                segments.append(current)
                current = []
            last_inside = inside
        if current:
            segments.append(current)
        for seg in segments:
            if len(seg) >= 2:
                parts.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{s.color}" stroke-width="{s.width}" stroke-linejoin="round" stroke-linecap="round"/>')

    # Legend.
    legend_x = margin_l + plot_w - 170
    legend_y = margin_t + 16
    parts.append(f'<rect x="{legend_x-12}" y="{legend_y-16}" width="164" height="{24*len(series)+12}" fill="white" opacity="0.86" stroke="#D1D5DB"/>')
    for i, s in enumerate(series):
        y = legend_y + i * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{s.color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+36}" y="{y+4}" font-family="Arial" font-size="13" fill="#111827">{svg_escape(s.label)}</text>')

    parts.append(f'<text x="{margin_l+plot_w/2:.1f}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="15">{svg_escape(x_label)}</text>')
    parts.append(f'<text x="20" y="{margin_t+plot_h/2:.1f}" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 20 {margin_t+plot_h/2:.1f})">{svg_escape(y_label)}</text>')
    parts.append("</svg>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_experiment(cfg: ToyConfig = CFG) -> str:
    out_dir = next_run_dir()
    curves_dir = os.path.join(out_dir, "curves")
    amb_lo, amb_hi = find_ambiguous_lambda_interval(cfg)

    # 1. Schedule-independent R(lambda).
    lambda_grid = linspace(cfg.lambda_min, cfg.lambda_max, cfg.n_lambda)
    r_lambda = [mode_separability_from_lambda(lam, cfg.d, cfg.sigma0) for lam in lambda_grid]
    write_csv(
        os.path.join(curves_dir, "mode_separability_lambda.csv"),
        ["lambda", "R_lambda"],
        zip(lambda_grid, r_lambda),
    )

    make_svg_line_plot(
        os.path.join(out_dir, "mode_separability_lambda.svg"),
        "Mode separability is a function of log SNR, not of the schedule",
        "lambda = log SNR",
        "R(lambda)",
        [Series(lambda_grid, r_lambda, "R(lambda)", COLORS["green"])],
        xlim=(cfg.lambda_min, cfg.lambda_max),
        ylim=(0.0, 2.0 * cfg.d / cfg.sigma0 * 1.08),
        vlines=[(0.0, COLORS["orange"], "lambda=0")],
        hlines=[
            (cfg.ambiguous_r_low, COLORS["gray"], "ambiguous low"),
            (cfg.ambiguous_r_high, COLORS["gray"], "ambiguous high"),
        ],
        shade_x=(amb_lo, amb_hi),
    )

    # 2. R(t) and p(lambda) for schedules.
    schedule_curves: Dict[str, Dict[str, List[float]]] = {}
    p_lambda_series: List[Series] = []
    r_t_series: List[Series] = []
    metrics_rows: List[List[object]] = []
    colors = {
        "cosine": COLORS["orange"],
        "linear_ddpm": COLORS["blue"],
        "laplace_centered": COLORS["purple"],
    }

    for name, fn in SCHEDULES.items():
        curve = curve_for_schedule(name, fn, cfg)
        schedule_curves[name] = curve
        write_csv(
            os.path.join(curves_dir, f"{name}_t_lambda_R.csv"),
            ["t", "lambda_t", "R_t"],
            zip(curve["t"], curve["lambda"], curve["R"]),
        )

        centers, density = histogram_density(
            curve["lambda"],
            cfg.lambda_min,
            cfg.lambda_max,
            cfg.histogram_bins,
        )
        write_csv(
            os.path.join(curves_dir, f"{name}_p_lambda_histogram.csv"),
            ["lambda_center", "p_lambda_density"],
            zip(centers, density),
        )

        p_lambda_series.append(Series(centers, density, name, colors[name]))
        r_t_series.append(Series(curve["t"], curve["R"], name, colors[name]))

        mass = approximate_ambiguous_mass(curve["lambda"], amb_lo, amb_hi)
        t_r_low = first_crossing_t(curve["t"], curve["R"], cfg.ambiguous_r_low)
        t_r_high = first_crossing_t(curve["t"], curve["R"], cfg.ambiguous_r_high)
        metrics_rows.append(
            [
                name,
                f"{mass:.6f}",
                f"{min(curve['lambda']):.6f}",
                f"{quantile(curve['lambda'], 0.25):.6f}",
                f"{median(curve['lambda']):.6f}",
                f"{quantile(curve['lambda'], 0.75):.6f}",
                f"{max(curve['lambda']):.6f}",
                fmt(t_r_high),
                fmt(t_r_low),
            ]
        )

    write_csv(
        os.path.join(out_dir, "schedule_metrics.csv"),
        [
            "schedule",
            "ambiguous_mass",
            "lambda_min",
            "lambda_q25",
            "lambda_median",
            "lambda_q75",
            "lambda_max",
            f"first_t_R_below_{cfg.ambiguous_r_high}",
            f"first_t_R_below_{cfg.ambiguous_r_low}",
        ],
        metrics_rows,
    )

    make_svg_line_plot(
        os.path.join(out_dir, "schedule_R_t.svg"),
        "Schedule-dependent collapse speed: R(t) = R(lambda(t))",
        "normalized timestep t",
        "R(t)",
        r_t_series,
        xlim=(0.0, 1.0),
        ylim=(0.0, 2.0 * cfg.d / cfg.sigma0 * 1.08),
        hlines=[
            (cfg.ambiguous_r_low, COLORS["gray"], "ambiguous low"),
            (cfg.ambiguous_r_high, COLORS["gray"], "ambiguous high"),
        ],
    )

    make_svg_line_plot(
        os.path.join(out_dir, "p_lambda_allocation.svg"),
        "Induced p(lambda): where uniform timestep sampling spends probability",
        "lambda = log SNR",
        "p(lambda)",
        p_lambda_series,
        xlim=(cfg.lambda_min, cfg.lambda_max),
        ylim=(0.0, 1.35),
        vlines=[(0.0, COLORS["orange"], "lambda=0")],
        shade_x=(amb_lo, amb_hi),
    )

    # 3. Optional toy density visualization: noised Gaussian mixture at lambdas.
    x_grid = linspace(-8.0, 8.0, 1000)
    density_series: List[Series] = []
    density_colors = [COLORS["teal"], COLORS["orange"], COLORS["blue"], COLORS["red"]]
    density_rows: List[Tuple[float, float, float]] = []
    for lam, color in zip(cfg.density_lambda_values, density_colors):
        ys = [noised_gmm_density(x, lam, cfg.d, cfg.sigma0) for x in x_grid]
        density_series.append(Series(x_grid, ys, f"lambda={lam:g}, R={mode_separability_from_lambda(lam, cfg.d, cfg.sigma0):.2f}", color))
        for x, y in zip(x_grid, ys):
            density_rows.append((lam, x, y))
    write_csv(
        os.path.join(curves_dir, "noised_gmm_densities.csv"),
        ["lambda", "x", "density"],
        density_rows,
    )
    make_svg_line_plot(
        os.path.join(out_dir, "gmm_noised_densities.svg"),
        "1D Gaussian mixture after noising at representative log-SNR levels",
        "x_lambda",
        "density",
        density_series,
        xlim=(-8.0, 8.0),
        ylim=(0.0, 0.32),
        vlines=[(-cfg.d, COLORS["gray"], ""), (cfg.d, COLORS["gray"], "")],
    )

    write_summary(out_dir, amb_lo, amb_hi, metrics_rows, cfg)
    write_log(out_dir, amb_lo, amb_hi, metrics_rows, cfg)
    return out_dir


def write_summary(
    out_dir: str,
    amb_lo: float,
    amb_hi: float,
    metrics_rows: Sequence[Sequence[object]],
    cfg: ToyConfig,
) -> None:
    metrics_md = "\n".join(
        f"| {row[0]} | {row[1]} | {row[7]} | {row[8]} |"
        for row in metrics_rows
    )
    path = os.path.join(out_dir, "EXPERIMENT_SUMMARY.md")
    text = f"""# Toy Log-SNR Noise-Schedule Experiment

Generated: {_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This run analyzes a 1D Gaussian mixture:

```text
p(x0) = 0.5 N(-d, sigma0^2) + 0.5 N(+d, sigma0^2)
d = {cfg.d}, sigma0 = {cfg.sigma0}
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
{cfg.ambiguous_r_low} <= R(lambda) <= {cfg.ambiguous_r_high}
lambda interval approximately [{amb_lo:.4f}, {amb_hi:.4f}]
```

The exact thresholds are a toy analysis choice, not a theorem.  They are chosen
so that the transition includes `lambda = 0` for the default mixture.

## Schedule Comparison

| schedule | ambiguous mass | first t where R < {cfg.ambiguous_r_high} | first t where R < {cfg.ambiguous_r_low} |
|---|---:|---:|---:|
{metrics_md}

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
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_log(
    out_dir: str,
    amb_lo: float,
    amb_hi: float,
    metrics_rows: Sequence[Sequence[object]],
    cfg: ToyConfig,
) -> None:
    lines = [
        "=== Toy log-SNR schedule analysis ===",
        f"Timestamp: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output directory: {out_dir}",
        "",
        "Parameters:",
        f"  d = {cfg.d}",
        f"  sigma0 = {cfg.sigma0}",
        f"  ambiguous R interval = [{cfg.ambiguous_r_low}, {cfg.ambiguous_r_high}]",
        f"  ambiguous lambda interval ~= [{amb_lo:.6f}, {amb_hi:.6f}]",
        f"  lambda=0 gives R = {mode_separability_from_lambda(0.0, cfg.d, cfg.sigma0):.6f}",
        "",
        "Schedule metrics:",
    ]
    for row in metrics_rows:
        lines.append(
            f"  {row[0]:>16}: ambiguous_mass={row[1]}, "
            f"lambda_median={row[4]}, first_t_R<{cfg.ambiguous_r_high}={row[7]}, "
            f"first_t_R<{cfg.ambiguous_r_low}={row[8]}"
        )
    lines.extend(
        [
            "",
            "Conceptual note:",
            "  The paper is used as motivation only; this run is an R-based toy analysis.",
            "  R(lambda) describes the toy data geometry at a log-SNR value.",
            "  p(lambda) describes how often a schedule samples each log-SNR value.",
            "  The key question is whether p(lambda) overlaps the ambiguous R(lambda) region.",
        ]
    )
    with open(os.path.join(out_dir, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    out_dir = run_experiment(CFG)
    print("")
    print("[saved] " + out_dir)
    print("[saved] " + os.path.join(out_dir, "EXPERIMENT_SUMMARY.md"))
    print("[saved] " + os.path.join(out_dir, "schedule_metrics.csv"))


if __name__ == "__main__":
    main()
