#!/usr/bin/env python3
"""Phase 1: 1D Gaussian mixture toy experiment for R-transition schedules.

This file is intentionally self-contained. It implements the full Phase 1
pipeline from the experiment plan:

1. sample a two-mode 1D Gaussian mixture,
2. compute analytic EDM/VE R(sigma) and its transition region,
3. train the same small MLP denoiser under multiple training noise samplers,
4. evaluate per-noise denoising MSE, mode classification error, and
   R-transition coverage, and
5. save publication-friendly CSV, Markdown, and plot artifacts.

Run:
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
        phase1/phase1_toy_experiment.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "results" / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class ExperimentConfig:
    d: float = 2.0
    sigma0: float = 0.5
    sigma_min: float = 0.01
    sigma_max: float = 20.0
    rho: float = 0.5
    train_steps: int = 2500
    batch_size: int = 512
    eval_batch_size: int = 4096
    eval_grid_size: int = 80
    lr: float = 2e-3
    hidden_dim: int = 96
    depth: int = 3
    seed: int = 20260526
    num_seeds: int = 1
    device: str = "cpu"
    run_name: str = "phase1_toy"


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    kind: str
    center_log_sigma: float | None = None
    scale: float | None = None
    note: str = ""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_mpl_config_dir(out_dir: Path) -> None:
    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def sample_clean_mixture(batch_size: int, d: float, sigma0: float, device: str) -> tuple[Tensor, Tensor]:
    labels = torch.randint(0, 2, (batch_size, 1), device=device)
    signs = labels.float() * 2.0 - 1.0
    x0 = signs * d + sigma0 * torch.randn(batch_size, 1, device=device)
    return x0, labels


def sinusoidal_logsigma_embedding(log_sigma: Tensor) -> Tensor:
    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=log_sigma.device).view(1, -1)
    phase = log_sigma * freqs
    return torch.cat([log_sigma, torch.sin(phase), torch.cos(phase)], dim=1)


class MLPDenoiser(nn.Module):
    def __init__(self, hidden_dim: int, depth: int) -> None:
        super().__init__()
        in_dim = 1 + 9
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_sigma: Tensor, sigma: Tensor) -> Tensor:
        log_sigma = torch.log(sigma.clamp_min(1e-12))
        features = torch.cat([x_sigma, sinusoidal_logsigma_embedding(log_sigma)], dim=1)
        return self.net(features)


def r_edm(sigma: np.ndarray, d: float, sigma0: float) -> np.ndarray:
    return 2.0 * d / np.sqrt(sigma0**2 + sigma**2)


def r_slope_abs(sigma: np.ndarray, d: float, sigma0: float) -> np.ndarray:
    return 2.0 * d * sigma**2 / np.power(sigma0**2 + sigma**2, 1.5)


def bayes_optimal_denoiser(x_sigma: Tensor, sigma: Tensor, d: float, sigma0: float) -> Tensor:
    """E[x0 | x_sigma, sigma] for the two-component Gaussian mixture.

    For sigma0 > 0 this is not just d * tanh(d * x / variance). The posterior
    mean also includes the within-mode Gaussian shrinkage term.
    """
    variance = sigma0**2 + sigma**2
    shrink = sigma0**2 / variance
    posterior_mode_mean = d * torch.tanh(d * x_sigma / variance)
    return shrink * x_sigma + (1.0 - shrink) * posterior_mode_mean


def transition_bounds(config: ExperimentConfig) -> tuple[float, float, float]:
    grid = np.exp(np.linspace(math.log(config.sigma_min), math.log(config.sigma_max), 4000))
    slope = r_slope_abs(grid, config.d, config.sigma0)
    mask = slope >= config.rho * slope.max()
    return float(grid[mask][0]), float(math.sqrt(2.0) * config.sigma0), float(grid[mask][-1])


def clamp_sigma(sigma: Tensor, config: ExperimentConfig) -> Tensor:
    return sigma.clamp(config.sigma_min, config.sigma_max)


def sample_schedule(spec: ScheduleSpec, n: int, config: ExperimentConfig, device: str) -> Tensor:
    eps = 1e-5
    if spec.kind == "edm_lognormal":
        log_sigma = -1.2 + 1.2 * torch.randn(n, 1, device=device)
        return clamp_sigma(torch.exp(log_sigma), config)

    if spec.kind == "r_normal":
        assert spec.center_log_sigma is not None and spec.scale is not None
        log_sigma = spec.center_log_sigma + spec.scale * torch.randn(n, 1, device=device)
        return clamp_sigma(torch.exp(log_sigma), config)

    if spec.kind == "r_laplace":
        assert spec.center_log_sigma is not None and spec.scale is not None
        dist = torch.distributions.Laplace(spec.center_log_sigma, spec.scale)
        return clamp_sigma(torch.exp(dist.sample((n, 1)).to(device)), config)

    if spec.kind == "hang_laplace_lambda":
        b = spec.scale if spec.scale is not None else 0.5
        lam = torch.distributions.Laplace(0.0, b).sample((n, 1)).to(device)
        return clamp_sigma(torch.exp(-0.5 * lam), config)

    if spec.kind == "cosine_vp_as_ve":
        t = torch.rand(n, 1, device=device).clamp(eps, 1.0 - eps)
        sigma = torch.tan(0.5 * math.pi * t)
        return clamp_sigma(sigma, config)

    if spec.kind == "linear_gamma_as_ve":
        t = torch.rand(n, 1, device=device).clamp(eps, 1.0 - eps)
        gamma = (1.0 - t).clamp(eps, 1.0 - eps)
        sigma = torch.sqrt((1.0 - gamma) / gamma)
        return clamp_sigma(sigma, config)

    raise ValueError(f"Unknown schedule kind: {spec.kind}")


def build_schedules(config: ExperimentConfig) -> list[ScheduleSpec]:
    center = math.log(math.sqrt(2.0) * config.sigma0)
    return [
        ScheduleSpec("cosine_vp_as_ve", "cosine_vp_as_ve", note="Cosine VP density mapped with sigma ~= exp(-lambda/2)."),
        ScheduleSpec("linear_gamma_as_ve", "linear_gamma_as_ve", note="Chen-style gamma(t)=1-t mapped to VE sigma."),
        ScheduleSpec("hang_laplace_lambda_b0.5", "hang_laplace_lambda", scale=0.5, note="Hang-style Laplace around lambda=0."),
        ScheduleSpec("edm_lognormal", "edm_lognormal", note="EDM baseline log sigma ~ N(-1.2, 1.2^2)."),
        ScheduleSpec("r_normal_wide", "r_normal", center_log_sigma=center, scale=0.90, note="R-matched normal, broad coverage."),
        ScheduleSpec("r_normal_mid", "r_normal", center_log_sigma=center, scale=0.45, note="R-matched normal, moderate focus."),
        ScheduleSpec("r_normal_narrow", "r_normal", center_log_sigma=center, scale=0.20, note="R-matched normal, narrow focus."),
        ScheduleSpec("r_laplace_mid", "r_laplace", center_log_sigma=center, scale=0.45, note="R-matched Laplace in log sigma."),
    ]


def train_one_schedule(spec: ScheduleSpec, config: ExperimentConfig, run_seed: int) -> tuple[MLPDenoiser, list[dict[str, float]]]:
    seed_everything(run_seed)
    model = MLPDenoiser(config.hidden_dim, config.depth).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []
    report_every = max(1, config.train_steps // 10)

    for step in range(1, config.train_steps + 1):
        x0, _labels = sample_clean_mixture(config.batch_size, config.d, config.sigma0, config.device)
        sigma = sample_schedule(spec, config.batch_size, config, config.device)
        x_sigma = x0 + sigma * torch.randn_like(x0)
        pred_x0 = model(x_sigma, sigma)
        loss = F.mse_loss(pred_x0, x0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % report_every == 0 or step == config.train_steps:
            history.append({"step": float(step), "train_mse": float(loss.detach().cpu())})

    return model, history


@torch.no_grad()
def evaluate_model(model: MLPDenoiser, spec: ScheduleSpec, config: ExperimentConfig, run_seed: int) -> dict[str, float | int | list[float]]:
    seed_everything(run_seed + 100_000)
    low, center, high = transition_bounds(config)
    sigma_grid = np.exp(np.linspace(math.log(config.sigma_min), math.log(config.sigma_max), config.eval_grid_size))
    per_sigma_mse: list[float] = []
    per_sigma_bayes_mse: list[float] = []
    per_sigma_excess_mse: list[float] = []
    per_sigma_mode_error: list[float] = []

    model.eval()
    for sigma_value in sigma_grid:
        x0, labels = sample_clean_mixture(config.eval_batch_size, config.d, config.sigma0, config.device)
        sigma = torch.full_like(x0, float(sigma_value))
        x_sigma = x0 + sigma * torch.randn_like(x0)
        pred_x0 = model(x_sigma, sigma)
        bayes_x0 = bayes_optimal_denoiser(x_sigma, sigma, config.d, config.sigma0)
        mse = F.mse_loss(pred_x0, x0).item()
        bayes_mse = F.mse_loss(bayes_x0, x0).item()
        excess_mse = F.mse_loss(pred_x0, bayes_x0).item()
        pred_labels = (pred_x0 >= 0).long()
        mode_error = (pred_labels != labels).float().mean().item()
        per_sigma_mse.append(float(mse))
        per_sigma_bayes_mse.append(float(bayes_mse))
        per_sigma_excess_mse.append(float(excess_mse))
        per_sigma_mode_error.append(float(mode_error))

    sampled_sigma = sample_schedule(spec, 200_000, config, config.device).detach().cpu().numpy().reshape(-1)
    slopes = r_slope_abs(sampled_sigma, config.d, config.sigma0)
    coverage_m = float(np.mean((sampled_sigma >= low) & (sampled_sigma <= high)))
    expected_s = float(np.mean(slopes))
    expected_s_norm = float(expected_s / r_slope_abs(np.array([center]), config.d, config.sigma0)[0])

    transition_mask = (sigma_grid >= low) & (sigma_grid <= high)
    low_mask = sigma_grid < low
    high_mask = sigma_grid > high

    return {
        "schedule": spec.name,
        "seed": run_seed,
        "transition_low": low,
        "sigma_r_star": center,
        "transition_high": high,
        "coverage_m": coverage_m,
        "expected_s": expected_s,
        "expected_s_norm": expected_s_norm,
        "mean_mse": float(np.mean(per_sigma_mse)),
        "mean_bayes_mse": float(np.mean(per_sigma_bayes_mse)),
        "mean_excess_mse": float(np.mean(per_sigma_excess_mse)),
        "transition_mse": float(np.mean(np.array(per_sigma_mse)[transition_mask])),
        "transition_bayes_mse": float(np.mean(np.array(per_sigma_bayes_mse)[transition_mask])),
        "transition_excess_mse": float(np.mean(np.array(per_sigma_excess_mse)[transition_mask])),
        "low_noise_mse": float(np.mean(np.array(per_sigma_mse)[low_mask])),
        "high_noise_mse": float(np.mean(np.array(per_sigma_mse)[high_mask])),
        "mean_mode_error": float(np.mean(per_sigma_mode_error)),
        "transition_mode_error": float(np.mean(np.array(per_sigma_mode_error)[transition_mask])),
        "sigma_grid": sigma_grid.tolist(),
        "per_sigma_mse": per_sigma_mse,
        "per_sigma_bayes_mse": per_sigma_bayes_mse,
        "per_sigma_excess_mse": per_sigma_excess_mse,
        "per_sigma_mode_error": per_sigma_mode_error,
    }


def save_metrics_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    keys = [
        "schedule",
        "seed",
        "coverage_m",
        "expected_s_norm",
        "mean_mse",
        "mean_bayes_mse",
        "mean_excess_mse",
        "transition_mse",
        "transition_bayes_mse",
        "transition_excess_mse",
        "low_noise_mse",
        "high_noise_mse",
        "mean_mode_error",
        "transition_mode_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


def save_per_sigma_csv(results: list[dict[str, float | list[float]]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "seed", "sigma", "mse", "bayes_mse", "excess_mse", "mode_error"])
        for result in results:
            schedule = str(result["schedule"])
            seed = int(result["seed"])
            for sigma, mse, bayes_mse, excess_mse, err in zip(
                result["sigma_grid"],
                result["per_sigma_mse"],
                result["per_sigma_bayes_mse"],
                result["per_sigma_excess_mse"],
                result["per_sigma_mode_error"],
            ):
                writer.writerow([schedule, seed, sigma, mse, bayes_mse, excess_mse, err])


def plot_r_profile(config: ExperimentConfig, out_path: Path) -> None:
    low, center, high = transition_bounds(config)
    sigma = np.exp(np.linspace(math.log(config.sigma_min), math.log(config.sigma_max), 600))
    r = r_edm(sigma, config.d, config.sigma0)
    slope = r_slope_abs(sigma, config.d, config.sigma0)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(sigma, r, label="R_EDM(sigma)", color="#2457A6", linewidth=2.2)
    ax1.set_xscale("log")
    ax1.set_xlabel("sigma")
    ax1.set_ylabel("R separability", color="#2457A6")
    ax1.tick_params(axis="y", labelcolor="#2457A6")
    ax1.axvspan(low, high, color="#F2B84B", alpha=0.25, label="R-transition region")
    ax1.axvline(center, color="#C23B22", linestyle="--", linewidth=1.8, label="sigma_R*")

    ax2 = ax1.twinx()
    ax2.plot(sigma, slope, label="|dR/dlog sigma|", color="#268C6C", linewidth=2.0)
    ax2.set_ylabel("|dR/dlog sigma|", color="#268C6C")
    ax2.tick_params(axis="y", labelcolor="#268C6C")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)
    ax1.set_title(f"Analytic R profile: d={config.d}, sigma0={config.sigma0}, rho={config.rho}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_schedule_densities(specs: list[ScheduleSpec], config: ExperimentConfig, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bins = np.linspace(math.log(config.sigma_min), math.log(config.sigma_max), 90)
    low, center, high = transition_bounds(config)
    for spec in specs:
        sigma = sample_schedule(spec, 80_000, config, config.device).detach().cpu().numpy().reshape(-1)
        ax.hist(np.log(sigma), bins=bins, density=True, histtype="step", linewidth=1.7, label=spec.name)
    ax.axvspan(math.log(low), math.log(high), color="#F2B84B", alpha=0.20)
    ax.axvline(math.log(center), color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("log sigma")
    ax.set_ylabel("density")
    ax.set_title("Training noise distributions")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_sigma(results: list[dict[str, float | list[float]]], config: ExperimentConfig, key: str, ylabel: str, out_path: Path) -> None:
    low, center, high = transition_bounds(config)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for result in results:
        ax.plot(result["sigma_grid"], result[key], linewidth=1.9, label=str(result["schedule"]))
    ax.set_xscale("log")
    ax.axvspan(low, high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("sigma")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " across noise levels")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_coverage_tradeoff(summary_rows: list[dict[str, float | str]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    color_values = [float(row["mean_mse"]) for row in summary_rows]
    for row in summary_rows:
        point = ax.scatter(row["coverage_m"], row["transition_excess_mse"], c=[row["mean_mse"]], vmin=min(color_values), vmax=max(color_values), cmap="viridis", s=70)
        label = str(row["schedule"])
        if "seed" in row:
            label += f" s{int(row['seed'])}"
        ax.annotate(label, (row["coverage_m"], row["transition_excess_mse"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    fig.colorbar(point, ax=ax, label="mean MSE across all sigma")
    ax.set_xlabel("R-transition mass M")
    ax.set_ylabel("transition excess MSE vs Bayes denoiser")
    ax.set_title("Transition coverage helps only if the full range is not neglected")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_summary_md(config: ExperimentConfig, specs: list[ScheduleSpec], rows: list[dict[str, float | str]], out_path: Path) -> None:
    low, center, high = transition_bounds(config)
    sorted_rows = sorted(rows, key=lambda row: float(row["transition_excess_mse"]))
    lines = [
        "# Phase 1 Toy Experiment Summary",
        "",
        "## Purpose",
        "",
        "This run tests whether training noise distributions that cover the analytic R-transition region improve denoising exactly where two Gaussian mixture modes become hard to separate.",
        "",
        "## Configuration",
        "",
        f"- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d={config.d}, sigma0={config.sigma0}",
        f"- sigma range: [{config.sigma_min}, {config.sigma_max}]",
        f"- rho threshold: {config.rho}",
        f"- sigma_R*: {center:.6f}",
        f"- transition region: [{low:.6f}, {high:.6f}]",
        f"- train steps per schedule: {config.train_steps}",
        f"- batch size: {config.batch_size}",
        f"- seeds per schedule: {config.num_seeds}",
        "",
        "## Schedules",
        "",
    ]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.note}")
    lines += [
        "",
        "## Main Results",
        "",
        "| rank | schedule | seed | M coverage | S norm | transition MSE | Bayes transition MSE | transition excess MSE | mean MSE | transition mode error |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(sorted_rows, start=1):
        lines.append(
            f"| {rank} | {row['schedule']} | {int(row['seed'])} | {float(row['coverage_m']):.4f} | "
            f"{float(row['expected_s_norm']):.4f} | {float(row['transition_mse']):.6f} | "
            f"{float(row['transition_bayes_mse']):.6f} | {float(row['transition_excess_mse']):.6f} | "
            f"{float(row['mean_mse']):.6f} | {float(row['transition_mode_error']):.4f} |"
        )
    lines += [
        "",
        "## Interpretation Guide",
        "",
        "- `transition_bayes_mse` is the irreducible denoising error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal denoiser.",
        "- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.",
        "- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.",
        "- This is not an image-generation or FID experiment; it validates the R/mode-separability mechanism before MNIST/CIFAR phases.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: ExperimentConfig, out_root: Path) -> Path:
    seed_everything(config.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / "phase1" / f"{run_id}_{config.run_name}_d{config.d}_s0{config.sigma0}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ensure_mpl_config_dir(out_dir)

    specs = build_schedules(config)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (out_dir / "schedules.json").write_text(json.dumps([asdict(spec) for spec in specs], indent=2), encoding="utf-8")

    plot_r_profile(config, plots_dir / "r_profile.png")
    plot_schedule_densities(specs, config, plots_dir / "schedule_densities.png")

    all_results: list[dict[str, float | list[float]]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    summary_rows: list[dict[str, float | str]] = []

    for seed_index in range(config.num_seeds):
        run_seed = config.seed + seed_index
        for spec in specs:
            print(f"[Phase 1] training {spec.name} with seed {run_seed}...")
            model, history = train_one_schedule(spec, config, run_seed)
            histories[f"{spec.name}_seed{run_seed}"] = history
            result = evaluate_model(model, spec, config, run_seed)
            all_results.append(result)
            summary_rows.append({key: value for key, value in result.items() if not isinstance(value, list)})

    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_metrics_csv(summary_rows, out_dir / "metrics_summary.csv")
    save_per_sigma_csv(all_results, out_dir / "per_sigma_metrics.csv")
    save_summary_md(config, specs, summary_rows, out_dir / "summary.md")
    plot_per_sigma(all_results, config, "per_sigma_mse", "Denoising MSE", plots_dir / "per_sigma_mse.png")
    plot_per_sigma(all_results, config, "per_sigma_bayes_mse", "Bayes-optimal MSE", plots_dir / "per_sigma_bayes_mse.png")
    plot_per_sigma(all_results, config, "per_sigma_excess_mse", "Excess MSE vs Bayes denoiser", plots_dir / "per_sigma_excess_mse.png")
    plot_per_sigma(all_results, config, "per_sigma_mode_error", "Mode classification error", plots_dir / "per_sigma_mode_error.png")
    plot_coverage_tradeoff(summary_rows, plots_dir / "coverage_vs_transition_mse.png")

    print(f"[Phase 1] saved results to {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 R-transition toy experiment.")
    parser.add_argument("--d", type=float, default=2.0)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--train-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        d=args.d,
        sigma0=args.sigma0,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=args.device,
    )
    run(config, args.out_root)


if __name__ == "__main__":
    main()
