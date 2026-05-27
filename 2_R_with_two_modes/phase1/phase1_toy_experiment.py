#!/usr/bin/env python3
"""Phase 1: 1D Gaussian mixture toy experiment for DMSR-transition schedules.

This file is intentionally self-contained. It implements the full Phase 1
pipeline from the experiment plan:

1. sample a two-mode 1D Gaussian mixture,
2. compute analytic VP DMSR(lambda) and its transition region,
3. train the same small MLP denoiser under multiple training noise samplers,
4. evaluate per-lambda epsilon-prediction MSE, mode classification error, and
   DMSR-transition coverage, and
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
    lambda_min: float = -10.0
    lambda_max: float = 10.0
    rho: float = 0.5
    train_steps: int = 10000
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
    center_lambda: float | None = None
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


def sinusoidal_lambda_embedding(lam: Tensor) -> Tensor:
    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=lam.device).view(1, -1)
    phase = lam * freqs
    return torch.cat([lam, torch.sin(phase), torch.cos(phase)], dim=1)


class MLPDenoiser(nn.Module):
    def __init__(self, hidden_dim: int, depth: int) -> None:
        super().__init__()
        in_dim = 1 + 9
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_lambda: Tensor, lam: Tensor) -> Tensor:
        features = torch.cat([x_lambda, sinusoidal_lambda_embedding(lam)], dim=1)
        return self.net(features)


def vp_alpha_sigma(lam: Tensor) -> tuple[Tensor, Tensor]:
    alpha_sq = torch.sigmoid(lam)
    sigma_sq = torch.sigmoid(-lam)
    return torch.sqrt(alpha_sq), torch.sqrt(sigma_sq)


def vp_alpha_sigma_np(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha_sq = 1.0 / (1.0 + np.exp(-lam))
    sigma_sq = 1.0 / (1.0 + np.exp(lam))
    return np.sqrt(alpha_sq), np.sqrt(sigma_sq)


def dmsr_vp(lam: np.ndarray, d: float, sigma0: float) -> np.ndarray:
    alpha, sigma = vp_alpha_sigma_np(lam)
    return 2.0 * alpha * d / np.sqrt((alpha * sigma0) ** 2 + sigma**2)


def dmsr_slope_abs(lam: np.ndarray, d: float, sigma0: float) -> np.ndarray:
    exp_lam = np.exp(lam)
    return d * np.sqrt(exp_lam) / np.power(1.0 + sigma0**2 * exp_lam, 1.5)


def bayes_optimal_x0_vp(x_lambda: Tensor, lam: Tensor, d: float, sigma0: float) -> Tensor:
    """E[x0 | x_lambda, lambda] for the VP-corrupted Gaussian mixture."""
    alpha, sigma = vp_alpha_sigma(lam)
    variance = (alpha * sigma0) ** 2 + sigma**2
    shrink = alpha * sigma0**2 / variance
    posterior_mode_mean = d * torch.tanh(alpha * d * x_lambda / variance)
    return shrink * x_lambda + (1.0 - alpha * shrink) * posterior_mode_mean


def bayes_optimal_epsilon_vp(x_lambda: Tensor, lam: Tensor, d: float, sigma0: float) -> Tensor:
    alpha, sigma = vp_alpha_sigma(lam)
    bayes_x0 = bayes_optimal_x0_vp(x_lambda, lam, d, sigma0)
    return (x_lambda - alpha * bayes_x0) / sigma.clamp_min(1e-12)


def transition_bounds(config: ExperimentConfig) -> tuple[float, float, float]:
    grid = np.linspace(config.lambda_min, config.lambda_max, 4000)
    slope = dmsr_slope_abs(grid, config.d, config.sigma0)
    mask = slope >= config.rho * slope.max()
    return float(grid[mask][0]), float(-math.log(2.0 * config.sigma0**2)), float(grid[mask][-1])


def clamp_lambda(lam: Tensor, config: ExperimentConfig) -> Tensor:
    return lam.clamp(config.lambda_min, config.lambda_max)


def sample_schedule(spec: ScheduleSpec, n: int, config: ExperimentConfig, device: str) -> Tensor:
    eps = 1e-5
    if spec.kind == "dmsr_normal":
        assert spec.center_lambda is not None and spec.scale is not None
        lam = spec.center_lambda + spec.scale * torch.randn(n, 1, device=device)
        return clamp_lambda(lam, config)

    if spec.kind == "dmsr_laplace":
        assert spec.center_lambda is not None and spec.scale is not None
        dist = torch.distributions.Laplace(spec.center_lambda, spec.scale)
        return clamp_lambda(dist.sample((n, 1)).to(device), config)

    if spec.kind == "hang_laplace_lambda":
        b = spec.scale if spec.scale is not None else 0.5
        lam = torch.distributions.Laplace(0.0, b).sample((n, 1)).to(device)
        return clamp_lambda(lam, config)

    if spec.kind == "cosine_vp":
        t = torch.rand(n, 1, device=device).clamp(eps, 1.0 - eps)
        lam = -2.0 * torch.log(torch.tan(0.5 * math.pi * t))
        return clamp_lambda(lam, config)

    if spec.kind == "linear_gamma":
        t = torch.rand(n, 1, device=device).clamp(eps, 1.0 - eps)
        gamma = (1.0 - t).clamp(eps, 1.0 - eps)
        lam = torch.log(gamma / (1.0 - gamma))
        return clamp_lambda(lam, config)

    raise ValueError(f"Unknown schedule kind: {spec.kind}")


def build_schedules(config: ExperimentConfig) -> list[ScheduleSpec]:
    center = -math.log(2.0 * config.sigma0**2)
    return [
        ScheduleSpec("cosine_vp", "cosine_vp", note="Cosine VP schedule induced density over lambda."),
        ScheduleSpec("linear_gamma", "linear_gamma", note="Chen-style gamma(t)=1-t baseline in VP lambda space."),
        ScheduleSpec("hang_laplace_lambda_b0.5", "hang_laplace_lambda", scale=0.5, note="Hang-style Laplace around lambda=0."),
        ScheduleSpec("dmsr_normal_wide_s1.5", "dmsr_normal", center_lambda=center, scale=1.5, note="DMSR-centered normal, wide s=1.5."),
        ScheduleSpec("dmsr_normal_mid_s0.8", "dmsr_normal", center_lambda=center, scale=0.8, note="DMSR-centered normal, middle s=0.8."),
        ScheduleSpec("dmsr_normal_narrow_s0.3", "dmsr_normal", center_lambda=center, scale=0.3, note="DMSR-centered normal, narrow s=0.3."),
        ScheduleSpec("dmsr_laplace_b0.5", "dmsr_laplace", center_lambda=center, scale=0.5, note="Hang-style Laplace shifted to lambda_R*."),
    ]


def train_one_schedule(spec: ScheduleSpec, config: ExperimentConfig, run_seed: int) -> tuple[MLPDenoiser, list[dict[str, float]]]:
    seed_everything(run_seed)
    model = MLPDenoiser(config.hidden_dim, config.depth).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []
    report_every = max(1, config.train_steps // 10)

    for step in range(1, config.train_steps + 1):
        x0, _labels = sample_clean_mixture(config.batch_size, config.d, config.sigma0, config.device)
        lam = sample_schedule(spec, config.batch_size, config, config.device)
        alpha, sigma = vp_alpha_sigma(lam)
        eps = torch.randn_like(x0)
        x_lambda = alpha * x0 + sigma * eps
        pred_eps = model(x_lambda, lam)
        loss = F.mse_loss(pred_eps, eps)

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
    lambda_grid = np.linspace(config.lambda_min, config.lambda_max, config.eval_grid_size)
    per_lambda_mse: list[float] = []
    per_lambda_bayes_mse: list[float] = []
    per_lambda_excess_mse: list[float] = []
    per_lambda_mode_error: list[float] = []

    model.eval()
    for lambda_value in lambda_grid:
        x0, labels = sample_clean_mixture(config.eval_batch_size, config.d, config.sigma0, config.device)
        lam = torch.full_like(x0, float(lambda_value))
        alpha, sigma = vp_alpha_sigma(lam)
        eps = torch.randn_like(x0)
        x_lambda = alpha * x0 + sigma * eps
        pred_eps = model(x_lambda, lam)
        bayes_eps = bayes_optimal_epsilon_vp(x_lambda, lam, config.d, config.sigma0)
        pred_x0 = (x_lambda - sigma * pred_eps) / alpha.clamp_min(1e-12)
        mse = F.mse_loss(pred_eps, eps).item()
        bayes_mse = F.mse_loss(bayes_eps, eps).item()
        excess_mse = F.mse_loss(pred_eps, bayes_eps).item()
        pred_labels = (pred_x0 >= 0).long()
        mode_error = (pred_labels != labels).float().mean().item()
        per_lambda_mse.append(float(mse))
        per_lambda_bayes_mse.append(float(bayes_mse))
        per_lambda_excess_mse.append(float(excess_mse))
        per_lambda_mode_error.append(float(mode_error))

    sampled_lambda = sample_schedule(spec, 200_000, config, config.device).detach().cpu().numpy().reshape(-1)
    slopes = dmsr_slope_abs(sampled_lambda, config.d, config.sigma0)
    coverage_m = float(np.mean((sampled_lambda >= low) & (sampled_lambda <= high)))
    expected_s = float(np.mean(slopes))
    expected_s_norm = float(expected_s / dmsr_slope_abs(np.array([center]), config.d, config.sigma0)[0])

    transition_mask = (lambda_grid >= low) & (lambda_grid <= high)
    low_mask = lambda_grid > high
    high_mask = lambda_grid < low

    return {
        "schedule": spec.name,
        "seed": run_seed,
        "transition_low": low,
        "lambda_r_star": center,
        "transition_high": high,
        "coverage_m": coverage_m,
        "expected_s": expected_s,
        "expected_s_norm": expected_s_norm,
        "mean_mse": float(np.mean(per_lambda_mse)),
        "mean_bayes_mse": float(np.mean(per_lambda_bayes_mse)),
        "mean_excess_mse": float(np.mean(per_lambda_excess_mse)),
        "transition_mse": float(np.mean(np.array(per_lambda_mse)[transition_mask])),
        "transition_bayes_mse": float(np.mean(np.array(per_lambda_bayes_mse)[transition_mask])),
        "transition_excess_mse": float(np.mean(np.array(per_lambda_excess_mse)[transition_mask])),
        "low_noise_mse": float(np.mean(np.array(per_lambda_mse)[low_mask])),
        "high_noise_mse": float(np.mean(np.array(per_lambda_mse)[high_mask])),
        "mean_mode_error": float(np.mean(per_lambda_mode_error)),
        "transition_mode_error": float(np.mean(np.array(per_lambda_mode_error)[transition_mask])),
        "lambda_grid": lambda_grid.tolist(),
        "per_lambda_mse": per_lambda_mse,
        "per_lambda_bayes_mse": per_lambda_bayes_mse,
        "per_lambda_excess_mse": per_lambda_excess_mse,
        "per_lambda_mode_error": per_lambda_mode_error,
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


def save_per_lambda_csv(results: list[dict[str, float | list[float]]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "seed", "lambda", "mse", "bayes_mse", "excess_mse", "mode_error"])
        for result in results:
            schedule = str(result["schedule"])
            seed = int(result["seed"])
            for lam, mse, bayes_mse, excess_mse, err in zip(
                result["lambda_grid"],
                result["per_lambda_mse"],
                result["per_lambda_bayes_mse"],
                result["per_lambda_excess_mse"],
                result["per_lambda_mode_error"],
            ):
                writer.writerow([schedule, seed, lam, mse, bayes_mse, excess_mse, err])


def plot_dmsr_profile(config: ExperimentConfig, out_path: Path) -> None:
    low, center, high = transition_bounds(config)
    lam = np.linspace(config.lambda_min, config.lambda_max, 600)
    r = dmsr_vp(lam, config.d, config.sigma0)
    slope = dmsr_slope_abs(lam, config.d, config.sigma0)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(lam, r, label="DMSR_VP(lambda)", color="#2457A6", linewidth=2.2)
    ax1.set_xlabel("lambda = log SNR")
    ax1.set_ylabel("DMSR separability", color="#2457A6")
    ax1.tick_params(axis="y", labelcolor="#2457A6")
    ax1.axvspan(low, high, color="#F2B84B", alpha=0.25, label="DMSR-transition region")
    ax1.axvline(center, color="#C23B22", linestyle="--", linewidth=1.8, label="lambda_R*")

    ax2 = ax1.twinx()
    ax2.plot(lam, slope, label="|dDMSR/dlambda|", color="#268C6C", linewidth=2.0)
    ax2.set_ylabel("|dDMSR/dlambda|", color="#268C6C")
    ax2.tick_params(axis="y", labelcolor="#268C6C")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)
    ax1.set_title(f"Analytic VP DMSR profile: d={config.d}, sigma0={config.sigma0}, rho={config.rho}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_schedule_densities(specs: list[ScheduleSpec], config: ExperimentConfig, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bins = np.linspace(config.lambda_min, config.lambda_max, 90)
    low, center, high = transition_bounds(config)
    for spec in specs:
        lam = sample_schedule(spec, 80_000, config, config.device).detach().cpu().numpy().reshape(-1)
        ax.hist(lam, bins=bins, density=True, histtype="step", linewidth=1.7, label=spec.name)
    ax.axvspan(low, high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("density")
    ax.set_title("Training noise distributions")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_lambda(results: list[dict[str, float | list[float]]], config: ExperimentConfig, key: str, ylabel: str, out_path: Path) -> None:
    low, center, high = transition_bounds(config)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for result in results:
        ax.plot(result["lambda_grid"], result[key], linewidth=1.9, label=str(result["schedule"]))
    ax.axvspan(low, high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("lambda = log SNR")
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
    fig.colorbar(point, ax=ax, label="mean MSE across all lambda")
    ax.set_xlabel("DMSR-transition mass M")
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
        "This run tests whether training noise distributions that cover the analytic VP DMSR-transition region improve epsilon prediction where two Gaussian mixture modes become hard to separate.",
        "",
        "## Configuration",
        "",
        f"- mixture: 0.5 N(-d, sigma0^2) + 0.5 N(d, sigma0^2), d={config.d}, sigma0={config.sigma0}",
        f"- lambda range: [{config.lambda_min}, {config.lambda_max}]",
        f"- rho threshold: {config.rho}",
        f"- lambda_R*: {center:.6f}",
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
        "- `transition_bayes_mse` is the irreducible epsilon-prediction error for this GMM; `transition_excess_mse` measures how far the learned MLP is from the Bayes-optimal epsilon predictor.",
        "- High `M` alone is not sufficient. A schedule can put nearly all mass in the transition region and still perform poorly if it neglects the broader denoising range.",
        "- The intended claim is therefore about adequate transition coverage with enough full-range support, not monotonic improvement as `M` increases.",
        "- This is not an image-generation or FID experiment; it validates the DMSR/mode-separability mechanism before MNIST/CIFAR phases.",
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

    plot_dmsr_profile(config, plots_dir / "dmsr_profile.png")
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
    save_per_lambda_csv(all_results, out_dir / "per_lambda_metrics.csv")
    save_summary_md(config, specs, summary_rows, out_dir / "summary.md")
    plot_per_lambda(all_results, config, "per_lambda_mse", "Epsilon-prediction MSE", plots_dir / "per_lambda_mse.png")
    plot_per_lambda(all_results, config, "per_lambda_bayes_mse", "Bayes-optimal epsilon MSE", plots_dir / "per_lambda_bayes_mse.png")
    plot_per_lambda(all_results, config, "per_lambda_excess_mse", "Excess MSE vs Bayes epsilon predictor", plots_dir / "per_lambda_excess_mse.png")
    plot_per_lambda(all_results, config, "per_lambda_mode_error", "Mode classification error", plots_dir / "per_lambda_mode_error.png")
    plot_coverage_tradeoff(summary_rows, plots_dir / "coverage_vs_transition_mse.png")

    print(f"[Phase 1] saved results to {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 VP DMSR-transition toy experiment.")
    parser.add_argument("--d", type=float, default=2.0)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--lambda-min", type=float, default=-10.0)
    parser.add_argument("--lambda-max", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--eval-grid-size", type=int, default=80)
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
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        rho=args.rho,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_grid_size=args.eval_grid_size,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=args.device,
    )
    run(config, args.out_root)


if __name__ == "__main__":
    main()
