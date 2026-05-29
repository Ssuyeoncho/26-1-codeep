"""Plotting, artifact saving, and main experiment runner for Phase 2."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ExperimentConfig, ScheduleSpec
from .models import (
    FeatureClassifier, load_mnist_two_class, resolve_device,
    seed_everything, train_feature_classifier,
)
from .experiment import (
    build_schedules, compute_classifier_metrics, compute_coverage_metrics,
    compute_empirical_dmsr, compute_empirical_transition, compute_fid_phi,
    ddim_generate, evaluate_per_lambda, extract_features,
    sample_schedule, train_one_schedule,
)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _span_kw() -> dict:
    return dict(color="#F2B84B", alpha=0.22)


def plot_dmsr_profile(
    lambda_grid, dmsr_values, dmsr_slope,
    transition_low, transition_high, lambda_r_star,
    config: ExperimentConfig, out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(lambda_grid, dmsr_values, label="DMSR_φ(λ)", color="#2457A6", linewidth=2.2)
    ax1.set_xlabel("λ = log SNR")
    ax1.set_ylabel("DMSR_φ separability", color="#2457A6")
    ax1.tick_params(axis="y", labelcolor="#2457A6")
    ax1.axvspan(transition_low, transition_high, **_span_kw(), label="DMSR-transition region")
    ax1.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.8, label="λ_R*")
    ax2 = ax1.twinx()
    ax2.plot(lambda_grid, dmsr_slope, label="|dDMSR_φ/dλ|", color="#268C6C", linewidth=2.0)
    ax2.set_ylabel("|dDMSR_φ/dλ|", color="#268C6C")
    ax2.tick_params(axis="y", labelcolor="#268C6C")
    lines, labels = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lb2, loc="upper right", frameon=False)
    d0, d1 = config.digits
    ax1.set_title(f"Empirical VP DMSR_φ(λ): MNIST {d0} vs {d1}, ρ={config.rho}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_schedule_densities(
    specs, config: ExperimentConfig,
    transition_low, transition_high, lambda_r_star, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bins = np.linspace(config.lambda_min, config.lambda_max, 90)
    for spec in specs:
        lam = sample_schedule(spec, 80_000, config, config.device).detach().cpu().numpy().reshape(-1)
        ax.hist(lam, bins=bins, density=True, histtype="step", linewidth=1.7, label=spec.name)
    ax.axvspan(transition_low, transition_high, **_span_kw())
    ax.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("λ = log SNR")
    ax.set_ylabel("density")
    ax.set_title("Training noise distributions (shading = DMSR transition region)")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_lambda_mse(
    results, config: ExperimentConfig,
    transition_low, transition_high, lambda_r_star, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for r in results:
        ax.plot(r["lambda_grid"], r["per_lambda_mse"], linewidth=1.9, label=str(r["schedule"]))
    ax.axvspan(transition_low, transition_high, **_span_kw())
    ax.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("λ = log SNR")
    ax.set_ylabel("Epsilon-prediction MSE")
    ax.set_title("Per-λ denoising MSE by schedule")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fid_summary(summary_rows, out_path: Path) -> None:
    names = [r["schedule"] for r in summary_rows]
    fids  = [float(r.get("fid_phi", float("nan"))) for r in summary_rows]
    order = np.argsort(fids)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.barh([names[i] for i in order], [fids[i] for i in order], color="#2457A6", alpha=0.8)
    ax.set_xlabel("FID (φ-feature space)")
    ax.set_title("FID comparison by training schedule")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_coverage_tradeoff(summary_rows, out_path: Path) -> None:
    fids = [float(r.get("fid_phi", 0.0)) for r in summary_rows]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    sc = ax.scatter(
        [float(r["coverage_m"]) for r in summary_rows],
        [float(r.get("fid_phi", float("nan"))) for r in summary_rows],
        c=fids, cmap="viridis", s=70,
    )
    for r in summary_rows:
        ax.annotate(str(r["schedule"]), (float(r["coverage_m"]), float(r.get("fid_phi", 0))),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)
    fig.colorbar(sc, ax=ax, label="FID (φ)")
    ax.set_xlabel("DMSR-transition mass M")
    ax.set_ylabel("FID (φ-feature space)")
    ax.set_title("Transition coverage vs FID")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_noisy_images(imgs_a, imgs_b, config: ExperimentConfig, lambda_r_star, out_path: Path) -> None:
    from .models import vp_alpha_sigma
    lam_vals = [lambda_r_star + d for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    n_show = 3
    fig, axes = plt.subplots(2, len(lam_vals), figsize=(len(lam_vals) * 2.2, 5.0))
    for col, lv in enumerate(lam_vals):
        lv = float(np.clip(lv, config.lambda_min, config.lambda_max))
        alpha, sigma = vp_alpha_sigma(torch.tensor(lv))
        for row, imgs in enumerate([imgs_a, imgs_b]):
            x_lam = alpha * imgs[:n_show] + sigma * torch.randn_like(imgs[:n_show])
            axes[row, col].imshow(x_lam[0, 0].clamp(-1, 1).numpy() * 0.5 + 0.5,
                                  cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"λ={lv:.1f}", fontsize=8)
    axes[0, 0].set_ylabel(f"digit {config.digits[0]}", fontsize=8)
    axes[1, 0].set_ylabel(f"digit {config.digits[1]}", fontsize=8)
    fig.suptitle(f"Noisy images around λ_R*={lambda_r_star:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_grid(schedule_name, gen_imgs, out_path: Path, n_rows=4, n_cols=8) -> None:
    n    = min(n_rows * n_cols, len(gen_imgs))
    imgs = gen_imgs[:n].clamp(-1, 1) * 0.5 + 0.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.1))
    for idx, ax in enumerate(axes.reshape(-1)):
        if idx < n:
            ax.imshow(imgs[idx, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"Generated samples: {schedule_name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Artifact Saving ────────────────────────────────────────────────────────────

_SUMMARY_KEYS = [
    "schedule", "seed",
    "coverage_m", "expected_s_norm",
    "fid_phi",
    "classifier_confidence", "balance_error",
    "mean_mse", "transition_mse", "low_noise_mse", "high_noise_mse",
]


def save_metrics_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_KEYS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _SUMMARY_KEYS})


def save_per_lambda_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "seed", "lambda", "mse"])
        for r in results:
            for lam, mse in zip(r["lambda_grid"], r["per_lambda_mse"]):
                writer.writerow([r["schedule"], int(r["seed"]), lam, mse])


def save_summary_md(
    config: ExperimentConfig,
    specs: list[ScheduleSpec],
    summary_rows: list[dict],
    dmsr_info: dict,
    out_path: Path,
) -> None:
    d0, d1  = config.digits
    low     = dmsr_info["transition_low"]
    high    = dmsr_info["transition_high"]
    r_star  = dmsr_info["lambda_r_star"]
    sorted_rows = sorted(summary_rows, key=lambda r: float(r.get("fid_phi", float("inf"))))

    lines = [
        "# Phase 2 MNIST Pilot Summary", "",
        "## Configuration", "",
        f"- digits: {d0} vs {d1}",
        f"- λ range: [{config.lambda_min}, {config.lambda_max}]",
        f"- ρ threshold: {config.rho}",
        f"- empirical λ_R*: {r_star:.4f}",
        f"- transition region: [{low:.4f}, {high:.4f}]",
        f"- train steps per schedule: {config.train_steps}",
        f"- DDIM steps: {config.ddim_steps}",
        f"- n_generate (FID): {config.n_generate}", "",
        "## Schedules", "",
    ]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.note}")
    lines += [
        "",
        "## Main Results", "",
        "| rank | schedule | seed | M coverage | S norm | FID (φ) | clf conf | balance err | mean MSE | transition MSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(sorted_rows, 1):
        lines.append(
            f"| {rank} | {row['schedule']} | {int(row['seed'])} "
            f"| {float(row['coverage_m']):.4f} | {float(row['expected_s_norm']):.4f} "
            f"| {float(row.get('fid_phi', float('nan'))):.2f} "
            f"| {float(row.get('classifier_confidence', float('nan'))):.4f} "
            f"| {float(row.get('balance_error', float('nan'))):.4f} "
            f"| {float(row['mean_mse']):.4f} | {float(row['transition_mse']):.4f} |"
        )
    lines += [
        "",
        "## Interpretation Guide", "",
        "- FID is computed in φ-feature space (not InceptionV3). Lower is better.",
        "- `coverage_m` alone is not sufficient; the balance with full-range support matters.",
        "- λ_R* is estimated empirically from the numerical slope of DMSR_φ(λ).",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main Run ───────────────────────────────────────────────────────────────────

def run(config: ExperimentConfig, out_root: Path) -> Path:
    seed_everything(config.seed)
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    d0, d1    = config.digits
    out_dir   = out_root / "phase2" / f"{run_id}_{config.run_name}_d{d0}v{d1}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    tr_imgs, tr_labels, te_imgs, te_labels = load_mnist_two_class(config)
    imgs_a = tr_imgs[tr_labels == 0]
    imgs_b = tr_imgs[tr_labels == 1]

    # 2. Train classifier φ
    print("[Phase 2] Training feature classifier φ...")
    clf = train_feature_classifier(tr_imgs, tr_labels, te_imgs, te_labels, config)
    torch.save(clf.state_dict(), out_dir / "classifier_phi.pt")

    # 3. Compute empirical DMSR_φ(λ)
    print("[Phase 2] Computing empirical DMSR_φ(λ)...")
    dmsr_grid, dmsr_vals = compute_empirical_dmsr(clf, imgs_a, imgs_b, config)
    dmsr_slope_abs = np.abs(np.gradient(dmsr_vals, dmsr_grid))
    trans_low, lambda_r_star, trans_high = compute_empirical_transition(
        dmsr_grid, dmsr_vals, config.rho
    )
    print(f"[Phase 2] λ_R* = {lambda_r_star:.4f}  T_R = [{trans_low:.4f}, {trans_high:.4f}]")

    dmsr_info = {"transition_low": trans_low, "lambda_r_star": lambda_r_star, "transition_high": trans_high}
    (out_dir / "dmsr_info.json").write_text(json.dumps({
        "lambda_grid": dmsr_grid.tolist(), "dmsr_values": dmsr_vals.tolist(),
        "dmsr_slope_abs": dmsr_slope_abs.tolist(), **dmsr_info,
    }, indent=2), encoding="utf-8")

    plot_dmsr_profile(dmsr_grid, dmsr_vals, dmsr_slope_abs, trans_low, trans_high, lambda_r_star,
                      config, plots_dir / "dmsr_profile.png")
    plot_noisy_images(imgs_a, imgs_b, config, lambda_r_star, plots_dir / "noisy_images_at_transition.png")

    # 4. Build schedules
    specs = build_schedules(config, lambda_r_star)
    (out_dir / "schedules.json").write_text(json.dumps([asdict(s) for s in specs], indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    plot_schedule_densities(specs, config, trans_low, trans_high, lambda_r_star,
                            plots_dir / "schedule_densities.png")

    # 5. Pre-compute real features for FID
    real_imgs_fid = tr_imgs[:config.n_generate]

    # 6. Train + evaluate each schedule
    all_per_lambda: list[dict] = []
    summary_rows:   list[dict] = []
    histories:      dict       = {}

    for seed_idx in range(config.num_seeds):
        run_seed = config.seed + seed_idx
        for spec in specs:
            print(f"[Phase 2] Training {spec.name} (seed={run_seed})...")
            model, history = train_one_schedule(spec, tr_imgs, config, run_seed)
            histories[f"{spec.name}_seed{run_seed}"] = history

            per_lam = evaluate_per_lambda(model, tr_imgs, config, trans_low, trans_high)
            per_lam.update({"schedule": spec.name, "seed": run_seed})
            all_per_lambda.append(per_lam)

            cov = compute_coverage_metrics(spec, config, trans_low, trans_high, dmsr_grid, dmsr_slope_abs)

            if config.n_generate > 0:
                print(f"[Phase 2]   Generating {config.n_generate} samples...")
                gen_imgs    = ddim_generate(model, config.n_generate, config)
                fid         = compute_fid_phi(clf, real_imgs_fid, gen_imgs, config.eval_batch_size, config.device)
                clf_metrics = compute_classifier_metrics(clf, gen_imgs, config)
                plot_sample_grid(spec.name, gen_imgs, plots_dir / f"samples_{spec.name}.png")
            else:
                fid         = float("nan")
                clf_metrics = {"classifier_confidence": float("nan"), "balance_error": float("nan"), "frac_class_0": float("nan")}

            row: dict = {
                "schedule": spec.name, "seed": run_seed,
                **cov, "fid_phi": fid, **clf_metrics,
                "mean_mse":       per_lam["mean_mse"],
                "transition_mse": per_lam["transition_mse"],
                "low_noise_mse":  per_lam["low_noise_mse"],
                "high_noise_mse": per_lam["high_noise_mse"],
            }
            summary_rows.append(row)
            print(f"[Phase 2]   FID={fid:.2f}  M={cov['coverage_m']:.4f}  mean_mse={per_lam['mean_mse']:.4f}")

    # 7. Save artifacts
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_metrics_csv(summary_rows, out_dir / "metrics_summary.csv")
    save_per_lambda_csv(all_per_lambda, out_dir / "per_lambda_metrics.csv")
    save_summary_md(config, specs, summary_rows, dmsr_info, out_dir / "summary.md")

    plot_per_lambda_mse(all_per_lambda, config, trans_low, trans_high, lambda_r_star,
                        plots_dir / "per_lambda_mse.png")
    if config.n_generate > 0 and not any(math.isnan(r.get("fid_phi", float("nan"))) for r in summary_rows):
        plot_fid_summary(summary_rows, plots_dir / "fid_summary.png")
        plot_coverage_tradeoff(summary_rows, plots_dir / "coverage_vs_fid.png")

    print(f"[Phase 2] Results saved to {out_dir}")
    return out_dir


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 2 MNIST DMSR noise schedule experiment.")
    p.add_argument("--digits",         type=int,   nargs=2, default=[0, 1])
    p.add_argument("--lambda-min",     type=float, default=-10.0)
    p.add_argument("--lambda-max",     type=float, default=10.0)
    p.add_argument("--rho",            type=float, default=0.5)
    p.add_argument("--clf-epochs",     type=int,   default=10)
    p.add_argument("--dmsr-grid-size", type=int,   default=40)
    p.add_argument("--dmsr-n-samples", type=int,   default=512)
    p.add_argument("--base-ch",        type=int,   default=32)
    p.add_argument("--train-steps",    type=int,   default=20000)
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--eval-grid-size", type=int,   default=40)
    p.add_argument("--seed",           type=int,   default=20260526)
    p.add_argument("--num-seeds",      type=int,   default=1)
    p.add_argument("--device",         type=str,   default="auto",
                   help="'auto', 'cuda', 'mps', or 'cpu'.")
    p.add_argument("--ddim-steps",     type=int,   default=50)
    p.add_argument("--n-generate",     type=int,   default=5000)
    p.add_argument("--gen-batch-size", type=int,   default=100)
    p.add_argument("--data-root",      type=str,   default="./data")
    p.add_argument("--out-root",       type=Path,  default=Path("results"))
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    print(f"[Phase 2] Using device: {device}")
    config = ExperimentConfig(
        digits=tuple(args.digits),
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        rho=args.rho,
        clf_epochs=args.clf_epochs,
        dmsr_grid_size=args.dmsr_grid_size,
        dmsr_n_samples=args.dmsr_n_samples,
        base_ch=args.base_ch,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_grid_size=args.eval_grid_size,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=device,
        ddim_steps=args.ddim_steps,
        n_generate=args.n_generate,
        gen_batch_size=args.gen_batch_size,
        data_root=args.data_root,
    )
    run(config, args.out_root)


if __name__ == "__main__":
    main()
