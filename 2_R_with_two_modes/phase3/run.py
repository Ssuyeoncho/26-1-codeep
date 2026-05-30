"""Plotting, artifact saving, and main experiment runner for Phase 3."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CLASS_PAIRS, PRESETS, ExperimentConfig, ScheduleSpec
from .experiment import (
    build_schedules, compute_coverage_metrics, compute_dmsr_phi, compute_fid,
    ddim_sample, eval_classifier_confidence, eval_per_lambda_mse,
    generate_samples_batched, sample_schedule, train_one_schedule,
)
from .models import resolve_device, seed_everything, train_classifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_DIR / "results"


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_dmsr_profile(
    lambda_grid: np.ndarray,
    dmsr_vals: np.ndarray,
    slope: np.ndarray,
    t_low: float,
    center: float,
    t_high: float,
    out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(lambda_grid, dmsr_vals, color="#2457A6", linewidth=2.2, label="DMSR_phi(lambda)")
    ax1.axvspan(t_low, t_high, color="#F2B84B", alpha=0.25, label="T_R")
    ax1.axvline(center, color="#C23B22", linestyle="--", linewidth=1.8, label="lambda_R*")
    ax1.set_xlabel("lambda = log SNR")
    ax1.set_ylabel("DMSR separability", color="#2457A6")
    ax1.tick_params(axis="y", labelcolor="#2457A6")
    ax2 = ax1.twinx()
    ax2.plot(lambda_grid, slope, color="#268C6C", linewidth=2.0, label="|dDMSR/dlambda|")
    ax2.set_ylabel("|dDMSR/dlambda|", color="#268C6C")
    ax2.tick_params(axis="y", labelcolor="#268C6C")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right", frameon=False)
    ax1.set_title("Empirical DMSR_phi profile (CIFAR-10)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_schedule_densities(
    specs: list[ScheduleSpec],
    cfg: ExperimentConfig,
    t_low: float,
    center: float,
    t_high: float,
    device: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bins = np.linspace(cfg.lambda_min, cfg.lambda_max, 90)
    for spec in specs:
        lam = sample_schedule(spec, 80_000, cfg, device).cpu().numpy().reshape(-1)
        ax.hist(lam, bins=bins, density=True, histtype="step", linewidth=1.7, label=spec.name)
    ax.axvspan(t_low, t_high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("density")
    ax.set_title("Training noise distributions + T_R")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_lambda_mse(
    records: list[dict],
    lambda_grid: np.ndarray,
    t_low: float,
    center: float,
    t_high: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for rec in records:
        ax.plot(rec["lambda_grid"], rec["per_lambda_mse"],
                linewidth=1.9, label=str(rec["schedule"]))
    ax.axvspan(t_low, t_high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("epsilon-prediction MSE")
    ax.set_title("Per-lambda denoising MSE by schedule")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fid_vs_s(rows: list[dict], out_path: Path) -> None:
    dmsr_rows = [r for r in rows if "dmsr_normal" in str(r["schedule"])]
    if not dmsr_rows:
        return
    s_vals = [float(str(r["schedule"]).split("_s")[-1]) for r in dmsr_rows]
    fid_vals = [float(r.get("fid") or float("nan")) for r in dmsr_rows]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(s_vals, fid_vals, s=80, zorder=3)
    ax.plot(s_vals, fid_vals, linewidth=1.5, linestyle="--")
    for s, f, r in zip(s_vals, fid_vals, dmsr_rows):
        ax.annotate(r["schedule"], (s, f), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("s (schedule width)")
    ax.set_ylabel("FID")
    ax.set_title("FID vs s (DMSR-Normal schedules)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fid_vs_m(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for r in rows:
        fid = float(r.get("fid") or float("nan"))
        m = float(r["coverage_m"])
        ax.scatter(m, fid, s=70, zorder=3)
        ax.annotate(str(r["schedule"]), (m, fid),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("M (transition mass)")
    ax.set_ylabel("FID")
    ax.set_title("FID vs T_R coverage M")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_sample_grid(imgs: torch.Tensor, out_path: Path, nrow: int = 8, title: str = "") -> None:
    try:
        from torchvision.utils import make_grid
    except ImportError:
        return
    grid = make_grid(imgs[:nrow * nrow].clamp(-1, 1), nrow=nrow, normalize=True, value_range=(-1, 1))
    npgrid = grid.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(nrow * 1.2, nrow * 1.2))
    ax.imshow(npgrid)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Artifact saving ────────────────────────────────────────────────────────────

_SUMMARY_KEYS = [
    "schedule", "seed", "fid", "classifier_confidence", "balance_error",
    "coverage_m", "expected_s", "mean_mse", "transition_mse",
    "transition_low", "lambda_r_star", "transition_high",
]


def save_metrics_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_KEYS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in _SUMMARY_KEYS})


def save_per_lambda_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "seed", "lambda", "mse"])
        for rec in records:
            for lam, mse in zip(rec["lambda_grid"], rec["per_lambda_mse"]):
                w.writerow([rec["schedule"], rec["seed"], lam, mse])


def save_summary_md(
    cfg: ExperimentConfig,
    specs: list[ScheduleSpec],
    rows: list[dict],
    lambda_r_star: float,
    t_low: float,
    t_high: float,
    out_path: Path,
) -> None:
    sorted_rows = sorted(rows, key=lambda r: float(r.get("fid") or 1e9))
    lines = [
        "# Phase 3 CIFAR-10 Two-Class Experiment Summary", "",
        "## Config", "",
        f"- class pair: {cfg.class_pair}",
        f"- lambda range: [{cfg.lambda_min}, {cfg.lambda_max}]",
        f"- rho: {cfg.rho}",
        f"- lambda_R*: {lambda_r_star:.4f}",
        f"- T_R: [{t_low:.4f}, {t_high:.4f}]",
        f"- train steps: {cfg.train_steps}",
        f"- batch size: {cfg.batch_size}", "",
        "## Schedules", "",
    ]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.note}")
    lines += [
        "",
        "## Main Results (sorted by FID)", "",
        "| rank | schedule | seed | FID | M | S | mean MSE | transition MSE | clf conf | balance err |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(sorted_rows, 1):
        lines.append(
            f"| {rank} | {r['schedule']} | {r['seed']} | "
            f"{float(r.get('fid') or float('nan')):.2f} | "
            f"{float(r['coverage_m']):.4f} | {float(r['expected_s']):.4f} | "
            f"{float(r['mean_mse']):.5f} | {float(r['transition_mse']):.5f} | "
            f"{float(r['classifier_confidence']):.4f} | {float(r['balance_error']):.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main runner ────────────────────────────────────────────────────────────────

def run(cfg: ExperimentConfig, out_root: Path) -> Path:
    seed_everything(cfg.seed)
    device = cfg.device   # already resolved before building cfg

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / "phase3" / f"{run_id}_{cfg.run_name}_{cfg.class_pair}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Keep matplotlib config dir inside the run output, not cwd
    mpl_cfg_dir = out_dir / ".matplotlib"
    mpl_cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg_dir))

    print(f"\n=== Phase 3: {cfg.class_pair} | device={device} ===")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    # 1. Train classifier
    print("\n[1/5] Training feature classifier...")
    classifier = train_classifier(cfg, device)
    torch.save(classifier.state_dict(), out_dir / "classifier.pt")

    # 2. Compute DMSR_phi
    print("\n[2/5] Computing empirical DMSR_phi(lambda)...")
    lambda_grid, dmsr_vals, slope, t_low, lambda_r_star, t_high = compute_dmsr_phi(
        classifier, cfg, device
    )
    print(f"  lambda_R* = {lambda_r_star:.4f}  T_R = [{t_low:.4f}, {t_high:.4f}]")
    np.save(out_dir / "dmsr_grid.npy", lambda_grid)
    np.save(out_dir / "dmsr_vals.npy", dmsr_vals)
    np.save(out_dir / "dmsr_slope.npy", slope)
    plot_dmsr_profile(lambda_grid, dmsr_vals, slope, t_low, lambda_r_star, t_high,
                      plots_dir / "dmsr_profile.png")

    # 3. Build schedules
    print("\n[3/5] Building noise schedules...")
    specs = build_schedules(lambda_r_star, cfg)
    (out_dir / "schedules.json").write_text(
        json.dumps([asdict(s) for s in specs], indent=2), encoding="utf-8"
    )
    plot_schedule_densities(specs, cfg, t_low, lambda_r_star, t_high, device,
                             plots_dir / "schedule_densities.png")

    # 4. Train + evaluate each schedule
    print("\n[4/5] Training diffusion models...")
    all_per_lambda: list[dict] = []
    summary_rows: list[dict] = []
    histories: dict[str, list[dict]] = {}

    for seed_idx in range(cfg.num_seeds):
        run_seed = cfg.seed + seed_idx
        for spec in specs:
            print(f"\n  Schedule: {spec.name}  seed={run_seed}")
            model, ema, history = train_one_schedule(spec, cfg, device, run_seed)
            histories[f"{spec.name}_seed{run_seed}"] = history

            ema.copy_to(model)
            model.eval()

            print("  Evaluating per-lambda MSE...")
            lam_g, per_mse = eval_per_lambda_mse(model, cfg, device)
            transition_mask = (lam_g >= t_low) & (lam_g <= t_high)
            mean_mse = float(np.mean(per_mse))
            transition_mse = (
                float(np.mean(per_mse[transition_mask])) if transition_mask.any() else float("nan")
            )

            print(f"  Generating {cfg.n_gen_samples} samples...")
            gen_imgs = generate_samples_batched(model, cfg.n_gen_samples, cfg, device)
            save_sample_grid(gen_imgs, plots_dir / f"samples_{spec.name}_seed{run_seed}.png",
                             title=spec.name)

            print("  Classifier confidence...")
            clf_conf, bal_err = eval_classifier_confidence(classifier, gen_imgs, device)

            fid_val = float("nan")
            if cfg.compute_fid:
                print("  FID...")
                fid_val = compute_fid(model, cfg, device, cfg.n_gen_samples)

            print("  Coverage metrics...")
            m, s = compute_coverage_metrics(spec, lambda_grid, slope, t_low, t_high, cfg, device)

            print(f"  [{spec.name}] FID={fid_val:.2f}  M={m:.4f}  S={s:.4f}  "
                  f"mean_mse={mean_mse:.5f}")

            rec = {
                "schedule": spec.name, "seed": run_seed,
                "fid": fid_val,
                "classifier_confidence": clf_conf,
                "balance_error": bal_err,
                "coverage_m": m, "expected_s": s,
                "mean_mse": mean_mse, "transition_mse": transition_mse,
                "transition_low": t_low, "lambda_r_star": lambda_r_star,
                "transition_high": t_high,
                "lambda_grid": lam_g.tolist(),
                "per_lambda_mse": per_mse.tolist(),
            }
            all_per_lambda.append(rec)
            summary_rows.append({k: v for k, v in rec.items() if not isinstance(v, list)})
            torch.save(model.state_dict(), out_dir / f"model_{spec.name}_seed{run_seed}.pt")

    # 5. Save all results
    print("\n[5/5] Saving results...")
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_metrics_csv(summary_rows, out_dir / "metrics_summary.csv")
    save_per_lambda_csv(all_per_lambda, out_dir / "per_lambda_metrics.csv")
    save_summary_md(cfg, specs, summary_rows, lambda_r_star, t_low, t_high,
                    out_dir / "summary.md")
    plot_per_lambda_mse(all_per_lambda, lambda_grid, t_low, lambda_r_star, t_high,
                        plots_dir / "per_lambda_mse.png")
    plot_fid_vs_s(summary_rows, plots_dir / "fid_vs_s.png")
    plot_fid_vs_m(summary_rows, plots_dir / "fid_vs_m.png")

    print(f"\nDone. Results saved to: {out_dir}")
    return out_dir


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 CIFAR-10 DMSR experiment")
    p.add_argument("--preset", choices=list(PRESETS), default="full")
    p.add_argument("--class-pair", choices=list(CLASS_PAIRS), default="airplane_vs_automobile")
    p.add_argument("--device", default="auto",
                   help="'auto' (best available), 'cuda', 'mps', or 'cpu'.")
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--num-seeds", type=int, default=None)
    p.add_argument("--s-values", type=float, nargs="+", default=None)
    p.add_argument("--no-fid", action="store_true")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[Phase 3] Using device: {device}")

    preset = PRESETS[args.preset]
    cfg = ExperimentConfig(
        class_pair=args.class_pair,
        clf_epochs=preset["clf_epochs"],
        train_steps=args.train_steps if args.train_steps is not None else preset["train_steps"],
        batch_size=args.batch_size if args.batch_size is not None else 128,
        n_gen_samples=preset["n_gen_samples"],
        dmsr_n_samples=preset["dmsr_n_samples"],
        dmsr_grid_size=preset["dmsr_grid_size"],
        eval_n_samples=preset["eval_n_samples"],
        compute_fid=(not args.no_fid) and preset["compute_fid"],
        seed=args.seed,
        num_seeds=args.num_seeds if args.num_seeds is not None else preset["num_seeds"],
        s_values=tuple(args.s_values) if args.s_values else (1.5, 0.8, 0.3),
        device=device,
        num_workers=args.num_workers,
        data_root=str(PROJECT_DIR / "data"),
    )
    run(cfg, args.out_root)


if __name__ == "__main__":
    main()
