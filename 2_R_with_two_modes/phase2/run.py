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

from gpu_perf import configure_backends

from .config import ExperimentConfig, ScheduleSpec
from .models import (
    FeatureClassifier, load_mnist_two_class, resolve_device,
    seed_everything, train_feature_classifier,
)
from .experiment import (
    LOWER_IS_BETTER,
    aggregate_over_seeds, aggregate_per_lambda, aggregated_csv_fieldnames,
    build_schedules, compute_classifier_metrics, compute_coverage_metrics,
    compute_empirical_dmsr, compute_empirical_transition, compute_phi_metrics,
    ddim_generate, evaluate_per_lambda,
    sample_schedule, significance_tests, train_one_schedule,
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


def plot_fid_with_error_bars(agg_rows, out_path: Path) -> None:
    """schedule별 FID 평균 ± 표준편차(seed 간)를 막대그래프로 그린다.

    seed가 1개뿐이면 오차막대가 없어 단일 막대로 표시된다. seed를 늘릴수록
    이 그림이 "차이가 분산보다 큰지"를 시각적으로 보여 주는 핵심 통계 그래프가 된다.
    """
    rows  = [r for r in agg_rows if np.isfinite(r.get("fid_phi_mean", float("nan")))]
    if not rows:
        return
    rows  = sorted(rows, key=lambda r: r["fid_phi_mean"])
    names = [r["schedule"] for r in rows]
    means = [r["fid_phi_mean"] for r in rows]
    # std가 NaN(seed 1개)이면 오차막대 0으로 처리
    errs  = [0.0 if not np.isfinite(r.get("fid_phi_std", float("nan"))) else r["fid_phi_std"]
             for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.barh(names, means, xerr=errs, color="#2457A6", alpha=0.8, capsize=4)
    ax.set_xlabel("FID (φ-feature space), mean ± std over seeds")
    ax.set_title(f"FID by schedule (n_seeds={rows[0]['n_seeds']})")
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
    # 생성 품질 지표 (φ 공간) — Phase 3와 동일한 정의
    "fid_phi", "kid_phi", "kid_phi_std",
    "precision_phi", "recall_phi", "density_phi", "coverage_phi",
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


def save_aggregated_csv(agg_rows: list[dict], path: Path) -> None:
    """seed 간 집계 결과(mean/std/sem/n)를 CSV로 저장한다.

    컬럼은 집계 결과에서 동적으로 만든다(지표 자동 탐지와 일관). 따라서 새로운
    지표가 추가돼도 CSV가 자동으로 그 컬럼을 포함한다.
    """
    if not agg_rows:
        return
    fieldnames = aggregated_csv_fieldnames(agg_rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_significance_md(
    sig_rows: list[dict],
    baseline: str,
    metric: str,
    num_seeds: int,
    out_path: Path,
) -> None:
    """baseline 대비 paired 유의성 검정 결과를 사람이 읽을 수 있게 저장한다."""
    arrow = "낮을수록" if metric in LOWER_IS_BETTER else "높을수록"
    lines = [
        f"# Phase 2 유의성 검정 ({metric}, baseline = {baseline})", "",
        f"- 지표 해석: **{metric}** 은 {arrow} 좋다.",
        f"- 설계: 동일 seed에서 측정한 값끼리 짝지은 **paired** 비교 (n_seeds={num_seeds}).",
        "- mean_diff = (해당 schedule − baseline)의 seed 평균. "
        "낮을수록 좋은 지표면 음수가 개선을 뜻한다.",
        "- p-value 는 paired t-test 기준 (seed ≥ 5 이면 Wilcoxon 보조 제시).", "",
        "| schedule | n_pairs | mean_diff | improved | t p-value | wilcoxon p-value | status |",
        "|---|---:|---:|:---:|---:|---:|---|",
    ]
    for r in sig_rows:
        imp = "—" if r["improved_vs_baseline"] is None else ("✓" if r["improved_vs_baseline"] else "✗")
        def _fmt(x: float) -> str:
            return "nan" if not np.isfinite(x) else f"{x:.4g}"
        lines.append(
            f"| {r['schedule']} | {r['n_pairs']} | {_fmt(r['mean_diff'])} | {imp} "
            f"| {_fmt(r['t_pvalue'])} | {_fmt(r['wilcoxon_pvalue'])} | {r['status']} |"
        )
    if num_seeds < 2:
        lines += [
            "",
            "> ⚠️ seed가 1개뿐이라 유의성 검정을 수행할 수 없습니다. "
            "Phase 3 본 실험에서는 `--num-seeds 3` 이상으로 재실행해 이 표를 채웁니다.",
        ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary_md(
    config: ExperimentConfig,
    specs: list[ScheduleSpec],
    summary_rows: list[dict],
    agg_rows: list[dict],
    sig_rows: list[dict],
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
    # ── per-(schedule, seed) raw 결과 (재현·디버깅용) ──────────────────────────
    lines += [
        "",
        "## Per-run Results (schedule × seed)", "",
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

    # ── seed 간 집계 결과 (mean ± std) — 통계 해석의 본체 ───────────────────────
    agg_sorted = sorted(
        agg_rows,
        key=lambda r: r.get("fid_phi_mean", float("inf"))
        if np.isfinite(r.get("fid_phi_mean", float("nan"))) else float("inf"),
    )
    lines += [
        "",
        f"## Aggregated over seeds (n_seeds = {config.num_seeds})", "",
        "생성 품질 지표는 mean ± std(seed 간)로 보고한다. seed가 1개면 std가 표시되지 않는다.",
        "FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다.",
        "",
        "| rank | schedule | n | FID (φ) | KID (φ) | Precision (φ) | Coverage (φ) | mean MSE |",
        "|---:|---|---:|---|---|---|---|---|",
    ]

    def _ms(row: dict, metric: str, fmt: str = ".3f") -> str:
        mean, std = row.get(f"{metric}_mean", float("nan")), row.get(f"{metric}_std", float("nan"))
        if not np.isfinite(mean):
            return "nan"
        if not np.isfinite(std):
            return f"{mean:{fmt}}"
        return f"{mean:{fmt}} ± {std:{fmt}}"

    for rank, row in enumerate(agg_sorted, 1):
        lines.append(
            f"| {rank} | {row['schedule']} | {int(row['n_seeds'])} "
            f"| {_ms(row, 'fid_phi', '.2f')} | {_ms(row, 'kid_phi', '.4f')} "
            f"| {_ms(row, 'precision_phi')} | {_ms(row, 'coverage_phi')} | {_ms(row, 'mean_mse', '.4f')} |"
        )

    # ── baseline 대비 paired 유의성 검정 ───────────────────────────────────────
    lines += [
        "",
        f"## Significance vs baseline ({config.baseline_schedule}, metric = FID φ)", "",
        "동일 seed에서 짝지은 paired 비교. mean_diff<0 이면 baseline보다 FID가 낮다(개선).",
        "",
        "| schedule | n_pairs | mean_diff | improved | t p-value | status |",
        "|---|---:|---:|:---:|---:|---|",
    ]
    for r in sig_rows:
        imp = "—" if r["improved_vs_baseline"] is None else ("✓" if r["improved_vs_baseline"] else "✗")
        p = "nan" if not np.isfinite(r["t_pvalue"]) else f"{r['t_pvalue']:.4g}"
        md = "nan" if not np.isfinite(r["mean_diff"]) else f"{r['mean_diff']:.3f}"
        lines.append(f"| {r['schedule']} | {r['n_pairs']} | {md} | {imp} | {p} | {r['status']} |")

    lines += [
        "",
        "## Interpretation Guide", "",
        "- 이 단계의 목적은 생성 성능 주장이 아니라 **파이프라인·통계 틀 검증**이다.",
        "  MNIST는 쉬운 데이터라 schedule 간 FID 차이가 작아도 실패가 아니다.",
        "- 생성 품질은 φ-feature space에서 FID·KID·Precision/Recall/Density/Coverage로 잰다.",
        "  FID는 품질·다양성을 뭉뚱그리므로, Precision(품질)과 Recall/Coverage(다양성)를 "
        "함께 보면 mode collapse 같은 실패를 분리해 볼 수 있다(Phase 3와 동일한 지표).",
        "- KID는 표본이 적을 때 FID보다 신뢰성이 높고 부분표본 분산을 함께 준다.",
        "- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며, "
        "full-range support 와의 균형이 중요하다(Phase 1 결론).",
        "- λ_R*는 DMSR_φ(λ)의 수치 미분 peak에서 경험적으로 추정한다. "
        "이 값은 Phase 3로 넘어가지 않으며 CIFAR에서 독립적으로 재추정한다.",
        f"- 유의성 검정은 seed가 부족하면(현재 {config.num_seeds}개) 건너뛴다. "
        "Phase 3에서 `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main Run ───────────────────────────────────────────────────────────────────

def run(config: ExperimentConfig, out_root: Path) -> Path:
    seed_everything(config.seed)
    configure_backends()   # cudnn autotuner + TF32 (CUDA가 아니면 no-op)
    print(f"[Phase 2] device={config.device}  amp={config.amp}  compile={config.compile_model}")
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
                # φ 공간 생성 품질 일습(FID/KID/Precision/Recall/Density/Coverage).
                # Phase 3와 동일한 gen_metrics 함수를 사용한다.
                phi_metrics = compute_phi_metrics(clf, real_imgs_fid, gen_imgs,
                                                  config.eval_batch_size, config.device)
                clf_metrics = compute_classifier_metrics(clf, gen_imgs, config)
                plot_sample_grid(spec.name, gen_imgs, plots_dir / f"samples_{spec.name}.png")
            else:
                phi_metrics = {"fid_phi": float("nan"), "kid_phi": float("nan"),
                               "kid_phi_std": float("nan"), "precision_phi": float("nan"),
                               "recall_phi": float("nan"), "density_phi": float("nan"),
                               "coverage_phi": float("nan")}
                clf_metrics = {"classifier_confidence": float("nan"), "balance_error": float("nan"), "frac_class_0": float("nan")}

            row: dict = {
                "schedule": spec.name, "seed": run_seed,
                **cov, **phi_metrics, **clf_metrics,
                "mean_mse":       per_lam["mean_mse"],
                "transition_mse": per_lam["transition_mse"],
                "low_noise_mse":  per_lam["low_noise_mse"],
                "high_noise_mse": per_lam["high_noise_mse"],
            }
            summary_rows.append(row)
            print(f"[Phase 2]   FID={phi_metrics['fid_phi']:.2f}  KID={phi_metrics['kid_phi']:.4f}  "
                  f"M={cov['coverage_m']:.4f}  mean_mse={per_lam['mean_mse']:.4f}")

    # 7. Statistical aggregation across seeds (통계 분석 틀)
    #    - aggregate_over_seeds : schedule별 mean/std/sem/n 집계
    #    - significance_tests   : baseline 대비 paired t-test/Wilcoxon (seed≥2일 때)
    #    - aggregate_per_lambda : per-λ MSE 곡선의 seed 간 평균/표준편차
    agg_rows  = aggregate_over_seeds(summary_rows)
    sig_rows  = significance_tests(summary_rows, config.baseline_schedule, metric="fid_phi")
    per_lambda_agg = aggregate_per_lambda(all_per_lambda)

    # 8. Save artifacts
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_metrics_csv(summary_rows, out_dir / "metrics_summary.csv")
    save_per_lambda_csv(all_per_lambda, out_dir / "per_lambda_metrics.csv")
    save_aggregated_csv(agg_rows, out_dir / "metrics_aggregated.csv")
    save_significance_md(sig_rows, config.baseline_schedule, "fid_phi",
                         config.num_seeds, out_dir / "significance.md")
    (out_dir / "stats.json").write_text(json.dumps({
        "baseline": config.baseline_schedule, "num_seeds": config.num_seeds,
        "aggregated": agg_rows, "significance": sig_rows,
        "per_lambda_aggregated": per_lambda_agg,
    }, indent=2), encoding="utf-8")
    save_summary_md(config, specs, summary_rows, agg_rows, sig_rows, dmsr_info,
                    out_dir / "summary.md")

    # 9. Plots
    plot_per_lambda_mse(all_per_lambda, config, trans_low, trans_high, lambda_r_star,
                        plots_dir / "per_lambda_mse.png")
    if config.n_generate > 0 and not any(math.isnan(r.get("fid_phi", float("nan"))) for r in summary_rows):
        plot_fid_summary(summary_rows, plots_dir / "fid_summary.png")
        plot_fid_with_error_bars(agg_rows, plots_dir / "fid_mean_std.png")
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
    p.add_argument("--micro-batch-size", type=int, default=None,
                   help="GPU에 한 번에 올릴 학습 batch 크기. batch-size는 유지하고 gradient accumulation으로 업데이트한다.")
    p.add_argument("--eval-grid-size", type=int,   default=40)
    p.add_argument("--seed",           type=int,   default=20260526)
    # num_seeds ≥ 3 으로 주면 seed 간 통계 집계·유의성 검정이 의미를 갖는다.
    p.add_argument("--num-seeds",      type=int,   default=1,
                   help="seed 반복 횟수. ≥3 이면 유의성 검정이 활성화된다.")
    p.add_argument("--device",         type=str,   default="auto",
                   help="'auto', 'cuda', 'mps', or 'cpu'.")
    p.add_argument("--ddim-steps",     type=int,   default=50)
    p.add_argument("--n-generate",     type=int,   default=5000)
    p.add_argument("--gen-batch-size", type=int,   default=500)
    # ── GPU 가속 옵션 ─────────────────────────────────────────────────────────
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="혼합정밀. auto=CUDA면 bf16. 정확 재현이 필요하면 fp32.")
    p.add_argument("--compile", dest="compile_model", action="store_true",
                   help="torch.compile 활성화(기본 OFF; nvcc 등 환경이 받쳐줄 때만).")
    # 비교 schedule 구성 (Phase 3와 동일한 sweep 인자)
    p.add_argument("--s-values",       type=float, nargs="+", default=None,
                   help="DMSR-Normal의 폭 s 후보들 (예: --s-values 1.5 0.8 0.3).")
    p.add_argument("--laplace-b",      type=float, default=0.5,
                   help="DMSR-Laplace의 폭 b.")
    p.add_argument("--baseline-schedule", type=str, default="cosine_vp",
                   help="유의성 검정에서 기준이 되는 schedule 이름.")
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
        micro_batch_size=args.micro_batch_size,
        eval_grid_size=args.eval_grid_size,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=device,
        ddim_steps=args.ddim_steps,
        n_generate=args.n_generate,
        gen_batch_size=args.gen_batch_size,
        amp=args.amp,
        compile_model=args.compile_model,
        s_values=tuple(args.s_values) if args.s_values else (0.3, 0.8, 1.5, 2.5, 4.0),
        laplace_b=args.laplace_b,
        baseline_schedule=args.baseline_schedule,
        data_root=args.data_root,
    )
    run(config, args.out_root)


if __name__ == "__main__":
    main()
