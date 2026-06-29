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
# 모든 plot 텍스트는 ASCII + 그리스문자만 쓴다(한글은 DejaVu에 글리프 없어 □로 깨짐).
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch

from classification_metrics import classification_report
from gpu_perf import configure_backends
from stats_analysis import (
    LOWER_IS_BETTER,
    aggregate_over_seeds, aggregate_per_lambda, aggregated_csv_fieldnames,
    per_lambda_excess_and_skill,
    significance_tests,
)

from .config import CLASS_PAIRS, PRESETS, ExperimentConfig, ScheduleSpec
from .experiment import (
    build_schedules, compute_coverage_metrics, compute_dmsr_phi, compute_fid,
    compute_phi_metrics, eval_classifier_confidence, evaluate_classifier_predictions,
    eval_per_lambda_mse, gather_real_images, generate_samples_batched, sample_schedule,
    schedule_density, train_one_schedule,
)
from .models import resolve_device, seed_everything, train_classifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_DIR / "results"


# ── Plotting ───────────────────────────────────────────────────────────────────

def _pair_tag(cfg: ExperimentConfig) -> str:
    """plot 제목 앞에 붙일 클래스 쌍 태그 (낱장 PNG도 어떤 쌍인지 알아보게)."""
    return f"[{cfg.class_pair}]"


def _schedule_colors(names: list[str]) -> dict:
    cmap = plt.get_cmap("tab20" if len(names) > 10 else "tab10")
    return {n: cmap(i % cmap.N) for i, n in enumerate(names)}


def _maybe_log_x(ax, values: list[float], ratio: float = 8.0) -> bool:
    pos = [v for v in values if np.isfinite(v) and v > 0]
    if len(pos) >= 2 and max(pos) / max(min(pos), 1e-9) > ratio:
        ax.set_xscale("log")
        return True
    return False


def _annotate_barh(ax, values: list[float], fmt: str = "{:.1f}") -> None:
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.annotate(fmt.format(v), (v, i), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7.5)


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
    """설계된 p_train(λ)의 *해석적* 밀도를 매끄러운 곡선으로 그린다(샘플 히스토그램 X).

    이전 히스토그램은 계단형 + 표본잡음으로 대칭 분포도 비대칭처럼 보였다. 닫힌 형태
    밀도를 직접 평가하므로 매끄럽고 정확히 대칭이다.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    lam = np.linspace(cfg.lambda_min, cfg.lambda_max, 1000)
    colors = _schedule_colors([s.name for s in specs])
    peak_heights: list[float] = []
    for spec in specs:
        dens = schedule_density(spec, lam, cfg)
        ax.plot(lam, dens, linewidth=1.8, label=spec.name, color=colors[spec.name])
        peak_heights.append(float(np.nanmax(dens)))
    ax.axvspan(t_low, t_high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    # y 상한을 '가장 높은 봉우리'에 맞춰 어떤 곡선도 위로 잘리지 않게 한다(overflow 방지).
    ymax = (max(peak_heights) * 1.05) if peak_heights else 1.0
    ax.set_ylim(0, max(ymax, 1e-3))
    ax.set_xlim(cfg.lambda_min, cfg.lambda_max)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("density p_train(lambda)")
    ax.set_title(f"{_pair_tag(cfg)} Training noise distributions p_train(lambda)  [analytic]\n"
                 "(shading = DMSR transition; dashed = lambda_R*)")
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
    pair_tag: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    colors = _schedule_colors([str(r["schedule"]) for r in records])
    for rec in records:
        ax.plot(rec["lambda_grid"], rec["per_lambda_mse"], linewidth=1.9,
                label=str(rec["schedule"]), color=colors[str(rec["schedule"])])
    ax.axvspan(t_low, t_high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("epsilon-prediction MSE")
    ax.set_title(f"{pair_tag} Per-lambda denoising MSE by schedule".strip())
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_lambda_learnability(
    per_lambda_agg: dict, baseline: str, t_low: float, center: float, t_high: float,
    out_path: Path, pair_tag: str = "",
) -> None:
    """의사결정용: λ별 학습난이도 + headroom (어디에 학습 mass를 더 주면 도움이 되나).

    초록=best achievable(λ별 최소 MSE, 도달 가능한 바닥), 파랑=baseline, 음영=headroom
    (baseline−envelope, >0이면 baseline이 그 λ를 덜 뽑아 개선 여지가 있음). 고-λ의 MSE→1은
    환원불가(ε 예측불가)라 absolute가 아니라 headroom을 본다.
    """
    if not per_lambda_agg:
        return
    grid = np.array(next(iter(per_lambda_agg.values()))["lambda_grid"], dtype=float)
    curves = [np.array(v["mse_mean"], dtype=float) for v in per_lambda_agg.values()
              if np.array(v["lambda_grid"], dtype=float).shape == grid.shape]
    if not curves:
        return
    M = np.vstack(curves)
    envelope = M.min(axis=0)
    base = (np.array(per_lambda_agg[baseline]["mse_mean"], dtype=float)
            if baseline in per_lambda_agg else M.mean(axis=0))
    headroom = np.clip(base - envelope, 0.0, None)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for v in per_lambda_agg.values():
        ax.plot(v["lambda_grid"], v["mse_mean"], color="#BBBBBB", linewidth=0.9, zorder=1)
    ax.fill_between(grid, envelope, base, where=(base > envelope),
                    color="#F2B84B", alpha=0.40, zorder=2, label="headroom (room to improve)")
    ax.plot(grid, envelope, color="#268C6C", linewidth=2.4, zorder=3,
            label="best achievable (min over schedules)")
    ax.plot(grid, base, color="#2457A6", linewidth=2.2, zorder=3, label=f"baseline ({baseline})")
    ax.axvspan(t_low, t_high, color="#F2B84B", alpha=0.20)
    ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.3)
    j = int(np.argmax(headroom))
    if headroom[j] > 1e-4:
        ax.annotate(f"max headroom @ lambda={grid[j]:.2f}", (grid[j], base[j]),
                    xytext=(0, 20), textcoords="offset points", ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("epsilon-prediction MSE (lower = better learned)")
    ax.set_title(f"{pair_tag} Per-lambda learnability: where does adding training mass help?\n"
                 "grey=schedules, green=best achievable, blue=baseline, shaded=headroom".strip())
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_curves(histories: dict, specs, seed: int, out_path: Path, pair_tag: str = "") -> None:
    """schedule별 학습 loss 곡선(log-y). 수렴/발산 진단."""
    colors = _schedule_colors([s.name for s in specs])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    plotted = False
    for spec in specs:
        hist = histories.get(f"{spec.name}_seed{seed}")
        if not hist:
            continue
        ax.plot([h["step"] for h in hist], [h["train_mse"] for h in hist],
                marker="o", markersize=2.5, linewidth=1.6, label=spec.name, color=colors[spec.name])
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("train epsilon-prediction MSE  [log]")
    ax.set_title(f"{pair_tag} Training loss by schedule (seed={seed})".strip())
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_precision_recall(summary_rows, out_path: Path, pair_tag: str = "") -> None:
    """Precision(품질) vs Recall(다양성) 산점도 — mode collapse 진단. 색 = Inception-FID."""
    rows = [r for r in summary_rows
            if np.isfinite(float(r.get("precision_phi", float("nan"))))
            and np.isfinite(float(r.get("recall_phi", float("nan"))))]
    if not rows:
        return
    rec  = [float(r["recall_phi"]) for r in rows]
    prec = [float(r["precision_phi"]) for r in rows]
    col  = [float(r.get("fid_inception", float("nan"))) for r in rows]
    if not any(np.isfinite(c) for c in col):
        col = [float(r.get("fid_phi", float("nan"))) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sc = ax.scatter(rec, prec, c=col, cmap="viridis_r", s=90, edgecolors="k", linewidths=0.5)
    for r, x, y in zip(rows, rec, prec):
        ax.annotate(str(r["schedule"]), (x, y), xytext=(5, 3), textcoords="offset points", fontsize=7)
    fig.colorbar(sc, ax=ax, label="FID - lower (brighter) is better")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Recall (phi) - diversity ->")
    ax.set_ylabel("Precision (phi) - quality ->")
    ax.set_title(f"{pair_tag} Precision-Recall by schedule (top-right is better)".strip())
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _fmt_metric(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def render_metric_table(agg_rows, panels, title: str, out_path: Path) -> None:
    """생성 품질 지표를 '표 이미지'로 (행=schedule, 열=지표). 열별 best 수치만 bold.

    panels: [(key, 표시이름, lower_is_better), ...]. headline(첫 지표) 기준 좋은 순 정렬.
    """
    rows = [r for r in agg_rows if r.get("schedule")]
    if not rows:
        return
    hk, _, hlow = panels[0]
    def _sortkey(r):
        v = float(r.get(f"{hk}_mean", float("nan")))
        if not np.isfinite(v):
            return float("inf")
        return v if hlow else -v
    rows = sorted(rows, key=_sortkey)

    col_labels = ["schedule"] + [lab for _, lab, _ in panels]
    cell_text: list[list[str]] = []
    cols_vals: list[list[float]] = [[] for _ in panels]
    for r in rows:
        line = [str(r["schedule"])]
        for ci, (k, _, _) in enumerate(panels):
            v = float(r.get(f"{k}_mean", float("nan")))
            cols_vals[ci].append(v)
            line.append(_fmt_metric(v))
        cell_text.append(line)

    best_row_per_col: list[int | None] = []
    for ci, (_, _, lower) in enumerate(panels):
        col = np.asarray(cols_vals[ci], dtype=float)
        fin = np.where(np.isfinite(col))[0]
        if fin.size == 0:
            best_row_per_col.append(None)
            continue
        sub = col[fin]
        best_row_per_col.append(int(fin[np.argmin(sub) if lower else np.argmax(sub)]))

    nrow, ncol = len(rows), len(col_labels)
    fig, ax = plt.subplots(figsize=(1.7 * ncol + 1.0, 0.42 * nrow + 1.3))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    table.auto_set_column_width(col=list(range(ncol)))
    for j in range(ncol):
        table[(0, j)].set_facecolor("#EAEAEA")
    for i in range(1, nrow + 1):
        table[(i, 0)].set_facecolor("#F7F7F7")
    for ci, br in enumerate(best_row_per_col):
        if br is not None:
            table[(br + 1, ci + 1)].get_text().set_fontweight("bold")
    ax.set_title(title, fontsize=12, pad=14)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_metric_overview(agg_rows, out_path: Path, pair_tag: str = "") -> None:
    """생성 품질 지표 표 (행=schedule, 열=지표). 헤드라인=Inception-FID. best 수치만 bold."""
    panels = [
        ("fid_inception", "FID-Incep ↓", True),
        ("fid_phi",       "FID(phi) ↓", True),
        ("kid_phi",       "KID(phi) ↓", True),
        ("precision_phi", "Precision ↑", False),
        ("recall_phi",    "Recall ↑", False),
        ("mean_mse",      "meanMSE ↓", True),
    ]
    title = f"{pair_tag} Generation-quality metrics (best per column in bold)".strip()
    render_metric_table(agg_rows, panels, title, out_path)


def plot_confusion_matrix(report: dict, out_path: Path, pair_tag: str = "") -> None:
    """φ 분류기 confusion matrix 히트맵 (행=정답, 열=예측; 행 정규화 비율로 색)."""
    cm = np.array(report["confusion_matrix"], dtype=float)
    names = report["class_names"]
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.9 * n + 3.0, 0.9 * n + 2.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row (true-class) normalized ratio")
    ax.set_xticks(range(n)); ax.set_xticklabels(names)
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)", ha="center", va="center",
                    fontsize=9, color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_title(f"{pair_tag} Confusion matrix (phi classifier, test set)\n"
                 f"accuracy = {report['accuracy']*100:.2f}%  (n={report['n_samples']})".strip())
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_classification_report(report: dict, out_path: Path, pair_tag: str = "") -> None:
    """클래스별 Precision/Recall/F1 묶음 막대 (y축 [0,1] 고정)."""
    per = report["per_class"]
    names = [f"class {p['class']}\n(n={p['support']})" for p in per]
    prec  = [p["precision"] for p in per]
    rec   = [p["recall"] for p in per]
    f1    = [p["f1"] for p in per]
    x = np.arange(len(per)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * len(per) + 3.0), 5.0))
    for bar, vals in [(ax.bar(x - w, prec, w, label="Precision", color="#2457A6"), prec),
                      (ax.bar(x, rec, w, label="Recall", color="#268C6C"), rec),
                      (ax.bar(x + w, f1, w, label="F1", color="#C28A2B"), f1)]:
        for rect, v in zip(bar, vals):
            ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7)
    ax.axhline(report["macro_f1"], color="#C28A2B", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"macro-F1={report['macro_f1']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score (0-1, higher is better)")
    ax.set_title(f"{pair_tag} Per-class Precision / Recall / F1 (phi classifier, test set)\n"
                 f"accuracy={report['accuracy']*100:.2f}%  macro-F1={report['macro_f1']:.3f}".strip())
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_lambda_decomposition(
    diag: dict, baseline: str, t_low: float, center: float, t_high: float, out_path: Path,
    pair_tag: str = "",
) -> None:
    """per-lambda MSE decomposition: (top) excess vs baseline, (bottom) skill = 1 - MSE.

    excess_s(lam) = mse_s(lam) - mse_baseline(lam): the common (data, lam)-dependent Bayes
    floor cancels in the difference, so this isolates the p_train effect per noise level
    (>0 means this schedule learned that noise band worse than the baseline). skill = 1 - MSE
    is bounded by 1 (trivial eps-predictor 0 gives MSE = Var(eps) = 1), showing whether the
    U-Net actually learned at each noise level (1 perfect, 0 trivial, <0 worse than trivial).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 8.2), sharex=True)
    for name, v in diag.items():
        grid = v["lambda_grid"]
        ax1.plot(grid, v["excess_mean"], linewidth=1.8, label=name)
        ax2.plot(grid, v["skill_mean"], linewidth=1.8, label=name)
    for ax in (ax1, ax2):
        ax.axvspan(t_low, t_high, color="#F2B84B", alpha=0.20)
        ax.axvline(center, color="#C23B22", linestyle="--", linewidth=1.3)
    ax1.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax1.set_ylabel(f"excess eps-MSE (vs {baseline})")
    ax1.set_title(f"{pair_tag} Per-lambda MSE decomposition\n(top) excess vs baseline: Bayes floor cancels, p_train effect remains".strip())
    ax1.legend(ncol=2, fontsize=7.5, frameon=False)
    ax2.axhline(1.0, color="#268C6C", linewidth=1.0, alpha=0.6, linestyle=":")
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax2.set_ylim(top=1.02)
    ax2.set_xlabel("lambda = log SNR")
    ax2.set_ylabel("skill = 1 - MSE")
    ax2.set_title("(bottom) skill / R^2: did the U-Net learn at each noise level")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fid_vs_s(rows: list[dict], out_path: Path) -> None:
    dmsr_rows = [r for r in rows if "dmsr_normal" in str(r["schedule"])]
    if not dmsr_rows:
        return
    s_vals = [float(str(r["schedule"]).split("_s")[-1]) for r in dmsr_rows]
    fid_vals = [float(r.get("fid_inception") or float("nan")) for r in dmsr_rows]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(s_vals, fid_vals, s=80, zorder=3)
    ax.plot(s_vals, fid_vals, linewidth=1.5, linestyle="--")
    for s, f, r in zip(s_vals, fid_vals, dmsr_rows):
        ax.annotate(r["schedule"], (s, f), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("s (schedule width)")
    ax.set_ylabel("FID (Inception)")
    ax.set_title("FID vs s (DMSR-Normal schedules)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fid_vs_m(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for r in rows:
        fid = float(r.get("fid_inception") or float("nan"))
        m = float(r["coverage_m"])
        ax.scatter(m, fid, s=70, zorder=3)
        ax.annotate(str(r["schedule"]), (m, fid),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("M (transition mass)")
    ax.set_ylabel("FID (Inception)")
    ax.set_title("FID vs T_R coverage M")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_classifier_report_md(report: dict, out_path: Path) -> None:
    """φ 분류기 테스트셋 평가(Accuracy·Confusion·per-class P/R/F1)를 사람이 읽게 저장."""
    names = report["class_names"]; cm = report["confusion_matrix"]
    lines = [
        "# Phase 3 — φ Classifier Evaluation (test set)", "",
        "DMSR_φ·FID-φ 지표가 모두 이 분류기 φ의 feature 위에서 계산되므로 φ 자체의 분류 성능을 점검한다.",
        "(생성 샘플은 unconditional이라 정답이 없어 이 지표들은 테스트셋에서만 정의된다.)", "",
        f"- classes: {', '.join(names)}",
        f"- test samples: {report['n_samples']}",
        f"- **accuracy: {report['accuracy']*100:.2f}%**",
        f"- macro avg — precision {report['macro_precision']:.4f}, recall {report['macro_recall']:.4f}, F1 {report['macro_f1']:.4f}",
        "", "## Per-class metrics", "",
        "| class | precision | recall | F1 | support |", "|---|---:|---:|---:|---:|",
    ]
    for p in report["per_class"]:
        lines.append(f"| {p['class']} | {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} | {p['support']} |")
    lines += ["", "## Confusion matrix (행=정답 true, 열=예측 pred)", "",
              "| true \\\\ pred | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for i, name in enumerate(names):
        lines.append(f"| **{name}** | " + " | ".join(str(c) for c in cm[i]) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    "schedule", "seed",
    # 헤드라인 + φ 공간 생성 품질 지표 (Phase 2와 동일한 φ 지표 정의)
    "fid_inception",
    "fid_phi", "kid_phi", "kid_phi_std",
    "precision_phi", "recall_phi", "density_phi", "coverage_phi",
    "classifier_confidence", "balance_error",
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


def save_aggregated_csv(agg_rows: list[dict], path: Path) -> None:
    """seed 간 집계 결과(mean/std/sem/n)를 CSV로 저장한다 (컬럼은 동적 생성)."""
    if not agg_rows:
        return
    fieldnames = aggregated_csv_fieldnames(agg_rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in agg_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


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
        f"# Phase 3 유의성 검정 ({metric}, baseline = {baseline})", "",
        f"- 지표 해석: **{metric}** 은 {arrow} 좋다.",
        f"- 설계: 동일 seed에서 측정한 값끼리 짝지은 **paired** 비교 (n_seeds={num_seeds}).",
        "- mean_diff = (해당 schedule − baseline)의 seed 평균. "
        "낮을수록 좋은 지표면 음수가 개선을 뜻한다.",
        "- p-value 는 paired t-test 기준 (seed ≥ 5 이면 Wilcoxon 보조 제시).", "",
        "| schedule | n_pairs | mean_diff | improved | t p-value | wilcoxon p-value | status |",
        "|---|---:|---:|:---:|---:|---:|---|",
    ]

    def _fmt(x: float) -> str:
        return "nan" if not np.isfinite(x) else f"{x:.4g}"

    for r in sig_rows:
        imp = "—" if r["improved_vs_baseline"] is None else ("✓" if r["improved_vs_baseline"] else "✗")
        lines.append(
            f"| {r['schedule']} | {r['n_pairs']} | {_fmt(r['mean_diff'])} | {imp} "
            f"| {_fmt(r['t_pvalue'])} | {_fmt(r['wilcoxon_pvalue'])} | {r['status']} |"
        )
    if num_seeds < 2:
        lines += [
            "",
            "> ⚠️ seed가 1개뿐이라 유의성 검정을 수행할 수 없습니다. "
            "`--num-seeds 3` 이상으로 재실행해 이 표를 채웁니다.",
        ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary_md(
    cfg: ExperimentConfig,
    specs: list[ScheduleSpec],
    rows: list[dict],
    agg_rows: list[dict],
    sig_rows: list[dict],
    lambda_r_star: float,
    t_low: float,
    t_high: float,
    out_path: Path,
) -> None:
    sorted_rows = sorted(rows, key=lambda r: float(r.get("fid_inception") or 1e9))
    lines = [
        "# Phase 3 CIFAR-10 Two-Class Experiment Summary", "",
        "## Config", "",
        f"- class pair: {cfg.class_pair}",
        f"- lambda range: [{cfg.lambda_min}, {cfg.lambda_max}]",
        f"- rho: {cfg.rho}",
        f"- lambda_R*: {lambda_r_star:.4f}",
        f"- T_R: [{t_low:.4f}, {t_high:.4f}]",
        f"- train steps: {cfg.train_steps}",
        f"- batch size: {cfg.batch_size}",
        f"- num_seeds: {cfg.num_seeds}", "",
        "## Schedules", "",
    ]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.note}")

    # ── per-run 결과 (schedule × seed) ────────────────────────────────────────
    lines += [
        "",
        "## Per-run Results (sorted by Inception-FID)", "",
        "| rank | schedule | seed | FID(Incep) | FID(φ) | KID(φ) | M | mean MSE | clf conf |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(sorted_rows, 1):
        lines.append(
            f"| {rank} | {r['schedule']} | {r['seed']} | "
            f"{float(r.get('fid_inception') or float('nan')):.2f} | "
            f"{float(r.get('fid_phi') or float('nan')):.2f} | "
            f"{float(r.get('kid_phi') or float('nan')):.4f} | "
            f"{float(r['coverage_m']):.4f} | {float(r['mean_mse']):.5f} | "
            f"{float(r['classifier_confidence']):.4f} |"
        )

    # ── seed 간 집계 (mean ± std) ─────────────────────────────────────────────
    def _ms(row: dict, metric: str, fmt: str = ".3f") -> str:
        mean, std = row.get(f"{metric}_mean", float("nan")), row.get(f"{metric}_std", float("nan"))
        if not np.isfinite(mean):
            return "nan"
        if not np.isfinite(std):
            return f"{mean:{fmt}}"
        return f"{mean:{fmt}} ± {std:{fmt}}"

    agg_sorted = sorted(
        agg_rows,
        key=lambda r: r.get("fid_inception_mean", float("inf"))
        if np.isfinite(r.get("fid_inception_mean", float("nan"))) else float("inf"),
    )
    lines += [
        "",
        f"## Aggregated over seeds (n_seeds = {cfg.num_seeds})", "",
        "FID·KID는 낮을수록, Precision·Coverage는 높을수록 좋다. seed 1개면 std 미표시.",
        "",
        "| rank | schedule | n | FID(Incep) | FID(φ) | KID(φ) | Precision(φ) | Coverage(φ) |",
        "|---:|---|---:|---|---|---|---|---|",
    ]
    for rank, row in enumerate(agg_sorted, 1):
        lines.append(
            f"| {rank} | {row['schedule']} | {int(row['n_seeds'])} "
            f"| {_ms(row, 'fid_inception', '.2f')} | {_ms(row, 'fid_phi', '.2f')} "
            f"| {_ms(row, 'kid_phi', '.4f')} | {_ms(row, 'precision_phi')} "
            f"| {_ms(row, 'coverage_phi')} |"
        )

    # ── baseline 대비 paired 유의성 검정 (Inception-FID 기준) ──────────────────
    lines += [
        "",
        f"## Significance vs baseline ({cfg.baseline_schedule}, metric = Inception-FID)", "",
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
        "- 헤드라인은 InceptionV3 기반 FID(CIFAR 표준). φ 공간 FID/KID/PRDC는 Phase 2와 "
        "동일한 정의로, 두 phase를 잇는 비교축이다.",
        "- FID는 품질·다양성을 뭉뚱그리므로 Precision(품질)·Recall/Coverage(다양성)를 함께 본다.",
        "- KID는 표본이 적을 때 FID보다 신뢰성이 높다.",
        "- `coverage_m`(transition 질량) 하나만으로 우수성을 말할 수 없으며 full-range support "
        "와의 균형이 중요하다(Phase 1 결론).",
        f"- 유의성 검정은 seed가 부족하면(현재 {cfg.num_seeds}개) 건너뛴다. "
        "`--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main runner ────────────────────────────────────────────────────────────────

def _write_run_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_console_summary(cfg, clf_report, agg_rows, per_lambda_diag, out_dir) -> None:
    """실행이 끝나면 핵심 결과를 터미널에 한 화면으로 요약(ASCII만). 헤드라인=Inception-FID."""
    bar = "=" * 70
    print("\n" + bar)
    print(f" PHASE 3 SUMMARY  |  CIFAR {cfg.class_pair}  |  n_seeds={cfg.num_seeds}")
    print(bar)
    if clf_report is not None:
        print(f"  phi classifier test accuracy = {clf_report['accuracy']*100:.2f}%  "
              f"(macro-F1 = {clf_report['macro_f1']:.3f})")
    # 헤드라인 = Inception-FID, 없으면 fid_phi로 폴백
    key = "fid_inception" if any(np.isfinite(r.get("fid_inception_mean", float("nan"))) for r in agg_rows) else "fid_phi"
    rows = [r for r in agg_rows if np.isfinite(r.get(f"{key}_mean", float("nan")))]
    rows = sorted(rows, key=lambda r: r[f"{key}_mean"])
    if rows:
        skill_of = {s: float(np.nanmean(np.asarray(v["skill_mean"], dtype=float)))
                    for s, v in per_lambda_diag.items()}
        print(f"\n  {key} ranking (lower = better):")
        print(f"  {'rank':>4}  {'schedule':<24} {'FID':>9}   {'mean skill':>10}")
        for i, r in enumerate(rows, 1):
            mean = r[f"{key}_mean"]; std = r.get(f"{key}_std", float("nan"))
            std_s = f" +/-{std:.1f}" if np.isfinite(std) else ""
            mk = "  <== BEST" if i == 1 else ("   (baseline)" if r["schedule"] == cfg.baseline_schedule else "")
            sk = skill_of.get(r["schedule"], float("nan"))
            print(f"  {i:>4}. {r['schedule']:<24} {mean:>8.2f}{std_s:<6} {sk:>10.3f}{mk}")
        best = rows[0]["schedule"]
        base_row = next((r for r in rows if r["schedule"] == cfg.baseline_schedule), None)
        print()
        if base_row is not None and best != cfg.baseline_schedule:
            d = rows[0][f"{key}_mean"] - base_row[f"{key}_mean"]
            print(f"  Best: {best} (FID {rows[0][f'{key}_mean']:.2f}); baseline {cfg.baseline_schedule} "
                  f"FID {base_row[f'{key}_mean']:.2f} -> best is {'better' if d < 0 else 'worse'} by {abs(d):.2f}.")
        elif best == cfg.baseline_schedule:
            print(f"  Best = baseline ({cfg.baseline_schedule}). No schedule beat it this run.")
    if cfg.num_seeds < 2:
        print("  Note: single seed -> significance test skipped (use --num-seeds >= 3).")
    print(f"\n  Full report : {out_dir}/summary.md")
    print(f"  Key plots   : metric_overview.png, lambda_learnability.png, per_lambda_decomposition.png")
    print(bar + "\n")


def run(cfg: ExperimentConfig, out_root: Path) -> Path:
    """실험 실행. 폴더명에 실행시각+모드+종료상태(__RUNNING/__OK/__FAILED_<stage>)를 담고,
    run_meta.json에 시작/종료/소요/단계/에러를 기록한다 (Phase 2와 동일)."""
    seed_everything(cfg.seed)
    device = cfg.device
    configure_backends()

    start_dt = datetime.now()
    run_id = start_dt.strftime("%Y%m%d_%H%M%S")
    base_name = f"{run_id}_{cfg.run_name}_{cfg.class_pair}_steps{cfg.train_steps}_seed{cfg.num_seeds}"
    out_dir = out_root / "phase3" / (base_name + "__RUNNING")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # matplotlib 캐시 dir은 rename되는 out_dir 밖(out_root)에 둔다(rename 충돌 방지).
    mpl_cfg_dir = out_root / ".matplotlib"
    mpl_cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg_dir))

    meta: dict = {
        "status": "RUNNING", "stage": "setup", "run_name": cfg.run_name,
        "class_pair": cfg.class_pair, "train_steps": cfg.train_steps,
        "num_seeds": cfg.num_seeds, "device": device, "amp": cfg.amp,
        "started_at": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None, "duration_sec": None, "error": None,
    }
    meta_path = out_dir / "run_meta.json"
    _write_run_meta(meta_path, meta)

    def _stage(name: str) -> None:
        meta["stage"] = name
        _write_run_meta(meta_path, meta)

    def _finalize(status: str) -> Path:
        end_dt = datetime.now()
        meta["status"] = status
        meta["finished_at"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        meta["duration_sec"] = round((end_dt - start_dt).total_seconds(), 1)
        _write_run_meta(meta_path, meta)
        suffix = "__OK" if status == "OK" else f"__FAILED_{meta['stage']}"
        final_dir = out_dir.with_name(base_name + suffix)
        if final_dir.exists():
            final_dir = out_dir.with_name(base_name + suffix + f"_{end_dt.strftime('%H%M%S')}")
        out_dir.rename(final_dir)
        return final_dir

    try:
        return _run_body(cfg, out_dir, plots_dir, device, meta, _stage, _finalize)
    except BaseException as exc:  # noqa: BLE001 — 모든 중단을 폴더명에 남긴다
        meta["error"] = f"{type(exc).__name__}: {exc}"
        failed_dir = _finalize("FAILED")
        print(f"[Phase 3] FAILED at stage '{meta['stage']}' -> {failed_dir}")
        raise


def _run_body(cfg, out_dir, plots_dir, device, meta, _stage, _finalize) -> Path:
    print(f"\n=== Phase 3: {cfg.class_pair} | device={device} | amp={cfg.amp} "
          f"| compile={cfg.compile_model} ===")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    tag = _pair_tag(cfg)

    # 1. Train classifier
    _stage("train_classifier")
    print("\n[1/6] Training feature classifier...")
    classifier = train_classifier(cfg, device)
    torch.save(classifier.state_dict(), out_dir / "classifier.pt")

    # 1b. φ 분류기 테스트셋 평가 (정답 레이블 있는 유일한 곳; Accuracy/Confusion/P/R/F1)
    _stage("eval_classifier")
    y_true, y_pred = evaluate_classifier_predictions(classifier, cfg, device)
    cls_names = cfg.class_pair.split("_vs_")
    if len(cls_names) != 2:
        cls_names = ["0", "1"]
    clf_report = classification_report(y_true, y_pred, n_classes=2, class_names=cls_names)
    (out_dir / "classifier_report.json").write_text(
        json.dumps(clf_report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_classifier_report_md(clf_report, out_dir / "classifier_report.md")
    plot_confusion_matrix(clf_report, plots_dir / "classifier_confusion_matrix.png", pair_tag=tag)
    plot_classification_report(clf_report, plots_dir / "classifier_pr_f1.png", pair_tag=tag)
    print(f"  phi classifier: acc={clf_report['accuracy']*100:.2f}%  macro-F1={clf_report['macro_f1']:.4f}")

    # 2. Compute DMSR_phi
    _stage("compute_dmsr")
    print("\n[2/6] Computing empirical DMSR_phi(lambda)...")
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
    _stage("build_schedules")
    print("\n[3/6] Building noise schedules...")
    specs = build_schedules(lambda_r_star, cfg)
    meta["n_schedules"] = len(specs)
    meta["schedules"] = [s.name for s in specs]
    (out_dir / "schedules.json").write_text(
        json.dumps([asdict(s) for s in specs], indent=2), encoding="utf-8"
    )
    plot_schedule_densities(specs, cfg, t_low, lambda_r_star, t_high, device,
                             plots_dir / "schedule_densities.png")

    # 4. Train + evaluate each schedule
    print("\n[4/6] Training diffusion models...")
    all_per_lambda: list[dict] = []
    summary_rows: list[dict] = []
    histories: dict[str, list[dict]] = {}

    # φ-feature 기반 지표(FID/KID/PRDC)의 '진짜 분포' 쪽 표본을 한 번만 모아 재사용
    real_imgs_phi = gather_real_images(cfg, device, cfg.n_gen_samples)

    for seed_idx in range(cfg.num_seeds):
        run_seed = cfg.seed + seed_idx
        for spec in specs:
            _stage(f"train:{spec.name}:seed{run_seed}")
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

            # 헤드라인 지표: InceptionV3 기반 FID(CIFAR 표준)
            fid_inception = float("nan")
            if cfg.compute_fid:
                print("  Inception-FID...")
                fid_inception = compute_fid(gen_imgs, cfg, device)

            # 보조·일관 비교축: φ 공간 FID/KID/Precision/Recall/Density/Coverage
            # (Phase 2와 동일한 gen_metrics 함수 — 두 phase를 같은 정의로 잇는다)
            print("  phi-space metrics (FID/KID/PRDC)...")
            phi_metrics = compute_phi_metrics(classifier, real_imgs_phi, gen_imgs, device)

            print("  Coverage metrics...")
            m, s = compute_coverage_metrics(spec, lambda_grid, slope, t_low, t_high, cfg, device)

            print(f"  [{spec.name}] FID(Inception)={fid_inception:.2f}  "
                  f"FID(phi)={phi_metrics['fid_phi']:.2f}  KID(phi)={phi_metrics['kid_phi']:.4f}  "
                  f"M={m:.4f}  S={s:.4f}  mean_mse={mean_mse:.5f}")

            rec = {
                "schedule": spec.name, "seed": run_seed,
                "fid_inception": fid_inception,
                **phi_metrics,
                "classifier_confidence": clf_conf,
                "balance_error": bal_err,
                "coverage_m": m, "expected_s": s,
                "mean_mse": mean_mse, "transition_mse": transition_mse,
                # skill = 1 - MSE (trivial eps-hat=0 -> MSE=1). U-Net 학습 정도(R^2/skill).
                "mean_skill": 1.0 - mean_mse,
                "transition_skill": 1.0 - transition_mse,
                "transition_low": t_low, "lambda_r_star": lambda_r_star,
                "transition_high": t_high,
                "lambda_grid": lam_g.tolist(),
                "per_lambda_mse": per_mse.tolist(),
            }
            all_per_lambda.append(rec)
            summary_rows.append({k: v for k, v in rec.items() if not isinstance(v, list)})
            torch.save(model.state_dict(), out_dir / f"model_{spec.name}_seed{run_seed}.pt")

    # 5. Statistical aggregation across seeds (Phase 2와 동일한 공용 통계 틀)
    #    헤드라인 지표(Inception-FID)에 대해 baseline 대비 paired 유의성을 검정한다.
    _stage("aggregate")
    agg_rows = aggregate_over_seeds(summary_rows)
    sig_rows = significance_tests(summary_rows, cfg.baseline_schedule, metric="fid_inception")
    sig_rows_phi = significance_tests(summary_rows, cfg.baseline_schedule, metric="fid_phi")
    per_lambda_agg = aggregate_per_lambda(all_per_lambda)
    # per-lambda MSE -> excess(vs baseline) + skill(=1-MSE) 분해 (재실행 없는 후처리, Phase 2와 동일).
    per_lambda_diag = per_lambda_excess_and_skill(
        per_lambda_agg, cfg.baseline_schedule, trivial_mse=1.0)

    # 6. Save all results
    _stage("save_artifacts")
    print("\n[5/6] Saving results...")
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_metrics_csv(summary_rows, out_dir / "metrics_summary.csv")
    save_per_lambda_csv(all_per_lambda, out_dir / "per_lambda_metrics.csv")
    save_aggregated_csv(agg_rows, out_dir / "metrics_aggregated.csv")
    save_significance_md(sig_rows, cfg.baseline_schedule, "fid_inception",
                         cfg.num_seeds, out_dir / "significance.md")
    (out_dir / "stats.json").write_text(json.dumps({
        "baseline": cfg.baseline_schedule, "num_seeds": cfg.num_seeds,
        "aggregated": agg_rows,
        "significance_fid_inception": sig_rows,
        "significance_fid_phi": sig_rows_phi,
        "per_lambda_aggregated": per_lambda_agg,
        "per_lambda_diagnostics": per_lambda_diag,
    }, indent=2), encoding="utf-8")
    save_summary_md(cfg, specs, summary_rows, agg_rows, sig_rows, lambda_r_star, t_low, t_high,
                    out_dir / "summary.md")

    # 7. Plots
    _stage("plots")
    print("\n[6/6] Plotting...")
    plot_per_lambda_mse(all_per_lambda, lambda_grid, t_low, lambda_r_star, t_high,
                        plots_dir / "per_lambda_mse.png", pair_tag=tag)
    plot_per_lambda_decomposition(per_lambda_diag, cfg.baseline_schedule,
                                  t_low, lambda_r_star, t_high,
                                  plots_dir / "per_lambda_decomposition.png", pair_tag=tag)
    plot_lambda_learnability(per_lambda_agg, cfg.baseline_schedule,
                             t_low, lambda_r_star, t_high,
                             plots_dir / "lambda_learnability.png", pair_tag=tag)
    plot_training_curves(histories, specs, cfg.seed, plots_dir / "training_curves.png", pair_tag=tag)
    plot_fid_vs_s(summary_rows, plots_dir / "fid_vs_s.png")
    plot_fid_vs_m(summary_rows, plots_dir / "fid_vs_m.png")
    # fid_mean_std는 metric_overview의 FID(Inception) 패널(오차막대 포함)에 흡수되어 삭제.
    plot_precision_recall(summary_rows, plots_dir / "precision_recall.png", pair_tag=tag)
    plot_metric_overview(agg_rows, plots_dir / "metric_overview.png", pair_tag=tag)

    final_dir = _finalize("OK")
    print(f"\n[Phase 3] OK ({meta['duration_sec']}s) -> {final_dir}")
    _print_console_summary(cfg, clf_report, agg_rows, per_lambda_diag, final_dir)
    return final_dir


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 CIFAR-10 DMSR experiment")
    p.add_argument("--preset", choices=list(PRESETS), default="full")
    p.add_argument("--class-pair", choices=list(CLASS_PAIRS), default="airplane_vs_automobile")
    p.add_argument("--device", default="auto",
                   help="'auto' (best available), 'cuda', 'mps', or 'cpu'.")
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None,
                   help="GPU에 한 번에 올릴 학습 batch 크기. batch-size는 유지하고 gradient accumulation으로 업데이트한다.")
    p.add_argument("--clf-batch-size", type=int, default=None,
                   help="classifier 학습 batch 크기. None이면 기본값 256.")
    p.add_argument("--gen-batch-size", type=int, default=None,
                   help="DDIM 생성 시 한 번에 만들 이미지 수. 총 생성 수는 바꾸지 않고 GPU 순간 부하만 낮춘다.")
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--num-seeds", type=int, default=None)
    p.add_argument("--width-values", type=float, nargs="+", default=None,
                   help="Normal s / Laplace b 공통 폭 sweep (예: --width-values 0.5 1.5 4.0).")
    p.add_argument("--no-center0", dest="include_center0", action="store_false",
                   help="중심 0 대조군(Normal@0, Laplace@0=Hang)을 빼고 λ_R* 중심만 실행.")
    p.set_defaults(include_center0=True)
    p.add_argument("--studentt-scales", type=float, nargs="+", default=None,
                   help="DMSR-Student-t(λ_R* 중심, 무거운 꼬리) 폭 sweep (예: --studentt-scales 1.0 2.5).")
    p.add_argument("--studentt-df", type=float, default=3.0,
                   help="Student-t 자유도 ν (꼬리 두께; 작을수록 두꺼움, 1=Cauchy, ∞=Normal).")
    p.add_argument("--no-linear", dest="include_linear", action="store_false",
                   help="VP linear-β baseline schedule을 빼고 실행.")
    p.add_argument("--no-uniform", dest="include_uniform", action="store_false",
                   help="uniform baseline schedule을 빼고 실행.")
    p.add_argument("--include-cosmix", action="store_true",
                   help="(구) DMSR×cosine 혼합 schedule도 추가(기본 OFF; Student-t로 대체됨).")
    p.add_argument("--mix-weights", type=float, nargs="+", default=None,
                   help="(--include-cosmix 일 때) 혼합 N 비율 w 후보들.")
    p.add_argument("--mix-scale", type=float, default=1.0,
                   help="(--include-cosmix 일 때) 혼합 N 폭.")
    p.set_defaults(include_linear=True, include_uniform=True)
    p.add_argument("--baseline-schedule", type=str, default="cosine_vp",
                   help="유의성 검정에서 기준이 되는 schedule 이름.")
    p.add_argument("--no-fid", action="store_true")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    # ── GPU 가속 옵션 ─────────────────────────────────────────────────────────
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader 병렬 워커 수 (서버 CPU 코어에 맞춰 조정).")
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="DataLoader 워커당 미리 읽을 batch 수. 낮출수록 CPU/RAM 순간 부하 감소.")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="혼합정밀. auto=CUDA면 bf16. 정확 재현이 필요하면 fp32.")
    p.add_argument("--compile", dest="compile_model", action="store_true",
                   help="torch.compile 활성화(기본 OFF; nvcc 등 환경이 받쳐줄 때만).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[Phase 3] Using device: {device}")

    preset = PRESETS[args.preset]
    cfg = ExperimentConfig(
        class_pair=args.class_pair,
        clf_epochs=preset["clf_epochs"],
        clf_batch_size=args.clf_batch_size if args.clf_batch_size is not None else 256,
        train_steps=args.train_steps if args.train_steps is not None else preset["train_steps"],
        batch_size=args.batch_size if args.batch_size is not None else 128,
        micro_batch_size=args.micro_batch_size,
        n_gen_samples=preset["n_gen_samples"],
        gen_batch_size=args.gen_batch_size if args.gen_batch_size is not None else 512,
        dmsr_n_samples=preset["dmsr_n_samples"],
        dmsr_grid_size=preset["dmsr_grid_size"],
        eval_n_samples=preset["eval_n_samples"],
        compute_fid=(not args.no_fid) and preset["compute_fid"],
        seed=args.seed,
        num_seeds=args.num_seeds if args.num_seeds is not None else preset["num_seeds"],
        width_values=tuple(args.width_values) if args.width_values else (0.5, 1.5, 4.0),
        include_center0=args.include_center0,
        studentt_scales=tuple(args.studentt_scales) if args.studentt_scales else (1.0, 2.5),
        studentt_df=args.studentt_df,
        include_linear=args.include_linear,
        include_uniform=args.include_uniform,
        include_cosmix=args.include_cosmix,
        mix_weights=tuple(args.mix_weights) if args.mix_weights else (0.5, 0.8),
        mix_scale=args.mix_scale,
        baseline_schedule=args.baseline_schedule,
        device=device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        amp=args.amp,
        compile_model=args.compile_model,
        data_root=str(PROJECT_DIR / "data"),
    )
    run(cfg, args.out_root)


if __name__ == "__main__":
    main()
