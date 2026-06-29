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
# 모든 plot 텍스트는 ASCII + 그리스문자(λ,φ,ρ)·기본 수학기호만 쓴다. 한글은 기본 폰트
# (DejaVu Sans)에 글리프가 없어 □로 깨지고, 서버(Linux)에도 한글 폰트가 없을 수 있어
# 의도적으로 영어만 렌더한다. 유니코드 마이너스(−)도 폰트에 따라 깨질 수 있어 ASCII로.
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch

from gpu_perf import configure_backends

from .config import ExperimentConfig, ScheduleSpec
from .models import (
    FeatureClassifier, load_mnist_two_class, resolve_device,
    seed_everything, train_feature_classifier,
)
from classification_metrics import classification_report

from .experiment import (
    LOWER_IS_BETTER,
    aggregate_over_seeds, aggregate_per_lambda, aggregated_csv_fieldnames,
    build_schedules, compute_classifier_metrics, compute_coverage_metrics,
    compute_empirical_dmsr, compute_empirical_transition, compute_phi_metrics,
    ddim_generate, evaluate_classifier_predictions, evaluate_per_lambda,
    per_lambda_excess_and_skill, region_curve_mean,
    sample_schedule, schedule_density, significance_tests, train_one_schedule,
)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _span_kw() -> dict:
    return dict(color="#F2B84B", alpha=0.22)


def _pair_tag(config: ExperimentConfig) -> str:
    """plot 제목 앞에 붙일 클래스 쌍 태그 (낱장 PNG도 어떤 쌍인지 알아보게)."""
    return f"[{config.digits[0]} vs {config.digits[1]}]"


def _schedule_colors(names: list[str]) -> dict:
    """schedule 이름 → 색을 일관되게 매핑(여러 그림에서 같은 색을 쓰도록).

    10개를 넘어가면 tab20을 써서 색이 겹치지 않게 한다(넓은 s + 혼합까지 추가하면
    schedule 수가 10개를 넘는다).
    """
    cmap = plt.get_cmap("tab20" if len(names) > 10 else "tab10")
    return {n: cmap(i % cmap.N) for i, n in enumerate(names)}


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
    """설계된 p_train(λ)의 *해석적* 밀도를 매끄러운 곡선으로 그린다.

    예전에는 80k 샘플을 뽑아 히스토그램으로 그려서 (1) 계단형이고 (2) 표본잡음 때문에
    대칭 분포도 비대칭처럼 보였다. schedule은 우리가 닫힌 형태로 설계한 분포이므로,
    진짜 밀도 p_train(λ)를 직접 평가해 그리면 매끄럽고 정확히 대칭이다.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    lam = np.linspace(config.lambda_min, config.lambda_max, 1000)
    colors = _schedule_colors([s.name for s in specs])
    peak_heights: list[float] = []
    for spec in specs:
        dens = schedule_density(spec, lam, config)
        ax.plot(lam, dens, linewidth=1.8, label=spec.name, color=colors[spec.name])
        peak_heights.append(float(np.nanmax(dens)))
    ax.axvspan(transition_low, transition_high, **_span_kw())
    ax.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.4)
    # y 상한을 '가장 높은 봉우리'에 맞춰 어떤 곡선도 위로 잘리지 않게 한다(overflow 방지).
    ymax = (max(peak_heights) * 1.05) if peak_heights else 1.0
    ax.set_ylim(0, max(ymax, 1e-3))
    ax.set_xlim(config.lambda_min, config.lambda_max)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("density p_train(lambda)")
    ax.set_title(f"{_pair_tag(config)} Training noise distributions p_train(lambda)  [analytic]\n"
                 "(shading = DMSR transition region; dashed = lambda_R*)")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_lambda_mse(
    results, config: ExperimentConfig,
    transition_low, transition_high, lambda_r_star, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    colors = _schedule_colors([str(r["schedule"]) for r in results])
    for r in results:
        ax.plot(r["lambda_grid"], r["per_lambda_mse"], linewidth=1.9,
                label=str(r["schedule"]), color=colors[str(r["schedule"])])
    ax.axvspan(transition_low, transition_high, **_span_kw())
    ax.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("epsilon-prediction MSE")
    ax.set_title(f"{_pair_tag(config)} Per-lambda denoising MSE by schedule\n"
                 "(clean end = large lambda, noisy end = small lambda)")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_lambda_decomposition(
    diag: dict, baseline: str, config: ExperimentConfig,
    transition_low, transition_high, lambda_r_star, out_path: Path,
) -> None:
    """per-λ MSE를 두 view로 분해: (위) baseline 대비 excess, (아래) skill/R².

    - excess: 같은 λ의 공통 Bayes floor가 소거돼 p_train 차이만 남는다. 0선(=baseline)
      위면 그 noise 구간을 baseline보다 못 배운 것. 절대 MSE 곡선의 공통 U자 모양에
      가려 안 보이던 schedule 간 차이가 여기서 드러난다.
    - skill = 1 − MSE: trivial 예측(ε̂=0, MSE=1) 대비 학습 정도. 1=완벽, 0=trivial과
      동급, <0=trivial보다 나쁨. U-Net이 각 구간에서 실제로 학습됐는지 본다.
    """
    names = list(diag.keys())
    colors = _schedule_colors(names)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 8.2), sharex=True)

    for name in names:
        v = diag[name]
        grid = v["lambda_grid"]
        ax1.plot(grid, v["excess_mean"], linewidth=1.8, label=name, color=colors[name])
        ax2.plot(grid, v["skill_mean"], linewidth=1.8, label=name, color=colors[name])

    for ax in (ax1, ax2):
        ax.axvspan(transition_low, transition_high, **_span_kw())
        ax.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.3)
    ax1.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax1.set_ylabel(f"excess eps-MSE\n(vs {baseline}; >0 = learned worse)")
    ax1.set_title(f"{_pair_tag(config)} Per-lambda MSE decomposition (isolate p_train effect by noise level)\n"
                  "(top) excess vs baseline: common Bayes floor cancels, only schedule diff remains")
    ax1.legend(ncol=2, fontsize=7.5, frameon=False)

    ax2.axhline(1.0, color="#268C6C", linewidth=1.0, alpha=0.6, linestyle=":")
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax2.set_ylim(top=1.02)
    ax2.set_xlabel("lambda = log SNR  (clean end = large lambda, noisy end = small lambda)")
    ax2.set_ylabel("skill = 1 - MSE\n(1 perfect, 0 trivial, <0 worse than trivial)")
    ax2.set_title("(bottom) skill / R^2: did the U-Net learn at each noise level")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_lambda_learnability(
    per_lambda_agg: dict, baseline: str, config: ExperimentConfig,
    transition_low, transition_high, lambda_r_star, out_path: Path,
) -> None:
    """의사결정용 한 장: 'λ별로 어디에 학습 mass를 더 주면 도움이 되나'.

    - 회색  : 각 schedule의 per-λ MSE 곡선.
    - 초록  : best achievable = schedule들의 λ별 최소 MSE(=경험적으로 도달 가능한 바닥).
              고-λ(clean끝)에서 MSE→1로 오르는 건 ε가 본질적으로 예측 불가라 '환원 불가'한
              부분이지 학습 부족이 아니다. 그래서 절대 MSE가 아니라 아래 'headroom'을 본다.
    - 파랑  : baseline(보통 cosine)의 per-λ MSE.
    - 음영  : headroom = baseline − envelope (>0). 이 구간은 어떤 schedule이 baseline보다
              잘 배웠다는 뜻 → baseline이 그 λ를 덜 뽑고 있다는 신호 → **거기서 더 뽑으면
              개선 여지가 있다**. headroom이 0이면 모두 바닥이라 더 뽑아도 의미 없다.
    """
    if not per_lambda_agg:
        return
    grid = np.array(next(iter(per_lambda_agg.values()))["lambda_grid"], dtype=float)
    curves = [np.array(v["mse_mean"], dtype=float)
              for v in per_lambda_agg.values()
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
    ax.axvspan(transition_low, transition_high, **_span_kw())
    ax.axvline(lambda_r_star, color="#C23B22", linestyle="--", linewidth=1.3)
    j = int(np.argmax(headroom))
    if headroom[j] > 1e-4:
        ax.annotate(f"max headroom @ lambda={grid[j]:.2f}", (grid[j], base[j]),
                    xytext=(0, 20), textcoords="offset points", ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("lambda = log SNR")
    ax.set_ylabel("epsilon-prediction MSE (lower = better learned)")
    ax.set_title(f"{_pair_tag(config)} Per-lambda learnability: where does adding training mass help?\n"
                 "grey=schedules, green=best achievable, blue=baseline, shaded=headroom (sample more here)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _maybe_log_x(ax, values: list[float], ratio: float = 8.0) -> bool:
    """양수 값들의 동적 범위가 크면(기본 >8배) x축을 log로 바꾼다(아웃라이어 가독성).

    s=0.3처럼 붕괴한 schedule의 FID(수백)가 선형축을 지배해 나머지 막대를 0처럼
    보이게 만드는 스케일 왜곡을 막는다(실측 예: cosine 36 vs s0.3 478 ≈ 13배).
    반환값으로 log 적용 여부를 알린다.
    """
    pos = [v for v in values if np.isfinite(v) and v > 0]
    if len(pos) >= 2 and max(pos) / max(min(pos), 1e-9) > ratio:
        ax.set_xscale("log")
        return True
    return False


def _annotate_barh(ax, values: list[float], fmt: str = "{:.1f}", log: bool = False) -> None:
    """수평 막대 끝에 실제 수치를 적어 CSV와 대조 가능하게 한다."""
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        ax.annotate(fmt.format(v), (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.5)


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


def plot_training_curves(histories: dict, specs, config: ExperimentConfig, out_path: Path) -> None:
    """schedule별 학습 loss(train ε-MSE) 곡선. 수렴/발산을 한눈에 본다.

    seed가 여럿이면 첫 seed의 곡선을 그린다(형태 비교가 목적). loss는 schedule에 따라
    수십 배 차이가 나므로 log-y로 그려 작은 값과 큰 값이 함께 보이게 한다.
    """
    colors = _schedule_colors([s.name for s in specs])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    plotted = False
    for spec in specs:
        hist = histories.get(f"{spec.name}_seed{config.seed}")
        if not hist:
            continue
        steps = [h["step"] for h in hist]
        mse   = [h["train_mse"] for h in hist]
        ax.plot(steps, mse, marker="o", markersize=2.5, linewidth=1.6,
                label=spec.name, color=colors[spec.name])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("train ε-prediction MSE  [log]")
    ax.set_title(f"Training loss by schedule (seed={config.seed})")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_precision_recall(summary_rows, out_path: Path, pair_tag: str = "") -> None:
    """Precision(품질) vs Recall(다양성) 산점도 — mode collapse 진단용.

    좌하단(둘 다 낮음)=붕괴, 우상단=품질·다양성 모두 좋음. FID 한 값으로는 안 보이는
    "왜 나쁜가"(품질이 문제인지 다양성이 문제인지)를 분리해 보여 준다.
    """
    rows = [r for r in summary_rows
            if np.isfinite(float(r.get("precision_phi", float("nan"))))
            and np.isfinite(float(r.get("recall_phi", float("nan"))))]
    if not rows:
        return
    rec  = [float(r["recall_phi"]) for r in rows]
    prec = [float(r["precision_phi"]) for r in rows]
    fids = [float(r.get("fid_phi", float("nan"))) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sc = ax.scatter(rec, prec, c=fids, cmap="viridis_r", s=90, edgecolors="k", linewidths=0.5)
    for r, x, y in zip(rows, rec, prec):
        ax.annotate(str(r["schedule"]), (x, y), xytext=(5, 3),
                    textcoords="offset points", fontsize=7)
    fig.colorbar(sc, ax=ax, label="FID (phi) - lower (brighter) is better")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Recall (phi) - diversity ->")
    ax.set_ylabel("Precision (phi) - quality ->")
    ax.set_title(f"{pair_tag} Precision-Recall by schedule (top-right is better)".strip())
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _fmt_metric(v: float) -> str:
    """지표 값을 자릿수 자동으로 포맷(큰 수는 적게, 0~1은 소수 3자리)."""
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
    """생성 품질 지표를 '표 이미지'로 렌더한다 (행=schedule, 열=지표).

    각 열(지표)의 best — ↓지표는 최소, ↑지표는 최대 — 수치만 **bold** 처리한다.
    panels: [(key, 표시이름, lower_is_better), ...]. headline(첫 지표) 기준 좋은 순 정렬.
    """
    rows = [r for r in agg_rows if r.get("schedule")]
    if not rows:
        return
    # headline(첫 지표) 기준 정렬 — 유한값 우선, 좋은(↓면 작은) 순으로 위에서 아래
    hk, _, hlow = panels[0]
    def _sortkey(r):
        v = float(r.get(f"{hk}_mean", float("nan")))
        if not np.isfinite(v):
            return float("inf")
        return v if hlow else -v
    rows = sorted(rows, key=_sortkey)
    sched = [str(r["schedule"]) for r in rows]

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

    # 열별 best 행 index (유한값 중 ↓=최소 / ↑=최대)
    best_row_per_col: list[int | None] = []
    for ci, (_, _, lower) in enumerate(panels):
        col = np.asarray(cols_vals[ci], dtype=float)
        fin = np.where(np.isfinite(col))[0]
        if fin.size == 0:
            best_row_per_col.append(None)
            continue
        sub = col[fin]
        best_row_per_col.append(int(fin[np.argmin(sub) if lower else np.argmax(sub)]))

    nrow, ncol = len(sched), len(col_labels)
    fig, ax = plt.subplots(figsize=(1.7 * ncol + 1.0, 0.42 * nrow + 1.3))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    table.auto_set_column_width(col=list(range(ncol)))
    # 헤더 행 옅은 회색 배경(가독성). bold는 'best 수치'에만 쓰기 위해 헤더는 일반체.
    for j in range(ncol):
        table[(0, j)].set_facecolor("#EAEAEA")
    # schedule 이름 열도 옅은 배경
    for i in range(1, nrow + 1):
        table[(i, 0)].set_facecolor("#F7F7F7")
    # 열별 best 수치만 bold
    for ci, br in enumerate(best_row_per_col):
        if br is not None:
            table[(br + 1, ci + 1)].get_text().set_fontweight("bold")
    ax.set_title(title, fontsize=12, pad=14)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_metric_overview(agg_rows, out_path: Path, pair_tag: str = "") -> None:
    """생성 품질 지표 표 (행=schedule, 열=지표). 지표별 best 수치만 bold."""
    panels = [
        ("fid_phi",       "FID(phi) ↓", True),
        ("kid_phi",       "KID(phi) ↓", True),
        ("precision_phi", "Precision ↑", False),
        ("recall_phi",    "Recall ↑", False),
        ("coverage_phi",  "Coverage ↑", False),
        ("mean_mse",      "meanMSE ↓", True),
    ]
    title = f"{pair_tag} Generation-quality metrics (best per column in bold; phi space)".strip()
    render_metric_table(agg_rows, panels, title, out_path)


def plot_confusion_matrix(report: dict, out_path: Path) -> None:
    """φ 분류기 confusion matrix 히트맵 (행=정답 digit, 열=예측 digit).

    각 칸에 개수와, 행 기준 정규화 비율(=recall 관점)을 함께 적어 어떤 숫자를 어떤
    숫자로 헷갈렸는지 한눈에 본다. 색은 행 정규화 비율(0~1)로 칠해 클래스 불균형이
    있어도 오분류 패턴이 드러나게 한다(스케일 고정 0~1).
    """
    cm = np.array(report["confusion_matrix"], dtype=float)
    names = report["class_names"]
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sum, 1.0)        # 행(정답) 기준 비율
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.9 * n + 3.0, 0.9 * n + 2.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row (true-class) normalized ratio")
    ax.set_xticks(range(n)); ax.set_xticklabels(names)
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("true digit")
    for i in range(n):
        for j in range(n):
            txt = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_title(f"Confusion matrix (φ classifier, test set)\n"
                 f"accuracy = {report['accuracy']*100:.2f}%  (n={report['n_samples']})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_classification_report(report: dict, out_path: Path) -> None:
    """클래스별 Precision/Recall/F1 묶음 막대 + macro 평균. y축은 [0,1] 고정.

    세 지표 모두 비율(0~1)이라 같은 축에 올려도 스케일 왜곡이 없다. 막대 위에 수치를
    적어 classifier_report.json 과 대조 가능하게 한다.
    """
    per = report["per_class"]
    names = [f"digit {p['class']}\n(n={p['support']})" for p in per]
    prec  = [p["precision"] for p in per]
    rec   = [p["recall"] for p in per]
    f1    = [p["f1"] for p in per]
    x = np.arange(len(per))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * len(per) + 3.0), 5.0))
    bars = [
        (ax.bar(x - w, prec, w, label="Precision", color="#2457A6"), prec),
        (ax.bar(x,      rec,  w, label="Recall",    color="#268C6C"), rec),
        (ax.bar(x + w,  f1,   w, label="F1",        color="#C28A2B"), f1),
    ]
    for bar, vals in bars:
        for rect, v in zip(bar, vals):
            ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    # macro 평균을 가로 점선으로 표시
    ax.axhline(report["macro_f1"], color="#C28A2B", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"macro-F1={report['macro_f1']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score (0-1, higher is better)")
    ax.set_title(f"Per-class Precision / Recall / F1 (φ classifier, test set)\n"
                 f"accuracy={report['accuracy']*100:.2f}%  macro-F1={report['macro_f1']:.3f}")
    ax.legend(fontsize=8, frameon=False, ncol=2)
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


def save_classifier_report_md(report: dict, out_path: Path) -> None:
    """φ 분류기 테스트셋 평가(Accuracy·Confusion·per-class P/R/F1)를 사람이 읽게 저장."""
    names = report["class_names"]
    cm = report["confusion_matrix"]
    lines = [
        "# Phase 2 — φ Classifier Evaluation (test set)", "",
        "DMSR_φ(λ)와 FID-φ 계열 지표가 모두 이 분류기 φ의 feature 위에서 계산되므로, "
        "φ 자체의 분류 성능을 먼저 점검한다. (생성 샘플은 unconditional이라 정답이 없어 "
        "이 지표들은 **테스트셋**에서만 정의된다.)", "",
        f"- digits: {', '.join(names)}",
        f"- test samples: {report['n_samples']}",
        f"- **accuracy: {report['accuracy']*100:.2f}%**",
        f"- macro avg — precision {report['macro_precision']:.4f}, "
        f"recall {report['macro_recall']:.4f}, F1 {report['macro_f1']:.4f}",
        f"- weighted avg — precision {report['weighted_precision']:.4f}, "
        f"recall {report['weighted_recall']:.4f}, F1 {report['weighted_f1']:.4f}", "",
        "## Per-class metrics", "",
        "| digit | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in report["per_class"]:
        lines.append(f"| {p['class']} | {p['precision']:.4f} | {p['recall']:.4f} "
                     f"| {p['f1']:.4f} | {p['support']} |")
    # confusion matrix (행=정답, 열=예측)
    lines += ["", "## Confusion matrix (행=정답 true, 열=예측 pred)", "",
              "| true \\\\ pred | " + " | ".join(names) + " |",
              "|---" * (len(names) + 1) + "|"]
    for i, name in enumerate(names):
        lines.append(f"| **{name}** | " + " | ".join(str(c) for c in cm[i]) + " |")
    lines += ["", "> precision=특정 숫자로 예측한 것 중 실제로 그 숫자인 비율, "
              "recall=실제 그 숫자인 것 중 맞춘 비율, F1=둘의 조화평균(불균형에 강건)."]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary_md(
    config: ExperimentConfig,
    specs: list[ScheduleSpec],
    summary_rows: list[dict],
    agg_rows: list[dict],
    sig_rows: list[dict],
    dmsr_info: dict,
    clf_report: dict,
    per_lambda_diag: dict,
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
        "## φ Classifier Evaluation (test set)", "",
        "DMSR_φ·FID-φ의 기반인 분류기 φ의 테스트셋 분류 성능 (자세한 표·confusion은 "
        "`classifier_report.md` / `classifier_report.json`, 그림은 plots 참고).",
        f"- accuracy: **{clf_report['accuracy']*100:.2f}%** "
        f"(n={clf_report['n_samples']}), macro-F1: {clf_report['macro_f1']:.4f}",
        "  " + ", ".join(
            f"digit {p['class']}: P={p['precision']:.3f}/R={p['recall']:.3f}/F1={p['f1']:.3f}"
            for p in clf_report["per_class"]
        ), "",
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

    # ── per-λ MSE 분해 (excess vs baseline + skill) ────────────────────────────
    lines += [
        "",
        "## Per-λ MSE 분해 (transition 구간)", "",
        "per-λ denoising MSE를 noise level별로 분해해 p_train 효과를 본다. "
        f"**excess** = (해당 schedule − {config.baseline_schedule})의 같은 λ MSE 차이(공통 "
        "Bayes floor가 소거됨; transition 구간 평균, >0이면 baseline보다 못 배움). "
        "**skill** = 1 − MSE (1 완벽, 0 trivial 예측과 동급). 그림은 "
        "`plots/per_lambda_decomposition.png`.", "",
        "| schedule | transition excess (vs baseline) | transition skill | mean skill |",
        "|---|---:|---:|---:|",
    ]
    diag_rows = []
    for name, v in per_lambda_diag.items():
        t_exc = region_curve_mean(v["lambda_grid"], v["excess_mean"], low, high, "transition")
        t_skl = region_curve_mean(v["lambda_grid"], v["skill_mean"], low, high, "transition")
        m_skl = float(np.nanmean(np.asarray(v["skill_mean"], dtype=float)))
        diag_rows.append((name, t_exc, t_skl, m_skl))
    for name, t_exc, t_skl, m_skl in sorted(diag_rows, key=lambda r: (np.inf if not np.isfinite(r[1]) else r[1])):
        exc_s = "0 (baseline)" if name == config.baseline_schedule else (
            "nan" if not np.isfinite(t_exc) else f"{t_exc:+.4f}")
        lines.append(f"| {name} | {exc_s} | {t_skl:.4f} | {m_skl:.4f} |")

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
        "- `dmsr_normal_s*`는 폭을 좁게→넓게 sweep한다. narrow는 clean끝을 못 배워 "
        "붕괴하기 쉽고, 넓힐수록(또는 `dmsr_cosmix_w*`처럼 cosine을 섞어 full-range "
        "support를 확보할수록) 회복하는지를 본다. Precision–Recall 그림으로 붕괴(좌하단) "
        "여부를 확인하라.",
        "- λ_R*는 DMSR_φ(λ)의 수치 미분 peak에서 경험적으로 추정한다. "
        "이 값은 Phase 3로 넘어가지 않으며 CIFAR에서 독립적으로 재추정한다.",
        f"- 유의성 검정은 seed가 부족하면(현재 {config.num_seeds}개) 건너뛴다. "
        "Phase 3에서 `--num-seeds 3` 이상으로 재실행하면 위 표가 채워진다.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main Run ───────────────────────────────────────────────────────────────────

def _write_run_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def run(config: ExperimentConfig, out_root: Path) -> Path:
    """실험을 실행하고 결과 폴더를 반환한다.

    폴더명에 **실행 시각 + 모드 정보 + 종료 상태**를 담아 한눈에 구분되게 한다.
      - 실행 중       : <ts>_<run_name>_d0v1_steps20000_seed1__RUNNING
      - 정상 완료     : ...__OK
      - 중간 실패     : ...__FAILED_<stage>   (어느 단계에서 멈췄는지)
    또 run_meta.json 에 시작/종료 시각·소요 시간·단계·에러 메시지를 기록한다.
    """
    seed_everything(config.seed)
    configure_backends()   # cudnn autotuner + TF32 (CUDA가 아니면 no-op)
    print(f"[Phase 2] device={config.device}  amp={config.amp}  compile={config.compile_model}")

    start_dt  = datetime.now()
    run_id    = start_dt.strftime("%Y%m%d_%H%M%S")
    d0, d1    = config.digits
    base_name = (f"{run_id}_{config.run_name}_d{d0}v{d1}"
                 f"_steps{config.train_steps}_seed{config.num_seeds}")
    out_dir   = out_root / "phase2" / (base_name + "__RUNNING")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "status": "RUNNING", "stage": "setup",
        "run_name": config.run_name, "digits": [d0, d1],
        "train_steps": config.train_steps, "num_seeds": config.num_seeds,
        "device": config.device, "amp": config.amp,
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
        if final_dir.exists():                       # 동일 이름 충돌 시 초까지 붙여 유일화
            final_dir = out_dir.with_name(base_name + suffix + f"_{end_dt.strftime('%H%M%S')}")
        out_dir.rename(final_dir)
        return final_dir

    try:
        return _run_body(config, out_dir, plots_dir, meta, _stage, _finalize)
    except BaseException as exc:                      # noqa: BLE001 — 모든 중단을 폴더명에 남긴다
        meta["error"] = f"{type(exc).__name__}: {exc}"
        failed_dir = _finalize("FAILED")
        print(f"[Phase 2] FAILED at stage '{meta['stage']}' -> {failed_dir}")
        raise


def _print_console_summary(config, dmsr_info, clf_report, agg_rows, sig_rows,
                           per_lambda_diag, out_dir) -> None:
    """실행이 끝나면 핵심 결과를 터미널에 한 화면으로 요약 출력한다 (ASCII만)."""
    bar = "=" * 70
    d0, d1 = config.digits
    print("\n" + bar)
    print(f" PHASE 2 SUMMARY  |  MNIST digits {d0} vs {d1}  |  n_seeds={config.num_seeds}")
    print(bar)
    print(f"  lambda_R* = {dmsr_info['lambda_r_star']:+.3f}   "
          f"transition T_R = [{dmsr_info['transition_low']:+.2f}, {dmsr_info['transition_high']:+.2f}]")
    print(f"  phi classifier test accuracy = {clf_report['accuracy']*100:.2f}%  "
          f"(macro-F1 = {clf_report['macro_f1']:.3f})")

    rows = [r for r in agg_rows if np.isfinite(r.get("fid_phi_mean", float("nan")))]
    rows = sorted(rows, key=lambda r: r["fid_phi_mean"])
    if rows:
        # skill 조회용 매핑 (mean over lambda)
        skill_of = {s: float(np.nanmean(np.asarray(v["skill_mean"], dtype=float)))
                    for s, v in per_lambda_diag.items()}
        print(f"\n  FID(phi) ranking  (lower = better):")
        print(f"  {'rank':>4}  {'schedule':<24} {'FID':>9}   {'mean skill':>10}")
        for i, r in enumerate(rows, 1):
            mean = r["fid_phi_mean"]; std = r.get("fid_phi_std", float("nan"))
            std_s = f" +/-{std:.1f}" if np.isfinite(std) else ""
            mk = "  <== BEST" if i == 1 else ("   (baseline)" if r["schedule"] == config.baseline_schedule else "")
            sk = skill_of.get(r["schedule"], float("nan"))
            print(f"  {i:>4}. {r['schedule']:<24} {mean:>8.2f}{std_s:<6} {sk:>10.3f}{mk}")

        best = rows[0]["schedule"]
        base_row = next((r for r in rows if r["schedule"] == config.baseline_schedule), None)
        print()
        if base_row is not None and best != config.baseline_schedule:
            delta = rows[0]["fid_phi_mean"] - base_row["fid_phi_mean"]
            rel = 100.0 * delta / base_row["fid_phi_mean"] if base_row["fid_phi_mean"] else float("nan")
            print(f"  Best: {best}  (FID {rows[0]['fid_phi_mean']:.2f})")
            print(f"  vs baseline {config.baseline_schedule} (FID {base_row['fid_phi_mean']:.2f}): "
                  f"{'better' if delta < 0 else 'worse'} by {abs(delta):.2f} ({abs(rel):.1f}%).")
        elif best == config.baseline_schedule:
            print(f"  Best = baseline ({config.baseline_schedule}). "
                  f"No schedule beat the baseline this run.")
    else:
        print("\n  (no FID computed -- n_generate=0?)")

    if config.num_seeds < 2:
        print("  Note: single seed -> significance test skipped (use 'final' / --num-seeds >= 3).")
    print(f"\n  Full report : {out_dir}/summary.md")
    print(f"  Key plots   : metric_overview.png, lambda_learnability.png, per_lambda_decomposition.png")
    print(bar + "\n")


def _run_body(config, out_dir, plots_dir, meta, _stage, _finalize) -> Path:
    # 1. Load data
    _stage("load_data")
    tr_imgs, tr_labels, te_imgs, te_labels = load_mnist_two_class(config)
    imgs_a = tr_imgs[tr_labels == 0]
    imgs_b = tr_imgs[tr_labels == 1]

    # 2. Train classifier φ
    _stage("train_classifier")
    print("[Phase 2] Training feature classifier φ...")
    clf = train_feature_classifier(tr_imgs, tr_labels, te_imgs, te_labels, config)
    torch.save(clf.state_dict(), out_dir / "classifier_phi.pt")

    # 2b. φ 분류기 자체의 분류 성능을 테스트셋에서 평가 (정답 레이블이 있는 유일한 곳).
    #     Accuracy/Precision/Recall/F1/Confusion — φ가 DMSR·FID-φ의 기반이라 신뢰도 점검.
    _stage("eval_classifier")
    y_true, y_pred = evaluate_classifier_predictions(clf, te_imgs, te_labels, config)
    clf_report = classification_report(
        y_true, y_pred, n_classes=len(config.digits),
        class_names=[str(d) for d in config.digits],
    )
    (out_dir / "classifier_report.json").write_text(
        json.dumps(clf_report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_confusion_matrix(clf_report, plots_dir / "classifier_confusion_matrix.png")
    plot_classification_report(clf_report, plots_dir / "classifier_pr_f1.png")
    print(f"[Phase 2] φ classifier: acc={clf_report['accuracy']*100:.2f}%  "
          f"macro-F1={clf_report['macro_f1']:.4f}")

    # 3. Compute empirical DMSR_φ(λ)
    _stage("compute_dmsr")
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
    _stage("build_schedules")
    specs = build_schedules(config, lambda_r_star)
    meta["n_schedules"] = len(specs)
    meta["schedules"] = [s.name for s in specs]
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
            _stage(f"train:{spec.name}:seed{run_seed}")
            print(f"[Phase 2] Training {spec.name} (seed={run_seed})...")
            model, history = train_one_schedule(spec, tr_imgs, config, run_seed)
            histories[f"{spec.name}_seed{run_seed}"] = history

            per_lam = evaluate_per_lambda(model, tr_imgs, config, trans_low, trans_high)
            per_lam.update({"schedule": spec.name, "seed": run_seed})
            all_per_lambda.append(per_lam)

            cov = compute_coverage_metrics(spec, config, trans_low, trans_high, dmsr_grid, dmsr_slope_abs)

            if config.n_generate > 0:
                _stage(f"generate:{spec.name}:seed{run_seed}")
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
                # skill = 1 − MSE (trivial ε̂=0 → MSE=1). U-Net 학습 정도(R²/skill score).
                "mean_skill":       1.0 - per_lam["mean_mse"],
                "transition_skill": 1.0 - per_lam["transition_mse"],
            }
            summary_rows.append(row)
            print(f"[Phase 2]   FID={phi_metrics['fid_phi']:.2f}  KID={phi_metrics['kid_phi']:.4f}  "
                  f"M={cov['coverage_m']:.4f}  mean_mse={per_lam['mean_mse']:.4f}")

    # 7. Statistical aggregation across seeds (통계 분석 틀)
    #    - aggregate_over_seeds : schedule별 mean/std/sem/n 집계
    #    - significance_tests   : baseline 대비 paired t-test/Wilcoxon (seed≥2일 때)
    #    - aggregate_per_lambda : per-λ MSE 곡선의 seed 간 평균/표준편차
    _stage("aggregate")
    agg_rows  = aggregate_over_seeds(summary_rows)
    sig_rows  = significance_tests(summary_rows, config.baseline_schedule, metric="fid_phi")
    per_lambda_agg = aggregate_per_lambda(all_per_lambda)
    # per-λ MSE를 baseline 대비 excess + skill(=1−MSE)로 분해 (재실행 없는 후처리).
    per_lambda_diag = per_lambda_excess_and_skill(
        per_lambda_agg, config.baseline_schedule, trivial_mse=1.0)

    # 8. Save artifacts
    _stage("save_artifacts")
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_metrics_csv(summary_rows, out_dir / "metrics_summary.csv")
    save_per_lambda_csv(all_per_lambda, out_dir / "per_lambda_metrics.csv")
    save_aggregated_csv(agg_rows, out_dir / "metrics_aggregated.csv")
    save_significance_md(sig_rows, config.baseline_schedule, "fid_phi",
                         config.num_seeds, out_dir / "significance.md")
    save_classifier_report_md(clf_report, out_dir / "classifier_report.md")
    (out_dir / "stats.json").write_text(json.dumps({
        "baseline": config.baseline_schedule, "num_seeds": config.num_seeds,
        "aggregated": agg_rows, "significance": sig_rows,
        "per_lambda_aggregated": per_lambda_agg,
        "per_lambda_diagnostics": per_lambda_diag,
    }, indent=2), encoding="utf-8")
    save_summary_md(config, specs, summary_rows, agg_rows, sig_rows, dmsr_info,
                    clf_report, per_lambda_diag, out_dir / "summary.md")

    # 9. Plots
    _stage("plots")
    plot_per_lambda_mse(all_per_lambda, config, trans_low, trans_high, lambda_r_star,
                        plots_dir / "per_lambda_mse.png")
    plot_per_lambda_decomposition(per_lambda_diag, config.baseline_schedule, config,
                                  trans_low, trans_high, lambda_r_star,
                                  plots_dir / "per_lambda_decomposition.png")
    # 의사결정용: λ별 학습난이도 + headroom (어디에 더 뽑으면 좋을지)
    plot_lambda_learnability(per_lambda_agg, config.baseline_schedule, config,
                             trans_low, trans_high, lambda_r_star,
                             plots_dir / "lambda_learnability.png")
    plot_training_curves(histories, specs, config, plots_dir / "training_curves.png")
    # 생성 품질 지표 종합. fid_summary/fid_mean_std는 metric_overview에 흡수되어 삭제됨.
    if config.n_generate > 0 and not any(math.isnan(r.get("fid_phi", float("nan"))) for r in summary_rows):
        tag = _pair_tag(config)
        plot_coverage_tradeoff(summary_rows, plots_dir / "coverage_vs_fid.png")
        plot_precision_recall(summary_rows, plots_dir / "precision_recall.png", pair_tag=tag)
        plot_metric_overview(agg_rows, plots_dir / "metric_overview.png", pair_tag=tag)

    final_dir = _finalize("OK")
    print(f"[Phase 2] OK ({meta['duration_sec']}s) -> {final_dir}")
    _print_console_summary(config, dmsr_info, clf_report, agg_rows, sig_rows,
                           per_lambda_diag, final_dir)
    return final_dir


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
    p.add_argument("--width-values",   type=float, nargs="+", default=None,
                   help="Normal s / Laplace b 공통 폭 sweep (예: --width-values 0.5 1.5 4.0).")
    p.add_argument("--no-center0",     dest="include_center0", action="store_false",
                   help="중심 0 대조군(Normal@0, Laplace@0=Hang)을 빼고 λ_R* 중심만 실행.")
    p.set_defaults(include_center0=True)
    p.add_argument("--studentt-scales", type=float, nargs="+", default=None,
                   help="DMSR-Student-t(λ_R* 중심, 무거운 꼬리)의 폭 sweep (예: --studentt-scales 1.0 2.5).")
    p.add_argument("--studentt-df",    type=float, default=3.0,
                   help="Student-t 자유도 ν (꼬리 두께; 작을수록 두꺼움, 1=Cauchy, ∞=Normal).")
    p.add_argument("--include-cosmix", action="store_true",
                   help="(구) DMSR×cosine 혼합 schedule도 추가(기본 OFF; Student-t로 대체됨).")
    p.add_argument("--mix-weights",    type=float, nargs="+", default=None,
                   help="(--include-cosmix 일 때) 혼합의 N 비율 w 후보들 (예: --mix-weights 0.5 0.8).")
    p.add_argument("--mix-scale",      type=float, default=1.0,
                   help="(--include-cosmix 일 때) 혼합 schedule에 쓰는 N(λ_R*, ·)의 폭.")
    p.add_argument("--no-linear",      dest="include_linear", action="store_false",
                   help="VP linear-β baseline schedule을 빼고 실행.")
    p.add_argument("--no-uniform",     dest="include_uniform", action="store_false",
                   help="uniform baseline schedule을 빼고 실행.")
    p.set_defaults(include_linear=True, include_uniform=True)
    p.add_argument("--baseline-schedule", type=str, default="cosine_vp",
                   help="유의성 검정에서 기준이 되는 schedule 이름.")
    p.add_argument("--run-name",       type=str,   default="phase2_mnist",
                   help="출력 폴더명 prefix (예: smoke 실행이면 phase2_mnist_smoke).")
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
        width_values=tuple(args.width_values) if args.width_values else (0.5, 1.5, 4.0),
        include_center0=args.include_center0,
        studentt_scales=tuple(args.studentt_scales) if args.studentt_scales else (1.0, 2.5),
        studentt_df=args.studentt_df,
        include_cosmix=args.include_cosmix,
        mix_weights=tuple(args.mix_weights) if args.mix_weights else (0.5, 0.8),
        mix_scale=args.mix_scale,
        include_linear=args.include_linear,
        include_uniform=args.include_uniform,
        baseline_schedule=args.baseline_schedule,
        run_name=args.run_name,
        data_root=args.data_root,
    )
    run(config, args.out_root)


if __name__ == "__main__":
    main()
