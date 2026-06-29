"""Phase 2·3 공용 통계 분석 모듈 (seed 간 집계 + 유의성 검정).

이 모듈은 Phase 2(MNIST 파이프라인 검증)와 Phase 3(CIFAR 본 실험)이 **동일한
통계 절차**로 결과를 해석하도록 하기 위해 분리해 둔 것이다. 두 phase가 같은 함수를
쓰므로 "Phase 2에서 검증한 통계 틀을 Phase 3에 그대로 적용한다"는 논리가 성립한다.

설계 원칙:
  (1) 집계(aggregation): 같은 schedule을 여러 seed로 돌린 결과를 mean±std±sem로 요약.
  (2) paired 비교: 동일한 seed_idx 안에서 모든 schedule이 같은 run_seed를 공유하므로
      (각 phase의 학습 루프 참고), schedule 간 비교는 '대응표본(paired)'이 된다.
      따라서 baseline 대비 seed별 차이 Δ를 만들어 paired t-test / Wilcoxon을 적용한다.

주의: seed가 1개뿐이면 분산을 추정할 수 없으므로 std/sem은 NaN이 되고 유의성 검정은
"seed 부족"으로 안전하게 건너뛴다. 의미 있는 검정은 num_seeds ≥ 3 을 권장한다.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


# "값이 작을수록 좋은" 지표 집합 (유의성 해석 시 개선 방향 판단용).
# 두 phase가 쓰는 지표 키를 모두 포함한다.
LOWER_IS_BETTER: set[str] = {
    "fid_phi", "fid_inception", "kid_phi",          # 분포 거리 (작을수록 좋음)
    "balance_error",                                # class 균형 오차
    "mean_mse", "transition_mse", "low_noise_mse", "high_noise_mse",
}

# 집계·검정에서 지표로 취급하지 않을 식별/메타 컬럼
_NON_METRIC_KEYS: set[str] = {"schedule", "seed"}


def detect_metric_keys(rows: list[dict]) -> list[str]:
    """rows에서 '수치형 scalar 지표' 키들을 자동으로 찾아 정렬해 반환한다.

    schedule/seed 같은 메타 컬럼과 list 형태(per-λ 곡선 등)는 제외한다. 이렇게 하면
    Phase 2와 Phase 3가 서로 다른 지표 집합을 갖더라도 같은 집계 함수를 쓸 수 있다.
    """
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k, v in r.items():
            if k in _NON_METRIC_KEYS or k in seen:
                continue
            if isinstance(v, bool) or isinstance(v, (list, dict, str)):
                continue
            if isinstance(v, (int, float)):
                seen.add(k)
                keys.append(k)
    return sorted(keys)


def _finite_values(rows: list[dict], metric: str) -> np.ndarray:
    """rows에서 metric의 유한한 값만 모아 1D array로 반환."""
    vals = [float(r[metric]) for r in rows if metric in r and r[metric] is not None]
    arr = np.array(vals, dtype=float)
    return arr[np.isfinite(arr)]


def aggregate_over_seeds(
    summary_rows: list[dict],
    metrics: list[str] | None = None,
) -> list[dict]:
    """schedule별로 seed들에 걸쳐 mean / std / sem / n 을 집계한다.

    metrics 가 None 이면 수치형 지표를 자동 탐지한다. 각 지표 m에 대해
        f"{m}_mean", f"{m}_std", f"{m}_sem", f"{m}_n" 키를 채운다.
    std는 표본표준편차(ddof=1), sem = std / sqrt(n) (n≥2일 때만 정의).
    """
    if metrics is None:
        metrics = detect_metric_keys(summary_rows)

    by_sched: dict[str, list[dict]] = defaultdict(list)
    for r in summary_rows:
        by_sched[str(r["schedule"])].append(r)

    out: list[dict] = []
    for sched, rows in by_sched.items():
        agg: dict = {"schedule": sched, "n_seeds": len(rows)}
        for m in metrics:
            vals = _finite_values(rows, m)
            n = int(vals.size)
            mean = float(vals.mean()) if n >= 1 else float("nan")
            std = float(vals.std(ddof=1)) if n >= 2 else float("nan")
            sem = float(std / math.sqrt(n)) if n >= 2 else float("nan")
            agg[f"{m}_mean"], agg[f"{m}_std"] = mean, std
            agg[f"{m}_sem"], agg[f"{m}_n"] = sem, n
        out.append(agg)
    return out


def aggregated_csv_fieldnames(agg_rows: list[dict]) -> list[str]:
    """집계 CSV 컬럼 순서를 집계 결과로부터 동적으로 만든다."""
    if not agg_rows:
        return ["schedule", "n_seeds"]
    fields = ["schedule", "n_seeds"]
    for k in agg_rows[0]:
        if k not in fields:
            fields.append(k)
    return fields


def paired_differences_vs_baseline(
    summary_rows: list[dict],
    baseline: str,
    metric: str = "fid_phi",
) -> dict[str, np.ndarray]:
    """baseline schedule 대비 seed별 차이(Δ = other − baseline)를 구한다.

    같은 seed에서 측정한 값끼리만 짝지어(paired) 차이를 만든다. 이 Δ 배열이
    paired t-test / Wilcoxon signed-rank test의 입력이 된다.
    반환: {schedule_name: Δ array} (baseline 자신은 제외).
    """
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for r in summary_rows:
        if metric in r and r[metric] is not None and np.isfinite(float(r[metric])):
            by_seed[int(r["seed"])][str(r["schedule"])] = float(r[metric])

    schedules = sorted({str(r["schedule"]) for r in summary_rows} - {baseline})
    diffs: dict[str, list[float]] = {s: [] for s in schedules}
    for _seed, sched_vals in by_seed.items():
        if baseline not in sched_vals:
            continue
        base_val = sched_vals[baseline]
        for s in schedules:
            if s in sched_vals:
                diffs[s].append(sched_vals[s] - base_val)
    return {s: np.array(v, dtype=float) for s, v in diffs.items()}


def significance_tests(
    summary_rows: list[dict],
    baseline: str,
    metric: str = "fid_phi",
) -> list[dict]:
    """각 schedule을 baseline과 paired 비교해 유의성을 검정한다.

    seed 쌍이 2개 이상이면 paired t-test를, 5개 이상이면 Wilcoxon도 함께 보고한다.
    (Wilcoxon은 표본이 매우 작으면 신뢰할 수 없으므로 보조 지표로만 사용.)
    scipy가 없거나 seed가 부족하면 p-value를 NaN으로 두고 status에 사유를 남긴다.
    """
    diffs = paired_differences_vs_baseline(summary_rows, baseline, metric)
    lower_better = metric in LOWER_IS_BETTER

    try:
        from scipy import stats as _stats
        have_scipy = True
    except ImportError:
        have_scipy = False

    results: list[dict] = []
    for sched, dvals in sorted(diffs.items()):
        n = int(dvals.size)
        mean_diff = float(dvals.mean()) if n >= 1 else float("nan")
        improved = (mean_diff < 0) if lower_better else (mean_diff > 0)

        row: dict = {
            "schedule": sched, "baseline": baseline, "metric": metric,
            "n_pairs": n, "mean_diff": mean_diff,
            "improved_vs_baseline": bool(improved) if n >= 1 else None,
            "t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan"),
            "status": "ok",
        }

        if n < 2:
            row["status"] = "seed 부족 (paired 검정에는 num_seeds≥2 필요)"
        elif not have_scipy:
            row["status"] = "scipy 미설치 (pip install scipy)"
        elif np.allclose(dvals, dvals[0]):
            row["status"] = "차이의 분산이 0 (검정 불가)"
        else:
            t_res = _stats.ttest_rel(dvals, np.zeros_like(dvals))
            row["t_pvalue"] = float(t_res.pvalue)
            if n >= 5:
                try:
                    w_res = _stats.wilcoxon(dvals)
                    row["wilcoxon_pvalue"] = float(w_res.pvalue)
                except ValueError:
                    pass  # 모든 차이가 0이면 wilcoxon이 실패할 수 있음
        results.append(row)
    return results


def per_lambda_excess_and_skill(
    per_lambda_agg: dict[str, dict],
    baseline: str,
    trivial_mse: float = 1.0,
) -> dict[str, dict]:
    """per-λ MSE 집계 곡선에 두 가지 분해 view를 더한다 (Phase 2·3 공용, 재실행 없음).

    입력은 aggregate_per_lambda() 의 출력 {schedule: {lambda_grid, mse_mean, ...}}.
    각 schedule에 다음을 채워 같은 구조로 돌려준다.

    (1) excess_mean[λ] = mse_mean[λ] − mse_baseline[λ]
        같은 λ에서 '이론상 최소(Bayes) MSE'는 데이터·λ에만 의존하므로 **모든 schedule이
        공유**한다. 따라서 baseline과의 차분에서 그 미지의 floor가 소거되어, 남는 것은
        순수하게 p_train 차이에 의한 '초과 손실'이다(실데이터엔 analytic Bayes가 없어
        절대 excess는 못 구하지만, 차분으로는 honest하게 구할 수 있다). baseline은
        excess_mean=0 이 된다. >0 이면 그 λ에서 baseline보다 못 배웠다는 뜻.

    (2) skill_mean[λ] = 1 − mse_mean[λ]/trivial_mse
        ε-prediction에서 best constant predictor(=0)의 MSE = Var(ε) = 1 이므로
        trivial_mse=1 이 기본. skill=1 완벽, 0 trivial과 동급, <0 trivial보다 나쁨.
        "U-Net이 각 noise 구간에서 실제로 학습됐는지"를 보여주는 bounded 지표(R²/skill score).
    """
    out: dict[str, dict] = {s: dict(v) for s, v in per_lambda_agg.items()}
    base = per_lambda_agg.get(baseline)
    base_mse = np.array(base["mse_mean"], dtype=float) if base is not None else None
    for _s, v in out.items():
        mse = np.array(v["mse_mean"], dtype=float)
        v["skill_mean"] = (1.0 - mse / trivial_mse).tolist()
        if base_mse is not None and base_mse.shape == mse.shape:
            v["excess_mean"] = (mse - base_mse).tolist()
        else:
            v["excess_mean"] = [float("nan")] * len(mse)
    return out


def region_curve_mean(
    lambda_grid: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    low: float,
    high: float,
    region: str = "transition",
) -> float:
    """λ 곡선을 구간별 스칼라로 요약한다 (transition / low_noise / high_noise).

    region:
      - "transition" : low ≤ λ ≤ high
      - "low_noise"  : λ > high  (신호 강함)
      - "high_noise" : λ < low   (잡음 강함)
    """
    grid = np.asarray(lambda_grid, dtype=float)
    vals = np.asarray(values, dtype=float)
    if region == "transition":
        mask = (grid >= low) & (grid <= high)
    elif region == "low_noise":
        mask = grid > high
    elif region == "high_noise":
        mask = grid < low
    else:
        raise ValueError(f"unknown region: {region}")
    finite = mask & np.isfinite(vals)
    return float(vals[finite].mean()) if finite.any() else float("nan")


def aggregate_per_lambda(all_per_lambda: list[dict]) -> dict[str, dict]:
    """per-λ denoising MSE 곡선을 schedule별로 seed에 걸쳐 집계한다.

    같은 schedule의 seed별 MSE 곡선을 쌓아 λ마다 mean/std를 낸다. 나중에
    MSE 곡선을 신뢰구간과 함께 그리거나 곡선 간 차이를 검정할 때 쓰인다.
    반환: {schedule: {"lambda_grid", "mse_mean", "mse_std", "n_seeds"}}.
    """
    by_sched: dict[str, list[dict]] = defaultdict(list)
    for rec in all_per_lambda:
        by_sched[str(rec["schedule"])].append(rec)

    out: dict[str, dict] = {}
    for sched, recs in by_sched.items():
        grid = np.array(recs[0]["lambda_grid"], dtype=float)
        stacked = np.array([rec["per_lambda_mse"] for rec in recs], dtype=float)  # (n_seeds, n_λ)
        out[sched] = {
            "lambda_grid": grid.tolist(),
            "mse_mean": stacked.mean(0).tolist(),
            "mse_std": (stacked.std(0, ddof=1) if len(recs) >= 2
                        else np.zeros_like(grid)).tolist(),
            "n_seeds": len(recs),
        }
    return out
