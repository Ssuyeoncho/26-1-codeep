"""Phase 2·3 공용 분류(classification) 평가 지표 (순수 numpy).

생성 품질 지표(gen_metrics.py)와 별개로, **정답 레이블이 있는** 분류기 평가용 지표를
모았다. Phase 2에서는 feature extractor φ(= MNIST 두 숫자 분류기)를 테스트셋에서
평가하는 데 쓴다. φ는 empirical DMSR_φ(λ)와 모든 FID-φ 계열 지표의 기반이므로,
φ 자체가 얼마나 잘 분류하는지(Accuracy·Precision·Recall·F1·Confusion)를 보고하면
하위 지표들의 신뢰도를 함께 점검할 수 있다.

모든 함수는 정수 레이블 배열 (y_true, y_pred)을 받는다. 이미지를 레이블로 바꾸는 부분
(어떤 분류기를 쓸지)은 각 phase가 책임진다 — gen_metrics와 동일한 분업 원칙.

지표 정의(다중 클래스 일반화, 2클래스면 자동으로 2×2):
  - accuracy           : (맞춘 수) / (전체)  = TP·합 / N
  - confusion_matrix   : cm[i, j] = 실제 i 를 j 로 예측한 개수 (행=정답, 열=예측)
  - per-class precision: TP / (TP + FP) = cm[c,c] / (열 c 합)
  - per-class recall   : TP / (TP + FN) = cm[c,c] / (행 c 합)
  - per-class f1       : precision·recall 의 조화평균 (불균형에 강건)
  - macro avg          : 클래스별 지표의 단순 평균 (클래스 수로 가중)
  - weighted avg       : support(정답 개수)로 가중한 평균
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """cm[i, j] = (실제=i, 예측=j) 개수. 행=정답, 열=예측."""
    yt = np.asarray(y_true, dtype=int).reshape(-1)
    yp = np.asarray(y_pred, dtype=int).reshape(-1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # bincount로 한 번에 채운다(빠르고 명확).
    flat = yt * n_classes + yp
    counts = np.bincount(flat, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes) + cm


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Accuracy·per-class P/R/F1·support·macro/weighted 평균 + confusion matrix를 dict로.

    n_classes 를 명시하면(예: len(config.digits)) 예측에 빠진 클래스가 있어도 항상
    그 크기의 정사각 confusion matrix를 만든다(2클래스면 2×2 보장).
    """
    yt = np.asarray(y_true, dtype=int).reshape(-1)
    yp = np.asarray(y_pred, dtype=int).reshape(-1)
    if n_classes is None:
        n_classes = int(max(yt.max(initial=0), yp.max(initial=0))) + 1
    if class_names is None:
        class_names = [str(c) for c in range(n_classes)]

    cm = confusion_matrix(yt, yp, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)     # 행 합 = 실제 각 클래스 개수
    pred_cnt = cm.sum(axis=0).astype(float)    # 열 합 = 예측한 각 클래스 개수

    # 분모 0 보호: 해당 분모가 0이면 그 지표를 0으로 둔다(sklearn zero_division=0과 동일).
    precision = np.where(pred_cnt > 0, tp / np.maximum(pred_cnt, 1.0), 0.0)
    recall    = np.where(support  > 0, tp / np.maximum(support,  1.0), 0.0)
    denom     = precision + recall
    f1        = np.where(denom > 0, 2.0 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    total = float(len(yt))
    accuracy = float(tp.sum() / total) if total > 0 else float("nan")

    w = support / max(support.sum(), 1.0)      # support 가중치
    per_class = []
    for c in range(n_classes):
        per_class.append({
            "class": class_names[c],
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        })

    return {
        "class_names": class_names,
        "n_classes": int(n_classes),
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float((precision * w).sum()),
        "weighted_recall": float((recall * w).sum()),
        "weighted_f1": float((f1 * w).sum()),
        "n_samples": int(total),
    }
