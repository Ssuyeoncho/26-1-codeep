#!/usr/bin/env bash
# Phase 3 (CIFAR-10) 실행 헬퍼 — Phase 2와 동일한 사용 패턴.
#
# 사용법:
#   bash run_phase3.sh smoke    # 1) 스모크 테스트 (파이프라인 점검)
#   bash run_phase3.sh full     # 2) 전체 실행 (단일 seed, FID 포함)
#   bash run_phase3.sh final    # 3) 최종 실행 (seed 3개, 통계·유의성까지)
#
# torch.compile은 TORCHDYNAMO_DISABLE=1로 끈다(서버 nvcc 권한 문제). AMP/TF32/cudnn은 작동.
set -e
cd "$(dirname "$0")"
export TORCHDYNAMO_DISABLE=1

SCRIPT="phase3/phase3_cifar_experiment.py"
MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    echo "[run_phase3] SMOKE TEST"
    python3 "$SCRIPT" --preset smoke --device cuda
    ;;
  full)
    echo "[run_phase3] FULL (단일 seed)"
    python3 "$SCRIPT" --preset full --device cuda
    ;;
  final)
    echo "[run_phase3] FINAL (seed 3개 — 통계·유의성 검정)"
    python3 "$SCRIPT" --preset full --device cuda --num-seeds 3
    ;;
  *)
    echo "Usage: bash run_phase3.sh [smoke|full|final]"
    exit 1
    ;;
esac
