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
# RTX 2080 Ti(11GB)에서 서버 GUI/GPU가 굳지 않도록 CPU BLAS/OpenMP 스레드와 DataLoader 선읽기를 낮춘다.
# 실험 횟수(train steps, seed 수, 생성 이미지 수)는 유지하고, 한 번에 처리하는 양만 줄인다.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SCRIPT="phase3/phase3_cifar_experiment.py"
MODE="${1:-smoke}"
SAFE_ARGS=(
  --amp fp16
  --num-workers 1
  --prefetch-factor 1
  --micro-batch-size 16
  --clf-batch-size 64
  --gen-batch-size 32
)

case "$MODE" in
  smoke)
    echo "[run_phase3] SMOKE TEST"
    python3 "$SCRIPT" --preset smoke --device cuda "${SAFE_ARGS[@]}"
    ;;
  full)
    echo "[run_phase3] FULL (단일 seed, RTX 2080 Ti safe)"
    python3 "$SCRIPT" --preset full --device cuda "${SAFE_ARGS[@]}"
    ;;
  final)
    echo "[run_phase3] FINAL (seed 3개 — 통계·유의성 검정, RTX 2080 Ti safe)"
    python3 "$SCRIPT" --preset full --device cuda --num-seeds 3 "${SAFE_ARGS[@]}"
    ;;
  *)
    echo "Usage: bash run_phase3.sh [smoke|full|final]"
    exit 1
    ;;
esac
