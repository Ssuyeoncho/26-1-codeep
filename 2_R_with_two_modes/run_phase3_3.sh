#!/usr/bin/env bash
# Phase 3.3 (ImageNette-64, mixture schedule) 실행 헬퍼.
#
# 설계: ImageNette-160을 64x64로 resize해 two-class natural-image subset에서
# p_train(lambda)의 center 위치(lambda_R* vs 0)를 비교한다.
set -e
cd "$(dirname "$0")"
export TORCHDYNAMO_DISABLE=1

SCRIPT="phase3_3/phase3_imagenette_experiment.py"
MODE="${1:-smoke}"
PAIR="${2:-gas_pump_vs_golf_ball}"

SAFE_ARGS=(
  --amp bf16
  --num-workers 8
  --prefetch-factor 4
  --batch-size 64
  --micro-batch-size 16
  --clf-batch-size 128
  --gen-batch-size 128
)

case "$MODE" in
  smoke)
    echo "[run_phase3_3] SMOKE TEST — pair=$PAIR"
    python3 "$SCRIPT" --preset smoke --device cuda --class-pair "$PAIR" \
      --run-name phase3_3_smoke "${SAFE_ARGS[@]}"
    ;;
  stageA)
    echo "[run_phase3_3] STAGE A (탐색, seed1, eta/b sweep) — pair=$PAIR"
    python3 "$SCRIPT" --preset fast --device cuda --class-pair "$PAIR" --num-seeds 1 \
      --run-name phase3_3_stageA \
      --mixture-etas 0.5 0.75 --mixture-bs 0.5 1.0 "${SAFE_ARGS[@]}"
    ;;
  stageB)
    ETA="${3:-0.5}"; B="${4:-1.0}"
    echo "[run_phase3_3] STAGE B (확정, seed3, eta=$ETA b=$B) — pair=$PAIR"
    python3 "$SCRIPT" --preset full --device cuda --class-pair "$PAIR" --num-seeds 3 \
      --run-name phase3_3_stageB_eta${ETA}_b${B} \
      --mixture-etas "$ETA" --mixture-bs "$B" "${SAFE_ARGS[@]}"
    ;;
  resume)
    RDIR="${2:-}"
    if [[ -z "$RDIR" ]]; then echo "Usage: bash run_phase3_3.sh resume <folder> [max_new]"; exit 1; fi
    MAXNEW="${3:-}"; EXTRA=()
    [[ -n "$MAXNEW" ]] && EXTRA=(--max-new "$MAXNEW")
    echo "[run_phase3_3] RESUME $(basename "$RDIR") ${MAXNEW:+(새로 최대 $MAXNEW개)}"
    python3 "$SCRIPT" --device cuda --resume "$RDIR" "${SAFE_ARGS[@]}" "${EXTRA[@]}"
    ;;
  *)
    echo "Usage: bash run_phase3_3.sh [smoke|stageA|stageB|resume] [pair] ..."
    echo "  pairs: gas_pump_vs_golf_ball, church_vs_garbage_truck, english_springer_vs_garbage_truck"
    exit 1
    ;;
esac
