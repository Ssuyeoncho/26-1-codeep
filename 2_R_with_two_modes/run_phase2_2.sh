#!/usr/bin/env bash
# Phase 2.2 (MNIST pipeline pilot) 실행 헬퍼.
#
# 설계: Phase 3.2와 같은 mixture family를 MNIST 0 vs 1에서 작게 검증한다.
#   p_train(lambda) = (1-eta)U[-8,8] + eta TruncatedLaplace[-8,8](center, beta)
#   center in {0, lambda_R*}, eta=0.5, beta=2.0 기본.
set -e
cd "$(dirname "$0")"
export TORCHDYNAMO_DISABLE=1

SCRIPT="phase2_2/phase2_mnist_experiment.py"
MODE="${1:-smoke}"

SAFE_ARGS=(
  --amp bf16
  --micro-batch-size 128
  --gen-batch-size 1000
)

run_one() {
  python3 "$SCRIPT" --device cuda "$@" "${SAFE_ARGS[@]}"
}

case "$MODE" in
  smoke)
    echo "[run_phase2_2] SMOKE TEST"
    run_one --run-name phase2_2_mnist_smoke \
      --train-steps 200 --n-generate 64 --ddim-steps 5 \
      --dmsr-grid-size 12 --eval-grid-size 12 --clf-epochs 2
    ;;
  pilot)
    echo "[run_phase2_2] PILOT (MNIST 0 vs 1, 4 schedules, seed1)"
    run_one --run-name phase2_2_mnist_pilot
    ;;
  paired)
    echo "[run_phase2_2] PAIRED PIPELINE CHECK (2 seeds, no significance claim)"
    run_one --run-name phase2_2_mnist_paired --num-seeds 2
    ;;
  *)
    echo "Usage: bash run_phase2_2.sh [smoke|pilot|paired]"
    exit 1
    ;;
esac
