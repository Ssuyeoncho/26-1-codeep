#!/usr/bin/env bash
# Phase 2 (MNIST) 실행 헬퍼 — 긴 인자를 매번 안 쳐도 되게 묶어둔 스크립트.
#
# 사용법:
#   bash run_phase2.sh smoke    # 1) 스모크 테스트 (1~2분, 파이프라인 점검)
#   bash run_phase2.sh full     # 2) 전체 실행 (단일 seed)
#   bash run_phase2.sh final    # 3) 최종 실행 (seed 3개, 통계·유의성까지)
#
# 이 서버는 torch.compile이 nvcc 권한 문제로 막혀 있어 TORCHDYNAMO_DISABLE=1로 끈다.
# (bf16 AMP·TF32·cudnn autotuner 가속은 그대로 작동한다.)
set -e
cd "$(dirname "$0")"            # 항상 2_R_with_two_modes/ 에서 실행
export TORCHDYNAMO_DISABLE=1    # torch.compile 완전 비활성화 (어떤 코드 버전에서도 안전)

SCRIPT="phase2/phase2_mnist_experiment.py"
MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    echo "[run_phase2] SMOKE TEST (빠른 동작 확인)"
    python3 "$SCRIPT" --device cuda \
      --train-steps 500 --n-generate 100 --ddim-steps 5 \
      --dmsr-grid-size 12 --eval-grid-size 12 --clf-epochs 2
    ;;
  full)
    echo "[run_phase2] FULL (단일 seed)"
    python3 "$SCRIPT" --device cuda
    ;;
  final)
    echo "[run_phase2] FINAL (seed 3개 — 통계·유의성 검정)"
    python3 "$SCRIPT" --device cuda --num-seeds 3
    ;;
  *)
    echo "Usage: bash run_phase2.sh [smoke|full|final]"
    exit 1
    ;;
esac
