#!/usr/bin/env bash
# Phase 2 (MNIST) 실행 헬퍼 — 긴 인자를 매번 안 쳐도 되게 묶어둔 스크립트.
#
# 사용법:
#   bash run_phase2.sh smoke            # 스모크 테스트 (1~2분, 파이프라인 점검)
#   bash run_phase2.sh full             # 전체 실행 (단일 seed, 기본 0 vs 1)
#   bash run_phase2.sh full 3 8         # 전체 실행, 클래스 쌍 지정 (digit 3 vs 8)
#   bash run_phase2.sh final            # 최종 실행 (seed 3개, 통계·유의성)
#   bash run_phase2.sh pairs            # ★ 큐레이션된 여러 쌍을 한 번에 전부 실행(무인 배치)
#
# torch.compile은 환경 의존성이 있어 TORCHDYNAMO_DISABLE=1로 끈다(안전).
# (bf16 AMP·TF32·cudnn autotuner 가속은 그대로 작동한다.)
set -e
cd "$(dirname "$0")"            # 항상 2_R_with_two_modes/ 에서 실행
export TORCHDYNAMO_DISABLE=1    # torch.compile 완전 비활성화 (어떤 코드 버전에서도 안전)

SCRIPT="phase2/phase2_mnist_experiment.py"
MODE="${1:-smoke}"

# ── RTX 4090 (Ada Lovelace, 24GB, CUDA 12.2) 설정 ─────────────────────────────
#  - Ada는 bf16을 하드웨어로 지원 → fp16+scaler보다 안정적인 bf16 사용.
#  - VRAM 24GB라 MNIST Mini U-Net은 매우 작아 여유가 크다. 학습 batch_size(=128)는
#    공정 비교를 위해 그대로 두고(=micro-batch도 128로 분할 없음), 처리량만 영향받는
#    생성 batch는 크게 키운다. 결과값은 동일하고 속도만 오른다.
#  - 만약 OOM(거의 없음)이 나면 --gen-batch-size 를 낮추면 된다.
SAFE_ARGS=(
  --amp bf16
  --micro-batch-size 128
  --gen-batch-size 1000
)

# digit 쌍을 2·3번째 인자로 줄 수 있다 (pairs 모드가 아닐 때).
DIGIT_ARGS=()
if [[ "$MODE" != "pairs" && -n "${2:-}" && -n "${3:-}" ]]; then
  DIGIT_ARGS=(--digits "$2" "$3")
  echo "[run_phase2] digits = $2 vs $3"
fi

# pairs 모드에서 한 번에 돌릴 큐레이션 쌍 (비슷/다름 난이도를 섞어 DMSR 변화를 본다).
#   distinct(쉬움): 0 vs 1, 1 vs 8   /   similar(어려움): 3 vs 8, 4 vs 9
MNIST_PAIRS=("0 1" "1 8" "3 8" "4 9")

run_one() {  # $1..: 추가 인자
  python3 "$SCRIPT" --device cuda "$@" "${SAFE_ARGS[@]}"
}

case "$MODE" in
  smoke)
    echo "[run_phase2] SMOKE TEST (빠른 동작 확인)"
    run_one --run-name phase2_mnist_smoke \
      --train-steps 500 --n-generate 100 --ddim-steps 5 \
      --dmsr-grid-size 12 --eval-grid-size 12 --clf-epochs 2 "${DIGIT_ARGS[@]}"
    ;;
  full)
    echo "[run_phase2] FULL (단일 seed, RTX 4090)"
    run_one "${DIGIT_ARGS[@]}"
    ;;
  final)
    echo "[run_phase2] FINAL (seed 3개 — 통계·유의성, RTX 4090)"
    run_one --num-seeds 3 "${DIGIT_ARGS[@]}"
    ;;
  pairs)
    # 여러 쌍을 무인 배치로 실행. 한 쌍이 실패해도 멈추지 않고 다음으로 넘어간다
    # (각 쌍은 자체 결과 폴더에 저장되므로 부분 완료도 보존된다).
    echo "[run_phase2] PAIRS BATCH — ${#MNIST_PAIRS[@]} pairs: ${MNIST_PAIRS[*]}"
    FAILED=()
    for pair in "${MNIST_PAIRS[@]}"; do
      echo ""
      echo "================ [run_phase2] PAIR: $pair ================"
      # shellcheck disable=SC2086
      run_one --run-name phase2_mnist_pairs --digits $pair \
        || { echo "[run_phase2] !! FAILED pair: $pair (계속 진행)"; FAILED+=("$pair"); }
    done
    echo ""
    echo "[run_phase2] PAIRS DONE. 성공 $(( ${#MNIST_PAIRS[@]} - ${#FAILED[@]} ))/${#MNIST_PAIRS[@]}"
    [[ ${#FAILED[@]} -gt 0 ]] && echo "[run_phase2] 실패한 쌍: ${FAILED[*]}"
    echo "[run_phase2] 결과: results/phase2/ (폴더명에 dXvY 로 쌍 구분, __OK/__FAILED 로 상태 구분)"
    ;;
  *)
    echo "Usage: bash run_phase2.sh [smoke|full|final|pairs] [digitA digitB]"
    echo "  예) bash run_phase2.sh full 3 8     # 단일 쌍"
    echo "      bash run_phase2.sh pairs        # 큐레이션 쌍 전부 (무인 배치)"
    exit 1
    ;;
esac
