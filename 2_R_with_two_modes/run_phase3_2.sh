#!/usr/bin/env bash
# Phase 3.2 (CIFAR-10, mixture schedule) 실행 헬퍼.
#
# 설계: p_train(λ)만 변수. model/objective/sampler/optimizer/steps/eval 전부 고정.
#   p_train(λ) = (1-η)·Uniform[-8,8] + η·TruncatedLaplace[-8,8](center, b)
#   center ∈ {λ_R*(DMSR), 0(Hang)} 을 η·b 동일 조건에서 통제비교.
#
# 사용법:
#   bash run_phase3_2.sh smoke                         # 파이프라인 점검
#   bash run_phase3_2.sh stageA [pair]                 # 탐색: seed1, (η,b) sweep 8+2 schedules
#   bash run_phase3_2.sh stageB [pair]                 # 확정: seed3, 단일(η,b) 4 schedules
#   bash run_phase3_2.sh resume <folder> [max_new]     # 끊긴 폴더 이어서
#
# class-pair 키: cat_vs_dog, deer_vs_horse, automobile_vs_truck,
#               airplane_vs_automobile, airplane_vs_frog (phase3_2/config.py CLASS_PAIRS)
set -e
cd "$(dirname "$0")"
export TORCHDYNAMO_DISABLE=1

SCRIPT="phase3_2/phase3_cifar_experiment.py"
MODE="${1:-smoke}"
PAIR="${2:-airplane_vs_frog}"

# ── RTX 4090 (Ada, 24GB) 설정 — Phase 3와 동일 ────────────────────────────────
SAFE_ARGS=(
  --amp bf16
  --num-workers 8
  --prefetch-factor 4
  --micro-batch-size 128
  --clf-batch-size 256
  --gen-batch-size 512
)

case "$MODE" in
  smoke)
    echo "[run_phase3_2] SMOKE TEST"
    python3 "$SCRIPT" --preset smoke --device cuda --class-pair "$PAIR" \
      --run-name phase3_2_smoke \
      --mixture-etas 0.5 0.75 --mixture-bs 0.5 1.0 "${SAFE_ARGS[@]}"
    ;;
  stageA)
    # 탐색(seed 1): 붕괴/무효 없이 emphasis가 보이는 (η,b) 지점 찾기.
    #   schedules = cosine_vp + uniform + {dmsr,at0}×{η 0.5,0.75}×{b 0.5,1.0} = 10개
    echo "[run_phase3_2] STAGE A (탐색, seed1, (η,b) sweep) — pair=$PAIR"
    python3 "$SCRIPT" --preset full --device cuda --class-pair "$PAIR" --num-seeds 1 \
      --run-name phase3_2_stageA \
      --mixture-etas 0.5 0.75 --mixture-bs 0.5 1.0 "${SAFE_ARGS[@]}"
    ;;
  stageB)
    # 확정(seed 3): Stage A에서 고른 단일 (η,b)로 중심 0 vs λ_R* 정면 비교 + baseline.
    #   기본 (η,b)=(0.5,1.0). 필요시 3·4번째 인자로 덮어쓰기: stageB pair 0.5 1.0
    ETA="${3:-0.5}"; B="${4:-1.0}"
    echo "[run_phase3_2] STAGE B (확정, seed3, η=$ETA b=$B) — pair=$PAIR"
    python3 "$SCRIPT" --preset full --device cuda --class-pair "$PAIR" --num-seeds 3 \
      --run-name phase3_2_stageB_eta${ETA}_b${B} \
      --mixture-etas "$ETA" --mixture-bs "$B" "${SAFE_ARGS[@]}"
    ;;
  resume)
    RDIR="${2:-}"
    if [[ -z "$RDIR" ]]; then echo "Usage: bash run_phase3_2.sh resume <folder> [max_new]"; exit 1; fi
    MAXNEW="${3:-}"; EXTRA=()
    [[ -n "$MAXNEW" ]] && EXTRA=(--max-new "$MAXNEW")
    echo "[run_phase3_2] RESUME $(basename "$RDIR") ${MAXNEW:+(새로 최대 $MAXNEW개)}"
    python3 "$SCRIPT" --device cuda --resume "$RDIR" "${SAFE_ARGS[@]}" "${EXTRA[@]}"
    ;;
  *)
    echo "Usage: bash run_phase3_2.sh [smoke|stageA|stageB|resume] [pair] ..."
    echo "  bash run_phase3_2.sh stageA airplane_vs_frog          # 탐색 (seed1, 10 schedules)"
    echo "  bash run_phase3_2.sh stageB airplane_vs_frog 0.5 1.0  # 확정 (seed3, 4 schedules)"
    echo "  bash run_phase3_2.sh resume <folder> [max_new]"
    exit 1
    ;;
esac
