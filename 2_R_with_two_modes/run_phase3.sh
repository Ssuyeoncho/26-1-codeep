#!/usr/bin/env bash
# Phase 3 (CIFAR-10) 실행 헬퍼 — Phase 2와 동일한 사용 패턴.
#
# 사용법:
#   bash run_phase3.sh smoke                    # 스모크 테스트 (파이프라인 점검)
#   bash run_phase3.sh full                      # 전체 실행 (단일 seed, FID 포함, 기본 쌍)
#   bash run_phase3.sh full cat_vs_dog           # 클래스 쌍 지정 (CLASS_PAIRS 키)
#   bash run_phase3.sh final                      # 최종 (seed 3개, 통계·유의성)
#   bash run_phase3.sh pairs                       # ★ 큐레이션된 여러 쌍을 한 번에(무인 배치)
#
# 사용 가능한 class-pair 키: cat_vs_dog, deer_vs_horse, automobile_vs_truck,
#                          airplane_vs_automobile, airplane_vs_frog (phase3/config.py CLASS_PAIRS)
#
# torch.compile은 환경 의존성이 있어 TORCHDYNAMO_DISABLE=1로 끈다(안전).
set -e
cd "$(dirname "$0")"
export TORCHDYNAMO_DISABLE=1

SCRIPT="phase3/phase3_cifar_experiment.py"
MODE="${1:-smoke}"

# ── RTX 4090 (Ada Lovelace, 24GB, CUDA 12.2) 설정 ─────────────────────────────
#  - Ada는 bf16 하드웨어 지원 → bf16 사용(fp16+scaler보다 안정적).
#  - 24GB라 CIFAR UNet도 full batch(128)가 여유롭게 들어간다. 학습 batch_size는 공정
#    비교 위해 유지(micro-batch=128, 분할 없음). 처리량용 batch(생성/분류/평가)는 키운다.
#  - DataLoader 워커를 늘려 GPU를 굶기지 않게 한다(전용 서버 가정). RAM/IO 문제 시 낮추면 됨.
#  - OOM이 거의 없지만 나면 --gen-batch-size / --micro-batch-size 를 낮추면 된다.
SAFE_ARGS=(
  --amp bf16
  --num-workers 8
  --prefetch-factor 4
  --micro-batch-size 128
  --clf-batch-size 256
  --gen-batch-size 512
)

# class-pair를 2번째 인자로 줄 수 있다 (pairs 모드가 아닐 때).
PAIR_ARGS=()
if [[ "$MODE" != "pairs" && -n "${2:-}" ]]; then
  PAIR_ARGS=(--class-pair "$2")
  echo "[run_phase3] class-pair = $2"
fi

# pairs 모드에서 돌릴 기본 큐레이션 쌍 (비슷/다름 난이도를 섞음). 가장 대비가 큰 쌍을
# 먼저 둬서, 밤새 다 못 돌더라도 비슷↔다름 대조를 먼저 확보한다.
#   similar(어려움): cat_vs_dog, deer_vs_horse  /  distinct(쉬움): airplane_vs_frog, airplane_vs_automobile
CIFAR_PAIRS=("cat_vs_dog" "airplane_vs_frog" "deer_vs_horse" "airplane_vs_automobile")
# CIFAR는 무거우므로 pairs 배치는 'fast' 프리셋(50k steps, FID 포함)으로 돈다.
PAIRS_PRESET="fast"

# pairs 배치는 full(15개) 대신 '중심 위치 통제비교'를 살린 핵심 9개만 돈다(폭 2개로 축소).
# 남기는 schedule:
#   cosine_vp(baseline)
#   + {Normal, Laplace} × {중심 λ_R*(dmsr), 중심 0(at0=Hang)} × 폭{0.5(좁음), 4.0(넓음)} = 8개
#     → 같은 모양·폭에서 '중심만' 다를 때 성능 비교(우리 핵심 가설)
# 빠지는 것: linear/uniform, 중간 폭 1.5.
#   → 1쌍당 약 9 schedule × ~28분 ≈ 약 4.2시간 (Inception-FID 켜지면 +조금).
LEAN_ARGS=(--no-linear --no-uniform --width-values 0.5 4.0)

case "$MODE" in
  smoke)
    echo "[run_phase3] SMOKE TEST"
    python3 "$SCRIPT" --preset smoke --device cuda "${PAIR_ARGS[@]}" "${SAFE_ARGS[@]}"
    ;;
  full)
    echo "[run_phase3] FULL (단일 seed, 13 schedules 전부, RTX 4090)"
    python3 "$SCRIPT" --preset full --device cuda "${PAIR_ARGS[@]}" "${SAFE_ARGS[@]}"
    ;;
  final)
    echo "[run_phase3] FINAL (seed 3개 — 통계·유의성, RTX 4090)"
    python3 "$SCRIPT" --preset full --device cuda --num-seeds 3 "${PAIR_ARGS[@]}" "${SAFE_ARGS[@]}"
    ;;
  pairs)
    # 2번째 인자부터 돌릴 쌍을 직접 지정할 수 있다. 없으면 기본 4쌍 전부.
    #   bash run_phase3.sh pairs cat_vs_dog                    # 1쌍 (~4.2h)
    #   bash run_phase3.sh pairs cat_vs_dog airplane_vs_frog   # 2쌍 (~8.4h)
    #   bash run_phase3.sh pairs                               # 4쌍 전부 (~17h)
    SEL_PAIRS=("${@:2}")
    if [[ ${#SEL_PAIRS[@]} -eq 0 ]]; then SEL_PAIRS=("${CIFAR_PAIRS[@]}"); fi
    echo "[run_phase3] PAIRS BATCH (preset=$PAIRS_PRESET, lean 9 schedules) — ${#SEL_PAIRS[@]} pairs: ${SEL_PAIRS[*]}"
    echo "[run_phase3] 예상: 약 4.2시간/쌍. 일부만 끝나도 각 쌍은 자체 폴더에 저장되어 보존된다."
    FAILED=()
    for pair in "${SEL_PAIRS[@]}"; do
      echo ""
      echo "================ [run_phase3] PAIR: $pair ================"
      python3 "$SCRIPT" --preset "$PAIRS_PRESET" --device cuda --class-pair "$pair" \
        "${SAFE_ARGS[@]}" "${LEAN_ARGS[@]}" \
        || { echo "[run_phase3] !! FAILED pair: $pair (계속 진행)"; FAILED+=("$pair"); }
    done
    echo ""
    echo "[run_phase3] PAIRS DONE. 성공 $(( ${#SEL_PAIRS[@]} - ${#FAILED[@]} ))/${#SEL_PAIRS[@]}"
    [[ ${#FAILED[@]} -gt 0 ]] && echo "[run_phase3] 실패한 쌍: ${FAILED[*]}"
    echo "[run_phase3] 결과: results/phase3/ (폴더명에 클래스쌍 이름 + __OK/__FAILED 로 구분)"
    ;;
  resume)
    # 기존 폴더를 이어서. 원래 설정(steps/width/class 등)은 폴더의 config.json에서 그대로
    # 복원되므로 preset/쌍을 따로 안 줘도 된다. 장치/처리량(SAFE_ARGS)만 적용.
    #   bash run_phase3.sh resume <folder>        # 남은 schedule 전부 이어서
    #   bash run_phase3.sh resume <folder> 1      # 이번엔 새로 1개만 학습(끊어 돌리기)
    RDIR="${2:-}"
    if [[ -z "$RDIR" ]]; then echo "Usage: bash run_phase3.sh resume <folder> [max_new]"; exit 1; fi
    MAXNEW="${3:-}"
    EXTRA=()
    [[ -n "$MAXNEW" ]] && EXTRA=(--max-new "$MAXNEW")
    echo "[run_phase3] RESUME $(basename "$RDIR")  ${MAXNEW:+(이번에 새로 학습 최대 $MAXNEW개)}"
    python3 "$SCRIPT" --device cuda --resume "$RDIR" "${SAFE_ARGS[@]}" "${EXTRA[@]}"
    ;;
  *)
    echo "Usage: bash run_phase3.sh [smoke|full|final|pairs|resume] ..."
    echo "  bash run_phase3.sh full cat_vs_dog                 # 단일 쌍, 15 schedules 전부"
    echo "  bash run_phase3.sh pairs cat_vs_dog                # lean 9 schedules, 1쌍 (~4.2h)"
    echo "  bash run_phase3.sh pairs cat_vs_dog airplane_vs_frog  # 2쌍 (~8.4h)"
    echo "  bash run_phase3.sh pairs                           # 기본 4쌍 (~17h)"
    echo "  bash run_phase3.sh resume <folder> [max_new]       # 끊긴 폴더 이어서(끝난 건 재사용)"
    exit 1
    ;;
esac
