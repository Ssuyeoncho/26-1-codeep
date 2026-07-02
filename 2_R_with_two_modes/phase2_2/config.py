"""Phase 2.2 MNIST pilot configuration.

Phase 2.2는 성능 우위 주장용 실험이 아니라 Phase 3.2 전에 전체 파이프라인을
검증하는 pilot이다. 따라서 MNIST 0 vs 1 하나만 기본으로 두고, Phase 3.2와 같은
mixture family를 4개 schedule로 작게 실행한다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    # ── 데이터 ────────────────────────────────────────────────────────────────
    digits: tuple = (0, 1)          # two-class setting: MNIST 두 숫자
    img_size: int = 28

    # ── VP forward process의 noise(=log-SNR λ) 범위 ──────────────────────────
    # λ = log SNR. 클수록 signal이 강한 low-noise, 작을수록 high-noise 상태.
    lambda_min: float = -8.0
    lambda_max: float = 8.0
    # ρ: transition region T_R을 자르는 분석용 threshold(|dDMSR/dλ| ≥ ρ·max).
    # 모델 학습에는 영향이 없고 "어디까지를 transition으로 볼지"만 결정한다.
    rho: float = 0.5

    # ── Feature extractor φ (two-class CNN) ───────────────────────────────────
    # penultimate layer 출력을 φ(x)로 사용해 empirical DMSR_φ(λ)를 계산한다.
    clf_epochs: int = 10
    clf_lr: float = 1e-3
    clf_batch_size: int = 256
    clf_feature_dim: int = 64

    # ── Empirical DMSR_φ(λ) 계산 ──────────────────────────────────────────────
    dmsr_grid_size: int = 40        # λ grid 해상도
    dmsr_n_samples: int = 512       # class별 DMSR 추정에 쓰는 이미지 수

    # ── Denoiser (Mini U-Net) ────────────────────────────────────────────────
    base_ch: int = 32
    time_emb_dim: int = 128

    # ── Denoiser 학습 ─────────────────────────────────────────────────────────
    train_steps: int = 20000
    batch_size: int = 128
    # Effective batch_size는 유지하되, GPU에는 micro_batch_size 단위로 쪼개 올린다.
    # None이면 기존처럼 한 번에 batch_size 전체를 처리한다.
    micro_batch_size: int | None = None
    lr: float = 2e-4
    eval_batch_size: int = 256
    eval_grid_size: int = 40

    # ── Seed 통제 (통계 분석의 핵심) ──────────────────────────────────────────
    # 동일한 seed_idx 안에서는 모든 schedule이 같은 run_seed를 공유하도록 구성해
    # schedule 간 비교가 "paired(대응표본)" 설계가 되게 한다.
    # num_seeds ≥ 2 이면 seed 간 mean±std 집계가, ≥ 3 이면 유의성 검정이 의미를 갖는다.
    # (Phase 2는 검증 단계라 기본 1, Phase 3 본 실험에서 여러 seed로 재실행)
    seed: int = 20260526
    num_seeds: int = 1
    device: str = "cpu"

    # ── GPU 가속 (CUDA 서버용, 현재 서버 = RTX 4090 / Ada Lovelace, 24GB) ──────
    # 모든 schedule에 동일하게 적용되는 처리량 옵션이라 비교 공정성에 영향 없음.
    # 4090(Ada)은 bf16·TF32를 하드웨어로 지원하므로 auto가 bf16(GradScaler 불필요)을
    # 고른다. (bf16 미지원 GPU에서는 auto가 fp16+GradScaler로 자동 폴백한다.)
    amp: str = "auto"             # 혼합정밀: auto(=CUDA면 bf16, 미지원 시 fp16)/bf16/fp16/fp32
    # torch.compile은 nvcc 권한 등 환경 의존성이 있어 기본 OFF. 가능한 환경에서만 --compile로 켠다.
    compile_model: bool = False

    # ── DDIM 생성 (모든 schedule 공통·고정: EDM식 통제 설계) ──────────────────
    ddim_steps: int = 50
    n_generate: int = 5000
    gen_batch_size: int = 500     # 생성 배치(학습과 무관, GPU 처리량용 — VRAM 여유 크면 키워도 됨)
    # DDIM 샘플링 λ 범위 (Phase 3와 공통). cosine 격자라 중앙에 step이 집중된다.
    ddim_lambda_min: float = -8.0
    ddim_lambda_max: float = 8.0

    # ── Phase 2.2 pilot schedules ─────────────────────────────────────────────
    # Phase 3.2와 동일한 family:
    #   p_train(λ) = (1-η)U[-8,8] + η TruncatedLaplace[-8,8](center, β)
    # 여기서는 sweep하지 않고 (η,β)=(0.5,2.0)으로 고정한다.
    mixture_etas: tuple[float, ...] = (0.5,)
    mixture_bs: tuple[float, ...] = (2.0,)
    include_center0: bool = True

    # ── 관행 baseline schedule (중심이 λ_R*가 아닌 '데이터 무관' 기준들) ─────────
    # cosine_vp 는 항상 포함. 추가로 아래 둘을 켜고 끌 수 있다.
    #   - linear : VP linear-β(DDPM/Song VP-SDE) schedule이 유도하는 λ 분포.
    #   - uniform: λ 범위 전체에 균일(=아무 noise level도 선호 안 함)한 분포.
    # 둘 다 λ_R* 정보를 안 쓰는 '관행/대조' 기준이라, DMSR 기반 schedule의 이득을
    # 가늠하는 비교축이 된다.
    include_linear: bool = False
    include_uniform: bool = True
    # VP linear-β schedule 파라미터(연속형 VP-SDE 표준값, Song et al. 2021).
    linear_beta_min: float = 0.1
    linear_beta_max: float = 20.0

    # ── class-conditional + classifier-free guidance (CFG) ────────────────────
    # 생성 품질용 표준 기법(Ho & Salimans 2022). 모든 schedule에 동일 적용이라 비교
    # 공정성엔 영향 없음. 보편 관행값으로 둠 (Phase 3와 동일).
    class_cond: bool = True          # 클래스(digit 0/1) 조건부 학습 + 샘플링 시 CFG
    num_classes: int = 2
    cond_dropout_prob: float = 0.1   # 학습 중 라벨을 null로 떨구는 비율(표준 0.1)
    cfg_scale: float = 1.5           # guidance scale(표준 ~1.5)

    # 유의성 검정에서 다른 schedule들과 비교할 기준(baseline) schedule 이름.
    baseline_schedule: str = "cosine_vp"

    # ── 출력 ──────────────────────────────────────────────────────────────────
    run_name: str = "phase2_2_mnist_pilot"
    data_root: str = "./data"


@dataclass(frozen=True)
class ScheduleSpec:
    """하나의 training noise distribution p_train(λ)을 기술하는 명세."""
    name: str                       # 식별용 이름 (그래프·CSV에 그대로 사용)
    kind: str                       # 샘플러 종류 (sample_schedule에서 분기)
    center_lambda: float | None = None   # 중심 λ (DMSR 기반 schedule이면 λ_R*)
    scale: float | None = None      # 폭/scale (mixture에서는 TruncatedLaplace β)
    eta: float | None = None        # mixture peak 혼합비 η
    note: str = ""                  # 사람이 읽을 설명
