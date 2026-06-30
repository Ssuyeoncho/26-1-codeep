from __future__ import annotations

from dataclasses import dataclass


CIFAR_CLASS_IDX: dict[str, int] = {
    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
    "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9,
}

CLASS_PAIRS: dict[str, tuple[int, int]] = {
    # 시각적으로 '비슷한' 쌍 (구분이 어려움 → DMSR 분리도가 낮고 transition이 늦게 옴 예상)
    "cat_vs_dog": (3, 5),                 # 비슷 (네발짐승, 고전적 난이도)
    "deer_vs_horse": (4, 7),              # 비슷 (네발짐승)
    "automobile_vs_truck": (1, 9),        # 비슷 (탈것)
    # 시각적으로 '확연히 다른' 쌍 (DMSR 분리도가 높고 transition이 일찍 올 것으로 예상)
    "airplane_vs_automobile": (0, 1),     # 다름 (하늘 탈것 vs 지상 탈것)
    "airplane_vs_frog": (0, 6),           # 확연히 다름 (큰 인공물 vs 작은 양서류)
}

PRESETS: dict[str, dict] = {
    "smoke": dict(clf_epochs=2, train_steps=50, n_gen_samples=64,
                  dmsr_n_samples=100, dmsr_grid_size=10, eval_n_samples=64,
                  compute_fid=False, num_seeds=1),
    "fast":  dict(clf_epochs=10, train_steps=50_000, n_gen_samples=5_000,
                  dmsr_n_samples=500, dmsr_grid_size=30, eval_n_samples=256,
                  compute_fid=True, num_seeds=1),
    "full":  dict(clf_epochs=20, train_steps=100_000, n_gen_samples=10_000,
                  dmsr_n_samples=1000, dmsr_grid_size=50, eval_n_samples=512,
                  compute_fid=True, num_seeds=1),
}


@dataclass(frozen=True)
class ExperimentConfig:
    class_pair: str = "airplane_vs_automobile"
    image_size: int = 32
    lambda_min: float = -15.0
    lambda_max: float = 15.0
    rho: float = 0.5
    # Classifier
    clf_epochs: int = 20
    clf_batch_size: int = 256
    clf_lr: float = 1e-3
    clf_feature_dim: int = 128
    # DMSR
    dmsr_grid_size: int = 50
    dmsr_n_samples: int = 1000
    # Diffusion training
    train_steps: int = 100_000
    batch_size: int = 128
    # Effective batch_size는 유지하되, GPU에는 micro_batch_size 단위로 쪼개 올린다.
    # None이면 기존처럼 한 번에 batch_size 전체를 처리한다.
    micro_batch_size: int | None = None
    lr: float = 2e-4
    ema_decay: float = 0.9999
    base_ch: int = 64
    num_res_blocks: int = 2
    # Eval
    eval_grid_size: int = 40
    eval_n_samples: int = 512
    n_gen_samples: int = 10_000
    ddim_steps: int = 50
    # DDIM 샘플링 λ 범위. DMSR 분석용 [lambda_min, lambda_max](=[-15,15])와 분리한다.
    # 샘플링은 cosine 격자로 중앙(λ≈0)에 step을 집중시키는 게 유리하므로 좁게 잡는다.
    # (Phase 2와 동일한 범위·격자 함수를 사용 → 두 phase sampler 완전 일치)
    ddim_lambda_min: float = -8.0
    ddim_lambda_max: float = 8.0
    # 생성 시 한 번에 만드는 배치 크기 (학습 결과와 무관 — GPU 처리량용. VRAM 여유 크면 키워도 됨)
    gen_batch_size: int = 512
    # Schedule 구성: {Normal, Laplace} × {중심 0, 중심 λ_R*} × {폭 sweep} 매칭 factorial.
    # 같은 모양·폭에서 중심만 0(선행연구 류)/λ_R*(우리 것)로 바꿔 '중심 위치 효과' 통제비교.
    # 폭은 '확연히 차이나는' 값들로(모임↔퍼짐). Phase 2와 동일 구성.
    width_values: tuple[float, ...] = (0.5, 1.5, 4.0)   # Normal s / Laplace b 공통 폭 sweep
    include_center0: bool = True    # 중심 0 대조군(Normal@0, Laplace@0=Hang) 포함 여부
    # (구) DMSR×cosine 혼합 — 기본 OFF (--include-cosmix로만 켬). 코드만 보존.
    include_cosmix: bool = False
    mix_weights: tuple[float, ...] = (0.5, 0.8)
    mix_scale: float = 1.0
    # 관행/대조 baseline (λ_R* 정보를 안 쓰는 데이터 무관 기준). cosine_vp는 항상 포함.
    include_linear: bool = True     # VP linear-β(DDPM) 유도분포
    include_uniform: bool = True    # λ 범위 균일분포
    linear_beta_min: float = 0.1    # VP linear-β 파라미터 (Song et al.)
    linear_beta_max: float = 20.0
    # 유의성 검정에서 다른 schedule들과 비교할 기준(baseline) schedule 이름.
    baseline_schedule: str = "cosine_vp"
    # Misc
    seed: int = 20260526
    # num_seeds ≥ 3 으로 주면 seed 간 통계 집계·유의성 검정이 의미를 갖는다.
    # 동일 seed_idx 안에서 모든 schedule이 같은 run_seed를 공유하므로 paired 비교가 된다.
    num_seeds: int = 1
    device: str = "auto"
    # ── GPU 가속 (CUDA 서버용, 현재 = RTX 4090 / Ada Lovelace, 24GB) ──────────
    # 4090(Ada)은 bf16/TF32 지원 → amp="auto"가 bf16을 고른다.
    num_workers: int = 4          # DataLoader 병렬 로딩 (서버 CPU 코어에 맞춰 조정)
    prefetch_factor: int = 2      # 워커당 미리 읽을 batch 수. 낮출수록 CPU/RAM 순간 부하 감소.
    amp: str = "auto"             # 혼합정밀: auto(=CUDA면 bf16)/bf16/fp16/fp32
    # torch.compile은 nvcc 권한 등 환경 의존성이 있어 기본 OFF. 가능한 환경에서만 --compile로 켠다.
    compile_model: bool = False
    data_root: str = ""          # set at runtime from PROJECT_DIR
    run_name: str = "phase3_cifar"
    compute_fid: bool = True


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    kind: str
    center_lambda: float | None = None
    scale: float | None = None
    weight: float | None = None     # 혼합 비율 w (dmsr_cosine_mix에서 N 성분 비중)
    note: str = ""
