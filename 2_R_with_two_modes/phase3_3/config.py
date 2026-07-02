from __future__ import annotations

from dataclasses import dataclass


IMAGENETTE_CLASSES: dict[str, str] = {
    "tench": "n01440764",
    "english_springer": "n02102040",
    "cassette_player": "n02979186",
    "chain_saw": "n03000684",
    "church": "n03028079",
    "french_horn": "n03394916",
    "garbage_truck": "n03417042",
    "gas_pump": "n03425413",
    "golf_ball": "n03445777",
    "parachute": "n03888257",
}


CLASS_PAIRS: dict[str, tuple[str, str]] = {
    # Easy, visually distinct pairs. Best first checks for 64x64 diffusion.
    "gas_pump_vs_golf_ball": ("gas_pump", "golf_ball"),
    "church_vs_garbage_truck": ("church", "garbage_truck"),
    "english_springer_vs_garbage_truck": ("english_springer", "garbage_truck"),
    # Harder/diagnostic pairs.
    "tench_vs_english_springer": ("tench", "english_springer"),
    "cassette_player_vs_chain_saw": ("cassette_player", "chain_saw"),
}


PRESETS: dict[str, dict] = {
    "smoke": dict(clf_epochs=1, train_steps=20, n_gen_samples=32,
                  dmsr_n_samples=64, dmsr_grid_size=8, eval_n_samples=32,
                  compute_fid=False, num_seeds=1),
    "fast":  dict(clf_epochs=6, train_steps=50_000, n_gen_samples=2_000,
                  dmsr_n_samples=400, dmsr_grid_size=30, eval_n_samples=192,
                  compute_fid=True, num_seeds=1),
    "full":  dict(clf_epochs=12, train_steps=100_000, n_gen_samples=5_000,
                  dmsr_n_samples=800, dmsr_grid_size=50, eval_n_samples=384,
                  compute_fid=True, num_seeds=1),
}


@dataclass(frozen=True)
class ExperimentConfig:
    class_pair: str = "gas_pump_vs_golf_ball"
    image_size: int = 64
    lambda_min: float = -8.0
    lambda_max: float = 8.0
    rho: float = 0.5
    # Classifier
    clf_epochs: int = 12
    clf_batch_size: int = 128
    clf_lr: float = 1e-3
    clf_feature_dim: int = 128
    # DMSR
    dmsr_grid_size: int = 50
    dmsr_n_samples: int = 800
    # Diffusion training
    train_steps: int = 100_000
    batch_size: int = 64
    micro_batch_size: int | None = 16
    lr: float = 2e-4
    ema_decay: float = 0.9999
    base_ch: int = 64
    num_res_blocks: int = 2
    # Eval
    eval_grid_size: int = 40
    eval_n_samples: int = 384
    n_gen_samples: int = 5_000
    ddim_steps: int = 50
    ddim_lambda_min: float = -8.0
    ddim_lambda_max: float = 8.0
    gen_batch_size: int = 128
    # Phase 3.3: same mixture family as Phase 3.2.
    mixture_etas: tuple[float, ...] = (0.5,)
    mixture_bs: tuple[float, ...] = (1.0,)
    include_center0: bool = True
    include_linear: bool = False
    include_uniform: bool = True
    linear_beta_min: float = 0.1
    linear_beta_max: float = 20.0
    # class-conditional + CFG
    class_cond: bool = True
    num_classes: int = 2
    cond_dropout_prob: float = 0.1
    cfg_scale: float = 1.5
    baseline_schedule: str = "cosine_vp"
    # Misc
    seed: int = 20260526
    num_seeds: int = 1
    device: str = "auto"
    num_workers: int = 4
    prefetch_factor: int = 2
    amp: str = "auto"
    compile_model: bool = False
    data_root: str = ""
    run_name: str = "phase3_3_imagenette64"
    compute_fid: bool = True


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    kind: str
    center_lambda: float | None = None
    scale: float | None = None
    eta: float | None = None
    note: str = ""
