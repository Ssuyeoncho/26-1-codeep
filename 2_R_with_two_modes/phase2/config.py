from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    # Data
    digits: tuple = (0, 1)
    img_size: int = 28

    # VP noise range
    lambda_min: float = -10.0
    lambda_max: float = 10.0
    rho: float = 0.5

    # Feature classifier φ
    clf_epochs: int = 10
    clf_lr: float = 1e-3
    clf_batch_size: int = 256
    clf_feature_dim: int = 64

    # Empirical DMSR computation
    dmsr_grid_size: int = 40
    dmsr_n_samples: int = 512

    # Mini U-Net
    base_ch: int = 32
    time_emb_dim: int = 128

    # Denoiser training
    train_steps: int = 20000
    batch_size: int = 128
    lr: float = 2e-4
    eval_batch_size: int = 256
    eval_grid_size: int = 40
    seed: int = 20260526
    num_seeds: int = 1
    device: str = "cpu"

    # DDIM generation
    ddim_steps: int = 50
    n_generate: int = 5000
    gen_batch_size: int = 100
    ddim_lambda_min: float = -8.0
    ddim_lambda_max: float = 6.0

    # Output
    run_name: str = "phase2_mnist"
    data_root: str = "./data"


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    kind: str
    center_lambda: float | None = None
    scale: float | None = None
    note: str = ""
