from __future__ import annotations

from dataclasses import dataclass


CIFAR_CLASS_IDX: dict[str, int] = {
    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
    "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9,
}

CLASS_PAIRS: dict[str, tuple[int, int]] = {
    "airplane_vs_automobile": (0, 1),
    "cat_vs_dog": (3, 5),
    "deer_vs_horse": (4, 7),
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
    lr: float = 2e-4
    ema_decay: float = 0.9999
    base_ch: int = 64
    num_res_blocks: int = 2
    # Eval
    eval_grid_size: int = 40
    eval_n_samples: int = 512
    n_gen_samples: int = 10_000
    ddim_steps: int = 50
    # Schedule sweep
    s_values: tuple[float, ...] = (1.5, 0.8, 0.3)
    laplace_b: float = 0.5
    # Misc
    seed: int = 20260526
    num_seeds: int = 1
    device: str = "auto"
    num_workers: int = 0
    data_root: str = ""          # set at runtime from PROJECT_DIR
    run_name: str = "phase3_cifar"
    compute_fid: bool = True


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    kind: str
    center_lambda: float | None = None
    scale: float | None = None
    note: str = ""
