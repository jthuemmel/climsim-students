from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class DatasetConfig:
    train: str = "data/train.zarr"
    test: str = "data/test.zarr"
    val_time: list[str] = field(default_factory=lambda: ["0008-02", "0008-02"])  # the first and the last month of the
    #                                              validation data, a part of the test store
    inputs: Optional[list[str]] = None  # None gives the inputs of the ClimSim paper
    targets: Optional[list[str]] = None  # None gives the targets of the ClimSim paper


@dataclass
class NetworkConfig:
    hidden: list[int] = field(default_factory=lambda: [768, 640, 512, 640, 640, 128])
    negative_slope: float = 0.15


@dataclass
class TrainerConfig:
    optimizer: str = "radam"  # radam, adam, or adamw
    lr: float = 2.5e-4
    weight_decay: float = 0.0
    batch_size: int = 3072
    max_steps: int = 11844
    val_every_n_steps: int = 500
    schedule: dict = field(default_factory=lambda: {"type": "constant"})  # or {type: cyclic, max_lr: ..., step_size: ...}
    num_workers: int = 0
    accelerator: str = "auto"  # as in Lightning: auto, gpu, mps, or cpu
    devices: Union[int, list] = 1
    precision: str = "bf16-mixed"  # as in Lightning; "32-true" gives float32
    progress_bar: bool = True
    store_predictions: bool = False  # write predictions.zarr in the test step, for score.py


@dataclass
class Config:
    name: str = "paper_mlp"
    seed: int = 0
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_dict(cls, cfg: dict) -> "Config":
        return cls(
            name=cfg.get("name", "paper_mlp"),
            seed=cfg.get("seed", 0),
            dataset=DatasetConfig(**cfg.get("dataset", {})),
            network=NetworkConfig(**cfg.get("network", {})),
            trainer=TrainerConfig(**cfg.get("trainer", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)
