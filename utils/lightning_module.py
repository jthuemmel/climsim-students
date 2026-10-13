import math
from pathlib import Path

import lightning as L
import torch
import xarray as xr
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from score import kaggle_score, score
from utils.config import Config
from utils.dataset import ClimSimData
from utils.paper_mlp import MLP


class ClimSimModule(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.to_dict())

        self.train_dataset = ClimSimData(self.data_cfg.train, self.data_cfg.inputs, self.data_cfg.targets)
        # An example of a validation loop.  These months are also in the test data: use other data to select a model.
        self.val_dataset = self.dataset(self.data_cfg.test, time=slice(*self.data_cfg.val_time))
        n_in, n_out = self.train_dataset.x.shape[1], self.train_dataset.y.shape[1]
        self.model = MLP(n_in, n_out, **vars(config.network), floor=self.surface_floor())

    @property
    def cfg(self):
        return self.config.trainer

    @property
    def data_cfg(self):
        return self.config.dataset

    def dataset(self, path: str, time: slice | None = None) -> ClimSimData:
        """A store with the variables of the configuration and the statistics of the training data."""
        return ClimSimData(path, self.data_cfg.inputs, self.data_cfg.targets, time=time, stats=self.train_dataset.stats)

    def surface_floor(self) -> torch.Tensor:
        """The standardised value of zero for each surface target (`cam_out_...`), and no floor for the other targets."""
        mean, std = self.train_dataset.stats["y"]
        surface = torch.cat([torch.full((width,), name.startswith("cam_out"))
                             for name, width in self.train_dataset.widths.items()])
        return torch.where(surface, -mean / torch.where(std > 0, std, 1.0), -torch.inf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).float()  # float32 also with mixed precision

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = ((self(x) - y) ** 2).mean()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val/loss", ((self(x) - y) ** 2).mean(), prog_bar=True)

    # TEST
    def on_test_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        self.log("test/loss", ((prediction - y) ** 2).mean())
        self.test_outputs.append(prediction.cpu())

    def on_test_epoch_end(self):
        """The scores of score.py for the test data.  With `store_predictions`, also predictions.zarr."""
        prediction = self.test_dataset.to_xarray(torch.cat(self.test_outputs))
        scores = score(prediction, xr.open_zarr(self.data_cfg.test))
        self.log("test/kaggle_score", kaggle_score(scores))
        run_dir = Path(self.trainer.default_root_dir)
        scores.to_zarr(run_dir / "scores.zarr", mode="w")
        if self.cfg.store_predictions:
            prediction.chunk({"time": 32}).to_zarr(run_dir / "predictions.zarr", mode="w")

    # OPTIMISATION
    def configure_optimizers(self):
        optimizers = {"radam": torch.optim.RAdam, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        optimizer = optimizers[self.cfg.optimizer](self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_factor)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def lr_factor(self, step: int) -> float:
        """The learning rate at an optimiser step, as a factor of `trainer.lr`."""
        schedule = self.cfg.schedule
        if schedule["type"] == "constant":
            return 1.0
        if schedule["type"] == "cyclic":  # triangular2: from lr to max_lr and back in 2 * step_size steps, half the peak each cycle
            cycle = math.floor(1 + step / (2 * schedule["step_size"]))
            x = abs(step / schedule["step_size"] - 2 * cycle + 1)
            return 1 + (schedule["max_lr"] / self.cfg.lr - 1) * max(0.0, 1 - x) / 2 ** (cycle - 1)
        raise ValueError(f"unknown schedule type: {schedule['type']}")

    # DATA
    def loader(self, dataset: ClimSimData, shuffle: bool) -> DataLoader:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batches = BatchSampler(sampler, self.cfg.batch_size, drop_last=shuffle)
        return DataLoader(dataset, batch_size=None, sampler=batches, num_workers=self.cfg.num_workers)

    def train_dataloader(self):
        return self.loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        self.test_dataset = self.dataset(self.data_cfg.test)
        return self.loader(self.test_dataset, shuffle=False)
