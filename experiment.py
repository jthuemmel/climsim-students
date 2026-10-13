"""Train the model of one configuration, then test it.

    python experiment.py --config configs/paper_mlp.yaml
    python experiment.py --config configs/paper_mlp.yaml --seed 1

The outputs are in runs/<name>/seed<k>/:
    config.yaml        the configuration of the run
    metrics.csv        the training loss, the validation loss, the test loss, and the Kaggle score
    model.ckpt         the trained weights
    scores.zarr        the scores of score.py for the test data
    predictions.zarr   with trainer.store_predictions: the predictions for the test data
"""
import argparse
from pathlib import Path

import lightning as L
import xarray as xr
import yaml
from lightning.pytorch.loggers import CSVLogger

from score import TARGETS, kaggle_score
from utils.config import Config
from utils.lightning_module import ClimSimModule


def run(config: Config) -> Path:
    """Train and test the model of a configuration.  The result is the directory of the run."""
    run_dir = Path("runs") / config.name / f"seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(config.to_dict(), open(run_dir / "config.yaml", "w"), sort_keys=False)

    L.seed_everything(config.seed, workers=True)
    module = ClimSimModule(config)
    cfg = config.trainer
    trainer = L.Trainer(
        max_steps=cfg.max_steps,
        max_epochs=-1,
        val_check_interval=cfg.val_every_n_steps,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=CSVLogger(run_dir, name="", version=""),
        default_root_dir=run_dir,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=cfg.progress_bar,
    )
    trainer.fit(module)
    trainer.save_checkpoint(run_dir / "model.ckpt")
    trainer.test(module)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.seed = args.seed if args.seed is not None else config.seed
    scores = xr.open_zarr(run(config) / "scores.zarr")[TARGETS]
    print(scores.mean("level").to_array("variable").to_pandas().to_string(float_format="{:.4g}".format))
    print(f"Kaggle score: {kaggle_score(scores):.4f}")


if __name__ == "__main__":
    main()
