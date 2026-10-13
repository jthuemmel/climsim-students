# ClimSim seminar

This repository trains a neural network on the ClimSim data (Yu et al. 2023) and scores its predictions.
The network learns the effect of clouds and radiation on one column of the atmosphere.

## Setup

```
conda env create -f environment.yaml
conda activate dlwp
```

If you have the environment `dlwp` from the lecture, use it.

## The four steps

```
python fetch_data.py                                  # download the data
python experiment.py --config configs/paper_mlp.yaml  # train the model, then test it
python score.py runs/paper_mlp/seed0/predictions.zarr # score stored predictions again
```

1. `fetch_data.py` downloads approximately 37 GB from Hugging Face in blocks and writes `data/train.zarr` and `data/test.zarr`.
   The script deletes each block of raw files after it writes the block to the stores.
   The download takes approximately 50 minutes.
   `python fetch_data.py --full` writes all variables.
2. `experiment.py` trains the MLP of the ClimSim paper and writes the results to `runs/<name>/seed<k>/`.
   During the training, a validation loss shows the progress. The validation data is one month of the test store.
   These months are also in the test data. Use other data to select a model.
3. The test step calculates the scores for `data/test.zarr` and writes `scores.zarr`.
   Set `trainer.store_predictions: true` in the configuration to write `predictions.zarr` also.
4. `score.py` calculates the same scores from a stored `predictions.zarr`.

## Start with the tutorial

`tutorial.ipynb` does all steps with a small quantity of data: a download of approximately 2 GB and a training of some minutes.
It shows the data, trains a small model, and plots the scores.

```
jupyter lab tutorial.ipynb
```

`GLOSSARY.md` explains the terms of atmospheric physics and the names of the variables.

## Files

| File | Contents |
| --- | --- |
| `fetch_data.py` | The download and the two stores. |
| `experiment.py` | One run: train, test, scores. |
| `score.py` | The metrics. Each metric is one function. |
| `configs/paper_mlp.yaml` | The model and the training recipe of the ClimSim paper. |
| `configs/tutorial.yaml` | The small model of the tutorial. |
| `utils/dataset.py` | `ClimSimData`: a store as standardised tensors, and predictions back to xarray. |
| `utils/paper_mlp.py` | The example model. |
| `utils/lightning_module.py` | The training step, the test step, the optimiser, and the data loaders. |
| `utils/config.py` | The configuration as dataclasses. |
| `tutorial.ipynb` | All steps with a small quantity of data. |
| `GLOSSARY.md` | The terms of atmospheric physics and the variables. |

## The data

A store has the dimensions `(time, ncol, level)`: 384 columns and 60 levels.
One row of the training data is one column at one time.
The model gets 124 input values for a row and predicts 128 target values.

```python
import xarray as xr

ds = xr.open_zarr("data/train.zarr")
ds["state_t"]                                 # temperature, (time, ncol, level)
ds.sel(time=slice("0001-02", "0005-12"))      # a selection of months
```

`ClimSimData("data/train.zarr", time=slice("0001-02", "0005-12"))` loads a selection of months.

## The scores

- The Kaggle score is the mean R2 of the 128 targets on all rows of the test data.
  The targets with weight zero in the Kaggle competition count as 1.
- The paper scores are the MAE, the MSE, and the R2 in W/m2, as in the ClimSim paper.
- `python score.py <predictions> --probabilistic` also calculates the CRPS and the PIT histogram.
  The docstring of `score.py` gives the format of a probabilistic prediction.
