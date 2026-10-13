# ClimSim seminar

This repository trains a neural network on the ClimSim data (Yu, Hu et al. 2024) and scores its predictions.
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

1. `fetch_data.py` downloads the raw files from Hugging Face in blocks and writes `data/train.zarr` and `data/test.zarr`.
   The script deletes each block of raw files after it writes the block to the stores.
   The download takes approximately 50 minutes.
   `python fetch_data.py --full` writes all variables.
   The two stores use approximately 4 GB of disk space, or 12 GB with `--full`.
3. `experiment.py` trains the MLP of the ClimSim paper and writes the results to `runs/<name>/seed<k>/`.
   During the training, a validation loss shows the progress. The validation data is one month of the test store.
   These months are also in the test data. Use other data to select a model.
4. The test step calculates the scores for `data/test.zarr` and writes `scores.zarr`.
   Set `trainer.store_predictions: true` in the configuration to write `predictions.zarr` also.
5. `score.py` calculates the same scores from a stored `predictions.zarr`.

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

## Recommended reading

### For everyone

- Yu, Hu et al. (2024). ClimSim-Online: A Large Multi-scale Dataset and Framework for Hybrid ML-physics Climate Emulation. [arXiv:2306.08754](https://arxiv.org/abs/2306.08754)
- Lin et al. (2025). Crowdsourcing the Frontier: Advancing Hybrid Physics-ML Climate Simulation via a $50,000 Kaggle Competition. [arXiv:2511.20963](https://arxiv.org/abs/2511.20963)

### Recommended for learned climate simulation
- Rasp, Pritchard & Gentine (2018). Deep learning to represent sub-grid processes in climate models. [arXiv:1806.04731](https://arxiv.org/abs/1806.04731)
- Kochkov et al. (2024). Neural general circulation models for weather and climate (NeuralGCM). Nature. [arXiv:2311.07222](https://arxiv.org/abs/2311.07222), [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)

### How to train your neural network
- Karpathy (2019). A Recipe for Training Neural Networks. <https://karpathy.github.io/2019/04/25/recipe/>
- Godbole et al. (2023). Deep Learning Tuning Playbook. <https://github.com/google-research/tuning_playbook>

### Baselines

The ClimSim paper compares the MLP with these models. The label is the name of the model in the paper.

- CNN: Cachay et al. (2021). ClimART: A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and Climate Models. [arXiv:2111.14671](https://arxiv.org/abs/2111.14671)
- ED: Behrens et al. (2022). Non-Linear Dimensionality Reduction With a Variational Encoder Decoder to Understand Convective Processes in Climate Models. [arXiv:2204.08708](https://arxiv.org/abs/2204.08708)
- HSR: Wong-Toi et al. (2023). Understanding Pathologies of Deep Heteroskedastic Regression. [arXiv:2306.16717](https://arxiv.org/abs/2306.16717)
- RPN: Osband, Aslanides & Cassirer (2018). Randomized Prior Functions for Deep Reinforcement Learning. [arXiv:1806.03335](https://arxiv.org/abs/1806.03335)
- cVAE: Kingma & Welling (2014). Auto-Encoding Variational Bayes. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- cVAE: Sohn, Lee & Yan (2015). Learning Structured Output Representation using Deep Conditional Generative Models. [NeurIPS 2015](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html).

### Extensions: probabilistic and generative models

- Alet et al. (2025). Skillful joint probabilistic weather forecasting from marginals (FGN). [arXiv:2506.10772](https://arxiv.org/abs/2506.10772)
- Seitzer et al. (2022). On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks. [arXiv:2203.09168](https://arxiv.org/abs/2203.09168)
- Lakshminarayanan, Pritzel & Blundell (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
- Gneiting & Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. JASA. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
- Higgins et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. [ICLR 2017](https://openreview.net/forum?id=Sy2fzU9gl).

### Extensions: physical constraints

- Beucler et al. (2021). Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems. [arXiv:1909.00912](https://arxiv.org/abs/1909.00912)
- Hu et al. (2024). Stable machine-learning parameterization of subgrid processes with real geography and full-physics emulation. [arXiv:2407.00124](https://arxiv.org/abs/2407.00124)

### Extensions: masking

- He et al. (2022). Masked Autoencoders Are Scalable Vision Learners. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

### Extensions: architecture and scaling

- Jaegle et al. (2022). Perceiver IO: A General Architecture for Structured Inputs & Outputs. [arXiv:2107.14795](https://arxiv.org/abs/2107.14795)
- Tolstikhin et al. (2021). MLP-Mixer: An all-MLP Architecture for Vision. [arXiv:2105.01601](https://arxiv.org/abs/2105.01601)
- Yu et al. (2022). MetaFormer Is Actually What You Need for Vision. [arXiv:2111.11418](https://arxiv.org/abs/2111.11418)
- Peebles & Xie (2023). Scalable Diffusion Models with Transformers. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
