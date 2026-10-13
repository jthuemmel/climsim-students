"""Scores of a prediction on the test data.

    python score.py runs/mlp/predictions.zarr                    # writes runs/mlp/scores.zarr
    python score.py runs/mlp/predictions.zarr --probabilistic    # also the CRPS, and pit.zarr
    python score.py runs/mlp/predictions.zarr --probabilistic --no-fair

A prediction has the dimensions (time, ncol, level) and the variable names of the targets, in physical units.
A probabilistic prediction has one additional dimension:
    member     an ensemble, each member is one prediction
    parameter  a Gaussian, with the coordinate values "mean" and "std"

Two styles:
    kaggle  R2 of each target on all rows.  The Kaggle score is the mean of these values.
    paper   MAE, MSE, and R2 in W/m2, as in ClimSim (Yu et al. 2023, SI section 5).

Each metric is one function of (prediction, truth).  To add a metric, write a function and add it to `score`.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.special import ndtr

from utils.dataset import TARGETS

ROWS = ("time", "ncol")

GRAV = 9.80616  # m/s2
CP = 1.00464e3  # J/kg/K
LV = 2.501e6  # J/kg
RHO_H2O = 1.0e3  # kg/m3
ZERO_WEIGHT = {"ptend_q0001": 12}  # Kaggle gives weight zero to the levels above this one


# UNITS
def layer_mass(truth: xr.Dataset) -> xr.DataArray:
    """The mass of air in each level, dp / g in kg/m2, from the surface pressure of each row."""
    interfaces = truth.P0 * truth.hyai + truth.state_ps * truth.hybi
    return interfaces.diff("ilev").drop_vars("ilev").rename(ilev="level") / GRAV


def to_energy_units(ds: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """Heating, moistening, and precipitation in W/m2.  The radiative fluxes are in W/m2 already."""
    mass = layer_mass(truth)
    factors = {"ptend_t": CP * mass, "ptend_q0001": LV * mass, "cam_out_PRECSC": LV * RHO_H2O, "cam_out_PRECC": LV * RHO_H2O}
    return ds.assign({name: ds[name] * factors[name] for name in ds.data_vars if name in factors})


def area_weights(truth: xr.Dataset) -> xr.DataArray:
    return truth.area / truth.area.mean()


# KAGGLE STYLE
def r2_kaggle(pred: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """R2 on all rows.  A target with weight zero has the score 1."""
    sse = ((pred - truth) ** 2).sum(ROWS)
    sst = ((truth - truth.mean(ROWS)) ** 2).sum(ROWS)
    score = xr.where(sst > 0, 1 - sse / sst.where(sst > 0), sse == 0)  # a constant target: 1 if the prediction is equal
    return score.assign({name: score[name].where(score.level >= ZERO_WEIGHT[name], 1.0)
                         for name in score.data_vars if name in ZERO_WEIGHT})


# PAPER STYLE
def mae(pred: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    return np.abs(pred - truth).mean("time").weighted(area_weights(truth)).mean("ncol")


def mse(pred: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    return ((pred - truth) ** 2).mean("time").weighted(area_weights(truth)).mean("ncol")


def r2_paper(pred: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """R2 along time in each column, then the mean of the columns.  A column with a constant target has no R2."""
    sse = ((pred - truth) ** 2).sum("time")
    sst = ((truth - truth.mean("time")) ** 2).sum("time")
    return (1 - sse / sst.where(sst > 1e-12 * sst.max())).mean("ncol")


# PROBABILISTIC
def crps(ensemble: xr.Dataset, truth: xr.Dataset, fair: bool = True) -> xr.Dataset:
    """The CRPS of an ensemble, E|X - y| - E|X - X'| / 2.  The second term comes from the sorted members.

    The fair form is for members that are random samples.  Use the other form for members that are fixed quantiles.
    """
    m = ensemble.sizes["member"]
    absolute_error = np.abs(ensemble - truth).mean("member")
    ranked = xr.apply_ufunc(np.sort, ensemble, input_core_dims=[["member"]], output_core_dims=[["member"]])
    coefficient = xr.DataArray(2 * np.arange(1, m + 1) - m - 1, dims="member")
    spread = (coefficient * ranked).sum("member") / (m * (m - 1) if fair else m ** 2)
    return (absolute_error - spread).mean("time").weighted(area_weights(truth)).mean("ncol")


def crps_gaussian(mean: xr.Dataset, std: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """The CRPS of a Gaussian in closed form."""
    std = std.clip(min=1e-12)
    z = (truth - mean) / std
    density = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    score = std * (z * (2 * xr.apply_ufunc(ndtr, z) - 1) + 2 * density - 1 / np.sqrt(np.pi))
    return score.mean("time").weighted(area_weights(truth)).mean("ncol")


def pit_histogram(pred: xr.DataArray, truth: xr.DataArray, bins: int = 10) -> np.ndarray:
    """The position of the truth in the predicted distribution.  A calibrated prediction gives a flat histogram.

    An ensemble of m members gives the m + 1 ranks of the truth.  A Gaussian gives `bins` intervals of its CDF.
    """
    if "member" in pred.dims:
        rank = (pred < truth).sum("member").values.ravel()
        return np.bincount(rank, minlength=pred.sizes["member"] + 1) / rank.size
    z = (truth - pred.sel(parameter="mean")) / pred.sel(parameter="std")
    pit = ndtr(z.values.ravel())
    pit = pit[np.isfinite(pit)]  # a target with std 0 has no position
    return np.histogram(pit, bins=bins, range=(0, 1))[0] / pit.size


# THE TABLE
def score_target(pred: xr.Dataset, truth: xr.Dataset, probabilistic: bool = False, fair: bool = True) -> xr.Dataset:
    """Every metric of one target, in float64.  Sums of 1.7 million float32 values are not accurate."""
    name, = pred.data_vars
    pred, truth = pred.astype("float64").compute(), truth[[name, "state_ps", "hyai", "hybi", "P0"]].astype("float64").compute()
    assert bool(np.isfinite(pred[name]).all()), f"The prediction of {name} contains NaN or infinite values."
    distribution = pred
    if "member" in pred.dims:
        pred = pred.mean("member")
    elif "parameter" in pred.dims:
        pred = pred.sel(parameter="mean", drop=True)

    metrics = {"r2_kaggle": r2_kaggle(pred, truth[[name]])}
    pred, target = to_energy_units(pred, truth), to_energy_units(truth[[name]], truth)
    metrics |= {"r2": r2_paper(pred, target), "mae": mae(pred, target), "mse": mse(pred, target)}
    if probabilistic:
        assert {"member", "parameter"} & set(distribution.dims), "A probabilistic score needs the dimension `member` or `parameter`."
        distribution = to_energy_units(distribution, truth)
        if "member" in distribution.dims:
            metrics["crps"] = crps(distribution, target, fair)
        else:
            mean, std = (distribution.sel(parameter=p, drop=True) for p in ("mean", "std"))
            metrics["crps"] = crps_gaussian(mean, std, target)
    return xr.concat(list(metrics.values()), dim=xr.DataArray(list(metrics), dims="metric", name="metric"))


def score(pred: xr.Dataset, truth: xr.Dataset, probabilistic: bool = False, fair: bool = True) -> xr.Dataset:
    """The scores of a prediction, with the dimensions (metric, level).  One target at a time, to limit the memory."""
    xr.align(pred[TARGETS], truth[TARGETS], join="exact", exclude=["member", "parameter"])
    return xr.merge([score_target(pred[[name]], truth, probabilistic, fair) for name in TARGETS], combine_attrs="drop")


def kaggle_score(scores: xr.Dataset) -> float:
    """The mean R2 of all targets.  Each level of a profile is one target."""
    r2 = scores.sel(metric="r2_kaggle")
    return float(sum(r2[name].sum() for name in TARGETS) / sum(r2[name].size for name in TARGETS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--truth", type=Path, default=Path("data/test.zarr"))
    parser.add_argument("--probabilistic", action="store_true", help="also the CRPS and the PIT histogram")
    parser.add_argument("--fair", action=argparse.BooleanOptionalAction, default=True, help="the fair CRPS of an ensemble")
    args = parser.parse_args()

    pred, truth = xr.open_zarr(args.predictions), xr.open_zarr(args.truth)
    scores = score(pred, truth, args.probabilistic, args.fair)
    scores.to_zarr(args.predictions.parent / "scores.zarr", mode="w")
    print(scores.mean("level").to_array("variable").to_pandas().to_string(float_format="{:.4g}".format))  # the mean along the levels
    print(f"Kaggle score: {kaggle_score(scores):.4f}")

    if args.probabilistic:
        histogram = xr.Dataset({name: ("bin", pit_histogram(pred[name], truth[name])) for name in TARGETS})
        histogram.to_zarr(args.predictions.parent / "pit.zarr", mode="w")


if __name__ == "__main__":
    main()
