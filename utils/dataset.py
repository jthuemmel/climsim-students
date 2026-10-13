import einops
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

# the variables of the ClimSim paper
INPUTS = ["state_t", "state_q0001", "state_ps", "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX"]
TARGETS = ["ptend_t", "ptend_q0001", "cam_out_NETSW", "cam_out_FLWDS", "cam_out_PRECSC", "cam_out_PRECC",
           "cam_out_SOLS", "cam_out_SOLL", "cam_out_SOLSD", "cam_out_SOLLD"]

# all variables, in the stores of `python fetch_data.py --full`
FULL_INPUTS = ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v", "pbuf_ozone",
               "state_ps", "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX", "pbuf_TAUX", "pbuf_TAUY", "pbuf_COSZRS",
               "cam_in_ALDIF", "cam_in_ALDIR", "cam_in_ASDIF", "cam_in_ASDIR", "cam_in_LWUP", "cam_in_ICEFRAC",
               "cam_in_LANDFRAC", "cam_in_OCNFRAC", "cam_in_SNOWHLAND", "cam_in_SNOWHICE"]
FULL_TARGETS = ["ptend_t", "ptend_q0001", "ptend_q0002", "ptend_q0003", "ptend_u", "ptend_v", *TARGETS[2:]]

# The smallest std of an input channel, as a part of the largest std of its variable.  Some levels of an input are
# almost constant in the training data.  Without this limit, a different value in other data becomes very large.
STD_FLOOR = 1e-3


class ClimSimData(Dataset):
    """One store in memory as standardised tensors.  A row is one column of the atmosphere at one time.

    `x` and `y` have the dimensions (row, channel).  A variable with levels gives one channel for each level.
    The rows have the order (time, ncol).  Each channel is standardised with its mean and std.
    The std of an input channel is not smaller than STD_FLOOR times the largest std of its variable.
    `stats` gives the statistics of another dataset: use the statistics of the training data for the test data.
    """

    def __init__(self, path: str, inputs: list[str] | None = None, targets: list[str] | None = None,
                 time: slice | None = None, stats: dict | None = None):
        self.inputs, self.targets = inputs or INPUTS, targets or TARGETS
        ds = xr.open_zarr(path)[self.inputs + self.targets].sel(time=time or slice(None))
        self.coords = ds.coords.to_dataset()  # time, ncol, level, lat, lon, area
        self.widths = {name: ds[name].sizes.get("level", 1) for name in self.targets}

        self.x, x_stats = self._to_tensor(ds, self.inputs, floor=STD_FLOOR)
        self.y, y_stats = self._to_tensor(ds, self.targets)
        self.stats = stats or {"x": x_stats, "y": y_stats}
        for values, (mean, std) in ((self.x, self.stats["x"]), (self.y, self.stats["y"])):
            values.sub_(mean).div_(torch.where(std > 0, std, 1.0))  # a constant channel becomes 0

    @staticmethod
    def _to_tensor(ds: xr.Dataset, names: list[str], floor: float = 0.0) -> tuple:
        """(row, channel) float32 with the mean and std of each channel.

        The tensor is in shared memory, and one variable at a time is read from the store.
        Thus the data is in memory only once, also with `num_workers` > 0.
        """
        rows = ds.sizes["time"] * ds.sizes["ncol"]
        widths = [ds[name].sizes.get("level", 1) for name in names]
        out = torch.empty(rows, sum(widths)).share_memory_()
        mean, std, start = [], [], 0
        for name, width in zip(names, widths):
            values = ds[name].transpose("time", "ncol", ...).values.reshape(rows, width)
            out[:, start:start + width] = torch.from_numpy(values)
            mean.append(values.mean(0, dtype=np.float64))
            std.append(values.std(0, dtype=np.float64))
            std[-1] = np.maximum(std[-1], floor * std[-1].max())
            start += width
        return out, (torch.tensor(np.concatenate(mean), dtype=torch.float32),
                     torch.tensor(np.concatenate(std), dtype=torch.float32))

    def to_xarray(self, y: torch.Tensor) -> xr.Dataset:
        """Standardised targets or predictions, (row, channel), in physical units on (time, ncol, level)."""
        mean, std = self.stats["y"]
        y = einops.rearrange(y.float().cpu() * std + mean, "(time ncol) channel -> time ncol channel",
                             ncol=self.coords.sizes["ncol"]).numpy()
        out, start = {}, 0
        for name, width in self.widths.items():
            out[name] = (("time", "ncol", "level"), y[..., start:start + width]) if width > 1 else (("time", "ncol"), y[..., start])
            start += width
        return xr.Dataset(out, coords=self.coords.coords)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """`idx` is one row or a list of rows.  The loaders give a list, thus one item is one batch."""
        return self.x[idx], self.y[idx]
