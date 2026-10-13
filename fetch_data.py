"""Download ClimSim (low resolution, real geography) from Hugging Face and write the two stores.

    python fetch_data.py            # the variables of the ClimSim paper: 6 inputs, 10 targets
    python fetch_data.py --full     # all variables: 24 inputs, 14 targets
    python fetch_data.py --dry-run  # only count the files and the bytes

    data/train.zarr   0001-02 to 0007-12, every 23rd timestep
    data/test.zarr    0008-02 to 0009-01, every 6th timestep

A store has the dimensions (time, ncol, level), with the latitude, longitude, and area of each column.
`time` is a calendar coordinate (years of 365 days), thus `ds.sel(time=slice("0001-02", "0003-12"))` selects months.
A tendency is (state after - state before) / 1200 s.

The script downloads one block of timesteps, writes the block to the stores, and deletes the raw files.
If the script stops, run it again: it continues after the last complete block.
The raw files contain all variables.  Thus the two variable sets need the same download.
To change the variable set, delete the stores and run the script again.

Hugging Face permits 3,000 file requests in 5 minutes without an account.  The script stays below this limit.
"""
import argparse
import datetime
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import dask.array
import numpy as np
import requests
import xarray as xr

from utils.dataset import FULL_INPUTS, FULL_TARGETS, INPUTS, TARGETS

DATA = Path(__file__).parent / "data"
HF = "https://huggingface.co/datasets/LEAP/ClimSim_low-res/resolve/main/"
GRID = "ClimSim_low-res_grid-info.nc"

# A larger TRAIN_STRIDE gives less data: the rows and the download change with 23 / TRAIN_STRIDE.
# A stride that has no common factor with 72 (the timesteps in one day) gives every time of the day.
TRAIN_MONTHS, TRAIN_STRIDE = ((1, 2), (7, 12)), 23
# The test data is the scoring data of the ClimSim paper.  Do not change these values.
TEST_MONTHS, TEST_STRIDE = ((8, 2), (9, 1)), 6

DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # the calendar has no leap years
TIMESTEP, STEPS_PER_DAY = 1200, 72  # seconds
BLOCK = 256  # timesteps; a multiple of the 32 timesteps in one chunk of a store


def timesteps(months: tuple, stride: int) -> list[tuple]:
    """(year, month, day, second) of every stride-th timestep of the months; the first and the last month are included."""
    (y, m), last, steps = months[0], months[1], []
    while (y, m) <= last:
        steps += [(y, m, d, s * TIMESTEP) for d in range(1, DAYS[m - 1] + 1) for s in range(STEPS_PER_DAY)]
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return steps[::stride]


def raw_path(step: tuple, kind: str) -> str:
    """The file of one timestep.  `mli` has the state before the timestep, `mlo` has the state after it."""
    y, m, d, s = step
    return f"train/{y:04d}-{m:02d}/E3SM-MMF.{kind}.{y:04d}-{m:02d}-{d:02d}-{s:05d}.nc"


# DOWNLOAD
class Pacer:
    """Gives all threads together a constant number of requests in one second."""

    def __init__(self, per_second: float = 9.5):
        self.interval, self.next, self.lock = 1.0 / per_second, time.monotonic(), threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next + self.interval, now)
            delay = self.next - now
        time.sleep(delay)


def fetch(session: requests.Session, pacer: Pacer, raw: Path, path: str, retries: int = 8) -> None:
    target = raw / path
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    for attempt in range(retries):
        try:
            pacer.wait()
            with session.get(HF + path, stream=True, timeout=60) as r:
                if r.status_code == 429:  # too many requests: wait until the limit resets
                    reset = re.search(r"t=(\d+)", r.headers.get("ratelimit", ""))
                    time.sleep(float(reset.group(1)) + 1 if reset else min(60, 2 ** attempt))
                    continue
                if r.status_code >= 500:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            part.rename(target)
            return
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"The download of {path} failed after {retries} attempts.")


def download(paths: list[str], raw: Path, session: requests.Session, pacer: Pacer, workers: int = 32) -> None:
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda path: fetch(session, pacer, raw, path), paths))


# STORES
def timestamp(step: tuple) -> cftime.DatetimeNoLeap:
    y, m, d, s = step
    return cftime.DatetimeNoLeap(y, m, d) + datetime.timedelta(seconds=s)


def read_step(raw: Path, step: tuple, names: list[str]) -> dict:
    """The variables of one timestep: a tendency from the two files, a `cam_out` flux from mlo, an input from mli."""
    with xr.open_dataset(raw / raw_path(step, "mli")) as mli, xr.open_dataset(raw / raw_path(step, "mlo")) as mlo:
        out = {}
        for name in names:
            if name.startswith("ptend_"):
                state = name.replace("ptend", "state")
                out[name] = (mlo[state].values - mli[state].values) / TIMESTEP
            else:
                out[name] = (mlo if name.startswith("cam_out_") else mli)[name].values
    # a raw profile is (level, ncol); a store has (ncol, level)
    return {name: (values.T if values.ndim == 2 else values).astype(np.float32) for name, values in out.items()}


def dims(values: np.ndarray) -> tuple:
    return ("time", "ncol", "level") if values.ndim == 3 else ("time", "ncol")


def create(path: Path, steps: list[tuple], block: dict, grid: xr.Dataset, attrs: dict) -> None:
    """An empty store with all timesteps.  The blocks are written into it one after the other."""
    empty = {name: (dims(values), dask.array.zeros((len(steps), *values.shape[1:]), chunks=(32, *values.shape[1:]), dtype=np.float32))
             for name, values in block.items()}
    coords = {"time": xr.CFTimeIndex([timestamp(step) for step in steps]), "ncol": np.arange(grid.sizes["ncol"]),
              "level": np.arange(grid.sizes["lev"]), "lat": ("ncol", grid.lat.values.astype(np.float32)),
              "lon": ("ncol", grid.lon.values.astype(np.float32)), "area": ("ncol", grid.area.values)}
    xr.Dataset(empty, coords=coords, attrs=attrs).to_zarr(path, mode="w", zarr_format=3, compute=False)


def build(path: Path, steps: list[tuple], names: list[str], grid: xr.Dataset, session: requests.Session, pacer: Pacer) -> None:
    raw, progress = path.parent / "raw", path.with_suffix(".progress")
    attrs = {"source": "LEAP/ClimSim_low-res on Hugging Face, E3SM-MMF low resolution, real geography",
             "timesteps": f"{timestamp(steps[0])} to {timestamp(steps[-1])}, {len(steps)} timesteps",
             "tendencies": "ptend = (mlo.state - mli.state) / 1200 s"}
    if path.exists():
        store = xr.open_zarr(path)
        assert set(store.data_vars) >= set(names) and store.sizes["time"] == len(steps), \
            f"{path} has other variables or other timesteps. Delete {path} and {progress}, then run the script again."
        if "hyai" in store:
            print(f"{path} is complete.")
            return
    done = int(progress.read_text()) if progress.exists() and path.exists() else 0

    for start in range(done, len(steps), BLOCK):
        chunk = steps[start:start + BLOCK]
        download([raw_path(step, kind) for step in chunk for kind in ("mli", "mlo")], raw, session, pacer)
        with ThreadPoolExecutor(8) as pool:
            arrays = list(pool.map(lambda step: read_step(raw, step, names), chunk))
        block = {name: np.stack([a[name] for a in arrays]) for name in names}
        if start == 0:
            create(path, steps, block, grid, attrs)
        ds = xr.Dataset({name: (dims(values), values) for name, values in block.items()})
        ds.to_zarr(path, region={"time": slice(start, start + len(chunk))})
        progress.write_text(str(start + len(chunk)))
        shutil.rmtree(raw / "train")
        print(f"{path.name}: {start + len(chunk)}/{len(steps)} timesteps", flush=True)

    # The coefficients of the levels, for the mass of each level in score.py.  They go last: they mark a complete store.
    constants = grid[["hyai", "hybi", "P0"]]
    constants.attrs = attrs  # this write replaces the attributes of the store
    constants.to_zarr(path, mode="a")
    progress.unlink(missing_ok=True)
    store = xr.open_zarr(path)
    print(f"{path}: {store.sizes['time']} timesteps x {store.sizes['ncol']} columns = "
          f"{store.sizes['time'] * store.sizes['ncol']} rows, {len(names)} variables")


def run(full: bool = False, dry_run: bool = False, data: Path = DATA,
        train_stride: int = TRAIN_STRIDE, test_stride: int = TEST_STRIDE) -> None:
    """Write `data`/train.zarr and `data`/test.zarr.  The tutorial calls this function with larger strides."""
    names = list(dict.fromkeys(FULL_INPUTS + FULL_TARGETS)) if full else INPUTS + TARGETS
    stores = {"train": timesteps(TRAIN_MONTHS, train_stride), "test": timesteps(TEST_MONTHS, test_stride)}
    count = sum(len(steps) for steps in stores.values())
    print(f"{count} timesteps, {2 * count} files, a download of approximately {count * 3.03e-3:.1f} GB, {len(names)} variables")
    if dry_run:
        return

    session, pacer = requests.Session(), Pacer()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))
    fetch(session, pacer, data / "raw", GRID)
    shutil.copy(data / "raw" / GRID, data / "grid_info.nc")
    grid = xr.open_dataset(data / "grid_info.nc").load()
    for name, steps in stores.items():
        build(data / f"{name}.zarr", steps, names, grid, session, pacer)
    shutil.rmtree(data / "raw", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="all variables instead of the variables of the ClimSim paper")
    parser.add_argument("--dry-run", action="store_true", help="only count the files and the bytes")
    args = parser.parse_args()
    run(args.full, args.dry_run)


if __name__ == "__main__":
    main()
