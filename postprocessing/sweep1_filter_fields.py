#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid, filter_fields
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for cross-scale energy transfer sweep")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-time-skip", type=int, default=1, help="Keep every n-th (consecutive) time step")
parser.add_argument("--scale-min", type=float, default=None, help="Smallest filter scale (FWHM, meters). Defaults to 2x the grid spacing, the smallest scale the horizontal Gaussian filter can meaningfully resolve.")
parser.add_argument("--scale-max", type=float, default=None, help="Largest filter scale (FWHM, meters). Defaults to 40%% of the domain width Lx, staying safely below the periodic half-domain.")
parser.add_argument("--n-scales", type=int, default=30, help="Number of log-spaced filter scales between --scale-min and --scale-max (default 30)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk(dict(time=1))

i = np.arange(ds.sizes["time"])
n_time_skip = args.n_time_skip
ds = ds.isel(time=(i // 2) % n_time_skip == 0)
print(f"Dataset loaded: {len(ds.time)} time steps")

# Filter scales (FWHM, meters): log-spaced between a data-driven min/max unless overridden on the CLI, so
# the sweep's range adapts automatically to whichever resolution/domain the dataset actually has.
scale_min = args.scale_min if args.scale_min is not None else 2 * float(max(ds.Δx_caa.min(), ds.Δy_aca.min()))
scale_max = args.scale_max if args.scale_max is not None else 0.4 * np.asarray(ds.attrs["Lx"]).item()
filter_scales = np.geomspace(scale_min, scale_max, args.n_scales)
print(f"Filter scales: {scale_min/1e3:.1f}km to {scale_max/1e3:.1f}km ({args.n_scales} log-spaced steps)")
#---

#+++ Filter velocity and buoyancy fields at each length scale
print("\n" + "="*60)
print("Filtering velocity and buoyancy fields in x and y...")

ds_filt = filter_fields(ds, filter_scales)
print("Done!")
#---

#+++ Save filtered fields
print("\n" + "="*60)
print("Saving filtered fields...")

output_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities_sweep.nc"))
# Force full computation before writing -- ds_filt is still fully dask-lazy here (GaussianFilter.apply uses
# xr.apply_ufunc(dask="parallelized"), which stays lazy on this chunked input). Writing a lazy Dataset via
# .to_netcdf() computes it via dask's threaded scheduler *during* the write, with multiple threads writing
# into the same HDF5 file handle -- a known hang risk, since the underlying HDF5 C library isn't reliably
# thread-safe for it (see local_potential_energies_timeseries() in aux01_pe_functions.py for a real observed
# stall of this kind). Loading here first makes the write purely synchronous.
print("  Computing filtered fields (forces the dask graph before the write)...")
with ProgressBar(minimum=5, dt=5):
    ds_filt = ds_filt.load()
ds_filt.to_netcdf(output_filename)
os.sync()
print(f"Filtered fields saved to: {output_filename}")
#---
