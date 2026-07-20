#!/usr/bin/env python
"""Depth vs. filter-scale structure of the time-averaged cross-scale KE/APE transfer."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.aux00_utils import load_dataset_and_grid
from src.aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot depth vs. filter-scale structure of the time-averaged cross-scale KE/APE transfer")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
parser.add_argument("--min-time-days", type=float, default=5.0, help="Exclude time samples before this (in days) from the time average -- the first several saved samples are dominated by a large initial transient unrelated to the ongoing cascade (default 5.0)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
#---

#+++ Load energy transfer data -- Π_K/Π_A are already full local (filter_scale, time, z, y, x) fields in
# this file (computed once by sweep2_energy_transfer.py, mainly for the volume-integrated Hovmöller in
# sweep3), so this plot needs no new heavy computation -- just a horizontal+time average of data that
# already exists on disk. Chunked by time so the (area-weighted) reduction below streams through one
# timestep at a time rather than requiring the whole (potentially large, full-depth, all-scales) file
# resident in memory at once.
print("Loading energy transfer data...")
input_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
et = xr.open_dataset(input_filename, decode_timedelta=False).chunk({"time": 1})
print(f"  Loaded: {input_filename}")
print(f"  Time steps: {len(et.time)}   Filter scales: {len(et.filter_scale)}   Depths: {len(et.z_aac)}")

# Horizontal area weights for the area-weighted mean below -- Δx_caa/Δy_aca live on the raw sim file, not
# the energy-transfer output.
ds_grid = load_dataset_and_grid(filename)
dA = ds_grid.Δx_caa * ds_grid.Δy_aca
#---

#+++ Exclude initial transient, then horizontally- and time-average Π_K/Π_A at each (filter_scale, z)
et_avg = et.sel(time=slice(args.min_time_days * 86400, None))
n_times_used = len(et_avg.time)
print(f"  Excluding t < {args.min_time_days} days -- averaging over x, y, and the remaining {n_times_used}/{len(et.time)} time samples...")

def horiz_time_mean(da):
    horiz_mean = (da * dA).sum(("x_caa", "y_aca")) / dA.sum(("x_caa", "y_aca"))
    return horiz_mean.mean("time").compute()

Pi_K_zl = horiz_time_mean(et_avg["Π_K"])
Pi_A_zl = horiz_time_mean(et_avg["Π_A"])
print("Done!")
#---

#+++ Plot: depth vs. 1/ℓ, one panel each for Π_K and Π_A
inv_scale = 1.0 / et.filter_scale.values
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True, sharey=True)

vmax = float(max(np.nanpercentile(np.abs(Pi_K_zl), 99), np.nanpercentile(np.abs(Pi_A_zl), 99)))
for ax, da, title in zip(axes, [Pi_K_zl, Pi_A_zl], ["KE cross-scale transfer  Π_K(z, ℓ)", "APE cross-scale transfer  Π_A(z, ℓ)"]):
    im = ax.pcolormesh(inv_scale, da.z_aac, da.transpose("z_aac", "filter_scale").values,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    ax.set_xscale("log")
    ax.set_xlabel(r"$1/\ell$  [m$^{-1}$]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
axes[0].set_ylabel("Depth  [m]")

label = run_label(et.attrs)
suptitle = f"Time-mean over t > {args.min_time_days}d  ({n_times_used} samples)"
if label:
    suptitle = f"{label}\n{suptitle}"
fig.suptitle(suptitle, fontsize=11)

plot_filename = str(REPO_ROOT / "figures" / (Path(filename).stem + f"_depth_scale_transfer{ref_suffix}.png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---
