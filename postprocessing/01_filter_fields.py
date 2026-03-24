#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import gcm_filters
from scipy.ndimage import gaussian_filter1d
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, condense_velocities
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for KE budget")
parser.add_argument("--filename", default="output/khi_128x1x256.nc",
                    help="Path to simulation NetCDF file")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
filter_length_scales = np.geomspace(0.1, 2, 4) # Length scales for filtering
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Filter velocity and buoyancy fields at each length scale
filter_in_2d = ds.dims["x_caa"] > 1 and ds.dims["y_aca"] > 1
print("\n" + "="*60)
if filter_in_2d:
    print("Filtering velocity and buoyancy fields in 2D (x and y)...")
else:
    print("Filtering velocity and buoyancy fields in 1D (x only)...")

dx_min = float(min(ds.Δx_caa.min(), ds.Δy_aca.min()))
dx_x   = float(ds.Δx_caa.min())

ds = condense_velocities(ds, indices=[1, 2, 3])

def apply_filter_1d(da, sigma_grid):
    """Apply a Gaussian filter along x only (periodic BC via mode='wrap').
    sigma_grid = ℓ / dx, consistent with gcm_filters' convention filter_scale = ℓ * sqrt(12)."""
    return xr.apply_ufunc(
        gaussian_filter1d, da,
        input_core_dims=[["x_caa"]],
        output_core_dims=[["x_caa"]],
        kwargs={"sigma": sigma_grid, "axis": -1, "mode": "wrap"},
        dask="parallelized",
        output_dtypes=[da.dtype],
    )

ds_filt_list = []
for ℓ in filter_length_scales:
    print(f"  filter_length_scale = {ℓ:.4f}...")

    if filter_in_2d:
        gaussian_filter = gcm_filters.Filter(
            filter_scale=ℓ * np.sqrt(12),
            dx_min=dx_min,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR,
        )
        ūᵢ = gaussian_filter.apply(ds["uᵢ"], dims=["x_caa", "y_aca"])
        b̄  = gaussian_filter.apply(ds["b"],  dims=["x_caa", "y_aca"])
    else:
        sigma_grid = ℓ / dx_x
        ūᵢ = apply_filter_1d(ds["uᵢ"], sigma_grid)
        b̄  = apply_filter_1d(ds["b"],  sigma_grid)

    ds_filt_list.append(xr.Dataset({"ūᵢ": ūᵢ, "b̄": b̄}))

scale_coord = xr.DataArray(filter_length_scales, dims="filter_length_scale",
                            name="filter_length_scale")
ds_filt = xr.concat(ds_filt_list, dim=scale_coord)
ds_filt["dV"] = ds["dV"]  # scale-independent, no filter_length_scale dimension
print("Done!")
#---

#+++ Save filtered fields
print("\n" + "="*60)
print("Saving filtered fields...")

output_filename = filename.replace(".nc", "_filtered_velocities.nc")
with ProgressBar():
    ds_filt.to_netcdf(output_filename)
print(f"Filtered fields saved to: {output_filename}")
#---
