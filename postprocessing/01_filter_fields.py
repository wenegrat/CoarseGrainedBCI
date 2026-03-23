#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import gcm_filters
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
print("\n" + "="*60)
print("Filtering velocity and buoyancy fields...")

filtered_dimensions = ["x_caa", "y_aca"]
dx_min = float(min(ds.Δx_caa.min(), ds.Δy_aca.min()))

ds = condense_velocities(ds, indices=[1, 2, 3])

ds_filt_list = []
for ℓ in filter_length_scales:
    print(f"  filter_length_scale = {ℓ:.4f}...")
    gaussian_filter = gcm_filters.Filter(
        filter_scale=ℓ * np.sqrt(12),
        dx_min=dx_min,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    ds_ℓ = xr.Dataset({
        "ūᵢ": gaussian_filter.apply(ds["uᵢ"], dims=filtered_dimensions),
        "b̄":  gaussian_filter.apply(ds["b"],  dims=filtered_dimensions),
    })
    ds_filt_list.append(ds_ℓ)

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