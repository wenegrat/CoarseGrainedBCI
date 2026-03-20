#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import gcm_filters
from aux00_utils import load_dataset_and_grid, condense_velocities, DaskParallelFilter
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for KE budget")
parser.add_argument("--filename", default="output/khi_128x1x256.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18,
                    help="Number of CPU workers for parallel filtering")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
n_workers = args.n_workers
filter_length_scale = 0.8  # Length scale for filtering
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Filter velocity and buoyancy fields
print("\n" + "="*60)
print("Filtering velocity and buoyancy fields...")

filtered_dimensions = ["x_caa", "y_aca"]
filter_scale = filter_length_scale * np.sqrt(12)
gaussian_filter = DaskParallelFilter(gcm_filters.Filter(
    filter_scale=filter_scale,
    dx_min=float(min(ds.Δx_caa.min(), ds.Δy_aca.min())),
    filter_shape=gcm_filters.FilterShape.GAUSSIAN,
    grid_type=gcm_filters.GridType.REGULAR,
), n_workers=n_workers)

ds = condense_velocities(ds, indices=[1, 2, 3])
ds["ūᵢ"] = gaussian_filter.apply(ds["uᵢ"], dims=filtered_dimensions)
ds["b̄"] = gaussian_filter.apply(ds["b"], dims=filtered_dimensions)

print(f"Fields filtered with length scale: {filter_length_scale}")
#---

#+++ Save filtered fields
print("\n" + "="*60)
print("Saving filtered fields...")

ds_filt = ds[["b̄", "dV", "ūᵢ"]].copy()

output_filename = filename.replace(".nc", "_filtered_velocities.nc")
ds_filt.to_netcdf(output_filename)
print(f"Filtered fields saved to: {output_filename}")
#---