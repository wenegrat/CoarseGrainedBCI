#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
from aux00_utils import load_dataset_and_grid, make_gaussian_filter
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
filter_length_scales = np.geomspace(0.1, 8, 5)
#---

#+++ Load data and grid
ds = load_dataset_and_grid(filename)
#---

#+++ Create circle DataArray (radius 2, centered at x=2, z=2)
x, z = ds.x_caa, ds.z_aac
ds["circle"] = xr.where((x - 3)**2 + (z - 2)**2 <= 4, 1.0, 0.0)
#---

#+++ Filter at each length scale
scale_coord = xr.DataArray(filter_length_scales, dims="filter_length_scale")
ds_filt = xr.concat(
    [make_gaussian_filter(ℓ, ds, filter_in_2d=True).apply(ds["circle"], dims=["x_caa", "z_aac"])
     for ℓ in filter_length_scales],
    dim=scale_coord,
).to_dataset(name="circle_filtered")
print(ds_filt)
#---

from matplotlib import pyplot as plt
ds_filt.circle_filtered.plot(col="filter_length_scale", x="x_caa")