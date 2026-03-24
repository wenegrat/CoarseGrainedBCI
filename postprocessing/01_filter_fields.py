#!/usr/bin/env python
import os
# Disable HDF5 advisory file locking — required for parallel writes on Lustre (Derecho/GPFS)
# and when multiple dask worker processes open the same HDF5/NetCDF4 file concurrently.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
#+++ Imports
from pathlib import Path
import numpy as np
from dask.distributed import Client, LocalCluster, progress
from aux00_utils import load_dataset_and_grid, filter_fields
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for KE budget")
parser.add_argument("--filename", default="output/khi_128x1x256.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=6,
                    help="Number of dask workers in LocalCluster")
parser.add_argument("--threads-per-worker", type=int, default=3,
                    help="Threads per dask worker")
#---

if __name__ == "__main__":
    args = parser.parse_args()
    REPO_ROOT = Path(__file__).resolve().parent.parent
    filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
    filter_length_scales = np.geomspace(0.1, 2, 4) # Length scales for filtering

    #+++ Start dask cluster
    print("\n" + "="*60)
    print("Starting dask LocalCluster...")
    cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
    client = Client(cluster)
    print(f"  Workers: {args.n_workers}  threads/worker: {args.threads_per_worker}  "
          f"(total CPUs: {args.n_workers * args.threads_per_worker})")
    print(f"  Dashboard: {client.dashboard_link}")
    #---

    #+++ Load data and grid
    print("\n" + "="*60)
    print("Loading data and grid...")
    ds = load_dataset_and_grid(filename)
    # Chunk only along time: each time step is one independent task.
    # x/y must stay whole (filter operates along those dims).
    # z is a batch dimension that gcm_filters handles internally via numpy — splitting
    # it would multiply an already complex task graph without improving performance.
    ds = ds.chunk({"time": 1})
    print(f"Dataset loaded: {len(ds.time)} time steps")
    #---

    #+++ Filter velocity and buoyancy fields at each length scale
    filter_in_2d = ds.sizes["x_caa"] > 1 and ds.sizes["y_aca"] > 1
    print("\n" + "="*60)
    if filter_in_2d:
        print("Filtering velocity and buoyancy fields in 2D (x and y)...")
    else:
        print("Filtering velocity and buoyancy fields in 1D (x only)...")

    ds_filt = filter_fields(ds, filter_length_scales, filter_in_2d=filter_in_2d)
    print("Done building lazy task graph — all filter scales will be computed in parallel")
    #---

    #+++ Save filtered fields
    print("\n" + "="*60)
    print("Saving filtered fields...")

    output_filename = filename.replace(".nc", "_filtered_velocities.nc")
    write = ds_filt.to_netcdf(output_filename, compute=False)
    future = client.compute(write)
    progress(future)
    future.result()
    print(f"Filtered fields saved to: {output_filename}")
    #---

    #+++ Shutdown cluster
    client.close()
    cluster.close()
    #---
