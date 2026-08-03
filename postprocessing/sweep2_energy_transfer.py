#!/usr/bin/env python
#+++ Imports
import gc
import os
from pathlib import Path
import time
import dask
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid, write_dataset
from src.aux02_ke_functions import calculate_energy_transfer
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale KE and APE transfer terms")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--write-mode", choices=["load", "synchronous"], default="load",
    help="How to avoid the dask-lazy .to_netcdf() write hang for each filter scale's checkpoint -- see "
         "write_dataset() in aux00_utils.py for what each mode does and the measured cost of 'synchronous' "
         "relative to 'load'.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
n_workers = args.n_workers
chunks = dict(time=1)
ref_suffix = "_fixed_ref" if fixed_reference else ""
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
ds = load_dataset_and_grid(filename)
ds = ds.chunk(chunks)
print(f"Dataset loaded: {len(ds.time)} time steps  ({time.time()-t0:.1f}s)")
#---

#+++ Load pre-filtered fields
print("\n" + "="*60)
print("Loading pre-filtered fields...")
t0 = time.time()
filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities_sweep.nc"))
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk(dict(time=1, filter_scale=1))
ds = ds.reindex(time=ds_filt.time).chunk(chunks)

filter_scales = ds_filt.filter_scale.values
print(f"  Loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_scales}")
print(f"  Filter dimensions: x and y (horizontal)")
#---

#+++ Load sorted density (only when using fixed reference)
rho_sorted = dz_sorted = None
if fixed_reference:
    sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
    ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk(chunks)
    ds_sorted = ds_sorted.reindex(time=ds_filt.time)
    rho_sorted = ds_sorted.rho_sorted
    dz_sorted  = ds_sorted.dz_sorted
    print(f"  Sorted density loaded from: {sorted_density_filename}")
#---

#+++ Calculate cross-scale transfer terms (checkpointed per filter scale)
print("\n" + "="*60)
print("Calculating cross-scale transfer terms...")
# Looped one scale at a time and checkpointed to disk -- same pattern as 03_energy_transfer.py -- so a
# scale's full lazy graph (and everything upstream of it: ds_filt_ℓ, w_full, b_r, ...) doesn't stay
# reachable for the rest of the sweep. calculate_energy_transfer() itself is unchanged (still called once
# per scale, just with a single-element filter_scales list) -- it's also used by 03_energy_transfer.py, so
# its own internals are left alone here rather than adding checkpointing inside the shared function.
#
# Confirmed as the cause of a real OOM on a 512x512x64, 40-day run (died mid-loop, at "Calculating
# cross-scale APE flux" -- calculate_energy_transfer() already .load()s Π_A and frees filt_local_pes per
# scale (an earlier fix), but ape_to_ke_exchange/w̄·b̄ᵣ and their integrals stay lazy, so every previous
# scale's not-yet-computed result -- and everything upstream of it -- stayed reachable via the growing
# Python list calculate_energy_transfer() itself builds internally before its own single xr.concat at the
# end). write_dataset() below forces all of that scale's remaining lazy pieces to compute and land on disk
# before moving to the next scale, same as it already does for calculate_energy_transfer()'s eager Π_A.
transfer_list = []
checkpoint_files = []
for ℓ in filter_scales:
    checkpoint_path = PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep_checkpoint_l{ℓ:.4f}{ref_suffix}.nc")
    checkpoint_files.append(checkpoint_path)

    if checkpoint_path.exists():
        print(f"\n--- filter_scale = {ℓ:.4f} (loading from checkpoint) ---")
        transfer_list.append(xr.open_dataset(str(checkpoint_path), decode_times=False).chunk(chunks))
        continue

    transfer_ℓ = calculate_energy_transfer(ds, [ℓ],
                                           ds_filt=ds_filt,
                                           rho_sorted=rho_sorted,
                                           dz_sorted=dz_sorted,
                                           n_workers=n_workers)
    print(f"  Computing and saving checkpoint (write-mode={args.write_mode})...")
    t0 = time.time()
    write_dataset(transfer_ℓ, str(checkpoint_path), write_mode=args.write_mode)
    print(f"  Checkpoint computed and saved  ({time.time()-t0:.1f}s)")
    del transfer_ℓ
    gc.collect()
    transfer_list.append(xr.open_dataset(str(checkpoint_path), decode_times=False).chunk(chunks))

energy_transfer = xr.concat(transfer_list, dim="filter_scale")
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")
energy_transfer.attrs.update(ds.attrs)
output_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
tmp_dir = PP_OUTPUT / (Path(output_filename).stem + "_tmp")
tmp_dir.mkdir(exist_ok=True)
tmp_files = []
# energy_transfer is dask-lazy here too -- each piece re-read from its own on-disk checkpoint above
# (open_dataset(...).chunk() is always lazy regardless of the checkpoint's own data having been eager on
# disk, same as 03_energy_transfer.py's identical re-read). Writing a lazy slice via .to_netcdf() computes it
# via dask's threaded scheduler *during* the write, with multiple threads writing into the same HDF5 file
# handle -- the same hang risk fixed for local_potential_energies_timeseries() in aux01_pe_functions.py.
# Loading one timestep at a time (not the whole energy_transfer up front -- that's the "hundreds of GB" case
# the comment below is about) keeps this bounded and makes each per-timestep write purely synchronous.
with ProgressBar(minimum=5, dt=5):
    for i in range(energy_transfer.sizes["time"]):
        tmp_f = str(tmp_dir / f"t{i:04d}.nc")
        energy_transfer.isel(time=[i]).load().to_netcdf(tmp_f)
        tmp_files.append(tmp_f)
        print(f"  wrote time {i+1}/{energy_transfer.sizes['time']}")

print("Merging per-timestep files...")
# Stream via dask (no .load(): the merged dataset is hundreds of GB and won't fit in RAM).
with xr.open_mfdataset(tmp_files, combine="by_coords", decode_timedelta=False,
                       parallel=False, chunks={"time": 1}) as merged:
    write_job = merged.to_netcdf(output_filename, compute=False)
    # Forced single-threaded: computing write_job under dask's default *threaded* scheduler means multiple
    # threads write into the same HDF5 file handle -- the same hang write_dataset() (aux00_utils.py) exists
    # to avoid for 01/03/04/05, confirmed as a real 3+ hour stall in sweep1's identical merge pattern (frozen
    # at 0% progress, not just slow). No "load"-style alternative offered, unlike write_dataset(): the whole
    # point of this per-timestep temp-file design is to never need the merged dataset fully in memory at once.
    with dask.config.set(scheduler="synchronous"):
        with ProgressBar(minimum=5, dt=5):
            write_job.compute()
for f in tmp_files:
    os.remove(f)
tmp_dir.rmdir()
print(f"Results saved to: {output_filename}")

print("\nDeleting intermediate checkpoint files...")
for f in checkpoint_files:
    f.unlink(missing_ok=True)
    print(f"  Deleted: {f.name}")
#---
