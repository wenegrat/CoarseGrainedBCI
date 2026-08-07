#!/usr/bin/env python
#+++ Imports
import gc
import os
from pathlib import Path
import shutil
import time
import dask
import numpy as np
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
# Scale batching, mirroring sweep1_filter_fields.py's own flags: lets the per-scale loop be split across
# several concurrently-running PBS jobs (see submit_sweep.sh's N_SCALE_JOBS), each computing a disjoint
# index range and checkpointing its own scales, followed by one --merge-only job that assembles them.
# Unlike sweep1, batches here can't disagree about which scales exist: every invocation reads the same
# already-merged ds_filt, so the scale list is a shared upstream artifact rather than one each job
# recomputes independently.
parser.add_argument("--scale-start-idx", type=int, default=None,
    help="First scale index (0-based, into ds_filt's own filter_scale list) this invocation computes, "
         "inclusive. Default: 0. Mutually exclusive with --merge-only.")
parser.add_argument("--scale-end-idx", type=int, default=None,
    help="Last scale index this invocation computes, EXCLUSIVE (half-open [start, end)). Default: all "
         "scales in ds_filt. Mutually exclusive with --merge-only.")
parser.add_argument("--merge-only", action="store_true", default=False,
    help="Skip computing any scale; require an up-to-date checkpoint to already exist for every one of "
         "ds_filt's filter_scale values, then concat + write the final merged output and delete the "
         "checkpoints. Hard-errors naming any scale whose checkpoint is missing or stale. Mutually "
         "exclusive with --scale-start-idx/--scale-end-idx.")
args = parser.parse_args()

if args.merge_only and (args.scale_start_idx is not None or args.scale_end_idx is not None):
    parser.error("--merge-only cannot be combined with --scale-start-idx/--scale-end-idx "
                 "(merge-only always covers every scale in ds_filt)")

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

filter_scales_full = ds_filt.filter_scale.values
n_scales_full = len(filter_scales_full)
print(f"  Loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_scales_full}")
print(f"  Filter dimensions: x and y (horizontal)")

# Resolve this invocation's slice of the scale list. Defaults cover everything, so an invocation passing
# neither flag behaves exactly as it always has. NOTE: filter_scales_full comes straight from ds_filt and
# is ascending (sweep1's combine="by_coords" merge guarantees it); the final xr.concat below relies on
# that ordering, since concat does NOT sort. Don't reorder it.
scale_start_idx = args.scale_start_idx if args.scale_start_idx is not None else 0
scale_end_idx   = args.scale_end_idx   if args.scale_end_idx   is not None else n_scales_full
if not (0 <= scale_start_idx < scale_end_idx <= n_scales_full):
    parser.error(f"--scale-start-idx/--scale-end-idx must satisfy 0 <= start < end <= n_scales "
                 f"(got start={scale_start_idx}, end={scale_end_idx}, n_scales={n_scales_full})")
is_full_range = (scale_start_idx == 0 and scale_end_idx == n_scales_full)
if not is_full_range:
    print(f"  This invocation computes scale indices [{scale_start_idx}, {scale_end_idx}) only")
#---

#+++ Load sorted density (always -- not just for --fixed-reference)
# The full-field density sort is scale-independent, but calculate_energy_transfer() only skips its own
# internal sorted_timeseries() call when BOTH rho_sorted and dz_sorted are passed in. Since this script
# calls it once per scale, leaving these as None (which is what the old --fixed-reference-only gate did
# on the default path) redid that expensive sort from scratch on every single scale. 03_energy_transfer.py
# already avoids this by loading 02_sort_density.py's output unconditionally; do the same here. Hard error
# rather than falling back to the old behavior: silently re-sorting n_scales times is the bug being fixed,
# and it's also what would make each parallel batch job redundantly redo the same sort.
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
if not os.path.exists(sorted_density_filename):
    raise FileNotFoundError(
        f"Required sorted-density file not found: {sorted_density_filename}\n"
        f"Run 02_sort_density.py --filename {args.filename}"
        f"{' --fixed-reference' if fixed_reference else ''} first "
        f"(sweep_sort.pbs does this on the HPC, and submit_sweep.sh submits it automatically when the "
        f"file is missing).")
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
n_scales = n_scales_full
filter_scales = filter_scales_full
for i, ℓ in enumerate(filter_scales):
    checkpoint_path = PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep_checkpoint_l{ℓ:.4f}{ref_suffix}.nc")
    checkpoint_files.append(checkpoint_path)

    # Batch mode: only compute the scales assigned to this invocation. Scales outside the range are
    # skipped entirely here (their checkpoints belong to a sibling job); the --merge-only pass is what
    # later requires all of them to be present. Checkpoint filenames are keyed by the ℓ value itself, so
    # concurrent batches never collide the way sweep1's index-named tmp files could.
    in_this_batch = scale_start_idx <= i < scale_end_idx
    if not (in_this_batch or args.merge_only):
        continue

    if checkpoint_path.exists():
        cached = xr.open_dataset(str(checkpoint_path), decode_times=False).chunk(chunks)
        # A checkpoint is keyed only by filter scale, not by which timesteps went into it -- if a *different*
        # run (a different --n-time-skip, an input that's since changed, or a pre-dedup-fix run) left a
        # checkpoint behind for this same scale without ever reaching the cleanup at the end of a successful
        # run, this run would otherwise trust it by filename alone. Confirmed as a real production issue:
        # mixing a stale full-resolution checkpoint with freshly-computed reduced-resolution ones produced an
        # xr.concat time-axis size mismatch (join='outer' silently unioned them) that inflated the final
        # output's time dimension well past what was actually requested, with no error, just a FutureWarning.
        # Validate the time axis actually matches this run's expected one (ds_filt.time) before trusting a
        # cached checkpoint; recompute (and overwrite) it otherwise.
        stale = cached.sizes["time"] != ds_filt.sizes["time"] or not np.array_equal(cached.time.values, ds_filt.time.values)
        if not stale:
            print(f"\n--- filter {i+1}/{n_scales}: scale = {ℓ:.4f} (loading from checkpoint) ---")
            transfer_list.append(cached)
            continue
        if args.merge_only:
            raise RuntimeError(
                f"--merge-only: checkpoint for filter_scale={ℓ:.4f} is stale (has "
                f"{cached.sizes['time']} timesteps, this run expects {ds_filt.sizes['time']}) -- a batch "
                f"covering this scale needs to (re)run before merging. Checkpoint: {checkpoint_path}")
        print(f"\n--- filter {i+1}/{n_scales}: scale = {ℓ:.4f}: checkpoint has {cached.sizes['time']} timesteps, "
              f"this run expects {ds_filt.sizes['time']} -- stale checkpoint from a different run, recomputing ---")
        cached.close()
        # Delete rather than just closing and letting write_dataset() overwrite it in place -- closing a
        # just-read netCDF4 handle doesn't reliably evict it from xarray's global file-handle cache before
        # the same path is immediately reopened for writing (confirmed directly: reusing the handle this way
        # raised a PermissionError from a stale cache entry). Removing the file first guarantees a clean open.
        checkpoint_path.unlink()
    elif args.merge_only:
        raise RuntimeError(
            f"--merge-only: no checkpoint found for filter_scale={ℓ:.4f} (expected {checkpoint_path}) -- "
            f"a batch covering this scale hasn't completed yet.")

    # Printed here (not inside calculate_energy_transfer(), which only knows the single-element list this
    # scale's call gets) so progress through the whole sweep is visible -- otherwise the "--- filter_scale =
    # ... ---" line the shared function itself prints right after gives no sense of how many scales remain,
    # which matters for a 30-scale sweep where each one can take minutes.
    print(f"\n=== Filter {i+1}/{n_scales} ===")
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

if not is_full_range and not args.merge_only:
    print(f"\nComputed scale indices [{scale_start_idx}, {scale_end_idx}) of [0, {n_scales}) -- a partial "
          f"batch, so not merging. Rerun with --merge-only once every batch covering [0, {n_scales}) has "
          f"finished.")
    raise SystemExit(0)

energy_transfer = xr.concat(transfer_list, dim="filter_scale")
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")
energy_transfer.attrs.update(ds.attrs)
output_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
tmp_dir = PP_OUTPUT / (Path(output_filename).stem + "_tmp")
# Same class of issue as the stale-checkpoint fix above: this run's own cleanup at the bottom only knows
# about the files *it* wrote (tmp_files), so any leftovers from an earlier run against this same output
# filename that didn't reach that cleanup (crashed, killed, walltime limit) -- especially a run with a
# different number of timesteps -- silently persist here. Confirmed as a real production issue: tmp_dir.rmdir()
# failed with "Directory not empty" after a successful write, because stale t{i:04d}.nc files from a prior,
# larger run were still sitting alongside this run's own (fewer) files. Clearing the directory unconditionally
# before this run writes anything guarantees a clean slate regardless of what an earlier run left behind.
shutil.rmtree(tmp_dir, ignore_errors=True)
tmp_dir.mkdir()
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
