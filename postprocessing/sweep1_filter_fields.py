#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import dask
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid, condense_velocities, GaussianFilter
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for cross-scale energy transfer sweep")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-time-skip", type=int, default=1, help="Keep every n-th (consecutive) time step")
parser.add_argument("--scale-min", type=float, default=None, help="Smallest filter scale (FWHM, meters). Defaults to 2x the grid spacing, the smallest scale the horizontal Gaussian filter can meaningfully resolve.")
parser.add_argument("--scale-max", type=float, default=None, help="Largest filter scale (FWHM, meters). Defaults to 40%% of the domain width Lx, staying safely below the periodic half-domain.")
parser.add_argument("--n-scales", type=int, default=30, help="Number of log-spaced filter scales between --scale-min and --scale-max (default 30)")
# Scale batching: lets the n_scales-long filtering loop be split across several concurrently-running PBS
# jobs (see submit_sweep.sh's N_SCALE_JOBS), each computing a disjoint index range into the SAME
# deterministic geomspace array and writing its own per-scale tmp files, followed by one --merge-only job
# that combines them. Every batch job must be given identical --filename/--scale-min/--scale-max/
# --n-scales/--n-time-skip so they all derive the same filter_scales array before slicing it -- the
# --merge-only validation below is the safety net that catches it if they weren't.
parser.add_argument("--scale-start-idx", type=int, default=None,
    help="First GLOBAL scale index (0-based, into the full --n-scales-length array) this invocation "
         "computes, inclusive. Default: 0. Mutually exclusive with --merge-only.")
parser.add_argument("--scale-end-idx", type=int, default=None,
    help="Last GLOBAL scale index this invocation computes, EXCLUSIVE (half-open [start, end)). "
         "Default: --n-scales. Mutually exclusive with --merge-only.")
parser.add_argument("--merge-only", action="store_true", default=False,
    help="Skip filtering entirely; merge the per-scale tmp files already on disk for the FULL "
         "[0, n_scales) range into the final output. Validates every expected file's filter_scale value "
         "and time axis, hard-erroring with the specific missing/stale indices rather than silently "
         "merging an incomplete set. Mutually exclusive with --scale-start-idx/--scale-end-idx.")
args = parser.parse_args()

if args.merge_only and (args.scale_start_idx is not None or args.scale_end_idx is not None):
    parser.error("--merge-only cannot be combined with --scale-start-idx/--scale-end-idx "
                 "(merge-only always covers the full [0, n_scales) range)")

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

# baroclinic_adjustment.jl's :fields writer uses schedule=ConsecutiveIterations(TimeInterval(...)), writing
# TWO consecutive model iterations (nominal output time, then the next iteration ~seconds-minutes later) at
# every nominal output time -- see plot5_vorticity_strain_flux.py's comment on this exact structure for why
# (04_sfs_ke_budget.py's tendency finite-difference needs a close pair). This sweep has no tendency term to
# compute, so it never needed the pairing, but never dropped it either -- every downstream filter/transfer
# step ran twice per real output time for no additional information (confirmed directly: a 40-day, 12h-
# output run processed 181 time steps here, not the ~81 a reader would expect). Keep only the first member
# of each pair -- same ::2 pattern already used for this exact structure elsewhere -- before the existing
# --n-time-skip logic, which can now be a plain slice instead of the i//2 trick previously needed to skip
# whole real output times while still keeping both members of whichever pairs it kept.
ds = ds.isel(time=slice(0, None, 2))
n_time_skip = args.n_time_skip
ds = ds.isel(time=slice(0, None, n_time_skip))
print(f"Dataset loaded: {len(ds.time)} time steps")

# Filter scales (FWHM, meters): log-spaced between a data-driven min/max unless overridden on the CLI, so
# the sweep's range adapts automatically to whichever resolution/domain the dataset actually has.
scale_min = args.scale_min if args.scale_min is not None else 2 * float(max(ds.Δx_caa.min(), ds.Δy_aca.min()))
scale_max = args.scale_max if args.scale_max is not None else 0.4 * np.asarray(ds.attrs["Lx"]).item()
filter_scales = np.geomspace(scale_min, scale_max, args.n_scales)
n_scales = args.n_scales
print(f"Filter scales: {scale_min/1e3:.1f}km to {scale_max/1e3:.1f}km ({n_scales} log-spaced steps)")

# Resolve this invocation's slice of the full scale list. Defaults cover everything, so an invocation
# that passes neither flag behaves exactly as it always has.
scale_start_idx = args.scale_start_idx if args.scale_start_idx is not None else 0
scale_end_idx   = args.scale_end_idx   if args.scale_end_idx   is not None else n_scales
if not (0 <= scale_start_idx < scale_end_idx <= n_scales):
    parser.error(f"--scale-start-idx/--scale-end-idx must satisfy 0 <= start < end <= n_scales "
                 f"(got start={scale_start_idx}, end={scale_end_idx}, n_scales={n_scales})")
is_full_range = (scale_start_idx == 0 and scale_end_idx == n_scales)
if not is_full_range:
    print(f"  This invocation computes global scale indices [{scale_start_idx}, {scale_end_idx}) only")
#---

#+++ Filter and save velocity/buoyancy fields, one filter scale at a time to bound memory
# filter_fields() (used by 01_filter_fields.py) computes every filter scale and concatenates them into one
# Dataset before returning -- fine for 01's 2 default scales, but this sweep's default of 30 scales means
# a single .load() of the concatenated result needs ~30x the memory of any one scale, all resident at once
# for the write (a 384x384x64 run OOM'd a 256GB node this way, dying at ~30% progress as dask accumulated
# more completed scales in memory). Instead, filter and write each scale to its own small tmp file, freeing
# memory before the next scale, then merge via dask's streaming to_netcdf (the same pattern already used
# for sweep2_energy_transfer.py's per-timestep writes) -- bounds peak memory to ~1 scale's worth (all
# times, one scale) instead of all n_scales.
print("\n" + "="*60)

ds = condense_velocities(ds, indices=(1, 2, 3))
output_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities_sweep.nc"))
tmp_dir = PP_OUTPUT / (Path(output_filename).stem + "_tmp")
tmp_dir.mkdir(exist_ok=True)
tmp_files = []

# Tmp files are named by GLOBAL scale index (position in the full n_scales-length array), not by the
# loop's own counter -- concurrent batch jobs each computing a different slice would otherwise all write
# scale000.nc, scale001.nc, ... and clobber each other. For the default full-range case scale_start_idx=0,
# so this produces exactly the same filenames as before.
if not args.merge_only:
    print("Filtering and saving velocity/buoyancy fields, one scale at a time...")

    # dx_min/dy_min don't depend on ℓ -- compute once outside the loop (and outside the ProgressBar scope
    # below) rather than via make_gaussian_filter(ℓ, ds) on every iteration. That call does two eager
    # float(...) scalar computes on grid spacing; triggering them inside the per-scale ProgressBar block was
    # measured to cost a flat ~5s (the dt=5 polling floor) *each*, purely for a scalar that's identical across
    # every scale -- 2 x n_scales wasted ~5s floors for no reason.
    dx_min = float(ds.Δx_caa.min())
    dy_min = float(ds.Δy_aca.min())

    with ProgressBar(minimum=5, dt=5):
        for local_idx, ℓ in enumerate(filter_scales[scale_start_idx:scale_end_idx]):
            global_idx = scale_start_idx + local_idx
            print(f"  filter_scale = {ℓ:.4f}  (global {global_idx+1}/{n_scales})...")
            gf = GaussianFilter(ℓ, dx_min, dy_min)
            ds_filt_ℓ = xr.Dataset({
                "ūᵢ": gf.apply(ds["uᵢ"], dims=["x_caa", "y_aca"]),
                "b̄":  gf.apply(ds["b"],  dims=["x_caa", "y_aca"]),
            }).expand_dims(filter_scale=[ℓ]).load()
            tmp_f = str(tmp_dir / f"scale{global_idx:03d}.nc")
            ds_filt_ℓ.to_netcdf(tmp_f)
            tmp_files.append(tmp_f)
            del ds_filt_ℓ
    print("Done!")

    if not is_full_range:
        print(f"\nComputed global scale indices [{scale_start_idx}, {scale_end_idx}) of [0, {n_scales}) -- "
              f"a partial batch, so not merging. Rerun with --merge-only once every batch covering "
              f"[0, {n_scales}) has finished.")
        raise SystemExit(0)
else:
    # --merge-only: collect the full range from disk instead of trusting whatever happens to be present.
    # Validate both the filter_scale value and the time axis of each file (same staleness-check idea
    # sweep2_energy_transfer.py already applies to its own per-scale checkpoints) so a batch that ran with
    # mismatched --scale-min/--scale-max/--n-scales/--n-time-skip is caught here rather than silently
    # merged into a wrong result.
    print(f"--merge-only: collecting per-scale tmp files for the full [0, {n_scales}) range...")
    problems = []
    for global_idx in range(n_scales):
        expected_ℓ = filter_scales[global_idx]
        f = tmp_dir / f"scale{global_idx:03d}.nc"
        if not f.exists():
            problems.append(f"  index {global_idx}: expected ℓ={expected_ℓ:.4f} -- MISSING ({f})")
            continue
        with xr.open_dataset(str(f), decode_times=False) as check_ds:
            found_ℓ = float(check_ds.filter_scale.values[0])
            ℓ_ok = np.isclose(found_ℓ, expected_ℓ, rtol=1e-6)
            time_ok = (check_ds.sizes["time"] == ds.sizes["time"]
                       and np.array_equal(check_ds.time.values, ds.time.values))
        if not (ℓ_ok and time_ok):
            problems.append(f"  index {global_idx}: expected ℓ={expected_ℓ:.4f} -- STALE "
                            f"(found ℓ={found_ℓ:.4f}, ℓ_ok={ℓ_ok}, time_ok={time_ok}) ({f})")
            continue
        tmp_files.append(str(f))
    if problems:
        raise RuntimeError(
            f"--merge-only: {len(problems)} of {n_scales} per-scale tmp files are missing or stale in "
            f"{tmp_dir} -- some batch job(s) haven't finished, or ran with different "
            f"--scale-min/--scale-max/--n-scales/--n-time-skip than this invocation:\n" + "\n".join(problems))
    print(f"  All {n_scales} per-scale tmp files present and validated")
#---

#+++ Merge per-scale files into the final filtered-fields output
print("\n" + "="*60)
print("Merging per-scale files...")
with xr.open_mfdataset(tmp_files, combine="by_coords", decode_timedelta=False,
                       parallel=False, chunks={"filter_scale": 1}) as merged:
    merged["dV"] = ds["dV"]
    merged.attrs.update(ds.attrs)
    merged.attrs["filter_dims"] = "x_caa,y_aca"
    write_job = merged.to_netcdf(output_filename, compute=False)
    # Forced single-threaded: computing write_job under dask's default *threaded* scheduler means multiple
    # threads write into the same HDF5 file handle -- the same hang write_dataset() (aux00_utils.py) exists
    # to avoid for 01/03/04/05, confirmed as a real 3+ hour stall here (frozen at 0% progress, not just slow).
    # No "load"-style alternative offered, unlike write_dataset(): the whole point of this per-scale temp-file
    # design is to never need the merged dataset fully in memory at once (see the comment above this loop).
    with dask.config.set(scheduler="synchronous"):
        with ProgressBar(minimum=5, dt=5):
            write_job.compute()
for f in tmp_files:
    os.remove(f)
tmp_dir.rmdir()
os.sync()
print(f"Filtered fields saved to: {output_filename}")
#---
