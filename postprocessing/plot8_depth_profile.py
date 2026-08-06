#!/usr/bin/env python
"""Depth profile of the horizontally- and time-averaged cross-scale KE/APE flux and buoyancy conversion,
at two filter scales side by side."""

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
# Default pooling window (days): matches plot5_vorticity_strain_flux.py/plot7_flux_phase_space.py's own
# default -- late enough that eddies should be fully developed, fixed rather than tied to each run's own
# length so repeated runs/comparisons default to the same physical window.
DEFAULT_TIME_MIN_DAYS = 15.0
DEFAULT_TIME_MAX_DAYS = 30.0

import argparse
parser = argparse.ArgumentParser(description="Depth profile of horizontally- and time-averaged Πₖ, Π_A, "
                                              "and buoyancy conversion (SFS APE->KE exchange), two filter scales side by side")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs=2, default=None,
    help="Two filter length scales (meters) for left and right panels. Defaults to the first two scales "
         "actually present in the budget file (avoids a silent nearest-match to the wrong scale if the "
         "file was built with non-default filter scales) -- same convention as plot3_budgets.py.")
parser.add_argument("--time-min", type=float, default=None, help=f"Start of the time range (days, inclusive; defaults to {DEFAULT_TIME_MIN_DAYS:g} days -- eddies should be fully developed by then)")
parser.add_argument("--time-max", type=float, default=None, help=f"End of the time range (days, inclusive; defaults to {DEFAULT_TIME_MAX_DAYS:g} days)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
FIGURES = REPO_ROOT / "figures" / stem  # one subfolder per run, keyed by input filename stem
FIGURES.mkdir(parents=True, exist_ok=True)
#---

#+++ Orientation fix: some pipeline fields (Π_A, the KE<->APE exchange term) are stored with dims
# (..., x, y) instead of (..., y, x) like most fields -- always transpose to (..., y_dim, x_dim) first.
# See anim3_panels.py/plot5_vorticity_strain_flux.py/plot6_snapshots.py for the same pattern.
def fix_orientation(da):
    y_dim = next(d for d in da.dims if d.startswith("y"))
    x_dim = next(d for d in da.dims if d.startswith("x"))
    other_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
    return da.transpose(*other_dims, y_dim, x_dim)
#---

#+++ Load Πₖ/conversion (ke_fields) and Π_A (ape_fields); pick the two filter scales, dedup + pool over
# the requested time window
print("Loading energy budget fields...")
ke_fields  = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields.nc",  decode_times=False)
ape_fields = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc", decode_times=False)
print(f"  Filter scales available: {ke_fields.filter_scale.values}")

# baroclinic_adjustment.jl's :fields writer uses schedule=ConsecutiveIterations(TimeInterval(...)), which
# writes TWO consecutive model iterations (nominal output time, then the next iteration ~seconds-minutes
# later) at every nominal output time -- see plot5_vorticity_strain_flux.py's comment on this same
# structure. Keep only the first member of each pair, on the full time axis before any time-based
# selection, so the pair parity stays anchored to the simulation start.
ke_fields = ke_fields.isel(time=slice(0, None, 2))

if args.filter_scales is not None:
    filter_scales = args.filter_scales
    print(f"  Filter scales (explicit --filter-scales): {filter_scales}")
else:
    filter_scales = list(ke_fields.filter_scale.values[:2])
    print(f"  Filter scales (from budget file, --filter-scales not given): {filter_scales}")

time_min_days = args.time_min if args.time_min is not None else DEFAULT_TIME_MIN_DAYS
time_max_days = args.time_max if args.time_max is not None else DEFAULT_TIME_MAX_DAYS

# Duration checks against this run's own actual length (filter-scale-independent, so done once): <
# time_min_days is a hard error -- the requested/default window can't even start, and silently plotting
# from whatever scraps exist near the run's actual end would be more confusing than failing loudly. <
# time_max_days (but >= time_min_days) is survivable -- clip and warn, since a shorter-than-requested-but-
# nonempty pooling window is still meaningful.
available_max_days = float(ke_fields.time.max()) / 86400
if available_max_days < time_min_days:
    raise ValueError(f"Requested time range starts at {time_min_days:g} days, but {stem} only has "
                     f"{available_max_days:.2f} days of output -- pass --time-min/--time-max explicitly "
                     f"for a shorter run.")
if available_max_days < time_max_days:
    print(f"  Warning: requested time range extends to {time_max_days:g} days, but {stem} only has "
          f"{available_max_days:.2f} days of output -- clipping to [{time_min_days:g}, {available_max_days:.2f}] days.")

t_min_sec = time_min_days * 86400
t_max_sec = time_max_days * 86400  # clipped to whatever's available by .sel(slice(...)) below if too large
ke_t = ke_fields.sel(time=slice(t_min_sec, t_max_sec))
n_times = ke_t.sizes["time"]
if n_times == 0:
    raise ValueError(f"No available times in [{t_min_sec/86400:.2f}, {t_max_sec/86400:.2f}] days")
t_days_actual = ke_t.time.values / 86400
t_min_days, t_max_days = float(t_days_actual.min()), float(t_days_actual.max())
print(f"t = {t_min_days:.2f}-{t_max_days:.2f} days ({n_times} snapshots)")
#---

#+++ Horizontal-area-weighted mean (Δx_caa/Δy_aca live on the raw sim file, not the budget-fields files),
# then mean/std across the pooled time window at each depth -- each pooled snapshot weighted equally
# regardless of Δt, matching sweep3/plot5's own convention for multi-snapshot pooling. Computed once per
# filter scale below.
ds_grid = load_dataset_and_grid(filename)
dA = ds_grid.Δx_caa * ds_grid.Δy_aca

def horiz_mean(da):
    return (da * dA).sum(("x_caa", "y_aca")) / dA.sum(("x_caa", "y_aca"))  # -> (time, z_aac)

# Colors match the inherited (KH-era, unadapted) postprocessing/S3_plot_sweep.py's own Πₖ/Π_A/SFS-exchange
# palette, for visual consistency with that precedent.
COLORS = {"Πₖ": "#2166ac", "Π_A": "#d6604d", "buoyancy conversion": "#1b7837"}

panels = []
for ℓ_target in filter_scales:
    ℓ = float(ke_t.filter_scale.sel(filter_scale=ℓ_target, method="nearest"))
    ℓ_km = int(round(ℓ / 1000))

    Pi_K = fix_orientation(ke_t["Π_K"].sel(filter_scale=ℓ, method="nearest"))
    conv = fix_orientation(ke_t["SFS APE->KE exchange"].sel(filter_scale=ℓ, method="nearest"))
    # Matched to ke_t's own (already deduped/windowed) times independently -- the two files can have
    # slightly different time grids/roundoff, same reasoning as plot5_vorticity_strain_flux.py's own Π_A
    # selection.
    Pi_A = fix_orientation(ape_fields["Π_A"].sel(filter_scale=ℓ, time=ke_t.time, method="nearest"))

    Pi_K_hz, Pi_A_hz, conv_hz = horiz_mean(Pi_K).compute(), horiz_mean(Pi_A).compute(), horiz_mean(conv).compute()
    profiles = {
        "Πₖ":                  (Pi_K_hz.mean("time"), Pi_K_hz.std("time")),
        "Π_A":                 (Pi_A_hz.mean("time"), Pi_A_hz.std("time")),
        "buoyancy conversion": (conv_hz.mean("time"), conv_hz.std("time")),
    }
    panels.append((ℓ_km, profiles))
    print(f"  ℓ = {ℓ:.4f} m ({ℓ_km} km): profiles computed")
#---

#+++ Plot: two panels side by side (one per filter scale), sharing the depth (y) axis but each with its
# own independent value (x) axis -- Πₖ/Π_A/conversion can differ enough in magnitude between the two
# filter scales that a shared x-range would compress the smaller-scale panel's own variation, unlike the
# earlier single-panel version where all three lines shared one scale (comparable magnitudes there since
# it was the same filter scale).
print("Building figure...")
FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 17, 15, 14, 13, 13

fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True, sharey=True)

for ax, (ℓ_km, profiles) in zip(axes, panels):
    for name, (mean, std) in profiles.items():
        color = COLORS[name]
        ax.plot(mean, mean.z_aac, color=color, lw=1.8, label=name)
        ax.fill_betweenx(mean.z_aac, mean - std, mean + std, color=color, alpha=0.2)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("m² s⁻³", fontsize=FS_LABEL)
    ax.set_title(f"ℓ = {ℓ_km} km", fontsize=FS_TITLE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)

axes[0].set_ylabel("Depth  [m]", fontsize=FS_LABEL)
axes[0].legend(fontsize=FS_LEGEND, loc="best")

label = run_label(ke_fields.attrs)
suptitle = f"{stem}, t={t_min_days:.1f}-{t_max_days:.1f}d ({n_times} snapshots)"
if label:
    suptitle = f"{label}\n{suptitle}"
fig.suptitle(suptitle, fontsize=FS_SUPTITLE)

ℓ_km_tag = "-".join(str(ℓ_km) for ℓ_km, _ in panels)
outfile = FIGURES / f"{stem}_depth_profile_l{ℓ_km_tag}km_t{t_min_days:.0f}-{t_max_days:.0f}d.pdf"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
#---
