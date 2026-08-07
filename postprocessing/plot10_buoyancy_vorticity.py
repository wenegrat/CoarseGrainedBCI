#!/usr/bin/env python
"""Buoyancy at mid-depth (initial vs. final time) alongside mid-depth Rossby number ζ/f (final time)."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
#---

#+++ Configuration
# Defaults match this codebase's own simulation defaults (baroclinic_adjustment.jl's --stop_time is 20
# days by default), not an arbitrary choice.
DEFAULT_T_INITIAL_DAYS = 0.0
DEFAULT_T_FINAL_DAYS = 20.0

import argparse
parser = argparse.ArgumentParser(description="1x3 panel: mid-depth buoyancy at t-initial, mid-depth buoyancy at t-final, mid-depth Rossby number ζ/f at t-final")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--z-mid", type=float, default=-500.0, help="Target mid-depth in meters (nearest available cell center; default -500)")
parser.add_argument("--t-initial", type=float, default=DEFAULT_T_INITIAL_DAYS, help=f"Initial time in days (nearest available; default {DEFAULT_T_INITIAL_DAYS:g})")
parser.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL_DAYS, help=f"Final time in days (nearest available; default {DEFAULT_T_FINAL_DAYS:g})")
parser.add_argument("--clim-percentile", type=float, default=99.5, help="Percentile of |data| used to set symmetric color limits")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)

REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
FIGURES = REPO_ROOT / "figures" / stem  # one subfolder per run, keyed by input filename stem
FIGURES.mkdir(parents=True, exist_ok=True)
#---

#+++ Orientation fix: some pipeline fields are stored with dims (..., x, y) instead of (..., y, x) --
# always transpose to (..., y_dim, x_dim) first. Applied defensively here too (a no-op if a field is
# already correctly ordered), matching plot5/plot6/plot8/plot9/anim3_panels.py's own convention.
def fix_orientation(da):
    y_dim = next(d for d in da.dims if d.startswith("y"))
    x_dim = next(d for d in da.dims if d.startswith("x"))
    other_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
    return da.transpose(*other_dims, y_dim, x_dim)

def coriolis_f(attrs):
    """Coriolis parameter from this run's own attrs (for the ζ/f Rossby-number normalization)."""
    Omega_earth = 7.2921159e-5
    return 2 * Omega_earth * np.sin(np.radians(attrs["latitude"]))
#---

#+++ Load buoyancy (initial and final, mid-depth) and vorticity (final, mid-depth) -- single-snapshot
# nearest-match, same convention as plot6_snapshots.py/plot9_buoyancy_evolution.py (no hard duration
# check: unlike the multi-snapshot-pooling plot5/plot7/plot8, a request beyond the run's actual length
# just silently resolves to the last available time, and the resolved value is printed/titled so a
# mismatch is still visible).
print("Opening dataset...")
ds_raw = xr.open_dataset(filename, decode_times=False)
x_km = ds_raw.x_caa.values / 1e3
y_km = ds_raw.y_aca.values / 1e3
f_cor = coriolis_f(ds_raw.attrs)

t_initial_sec = args.t_initial * 86400
t_final_sec = args.t_final * 86400

b_mid_t0    = fix_orientation(ds_raw["b"]).sel(time=t_initial_sec, method="nearest").sel(z_aac=args.z_mid, method="nearest")
b_mid_t1    = fix_orientation(ds_raw["b"]).sel(time=t_final_sec,   method="nearest").sel(z_aac=args.z_mid, method="nearest")
zeta_mid_t1 = fix_orientation(ds_raw["ζ"]).sel(time=t_final_sec,   method="nearest").sel(z_aac=args.z_mid, method="nearest") / f_cor

t0_days = float(b_mid_t0.time) / 86400
t1_days = float(b_mid_t1.time) / 86400
z_mid   = float(b_mid_t0.z_aac)
print(f"t_initial = {t0_days:.2f}d, t_final = {t1_days:.2f}d, z_mid = {z_mid:.1f}m, f0 = {f_cor:.4e} s^-1")
#---

#+++ Color scales: the two buoyancy panels share one scale (vmax = max of each panel's own clim_percentile,
# robust to a single outlier pixel) so the initial-vs-final comparison is directly meaningful; the Rossby
# number panel is a different, dimensionless quantity and gets its own independent scale.
b_panels = [
    (b_mid_t0, f"buoyancy b, z={z_mid:.0f}m\nt={t0_days:.1f}d"),
    (b_mid_t1, f"buoyancy b, z={z_mid:.0f}m\nt={t1_days:.1f}d"),
]
vmax_b = float(max(np.nanpercentile(np.abs(da.values), args.clim_percentile) for da, _ in b_panels))

zeta_title = f"Rossby number ζ/f, z={z_mid:.0f}m\nt={t1_days:.1f}d"
vmax_zeta = float(np.nanpercentile(np.abs(zeta_mid_t1.values), args.clim_percentile))
#---

#+++ Plot
print("Building figure...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True, sharex=True, sharey=True)

for ax, (da, title) in zip(axes[:2], b_panels):
    # set_edgecolor("face") avoids a known pcolormesh rendering artifact -- see plot6_snapshots.py's
    # comment for why linewidth is deliberately left untouched (linewidth=0 does not fix it).
    # rasterized=True: without it, a PDF pcolormesh draws one vector polygon per grid cell -- at a high
    # resolution (e.g. 384x384), three panels' worth of cells balloon the PDF to 10s of MB. Rasterizing
    # just this Artist embeds the mesh as a single bitmap (at the savefig dpi= below) while titles/axes/
    # text stay vector, cutting file size by roughly an order of magnitude with no visible quality loss.
    im_b = ax.pcolormesh(x_km, y_km, da.values, cmap="RdBu_r", vmin=-vmax_b, vmax=vmax_b, shading="auto", rasterized=True)
    im_b.set_edgecolor("face")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x [km]")

im_zeta = axes[2].pcolormesh(x_km, y_km, zeta_mid_t1.values, cmap="RdBu_r", vmin=-vmax_zeta, vmax=vmax_zeta, shading="auto", rasterized=True)
im_zeta.set_edgecolor("face")
axes[2].set_aspect("equal")
axes[2].set_title(zeta_title, fontsize=12)
axes[2].set_xlabel("x [km]")

axes[0].set_ylabel("y [km]")

# All three panels share the same x/y range (sharex/sharey=True), so the tick labels on columns 2/3 are
# pure repetition of column 1's -- keep the ticks (for gridline alignment) but drop their labels.
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

# One colorbar per quantity, each attached to its own rightmost panel with fraction/pad tuned to that one
# equal-aspect axes -- same idiom used for every other single-panel colorbar in this pipeline (e.g.
# anim3_panels.py's setup_map()) -- so each colorbar's height matches its panel height exactly. The
# buoyancy colorbar is attached to axes[1] (not axes[0]), matching plot9_buoyancy_evolution.py's own
# choice to place a shared colorbar at the right edge of the group of panels it applies to.
fig.colorbar(im_b,    ax=axes[1], fraction=0.046, pad=0.04, label="m s⁻²")
fig.colorbar(im_zeta, ax=axes[2], fraction=0.046, pad=0.04, label="")

fig.suptitle(f"{stem}: buoyancy evolution and final-time vorticity", fontsize=13)

outfile = FIGURES / f"{stem}_buoyancy_vorticity_t{t0_days:.0f}-{t1_days:.0f}d.pdf"
fig.savefig(outfile, dpi=450, bbox_inches="tight")
print(f"Saved: {outfile}")
#---
