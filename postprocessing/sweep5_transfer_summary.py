#!/usr/bin/env python
"""Combined KE/APE cross-scale transfer summary: KE and APE spectra overlaid on one wide top panel (from
sweep3_plot_transfer_spectrum.py's own spectrum panels), KE/APE/total depth structure below (three panels,
from sweep4_plot_depth_scale.py's own first two panels plus their sum)."""

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
import argparse
parser = argparse.ArgumentParser(description="Summary figure: KE+APE cross-scale transfer spectrum overlaid "
                                              "on one wide top panel, KE/APE/total depth structure below (three panels)")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
# One shared time window for both rows.
parser.add_argument("--min-time-days", type=float, default=10.0, help="Exclude time samples before this (in days), for both rows (default 10.0)")
parser.add_argument("--max-time-days", type=float, default=None, help="Exclude time samples after this (in days). Defaults to the latest available time in the run (no upper restriction).")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
FIGURES = REPO_ROOT / "figures" / Path(filename).stem  # one subfolder per run, keyed by input filename stem
FIGURES.mkdir(parents=True, exist_ok=True)
#---

#+++ Load energy transfer data, dedup pairs, restrict to the shared time window -- used by both rows below
print("Loading energy transfer data...")
input_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
et = xr.open_dataset(input_filename, decode_timedelta=False).chunk({"time": 1})

# baroclinic_adjustment.jl's :fields writer uses schedule=ConsecutiveIterations(TimeInterval(...)), which
# writes TWO consecutive model iterations at every nominal output time -- see sweep3/sweep4's own
# identical comment on this structure. Keep only the first member of each pair, on the full time axis
# before any time-based selection, so the pair parity stays anchored to the simulation start.
et = et.isel(time=slice(0, None, 2))

t_max_sec = args.max_time_days * 86400 if args.max_time_days is not None else float(et.time.max())
et = et.sel(time=slice(args.min_time_days * 86400, t_max_sec))
n_times = et.sizes["time"]

et = et.assign_coords(inv_scale=("filter_scale", 1.0 / et.filter_scale.values))
print(f"  Loaded: {input_filename}")
print(f"  Time steps used: {n_times}   Filter scales: {len(et.filter_scale)}")

# Horizontal area weights for the bottom row's area-weighted mean -- Δx_caa/Δy_aca live on the raw sim
# file, not the energy-transfer output.
ds_grid = load_dataset_and_grid(filename)
dA = ds_grid.Δx_caa * ds_grid.Δy_aca
#---

#+++ Theoretical (Eady) deformation radius Ld = N*Lz/|f0|, from this run's own N2/Lz/latitude attrs
Omega_earth = 7.2921159e-5
f0 = 2 * Omega_earth * np.sin(np.radians(float(et.attrs["latitude"])))
N = np.sqrt(float(et.attrs["N2"]))
Lz = float(et.attrs["Lz"])
Ld = N * Lz / abs(f0)
inv_Ld = 1.0 / Ld
print(f"  Deformation radius Ld = {Ld/1e3:.2f} km")
#---

#+++ Top panel data: time-averaged spectrum (mean, +/-1 std across the pooled snapshots, vs 1/ℓ) -- same
# quantity/normalization as sweep3_plot_transfer_spectrum.py's own spectrum panels, just overlaid on one
# shared axes instead of two separate ones.
V_total = et.attrs["Lx"] * et.attrs["Ly"] * et.attrs["Lz"]
print(f"  Domain volume V = {V_total:.4e} m^3 (normalizing ∫Π dV terms to domain-averaged m² s⁻³)")

mean_ke,  std_ke  = et["∫Π_K dV"].mean("time") / V_total, et["∫Π_K dV"].std("time") / V_total
mean_ape, std_ape = et["∫Π_A dV"].mean("time") / V_total, et["∫Π_A dV"].std("time") / V_total
#---

#+++ Bottom row data: depth vs 1/ℓ structure of the horizontally- and time-averaged Π_K/Π_A, plus their sum
# -- same quantity/normalization as sweep4_plot_depth_scale.py's own first two panels (its 3rd/4th panels --
# buoyancy conversion ℓ-derivative and the Πₖ+Π_A total -- were dropped from those two originally; the
# total is recomputed here as a third panel instead).
def horiz_time_mean(da):
    horiz_mean = (da * dA).sum(("x_caa", "y_aca")) / dA.sum(("x_caa", "y_aca"))
    return horiz_mean.mean("time").compute()

Pi_K_zl = horiz_time_mean(et["Π_K"])
Pi_A_zl = horiz_time_mean(et["Π_A"])
Pi_total_zl = Pi_K_zl + Pi_A_zl
depth_panels = [
    (Pi_K_zl,     "KE cross-scale transfer  Π_K(z, ℓ)"),
    (Pi_A_zl,     "APE cross-scale transfer  Π_A(z, ℓ)"),
    (Pi_total_zl, "Total cross-scale transfer  Π_K+Π_A(z, ℓ)"),
]
print("Done!")
#---

#+++ Plot: one wide top panel (Πₖ and Π_A overlaid, symlog y) over three depth-structure panels below.
# Colors match plot8_depth_profile.py's own Πₖ/Π_A palette (itself matching the inherited, unadapted
# postprocessing/S3_plot_sweep.py). Πₖ and Π_A differ by roughly an order of magnitude here (confirmed on
# real data: Πₖ ~1e-7, Π_A ~1e-6) -- a shared *linear* y-axis would flatten Πₖ's own variation to a barely
# visible sliver near zero. S3_plot_sweep.py hit this exact problem (its own top panel overlays Πₖ, Π_A,
# and two exchange terms on one axes) and used a symlog y-scale to fix it; this mirrors that solution
# rather than inventing a new one, and is also the same class of fix as the SymLogNorm already used for
# the Hovmöller color scales elsewhere in this pipeline (sweep3_plot_transfer_spectrum.py).
print("Building figure...")
# One place to tune every text element's size, rather than scattering ad hoc numbers through the plot
# code below.
FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 17, 15, 14, 13, 13

fig = plt.figure(figsize=(15, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
ax_top = fig.add_subplot(gs[0, :])
ax_ke_z    = fig.add_subplot(gs[1, 0])
ax_ape_z   = fig.add_subplot(gs[1, 1], sharex=ax_ke_z, sharey=ax_ke_z)
ax_total_z = fig.add_subplot(gs[1, 2], sharex=ax_ke_z, sharey=ax_ke_z)

C_PI_K, C_PI_A = "#2166ac", "#d6604d"

ax_top.plot(et.inv_scale, mean_ke, color=C_PI_K, lw=1.8, marker="o", ms=4, label=r"$\Pi_K$")
ax_top.fill_between(et.inv_scale, mean_ke - std_ke, mean_ke + std_ke, color=C_PI_K, alpha=0.2)
ax_top.plot(et.inv_scale, mean_ape, color=C_PI_A, lw=1.8, marker="o", ms=4, label=r"$\Pi_A$")
ax_top.fill_between(et.inv_scale, mean_ape - std_ape, mean_ape + std_ape, color=C_PI_A, alpha=0.2)
ax_top.axhline(0, color="gray", lw=0.5)
ax_top.axvline(inv_Ld, color="k", ls="--", lw=1.2, label=f"$L_d$={Ld/1e3:.1f}km")
# linthresh sized off Πₖ (the weaker of the two) so its own variation lands in the log-scaled region,
# rather than off the shared/Π_A-dominated vmax -- a fixed fraction of the overall max would put most of
# Πₖ's range inside the flat linear zone and defeat the point of using symlog at all.
vmax_ke = float(max(abs(mean_ke - std_ke).max(), abs(mean_ke + std_ke).max()))
ax_top.set_yscale("symlog", linthresh=0.1 * vmax_ke)
ax_top.set_xscale("log")
# Π_A's positive extent and Πₖ's negative extent differ in magnitude, so matplotlib's autoscaled ylim is
# lopsided by default (e.g. -1e-7 to 1e-6) -- symmetrize about zero using whichever side reaches farther,
# so 0 isn't visually off-center on an axis for a signed quantity. Padded by Y_PAD (multiplicative, not
# additive, since the axis is symlog) so the data doesn't sit flush against the top/bottom edges.
Y_PAD = 1.5
y_abs = max(abs(v) for v in ax_top.get_ylim()) * Y_PAD
ax_top.set_ylim(-y_abs, y_abs)
ax_top.legend(fontsize=FS_LEGEND, loc="best")
ax_top.set_ylabel("Cross-scale transfer  [m² s⁻³]", fontsize=FS_LABEL)
ax_top.set_title("KE and APE cross-scale transfer spectrum", fontsize=FS_TITLE)
ax_top.grid(True, alpha=0.3)
# ax_top isn't in a matplotlib sharex group with the bottom row (it spans all three columns, so there's no
# single bottom-row axes it could truly share with) -- its x-axis is only *visually* aligned below via the
# explicit set_xlim/set_xscale calls, so it needs its own real tick labels/axis label rather than relying
# on the row below to supply them.
ax_top.set_xlabel(r"$1/\ell$  [m$^{-1}$]", fontsize=FS_LABEL)
ax_top.tick_params(labelsize=FS_TICK)

# All three depth-structure panels share one color limit and one colorbar -- Πₖ/Π_A/total differ by
# roughly an order of magnitude (see the top panel's own comment on this), so Πₖ will look comparatively
# muted here, but that's the deliberate point of a shared scale: it shows the true relative magnitude
# across panels rather than each one's own internal contrast.
vmax_zl = float(max(np.nanpercentile(np.abs(da), 99) for da, _ in depth_panels))

inv_scale = et.inv_scale.values
depth_axes = [ax_ke_z, ax_ape_z, ax_total_z]
for ax, (da, title) in zip(depth_axes, depth_panels):
    im = ax.pcolormesh(inv_scale, da.z_aac, da.transpose("z_aac", "filter_scale").values,
                       cmap="RdBu_r", vmin=-vmax_zl, vmax=vmax_zl, shading="nearest")
    ax.set_xlabel(r"$1/\ell$  [m$^{-1}$]", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
ax_ke_z.set_xscale("log")   # the depth-row axes share x with each other, but not with ax_top -- set
                            # explicitly here too, matching ax_top's own x-scale for visual consistency.
ax_ke_z.set_xlim(ax_top.get_xlim())
ax_ke_z.set_ylabel("Depth  [m]", fontsize=FS_LABEL)
for ax in (ax_ape_z, ax_total_z):
    ax.tick_params(labelleft=False)  # share y with ax_ke_z -- drop the repeated depth tick labels

# One colorbar for all three panels, attached to the rightmost one with fraction/pad tuned to that single
# axes -- same height-matching idiom used throughout this pipeline (e.g. plot9/plot10's shared colorbars).
cbar = fig.colorbar(im, ax=ax_total_z, fraction=0.046, pad=0.04)
cbar.set_label("m² s⁻³", fontsize=FS_LABEL)
cbar.ax.tick_params(labelsize=FS_TICK)

label = run_label(et.attrs)
max_label = f"{args.max_time_days}d" if args.max_time_days is not None else "end"
suptitle = f"t = {args.min_time_days}d–{max_label}  ({n_times} samples)"
if label:
    suptitle = f"{label}\n{suptitle}"
fig.suptitle(suptitle, fontsize=FS_SUPTITLE)

plot_filename = str(FIGURES / (Path(filename).stem + f"_transfer_summary{ref_suffix}.png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---
