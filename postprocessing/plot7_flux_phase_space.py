#!/usr/bin/env python
"""Net cross-scale KE/APE/total flux in vorticity-strain and vorticity-divergence phase space, side by side."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic_2d
from src.aux00_utils import load_dataset_and_grid
#---

#+++ Configuration
# Same default pooling window as plot5_vorticity_strain_flux.py, for consistency between the two -- see its
# own comment for the rationale (fixed window rather than "whatever this run's own length happens to be").
DEFAULT_TIME_MIN_DAYS = 15.0
DEFAULT_TIME_MAX_DAYS = 30.0

import argparse
parser = argparse.ArgumentParser(description="Net cross-scale KE/APE/total flux (Πₖ, Π_A, Πₖ+Π_A), conditioned on filtered-field vorticity-strain and vorticity-divergence phase space")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scale", type=float, default=None, help="Target filter length scale in meters (nearest available; defaults to the smallest available)")
parser.add_argument("--time-min", type=float, default=None, help=f"Start of the time range (days, inclusive; defaults to {DEFAULT_TIME_MIN_DAYS:g} days -- eddies should be fully developed by then)")
parser.add_argument("--time-max", type=float, default=None, help=f"End of the time range (days, inclusive; defaults to {DEFAULT_TIME_MAX_DAYS:g} days)")
parser.add_argument("--z", type=float, default=-500.0, help="Target depth in meters (nearest available cell center; default -500, mid-depth)")
parser.add_argument("--n-bins", type=int, default=40, help="Number of bins per axis for each phase-space JPDF (default 40)")
parser.add_argument("--min-count", type=int, default=5, help="Bins with fewer than this many points are masked out in the net-contribution panels (default 5)")
parser.add_argument("--clim-percentile", type=float, default=99.0, help="Percentile of |data| used to set symmetric color limits for flux panels")
parser.add_argument("--percentiles", type=float, nargs="+", default=[50, 90, 99], help="JPDF highest-density-region percentiles to contour on each flux panel (default 50 90 99)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Orientation fix: some pipeline fields (Π_A, the KE<->APE exchange term, and ūᵢ here) are stored with
# dims (..., x, y) instead of (..., y, x) like most fields -- plotting/differentiating without accounting
# for this silently produces rotated or transposed results. Always transpose to (..., y_dim, x_dim) first.
# See plot5_vorticity_strain_flux.py/plot6_snapshots.py/anim3_panels.py for the same pattern.
def fix_orientation(da):
    y_dim = next(d for d in da.dims if d.startswith("y"))
    x_dim = next(d for d in da.dims if d.startswith("x"))
    other_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
    return da.transpose(*other_dims, y_dim, x_dim)
#---

#+++ Load filtered velocities, pick filter scale/time/z, compute filtered vorticity, strain, divergence
print("Loading filtered velocities...")
filt = xr.open_dataset(PP_OUTPUT / f"{stem}_filtered_velocities.nc", decode_times=False)

# baroclinic_adjustment.jl's :fields writer uses schedule=ConsecutiveIterations(TimeInterval(...)), writing
# paired (t, t+ε) snapshots at every nominal output time -- see plot5_vorticity_strain_flux.py's comment on
# this exact structure. Keep only the first member of each pair before any time-range selection, same as
# plot5, or pooling would silently double the reported/actual sample count.
filt = filt.isel(time=slice(0, None, 2))

ℓ_target = args.filter_scale if args.filter_scale is not None else float(filt.filter_scale.min())
ℓ = float(filt.filter_scale.sel(filter_scale=ℓ_target, method="nearest"))
ℓ_km = int(round(ℓ / 1000))

time_min_days = args.time_min if args.time_min is not None else DEFAULT_TIME_MIN_DAYS
time_max_days = args.time_max if args.time_max is not None else DEFAULT_TIME_MAX_DAYS

# Same duration checks as plot5_vorticity_strain_flux.py -- see its comment for the reasoning (hard error if
# the window can't even start, clip-and-warn if it can start but not finish).
available_max_days = float(filt.time.max()) / 86400
if available_max_days < time_min_days:
    raise ValueError(f"Requested time range starts at {time_min_days:g} days, but {stem} only has "
                     f"{available_max_days:.2f} days of output -- pass --time-min/--time-max explicitly "
                     f"for a shorter run.")
if available_max_days < time_max_days:
    print(f"  Warning: requested time range extends to {time_max_days:g} days, but {stem} only has "
          f"{available_max_days:.2f} days of output -- clipping to [{time_min_days:g}, {available_max_days:.2f}] days.")

t_min_sec = time_min_days * 86400
t_max_sec = time_max_days * 86400
filt_t = filt.sel(time=slice(t_min_sec, t_max_sec))
n_times = filt_t.sizes["time"]
if n_times == 0:
    raise ValueError(f"No available times in [{t_min_sec/86400:.2f}, {t_max_sec/86400:.2f}] days")
t_days_actual = filt_t.time.values / 86400
t_min_days, t_max_days = float(t_days_actual.min()), float(t_days_actual.max())

ubar = fix_orientation(filt_t["ūᵢ"].sel(i=1, filter_scale=ℓ, method="nearest", drop=True)).sel(z_aac=args.z, method="nearest")
vbar = fix_orientation(filt_t["ūᵢ"].sel(i=2, filter_scale=ℓ, method="nearest", drop=True)).sel(z_aac=args.z, method="nearest")
z_sel = float(ubar.z_aac)
print(f"ℓ = {ℓ:.4f} m ({ℓ_km} km), t = {t_min_days:.2f}-{t_max_days:.2f} days ({n_times} snapshots), z = {z_sel:.1f} m")

sigma_n  = ubar.differentiate("x_caa") - vbar.differentiate("y_aca")
sigma_s  = vbar.differentiate("x_caa") + ubar.differentiate("y_aca")
zeta_bar = vbar.differentiate("x_caa") - ubar.differentiate("y_aca")
div_bar  = ubar.differentiate("x_caa") + vbar.differentiate("y_aca")
sigma_bar = np.sqrt(sigma_n**2 + sigma_s**2)

Omega_earth = 7.2921159e-5
f0 = 2 * Omega_earth * np.sin(np.radians(filt.attrs["latitude"]))
zeta_norm  = (zeta_bar / f0).values
sigma_norm = (sigma_bar / abs(f0)).values
# δ (unlike σ, a magnitude by construction) is signed -- convergence vs. divergence -- so it's normalized
# by |f0| the same way σ is (no inherent rotational sense to preserve, unlike ζ/f0's cyclonic/anticyclonic
# sign convention), but keeps its own sign rather than being folded into a positive-definite magnitude.
div_norm = (div_bar / abs(f0)).values
print(f"f0 = {f0:.4e} s^-1")
#---

#+++ Load Πₖ, Π_A at the same ℓ, t, z; form the total
print("Loading Πₖ, Π_A...")
ke_fields  = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields.nc",  decode_times=False)
ape_fields = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc", decode_times=False)

Pi_K = fix_orientation(ke_fields["Π_K"].sel(filter_scale=ℓ, time=filt_t.time, method="nearest")).sel(z_aac=args.z, method="nearest").values
Pi_A = fix_orientation(ape_fields["Π_A"].sel(filter_scale=ℓ, time=filt_t.time, method="nearest")).sel(z_aac=args.z, method="nearest").values
Pi_total = Pi_K + Pi_A
#---

#+++ Area weights and flattening
ds_grid = load_dataset_and_grid(filename)
area_2d = np.outer(ds_grid.Δy_aca.values, ds_grid.Δx_caa.values)  # matches (y_aca, x_caa) after fix_orientation
area_3d = np.broadcast_to(area_2d, zeta_norm.shape)
area_flat = area_3d.ravel()

zeta_flat  = zeta_norm.ravel()
sigma_flat = sigma_norm.ravel()
div_flat   = div_norm.ravel()
Pi_K_flat, Pi_A_flat, Pi_total_flat = Pi_K.ravel(), Pi_A.ravel(), Pi_total.ravel()
#---

#+++ Shared JPDF / net-contribution machinery, generalized over an arbitrary 2D phase-space pair (x, y) --
# same approach as plot5_vorticity_strain_flux.py, factored into functions here since this script applies
# it twice (vorticity-strain, vorticity-divergence) rather than once.
def compute_jpdf(x_flat, y_flat, x_edges, y_edges):
    """Area-weighted JPDF over (x_edges, y_edges), normalized so it integrates to 1."""
    counts_2d, _, _ = np.histogram2d(x_flat, y_flat, bins=[x_edges, y_edges], weights=area_flat)
    dx, dy = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]
    return counts_2d / area_flat.sum() / dx / dy, dx, dy

def cond_mean_and_net(x_flat, y_flat, values_flat, x_edges, y_edges, jpdf, min_count):
    counts, _, _, _ = binned_statistic_2d(x_flat, y_flat, None, statistic="count", bins=[x_edges, y_edges])
    mean, _, _, _ = binned_statistic_2d(x_flat, y_flat, values_flat, statistic="mean", bins=[x_edges, y_edges])
    mean = np.where(counts < min_count, np.nan, mean)
    return mean * jpdf

# JPDF highest-density-region contour levels: the level of P such that {P > level} contains a given
# percentile of the total probability mass (a "percentile contour", as in Balwada et al.'s gray contours) --
# NOT a fixed absolute-probability threshold, since that wouldn't compare meaningfully across filter scales/
# resolutions/phase spaces where the JPDF's overall magnitude differs.
def percentile_levels(density, mass, percentiles):
    """Return {percentile: density level} such that {density > level} contains that percentile of the mass."""
    order = np.argsort(density.ravel())[::-1]
    sorted_density = density.ravel()[order]
    cum_mass = np.cumsum(mass.ravel()[order])
    total = cum_mass[-1]
    levels = {}
    for p in percentiles:
        idx = min(np.searchsorted(cum_mass, p / 100 * total), len(sorted_density) - 1)
        levels[p] = sorted_density[idx]
    return levels
#---

#+++ Row 1: vorticity-strain (ζ, σ) phase space
zeta_max  = np.nanpercentile(np.abs(zeta_flat), 99.5)
sigma_max = np.nanpercentile(sigma_flat, 99.5)
zeta_edges  = np.linspace(-zeta_max, zeta_max, args.n_bins + 1)
sigma_edges = np.linspace(0, sigma_max, args.n_bins + 1)
zeta_centers  = 0.5 * (zeta_edges[:-1] + zeta_edges[1:])
sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])

jpdf_zs, dζ, dσ = compute_jpdf(zeta_flat, sigma_flat, zeta_edges, sigma_edges)
net_zs = {name: cond_mean_and_net(zeta_flat, sigma_flat, vals, zeta_edges, sigma_edges, jpdf_zs, args.min_count)
          for name, vals in [("Πₖ", Pi_K_flat), ("Π_A", Pi_A_flat), ("Πₖ+Π_A", Pi_total_flat)]}
levels_zs = percentile_levels(jpdf_zs, jpdf_zs * dζ * dσ, args.percentiles)
print(f"  Vorticity-strain JPDF percentile levels -> {levels_zs}")
#---

#+++ Row 2: vorticity-divergence (ζ, δ) phase space -- same ζ bins as row 1 (directly comparable x-axis
# across rows), but δ's own bins are symmetric about 0 (unlike σ, δ can be negative -- convergence).
div_max = np.nanpercentile(np.abs(div_flat), 99.5)
div_edges = np.linspace(-div_max, div_max, args.n_bins + 1)
div_centers = 0.5 * (div_edges[:-1] + div_edges[1:])

jpdf_zd, dζ2, dδ = compute_jpdf(zeta_flat, div_flat, zeta_edges, div_edges)
net_zd = {name: cond_mean_and_net(zeta_flat, div_flat, vals, zeta_edges, div_edges, jpdf_zd, args.min_count)
          for name, vals in [("Πₖ", Pi_K_flat), ("Π_A", Pi_A_flat), ("Πₖ+Π_A", Pi_total_flat)]}
levels_zd = percentile_levels(jpdf_zd, jpdf_zd * dζ2 * dδ, args.percentiles)
print(f"  Vorticity-divergence JPDF percentile levels -> {levels_zd}")
#---

#+++ Plot: 2 rows (vorticity-strain, vorticity-divergence) x 3 columns (Πₖ, Π_A, Πₖ+Π_A), net contribution only
print("Building figure...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)

_LINESTYLES = ["dotted", "dashed", "solid", "dashdot"]
names = ["Πₖ", "Π_A", "Πₖ+Π_A"]

# Percentile -> linestyle is fixed by position in args.percentiles (not by sorted contour value), so the
# same linestyle means "the Nth requested percentile" consistently across both rows even though rows 1 and
# 2 have different underlying JPDFs (and therefore different actual density thresholds per percentile).
def add_jpdf_contours(ax, x_centers, y_centers, jpdf, level_by_percentile):
    for i, p in enumerate(args.percentiles):
        ax.contour(x_centers, y_centers, jpdf.T, levels=[level_by_percentile[p]], colors="0.25",
                   linewidths=1.0, linestyles=_LINESTYLES[i % len(_LINESTYLES)])

for col, name in enumerate(names):
    ax = axes[0, col]
    net = net_zs[name]
    vmax = np.nanpercentile(np.abs(net), args.clim_percentile)
    im = ax.pcolormesh(zeta_edges, sigma_edges, net.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    im.set_edgecolor("face")  # see plot6_snapshots.py's comment for why linewidth=0 doesn't work here
    zmax = zeta_edges[-1]
    z = np.linspace(-zmax, zmax, 200)
    ax.plot(z, np.abs(z), "--", color="gray", lw=1)  # σ=|ζ| strain/vorticity-dominated boundary
    add_jpdf_contours(ax, zeta_centers, sigma_centers, jpdf_zs, levels_zs)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel(r"$\bar\zeta / f_0$")
    ax.set_ylabel(r"$\bar\sigma / |f_0|$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m² s⁻³")

for col, name in enumerate(names):
    ax = axes[1, col]
    net = net_zd[name]
    vmax = np.nanpercentile(np.abs(net), args.clim_percentile)
    im = ax.pcolormesh(zeta_edges, div_edges, net.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    im.set_edgecolor("face")
    ax.axhline(0, color="gray", lw=1, ls="--")  # convergence/divergence boundary
    add_jpdf_contours(ax, zeta_centers, div_centers, jpdf_zd, levels_zd)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel(r"$\bar\zeta / f_0$")
    ax.set_ylabel(r"$\bar\delta / |f_0|$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m² s⁻³")

# Row labels in the left margin, since "net contribution" + phase-space axes already say what each row is
# once labeled -- repeating a full title on every one of the 6 panels would be redundant.
fig.text(-0.01, 0.77, "vorticity–strain", rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
fig.text(-0.01, 0.27, "vorticity–divergence", rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")

legend_handles = [Line2D([0], [0], color="0.25", lw=1.2, linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                        label=f"JPDF {p:g}% HDR") for i, p in enumerate(args.percentiles)]
fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 1.06))

fig.suptitle(f"{stem}: net cross-scale flux, ℓ={ℓ_km}km, t={t_min_days:.1f}-{t_max_days:.1f}d "
             f"({n_times} snapshots), z={z_sel:.0f}m", fontsize=13, y=1.12)

z_m = int(round(z_sel))
outfile = FIGURES / f"{stem}_flux_phase_space_l{ℓ_km}km_z{z_m}m_t{t_min_days:.0f}-{t_max_days:.0f}d.pdf"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
#---
