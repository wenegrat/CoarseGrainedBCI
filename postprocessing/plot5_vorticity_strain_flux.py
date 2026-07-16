#!/usr/bin/env python
"""Cross-scale KE/APE flux conditioned on filtered-field vorticity and strain (Balwada et al. 2021 style)."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from src.aux00_utils import load_dataset_and_grid
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Conditional-mean and net-contribution maps of Πₖ, Π_A, and Πₖ+Π_A in filtered vorticity-strain space")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scale", type=float, default=None, help="Target filter length scale in meters (nearest available; defaults to the smallest available)")
parser.add_argument("--time", type=float, default=None, help="Target time in days (nearest available; defaults to the last available -- eddies should be fully developed by then)")
parser.add_argument("--z", type=float, default=-500.0, help="Target depth in meters (nearest available cell center; default -500, mid-depth)")
parser.add_argument("--n-bins", type=int, default=40, help="Number of bins per axis for the vorticity-strain JPDF (default 40)")
parser.add_argument("--min-count", type=int, default=5, help="Bins with fewer than this many points are masked out in the conditional-mean/net panels (default 5)")
parser.add_argument("--clim-percentile", type=float, default=99.0, help="Percentile of |data| used to set symmetric color limits for flux panels")
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
def fix_orientation(da):
    y_dim = next(d for d in da.dims if d.startswith("y"))
    x_dim = next(d for d in da.dims if d.startswith("x"))
    other_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
    return da.transpose(*other_dims, y_dim, x_dim)
#---

#+++ Load filtered velocities, pick filter scale/time/z, compute filtered vorticity and strain
print("Loading filtered velocities...")
filt = xr.open_dataset(PP_OUTPUT / f"{stem}_filtered_velocities.nc", decode_times=False)

ℓ_target = args.filter_scale if args.filter_scale is not None else float(filt.filter_scale.min())
ℓ = float(filt.filter_scale.sel(filter_scale=ℓ_target, method="nearest"))
ℓ_km = int(round(ℓ / 1000))

t_target = args.time * 86400 if args.time is not None else float(filt.time.max())
t_sel = float(filt.time.sel(time=t_target, method="nearest"))
t_days = t_sel / 86400

sel = dict(filter_scale=ℓ, time=t_sel, method="nearest")
ubar = fix_orientation(filt["ūᵢ"].sel(i=1, **sel, drop=True)).sel(z_aac=args.z, method="nearest")
vbar = fix_orientation(filt["ūᵢ"].sel(i=2, **sel, drop=True)).sel(z_aac=args.z, method="nearest")
z_sel = float(ubar.z_aac)
print(f"ℓ = {ℓ:.4f} m ({ℓ_km} km), t = {t_days:.2f} days, z = {z_sel:.1f} m")

sigma_n  = ubar.differentiate("x_caa") - vbar.differentiate("y_aca")
sigma_s  = vbar.differentiate("x_caa") + ubar.differentiate("y_aca")
zeta_bar = vbar.differentiate("x_caa") - ubar.differentiate("y_aca")
sigma_bar = np.sqrt(sigma_n**2 + sigma_s**2)

Omega_earth = 7.2921159e-5
f0 = 2 * Omega_earth * np.sin(np.radians(filt.attrs["latitude"]))
zeta_norm  = (zeta_bar / f0).values
sigma_norm = (sigma_bar / abs(f0)).values
print(f"f0 = {f0:.4e} s^-1")
#---

#+++ Load Πₖ, Π_A at the same ℓ, t, z; form the total
print("Loading Πₖ, Π_A...")
ke_fields  = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields.nc",  decode_times=False)
ape_fields = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc", decode_times=False)

t_flux = float(ke_fields.time.sel(time=t_sel, method="nearest"))
Pi_K = fix_orientation(ke_fields["Π_K"].sel(filter_scale=ℓ, time=t_flux, method="nearest")).sel(z_aac=args.z, method="nearest").values
Pi_A = fix_orientation(ape_fields["Π_A"].sel(filter_scale=ℓ, time=t_flux, method="nearest")).sel(z_aac=args.z, method="nearest").values
Pi_total = Pi_K + Pi_A
#---

#+++ Area weights (for the JPDF) and flattening
# Δx_caa/Δy_aca live on the raw sim file (not the filtered-velocities file), added by load_dataset_and_grid
ds_grid = load_dataset_and_grid(filename)
dx_1d = ds_grid.Δx_caa.values
dy_1d = ds_grid.Δy_aca.values
area_2d = np.outer(dy_1d, dx_1d)  # matches (y_aca, x_caa) after fix_orientation
A_total = area_2d.sum()

zeta_flat  = zeta_norm.ravel()
sigma_flat = sigma_norm.ravel()
area_flat  = area_2d.ravel()
Pi_K_flat, Pi_A_flat, Pi_total_flat = Pi_K.ravel(), Pi_A.ravel(), Pi_total.ravel()
#---

#+++ Bin edges (data-driven range) and JPDF
zeta_max  = np.nanpercentile(np.abs(zeta_flat), 99.5)
sigma_max = np.nanpercentile(sigma_flat, 99.5)
zeta_edges  = np.linspace(-zeta_max, zeta_max, args.n_bins + 1)
sigma_edges = np.linspace(0, sigma_max, args.n_bins + 1)

jpdf_counts, _, _ = np.histogram2d(zeta_flat, sigma_flat, bins=[zeta_edges, sigma_edges], weights=area_flat)
dζ = zeta_edges[1] - zeta_edges[0]
dσ = sigma_edges[1] - sigma_edges[0]
jpdf = jpdf_counts / A_total / dζ / dσ

counts, _, _, _ = binned_statistic_2d(zeta_flat, sigma_flat, None, statistic="count", bins=[zeta_edges, sigma_edges])
sparse_mask = counts < args.min_count
#---

#+++ Conditional means and net contributions
def cond_mean_and_net(values_flat):
    mean, _, _, _ = binned_statistic_2d(zeta_flat, sigma_flat, values_flat, statistic="mean", bins=[zeta_edges, sigma_edges])
    mean = np.where(sparse_mask, np.nan, mean)
    net = mean * jpdf
    return mean, net

Pi_K_mean, Pi_K_net = cond_mean_and_net(Pi_K_flat)
Pi_A_mean, Pi_A_net = cond_mean_and_net(Pi_A_flat)
Pi_total_mean, Pi_total_net = cond_mean_and_net(Pi_total_flat)
#---

#+++ Region (AVD/CVD/SD) flux fractions, using the net-contribution field
zeta_centers = 0.5 * (zeta_edges[:-1] + zeta_edges[1:])
sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
ZZ, SS = np.meshgrid(zeta_centers, sigma_centers, indexing="ij")
sd_mask, avd_mask, cvd_mask = SS >= np.abs(ZZ), (ZZ < 0) & (SS < np.abs(ZZ)), (ZZ > 0) & (SS < np.abs(ZZ))

def region_fractions(net):
    total = np.nansum(net)
    return {r: 100 * np.nansum(net[m]) / total for r, m in [("SD", sd_mask), ("AVD", avd_mask), ("CVD", cvd_mask)]}

fractions = {"Πₖ": region_fractions(Pi_K_net), "Π_A": region_fractions(Pi_A_net), "Πₖ+Π_A": region_fractions(Pi_total_net)}
for name, frac in fractions.items():
    print(f"  {name}: SD={frac['SD']:.1f}%  AVD={frac['AVD']:.1f}%  CVD={frac['CVD']:.1f}%")
#---

#+++ Plot
print("Building figure...")
fig = plt.figure(figsize=(18, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1, 1, 1])
ax_jpdf = fig.add_subplot(gs[:, 0])

flux_panels = [("Πₖ", Pi_K_mean, Pi_K_net), ("Π_A", Pi_A_mean, Pi_A_net), ("Πₖ+Π_A", Pi_total_mean, Pi_total_net)]
axes_mean = [fig.add_subplot(gs[0, c+1]) for c in range(3)]
axes_net  = [fig.add_subplot(gs[1, c+1]) for c in range(3)]

def add_sd_lines(ax):
    zmax = zeta_edges[-1]
    z = np.linspace(-zmax, zmax, 200)
    ax.plot(z, np.abs(z), "--", color="gray", lw=1)

im = ax_jpdf.pcolormesh(zeta_edges, sigma_edges, np.log10(jpdf.T + 1e-30), cmap="Reds", vmin=-5, vmax=1, shading="flat")
add_sd_lines(ax_jpdf)
ax_jpdf.set_xlabel(r"$\bar\zeta / f_0$")
ax_jpdf.set_ylabel(r"$\bar\sigma / |f_0|$")
ax_jpdf.set_title(f"JPDF  log$_{{10}}$P($\\bar\\zeta,\\bar\\sigma$)\n{stem}, ℓ={ℓ_km}km, t={t_days:.1f}d, z={z_sel:.0f}m")
ax_jpdf.text(0.02, 0.97, "SD", transform=ax_jpdf.transAxes, va="top", fontsize=11)
ax_jpdf.text(0.65, 0.05, "CVD", transform=ax_jpdf.transAxes, fontsize=11)
ax_jpdf.text(0.05, 0.05, "AVD", transform=ax_jpdf.transAxes, fontsize=11)
fig.colorbar(im, ax=ax_jpdf, fraction=0.046, pad=0.04)

for ax, (name, mean, _) in zip(axes_mean, flux_panels):
    vmax = np.nanpercentile(np.abs(mean), args.clim_percentile)
    im = ax.pcolormesh(zeta_edges, sigma_edges, mean.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    add_sd_lines(ax)
    ax.set_title(f"{name} conditional mean\nSD={fractions[name]['SD']:.0f}% AVD={fractions[name]['AVD']:.0f}% CVD={fractions[name]['CVD']:.0f}%", fontsize=10)
    ax.set_xlabel(r"$\bar\zeta / f_0$")
    ax.set_ylabel(r"$\bar\sigma / |f_0|$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax, (name, _, net) in zip(axes_net, flux_panels):
    vmax = np.nanpercentile(np.abs(net), args.clim_percentile)
    im = ax.pcolormesh(zeta_edges, sigma_edges, net.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    add_sd_lines(ax)
    ax.set_title(f"{name} net contribution", fontsize=10)
    ax.set_xlabel(r"$\bar\zeta / f_0$")
    ax.set_ylabel(r"$\bar\sigma / |f_0|$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

outfile = FIGURES / f"{stem}_vorticity_strain_flux_l{ℓ_km}km.pdf"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
#---
