#!/usr/bin/env python
"""2x3 snapshot + vorticity-strain phase-space summary of cross-scale KE/APE/total flux, at one shared
filter scale and depth: top row is a single-time map snapshot (as in plot6_snapshots.py), bottom row is
the conditional mean in filtered vorticity-strain space pooled over a time range (as in
plot5_vorticity_strain_flux.py)."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from scipy.stats import binned_statistic_2d
from src.aux00_utils import load_dataset_and_grid
#---

#+++ Configuration
# Default pooling window (days) for the bottom row -- matches plot5_vorticity_strain_flux.py's own default
# (late enough that eddies should be fully developed), fixed rather than tied to each run's own length so
# repeated runs/comparisons default to the same physical window.
DEFAULT_TIME_MIN_DAYS = 15.0
DEFAULT_TIME_MAX_DAYS = 30.0

import argparse
parser = argparse.ArgumentParser(description="2x3: top row = single-time Πₖ/Π_A/Πₖ+Π_A snapshot maps, "
                                              "bottom row = their conditional mean in vorticity-strain "
                                              "space pooled over a time range -- same filter scale and "
                                              "depth shared by both rows")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scale", type=float, default=None, help="Target filter length scale in meters (nearest available; defaults to the smallest available), shared by both rows")
parser.add_argument("--z", type=float, default=-500.0, help="Target depth in meters (nearest available cell center; default -500, mid-depth), shared by both rows")
parser.add_argument("--time", type=float, default=None, help="Top-row snapshot time in days (nearest available; defaults to the last available)")
parser.add_argument("--time-min", type=float, default=None, help=f"Bottom-row pooling window start (days, inclusive; defaults to {DEFAULT_TIME_MIN_DAYS:g} days -- eddies should be fully developed by then)")
parser.add_argument("--time-max", type=float, default=None, help=f"Bottom-row pooling window end (days, inclusive; defaults to {DEFAULT_TIME_MAX_DAYS:g} days)")
parser.add_argument("--n-bins", type=int, default=40, help="Number of bins per axis for the bottom row's vorticity-strain JPDF (default 40)")
parser.add_argument("--min-count", type=int, default=5, help="Bottom-row bins with fewer than this many points are masked out (default 5)")
parser.add_argument("--clim-percentile", type=float, default=99.0, help="Percentile of |data| used to set symmetric color limits (both rows)")
parser.add_argument("--percentiles", type=float, nargs="+", default=[50, 90, 99], help="JPDF highest-density-region percentiles to contour on each bottom-row panel (default 50 90 99)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Orientation fix: some pipeline fields (Π_A, the KE<->APE exchange term, and ūᵢ) are stored with dims
# (..., x, y) instead of (..., y, x) like most fields -- always transpose to (..., y_dim, x_dim) first. See
# plot5_vorticity_strain_flux.py/plot6_snapshots.py for the same pattern.
def fix_orientation(da):
    y_dim = next(d for d in da.dims if d.startswith("y"))
    x_dim = next(d for d in da.dims if d.startswith("x"))
    other_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
    return da.transpose(*other_dims, y_dim, x_dim)

def coriolis_f(attrs):
    """Coriolis parameter from this run's own attrs (for the ζ/f, σ/|f| normalization)."""
    Omega_earth = 7.2921159e-5
    return 2 * Omega_earth * np.sin(np.radians(attrs["latitude"]))
#---

#+++ Shared data: Πₖ/Π_A budget fields, filter-scale resolution (same scale used by both rows)
print("Opening budget field datasets...")
ke_fields  = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields.nc",  decode_times=False)
ape_fields = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc", decode_times=False)

ℓ_target = args.filter_scale if args.filter_scale is not None else float(ke_fields.filter_scale.min())
ℓ = float(ke_fields.filter_scale.sel(filter_scale=ℓ_target, method="nearest"))
ℓ_km = int(round(ℓ / 1000))
print(f"ℓ = {ℓ:.4f} m ({ℓ_km} km)")
#---

#+++ Top row: single-time snapshot maps of Πₖ, Π_A, Πₖ+Π_A (mirrors plot6_snapshots.py)
print("Building top-row snapshot fields...")
ds_raw = xr.open_dataset(filename, decode_times=False)
x_km = ds_raw.x_caa.values / 1e3
y_km = ds_raw.y_aca.values / 1e3

t_target = args.time * 86400 if args.time is not None else float(ke_fields.time.max())
t_sel = float(ke_fields.time.sel(time=t_target, method="nearest"))
t_days = t_sel / 86400
t_sel_raw = float(ds_raw.time.sel(time=t_sel, method="nearest"))

# Buoyancy contour overlay for visual reference to the front/eddy structure driving the flux, same
# convention as plot6_snapshots.py.
b_snap = fix_orientation(ds_raw["b"]).sel(time=t_sel_raw, method="nearest").sel(z_aac=args.z, method="nearest")
z_sel = float(b_snap.z_aac)

Pi_K_snap = fix_orientation(ke_fields["Π_K"].sel(filter_scale=ℓ, time=t_sel, method="nearest")).sel(z_aac=args.z, method="nearest")
Pi_A_snap = fix_orientation(ape_fields["Π_A"].sel(filter_scale=ℓ, time=t_sel, method="nearest")).sel(z_aac=args.z, method="nearest")
Pi_total_snap = Pi_K_snap + Pi_A_snap
print(f"  Snapshot: t={t_days:.2f}d, z={z_sel:.1f}m")
#---

#+++ Bottom row: conditional mean of Πₖ, Π_A, Πₖ+Π_A in filtered vorticity-strain space, pooled over
# [--time-min, --time-max] (mirrors plot5_vorticity_strain_flux.py)
print("Loading filtered velocities for vorticity/strain...")
filt = xr.open_dataset(PP_OUTPUT / f"{stem}_filtered_velocities.nc", decode_times=False)

# baroclinic_adjustment.jl's :fields writer uses schedule=ConsecutiveIterations(TimeInterval(...)), which
# writes TWO consecutive model iterations at every nominal output time -- see plot5_vorticity_strain_flux.py's
# own comment on this exact structure. Keep only the first member of each pair, on the full time axis
# before any time-based selection, so the pair parity stays anchored to the simulation start.
filt = filt.isel(time=slice(0, None, 2))

time_min_days = args.time_min if args.time_min is not None else DEFAULT_TIME_MIN_DAYS
time_max_days = args.time_max if args.time_max is not None else DEFAULT_TIME_MAX_DAYS

# Duration checks against this run's own actual length: < time_min_days is a hard error -- the requested/
# default window can't even start, and silently pooling from whatever scraps exist near the run's actual
# end would be more confusing than failing loudly. < time_max_days (but >= time_min_days) is survivable --
# clip and warn, since a shorter-than-requested-but-nonempty pooling window is still meaningful.
available_max_days = float(filt.time.max()) / 86400
if available_max_days < time_min_days:
    raise ValueError(f"Requested time range starts at {time_min_days:g} days, but {stem} only has "
                     f"{available_max_days:.2f} days of output -- pass --time-min/--time-max explicitly "
                     f"for a shorter run.")
if available_max_days < time_max_days:
    print(f"  Warning: requested time range extends to {time_max_days:g} days, but {stem} only has "
          f"{available_max_days:.2f} days of output -- clipping to [{time_min_days:g}, {available_max_days:.2f}] days.")

t_min_sec = time_min_days * 86400
t_max_sec = time_max_days * 86400  # clipped to whatever's available by .sel(slice(...)) below if too large
filt_t = filt.sel(time=slice(t_min_sec, t_max_sec))
n_times = filt_t.sizes["time"]
if n_times == 0:
    raise ValueError(f"No available times in [{t_min_sec/86400:.2f}, {t_max_sec/86400:.2f}] days")
t_days_actual = filt_t.time.values / 86400
t_min_days_actual, t_max_days_actual = float(t_days_actual.min()), float(t_days_actual.max())

ubar = fix_orientation(filt_t["ūᵢ"].sel(i=1, filter_scale=ℓ, method="nearest", drop=True)).sel(z_aac=args.z, method="nearest")
vbar = fix_orientation(filt_t["ūᵢ"].sel(i=2, filter_scale=ℓ, method="nearest", drop=True)).sel(z_aac=args.z, method="nearest")
z_sel_bottom = float(ubar.z_aac)
print(f"  Pooling: t={t_min_days_actual:.2f}-{t_max_days_actual:.2f}d ({n_times} snapshots), z={z_sel_bottom:.1f}m")

sigma_n  = ubar.differentiate("x_caa") - vbar.differentiate("y_aca")
sigma_s  = vbar.differentiate("x_caa") + ubar.differentiate("y_aca")
zeta_bar = vbar.differentiate("x_caa") - ubar.differentiate("y_aca")
sigma_bar = np.sqrt(sigma_n**2 + sigma_s**2)

f0 = coriolis_f(filt.attrs)
zeta_norm  = (zeta_bar / f0).values
sigma_norm = (sigma_bar / abs(f0)).values
print(f"  f0 = {f0:.4e} s^-1")

# Each of filt_t's own times is matched to its nearest neighbor in the flux files independently
# (vectorized nearest-time selection), same reasoning as plot5_vorticity_strain_flux.py's own Π_A
# selection -- the files can have slightly different time grids/roundoff.
Pi_K_pool = fix_orientation(ke_fields["Π_K"].sel(filter_scale=ℓ, time=filt_t.time, method="nearest")).sel(z_aac=args.z, method="nearest").values
Pi_A_pool = fix_orientation(ape_fields["Π_A"].sel(filter_scale=ℓ, time=filt_t.time, method="nearest")).sel(z_aac=args.z, method="nearest").values
Pi_total_pool = Pi_K_pool + Pi_A_pool
#---

#+++ Area weights (for the JPDF) and flattening
ds_grid = load_dataset_and_grid(filename)
dx_1d = ds_grid.Δx_caa.values
dy_1d = ds_grid.Δy_aca.values
area_2d = np.outer(dy_1d, dx_1d)  # matches (y_aca, x_caa) after fix_orientation

# Broadcast the (time-independent) area weights across the time dimension so every included snapshot
# contributes one full domain's worth of area-weighted samples -- snapshots are weighted equally
# regardless of Δt, consistent with plot5_vorticity_strain_flux.py's own convention.
area_3d = np.broadcast_to(area_2d, zeta_norm.shape)
A_total = area_3d.sum()

zeta_flat  = zeta_norm.ravel()
sigma_flat = sigma_norm.ravel()
area_flat  = area_3d.ravel()
Pi_K_flat, Pi_A_flat, Pi_total_flat = Pi_K_pool.ravel(), Pi_A_pool.ravel(), Pi_total_pool.ravel()
#---

#+++ Bin edges (data-driven range, fixed 99.5th percentile independent of --clim-percentile -- same as
# plot5_vorticity_strain_flux.py, since this sets the JPDF's own axis range, not a flux color limit) and JPDF
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

#+++ Conditional mean (+ net contribution, used only to derive the SD/AVD/CVD region fractions shown in
# each bottom-row panel's title -- the net-contribution maps themselves aren't plotted here, unlike
# plot5_vorticity_strain_flux.py's own 2x3 mean+net layout)
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

#+++ JPDF highest-density-region contour levels: the level of P such that {P > level} contains a given
# percentile of the total probability mass -- same "percentile contour" construction as
# plot5_vorticity_strain_flux.py, so it compares meaningfully across filter scales/resolutions.
def percentile_levels(density, mass, percentiles):
    order = np.argsort(density.ravel())[::-1]
    sorted_density = density.ravel()[order]
    cum_mass = np.cumsum(mass.ravel()[order])
    total = cum_mass[-1]
    levels = {}
    for p in percentiles:
        idx = min(np.searchsorted(cum_mass, p / 100 * total), len(sorted_density) - 1)
        levels[p] = sorted_density[idx]
    return levels

mass = jpdf * dζ * dσ
level_by_percentile = percentile_levels(jpdf, mass, args.percentiles)
contour_levels = sorted(level_by_percentile.values())
level_labels = {lvl: f"{p:g}%" for p, lvl in level_by_percentile.items()}
print(f"  JPDF percentile levels -> {level_by_percentile}")
#---

#+++ Plot: 2x3 -- top row single-time snapshot maps (real x,y space), bottom row net-contribution maps
# (vorticity-strain phase space). Every panel gets its own independent color scale and its own compact
# inset colorbar -- embedded via ax.inset_axes(), the exact technique plot7_flux_phase_space.py's own
# add_inset_colorbar() uses -- rather than a separate colorbar axes per panel. That removes the previous
# motivation for sharing a scale between Π_A/Πₖ+Π_A (it existed only to avoid *repeating* a full-size
# colorbar): an inset costs near-zero extra space, so every panel gets one, matching plot7's own choice to
# never share scales between Πₖ/Π_A/total. Axes are shared within each row (all three columns show the same
# physical/phase-space domain), so only the leftmost column keeps y-tick labels; the two rows use genuinely
# different x/y quantities (km vs ζ/f0, σ/|f0|), so there's no sharing *across* rows the way plot7's own two
# same-quantity phase-space rows share.
print("Building figure...")
# height_ratios gives the top row extra height so its panels come out square (matching the real x/y
# domain's own 1:1 aspect) *without* shrinking their width below the bottom row's -- tried set_aspect/
# set_box_aspect on the top-row axes first, but constrained_layout resolves either by shrinking whichever
# dimension it needs to (here: width, not height) to satisfy the ratio, which is the opposite of what's
# wanted. height_ratios instead controls row height directly, leaving each row's width tied purely to the
# (shared) GridSpec columns -- so top and bottom columns stay the same width by construction, and only the
# top row's box grows taller. 1.54 was computed once from this figure's own measured layout (a 3.81in-wide
# column needs 3.81in of height to be square, against a bottom row given 2.48in) -- specific to this
# figsize/panel count, not a portable constant.
fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.54, 1]})

def add_inset_colorbar(ax, im):
    # Thin horizontal colorbar in the panel's upper-right corner, backed by an opaque white patch so its
    # tick labels stay legible regardless of what data sits underneath -- verbatim from
    # plot7_flux_phase_space.py's own add_inset_colorbar(). Forced scientific notation
    # (set_powerlimits((0, 0)), meaning "always", not a real exponent range) keeps every panel's colorbar
    # the same style despite spanning different magnitudes, rather than matplotlib's default
    # ScalarFormatter switching between plain-decimal and offset-multiplier styles per panel.
    x0, y0, w, h = 0.46, 0.85, 0.50, 0.055
    pad = 0.018
    backing = ax.inset_axes([x0 - pad, y0 - 2.6*pad, w + 2*pad, h + 4*pad])
    backing.set_facecolor("white")
    backing.set_xticks([]); backing.set_yticks([])
    for spine in backing.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([x0, y0, w, h])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.locator = MaxNLocator(nbins=3)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar.formatter = fmt
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)
    cbar.ax.xaxis.get_offset_text().set_fontsize(6)
    cbar.set_label("m² s⁻³", fontsize=6.5, labelpad=1)
    return cbar

# --- Top row: snapshot maps ---
top_panels = [("Πₖ", Pi_K_snap), ("Π_A", Pi_A_snap), ("Πₖ+Π_A", Pi_total_snap)]
for col, (name, field) in enumerate(top_panels):
    ax = axes[0, col]
    vmax = np.nanpercentile(np.abs(field.values), args.clim_percentile)
    # set_edgecolor("face") avoids a known pcolormesh rendering artifact -- see plot6_snapshots.py's
    # comment for why linewidth is deliberately left untouched (linewidth=0 does not fix it).
    # rasterized=True: a PDF pcolormesh draws one vector polygon per grid cell -- at this row's full
    # simulation resolution (unlike the bottom row's much coarser --n-bins JPDF grid), that bloated the
    # output to 10s of MB (see plot9/plot10's identical fix). Rasterizing just this Artist embeds it as a
    # single bitmap (at the savefig dpi= below) while titles/axes/text stay vector.
    im = ax.pcolormesh(x_km, y_km, field.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    im.set_edgecolor("face")
    ax.contour(x_km, y_km, b_snap.values, levels=5, colors="0.4", linewidths=0.4)
    add_inset_colorbar(ax, im)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("x [km]")
    if col == 0:
        ax.set_ylabel("y [km]")
    else:
        ax.tick_params(labelleft=False)  # shares the y-axis (same x_km/y_km domain) with the leftmost panel

# --- Bottom row: net-contribution maps in vorticity-strain space ---
def add_sd_lines(ax):
    zmax = zeta_edges[-1]
    z = np.linspace(-zmax, zmax, 200)
    ax.plot(z, np.abs(z), "--", color="gray", lw=1)

_LINESTYLES = ["dotted", "dashed", "solid", "dashdot"]
def add_jpdf_contours(ax):
    # One contour call per level (not a single multi-level call) so each percentile gets its own linestyle
    # -- a single JPDF level from one noisy snapshot often breaks into several disconnected pieces, so a
    # single clabel call would stamp the same percentage text repeatedly across the panel. A shared
    # figure-level legend (below) avoids that clutter entirely.
    for i, lvl in enumerate(contour_levels):
        ax.contour(zeta_centers, sigma_centers, jpdf.T, levels=[lvl], colors="0.25", linewidths=1.0,
                   linestyles=_LINESTYLES[i % len(_LINESTYLES)])

# Net contribution (mean x JPDF), not the conditional mean -- shows how much each vorticity-strain regime
# actually contributes to the domain's total flux (accounting for how common/rare that regime is), rather
# than just how strong the flux is there. Same quantity the SD/AVD/CVD region fractions above are computed
# from.
bottom_panels = [("Πₖ", Pi_K_net), ("Π_A", Pi_A_net), ("Πₖ+Π_A", Pi_total_net)]
for col, (name, net) in enumerate(bottom_panels):
    ax = axes[1, col]
    vmax = np.nanpercentile(np.abs(net), args.clim_percentile)
    im = ax.pcolormesh(zeta_edges, sigma_edges, net.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    im.set_edgecolor("face")
    add_sd_lines(ax)
    add_jpdf_contours(ax)
    add_inset_colorbar(ax, im)
    frac = fractions[name]
    ax.set_title(f"{name}\nSD {frac['SD']:.0f}%  AVD {frac['AVD']:.0f}%  CVD {frac['CVD']:.0f}%", fontsize=10)
    ax.set_xlabel(r"$\bar\zeta / f_0$")
    if col == 0:
        ax.set_ylabel(r"$\bar\sigma / |f_0|$")
    else:
        ax.tick_params(labelleft=False)  # shares the y-axis (same ζ/σ bins) with the leftmost panel

# Row labels in the left margin -- the column titles (just the flux name) plus one row label already say
# what each row is, so repeating a full descriptive title on all six panels would be redundant. Same
# convention as plot7_flux_phase_space.py's own row labels.
# y positions are each row's own vertical center (measured from axes[r,c].get_position(), not guessed) --
# the top row is now taller than the bottom (see the height_ratios comment above), so its center sits
# higher than a naive equal-split guess would put it.
fig.text(-0.02, 0.705, f"snapshot\nt={t_days:.1f}d", rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
fig.text(-0.02, 0.215, f"net contribution\nt={t_min_days_actual:.0f}–{t_max_days_actual:.0f}d", rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")

legend_handles = [Line2D([0], [0], color="0.25", lw=1.2, linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                        label=f"JPDF {level_labels[lvl]} HDR") for i, lvl in enumerate(contour_levels)]
fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 1.04))

fig.suptitle(f"{stem}: ℓ={ℓ_km}km, z={z_sel:.0f}m", fontsize=13)

z_m = int(round(z_sel))
outfile = FIGURES / f"{stem}_flux_snapshot_phase_space_l{ℓ_km}km_z{z_m}m_t{t_days:.0f}d.pdf"
fig.savefig(outfile, dpi=450, bbox_inches="tight")
print(f"Saved: {outfile}")
#---
