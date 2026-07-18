#!/usr/bin/env python
"""Animate surface buoyancy/vorticity alongside cross-scale KE/APE flux maps and SFS budget time series."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Animate surface fields, cross-scale fluxes, and SFS budget time series")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scale", type=float, default=None, help="Target filter length scale in meters (nearest available will be used; defaults to the smallest available)")
parser.add_argument("--fps", type=int, default=6, help="Frames per second")
parser.add_argument("--dpi", type=int, default=100, help="DPI for output GIF")
parser.add_argument("--clim-percentile", type=float, default=99.0, help="Percentile of |data| used to set symmetric color limits")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
ANIMATIONS = REPO_ROOT / "animations"
ANIMATIONS.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Orientation fix: some offline APE-pipeline fields (Π_A, the KE<->APE exchange term) are stored with
# dims (..., x, y) instead of (..., y, x) like every other field -- plotting field.values directly against
# (x_km, y_km) then renders them rotated relative to b/ζ/Πₖ. Always transpose to (..., y_dim, x_dim) before
# plotting so this can't recur regardless of a given variable's stored dim order.
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

#+++ Open datasets and pick the filter scale
print("Opening datasets...")
surf = xr.open_dataset(filename.replace(".nc", "_surface.nc"), decode_times=False)
ke_fields  = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields.nc",  decode_times=False, chunks={})
ape_fields = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc", decode_times=False, chunks={})

ℓ_target = args.filter_scale if args.filter_scale is not None else float(ke_fields.filter_scale.min())
ℓ = float(ke_fields.filter_scale.sel(filter_scale=ℓ_target, method="nearest"))
ℓ_km = int(round(ℓ / 1000))
print(f"Filter scale: ℓ = {ℓ:.4f} m ({ℓ_km} km)")

x_km = surf.x_caa.values / 1e3
y_km = surf.y_aca.values / 1e3
f_cor = coriolis_f(surf.attrs)
#---

#+++ Load surface fields (light: b, ζ, Πₖ are all written directly to the surface writer)
print("Loading surface fields...")
b = fix_orientation(surf["b"].squeeze()).load()
zeta_norm = fix_orientation(surf["ζ"].squeeze()).load() / f_cor
t_surf_days = surf.time.values / 86400
#---

#+++ Load near-surface (top z-level) offline fields: Πₖ, Π_A, and the SFS APE->KE "conversion" term.
# All three sourced from the same offline fields files (not the online surface writer) so they share an
# identical time/z grid -- this matters for the Πₖ+Π_A "total flux" panel, a simple sum with no
# interpolation/nearest-match needed between the two.
print("Loading Πₖ, Π_A, and conversion fields (top z-level slice)...")
Pi_K_top = fix_orientation(ke_fields["Π_K"].sel(filter_scale=ℓ, method="nearest").isel(z_aac=-1)).load()
Pi_A_top = fix_orientation(ape_fields["Π_A"].sel(filter_scale=ℓ, method="nearest").isel(z_aac=-1)).load()
conv_top = fix_orientation(ke_fields["SFS APE->KE exchange"].sel(filter_scale=ℓ, method="nearest").isel(z_aac=-1)).load()
Pi_total = Pi_K_top + Pi_A_top
t_fields_days = ke_fields.time.values / 86400
n_frames = len(t_fields_days)
print(f"  Done. {n_frames} field frames (this drives the animation)")
#---

#+++ Load KE and APE budget integrated timeseries
print("Loading KE/APE budget timeseries...")
ke_int  = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_integrated.nc",  decode_times=False).sel(filter_scale=ℓ, method="nearest")
ape_int = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_integrated.nc", decode_times=False).sel(filter_scale=ℓ, method="nearest")
t_budget_days = ke_int.time.values / 86400
#---

def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))

def clim(field):
    return float(np.nanpercentile(np.abs(field.values), args.clim_percentile))

#+++ Build figure: 2 rows of 3 maps, then a full-width bottom row split into KE/APE budget panels.
# constrained_layout fights with equal-aspect square map axes mixed into the same GridSpec as the wide
# (non-square) budget row -- it silently fails ("axes sizes collapsed to zero") and produces uneven gaps.
# Explicit wspace/hspace plus fixed-fraction colorbars avoid the layout solver entirely.
print("Building figure...")
vmax_b, vmax_z = clim(b), clim(zeta_norm)
vmax_pik, vmax_pia, vmax_conv, vmax_tot = clim(Pi_K_top), clim(Pi_A_top), clim(conv_top), clim(Pi_total)

fig = plt.figure(figsize=(17, 12))
gs = fig.add_gridspec(3, 6, wspace=1.0, hspace=0.45, height_ratios=[1, 1, 0.8],
                       left=0.05, right=0.97, top=0.94, bottom=0.06)
ax_b, ax_z, ax_conv = fig.add_subplot(gs[0,0:2]), fig.add_subplot(gs[0,2:4]), fig.add_subplot(gs[0,4:6])
ax_pik, ax_pia, ax_tot = fig.add_subplot(gs[1,0:2]), fig.add_subplot(gs[1,2:4]), fig.add_subplot(gs[1,4:6])
ax_ts_ke, ax_ts_ape = fig.add_subplot(gs[2,0:3]), fig.add_subplot(gs[2,3:6])

def setup_map(ax, data0, vmax, title):
    im = ax.pcolormesh(x_km, y_km, data0, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

im_b    = setup_map(ax_b,    b.isel(time=0).values,         vmax_b,    "surface buoyancy b")
im_z    = setup_map(ax_z,    zeta_norm.isel(time=0).values, vmax_z,    "surface Rossby number ζ/f")
im_conv = setup_map(ax_conv, conv_top.isel(time=0).values,  vmax_conv, f"conversion (SFS APE→KE, ℓ={ℓ_km}km)")
im_pik  = setup_map(ax_pik,  Pi_K_top.isel(time=0).values,  vmax_pik,  f"cross-scale KE flux Πₖ (ℓ={ℓ_km}km)")
im_pia  = setup_map(ax_pia,  Pi_A_top.isel(time=0).values,  vmax_pia,  f"cross-scale APE flux Π_A (ℓ={ℓ_km}km)")
im_tot  = setup_map(ax_tot,  Pi_total.isel(time=0).values,  vmax_tot,  f"total cross-scale flux Πₖ+Π_A (ℓ={ℓ_km}km)")

ke_terms = [
    ("∫-∂ₜ SFS KE dV",    "C0", r"$\partial_t E_K^s$"),
    ("∫Π_K dV",           "C1", r"$\Pi_K$"),
    ("∫-ε_Kˢ dV",         "C2", r"$-\varepsilon_K^s$"),
    ("∫(SFS APE->KE) dV", "C3", r"$E_A^s \to E_K^s$"),
]
# --bottom_drag (baroclinic_adjustment.jl) only: SFS term is folded into residual_K already
# (04_sfs_ke_budget.py) and shown here for visibility; LS term is a standalone diagnostic, not part of
# any budget/residual sum here.
if "∫-(bottom drag work, SFS) dA" in ke_int.data_vars:
    ke_terms.append(("∫-(bottom drag work, SFS) dA", "C5",   r"$-\tau u_b^{s}$"))
if "∫-(bottom drag work, LS) dA" in ke_int.data_vars:
    ke_terms.append(("∫-(bottom drag work, LS) dA",  "gray", r"$-\bar\tau\cdot\bar u_b$"))
ape_terms = [
    ("∫-∂ₜ SFS APE dV",   "C0", r"$\partial_t E_A^s$"),
    ("∫Π_A dV",           "C1", r"$\Pi_A$"),
    ("∫-ε_Aˢ dV",         "C2", r"$-\varepsilon_A^s$"),
    ("∫(SFS KE->APE) dV", "C3", r"$E_K^s \to E_A^s$"),
    ("∫Rˢ dV",            "C4", r"$R^s$"),
]

for ax, ds_int, terms, resid_var, title in [
    (ax_ts_ke,  ke_int,  ke_terms,  "residual_K", "SFS KE budget (volume-integrated)"),
    (ax_ts_ape, ape_int, ape_terms, "residual_A", "SFS APE budget (volume-integrated)"),
]:
    for var, color, label in terms:
        ax.plot(t_budget_days, ds_int[var].values, color=color, lw=1.5, label=label)
    ax.plot(t_budget_days, ds_int[resid_var].values, color="k", ls="--", lw=1.0, label="residual")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_xlabel("Time (days)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)

marker_ke  = ax_ts_ke.axvline(t_budget_days[0], color="k", lw=1.2)
marker_ape = ax_ts_ape.axvline(t_budget_days[0], color="k", lw=1.2)

suptitle = fig.suptitle(f"{stem}: t={t_fields_days[0]:.2f} days", fontsize=13)
#---

#+++ Animate (driven by the offline fields' time grid; b/ζ nearest-matched onto it)
def update(frame_i):
    t_day = t_fields_days[frame_i]
    si = nearest_idx(t_surf_days, t_day)
    im_b.set_array(b.isel(time=si).values.ravel())
    im_z.set_array(zeta_norm.isel(time=si).values.ravel())

    im_pik.set_array(Pi_K_top.isel(time=frame_i).values.ravel())
    im_pia.set_array(Pi_A_top.isel(time=frame_i).values.ravel())
    im_conv.set_array(conv_top.isel(time=frame_i).values.ravel())
    im_tot.set_array(Pi_total.isel(time=frame_i).values.ravel())

    marker_ke.set_xdata([t_day, t_day])
    marker_ape.set_xdata([t_day, t_day])
    suptitle.set_text(f"{stem}: t={t_day:.2f} days")
    return im_b, im_z, im_pik, im_pia, im_conv, im_tot, marker_ke, marker_ape

print(f"Building animation over {n_frames} frames...")
anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
outfile = str(ANIMATIONS / f"{stem}_panels_l{ℓ_km}km.gif")
anim.save(outfile, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
print(f"Animation saved to: {outfile}")
#---
