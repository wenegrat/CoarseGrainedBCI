#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot cross-scale KE and APE transfer spectra")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
parser.add_argument("--min-time-days", type=float, default=1.0, help="Exclude time samples before this (in days) -- the first ~1-2 saved samples are dominated by a large initial transient unrelated to the ongoing cascade (default 1.0). Applies to both the Hovmöller and the time-averaged spectrum panel.")
parser.add_argument("--max-time-days", type=float, default=None, help="Exclude time samples after this (in days). Defaults to the latest available time (no upper restriction) -- e.g. pass 30 to see what the plots would look like using only the first 30 days of a longer run.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
FIGURES = REPO_ROOT / "figures" / Path(filename).stem  # one subfolder per run, keyed by input filename stem
FIGURES.mkdir(parents=True, exist_ok=True)
#---

#+++ Load energy transfer data
print("Loading energy transfer data...")
input_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
et = xr.open_dataset(input_filename, decode_timedelta=False)

# baroclinic_adjustment.jl's :fields writer uses schedule=ConsecutiveIterations(TimeInterval(...)), which
# writes TWO consecutive model iterations (nominal output time, then the next iteration ~seconds-minutes
# later) at every nominal output time -- see plot5_vorticity_strain_flux.py's comment on the same structure
# for why (04_sfs_ke_budget.py's tendency finite-difference needs a close pair). sweep2_energy_transfer.py
# doesn't compute any tendency term, so it never collapses this pairing -- et's raw time axis is still pairs
# (t, t+ε), not independent samples at the nominal output frequency. Averaging/coloring by both members over
# the --min-time-days-filtered range below would silently double-count near-identical snapshots. Keep only
# the first member of each pair (same ::2 pattern already used for this exact structure elsewhere), on the
# full time axis before any time-based selection so the pair parity stays anchored to the simulation start.
et = et.isel(time=slice(0, None, 2))

# Restrict to the requested [--min-time-days, --max-time-days] window -- default is the full available
# range (no restriction). Applied once, here, to the shared et used by both the Hovmöller and the
# time-averaged spectrum panel below, so --max-time-days genuinely shows "what the plots would look like"
# over a shorter window rather than only affecting the spectrum average while the Hovmöller still shows
# the full run.
t_max_sec = args.max_time_days * 86400 if args.max_time_days is not None else float(et.time.max())
et = et.sel(time=slice(args.min_time_days * 86400, t_max_sec))

# Add 1/ℓ as a non-dimension coordinate so plot.line can use it as the x axis
et = et.assign_coords(inv_scale=("filter_scale", 1.0 / et.filter_scale.values))
et["inv_scale"].attrs = {"long_name": "1/ℓ", "units": "m⁻¹"}
print(f"  Loaded: {input_filename}")
print(f"  Time steps: {len(et.time)}   Filter scales: {len(et.filter_scale)}")
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

#+++ Plot: Hovmöller (time vs. scale) on top, time-averaged spectrum (vs. scale) below -- one column each
# for Πₖ, Π_A, and the ℓ-derivative of the buoyancy conversion term (SFS APE->KE exchange). Πₖ(ℓ)/Π_A(ℓ)/
# the raw conversion term ∫(SFS APE->KE) dV are all *cumulative*-in-scale diagnostics -- their value at a
# given ℓ reflects exchange happening at all scales below ℓ, not the exchange happening *at* scale ℓ.
# d/dℓ recovers that scale-local transfer density (a spectral-transfer-function-style reconstruction from
# a cumulative flux). See postprocessing/S3_plot_sweep.py (an inherited, unadapted KH-era script, not part
# of this pipeline) for the same technique via xr.DataArray.differentiate("filter_scale"), which this
# mirrors -- including keeping 1/ℓ as the x-axis even though the derivative itself is w.r.t. plain ℓ.
fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

# et's own ∫Π_K dV/∫Π_A dV/∫(SFS APE->KE) dV are raw domain integrals of already-specific (per-unit-mass)
# fields, so they come out in m^5 s^-3, not m^2 s^-3 -- not comparable to the raw Πₖ/Π_A/conversion fields
# plotted elsewhere in the pipeline. Same fix as plot3_budgets.py/anim3_panels.py: divide by domain volume
# for a genuine domain-averaged rate in m^2 s^-3.
V_total = et.attrs["Lx"] * et.attrs["Ly"] * et.attrs["Lz"]
print(f"  Domain volume V = {V_total:.4e} m^3 (normalizing ∫Π dV terms to domain-averaged m² s⁻³)")

# d/dℓ divides out an extra factor of length -- m² s⁻³ becomes m s⁻³, no longer comparable to Πₖ/Π_A's own
# units, so this gets its own color scale/vmax below rather than sharing hovmoller_vars'/var_labels' loop.
# Differentiated here, before any time-reduction: differentiation (along filter_scale) and the mean/std
# reduction (along time) used below for the spectrum panel are independent axes and would commute for the
# mean, but NOT for std (a nonlinear, quadratic-based reduction) -- computing the derivative first keeps
# the eventual std band the genuine temporal spread of the exact quantity being plotted.
d_conv = (et["∫(SFS APE->KE) dV"] / V_total).differentiate("filter_scale")

hovmoller_vars = ["∫Π_K dV", "∫Π_A dV"]
vmax = float(max(abs(et[var] / V_total).max() for var in hovmoller_vars))
linthresh = vmax * 1e-3  # scales with the data's own magnitude, rather than a fixed absolute value
for ax, var in zip(axes[0, :2], hovmoller_vars):
    (et[var] / V_total).plot.pcolormesh(x="time", y="filter_scale", ax=ax,
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            norm=plt.matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
                            cbar_kwargs={"label": "m² s⁻³"})
    ax.set_yscale("log")
    ax.set_ylabel("ℓ  [m]")

vmax_conv = float(abs(d_conv).max())
linthresh_conv = vmax_conv * 1e-3
d_conv.plot.pcolormesh(x="time", y="filter_scale", ax=axes[0, 2],
                        cmap="RdBu_r", vmin=-vmax_conv, vmax=vmax_conv,
                        norm=plt.matplotlib.colors.SymLogNorm(linthresh=linthresh_conv, vmin=-vmax_conv, vmax=vmax_conv),
                        cbar_kwargs={"label": "m s⁻³"})
axes[0, 2].set_yscale("log")
axes[0, 2].set_ylabel("ℓ  [m]")

axes[0,0].set_title("KE cross-scale transfer (Hovmöller)")
axes[0,1].set_title("APE cross-scale transfer (Hovmöller)")
axes[0,2].set_title("Buoyancy conversion ℓ-derivative (Hovmöller)")

# Time-averaged spectrum: mean (and ±1 std across time) of Π_K/Π_A vs. 1/ℓ, over the [--min-time-days,
# --max-time-days] window already applied to et above.
var_labels = {"∫Π_K dV": r"$\langle\Pi_K\rangle$", "∫Π_A dV": r"$\langle\Pi_A\rangle$"}
spectrum_titles = ["KE cross-scale transfer spectrum", "APE cross-scale transfer spectrum"]
for ax, var, title in zip(axes[1, :2], hovmoller_vars, spectrum_titles):
    mean = et[var].mean("time") / V_total
    std = et[var].std("time") / V_total
    ax.plot(et.inv_scale, mean, marker="o", color="C0")
    ax.fill_between(et.inv_scale, mean - std, mean + std, color="C0", alpha=0.25)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(inv_Ld, color="k", ls="--", lw=1.2, label=f"$L_d$={Ld/1e3:.1f}km")
    ax.legend(fontsize=9, loc="best")
    ax.set_xscale("log")
    ax.set_xlabel(r"$1/\ell$  [m$^{-1}$]")
    ax.set_ylabel(f"{var_labels[var]}  [m² s⁻³]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

ax = axes[1, 2]
mean_conv = d_conv.mean("time")
std_conv = d_conv.std("time")
ax.plot(et.inv_scale, mean_conv, marker="o", color="C0")
ax.fill_between(et.inv_scale, mean_conv - std_conv, mean_conv + std_conv, color="C0", alpha=0.25)
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(inv_Ld, color="k", ls="--", lw=1.2, label=f"$L_d$={Ld/1e3:.1f}km")
ax.legend(fontsize=9, loc="best")
ax.set_xscale("log")
ax.set_xlabel(r"$1/\ell$  [m$^{-1}$]")
ax.set_ylabel(r"$\partial_\ell\langle E_A^s\to E_K^s\rangle$  [m s⁻³]")
ax.set_title("Buoyancy conversion ℓ-derivative spectrum")
ax.grid(True, alpha=0.3)

label = run_label(et.attrs)
if label:
    fig.suptitle(label, fontsize=11)

plot_filename = str(FIGURES / os.path.basename(input_filename).replace(".nc", ".png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---
