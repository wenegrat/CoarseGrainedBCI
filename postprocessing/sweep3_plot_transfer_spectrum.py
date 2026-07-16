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
parser.add_argument("--min-time-days", type=float, default=1.0, help="Exclude time samples before this (in days) from the time-averaged spectrum panel -- the first ~1-2 saved samples are dominated by a large initial transient unrelated to the ongoing cascade (default 1.0)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
#---

#+++ Load energy transfer data
print("Loading energy transfer data...")
input_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
et = xr.open_dataset(input_filename, decode_timedelta=False)

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

#+++ Plot: Hovmöller (time vs. scale) on top, time-averaged spectrum (vs. scale) below
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

vmax = float(max(abs(et["∫Π_K dV"]).max(), abs(et["∫Π_A dV"]).max()))
linthresh = vmax * 1e-3  # scales with the data's own magnitude, rather than a fixed absolute value
for ax, var in zip(axes[0], ["∫Π_K dV", "∫Π_A dV"]):
    et[var].plot.pcolormesh(x="time", y="filter_scale", ax=ax,
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            norm=plt.matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax))
    ax.set_yscale("log")
    ax.set_ylabel("ℓ  [m]")

axes[0,0].set_title("KE cross-scale transfer (Hovmöller)")
axes[0,1].set_title("APE cross-scale transfer (Hovmöller)")

# Time-averaged spectrum: mean (and ±1 std across time) of Π_K/Π_A vs. 1/ℓ, excluding the initial transient
et_avg = et.sel(time=slice(args.min_time_days * 86400, None))
for ax, var, title in zip(axes[1], ["∫Π_K dV", "∫Π_A dV"], ["KE cross-scale transfer spectrum", "APE cross-scale transfer spectrum"]):
    mean = et_avg[var].mean("time")
    std = et_avg[var].std("time")
    ax.plot(et_avg.inv_scale, mean, marker="o", color="C0")
    ax.fill_between(et_avg.inv_scale, mean - std, mean + std, color="C0", alpha=0.25)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(inv_Ld, color="k", ls="--", lw=1.2, label=f"$L_d$={Ld/1e3:.1f}km")
    ax.legend(fontsize=9, loc="best")
    ax.set_xscale("log")
    ax.set_xlabel(r"$1/\ell$  [m$^{-1}$]")
    ax.set_ylabel(var)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

label = run_label(et.attrs)
if label:
    fig.suptitle(label, fontsize=11)

plot_filename = str(REPO_ROOT / "figures" / os.path.basename(input_filename).replace(".nc", ".png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---
