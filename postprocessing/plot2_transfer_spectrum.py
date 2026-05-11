#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from src.aux00_utils import load_dataset_and_grid, integrate, condense_uw_velocities
from src.aux01_pe_functions import (calculate_density_fields_from_buoyancy,
                                    sorted_timeseries, calculate_b_r)
from src.aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot cross-scale KE and APE transfer spectra")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
def str2bool(s):
    if s.lower() in ("true", "1", "yes"):  return True
    if s.lower() in ("false", "0", "no"):  return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {s!r}")
parser.add_argument("--time", type=float, default=40, help="Snapshot time (nearest available will be used; ignored if --time-average true)")
parser.add_argument("--time-average", type=str2bool, default=True, metavar="BOOL", help="Average transfer terms over the whole time range (true/false)")
parser.add_argument("--max-average-time", type=float, default=140.0, help="Latest time included when averaging (only used if --time-average true)")
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
et = xr.open_dataset(input_filename, decode_timedelta=False).chunk(dict(time=1))
if args.time_average:
    et = et.sel(time=slice(None, args.max_average_time))
    t0, t1 = float(et.time.min()), float(et.time.max())
    et = et.mean("time", keep_attrs=True)
    time_label = f"$t \\in [{t0:.0f}, {t1:.0f}]$"
else:
    et = et.sel(time=args.time, method="nearest")
    time_label = f"$t = {float(et.time.values):.0f}$"

# Add 1/ℓ as a non-dimension coordinate so plot.line can use it as the x axis
et = et.assign_coords(inv_scale=("filter_scale", 1.0 / et.filter_scale.values))
et["inv_scale"].attrs = {"long_name": "Inverse of filter scale 1/ℓ",}
print(f"  Loaded: {input_filename}")
print(f"  Filter scales: {et.filter_scale.values}")
#---

#+++ Direct ∫w·b_r dV (no filter — for comparison with the SFS+resolved sum)
# The SFS exchange + resolved conversion sum to filter(w·b_r). Volume-integrated,
# this should equal the unfiltered ∫w·b_r dV (for filters that conserve integrals).
print("\nComputing direct ∫w·b_r dV from raw fields for comparison...")
ds_raw = load_dataset_and_grid(filename).chunk(dict(time=1))
ds_raw = ds_raw.reindex(time=et.time if not args.time_average else ds_raw.time)
ds_raw = condense_uw_velocities(ds_raw, indices=(1, 3))
ds_raw = calculate_density_fields_from_buoyancy(ds_raw, buoyancy_name="b", density_name="ρ")

sorted_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
if os.path.exists(sorted_filename):
    print(f"  Loading sorted density from: {sorted_filename}")
    ds_sorted = xr.open_dataset(sorted_filename, decode_times=False).chunk(dict(time=1))
    rho_sorted = ds_sorted.rho_sorted
else:
    print(f"  Sorted density file not found; computing on the fly (slow)...")
    _s = sorted_timeseries(ds_raw, field_to_sort="ρ", n_workers=4)
    rho_sorted = _s.rho_sorted

b_r_full = calculate_b_r(ds_raw["ρ"], rho_sorted)
w_full = ds_raw["uᵢ"].sel(i=3)
int_w_b_r = integrate(w_full * b_r_full, ds_raw.dV)
if args.time_average:
    int_w_b_r = int_w_b_r.sel(time=slice(None, args.max_average_time)).mean("time")
else:
    int_w_b_r = int_w_b_r.sel(time=args.time, method="nearest")
int_w_b_r_val = float(int_w_b_r.compute())

# Compare with sum (SFS + resolved) at each filter scale
sum_vals = (et["∫(SFS APE->KE) dV"] + et["∫w̄·b̄ᵣ dV"]).values
print("\nComparison: ∫w·b_r dV   vs   ∫(SFS exchange) dV + ∫w̄·b̄ᵣ dV")
print(f"  Direct ∫w·b_r dV       = {int_w_b_r_val:+.6e}")
for ℓ, s in zip(et.filter_scale.values, sum_vals):
    rel_err = abs(s - int_w_b_r_val) / max(abs(int_w_b_r_val), 1e-30)
    print(f"  ℓ={ℓ:6.3f}: sum = {s:+.6e}   |Δ|/|direct| = {rel_err:.3e}")
#---

#+++ Plot
fig, (ax, ax_b, ax_c) = plt.subplots(3, 1, figsize=(6, 10), constrained_layout=True, sharex=True)

for var, color, label_str in [
    ("∫Π_K dV",           "#2166ac", r"$\Pi_K$"),
    ("∫Π_A dV",           "#d6604d", r"$\Pi_A$"),
    ("∫(SFS APE->KE) dV", "#1b7837", r"SFS APE$\to$KE: $\overline{w\,b_r} - \bar{w}\,\bar{b}_r$"),
    ("∫w̄·b̄ᵣ dV",          "#762a83", r"Resolved conversion: $\bar{w}\,\bar{b}_r$"),
]:
    ax.plot(et.inv_scale, et[var].values, color=color, label=label_str)
ax.axhline(0, color="k", lw=0.8, ls="--")
for ℓ in [1, 7]:
    ax.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1e-2)
ax.grid(True, alpha=0.3)
ax.legend()
ax2 = ax.secondary_xaxis("top", functions=(lambda x: 1/x, lambda x: 1/x))
ax2.set_xlabel("Filter scale ℓ")

info_parts = []
label = run_label(et.attrs)
if label:
    info_parts.append(label)
info_parts.append(time_label)
ax.text(0.98, 0.04, ",  ".join(info_parts), transform=ax.transAxes, fontsize=10, ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.8))

#+++ Bottom panel: Π_K, Π_A, and d/dℓ of both APE->KE exchange rates
d_sfs   = et["∫(SFS APE->KE) dV"].differentiate("filter_scale")
d_resol = et["∫w̄·b̄ᵣ dV"].differentiate("filter_scale")
ax_b.plot(et.inv_scale, et["∫Π_K dV"].values, color="#2166ac", label=r"$\Pi_K$")
ax_b.plot(et.inv_scale, et["∫Π_A dV"].values, color="#d6604d", label=r"$\Pi_A$")
ax_b.plot(et.inv_scale, d_sfs.values,   color="#1b7837", label=r"$\partial_\ell$ SFS APE$\to$KE")
ax_b.plot(et.inv_scale, d_resol.values, color="#762a83", label=r"$\partial_\ell\,\bar{w}\,\bar{b}_r$")
ax_b.axhline(0, color="k", lw=0.8, ls="--")
for ℓ in [1, 7]:
    ax_b.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--")
ax_b.set_xscale("log")
ax_b.set_yscale("symlog", linthresh=1e-2)
ax_b.grid(True, alpha=0.3)
ax_b.legend()
#---

#+++ Third panel: Π_K, Π_A, and d/dℓ of total (SFS + resolved) APE->KE exchange
total_exchange = et["∫(SFS APE->KE) dV"] + et["∫w̄·b̄ᵣ dV"]
d_total = total_exchange.differentiate("filter_scale")
ax_c.plot(et.inv_scale, et["∫Π_K dV"].values, color="#2166ac", label=r"$\Pi_K$")
ax_c.plot(et.inv_scale, et["∫Π_A dV"].values, color="#d6604d", label=r"$\Pi_A$")
ax_c.plot(et.inv_scale, d_total.values, color="#000000",
          label=r"$\partial_\ell\left[\,\overline{w\,b_r}\,\right]$ "
                r"(total APE$\to$KE exchange = SFS + resolved)")
ax_c.axhline(0, color="k", lw=0.8, ls="--")
for ℓ in [1, 7]:
    ax_c.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--")
ax_c.set_xscale("log")
ax_c.set_yscale("symlog", linthresh=1e-2)
ax_c.grid(True, alpha=0.3)
ax_c.set_xlabel("Inverse of filter scale 1/ℓ")
ax_c.legend()
#---

plot_filename = str(REPO_ROOT / "figures" / os.path.basename(input_filename).replace("energy_transfer_sweep", "cross-scale_transfer_spectrum").replace(".nc", ".png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---

#+++ Separate figure: comparison of total APE->KE exchange via two routes
fig2, axc = plt.subplots(figsize=(6, 4), constrained_layout=True)
axc.plot(et.inv_scale, sum_vals, color="#1b7837", marker="o", lw=1.5,
         label=r"SFS + resolved: $\int(\overline{w\,b_r} - \bar w\,\bar b_r)\,dV + \int \bar w\,\bar b_r\,dV$")
axc.axhline(int_w_b_r_val, color="#762a83", lw=1.5, ls="--",
            label=r"Direct (no filter): $\int w\,b_r\,dV$")
axc.axhline(0, color="k", lw=0.8, ls=":")
for ℓ in [1, 7]:
    axc.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--", alpha=0.4)
axc.set_xscale("log")
axc.set_xlabel("Inverse of filter scale 1/ℓ")
axc.set_ylabel(r"Total APE$\to$KE exchange  $[\mathrm{m}^4\,\mathrm{s}^{-3}]$")
axc.set_title("Consistency check: total APE→KE exchange,\nfilter decomposition vs. direct integral")
axc.grid(True, alpha=0.3)
axc.legend(loc="best", fontsize=9)
axc2 = axc.secondary_xaxis("top", functions=(lambda x: 1/x, lambda x: 1/x))
axc2.set_xlabel("Filter scale ℓ")

info_parts2 = []
if label:
    info_parts2.append(label)
info_parts2.append(time_label)
axc.text(0.98, 0.04, ",  ".join(info_parts2), transform=axc.transAxes, fontsize=10,
         ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.8))

cmp_filename = str(REPO_ROOT / "figures" / os.path.basename(input_filename).replace("energy_transfer_sweep", "exchange_consistency_check").replace(".nc", ".png"))
fig2.savefig(cmp_filename, dpi=150, bbox_inches="tight")
print(f"Comparison plot saved to: {cmp_filename}")
#---
