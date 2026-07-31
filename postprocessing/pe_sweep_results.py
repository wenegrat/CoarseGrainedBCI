#!/usr/bin/env python
"""Aggregate SFS KE/APE budget residual-vs-dominant-term ratios across a Pe_cell diagonal sweep, at each of
one or more resolutions (see ../submit_pe_sweep.sh), plus a physical-realism guard-rail (max|u,v,w|)."""

#+++ Imports
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
#---

#+++ Configuration
parser = argparse.ArgumentParser(description="Aggregate residual/dominant-term ratios across a Pe_cell sweep")
parser.add_argument("--resolutions", type=str, nargs="+", required=True, help="Space-separated NxYxZx triples (e.g. 256x256x64 192x192x64), matching submit_pe_sweep.sh's RESOLUTIONS")
parser.add_argument("--pe-values", type=float, nargs="+", required=True, help="Pe_cell_h=Pe_cell_v values swept, matching submit_pe_sweep.sh's PE_VALUES")
parser.add_argument("--min-time-days", type=float, default=5.0, help="Exclude time samples before this (in days), matching submit_pe_sweep.sh's exclusion window (default 5.0)")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
SIM_OUTPUT = REPO_ROOT / "output"
t_min = args.min_time_days * 86400
#---

#+++ Residual ratio: |residual(t)| / max_over_other_terms(|term(t)|), not a fixed assumption about which
# term dominates (e.g. Πₖ) -- ε_Kˢ or the tendency could plausibly be larger at some Pe values, especially
# near the over-damped end of the sweep. Reported as both the mean and max over the post-transient window
# (mean = typical closure quality; max = worst-case single snapshot).
def residual_ratio(budget, residual_var, term_vars):
    terms = xr.concat([np.abs(budget[v]) for v in term_vars], dim="term")
    dominant = terms.max("term")
    ratio = np.abs(budget[residual_var]) / dominant
    return float(ratio.mean("time")), float(ratio.max("time"))
#---

#+++ Loop over resolutions x Pe values, computing guard-rail + residual ratios for each combination
rows = []
for res in args.resolutions:
    nx, ny, nz = (int(v) for v in res.lower().split("x"))

    for pe in args.pe_values:
        pe_str = f"{pe:g}"
        stem = f"bci_Nx{nx}_Ny{ny}_Nz{nz}_pe{pe_str}"
        print(f"--- Nx={nx},Ny={ny},Nz={nz}, Pe_cell_h=Pe_cell_v={pe_str} ({stem}) ---")

        sim_file = SIM_OUTPUT / f"{stem}.nc"
        ds = xr.open_dataset(sim_file, decode_times=False, chunks={}).sel(time=slice(t_min, None))
        max_u = float(np.abs(ds.u).max())
        max_v = float(np.abs(ds.v).max())
        max_w = float(np.abs(ds.w).max())
        print(f"  max|u,v,w| = ({max_u:.3g}, {max_v:.3g}, {max_w:.3g}) m/s")

        ke = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_integrated.nc", decode_timedelta=False).sel(time=slice(t_min, None))
        ape = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_integrated.nc", decode_timedelta=False).sel(time=slice(t_min, None))

        ke_terms  = ["∫-∂ₜ SFS KE dV", "∫Π_K dV", "∫-ε_Kˢ dV", "∫(SFS APE->KE) dV"]
        ape_terms = ["∫-∂ₜ SFS APE dV", "∫Π_A dV", "∫-ε_Aˢ dV", "∫(SFS KE->APE) dV", "∫Rˢ dV"]

        row = {"Nx": nx, "Ny": ny, "Nz": nz, "Pe_cell_h=Pe_cell_v": pe, "max|u|": max_u, "max|v|": max_v, "max|w|": max_w}
        for ell in ke.filter_scale.values:
            ell_km = round(float(ell) / 1000)
            mean_r, max_r = residual_ratio(ke.sel(filter_scale=ell), "residual_K", ke_terms)
            row[f"KE resid ratio (ℓ={ell_km}km) mean/max"] = f"{mean_r:.2%}/{max_r:.2%}"
            print(f"  KE  ℓ={ell_km}km: residual ratio mean={mean_r:.1%}, max={max_r:.1%}")
        for ell in ape.filter_scale.values:
            ell_km = round(float(ell) / 1000)
            mean_r, max_r = residual_ratio(ape.sel(filter_scale=ell), "residual_A", ape_terms)
            row[f"APE resid ratio (ℓ={ell_km}km) mean/max"] = f"{mean_r:.2%}/{max_r:.2%}"
            print(f"  APE ℓ={ell_km}km: residual ratio mean={mean_r:.1%}, max={max_r:.1%}")
        rows.append(row)
#---

#+++ Print and save summary table
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n" + "="*70 + "\nSummary\n" + "="*70)
print(df.to_string(index=False))

out_csv = PP_OUTPUT / "pe_sweep_results.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
#---
