#!/usr/bin/env python
"""
Independent sanity check on ε_Aˢ: compare it against the TOTAL (unfiltered, grid-resolved) APE
dissipation rate χ_APE = ∫[κh(∂ρ/∂x·∂Υ/∂x + ∂ρ/∂y·∂Υ/∂y) + κv(∂ρ/∂z·∂Υ/∂z)] dV, using the *unfiltered*
ρ field directly -- same formula as ε_Aˢ's "term1" but with no filter applied. This is the ℓ→∞
limit of ε_Aˢ (as the filter widens, ρ̄→const, ∇Υˡ→0, so ε_Aˢ→ this domain-total quantity), so ε_Aˢ(ℓ)
should stay below χ_APE and increase monotonically with ℓ toward it.

∇Υ is reconstructed analytically from D=dz*/dρ and ∇ρ (the same method ε_Aˢ itself now uses -- see
analytic_grad_upsilon()), NOT by differentiating the saved Υ field directly, so this is a fair
apples-to-apples bound: both sides of the comparison go through the same gradient-reconstruction method.

Note: this uses κ∇ρ·∇Υ (an APE dissipation rate, units of power), NOT κ|∇ρ|² (a density-variance
dissipation rate, different units entirely) -- the two are not interchangeable.
"""
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
from src.aux00_utils import load_dataset_and_grid, integrate, calculate_gradient
from src.aux01_pe_functions import analytic_grad_upsilon
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare epsilon_A^s against total unfiltered APE dissipation rate")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--min-time-days", type=float, default=1.0, help="Exclude initial transient (days)")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
print("\n" + "="*70 + f"\n  inv06_total_mixing_check.py  filename={args.filename}\n" + "="*70)
#---

#+++ Load raw grid (for dV) and the saved full (unfiltered) rho, D fields
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})

fields_filename = PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc"
ds_fields = xr.open_dataset(fields_filename, decode_timedelta=False).chunk({"time": 1})
rho = ds_fields["ρ"]
D = ds_fields["D"]

κh, κv = ds.attrs["nu_h"] / ds.attrs["Pr"], ds.attrs["nu_v"] / ds.attrs["Pr"]
print(f"κh={κh:.4g} m²/s   κv={κv:.4g} m²/s")
#---

#+++ Compute chi_APE = kh*(rho_x*ups_x + rho_y*ups_y) + kv*rho_z*ups_z, integrated over volume -- no filter
grad_rho = calculate_gradient(rho)      # dims (i, time, z, y, x), i=1,2,3 = x,y,z
grad_ups = analytic_grad_upsilon(grad_rho, D)
horiz = (grad_rho.sel(i=[1, 2]) * grad_ups.sel(i=[1, 2])).sum("i")
vert  = grad_rho.sel(i=3) * grad_ups.sel(i=3)
chi_local = κh * horiz + κv * vert
chi_total_t = integrate(chi_local, ds["dV"])
#---

#+++ Apply the same transient exclusion as the pytest budget tests, then compare
chi_total_t = chi_total_t.sel(time=slice(args.min_time_days * 86400, None))
chi_total_rms = float(np.sqrt((chi_total_t**2).mean()))
print(f"\nχ_APE = ∫ κ∇ρ·∇Υ dV  (unfiltered -- the ℓ→∞ limit of ε_Aˢ)")
print(f"  rms(χ_APE) over t > {args.min_time_days} days: {chi_total_rms:.4e}")

eps_filename = PP_OUTPUT / f"{stem}_sfs_ape_budget_integrated.nc"
if eps_filename.exists():
    ds_eps = xr.open_dataset(eps_filename, decode_timedelta=False)
    ds_eps = ds_eps.sel(time=slice(args.min_time_days * 86400, None))
    print(f"\n  {'filter_scale':<15} {'rms(ε_Aˢ)':<15} {'ratio to χ_total':<20}")
    for ell in ds_eps.filter_scale.values:
        eps = ds_eps["∫-ε_Aˢ dV"].sel(filter_scale=ell)
        eps_rms = float(np.sqrt((eps**2).mean()))
        print(f"  {ell:<15.0f} {eps_rms:<15.4e} {eps_rms/chi_total_rms:<20.3%}")
else:
    print(f"  (no {eps_filename} found -- run 05_sfs_ape_budget.py first)")
#---
