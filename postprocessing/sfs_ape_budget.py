#!/usr/bin/env python
"""
Calculate SFS APE budget from Kelvin-Helmholtz simulation output
"""

#+++ Imports
import os
import time
import numpy as np
import xarray as xr
import gcm_filters
from aux00_utils import load_dataset_and_grid, condense_velocities, integrate, DaskParallelFilter
from aux01_pe_functions import (
    calculate_density_fields_from_buoyancy,
    local_potential_energies_timeseries,
    calculate_sfs_ape_tendency,
    calculate_sfs_R_correction,
    calculate_cross_scale_ape_flux,
    calculate_sfs_ape_dissipation,
    calculate_ape_to_ke_exchange_term,
)
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS APE budget from Kelvin-Helmholtz simulation output")
parser.add_argument("--filename", default="output/kelvin_helmholtz_instability_128x1x256.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18,
                    help="Number of CPU workers for parallel filtering / APE sorting")
args = parser.parse_args()
filename = args.filename
filter_length_scale = 0.8  # Length scale for filtering
n_workers = args.n_workers
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
ds = load_dataset_and_grid(filename)
print(f"Dataset loaded: {len(ds.time)} time steps  ({time.time()-t0:.1f}s)")

# ds = ds.sel(time=slice(20, 81))
#---

#+++ Filter buoyancy field
print("\n" + "="*60)
print("Filtering buoyancy field...")

filtered_dimensions = ["x_caa", "y_aca"]
filter_scale = filter_length_scale * np.sqrt(12)
gaussian_filter = DaskParallelFilter(gcm_filters.Filter(
    filter_scale=filter_scale,
    dx_min=float(min(ds.Δx_caa.min(), ds.Δy_aca.min())),
    filter_shape=gcm_filters.FilterShape.GAUSSIAN,
    grid_type=gcm_filters.GridType.REGULAR,
), n_workers=n_workers)

t0 = time.time()
ds["b̄"] = gaussian_filter.apply(ds.b, dims=filtered_dimensions) # An overbar denotes a filtering operation
print(f"  b̄ filtered  ({time.time()-t0:.1f}s)")

ds = condense_velocities(ds, indices=[1, 2, 3]) # Condense velocity components into tensor form
t0 = time.time()
ds["ūᵢ"] = gaussian_filter.apply(ds["uᵢ"], dims=filtered_dimensions)
print(f"  ūᵢ filtered  ({time.time()-t0:.1f}s)")

ds_filt = ds[["b̄", "dV", "LxLy", "ūᵢ"]].copy()
ds_full = ds[["b", "dV", "LxLy", "uᵢ"]].copy()
#---

#+++ Calculate density fields
print("\n" + "="*60)
print("Calculating density fields...")
t0 = time.time()
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
ds_filt = calculate_density_fields_from_buoyancy(ds_filt, buoyancy_name="b̄", density_name="ρ̄")
print(f"Density fields calculated: ρ, Z, ρ̄  ({time.time()-t0:.1f}s)")
#---

#+++ Calculate local APE using precomputed_integral method
print("\n" + "="*60)
print("Calculating local APE...")

t0 = time.time()
full_local_pes = local_potential_energies_timeseries(ds_full, density_name="ρ", rho_to_sort=ds_full.ρ, ape_method="precomputed_integral", use_numpy_version=True, n_workers=n_workers)
print(f"  full_local_pes  ({time.time()-t0:.1f}s)")

t0 = time.time()
filt_local_pes = local_potential_energies_timeseries(ds_filt, density_name="ρ̄", rho_to_sort=ds_full.ρ, ape_method="precomputed_integral", use_numpy_version=True, n_workers=n_workers)
print(f"  filt_local_pes  ({time.time()-t0:.1f}s)")
#---

#+++ Filter local APE
print("\n" + "="*60)
print("Filtering local APE...")

t0 = time.time()
full_local_ape_filtered = gaussian_filter.apply(full_local_pes.ape, dims=filtered_dimensions)
subfilter_local_ape = full_local_ape_filtered - filt_local_pes.ape
print(f"Local APE filtered with length scale: {filter_length_scale}  ({time.time()-t0:.1f}s)")
#---

#+++ Calculate budget terms
print("\n" + "="*60)
print("Calculating budget terms...")

t0 = time.time()
cross_scale_ape_flux = calculate_cross_scale_ape_flux(ds_full.ρ, ds_full["uᵢ"], filt_local_pes.upsilon, gaussian_filter,
    filter_dims=filtered_dimensions,
    filtered_density=ds_filt.ρ̄,
    filtered_velocity_vector=ds_filt["ūᵢ"],)
print(f"  cross_scale_ape_flux  ({time.time()-t0:.1f}s)")

t0 = time.time()
sfs_ape_dissipation = calculate_sfs_ape_dissipation(ds_full.ρ, full_local_pes.upsilon, filt_local_pes.upsilon, ds.κ, gaussian_filter,
    filter_dims=filtered_dimensions,
    filtered_density=ds_filt.ρ̄,)
print(f"  sfs_ape_dissipation  ({time.time()-t0:.1f}s)")

t0 = time.time()
ape_to_ke_exchange = calculate_ape_to_ke_exchange_term(
    ds_full["uᵢ"].sel(i=3),   # full w
    ds_full.b,                # full buoyancy
    gaussian_filter,
    filter_dims=filtered_dimensions,
    filtered_w=ds_filt["ūᵢ"].sel(i=3),
    filtered_b=ds_filt["b̄"],)
print(f"  ape_to_ke_exchange  ({time.time()-t0:.1f}s)")

t0 = time.time()
R_s = calculate_sfs_R_correction(full_local_pes.rho_sorted, full_local_pes.z0, filt_local_pes.z0,
                                  full_local_pes.dz_sorted, gaussian_filter, filter_dims=filtered_dimensions)
print(f"  R_s  ({time.time()-t0:.1f}s)")
#---

#+++ Calculate SFS APE time derivatives
t0 = time.time()
dAPE_dt = calculate_sfs_ape_tendency(subfilter_local_ape)
print(f"  dAPE_dt  ({time.time()-t0:.1f}s)")
#---

#+++ Integrate and budget
print("\n" + "="*60)
print("Integrating SFS APE budget terms...")
t0 = time.time()

dV = ds_full.dV
int_dAPE_dt = integrate(dAPE_dt, dV)

int_sfs_ape_dissipation = integrate(sfs_ape_dissipation.reindex(time=dAPE_dt.time), dV)
int_cross_scale_ape_flux = integrate(cross_scale_ape_flux.reindex(time=dAPE_dt.time), dV)
int_ape_to_ke_exchange = integrate(ape_to_ke_exchange.reindex(time=dAPE_dt.time), dV)
int_R_s = integrate(R_s.reindex(time=dAPE_dt.time), dV)

residual = -int_dAPE_dt - int_ape_to_ke_exchange + int_cross_scale_ape_flux - int_sfs_ape_dissipation + int_R_s
print(f"Integration done  ({time.time()-t0:.1f}s)")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")
t0 = time.time()

sfs_ape_budget_terms = xr.Dataset({
    # Density fields
    "ρ": ds_full.ρ,
    "ρ̄": ds_filt.ρ̄,
    # Reference heights
    "z₀(ρ)": full_local_pes.z0,
    "z₀(ρ̄)": filt_local_pes.z0,
    # Buoyancy displacement potentials
    "Υ": full_local_pes.upsilon,
    "Υˡ": filt_local_pes.upsilon,
    # Local APE fields
    "Ea(ρ, z)": full_local_pes.ape,
    "Ea(ρ̄, z)": filt_local_pes.ape,
    "Ēa(ρ, z)": full_local_ape_filtered,
    "Eaˢ(ρ, z)": subfilter_local_ape,
    # Local budget terms
    "∂ₜ SFS APE": dAPE_dt,
    "Π_APE": cross_scale_ape_flux,
    "χₛ": sfs_ape_dissipation,
    "SFS KE->APE exchange": ape_to_ke_exchange,
    "Rˢ": R_s,
    # Integrated budget terms
    "∫-∂ₜ SFS APE dV": -int_dAPE_dt,
    "∫Π_APE dV": int_cross_scale_ape_flux,
    "∫-χₛ dV": -int_sfs_ape_dissipation,
    "∫(SFS KE->APE) dV": -int_ape_to_ke_exchange, # Flip the sign to make plotting easier
    "∫Rˢ dV": int_R_s,
    "residual_APE": residual,
})

output_filename = filename.replace(".nc", "_sfs_ape_budget.nc")
sfs_ape_budget_terms.to_netcdf(output_filename)
print(f"Results saved to: {output_filename}  ({time.time()-t0:.1f}s)")
#---

#+++ Plot integrated budget terms
print("\n" + "="*60)
print("Creating plots...")
t0 = time.time()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Colors shared with sfs_ke_budget.py — keep analogous terms the same colour
budget_colors = {
    "tendency":   "C0",
    "flux":       "C1",
    "dissipation":"C2",
    "exchange":   "C3",
    "residual":   "k",
}
integrated_vars = {
    "∫-∂ₜ SFS APE dV":    budget_colors["tendency"],
    "∫Π_APE dV":           budget_colors["flux"],
    "∫-χₛ dV":             budget_colors["dissipation"],
    "∫(SFS KE->APE) dV":   budget_colors["exchange"],
    "∫Rˢ dV":              "C4",
    "residual_APE":         budget_colors["residual"],
}
for var, color in integrated_vars.items():
    sfs_ape_budget_terms[var].dropna("time").plot.line(ax=ax, x="time", label=var, color=color)
    ax.legend()
ax.set_ylabel("Budget Terms [W or J s⁻¹]")
ax.set_title("Integrated SFS APE Budget Terms")
ax.grid(True, alpha=0.3)
plot_filename = os.path.join("figures", os.path.basename(output_filename).replace(".nc", ".png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Budget timeseries plot saved to: {plot_filename}  ({time.time()-t0:.1f}s)")
#---