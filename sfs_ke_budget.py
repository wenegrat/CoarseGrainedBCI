#!/usr/bin/env python
#+++ Imports
import numpy as np
import xarray as xr
import gcm_filters
from aux00_utils import load_dataset_and_grid, condense_velocities, integrate
from aux01_pe_functions import calculate_ape_to_ke_exchange_term
from aux02_ke_functions import (
    calculate_sfs_stress_tensor,
    calculate_strain_tensor,
    calculate_sfs_ke_dissipation,
    calculate_cross_scale_ke_flux,
    calculate_sfs_ke_tendency,
)
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS KE budget from Kelvin-Helmholtz simulation output")
parser.add_argument("--filename", default="output/kelvin_helmholtz_instability_64x1x256.nc",
                    help="Path to simulation NetCDF file")
args = parser.parse_args()
filename = args.filename
filter_length_scale = 0.8  # Length scale for filtering
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Filter velocity field
print("\n" + "="*60)
print("Filtering velocity field...")

filtered_dimensions = ["x_caa", "y_aca"]
filter_scale = filter_length_scale * np.sqrt(12)
gaussian_filter = gcm_filters.Filter(
    filter_scale=filter_scale,
    dx_min=float(min(ds.Δx_caa.min(), ds.Δy_aca.min())),
    filter_shape=gcm_filters.FilterShape.GAUSSIAN,
    grid_type=gcm_filters.GridType.REGULAR,
)

ds = condense_velocities(ds, indices=[1, 2, 3])  # uᵢ with i dimension
ds["ūᵢ"] = gaussian_filter.apply(ds["uᵢ"], dims=filtered_dimensions)
ds["b̄"] = gaussian_filter.apply(ds["b"], dims=filtered_dimensions)

ds_full = ds[["b", "dV", "uᵢ"]].copy()
ds_filt = ds[["b̄", "dV", "ūᵢ"]].copy()

print(f"Velocities filtered with length scale: {filter_length_scale}")
#---

#+++ Calculate SFS stress tensor
print("\n" + "="*60)
print("Calculating SFS stress tensor...")

# τⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ   shape: (i, j, time, z, y, x)
# Pre-pass filtered_u_i so the filter is not applied a second time
sfs_stress_tensor = calculate_sfs_stress_tensor(ds_full["uᵢ"], gaussian_filter,
                                                filter_dims=filtered_dimensions,
                                                filtered_u_i=ds_filt["ūᵢ"])

# Sanity check: trace/2 must equal KE_s pointwise
sfs_stress_tensor_trace = sfs_stress_tensor.sel(i=1, j=1) + sfs_stress_tensor.sel(i=2, j=2) + sfs_stress_tensor.sel(i=3, j=3)
sfs_ke_density = sfs_stress_tensor_trace / 2
print("Done!")
#---

#+++ Calculate strain rate tensor of the filtered flow
print("\n" + "="*60)
print("Calculating strain rate tensor of the filtered flow...")
strain_rate_tensor_l = calculate_strain_tensor(ds_filt["ūᵢ"])
print("Done!")
#---

#+++ Calculate cross-scale KE flux
print("\n" + "="*60)
print("Calculating cross-scale KE flux...")

# Πℓ = -ρ₀ τⁱʲ : S̄ⁱʲ  [m² s⁻³]
cross_scale_ke_flux = calculate_cross_scale_ke_flux(sfs_stress_tensor, strain_rate_tensor_l)
print("Done!")
#---

#+++ Calculate strain tensor of the full (unfiltered) flow
print("\n" + "="*60)
print("Calculating strain tensor for the full (unfiltered) flow...")
strain_rate_tensor = calculate_strain_tensor(ds_full["uᵢ"])
print("Done!")
#---

#+++ Calculate SFS KE dissipation
print("\n" + "="*60)
print("Calculating SFS KE dissipation...")

# ε<ℓ = 2ρ₀ν τ(S, S) = 2ρ₀ν Σᵢⱼ [ filter(Sⁱʲ Sⁱʲ) - filter(Sⁱʲ)² ]   [m² s⁻³]
sfs_ke_dissipation = calculate_sfs_ke_dissipation(strain_rate_tensor, ds.ν, gaussian_filter, filter_dims=filtered_dimensions)
print("Done!")
#---

#+++ Calculate SFS APE to KE exchange term
print("\n" + "="*60)
print("Calculating SFS KE->APE exchange term...")

# For SFS KE, use the vertical velocity (w component, i=3) and buoyancy fields.
# Filtered fields for collapsed/filtered form.
ape_to_ke_exchange = calculate_ape_to_ke_exchange_term(
    ds_full["uᵢ"].sel(i=3),   # full w
    ds_full.b,                # full buoyancy
    gaussian_filter,
    filter_dims=filtered_dimensions,
    filtered_w=ds_filt["ūᵢ"].sel(i=3),
    filtered_b=ds_filt["b̄"]
)
print("Done!")
#---

#+++ Calculate SFS KE tendency
print("\n" + "="*60)
print("Calculating SFS KE tendency...")

# ∂KE_s/∂t   centred finite difference, staggered time grid
dKE_dt = calculate_sfs_ke_tendency(sfs_ke_density)
print("Done!")
#---

#+++ Integrate
print("\n" + "="*60)
print("Integrating SFS KE budget terms...")

dV = ds_full.dV
int_dKE_dt = integrate(dKE_dt, dV)

int_ape_to_ke_exchange = integrate(ape_to_ke_exchange.reindex(time=dKE_dt.time), dV)
int_sfs_ke_density   = integrate(sfs_ke_density.reindex(time=dKE_dt.time), dV)
int_cross_scale_ke_flux  = integrate(cross_scale_ke_flux.reindex(time=dKE_dt.time), dV)
int_sfs_ke_dissipation = integrate(sfs_ke_dissipation.reindex(time=dKE_dt.time), dV)

residual = -int_dKE_dt + int_ape_to_ke_exchange + int_cross_scale_ke_flux - int_sfs_ke_dissipation

print("Done!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

sfs_ke_budget_terms = xr.Dataset({
    # Local fields
    "SFS KE":     sfs_ke_density,
    "∂ₜ SFS KE": dKE_dt,
    "Π_KE":   cross_scale_ke_flux,
    "εₛ":    sfs_ke_dissipation,
    "SFS APE->KE exchange": ape_to_ke_exchange,
    # Integrated scalars
    "∫SFS KE dV":         int_sfs_ke_density,
    "∫-∂ₜ SFS KE dV":     -int_dKE_dt,
    "∫Π_KE dV":           int_cross_scale_ke_flux,
    "∫-εₛ dV":           -int_sfs_ke_dissipation,
    "∫(SFS APE->KE) dV":  int_ape_to_ke_exchange,
    "residual_KE": residual,
})

output_filename = filename.replace(".nc", "_sfs_ke_budget.nc")
sfs_ke_budget_terms.to_netcdf(output_filename)
print(f"\nResults saved to: {output_filename}")
#---

#+++ Plot integrated KE decomposition
print("\n" + "="*60)
print("Creating plots...")
print("="*60)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Colors shared with sfs_ape_budget.py — keep analogous terms the same colour
budget_colors = {
    "tendency":   "C0",
    "flux":       "C1",
    "dissipation":"C2",
    "exchange":   "C3",
    "residual":   "k",
}
integrated_vars = {
    "∫-∂ₜ SFS KE dV":    budget_colors["tendency"],
    "∫Π_KE dV":           budget_colors["flux"],
    "∫-εₛ dV":            budget_colors["dissipation"],
    "∫(SFS APE->KE) dV":  budget_colors["exchange"],
    "residual_KE":         budget_colors["residual"],
}
for var, color in integrated_vars.items():
    sfs_ke_budget_terms[var].dropna("time").plot.line(ax=ax, x="time", label=var, color=color)
    ax.legend()
ax.set_ylabel("Budget Terms [W or J s⁻¹]")
ax.set_title("Integrated SFS KE Budget Terms")
ax.grid(True, alpha=0.3)
plot_filename = output_filename.replace(".nc", ".png")
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Budget timeseries plot saved to: {plot_filename}")
#---