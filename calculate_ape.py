#!/usr/bin/env python
"""
Calculate Available Potential Energy (APE) from Kelvin-Helmholtz simulation output

Workflow:
1. Load data and grid
2. Calculate density fields from buoyancy
3. Filter density field
4. Calculate local APE using precomputed_integral method
5. Filter local APE with length scale 0.8
"""

#+++ Imports
import numpy as np
import xarray as xr
import gcm_filters
from aux00_utils import load_dataset_and_grid
from aux01_ape_functions import (
    calculate_density_fields_from_buoyancy,
    local_potential_energies_timeseries,
)
from ape_plots import plot_dataset_variables
#---

#+++ Configuration
filename = "output/kelvin_helmholtz_instability_64x1x256.nc"
filter_length_scale = 0.8  # Length scale for filtering
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Filter buoyancy field
print("\n" + "="*60)
print("Filtering buoyancy field...")

filtered_dimensions = ["x_caa", "y_aca"]
filter_scale = filter_length_scale * np.sqrt(12)
gaussian_filter = gcm_filters.Filter(
    filter_scale=filter_scale,
    dx_min=float(min(ds.Δx_caa.min(), ds.Δy_aca.min())),
    filter_shape=gcm_filters.FilterShape.GAUSSIAN,
    grid_type=gcm_filters.GridType.REGULAR,
)

ds["b̄"] = gaussian_filter.apply(ds.b, dims=filtered_dimensions) # An overbar denotes a filtering operation
print(f"Buoyancy filtered with length scale: {filter_length_scale}")

ds_filt = ds[["b̄", "dV", "LxLy"]].copy()
ds_full = ds[["b", "dV", "LxLy"]].copy()
#---

#+++ Calculate density fields
print("\n" + "="*60)
print("Calculating density fields...")
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
ds_filt = calculate_density_fields_from_buoyancy(ds_filt, buoyancy_name="b̄", density_name="ρ̄")
print("Density fields calculated: ρ, Z, ρ̄")
#---

#+++ Calculate local APE using precomputed_integral method
print("\n" + "="*60)
print("Calculating local APE...")

full_local_potential_energies = local_potential_energies_timeseries(ds_full, use_numpy_version=True, ape_method="on_the_fly", density_name="ρ", rho_to_sort=ds_full.ρ)
filt_local_potential_energies = local_potential_energies_timeseries(ds_filt, use_numpy_version=True, ape_method="on_the_fly", density_name="ρ̄", rho_to_sort=ds_full.ρ)
#---

#+++ Filter local APE
print("\n" + "="*60)
print("Filtering local APE...")

full_local_ape_filtered = gaussian_filter.apply(full_local_potential_energies.ape, dims=filtered_dimensions)
print(f"Local APE filtered with length scale: {filter_length_scale}")

subfilter_local_ape = full_local_ape_filtered - filt_local_potential_energies.ape
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

output_ds = xr.Dataset({
    "Ea(ρ, z)": full_local_potential_energies.ape,
    "Ea(ρ̄, z)": filt_local_potential_energies.ape,
    "Ēa(ρ, z)": full_local_ape_filtered,
    "Ēa(ρ, z) - Ea(ρ̄, z)": subfilter_local_ape,
    "ρ": ds_full.ρ,
    "ρ̄": ds_filt.ρ̄,
})

output_filename = filename.replace(".nc", "_ape_local.nc")
output_ds.to_netcdf(output_filename)
print(f"\nResults saved to: {output_filename}")
#---

#+++ Plot all variables in output_ds
print("\n" + "="*60)
print("Creating plots...")
print("="*60)
# figures = plot_dataset_variables(output_ds[["Ea(ρ, z)", "Ea(ρ̄, z)", "Ēa(ρ, z) - Ea(ρ̄, z)"]], time_stride=1, col="time", col_wrap=5, cmap="viridis", vmin=0 ,vmax=3, x="x_caa")
figures = plot_dataset_variables(output_ds[["Ea(ρ, z)", "Ea(ρ̄, z)", "Ēa(ρ, z) - Ea(ρ̄, z)"]], time_stride=1, col="time", col_wrap=5, cmap="RdBu_r", vmin=-10, vmax=10, x="x_caa")
#---