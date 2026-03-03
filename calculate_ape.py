#!/usr/bin/env python
"""
Calculate Available Potential Energy (APE) from Kelvin-Helmholtz simulation output

Workflow:
1. Load data
2. Filter density field
3. Calculate local APE using precomputed_integral method
4. Filter local APE with length scale 0.8
"""

import numpy as np
import xarray as xr
import gcm_filters
from ape_calculations import load_data, local_potential_energies_timeseries
from aux00_utils import timeit

# ============================================================================
# Configuration
# ============================================================================
filename = "output/kelvin_helmholtz_instability_64x1x64.nc"
filter_length_scale = 0.8  # Length scale for filtering

# ============================================================================
# Load data
# ============================================================================
print("="*60)
print("Loading data...")
print("="*60)
ds = load_data(filename)
print(f"Dataset loaded: {len(ds.time)} time steps")

# ============================================================================
# Filter density field
# ============================================================================
print("\n" + "="*60)
print("Filtering density field...")
print("="*60)

filter_scale = filter_length_scale * np.sqrt(12)
gaussian_filter = gcm_filters.Filter(
    filter_scale=filter_scale,
    dx_min=float(min(ds.Δx_caa.min(), ds.Δy_aca.min())),
    filter_shape=gcm_filters.FilterShape.GAUSSIAN,
    grid_type=gcm_filters.GridType.REGULAR,
)

ds["rho_filtered"] = gaussian_filter.apply(ds.rho, dims=["x_caa", "y_aca"])
print(f"Density filtered with length scale: {filter_length_scale}")

# ============================================================================
# Calculate local APE using precomputed_integral method
# ============================================================================
print("\n" + "="*60)
print("Calculating local APE...")
print("="*60)

@timeit
def calculate_local_ape():
    return local_potential_energies_timeseries(
        ds,
        test=False,
        verbose_level=1,
        use_numpy_version=True,
        ape_method="precomputed_integral"
    )

local_potential_energies = calculate_local_ape()

# ============================================================================
# Filter local APE
# ============================================================================
print("\n" + "="*60)
print("Filtering local APE...")
print("="*60)

ape_filtered = gaussian_filter.apply(local_potential_energies.ape, dims=["x_caa", "y_aca"])
print(f"Local APE filtered with length scale: {filter_length_scale}")

# ============================================================================
# Save results
# ============================================================================
print("\n" + "="*60)
print("Saving results...")
print("="*60)

output_ds = xr.Dataset({
    "ape_local": local_potential_energies.ape,
    "ape_filtered": ape_filtered,
    "rho_filtered": ds.rho_filtered,
})

output_filename = "kelvin_helmholtz_ape_local.nc"
output_ds.to_netcdf(output_filename)
print(f"Results saved to: {output_filename}")
print("="*60)
