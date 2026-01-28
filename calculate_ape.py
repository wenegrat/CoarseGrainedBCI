#!/usr/bin/env python
"""
Calculate Available Potential Energy (APE) from Kelvin-Helmholtz simulation output

This script reads the NetCDF output from the Kelvin-Helmholtz instability simulation
and calculates the APE using the sorting method following Winters et al. (1995)
and the approaches used in CrossScaleAPE notebooks.
"""

import numpy as np
import xarray as xr
import scipy.integrate as integrate
from ape_calculations import (
    vertical_sort_density,
    calculate_ape_timeseries,
    calculate_ke_timeseries,
    calculate_reference_potential_energy_profile,
    integrated_reference_potential_energy,
    integrated_total_potential_energy,
    vectorized_summation_method_local_APE,
    vectorized_cumulative_method_local_APE,
    create_inverse_sort_lookup,
    load_data,
    integrate,
    g,
    rho_0,
)
from ape_plots import plot_energy_timeseries
from matplotlib import pyplot as plt
import pynanigans as pn

# File path to the simulation output
filename = "output/kelvin_helmholtz_instability_128x1x128.nc"
ds = load_data(filename)

#+++ Test that convertion between ρ and b is correct
ds0 = ds.sel(time=[100])
TPE_online_from_ρ = g * integrate(ds0.rho_z, ds0.dV) # g ∭ ρz dV
TPE_online_from_b = integrate(ds0.pe, ds0.dV) # ∭ (-ρ₀ bz) dV

TPE_offline_from_ρ = integrated_total_potential_energy(ds0.rho, ds=ds0) # g ∭ ρz dV
TPE_offline_from_b = - rho_0 * integrate(ds0.b * ds0.z_aac, ds0.dV)

assert np.isclose(TPE_online_from_ρ, TPE_online_from_b, rtol=1e-3), f"Mismatch: rho_z integral={TPE_online_from_ρ}, -pe integral={TPE_online_from_b}"
assert np.isclose(TPE_online_from_ρ, TPE_offline_from_ρ, rtol=1e-3), f"Mismatch: rho_z integral={TPE_online_from_ρ}, offline integral={TPE_offline_from_ρ}"
assert np.isclose(TPE_online_from_b, TPE_offline_from_b, rtol=1e-3), f"Mismatch: pe integral={TPE_online_from_b}, -b*z integral={TPE_offline_from_b}"
#---

#+++ Test that TPE at the initial time is close to the reference potential energy
ds0 = ds.isel(time=[0])
rho0 = ds.rho.isel(time=0)
vertically_sorted_ds = calculate_reference_potential_energy_profile(rho0, ds.dV, ds.LxLy, test=True, z_min=ds.z_min, Lz=ds.Lz)
if False:
    val3 = integrated_reference_potential_energy(vertically_sorted_ds, ds.LxLy.values)
    val4 = integrated_total_potential_energy(ds0.rho, ds=ds)
    assert np.isclose(val3, val4, rtol=1e-1), f"Mismatch: reference PE={val3}, total PE={val4}"
#---

step = 2
ds0 = ds.sel(time=[100])
vertically_sorted_ds, threed_sorted_ds = vertical_sort_density(ds0.rho, ds0.dV, ds0.LxLy, test=True, z_min=ds0.z_min, Lz=ds0.Lz)

vertically_sorted_ds["rho_1d_sorted_cumulative_integral"] = (vertically_sorted_ds.rho_1d_sorted * vertically_sorted_ds.dz_1d_sorted).cumsum("z_1d_sorted")
vertically_sorted_ds["dz_1d_sorted_cumulative_integral"] = (vertically_sorted_ds.dz_1d_sorted).cumsum("z_1d_sorted")

# Create inverse lookup table for fast z_0 retrieval
inverse_sort_indices, z_1d_sorted_values = create_inverse_sort_lookup(vertically_sorted_ds)

# Vectorized calculation of local APE
Ea_on_the_fly = vectorized_summation_method_local_APE(ds0, vertically_sorted_ds, threed_sorted_ds, inverse_sort_indices, z_1d_sorted_values)
Ea_integrated = integrate(Ea_on_the_fly, ds0.Δx_caa * ds0.Δy_aca * ds0.Δz_aac)
APE0, TPE0, RPE0 = calculate_ape_timeseries(ds0, test=False)

opts = dict(vmin=-4e-4, vmax=4e-4, cmap="RdBu_r")
pause
# Calculate PE time series
APE, TPE, RPE = calculate_ape_timeseries(ds, test=False)

# Calculate KE time series
KE = calculate_ke_timeseries(ds)

# Print summary statistics
print("\n" + "="*60)
print("APE Calculation Summary")
print("="*60)
print(f"APE Change:  {APE[-1] - APE[0]:.4e} ({(APE[-1]/APE[0] - 1)*100:.2f}%)")
print(f"KE Change:   {KE[-1] - KE[0]:.4e} ({(KE[-1]/KE[0] - 1)*100:.2f}%)")
print(f"\nTotal Energy Conservation: {(APE[-1] + KE[-1]) / (APE[0] + KE[0]):.6f}")
print("="*60)

# Save results to NetCDF
output_ds = xr.Dataset(dict(APE=("time", APE),
                            TPE=("time", TPE),
                            RPE=("time", RPE),
                            KE=("time", KE.values)))
output_ds.to_netcdf("kelvin_helmholtz_ape.nc")
print("\nResults saved to: kelvin_helmholtz_ape.nc")

# Create plots
print("\nCreating plots...")

pause
from os.path import basename
figname = f"figures/{basename(filename)}_energy_analysis.png"
fig = plot_energy_timeseries(ds, APE, TPE, RPE, KE)
fig.savefig(figname, dpi=150, bbox_inches="tight")
print(f"Saved: {figname}")