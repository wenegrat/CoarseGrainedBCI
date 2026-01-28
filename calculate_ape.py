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
    integrated_potential_energies_timeseries,
    integrated_KE_timeseries,
    local_potential_energies_timeseries,
    calculate_reference_potential_energy_profile,
    integrated_reference_potential_energy,
    integrated_total_potential_energy,
    load_data,
    integrate,
    g,
)
from ape_plots import plot_energy_timeseries, plot_potential_energies
from matplotlib import pyplot as plt
import pynanigans as pn

# File path to the simulation output
filename = "output/kelvin_helmholtz_instability_64x1x64.nc"
rho_0 = 1025
ds = load_data(filename, ρ0=rho_0)

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
    val4 = integrated_total_potential_energy(ds0.rho, ds=ds, ρ0=rho_0)
    assert np.isclose(val3, val4, rtol=1e-1), f"Mismatch: reference PE={val3}, total PE={val4}"
#---

#+++ Test local calculations
if True:
    ds0 = ds.sel(time=[70], method="nearest")
    local_potential_energies = local_potential_energies_timeseries(ds0, test=True, verbose_level=0, use_numpy_version=False)
    pause
    potential_energies = integrated_potential_energies_timeseries(ds0, test=True, verbose_level=0)
#---

# Calculate PE time series
global_potential_energies = integrated_potential_energies_timeseries(ds, test=False, verbose_level=1)

# Calculate local APE time series
local_potential_energies = local_potential_energies_timeseries(ds, test=False, verbose_level=1)
integrated_local_potential_energies = integrate(local_potential_energies[["ape", "tpe"]], ds.dV)
integrated_local_potential_energies["rpe"] = (local_potential_energies.rho_sorted * local_potential_energies.dz_sorted).sum("z_1d_sorted")

# local_potential_energies.ape.squeeze().sel(time=slice(None, None, 9)).plot(col="time", col_wrap=3, robust=True)

# Calculate KE time series
KE = integrated_KE_timeseries(ds, ρ0=rho_0)
# Print summary statistics
APE = global_potential_energies.APE

print("\n" + "="*60)
print("APE Calculation Summary")
print("="*60)
print(f"APE Change:  {APE[-1] - APE[0]:.4e} ({(APE[-1]/APE[0] - 1)*100:.2f}%)")
print(f"KE Change:   {KE[-1] - KE[0]:.4e} ({(KE[-1]/KE[0] - 1)*100:.2f}%)")
print(f"\nTotal Energy Conservation: {(APE[-1] + KE[-1]) / (APE[0] + KE[0]):.6f}")
print("="*60)

# Save results to NetCDF
output_ds = xr.Dataset(dict(APE = APE,
                            TPE = global_potential_energies.TPE,
                            RPE = global_potential_energies.RPE,
                            ape_int = integrated_local_potential_energies.ape,
                            tpe_int = integrated_local_potential_energies.tpe,
                            rpe_int = integrated_local_potential_energies.rpe,
                            KE  = KE))
output_ds.to_netcdf("kelvin_helmholtz_ape.nc")
print("\nResults saved to: kelvin_helmholtz_ape.nc")

# Create plots
print("\nCreating plots...")



# from os.path impInort basename
# figname_energy = f"figures/{basename(filename)}_energy_analysis.png"
# fig = plot_energy_timeseries(ds, APE=APE, TPE=global_potential_energies.TPE, RPE=global_potential_energies.RPE, KE=KE)
# fig.savefig(figname_energy, dpi=150, bbox_inches="tight")
# print(f"Saved: {figname_energy}")

fig_global = plot_potential_energies(ds.time, APE=global_potential_energies.APE)
fig_local = plot_potential_energies(ds.time, APE=integrated_local_potential_energies.ape)
