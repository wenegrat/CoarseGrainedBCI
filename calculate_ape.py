#!/usr/bin/env python
"""
Calculate Available Potential Energy (APE) from Kelvin-Helmholtz simulation output

This script reads the NetCDF output from the Kelvin-Helmholtz instability simulation
and calculates the APE using the sorting method following Winters et al. (1995)
and the approaches used in CrossScaleAPE notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.integrate as integrate
from ape_calculations import (
    calculate_ape_timeseries,
    calculate_ke_timeseries,
    calculate_potential_energies,
    calculate_reference_potential_energy,
    calculate_total_potential_energy,
    vertical_sort_density,
    g,
    rho_0,
)


#+++ Auxiliary functions
def sum(da):
    return da.sum(("x_caa", "y_aca", "z_aac"))

def integrate(da, dV):
    return (da * dV).sum(("x_caa", "y_aca", "z_aac"))
#---

#+++ Load data
def load_data(filename):
    """Load the simulation output"""
    print(f"Loading data from {filename}...")
    ds = xr.open_dataset(filename, decode_times=False)
    grid = xr.open_dataset(filename, group="underlying_grid_reconstruction_kwargs")

    ds.attrs["Lx"] = np.diff(grid.x)
    ds.attrs["Ly"] = np.diff(grid.y)
    ds.attrs["Lz"] = np.diff(grid.z)

    ds.attrs["x_min"] = grid.x.min()
    ds.attrs["x_max"] = grid.x.max()
    ds.attrs["y_min"] = grid.y.min()
    ds.attrs["y_max"] = grid.y.max()
    ds.attrs["z_min"] = grid.z.min()
    ds.attrs["z_max"] = grid.z.max()

    ds["dV"] = ds.Δx_caa * ds.Δy_aca * ds.Δz_aac
    ds["LxLy"] = ds.Lx * ds.Ly

    # Convert buoyancy to density
    # b = g * (rho_0 - rho) / rho_0  =>  rho = rho_0 * (1 - b/g)
    ds["rho"] = rho_0 * (1 - ds.b / g)
    ds["rho_z"] = (rho_0 * ds.z_aac + ds.pe / g) # pe  = -b*z

    # Add coordinate arrays
    if "z_aac" in ds.coords:
        ds["Z"] = ds.rho * 0 + ds.z_aac
    else:
        print("Warning: z_aac coordinate not found, trying to infer from data")

    return ds
#---

#+++ Plot energy timeseries
def plot_energy_timeseries(ds, APE, TPE, RPE, KE=None):
    """Plot APE and energy components over time"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: TPE and RPE
    ax1 = axes[0]
    ax1.plot(ds.time, TPE, label='Total PE', linewidth=2)
    ax1.plot(ds.time, RPE, label='Reference PE', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Potential Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Potential Energy Components')

    # Plot 2: APE and KE
    ax2 = axes[1]
    ax2.plot(ds.time, APE, label='APE', linewidth=2, color='red')
    if KE is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(ds.time, KE, label='KE', linewidth=2, color='blue', alpha=0.7)
        ax2_twin.set_ylabel('Kinetic Energy', color='blue')
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        ax2_twin.legend(loc='upper right')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Available Potential Energy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Available Potential Energy and Kinetic Energy')

    # Plot 3: Normalized energy budget
    ax3 = axes[2]
    if KE is not None:
        total_energy = APE + KE
        total_energy_norm = total_energy / total_energy[0]

        ax3.plot(ds.time, APE / APE[0], label='APE (normalized)', linewidth=2, color='red')
        ax3.plot(ds.time, KE / KE[0], label='KE (normalized)', linewidth=2, color='blue')
        ax3.plot(ds.time, total_energy_norm, label='Total (APE + KE, normalized)',
                linewidth=2, color='black', linestyle='--')

        ax3.set_xlabel('Time')
        ax3.set_ylabel('Normalized Energy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Energy Budget: APE to KE Conversion')

    plt.tight_layout()
    return fig
#---

# File path to the simulation output
filename = "output/kelvin_helmholtz_instability_128x1x128.nc"
ds = load_data(filename)

TPE_online_from_r = g * integrate(ds.rho_z, ds.dV) # g ∭ ρz dV
TPE_online_from_b = integrate(ds.pe, ds.dV) # ∭ (-ρ₀ bz) dV
TPE_offline_from_r = calculate_total_potential_energy(ds.rho, ds=ds) # g ∭ ρz dV
assert all(np.isclose(TPE_online_from_r, TPE_online_from_b, rtol=1e-3)), f"Mismatch: rho_z integral={TPE_online_from_r}, -pe integral={TPE_online_from_b}"

# Use a single time step to test the sorting method
ds0 = ds.isel(time=[0])

val1 = integrate(ds0.pe, ds0.dV)
val2 = - rho_0 * integrate(ds0.b * ds0.z_aac, ds0.dV)
assert np.isclose(val1, val2, rtol=1e-3), f"Mismatch: pe integral={val1.values}, -b*z integral={val2.values}"

val3 = calculate_reference_potential_energy(ds, 0, test=True)
val4 = calculate_total_potential_energy(ds0.rho, ds=ds)
assert np.isclose(val3, val4, rtol=1e-1), f"Mismatch: reference PE={val3}, total PE={val4}"

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
output_ds = xr.Dataset({
    'APE': (['time'], APE),
    'TPE': (['time'], TPE),
    'RPE': (['time'], RPE),
    'KE': (['time'], KE.values),
    'time': ds.time
})
output_ds.to_netcdf('kelvin_helmholtz_ape.nc')
print("\nResults saved to: kelvin_helmholtz_ape.nc")

# Create plots
print("\nCreating plots...")

from os.path import basename
figname = f"figures/{basename(filename)}_energy_analysis.png"
fig = plot_energy_timeseries(ds, APE, TPE, RPE, KE)
fig.savefig(figname, dpi=150, bbox_inches='tight')
print(f"Saved: {figname}")