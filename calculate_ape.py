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
from pathlib import Path

# Physical constants
g = 9.81  # gravitational acceleration [m/s^2]
rho_0 = 1025  # reference density [kg/m^3]

def load_data(filename):
    """Load the simulation output"""
    print(f"Loading data from {filename}...")
    ds = xr.open_dataset(filename)

    # Convert buoyancy to density
    # b = g * (rho_0 - rho) / rho_0  =>  rho = rho_0 * (1 - b/g)
    ds['rho'] = rho_0 * (1 - ds.b / g)

    # Add coordinate arrays
    if 'z_aac' in ds.coords:
        ds['Z'] = ds.rho * 0 + ds.z_aac
    else:
        print("Warning: z_aac coordinate not found, trying to infer from data")

    return ds

def sort_density_field(rho):
    """
    Sort the density field to obtain the reference state

    This creates a sorted density field by flattening, sorting, and reshaping.
    The sorted field represents the reference state with minimum potential energy.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field (x, y, z)

    Returns
    -------
    rho_sorted : xr.DataArray
        Sorted density field with same shape as input
    """
    rho_flat = np.sort(np.ravel(rho.copy(), order='C'))
    rho_sorted = xr.zeros_like(rho)
    rho_sorted.data = rho_flat.reshape(rho.shape, order='C')
    return rho_sorted

def calculate_ape_sorting(ds, time_idx):
    """
    Calculate APE using the sorting method

    Following the approach from IdealizedAPECalcs.ipynb:
    1. Sort the density field
    2. Calculate Total Potential Energy (TPE) = -∫ ρ z dV
    3. Calculate Reference Potential Energy (RPE) = -∫ ρ_sorted z dV
    4. APE = TPE - RPE

    Note: We use negative sign convention for PE (PE increases downward)
    """
    # Get the density field at this time (3D: x, y, z)
    rho = ds.rho.isel(time=time_idx)
    x = ds.x_caa
    y = ds.y_aca
    z = ds.z_aac

    # Get grid spacing from dataset
    dx = ds.Δx_caa
    dy = ds.Δy_aca
    dz = ds.Δz_aac

    # Create meshgrid for dV (can vary spatially)
    nx, ny, nz = rho.shape
    dV = dx * dy * dz

    # Create Z coordinate meshgrid
    Z = z + 0*x*y

    # Calculate Total Potential Energy (TPE)
    TPE = -np.sum(rho * Z * dV)

    # Sort the density field to get reference state
    rho_sorted = sort_density_field(rho)

    # Calculate Reference Potential Energy (RPE)
    RPE = -np.sum(rho_sorted * Z * dV)

    # Calculate APE
    APE = TPE - RPE

    return APE, TPE, RPE

def calculate_ape_timeseries(ds):
    """Calculate APE for all time steps"""
    print("Calculating APE time series...")

    n_times = len(ds.time)
    APE = np.zeros(n_times)
    TPE = np.zeros(n_times)
    RPE = np.zeros(n_times)

    for i in range(n_times):
        print(f"  Processing time step {i+1}/{n_times}", end='\r')
        APE[i], TPE[i], RPE[i] = calculate_ape_sorting(ds, i)

    print("\nDone!")
    return APE, TPE, RPE

def calculate_kinetic_energy(ds, time_idx):
    """Calculate total kinetic energy (3D: x, y, z)"""
    u = ds.u.isel(time=time_idx).values
    w = ds.w.isel(time=time_idx).values

    # Get grid spacing from dataset
    dx = ds.Δx_caa.values
    dy = ds.Δy_aca.values
    dz = ds.Δz_aac.values

    # Create dV array
    nx, ny, nz = u.shape
    dV = np.zeros_like(u)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dV[i, j, k] = dx[i] * dy[j] * dz[k]

    KE = 0.5 * rho_0 * np.sum((u**2 + w**2) * dV)
    return KE

def calculate_ke_timeseries(ds):
    """Calculate KE for all time steps"""
    print("Calculating KE time series...")

    n_times = len(ds.time)
    KE = np.zeros(n_times)

    for i in range(n_times):
        print(f"  Processing time step {i+1}/{n_times}", end='\r')
        KE[i] = calculate_kinetic_energy(ds, i)

    print("\nDone!")
    return KE

def plot_energy_timeseries(ds, APE, TPE, RPE, KE=None):
    """Plot APE and energy components over time"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: APE, TPE, RPE
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

    plt.tight_layout()
    return fig

def plot_energy_budget(ds, APE, KE):
    """Plot energy budget showing APE to KE conversion"""
    fig, ax = plt.subplots(figsize=(10, 6))

    total_energy = APE + KE
    total_energy_norm = total_energy / total_energy[0]

    ax.plot(ds.time, APE / APE[0], label='APE (normalized)', linewidth=2, color='red')
    ax.plot(ds.time, KE / KE[0], label='KE (normalized)', linewidth=2, color='blue')
    ax.plot(ds.time, total_energy_norm, label='Total (APE + KE, normalized)',
            linewidth=2, color='black', linestyle='--')

    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Energy Budget: APE to KE Conversion')

    plt.tight_layout()
    return fig


# File path to the simulation output
filename = "kelvin_helmholtz_instability.nc"
ds = load_data(filename)

# Calculate APE time series
APE, TPE, RPE = calculate_ape_timeseries(ds)

# Calculate KE time series
KE = calculate_ke_timeseries(ds)

# Print summary statistics
print("\n" + "="*60)
print("APE Calculation Summary")
print("="*60)
print(f"Initial APE: {APE[0]:.4e}")
print(f"Final APE:   {APE[-1]:.4e}")
print(f"APE Change:  {APE[-1] - APE[0]:.4e} ({(APE[-1]/APE[0] - 1)*100:.2f}%)")
print(f"\nInitial KE:  {KE[0]:.4e}")
print(f"Final KE:    {KE[-1]:.4e}")
print(f"KE Change:   {KE[-1] - KE[0]:.4e} ({(KE[-1]/KE[0] - 1)*100:.2f}%)")
print(f"\nTotal Energy Conservation: {(APE[-1] + KE[-1]) / (APE[0] + KE[0]):.6f}")
print("="*60)

# Save results to NetCDF
output_ds = xr.Dataset({
    'APE': (['time'], APE),
    'TPE': (['time'], TPE),
    'RPE': (['time'], RPE),
    'KE': (['time'], KE),
    'time': ds.time
})
output_ds.to_netcdf('kelvin_helmholtz_ape.nc')
print("\nResults saved to: kelvin_helmholtz_ape.nc")

# Create plots
print("\nCreating plots...")

fig1 = plot_energy_timeseries(ds, APE, TPE, RPE, KE)
fig1.savefig('kelvin_helmholtz_energy_timeseries.png', dpi=150, bbox_inches='tight')
print("Saved: kelvin_helmholtz_energy_timeseries.png")

fig2 = plot_energy_budget(ds, APE, KE)
fig2.savefig('kelvin_helmholtz_energy_budget.png', dpi=150, bbox_inches='tight')
print("Saved: kelvin_helmholtz_energy_budget.png")

plt.show()

