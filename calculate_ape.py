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

# Physical constants
g = 9.81  # gravitational acceleration [m/s^2]
rho_0 = 1025  # reference density [kg/m^3]

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
    ds = xr.open_dataset(filename)
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

#+++ Vertical sort density
def vertical_sort_density(rho, dV, LxLy, test=False, z_min=0):
    """
    Sort the density field to obtain the reference state

    This creates a sorted density field by flattening, sorting, and reshaping.
    The sorted field represents the reference state with minimum potential energy.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field (x, y, z)
    dV : xr.DataArray
        Volume of each grid cell (x, y, z)
    LxLy : xr.DataArray or float
        Horizontal area (Lx * Ly)
    test : bool
        Whether to test the sorting

    Returns
    -------
    dz_flat_1d_sorted : np.ndarray
        Sorted vertical coordinate in sorted space
    z_star : np.ndarray
        Cumulative vertical coordinate in sorted space
    rho_1d_sorted : np.ndarray
        Sorted density field with same shape as input
    """
    rho_1d = np.ravel(rho.copy(), order="C")

    dz_flat = dV / LxLy # 3D DataArray with the same shape as rho
    dz_flat_1d = np.ravel(dz_flat.values, order='C')
    if test: assert(dz_flat.sum().values == ds.Lz)

    # Get the permutation indices used to sort rho_1d
    sort_indices = np.argsort(-rho_1d) # descending order since this is density

    # Sort dz_flat using the same permutation
    dz_flat_1d_sorted = dz_flat_1d[sort_indices]
    rho_1d_sorted = rho_1d[sort_indices]
    # from IPython import embed; embed()
    z_star = np.cumsum(dz_flat_1d_sorted) + z_min + dz_flat_1d_sorted[0]/2

    return dz_flat_1d_sorted, z_star, rho_1d_sorted
#---

#+++ Calculate TPE
def calculate_total_potential_energy(rho, dV=None, ds=None):
    """Calculate Total Potential Energy (TPE)"""
    if dV is None:
        if ds is not None:
            dV = ds.dV
        else:
            raise ValueError("Either dV or ds must be provided")
    return g * integrate(rho * rho.z_aac, dV)
#---

#+++ Calculate reference state using sorting method
def calculate_reference_potential_energy(ds, time_idx, test=False):
    """Calculate Reference Potential Energy (RPE)"""
    # Get the density field at this time (3D: x, y, z)
    rho = ds.rho.isel(time=time_idx)

    # Sort the density field to get reference state
    dz_flat_1d_sorted, z_star, rho_1d_sorted = vertical_sort_density(rho, ds.dV, ds.LxLy, test=test, z_min=ds.z_min)

    if test:
        assert(all(np.diff(rho_1d_sorted) <= 0))
        assert(all(np.diff(z_star) > 0))
        assert(np.sum(dz_flat_1d_sorted) == ds.Lz)

    # Calculate Reference Potential Energy (RPE)
    dV_flat_1d_sorted = dz_flat_1d_sorted * ds.LxLy.values
    return g * np.sum(rho_1d_sorted * z_star * dV_flat_1d_sorted)

def calculate_potential_energies(ds, time_idx, test=False):
    """Calculate Available Potential Energy (APE)"""
    TPE = calculate_total_potential_energy(ds.rho.isel(time=time_idx), ds=ds)
    RPE = calculate_reference_potential_energy(ds, time_idx, test=test)
    APE = TPE - RPE
    return APE, TPE, RPE
#---

#+++ Calculate APE time series
def calculate_ape_timeseries(ds, test=False):
    """Calculate APE for all time steps"""
    print("Calculating APE time series...")

    n_times = len(ds.time)
    APE = np.zeros(n_times)
    TPE = np.zeros(n_times)
    RPE = np.zeros(n_times)

    for i in range(n_times):
        print(f"  Processing time step {i+1}/{n_times}", end='\r')
        APE[i], TPE[i], RPE[i] = calculate_potential_energies(ds, i, test=test)

    print("\nDone!")
    return APE, TPE, RPE
#---

#+++ Calculate kinetic energy time series
def calculate_ke_timeseries(ds):
    """Calculate KE for all time steps"""
    print("Calculating KE time series...")

    ke = 0.5 * rho_0 * (ds.u**2 + ds.v**2 + ds.w**2)

    # Get grid spacing from dataset
    dx = ds.Δx_caa
    dy = ds.Δy_aca
    dz = ds.Δz_aac

    # Create dV array
    dV = dx * dy * dz
    KE = (ke * dV).sum(("x_caa", "y_aca", "z_aac"))

    print("\nDone!")
    return KE
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
filename = "kelvin_helmholtz_instability.nc"
ds = load_data(filename)

TPE_online_from_r = g * integrate(ds.rho_z, ds.dV) # g ∭ ρz dV
TPE_online_from_b = integrate(ds.pe, ds.dV) # ∭ (-ρ₀ bz) dV
TPE_offline_from_r = calculate_total_potential_energy(ds.rho, ds=ds) # g ∭ ρz dV
assert all(np.isclose(TPE_online_from_r, TPE_online_from_b, rtol=1e-3)), f"Mismatch: rho_z integral={TPE_online_from_r}, -pe integral={TPE_online_from_b}"

# Use a single time step to test the sorting method
ds0 = ds.isel(time=[0])
APE, TPE, RPE = calculate_ape_timeseries(ds0, test=True)

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
    'KE': (['time'], KE.values),
    'time': ds.time
})
output_ds.to_netcdf('kelvin_helmholtz_ape.nc')
print("\nResults saved to: kelvin_helmholtz_ape.nc")

# Create plots
print("\nCreating plots...")

fig = plot_energy_timeseries(ds, APE, TPE, RPE, KE)
fig.savefig('kelvin_helmholtz_energy_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: kelvin_helmholtz_energy_analysis.png")