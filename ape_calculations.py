"""
Energy calculation functions for Available Potential Energy (APE) analysis

This module contains functions for calculating APE using the sorting method
following Winters et al. (1995).
"""

import numpy as np
import xarray as xr

# Physical constants
g = 9.81  # gravitational acceleration [m/s^2]
rho_0 = 1025  # reference density [kg/m^3]


def integrate(da, dV):
    """Integrate a DataArray over spatial dimensions"""
    return (da * dV).sum(("x_caa", "y_aca", "z_aac"))


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
    if test:
        assert(dz_flat.sum().values == ds.Lz)

    # Get the permutation indices used to sort rho_1d
    sort_indices = np.argsort(-rho_1d) # descending order since this is density

    # Sort dz_flat using the same permutation
    dz_flat_1d_sorted = dz_flat_1d[sort_indices]
    rho_1d_sorted = rho_1d[sort_indices]
    z_1d_sorted = np.cumsum(dz_flat_1d_sorted) + z_min + dz_flat_1d_sorted[0]/2

    # Reshape z_1d_sorted back into the original shape used by rho
    z_3d_sorted = z_1d_sorted.reshape(rho.shape, order="C")

    # Reshape rho_1d_sorted back into the original shape used by rho
    rho_3d_sorted = rho_1d_sorted.reshape(rho.shape, order="C")

    # Put verticall sorted data into a Dataset
    rho_1d_sorted = xr.DataArray(rho_1d_sorted, dims="z_1d_sorted", coords=dict(z_1d_sorted=z_1d_sorted))
    dz_1d_sorted = xr.DataArray(dz_flat_1d_sorted, dims="z_1d_sorted", coords=dict(z_1d_sorted=z_1d_sorted))
    vertically_sorted_ds = xr.Dataset(dict(rho_1d_sorted=rho_1d_sorted, dz_1d_sorted=dz_1d_sorted))

    # Put 3D sorted data into a Dataset
    z_3d_sorted = xr.DataArray(z_3d_sorted, dims=rho.dims, coords=rho.coords)
    rho_3d_sorted = xr.DataArray(rho_3d_sorted, dims=rho.dims, coords=rho.coords)
    threed_sorted_ds = xr.Dataset(dict(rho_3d_sorted=rho_3d_sorted, z_3d_sorted=z_3d_sorted))

    if test:
        rho_3d_reshaped = rho_1d.reshape(rho.shape, order="C")
        assert(np.all(rho == rho_3d_reshaped))

    return vertically_sorted_ds, threed_sorted_ds
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
    vertically_sorted_ds, threed_sorted_ds = vertical_sort_density(rho, ds.dV, ds.LxLy, test=test, z_min=ds.z_min)

    if test:
        assert(all(np.diff(vertically_sorted_ds.rho_1d_sorted) <= 0))
        assert(all(np.diff(vertically_sorted_ds.z_1d_sorted) > 0))
        assert(np.sum(vertically_sorted_ds.dz_1d_sorted).values == ds.Lz)

    # Calculate Reference Potential Energy (RPE)
    dV_flat_1d_sorted = vertically_sorted_ds.dz_1d_sorted * ds.LxLy.values
    return g * np.sum(vertically_sorted_ds.rho_1d_sorted * vertically_sorted_ds.z_1d_sorted * dV_flat_1d_sorted)

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
