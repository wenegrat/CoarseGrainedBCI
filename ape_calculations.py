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

#+++ Auxiliary functions
def volume_sum(da):
    return da.sum(("x_caa", "y_aca", "z_aac"))

def integrate(da, dV):
    """Integrate a DataArray over spatial dimensions"""
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

#+++ Vertical sort density
def vertical_sort_density(rho, dV, LxLy, test=False, z_min=0, Lz=None):
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
    Lz : float, optional
        Total vertical extent for testing

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
    dz_flat_1d = np.ravel(dz_flat.values, order="C")
    if test and Lz is not None:
        assert(dz_flat.sum().values == Lz)

    # Get the permutation indices used to sort rho_1d
    sort_indices = np.argsort(-rho_1d) # descending order since this is density

    # Sort dz_flat using the same permutation
    dz_flat_1d_sorted = dz_flat_1d[sort_indices]
    rho_1d_sorted = rho_1d[sort_indices]
    z_1d_sorted = np.cumsum(dz_flat_1d_sorted) + z_min + dz_flat_1d_sorted[0]/2

    # Reshape z_1d_sorted and rho_1d_sorted back into the original shape used by rho
    z_3d_sorted = z_1d_sorted.reshape(rho.shape, order="C")
    rho_3d_sorted = rho_1d_sorted.reshape(rho.shape, order="C")
    sort_indices_3d = sort_indices.reshape(rho.shape, order="C")

    # Put vertically sorted data into a Dataset
    sort_indices_1d = xr.DataArray(sort_indices, dims="z_1d_sorted", coords=dict(z_1d_sorted=z_1d_sorted))
    rho_1d_sorted = xr.DataArray(rho_1d_sorted, dims="z_1d_sorted", coords=dict(z_1d_sorted=z_1d_sorted))
    dz_1d_sorted = xr.DataArray(dz_flat_1d_sorted, dims="z_1d_sorted", coords=dict(z_1d_sorted=z_1d_sorted))
    vertically_sorted_ds = xr.Dataset(dict(rho_1d_sorted=rho_1d_sorted, dz_1d_sorted=dz_1d_sorted, sort_indices_1d=sort_indices_1d))

    # Put 3D sorted data into a Dataset
    sort_indices_3d = xr.DataArray(sort_indices_3d, dims=rho.dims, coords=rho.coords)
    z_3d_sorted = xr.DataArray(z_3d_sorted, dims=rho.dims, coords=rho.coords)
    rho_3d_sorted = xr.DataArray(rho_3d_sorted, dims=rho.dims, coords=rho.coords)
    threed_sorted_ds = xr.Dataset(dict(rho_3d_sorted=rho_3d_sorted, z_3d_sorted=z_3d_sorted, sort_indices_3d=sort_indices_3d))

    if test:
        rho_3d_reshaped = rho_1d.reshape(rho.shape, order="C")
        assert(np.all(rho == rho_3d_reshaped))

    return vertically_sorted_ds, threed_sorted_ds
#---

#+++ Calculate TPE
def calculate_total_potential_energy(rho):
    """
    Calculate local Total Potential Energy density

    Parameters
    ----------
    rho : xr.DataArray
        Density field

    Returns
    -------
    xr.DataArray
        Local TPE density: g * rho * z
    """
    return g * rho * rho.z_aac

def integrated_total_potential_energy(rho, dV=None, ds=None):
    """
    Calculate volume-integrated Total Potential Energy (TPE)

    Parameters
    ----------
    rho : xr.DataArray
        Density field
    dV : xr.DataArray, optional
        Volume element
    ds : xr.Dataset, optional
        Dataset containing dV

    Returns
    -------
    float
        Integrated TPE
    """
    if dV is None:
        if ds is not None:
            dV = ds.dV
        else:
            raise ValueError("Either dV or ds must be provided")
    tpe_local = calculate_total_potential_energy(rho)
    return integrate(tpe_local, dV)
#---

#+++ Calculate reference state using sorting method
def calculate_reference_potential_energy_profile(rho, dV, LxLy, test=False, z_min=0, Lz=None):
    """
    Calculate local Reference Potential Energy density using sorted density

    Parameters
    ----------
    rho : xr.DataArray
        Density field
    dV : xr.DataArray
        Volume element
    LxLy : float
        Horizontal area
    test : bool
        Whether to run tests
    z_min : float
        Minimum z coordinate
    Lz : float, optional
        Total vertical extent for testing

    Returns
    -------
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted 1D density and coordinate
    """
    # Sort the density field to get reference state
    vertically_sorted_ds, threed_sorted_ds = vertical_sort_density(rho, dV, LxLy, test=test, z_min=z_min, Lz=Lz)

    if test:
        assert(all(np.diff(vertically_sorted_ds.rho_1d_sorted) <= 0))
        assert(all(np.diff(vertically_sorted_ds.z_1d_sorted) > 0))
        if Lz is not None:
            assert(np.sum(vertically_sorted_ds.dz_1d_sorted).values == Lz)

    return vertically_sorted_ds

def integrated_reference_potential_energy(vertically_sorted_ds, LxLy):
    """
    Calculate volume-integrated Reference Potential Energy (RPE)

    Parameters
    ----------
    vertically_sorted_ds : xr.Dataset
        Sorted density dataset from calculate_reference_potential_energy_profile
    LxLy : float
        Horizontal area

    Returns
    -------
    float
        Integrated RPE
    """
    dV_flat_1d_sorted = vertically_sorted_ds.dz_1d_sorted * LxLy
    return g * np.sum(vertically_sorted_ds.rho_1d_sorted * vertically_sorted_ds.z_1d_sorted * dV_flat_1d_sorted)

def integrated_potential_energies(ds, time_idx, test=False):
    """
    Calculate volume-integrated potential energies (APE, TPE, RPE)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing simulation data
    time_idx : int
        Time index
    test : bool
        Whether to run tests

    Returns
    -------
    tuple
        (APE, TPE, RPE) - all volume-integrated scalars
    """
    rho = ds.rho.isel(time=time_idx)
    TPE = integrated_total_potential_energy(rho, ds=ds)
    vertically_sorted_ds = calculate_reference_potential_energy_profile(rho, ds.dV, ds.LxLy, test=test, z_min=ds.z_min, Lz=ds.Lz)
    RPE = integrated_reference_potential_energy(vertically_sorted_ds, ds.LxLy.values)
    APE = TPE - RPE
    return APE, TPE, RPE
#---

#+++ Calculate APE time series
def calculate_ape_timeseries(ds, test=False):
    """
    Calculate volume-integrated APE for all time steps

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing simulation data
    test : bool
        Whether to run tests

    Returns
    -------
    tuple
        (APE, TPE, RPE) - time series of volume-integrated energies
    """
    print("Calculating APE time series...")

    n_times = len(ds.time)
    APE = np.zeros(n_times)
    TPE = np.zeros(n_times)
    RPE = np.zeros(n_times)

    for i in range(n_times):
        print(f"  Processing time step {i+1}/{n_times}", end='\r')
        APE[i], TPE[i], RPE[i] = integrated_potential_energies(ds, i, test=test)

    print("\nDone!")
    return APE, TPE, RPE
#---

#+++ Create inverse lookup table for fast z_0 retrieval
def create_inverse_sort_lookup(vertically_sorted_ds):
    """
    Create inverse lookup table for fast z_0 coordinate retrieval

    This function creates a mapping from density indices to their positions in the
    sorted array and extracts z coordinates for fast access. This avoids the slow
    .where() operation when looking up reference z coordinates.

    Parameters
    ----------
    vertically_sorted_ds : xr.Dataset
        Dataset containing sort_indices_1d and z_1d_sorted

    Returns
    -------
    inverse_sort_indices : np.ndarray
        Array mapping density_index -> position in sorted array (O(1) lookup)
    z_1d_sorted_values : np.ndarray
        Z coordinate values in sorted order for fast indexing
    """
    print("Creating inverse lookup table for fast z_0 retrieval...")
    inverse_sort_indices = np.empty(len(vertically_sorted_ds.sort_indices_1d), dtype=int)
    for i, idx in enumerate(vertically_sorted_ds.sort_indices_1d.values):
        inverse_sort_indices[int(idx)] = i
    z_1d_sorted_values = vertically_sorted_ds.z_1d_sorted.values
    print("Done!")

    return inverse_sort_indices, z_1d_sorted_values
#---

#+++ Calculate local APE using summation method
def summation_method_local_APE(vertically_sorted_ds, rho, displacement, displacement_slice, z, z_0):
    """
    Calculate local APE using direct summation (straightforward but slower method)

    Parameters
    ----------
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    rho : float
        Density at current position
    displacement : float
        z - z_0 (vertical displacement)
    displacement_slice : slice
        Slice object for selecting the displacement range
    z : float
        Current z coordinate
    z_0 : float
        Reference z coordinate (sorted position)

    Returns
    -------
    float
        Local APE value: g * ∫(rho - rho_sorted) dz / rho_0
    """
    # Get signed dz based on displacement direction
    if displacement > 0:
        dz_flat = +vertically_sorted_ds.dz_1d_sorted.sel(z_1d_sorted=displacement_slice)
    else:
        dz_flat = -vertically_sorted_ds.dz_1d_sorted.sel(z_1d_sorted=displacement_slice)

    # Calculate buoyancy difference and integrate
    rho_sorted_profile = vertically_sorted_ds.rho_1d_sorted
    rho_sorted_profile_slice = rho_sorted_profile.sel(z_1d_sorted=displacement_slice)
    b_l = rho - rho_sorted_profile_slice
    bl_dz = b_l * dz_flat
    bl_integrated = bl_dz.sum("z_1d_sorted")

    local_ape = g * bl_integrated / rho_0
    return local_ape
#---

#+++ Calculate local APE using cumulative integral method
def cumulative_method_local_APE(vertically_sorted_ds, rho, displacement, displacement_slice, z, z_0):
    """
    Calculate local APE using cumulative integrals (fast method)

    Parameters
    ----------
    vertically_sorted_ds : xr.Dataset
        Dataset containing cumulative integrals of sorted density and dz
    rho : float
        Density at current position
    displacement : float
        z - z_0 (vertical displacement)
    displacement_slice : slice
        Slice object for selecting the displacement range
    z : float
        Current z coordinate
    z_0 : float
        Reference z coordinate (sorted position)

    Returns
    -------
    float
        Local APE value: g * (rho * ∫dz - ∫rho_sorted dz) / rho_0
    """
    # Get cumulative integral of sorted density profile
    cumulative_rho_sorted_integral = vertically_sorted_ds["rho_1d_sorted_cumulative_integral"].sel(z_1d_sorted=displacement_slice)
    rho_sorted_integral = np.sign(displacement) * (
        cumulative_rho_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
        cumulative_rho_sorted_integral.sel(z_1d_sorted=z_0)
    )

    # Get cumulative integral of dz
    cumulative_dz_sorted_integral = vertically_sorted_ds["dz_1d_sorted_cumulative_integral"].sel(z_1d_sorted=displacement_slice)
    dz_integral = np.sign(displacement) * (
        cumulative_dz_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
        cumulative_dz_sorted_integral.sel(z_1d_sorted=z_0)
    )

    # Calculate local APE
    rho_constant_integral = rho * dz_integral
    local_ape = g * (rho_constant_integral - rho_sorted_integral) / rho_0

    return local_ape
#---

#+++ Calculate kinetic energy
def calculate_kinetic_energy(u, v, w):
    """
    Calculate local kinetic energy density

    Parameters
    ----------
    u, v, w : xr.DataArray
        Velocity components

    Returns
    -------
    xr.DataArray
        Local KE density: 0.5 * rho_0 * (u^2 + v^2 + w^2)
    """
    return 0.5 * rho_0 * (u**2 + v**2 + w**2)

def integrated_kinetic_energy(ds, time_idx=None):
    """
    Calculate volume-integrated kinetic energy

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing velocity fields (u, v, w)
    time_idx : int, optional
        Time index to select. If None, integrates over all times.

    Returns
    -------
    float or xr.DataArray
        Integrated KE (scalar if time_idx given, time series otherwise)
    """
    if time_idx is not None:
        u = ds.u.isel(time=time_idx)
        v = ds.v.isel(time=time_idx)
        w = ds.w.isel(time=time_idx)
        dV = ds.dV
    else:
        u = ds.u
        v = ds.v
        w = ds.w
        dV = ds.dV

    ke_local = calculate_kinetic_energy(u, v, w)
    return (ke_local * dV).sum(("x_caa", "y_aca", "z_aac"))

def calculate_ke_timeseries(ds):
    """
    Calculate volume-integrated KE for all time steps

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing velocity fields

    Returns
    -------
    xr.DataArray
        Time series of volume-integrated KE
    """
    print("Calculating KE time series...")
    KE = integrated_kinetic_energy(ds)
    print("\nDone!")
    return KE
#---
