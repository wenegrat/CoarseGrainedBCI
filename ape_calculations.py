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
def calculate_local_TPE(rho):
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
    tpe_local = calculate_local_TPE(rho)
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

#+++ Calculate local APE and TPE time series
def calculate_local_energies_timeseries(ds, test=False):
    """
    Calculate local APE and TPE fields for all time steps

    This function computes the 3D local APE field and scalar TPE value
    for each time step in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing simulation data with time dimension
    test : bool
        Whether to run tests on the sorting

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - local_ape: 4D DataArray (time, x, y, z) with local APE values
        - TPE: 1D DataArray (time) with total potential energy
    """
    print("Calculating local APE and TPE time series...")

    n_times = len(ds.time)

    # Initialize list to store local APE fields for each time
    local_ape_list = []
    TPE_array = np.zeros(n_times)

    for i in range(n_times):
        print(f"  Processing time step {i+1}/{n_times}", end="\r")

        # Get data for this time step
        ds_t = ds.isel(time=i)

        # Perform vertical sorting
        vertically_sorted_ds, threed_sorted_ds = vertical_sort_density(
            ds_t.rho, ds_t.dV, ds.LxLy, test=test, z_min=ds.z_min, Lz=ds.Lz
        )

        # Create inverse lookup table
        inverse_sort_indices, z_1d_sorted_values = create_inverse_sort_lookup(vertically_sorted_ds)

        # Calculate local APE field
        local_ape = vectorized_summation_method_local_APE(
            ds_t, vertically_sorted_ds, threed_sorted_ds,
            inverse_sort_indices, z_1d_sorted_values
        )
        local_ape_list.append(local_ape)

        # Calculate TPE
        TPE_array[i] = integrated_total_potential_energy(ds_t.rho, ds=ds_t)

    print("\nDone!")

    # Concatenate local APE fields along time dimension
    local_ape_4d = xr.concat(local_ape_list, dim="time")
    local_ape_4d["time"] = ds.time

    # Create TPE DataArray
    TPE = xr.DataArray(TPE_array, dims="time", coords=dict(time=ds.time))

    # Combine into a Dataset
    result = xr.Dataset({
        "local_ape": local_ape_4d,
        "TPE": TPE
    })

    return result
#---

#+++ Calculate APE time series
def calculate_energies_timeseries(ds, test=False):
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
    print("Calculating energies time series...")

    n_times = len(ds.time)
    APE = np.zeros(n_times)
    TPE = np.zeros(n_times)
    RPE = np.zeros(n_times)

    for i in range(n_times):
        print(f"  Processing time step {i+1}/{n_times}", end="\r")
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

#+++ Local APE calculations using summation method
def _summation_local_APE_xarray(rho, z, density_index, vertically_sorted_ds, inverse_sort_indices, z_1d_sorted_values):
    """
    Compute APE for a single point using summation method (xarray inputs)

    This function is designed to be called by xr.apply_ufunc with vectorize=True

    Parameters
    ----------
    rho : xr.DataArray
        Density field
    z : xr.DataArray
        Z coordinate
    density_index : int
        Index of the density in the sorted array
    """
    # Get z_0 using inverse lookup
    sorted_position = inverse_sort_indices[int(density_index)]
    z_0 = z_1d_sorted_values[sorted_position]

    # Calculate displacement and slice
    displacement = z - z_0
    if displacement > 0:
        displacement_slice = slice(z_0, z)
    else:
        displacement_slice = slice(z, z_0)

    # Calculate buoyancy difference and integrate
    rho_sorted_profile = vertically_sorted_ds.rho_1d_sorted
    rho_sorted_profile_slice = rho_sorted_profile.sel(z_1d_sorted=displacement_slice)

    b_l = - g * (rho - rho_sorted_profile_slice)

    dz_flat = vertically_sorted_ds.dz_1d_sorted.sel(z_1d_sorted=displacement_slice)
    return -(b_l * dz_flat).sum("z_1d_sorted")

def _summation_local_APE_numpy(rho, z, density_index, vertically_sorted_ds, inverse_sort_indices, z_1d_sorted_values):
    """
    Compute APE for a single point using summation method (scalar inputs) according to the
    definition:

    E_a = -∫_{z_0}^{z} b_l dz = g ∫_{z_0}^{z} (ρ - ρ_ref) dz

    Thus the output units are: kg m^2 s^-2 / m^3, which is the APE by unit of volume.

    Parameters
    ----------
    rho : xr.DataArray
        Density field
    z : xr.DataArray
        Z coordinate
    density_index : int
        Index of the density in the sorted array
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    inverse_sort_indices : np.ndarray
        Inverse lookup table mapping density_index to sorted position
    z_1d_sorted_values : np.ndarray
        Z coordinate values in sorted order

    Returns
    -------
    float
        Local APE

    Notes
    -----
    This function is designed to be called by xr.apply_ufunc with vectorize=True
    and is optimized to use numpy array indexing instead of slow .sel() operations
    """
    # Get z_0 using inverse lookup
    sorted_position = inverse_sort_indices[int(density_index)]
    z_0 = z_1d_sorted_values[sorted_position]

    # Extract numpy arrays once (faster than repeated xarray operations)
    rho_sorted_array = vertically_sorted_ds.rho_1d_sorted.values
    dz_sorted_array = vertically_sorted_ds.dz_1d_sorted.values
    z_sorted_array = vertically_sorted_ds.z_1d_sorted.values

    # Find integer indices using binary search (much faster than .sel())
    idx_z = np.searchsorted(z_sorted_array, z)
    idx_z0 = sorted_position

    # Determine slice indices based on displacement direction
    if z > z_0:
        idx_start, idx_end = idx_z0, idx_z
    else:
        idx_start, idx_end = idx_z, idx_z0

    # Fast numpy array slicing (much faster than .sel())
    if idx_end > idx_start:
        rho_sorted_slice = rho_sorted_array[idx_start:idx_end]
        dz_slice = dz_sorted_array[idx_start:idx_end]

        # Calculate buoyancy difference and integrate
        b_l = -g * (rho - rho_sorted_slice)
        integral = np.sum(b_l * dz_slice)
        return -integral
    else:
        return 0.0


def vectorized_summation_method_local_APE(ds0, vertically_sorted_ds, threed_sorted_ds, inverse_sort_indices, z_1d_sorted_values, use_numpy_version=True):
    """
    Vectorized calculation of local APE using summation method for all grid points

    Uses xr.apply_ufunc with vectorize=True to apply the calculation to all points
    without explicit loops.

    Parameters
    ----------
    ds0 : xr.Dataset
        Dataset at a single time containing rho field
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    threed_sorted_ds : xr.Dataset
        Dataset containing 3D sort indices
    inverse_sort_indices : np.ndarray
        Inverse lookup table mapping density_index to sorted position
    z_1d_sorted_values : np.ndarray
        Z coordinate values in sorted order

    Returns
    -------
    xr.DataArray
        Local APE values with same dimensions as ds0.rho
    """
    print("Computing local APE using vectorized summation method (xr.apply_ufunc)...")

    # Broadcast z coordinates to match rho shape
    z_broadcast = xr.zeros_like(ds0.rho) + ds0.z_aac

    # Use apply_ufunc with vectorize=True to apply the scalar function to all points
    result = xr.apply_ufunc(
        _summation_local_APE_numpy if use_numpy_version else _summation_local_APE_xarray,
        ds0.rho,
        z_broadcast,
        threed_sorted_ds.sort_indices_3d,
        vectorize = True,
        dask = "allowed",
        kwargs = dict(
            vertically_sorted_ds = vertically_sorted_ds,
            inverse_sort_indices = inverse_sort_indices,
            z_1d_sorted_values = z_1d_sorted_values,
        )
    )

    print("Done!")
    return result
#---

#+++ local APE calculations using cumulative integral method
def _cumulative_local_APE_scalar(rho, z, density_index, vertically_sorted_ds, inverse_sort_indices, z_1d_sorted_values):
    """
    Compute APE for a single point using cumulative integral method (scalar inputs) according to the
    definition:

    E_a = g ∫_{z_0}^{z} (ρ - ρ_ref) dz

    Thus the output units are: kg m^2 s^-2 / m^3, which is the APE by unit of volume.

    Parameters
    ----------
    rho : xr.DataArray
        Density field
    z : xr.DataArray
        Z coordinate
    density_index : int
        Index of the density in the sorted array
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    inverse_sort_indices : np.ndarray
        Inverse lookup table mapping density_index to sorted position
    z_1d_sorted_values : np.ndarray
        Z coordinate values in sorted order

    Returns
    -------
    float
        Local APE

    Notes
    -----
    This function is designed to be called by xr.apply_ufunc with vectorize=True
    and is optimized to use numpy array indexing instead of slow .sel() operations
    """
    # Get z_0 using inverse lookup
    sorted_position = inverse_sort_indices[int(density_index)]
    z_0 = z_1d_sorted_values[sorted_position]

    # Calculate displacement
    displacement = z - z_0

    # Get cumulative integral of sorted density profile
    cumulative_rho_sorted_integral = vertically_sorted_ds["rho_1d_sorted_cumulative_integral"]
    rho_sorted_integral = np.sign(displacement) * (cumulative_rho_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
                                                   cumulative_rho_sorted_integral.sel(z_1d_sorted=z_0))

    # Get cumulative integral of dz
    cumulative_dz_sorted_integral = vertically_sorted_ds["dz_1d_sorted_cumulative_integral"]
    dz_integral = np.sign(displacement) * (cumulative_dz_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
                                           cumulative_dz_sorted_integral.sel(z_1d_sorted=z_0))

    # Calculate local APE
    rho_constant_integral = rho * dz_integral
    local_ape = g * (rho_constant_integral - rho_sorted_integral)

    return float(local_ape)

def vectorized_cumulative_method_local_APE(ds0, vertically_sorted_ds, threed_sorted_ds, inverse_sort_indices, z_1d_sorted_values):
    """
    Vectorized calculation of local APE using cumulative integral method for all grid points

    Uses xr.apply_ufunc with vectorize=True to apply the calculation to all points
    without explicit loops.

    Parameters
    ----------
    ds0 : xr.Dataset
        Dataset at a single time containing rho field
    vertically_sorted_ds : xr.Dataset
        Dataset containing cumulative integrals of sorted density and dz
    threed_sorted_ds : xr.Dataset
        Dataset containing 3D sort indices
    inverse_sort_indices : np.ndarray
        Inverse lookup table mapping density_index to sorted position
    z_1d_sorted_values : np.ndarray
        Z coordinate values in sorted order

    Returns
    -------
    xr.DataArray
        Local APE values with same dimensions as ds0.rho
    """
    print("Computing local APE using vectorized cumulative method (xr.apply_ufunc)...")

    # Broadcast z coordinates to match rho shape
    z_broadcast = xr.zeros_like(ds0.rho) + ds0.z_aac

    # Use apply_ufunc with vectorize=True to apply the scalar function to all points
    result = xr.apply_ufunc(
        _cumulative_local_ape_scalar,
        ds0.rho,
        z_broadcast,
        threed_sorted_ds.sort_indices_3d,
        vectorize=True,
        dask="allowed",
        kwargs={
            "vertically_sorted_ds": vertically_sorted_ds,
            "inverse_sort_indices": inverse_sort_indices,
            "z_1d_sorted_values": z_1d_sorted_values,
        }
    )

    print("Done!")
    return result
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
