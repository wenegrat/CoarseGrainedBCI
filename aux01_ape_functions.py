"""
Energy calculation functions for Available Potential Energy (APE) analysis

This module contains functions for calculating APE using the sorting method
following Winters et al. (1995).
"""

import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
import warnings
from aux00_utils import integrate

# Physical constants
g = 9.81  # gravitational acceleration [m/s^2]
ρ0 = 1025  # reference density [kg/m^3]

#+++ Calculate density fields (preprocessing step)
def calculate_density_fields_from_buoyancy(ds, ρ_ref=ρ0, buoyancy_name="b", density_name="rho"):
    """
    Calculate density and related fields from buoyancy

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing buoyancy field
    ρ_ref : float, optional
        Reference density [kg/m^3], default is global ρ0
    buoyancy_name : str, optional
        Name of the buoyancy field in the dataset, default is "b"
    density_name : str, optional
        Name for the calculated density field, default is "rho"

    Returns
    -------
    ds : xr.Dataset
        Dataset with added fields: {density_name}, {density_name}_z, Z
    """
    # Convert buoyancy to density
    # b = g * (ρ0 - rho) / ρ0  =>  rho = ρ0 * (1 - b/g)
    ds[density_name] = ρ_ref * (1 - ds[buoyancy_name] / g)
    ds[f"{density_name}_z"] = (ρ_ref * ds.z_aac + ds.pe / g)  # pe = -b*z

    # Add coordinate arrays
    if "z_aac" in ds.coords:
        ds["Z"] = ds[density_name] * 0 + ds.z_aac
    else:
        print("Warning: z_aac coordinate not found, trying to infer from data")

    return ds
#---

#+++ Vertical sort density by flattening, sorting, and reshaping
def vertical_sort_density_by_flattening(rho, dV, LxLy, test=False, z_min=0, Lz=None):
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

#+++ Vertical sort density by the PDF method
def vertical_sort_density_by_PDF(rho, Lz, nbins=1000):
    ρ_1d = np.ravel(rho.values, order="C")
    ε = 1e-3
    bin_edges = np.linspace(ρ_1d.min()-ε, ρ_1d.max()+ε, nbins+1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P_rho, bins_count = np.histogram(ρ_1d, bins=bin_edges, density=True)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    Z_r = Lz * cumulative_trapezoid(P_rho[::-1], x=bin_centers[::-1], initial=0)
    Z_r = Z_r - Z_r.mean()
    return xr.DataArray(Z_r, dims="z_1d_sorted", coords=dict(z_1d_sorted=bin_centers))
#---

#+++ Calculate TPE
def local_TPE(rho, z_name="z_aac"):
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
    return g * rho * rho[z_name]

def integrated_total_potential_energy(rho, dV=None, ds=None, z_name="z_aac"):
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
    tpe_local = local_TPE(rho, z_name=z_name)
    return integrate(tpe_local, dV)
#---

#+++ Calculate reference state using sorting method
def calculate_reference_potential_energy_profile(rho, dV, LxLy, test=False, z_min=0, Lz=None, sorting_method="vertically_flattened"):
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
    if sorting_method == "vertically_flattened":
        vertically_sorted_ds = vertical_sort_density_by_flattening(rho, dV, LxLy, test=test, z_min=z_min, Lz=Lz)[0]

        if test:
            assert(all(np.diff(vertically_sorted_ds.rho_1d_sorted) <= 0))
            assert(all(np.diff(vertically_sorted_ds.z_1d_sorted) > 0))
            if Lz is not None:
                assert(np.sum(vertically_sorted_ds.dz_1d_sorted).values == Lz)

    elif sorting_method == "PDF":
        vertically_sorted_ds = vertical_sort_density_by_PDF(rho, Lz, nbins=1000)[0]
    else:
        raise ValueError(f"Invalid sorting method: {sorting_method}")

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

def integrated_potential_energies(ds, time_idx, test=False, sorting_method="vertically_flattened",
                                  density_name="rho", dV_name="dV", LxLy_name="LxLy",
                                  z_min_name="z_min", Lz_name="Lz", z_name="z_aac"):
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
    sorting_method : str
        Method for sorting density
    density_name : str
        Name of density field in dataset
    dV_name : str
        Name of volume element field
    LxLy_name : str
        Name of horizontal area field/attribute
    z_min_name : str
        Name of minimum z attribute
    Lz_name : str
        Name of vertical extent attribute
    z_name : str
        Name of vertical coordinate

    Returns
    -------
    tuple
        (APE, TPE, RPE) - all volume-integrated scalars
    """
    rho = ds[density_name].isel(time=time_idx)
    dV = ds[dV_name]
    LxLy = ds[LxLy_name]
    z_min = ds.attrs[z_min_name] if isinstance(z_min_name, str) else z_min_name
    Lz = ds.attrs[Lz_name] if isinstance(Lz_name, str) else Lz_name

    TPE = integrated_total_potential_energy(rho, dV=dV, z_name=z_name)
    vertically_sorted_ds = calculate_reference_potential_energy_profile(
        rho, dV, LxLy, test=test, z_min=z_min, Lz=Lz, sorting_method=sorting_method
    )
    RPE = integrated_reference_potential_energy(vertically_sorted_ds, LxLy if isinstance(LxLy, float) else LxLy.values)
    APE = TPE - RPE
    return APE, TPE, RPE
#---

#+++ Integrated APE, TPE, RPE time series calculations
def integrated_potential_energies_timeseries(ds, test=False, verbose_level=1, sorting_method="vertically_flattened",
                                             density_name="rho", dV_name="dV", LxLy_name="LxLy",
                                             z_min_name="z_min", Lz_name="Lz", z_name="z_aac"):
    """
    Calculate volume-integrated potential energies for all time steps

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing simulation data
    test : bool
        Whether to run tests
    verbose_level : int
        Verbosity level
    sorting_method : str
        Method for sorting density
    density_name : str
        Name of density field in dataset
    dV_name : str
        Name of volume element field
    LxLy_name : str
        Name of horizontal area field/attribute
    z_min_name : str
        Name of minimum z attribute
    Lz_name : str
        Name of vertical extent attribute
    z_name : str
        Name of vertical coordinate

    Returns
    -------
    potential_energies_ds : xr.Dataset
        Dataset containing:
        - APE: 1D DataArray (time) with APE values
        - TPE: 1D DataArray (time) with TPE values
        - RPE: 1D DataArray (time) with RPE values
    """
    if verbose_level > 0: print("Calculating potential energies time series...")

    n_times = len(ds.time)
    APE = np.zeros(n_times)
    TPE = np.zeros(n_times)
    RPE = np.zeros(n_times)

    for i in range(n_times):
        if verbose_level > 0: print(f"  Processing time step {i+1}/{n_times}", end="\r")
        APE[i], TPE[i], RPE[i] = integrated_potential_energies(
            ds, i, test=test, sorting_method=sorting_method,
            density_name=density_name, dV_name=dV_name, LxLy_name=LxLy_name,
            z_min_name=z_min_name, Lz_name=Lz_name, z_name=z_name
        )

    if verbose_level > 0: print("\nDone!")

    potential_energies_ds = xr.Dataset(dict(APE=("time", APE), TPE=("time", TPE), RPE=("time", RPE)))
    potential_energies_ds["time"] = ds.time
    return potential_energies_ds
#---

#+++ Create inverse lookup table for fast z_0 retrieval (if needed)
def _create_inverse_sort_lookup(vertically_sorted_ds, verbose=False):
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
    if verbose: print("Creating inverse lookup table for fast z_0 retrieval...", end="")
    inverse_sort_indices = np.empty(len(vertically_sorted_ds.sort_indices_1d), dtype=int)
    for i, idx in enumerate(vertically_sorted_ds.sort_indices_1d.values):
        inverse_sort_indices[int(idx)] = i
    z_1d_sorted_values = vertically_sorted_ds.z_1d_sorted.values
    if verbose: print("Done!")

    return inverse_sort_indices, z_1d_sorted_values
#---

#+++ Local APE calculations using on-the-fly integral method
def _local_APE_on_the_fly_integral_xarray(ρ, z, vertically_sorted_ds):
    """
    Compute APE for a single point using summation method (xarray inputs)

    This function is designed to be called by xr.apply_ufunc with vectorize=True

    Parameters
    ----------
    ρ : xr.DataArray
        Density field
    z : xr.DataArray
        Z coordinate
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    ρ0 : float
        Reference density
    """
    # Get z_0
    ρ_sorted_profile = vertically_sorted_ds.rho_1d_sorted
    z_possibilities = ρ_sorted_profile.where(ρ_sorted_profile == ρ, drop=True).z_1d_sorted
    z_0 = z_possibilities[abs(z_possibilities - z).argmin()]

    # Calculate displacement and slice
    if z > z_0:
        displacement_slice = slice(z_0, z)
        signed_dz_flat = +vertically_sorted_ds.dz_1d_sorted.sel(z_1d_sorted=displacement_slice)
    else:
        displacement_slice = slice(z, z_0)
        signed_dz_flat = -vertically_sorted_ds.dz_1d_sorted.sel(z_1d_sorted=displacement_slice)

    # Calculate buoyancy difference and integrate
    ρ_sorted_profile_slice = ρ_sorted_profile.sel(z_1d_sorted=displacement_slice)

    b_l = - g * (ρ - ρ_sorted_profile_slice) / ρ0

    return -ρ0 * (b_l * signed_dz_flat).sum("z_1d_sorted") # Convert to APE by unit of volume


def _local_APE_on_the_fly_integral_numpy(ρ, z, vertically_sorted_ds):
    """
    Compute APE for a single point using summation method (scalar inputs) according to the
    definition:

    E_a = -ρ0 ∫_{z_0}^{z} b_l dz = g ∫_{z_0}^{z} (ρ - ρ_ref) dz

    Thus the output units are: kg m^2 s^-2 / m^3, which is the APE by unit of volume.

    Parameters
    ----------
    ρ : float
        Density at the point
    z : float
        Z coordinate at the point
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    ρ0 : float
        Reference density

    Returns
    -------
    float
        Local APE

    Notes
    -----
    This function is designed to be called by xr.apply_ufunc with vectorize=True
    and is optimized to use numpy array indexing instead of slow .sel() operations.

    z_0 is calculated by finding all z values where the sorted density equals ρ,
    then selecting the one closest to the current z coordinate.
    """
    # Extract numpy arrays once (faster than repeated xarray operations)
    ρ_sorted_array = vertically_sorted_ds.rho_1d_sorted.values
    dz_sorted_array = vertically_sorted_ds.dz_1d_sorted.values
    z_sorted_array = vertically_sorted_ds.z_1d_sorted.values

    # Get z_0: find all z values where sorted density equals ρ, then pick closest to z
    mask = ρ_sorted_array == ρ
    z_possibilities = z_sorted_array[mask]
    z_0 = z_possibilities[np.abs(z_possibilities - z).argmin()]

    # Find integer indices using binary search (much faster than .sel())
    idx_z = np.searchsorted(z_sorted_array, z)
    idx_z0 = np.searchsorted(z_sorted_array, z_0)

    # Determine slice indices based on displacement direction
    if z > z_0:
        idx_start, idx_end = idx_z0, idx_z
        signed_dz_slice = +dz_sorted_array[idx_start:idx_end]
    else:
        idx_start, idx_end = idx_z, idx_z0
        signed_dz_slice = -dz_sorted_array[idx_start:idx_end]

    # Fast numpy array slicing (much faster than .sel())
    ρ_sorted_slice = ρ_sorted_array[idx_start:idx_end]

    # Calculate buoyancy difference and integrate
    b_l = -g * (ρ - ρ_sorted_slice) / ρ0
    integral = np.sum(b_l * signed_dz_slice)
    return -ρ0 * integral # Convert to APE by unit of volume


def vectorized_local_APE_on_the_fly_integral(ds0, vertically_sorted_ds, use_numpy_version=True, verbose=True, z_name="z_aac"):
    """
    Vectorized calculation of local APE using summation method for all grid points

    Uses optimized numpy/numba implementation for maximum performance.

    Parameters
    ----------
    ds0 : xr.Dataset
        Dataset at a single time containing rho field
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    use_numpy_version : bool
        Whether to use the numpy optimized version
    verbose : bool
        Whether to print progress messages
    z_name : str
        Name of vertical coordinate

    Returns
    -------
    xr.DataArray
        Local APE values with same dimensions as ds0.rho
    """
    if use_numpy_version:
        if verbose:
            print("Computing local APE using fast cumulative integral method...")
    else:
        if verbose:
            print("Computing local APE using xarray method...")

    if use_numpy_version:
        # Broadcast z coordinates to match rho shape
        z_broadcast = xr.zeros_like(ds0.rho) + ds0[z_name]

        # Use apply_ufunc with vectorize=True to apply the scalar function to all points
        result = xr.apply_ufunc(
            _local_APE_on_the_fly_integral_numpy,
            ds0.rho,
            z_broadcast,
            vectorize = True,
            dask = "allowed",
            kwargs = dict(
                vertically_sorted_ds = vertically_sorted_ds,
            )
        )
    else:
        # Broadcast z coordinates to match rho shape
        z_broadcast = xr.zeros_like(ds0.rho) + ds0[z_name]

        # Use apply_ufunc with vectorize=True to apply the scalar function to all points
        result = xr.apply_ufunc(_local_APE_on_the_fly_integral_xarray,
                                ds0.rho,
                                z_broadcast,
                                vectorize = True,
                                dask = "allowed",
                                kwargs = dict(vertically_sorted_ds = vertically_sorted_ds))

    if verbose: print("Done!")
    return result
#---

#+++ Local APE calculations using precomputed integral method
def _local_APE_precomputed_integral_numpy(ρ_3d, z_3d, ρ_sorted_array, z_sorted_array, cumulative_rho_dz, cumulative_dz):
    """
    Compute APE for all points using cumulative integrals - fully vectorized, no loops!

    This is the fastest method - uses precomputed cumulative integrals so each point
    calculation becomes just array lookups and arithmetic operations.

    Strategy: Use broadcasting and vectorized operations to compute all points at once.

    Parameters
    ----------
    cumulative_rho_dz : np.ndarray
        Precomputed cumulative integral of ρ * dz, with leading 0
    cumulative_dz : np.ndarray
        Precomputed cumulative integral of dz, with leading 0
    """

    # Flatten inputs
    z_flat = z_3d.ravel()
    ρ_flat = ρ_3d.ravel()
    n_points = len(z_flat)

    # Find z indices for all points at once
    z_indices = np.searchsorted(z_sorted_array, z_flat)

    # Find z_0 for each point - this is the bottleneck
    # We need to find, for each point's density, the closest z in sorted profile
    z_0_indices = np.zeros(n_points, dtype=np.int32)

    # Get unique densities to process in batches
    unique_densities = np.unique(ρ_flat)

    for ρ_val in unique_densities:
        # All points with this density
        point_mask = (ρ_flat == ρ_val)
        n_points_with_density = np.sum(point_mask)

        if n_points_with_density == 0:
            continue

        # All sorted z positions with this density
        density_mask = (ρ_sorted_array == ρ_val)
        z_possibilities_idx = np.where(density_mask)[0]

        if len(z_possibilities_idx) == 0:
            z_0_indices[point_mask] = 0  # Fallback
            continue

        z_possibilities = z_sorted_array[z_possibilities_idx]

        # For all points with this density, find closest z_0 (vectorized)
        z_points = z_flat[point_mask]

        # Broadcasting: (n_points_with_density, n_possibilities)
        distances = np.abs(z_points[:, np.newaxis] - z_possibilities[np.newaxis, :])
        closest_possibility_idx = np.argmin(distances, axis=1)
        z_0_indices[point_mask] = z_possibilities_idx[closest_possibility_idx]

    # Now compute APE for all points at once (fully vectorized!)
    # Get cumulative integrals at z and z_0 for all points
    rho_integral_at_z = cumulative_rho_dz[z_indices]
    rho_integral_at_z0 = cumulative_rho_dz[z_0_indices]
    dz_integral_at_z = cumulative_dz[z_indices]
    dz_integral_at_z0 = cumulative_dz[z_0_indices]

    # Simpler approach: just compute absolute difference, then apply sign
    rho_integral = np.abs(rho_integral_at_z - rho_integral_at_z0)
    dz_integral = np.abs(dz_integral_at_z - dz_integral_at_z0)

    # Sign convention: positive if z > z_0, negative if z < z_0
    sign = np.where(z_indices > z_0_indices, 1.0, -1.0)
    ape_flat = sign * g * (ρ_flat * dz_integral - rho_integral)

    return ape_flat.reshape(ρ_3d.shape)


def _local_APE_precomputed_integral_xarray(ρ, z, vertically_sorted_ds):
    """
    Compute APE for a single point using cumulative integral method (scalar inputs) according to the
    definition:

    E_a = g ∫_{z_0}^{z} (ρ - ρ_ref) dz

    Thus the output units are: kg m^2 s^-2 / m^3, which is the APE by unit of volume.

    Parameters
    ----------
    ρ : xr.DataArray
        Density field
    z : xr.DataArray
        Z coordinate
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz

    Returns
    -------
    float
        Local APE

    Notes
    -----
    This function is designed to be called by xr.apply_ufunc with vectorize=True.

    z_0 is calculated by finding all z values where the sorted density equals ρ,
    then selecting the one closest to the current z coordinate.
    """
    # Get z_0: find all z values where sorted density equals ρ, then pick closest to z
    ρ_sorted_profile = vertically_sorted_ds.rho_1d_sorted
    z_possibilities = ρ_sorted_profile.where(ρ_sorted_profile == ρ, drop=True).z_1d_sorted
    z_0 = z_possibilities[abs(z_possibilities - z).argmin()]

    # Get cumulative integral of sorted density profile
    cumulative_ρ_sorted_integral = vertically_sorted_ds["rho_1d_sorted_cumulative_integral"]
    ρ_sorted_integral = (cumulative_ρ_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
                         cumulative_ρ_sorted_integral.sel(z_1d_sorted=z_0))

    # Get cumulative integral of dz
    cumulative_dz_sorted_integral = vertically_sorted_ds["dz_1d_sorted_cumulative_integral"]
    dz_integral = (cumulative_dz_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
                   cumulative_dz_sorted_integral.sel(z_1d_sorted=z_0))

    # Calculate local APE
    ρ_constant_integral = ρ * dz_integral
    local_ape = g * (ρ_constant_integral - ρ_sorted_integral)

    return float(local_ape)

def vectorized_local_APE_precomputed_integral(ds0, vertically_sorted_ds, use_numpy_version=True, verbose=False, z_name="z_aac"):
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
    use_numpy_version : bool
        Whether to use numpy optimized version
    verbose : bool
        Whether to print progress messages
    z_name : str
        Name of vertical coordinate

    Returns
    -------
    xr.DataArray
        Local APE values with same dimensions as ds0.rho
    """
    if verbose: print("Computing local APE using vectorized cumulative method (xr.apply_ufunc)...")

    # Broadcast z coordinates to match rho shape
    z_broadcast = xr.zeros_like(ds0.rho) + ds0[z_name]

    # Use apply_ufunc with vectorize=True to apply the scalar function to all points
    if use_numpy_version:
        # Extract sorted arrays for direct call to numpy function
        ρ_sorted_array = vertically_sorted_ds.rho_1d_sorted.values
        dz_sorted_array = vertically_sorted_ds.dz_1d_sorted.values
        z_sorted_array = vertically_sorted_ds.z_1d_sorted.values

        # Precompute cumulative integrals once (with leading 0)
        cumulative_rho_dz = np.concatenate([[0], np.cumsum(ρ_sorted_array * dz_sorted_array)])
        cumulative_dz = np.concatenate([[0], np.cumsum(dz_sorted_array)])

        # Call numpy function directly with 3D arrays
        ape_values = _local_APE_precomputed_integral_numpy(
            ds0.rho.values,
            z_broadcast.values,
            ρ_sorted_array,
            z_sorted_array,
            cumulative_rho_dz,
            cumulative_dz
        )

        # Wrap result in xarray
        result = xr.DataArray(
            ape_values,
            dims=ds0.rho.dims,
            coords=ds0.rho.coords,
            name='local_ape'
        )
    else:
        result = xr.apply_ufunc(
            _local_APE_precomputed_integral_xarray,
            ds0.rho,
            z_broadcast,
            vectorize=True,
            dask="allowed",
            kwargs={
                "vertically_sorted_ds": vertically_sorted_ds,
            }
        )

    if verbose: print("Done!")
    return result
#---

#+++ Local APE and TPE time series calculations
def local_potential_energies_timeseries(ds, test=False, verbose_level=1, sorting_method="vertically_flattened",
                                        ape_method="on_the_fly", use_numpy_version=True,
                                        density_name="rho", dV_name="dV", LxLy_name="LxLy",
                                        z_min_name="z_min", Lz_name="Lz", z_name="z_aac"):
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
    verbose_level : int
        Verbosity level (0=quiet, 1=progress, 2=detailed)
    sorting_method : str
        Method for sorting density ("sorting" or "PDF")
    ape_method : str
        Method for computing local APE ("on_the_fly" or "precomputed_integral")
    use_numpy_version : bool
        Whether to use numpy-optimized version (only for on_the_fly method)
    density_name : str
        Name of density field in dataset
    dV_name : str
        Name of volume element field
    LxLy_name : str
        Name of horizontal area field/attribute
    z_min_name : str
        Name of minimum z attribute
    Lz_name : str
        Name of vertical extent attribute
    z_name : str
        Name of vertical coordinate

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - local_ape: 4D DataArray (time, x, y, z) with local APE values
        - TPE: 1D DataArray (time) with total potential energy
    """
    if verbose_level > 0: print("Calculating local APE and TPE time series...")

    n_times = len(ds.time)

    # Get grid info
    dV = ds[dV_name]
    LxLy = ds[LxLy_name]
    z_min = ds.attrs[z_min_name] if isinstance(z_min_name, str) else z_min_name
    Lz = ds.attrs[Lz_name] if isinstance(Lz_name, str) else Lz_name

    # Initialize list to store local APE fields for each time
    local_ape_list = []
    local_rho_sorted_list = []
    local_dz_sorted_list = []

    for i in range(n_times):
        if verbose_level > 0: print(f"  Processing time step {i+1}/{n_times}", end="\r")

        # Get data for this time step
        ds_t = ds.isel(time=i)
        rho_t = ds_t[density_name]

        # Perform vertical sorting
        if sorting_method == "vertically_flattened":
            vertically_sorted_ds, threed_sorted_ds = vertical_sort_density_by_flattening(
                rho_t, dV, LxLy, test=test, z_min=z_min, Lz=Lz
            )
        elif sorting_method == "PDF":
            vertically_sorted_ds = vertical_sort_density_by_PDF(rho_t, Lz, nbins=1000)
        else:
            raise ValueError(f"Invalid sorting method: {sorting_method}")

        # Create a temporary dataset with the density field for APE calculation
        ds_t_with_rho = ds_t.copy()
        ds_t_with_rho["rho"] = rho_t
        ds_t_with_rho = ds_t_with_rho.assign_coords({z_name: ds_t[z_name]})

        # Calculate local APE field using selected method
        if ape_method == "on_the_fly":
            local_ape = vectorized_local_APE_on_the_fly_integral(
                ds_t_with_rho, vertically_sorted_ds,
                use_numpy_version=use_numpy_version,
                verbose=verbose_level > 1,
                z_name=z_name,
            )
        elif ape_method == "precomputed_integral":
            # Compute cumulative integrals for precomputed method
            # ∫_0^z ρ(z') dz' = cumsum(ρ * dz)
            vertically_sorted_ds["rho_1d_sorted_cumulative_integral"] = (
                vertically_sorted_ds.rho_1d_sorted * vertically_sorted_ds.dz_1d_sorted
            ).cumsum("z_1d_sorted")
            # ∫_0^z dz' = cumsum(dz)
            vertically_sorted_ds["dz_1d_sorted_cumulative_integral"] = (
                vertically_sorted_ds.dz_1d_sorted.cumsum("z_1d_sorted")
            )

            local_ape = vectorized_local_APE_precomputed_integral(
                ds_t_with_rho, vertically_sorted_ds,
                verbose=verbose_level > 1,
                use_numpy_version=use_numpy_version,
                z_name=z_name,
            )
        else:
            raise ValueError(f"Invalid ape_method: {ape_method}. Must be 'on_the_fly' or 'precomputed_integral'")

        # Append to lists in order to concatenate later
        local_ape_list.append(local_ape)
        local_rho_sorted_list.append(vertically_sorted_ds.rho_1d_sorted)
        local_dz_sorted_list.append(vertically_sorted_ds.dz_1d_sorted)

    if verbose_level > 0: print("\nDone!")

    # Concatenate local APE fields along time dimension
    local_ape_4d = xr.concat(local_ape_list, dim="time")
    local_ape_4d["time"] = ds.time

    local_rho_sorted_4d = xr.concat(local_rho_sorted_list, dim="time")
    local_rho_sorted_4d["time"] = ds.time

    local_dz_sorted_4d = xr.concat(local_dz_sorted_list, dim="time")
    local_dz_sorted_4d["time"] = ds.time

    tpe = local_TPE(ds[density_name], z_name=z_name)
    rpe = local_TPE(local_rho_sorted_4d, z_name="z_1d_sorted")

    # Combine into a Dataset
    local_potential_energies_ds = xr.Dataset(dict(
        ape = local_ape_4d,
        tpe = tpe,
        rpe = rpe,
        rho_sorted = local_rho_sorted_4d,
        dz_sorted = local_dz_sorted_4d,
    ))

    return local_potential_energies_ds
#---

#+++ Calculate kinetic energy
def local_KE(u, v, w):
    """
    Calculate local kinetic energy density

    Parameters
    ----------
    u, v, w : xr.DataArray
        Velocity components

    Returns
    -------
    xr.DataArray
        Local KE density: ρ0 * (u^2 + v^2 + w^2) / 2
    """
    return ρ0 * (u**2 + v**2 + w**2) / 2

def integrated_KE(ds, u_name="u", v_name="v", w_name="w", dV_name="dV",
                  x_dim="x_caa", y_dim="y_aca", z_dim="z_aac"):
    """
    Calculate volume-integrated kinetic energy

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing velocity fields
    u_name : str
        Name of u velocity component
    v_name : str
        Name of v velocity component
    w_name : str
        Name of w velocity component
    dV_name : str
        Name of volume element field
    x_dim, y_dim, z_dim : str
        Names of spatial dimensions

    Returns
    -------
    xr.DataArray
        Integrated KE
    """
    u = ds[u_name]
    v = ds[v_name]
    w = ds[w_name]
    dV = ds[dV_name]

    ke = local_KE(u, v, w)
    KE = (ke * dV).sum((x_dim, y_dim, z_dim))
    return KE

def integrated_KE_timeseries(ds, verbose=False, u_name="u", v_name="v", w_name="w",
                             dV_name="dV", x_dim="x_caa", y_dim="y_aca", z_dim="z_aac"):
    """
    Calculate volume-integrated KE for all time steps

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing velocity fields
    verbose : bool
        Whether to print progress
    u_name : str
        Name of u velocity component
    v_name : str
        Name of v velocity component
    w_name : str
        Name of w velocity component
    dV_name : str
        Name of volume element field
    x_dim, y_dim, z_dim : str
        Names of spatial dimensions

    Returns
    -------
    xr.DataArray
        Time series of volume-integrated KE
    """
    if verbose: print("Calculating KE time series...")
    KE = integrated_KE(ds, u_name=u_name, v_name=v_name, w_name=w_name,
                      dV_name=dV_name, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
    if verbose: print("\nDone!")
    return KE
#---
