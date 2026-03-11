"""
Potential energy calculation functions for Available Potential Energy (APE) analysis

This module contains functions for calculating TPE, RPE, and APE using the sorting method
following Winters et al. (1995).
"""

import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
import warnings
from aux00_utils import integrate, calculate_gradient

# Physical constants
g = 9.81  # gravitational acceleration [m/s^2]
ρ0 = 1025  # reference density [kg/m^3]

#+++ Calculate density fields (preprocessing step)
def calculate_density_fields_from_buoyancy(ds, ρ_ref=ρ0, buoyancy_name="b", density_name="rho", compute_density_z=False):
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
    compute_density_z : bool, optional
        If True, also compute {density_name}_z = ρ₀·z + pe/g (≡ ρ·z) for
        cross-checking TPE via the online pe field. Default is False.

    Returns
    -------
    ds : xr.Dataset
        Dataset with added fields: {density_name}, Z, and optionally {density_name}_z
    """
    # Convert buoyancy to density
    # b = g * (ρ0 - rho) / ρ0  =>  rho = ρ0 * (1 - b/g)
    ds[density_name] = ρ_ref * (1 - ds[buoyancy_name] / g)
    if compute_density_z:
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
    vertically_sorted_ds : xr.Dataset
        1D sorted dataset with variables: rho_1d_sorted, dz_1d_sorted,
        sort_indices_1d — all indexed by the virtual z_1d_sorted coordinate
        (cell centres in the sorted reference state, running bottom to top).
    threed_sorted_ds : xr.Dataset
        3D sorted fields in the original grid shape: rho_3d_sorted,
        z_3d_sorted, sort_indices_3d.
    """
    rho_1d = np.ravel(rho.copy(), order="C")

    dz_flat = dV / LxLy # 3D DataArray with the same shape as rho
    dz_flat_1d = np.ravel(dz_flat.values, order="C")
    if test and Lz is not None:
        assert(np.isclose(dz_flat.sum().values, Lz))

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
        Local TPE density: g * rho * z / ρ0  [m² s⁻²]
    """
    return g * rho * rho[z_name] / ρ0

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
    Compute the adiabatically sorted reference-state density profile

    Sorts rho to obtain the minimum-PE reference state and returns it as a
    1D Dataset indexed by virtual z in the sorted state.

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
                assert(np.isclose(np.sum(vertically_sorted_ds.dz_1d_sorted).values, Lz))

    elif sorting_method == "PDF":
        vertically_sorted_ds = vertical_sort_density_by_PDF(rho, Lz, nbins=1000)
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
                                  z_min_name="z_min", Lz_name="Lz", z_name="z_aac",
                                  rho_to_sort=None):
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
    rho_to_sort : xr.DataArray, optional
        Density field to use for sorting. If it has a "time" dimension it will
        be sliced at time_idx. If None, the density from the dataset is used.

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
    _rho_to_sort = rho_to_sort.isel(time=time_idx) if (rho_to_sort is not None and "time" in rho_to_sort.dims) else (rho_to_sort if rho_to_sort is not None else rho)
    vertically_sorted_ds = calculate_reference_potential_energy_profile(
        _rho_to_sort, dV, LxLy, test=test, z_min=z_min, Lz=Lz, sorting_method=sorting_method
    )
    RPE = integrated_reference_potential_energy(vertically_sorted_ds, LxLy if isinstance(LxLy, float) else LxLy.values)
    APE = TPE - RPE
    return APE, TPE, RPE
#---

#+++ Integrated APE, TPE, RPE time series calculations
def integrated_potential_energies_timeseries(ds, test=False, verbose_level=1, sorting_method="vertically_flattened",
                                             density_name="rho", dV_name="dV", LxLy_name="LxLy",
                                             z_min_name="z_min", Lz_name="Lz", z_name="z_aac",
                                             rho_to_sort=None):
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
    rho_to_sort : xr.DataArray, optional
        Density field to use for sorting instead of the dataset density. If it
        has a "time" dimension it will be sliced per step. If None, the density
        from the dataset is used.

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
            z_min_name=z_min_name, Lz_name=Lz_name, z_name=z_name,
            rho_to_sort=rho_to_sort,
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

#+++ z_0 lookup helpers
def _find_z0_xarray(ρ, z, ρ_sorted_profile):
    """
    Find the reference height z_0 for a single point (xarray inputs)

    z_0 is the height in the sorted reference state where density equals ρ,
    choosing the one closest to the actual height z. If ρ is not present
    exactly in the sorted profile (e.g. a filtered density not in the full
    sort), the position of the nearest density value is returned instead.

    Parameters
    ----------
    ρ : float
        Density at the point
    z : float
        Physical z coordinate at the point
    ρ_sorted_profile : xr.DataArray
        1D sorted density profile indexed by z_1d_sorted

    Returns
    -------
    z_0 : xr.DataArray
        Reference height (scalar DataArray with z_1d_sorted coordinate)
    """
    z_possibilities = ρ_sorted_profile.where(ρ_sorted_profile == ρ, drop=True).z_1d_sorted
    if len(z_possibilities) > 0:
        return z_possibilities[abs(z_possibilities - z).argmin()]
    idx_min = int(np.argmin(np.abs(ρ_sorted_profile.values - ρ)))
    return ρ_sorted_profile.z_1d_sorted[idx_min]


def _find_z0_numpy(ρ, z, ρ_sorted_array, z_sorted_array):
    """
    Find the reference height z_0 for a single point (numpy inputs)

    z_0 is the height in the sorted reference state where density equals ρ,
    choosing the one closest to the actual height z. If ρ is not present
    exactly in the sorted profile (e.g. a filtered density not in the full
    sort), the position of the nearest density value is returned instead.

    Parameters
    ----------
    ρ : float
        Density at the point
    z : float
        Physical z coordinate at the point
    ρ_sorted_array : np.ndarray
        1D sorted density profile values
    z_sorted_array : np.ndarray
        1D z coordinates corresponding to ρ_sorted_array

    Returns
    -------
    z_0 : float
        Reference height
    """
    mask = ρ_sorted_array == ρ
    if mask.any():
        z_possibilities = z_sorted_array[mask]
        return z_possibilities[np.abs(z_possibilities - z).argmin()]
    return z_sorted_array[np.argmin(np.abs(ρ_sorted_array - ρ))]
#---

#+++ Reference height (z_0) field
def calculate_z0_field_xarray(rho, vertically_sorted_ds, z_name="z_aac"):
    """
    Calculate the reference height z_0 at every grid point (xarray version)

    For each point in the 3D density field, z_0 is the height in the sorted
    reference state occupied by a fluid parcel of that density, choosing the
    candidate closest to the parcel's actual height z. Uses _find_z0_xarray
    internally via xr.apply_ufunc.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field (x, y, z)
    vertically_sorted_ds : xr.Dataset
        Sorted reference-state dataset containing rho_1d_sorted indexed by
        z_1d_sorted
    z_name : str, optional
        Name of the vertical coordinate in rho, default "z_aac"

    Returns
    -------
    z0_field : xr.DataArray
        3D field of reference heights z_0, same shape and coordinates as rho
    """
    ρ_sorted_profile = vertically_sorted_ds.rho_1d_sorted
    z_broadcast = xr.zeros_like(rho) + rho[z_name]
    return xr.apply_ufunc(
        _find_z0_xarray,
        rho,
        z_broadcast,
        vectorize=True,
        dask="allowed",
        kwargs=dict(ρ_sorted_profile=ρ_sorted_profile),
    )


def calculate_z0_field_numpy(rho, vertically_sorted_ds, z_name="z_aac"):
    """
    Calculate the reference height z_0 at every grid point (numpy version)

    For each point in the 3D density field, z_0 is the height in the sorted
    reference state occupied by a fluid parcel of that density, choosing the
    candidate closest to the parcel's actual height z. Uses _find_z0_numpy
    internally via xr.apply_ufunc.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field (x, y, z)
    vertically_sorted_ds : xr.Dataset
        Sorted reference-state dataset containing rho_1d_sorted and
        dz_1d_sorted indexed by z_1d_sorted
    z_name : str, optional
        Name of the vertical coordinate in rho, default "z_aac"

    Returns
    -------
    z0_field : xr.DataArray
        3D field of reference heights z_0, same shape and coordinates as rho
    """
    ρ_sorted_array = vertically_sorted_ds.rho_1d_sorted.values
    z_sorted_array = vertically_sorted_ds.z_1d_sorted.values
    z_broadcast = xr.zeros_like(rho) + rho[z_name]
    return xr.apply_ufunc(
        _find_z0_numpy,
        rho,
        z_broadcast,
        vectorize=True,
        dask="allowed",
        kwargs=dict(ρ_sorted_array=ρ_sorted_array, z_sorted_array=z_sorted_array),
    )


def calculate_Upsilon(z0_field, rho, z_name="z_aac"):
    """
    Calculate the buoyancy displacement potential Υ = g (z - z_0) / ρ0

    Υ represents the work done per unit mass in displacing a parcel from its
    reference height z_0 to its actual height z, against the background
    buoyancy gradient.

    Parameters
    ----------
    z0_field : xr.DataArray
        3D field of reference heights z_0, which is where the density rho is at
        neutral buoyancy in the sorted reference state (using the original non-filtered
        density for the sorting). Same shape as rho.
    rho : xr.DataArray
        Density field, used to supply the z coordinate
    z_name : str, optional
        Name of the vertical coordinate in rho, default "z_aac"

    Returns
    -------
    xr.DataArray
        3D field of Υ values [m² s⁻²], same shape as rho
    """
    return g * (rho[z_name] - z0_field) / ρ0
#---

#+++ Local APE calculations using on-the-fly integral method
def _local_APE_on_the_fly_integral_xarray(ρ, z, z_0, vertically_sorted_ds):
    """
    Compute APE for a single point using summation method (xarray inputs)

    This function is designed to be called by xr.apply_ufunc with vectorize=True

    Parameters
    ----------
    ρ : float
        Density at the point (scalar, passed by apply_ufunc with vectorize=True)
    z : float
        Physical z coordinate at the point (scalar)
    z_0 : float
        Reference height for this point (pre-computed)
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz

    Notes
    -----
    ρ0 is the module-level reference density constant, not a parameter.
    """
    ρ_sorted_profile = vertically_sorted_ds.rho_1d_sorted

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

    return -(b_l * signed_dz_flat).sum("z_1d_sorted")  # APE in Boussinesq units [m² s⁻²]



def vectorized_local_APE_on_the_fly_integral(ds0, vertically_sorted_ds, verbose=True, z_name="z_aac", z0=None):
    """
    Vectorized calculation of local APE using on-the-fly summation method for all grid points

    Parameters
    ----------
    ds0 : xr.Dataset
        Dataset at a single time containing rho field
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    verbose : bool
        Whether to print progress messages
    z_name : str
        Name of vertical coordinate
    z0 : xr.DataArray, optional
        Pre-computed reference height field (same shape as ds0.rho). If None,
        it is calculated from the sorted profile.

    Returns
    -------
    xr.DataArray
        Local APE values with same dimensions as ds0.rho
    """
    if verbose:
        print("Computing local APE using xarray on-the-fly method...")

    if z0 is None:
        z0 = calculate_z0_field_xarray(ds0.rho, vertically_sorted_ds, z_name=z_name)

    z_broadcast = xr.zeros_like(ds0.rho) + ds0[z_name]

    result = xr.apply_ufunc(
        _local_APE_on_the_fly_integral_xarray,
        ds0.rho,
        z_broadcast,
        z0,
        vectorize=True,
        dask="allowed",
        kwargs=dict(vertically_sorted_ds=vertically_sorted_ds),
    )

    if verbose: print("Done!")
    return result
#---

#+++ Local APE calculations using precomputed integral method
def _local_APE_precomputed_integral_numpy(ρ_3d, z_3d, ρ_sorted_array, z_sorted_array, cumulative_rho_dz, cumulative_dz, z_0_3d=None):
    """
    Compute APE for all points using cumulative integrals - fully vectorized, no loops!

    This is the fastest method - uses precomputed cumulative integrals so each point
    calculation becomes just array lookups and arithmetic operations.

    Strategy: Use broadcasting and vectorized operations to compute all points at once.

    Parameters
    ----------
    ρ_3d : np.ndarray
        3D density field
    z_3d : np.ndarray
        3D physical z coordinate field
    ρ_sorted_array : np.ndarray
        1D sorted density profile
    z_sorted_array : np.ndarray
        1D z coordinates of sorted profile
    cumulative_rho_dz : np.ndarray
        Precomputed cumulative integral of ρ * dz, with leading 0
    cumulative_dz : np.ndarray
        Precomputed cumulative integral of dz, with leading 0
    z_0_3d : np.ndarray, optional
        Pre-computed 3D field of reference heights z_0. If provided, the
        z_0 lookup loop is skipped entirely (significant speedup).
    """

    # Flatten inputs
    z_flat = z_3d.ravel()
    ρ_flat = ρ_3d.ravel()
    n_points = len(z_flat)

    # Find z indices for all points at once
    z_indices = np.searchsorted(z_sorted_array, z_flat)

    # Find z_0 indices — use pre-computed values if available, otherwise compute
    if z_0_3d is not None:
        z_0_indices = np.searchsorted(z_sorted_array, z_0_3d.ravel())
    else:
        z_0_indices = np.zeros(n_points, dtype=np.int32)
        unique_densities = np.unique(ρ_flat)

        for ρ_val in unique_densities:
            point_mask = (ρ_flat == ρ_val)
            if not point_mask.any():
                continue

            density_mask = (ρ_sorted_array == ρ_val)
            z_possibilities_idx = np.where(density_mask)[0]

            if len(z_possibilities_idx) == 0:
                # Density not in sorted profile (e.g. filtered ρ̄ not in full ρ sort):
                # fall back to nearest density, matching on-the-fly method behaviour
                z_0_indices[point_mask] = np.argmin(np.abs(ρ_sorted_array - ρ_val))
                continue

            z_possibilities = z_sorted_array[z_possibilities_idx]
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
    ape_flat = sign * g * (ρ_flat * dz_integral - rho_integral) / ρ0  # Boussinesq units [m² s⁻²]

    return ape_flat.reshape(ρ_3d.shape)


def _local_APE_precomputed_integral_xarray(ρ, z, z_0, vertically_sorted_ds):
    """
    Compute APE for a single point using cumulative integral method (scalar inputs) according to the
    definition:

    E_a = g ∫_{z_0}^{z} (ρ - ρ_ref) dz

    Thus the output units are: kg m^2 s^-2 / m^3, which is the APE by unit of volume.

    Parameters
    ----------
    ρ : float
        Density at the point
    z : float
        Z coordinate at the point
    z_0 : float
        Reference height for this point (pre-computed)
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz

    Returns
    -------
    float
        Local APE

    Notes
    -----
    This function is designed to be called by xr.apply_ufunc with vectorize=True.
    """

    # Get cumulative integral of sorted density profile
    cumulative_ρ_sorted_integral = vertically_sorted_ds["rho_1d_sorted_cumulative_integral"]
    ρ_sorted_integral = (cumulative_ρ_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
                         cumulative_ρ_sorted_integral.sel(z_1d_sorted=z_0))

    # Get cumulative integral of dz
    cumulative_dz_sorted_integral = vertically_sorted_ds["dz_1d_sorted_cumulative_integral"]
    dz_integral = (cumulative_dz_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
                   cumulative_dz_sorted_integral.sel(z_1d_sorted=z_0))

    # Calculate local APE in Boussinesq units [m² s⁻²]
    ρ_constant_integral = ρ * dz_integral
    local_ape = g * (ρ_constant_integral - ρ_sorted_integral) / ρ0

    return float(local_ape)

def vectorized_local_APE_precomputed_integral(ds0, vertically_sorted_ds, use_numpy_version=True, verbose=False, z_name="z_aac", z0=None):
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
    z0 : xr.DataArray, optional
        Pre-computed reference height field (same shape as ds0.rho). If None,
        it is calculated from the sorted profile.

    Returns
    -------
    xr.DataArray
        Local APE values with same dimensions as ds0.rho
    """
    if verbose: print("Computing local APE using vectorized cumulative method (xr.apply_ufunc)...")

    if z0 is None:
        if use_numpy_version:
            z0 = calculate_z0_field_numpy(ds0.rho, vertically_sorted_ds, z_name=z_name)
        else:
            z0 = calculate_z0_field_xarray(ds0.rho, vertically_sorted_ds, z_name=z_name)

    z_broadcast = xr.zeros_like(ds0.rho) + ds0[z_name]

    if use_numpy_version:
        ρ_sorted_array = vertically_sorted_ds.rho_1d_sorted.values
        dz_sorted_array = vertically_sorted_ds.dz_1d_sorted.values
        z_sorted_array = vertically_sorted_ds.z_1d_sorted.values

        # Precompute cumulative integrals once (with leading 0)
        cumulative_rho_dz = np.concatenate([[0], np.cumsum(ρ_sorted_array * dz_sorted_array)])
        cumulative_dz = np.concatenate([[0], np.cumsum(dz_sorted_array)])

        # Call numpy function directly with 3D arrays, passing pre-computed z0
        ape_values = _local_APE_precomputed_integral_numpy(
            ds0.rho.values,
            z_broadcast.values,
            ρ_sorted_array,
            z_sorted_array,
            cumulative_rho_dz,
            cumulative_dz,
            z_0_3d=z0.values,
        )

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
            z0,
            vectorize=True,
            dask="allowed",
            kwargs={"vertically_sorted_ds": vertically_sorted_ds},
        )

    if verbose: print("Done!")
    return result
#---

#+++ Local APE and TPE time series calculations
def local_potential_energies_timeseries(ds, test=False, verbose_level=1, sorting_method="vertically_flattened",
                                        ape_method="precomputed_integral", use_numpy_version=True,
                                        density_name="rho", dV_name="dV", LxLy_name="LxLy",
                                        z_min_name="z_min", Lz_name="Lz", z_name="z_aac",
                                        rho_to_sort=None):
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
    rho_to_sort : xr.DataArray, optional
        Density field to use for sorting instead of the dataset density. If it
        has a "time" dimension it will be sliced per step. If None, the density
        from the dataset is used.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - ape       : 4D DataArray (time, x, y, z) — local APE density [J m⁻³]
        - tpe       : 4D DataArray (time, x, y, z) — local TPE density g·ρ·z
        - rpe       : 2D DataArray (time, z_1d_sorted) — local RPE density in sorted state
        - rho_sorted: 2D DataArray (time, z_1d_sorted) — sorted reference density profile
        - dz_sorted : 2D DataArray (time, z_1d_sorted) — cell heights in sorted state
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
    local_z0_list = []
    local_upsilon_list = []
    local_rho_sorted_list = []
    local_dz_sorted_list = []

    for i in range(n_times):
        if verbose_level > 0: print(f"  Processing time step {i+1}/{n_times}", end="\r")

        # Get data for this time step
        ds_t = ds.isel(time=i)
        rho_t = ds_t[density_name]

        # Determine density field to sort and compute reference state
        _rho_to_sort = rho_to_sort.isel(time=i) if (rho_to_sort is not None and "time" in rho_to_sort.dims) else (rho_to_sort if rho_to_sort is not None else rho_t)
        if sorting_method == "vertically_flattened":
            _vertically_sorted_ds, threed_sorted_ds = vertical_sort_density_by_flattening(
                _rho_to_sort, dV, LxLy, test=test, z_min=z_min, Lz=Lz
            )
        elif sorting_method == "PDF":
            _vertically_sorted_ds = vertical_sort_density_by_PDF(_rho_to_sort, Lz, nbins=1000)
        else:
            raise ValueError(f"Invalid sorting method: {sorting_method}")

        # Create a temporary dataset with the density field for APE calculation
        ds_t_with_rho = ds_t.copy()
        ds_t_with_rho["rho"] = rho_t
        ds_t_with_rho = ds_t_with_rho.assign_coords({z_name: ds_t[z_name]})

        # Calculate z_0 field (used by all APE methods below)
        if use_numpy_version:
            local_z0 = calculate_z0_field_numpy(rho_t, _vertically_sorted_ds, z_name=z_name)
        else:
            local_z0 = calculate_z0_field_xarray(rho_t, _vertically_sorted_ds, z_name=z_name)

        # Calculate local APE field using selected method
        if ape_method == "on_the_fly":
            local_ape = vectorized_local_APE_on_the_fly_integral(
                ds_t_with_rho, _vertically_sorted_ds,
                verbose=verbose_level > 1,
                z_name=z_name,
                z0=local_z0,
            )
        elif ape_method == "precomputed_integral":
            # Compute cumulative integrals for precomputed method
            # ∫_0^z ρ(z') dz' = cumsum(ρ * dz)
            _vertically_sorted_ds["rho_1d_sorted_cumulative_integral"] = (_vertically_sorted_ds.rho_1d_sorted * _vertically_sorted_ds.dz_1d_sorted).cumsum("z_1d_sorted")
            # ∫_0^z dz' = cumsum(dz)
            _vertically_sorted_ds["dz_1d_sorted_cumulative_integral"] = _vertically_sorted_ds.dz_1d_sorted.cumsum("z_1d_sorted")

            local_ape = vectorized_local_APE_precomputed_integral(
                ds_t_with_rho, _vertically_sorted_ds,
                verbose=verbose_level > 1,
                use_numpy_version=use_numpy_version,
                z_name=z_name,
                z0=local_z0,
            )
        else:
            raise ValueError(f"Invalid ape_method: {ape_method}. Must be 'on_the_fly' or 'precomputed_integral'")

        local_upsilon = calculate_Upsilon(local_z0, rho_t, z_name=z_name)

        # Append to lists in order to concatenate later
        local_ape_list.append(local_ape)
        local_z0_list.append(local_z0)
        local_upsilon_list.append(local_upsilon)
        local_rho_sorted_list.append(_vertically_sorted_ds.rho_1d_sorted)
        local_dz_sorted_list.append(_vertically_sorted_ds.dz_1d_sorted)

    if verbose_level > 0: print("\nDone!")

    # Concatenate local APE fields along time dimension
    local_ape_4d = xr.concat(local_ape_list, dim="time")
    local_ape_4d["time"] = ds.time

    local_z0_4d = xr.concat(local_z0_list, dim="time")
    local_z0_4d["time"] = ds.time

    local_upsilon_4d = xr.concat(local_upsilon_list, dim="time")
    local_upsilon_4d["time"] = ds.time

    local_rho_sorted_4d = xr.concat(local_rho_sorted_list, dim="time")
    local_rho_sorted_4d["time"] = ds.time

    local_dz_sorted_4d = xr.concat(local_dz_sorted_list, dim="time")
    local_dz_sorted_4d["time"] = ds.time

    tpe = local_TPE(ds[density_name], z_name=z_name)
    rpe = local_TPE(local_rho_sorted_4d, z_name="z_1d_sorted")

    # Combine into a Dataset
    local_potential_energies_ds = xr.Dataset(dict(
        ape = local_ape_4d,
        z0 = local_z0_4d,
        upsilon = local_upsilon_4d,
        tpe = tpe,
        rpe = rpe,
        rho_sorted = local_rho_sorted_4d,
        dz_sorted = local_dz_sorted_4d,
    ))

    return local_potential_energies_ds
#---

#+++ SFS flux tensor (general)
def calculate_sfs_flux_tensor(a, b, filter, filter_dims=["x_caa", "y_aca"],
                              filtered_a=None, filtered_b=None):
    """
    Calculate the SFS flux tensor filtered(a·b) - filtered(a)·filtered(b)

    This is the general building block for subfilter-scale flux quantities: it
    measures the covariance between a and b at scales smaller than the filter
    width.

    Parameters
    ----------
    a : xr.DataArray
        First (unfiltered) field
    b : xr.DataArray
        Second (unfiltered) field
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_a : xr.DataArray, optional
        Pre-computed filtered(a). If None, it is computed from a.
    filtered_b : xr.DataArray, optional
        Pre-computed filtered(b). If None, it is computed from b.

    Returns
    -------
    xr.DataArray
        SFS flux tensor filtered(a·b) - filtered(a)·filtered(b)
    """
    if filtered_a is None:
        filtered_a = filter.apply(a, dims=filter_dims)

    if filtered_b is None:
        filtered_b = filter.apply(b, dims=filter_dims)

    return filter.apply(a * b, dims=filter_dims) - filtered_a * filtered_b
#---

#+++ Subfilter stress tensor
def calculate_subfilter_tracer_flux(rho, u_i, filter, filter_dims=["x_caa", "y_aca"],
                                    filtered_density=None, filtered_velocity_vector=None):
    """
    Calculate the subfilter stress tensor τᵢ = filtered(ρ uᵢ) - filtered(ρ) filtered(uᵢ)

    This represents the subfilter momentum flux arising from correlations between
    density and velocity fluctuations at scales smaller than the filter width.

    Parameters
    ----------
    rho : xr.DataArray
        Full (unfiltered) density field
    u_i : xr.DataArray
        Full (unfiltered) velocity vector with an "i" dimension indexing the
        three components (shape: i × time × z × y × x), as produced by
        condense_velocities()
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_density : xr.DataArray, optional
        Pre-computed filtered(ρ). If None, it is computed by applying
        filter to rho.
    filtered_velocity_vector : xr.DataArray, optional
        Pre-computed filtered(uᵢ). If None, it is computed by applying
        filter to u_i.

    Returns
    -------
    xr.DataArray
        Subfilter stress τᵢ [kg m⁻² s⁻¹] with the same dimensions as u_i
    """
    tau_i = calculate_sfs_flux_tensor(rho, u_i, filter,
                                      filter_dims=filter_dims,
                                      filtered_a=filtered_density,
                                      filtered_b=filtered_velocity_vector)
    tau_i.name = "τᵢ"
    return tau_i
#---

#+++ KE-APE exchange term
def calculate_ape_to_ke_exchange_term(w, b, filter, filter_dims=["x_caa", "y_aca"],
                                      filtered_w=None, filtered_b=None):
    """
    Calculate the SFS KE->APE exchange term +(filtered(w·b) - filtered(w)·filtered(b))

    This represents the SFS flux of KE to APE: the rate at which small-scale KE is converted to APE.

    Parameters
    ----------
    w : xr.DataArray
        Full (unfiltered) vertical velocity field
    b : xr.DataArray
        Full (unfiltered) buoyancy field
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_w : xr.DataArray, optional
        Pre-computed filtered(w). If None, it is computed from w.
    filtered_b : xr.DataArray, optional
        Pre-computed filtered(b). If None, it is computed from b.

    Returns
    -------
    xr.DataArray
        SFS KE->APE exchange term +(filtered(w·b) - filtered(w)·filtered(b))
    """
    result = calculate_sfs_flux_tensor(w, b, filter,
                                       filter_dims=filter_dims,
                                       filtered_a=filtered_w,
                                       filtered_b=filtered_b)
    result.name = "SFS KE->APE exchange"
    return result
#---

#+++ SFS APE dissipation
def calculate_sfs_ape_dissipation(rho, upsilon, upsilon_l, kappa, filter,
                                  filter_dims=["x_caa", "y_aca"],
                                  filtered_density=None, index_dim="i"):
    """
    Calculate the SFS APE dissipation ε_s = filtered(κ ∇ρ · ∇Υ) - κ ∇ρ̄ · ∇Υˡ

    The SFS APE dissipation quantifies the removal of large-scale APE by
    subfilter-scale diffusive processes:

        ε_s = filtered(κ ∇ρ · ∇Υ) - κ ∇ρ̄ · ∇Υˡ

    where:
        Υ  = g (z - z_*(ρ)) / ρ₀   — displacement potential using full density
        Υˡ = g (z - z_*(ρ̄)) / ρ₀   — displacement potential using filtered density
        κ  — diffusivity field

    Parameters
    ----------
    rho : xr.DataArray
        Full (unfiltered) density field ρ
    upsilon : xr.DataArray
        Buoyancy displacement potential Υ(ρ, z) = g(z - z_*(ρ))/ρ₀, computed
        from the full density sort (full_local_potential_energies.upsilon)
    upsilon_l : xr.DataArray
        Large-scale displacement potential Υˡ(ρ̄, z) = g(z - z_*(ρ̄))/ρ₀,
        computed from the filtered density sort (filt_local_potential_energies.upsilon)
    kappa : xr.DataArray
        Diffusivity field κ (e.g. ds.κ_e from SmagorinskyLilly)
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_density : xr.DataArray, optional
        Pre-computed filtered density ρ̄. If None, it is computed by applying
        filter to rho.
    index_dim : str, optional
        Name of the vector index dimension, default "i"

    Returns
    -------
    xr.DataArray
        SFS APE dissipation ε_s [J m⁻³ s⁻¹] with the same spatial dimensions as rho
    """
    # Term 1: filtered(κ ∇ρ · ∇Υ)
    grad_rho = calculate_gradient(rho)
    grad_upsilon = calculate_gradient(upsilon)
    kappa_grad_dot = kappa * (grad_rho * grad_upsilon).sum(dim=index_dim)
    term1 = filter.apply(kappa_grad_dot, dims=filter_dims)

    # Term 2: κ ∇ρ̄ · ∇Υˡ
    if filtered_density is None:
        filtered_density = filter.apply(rho, dims=filter_dims)
    grad_rho_bar = calculate_gradient(filtered_density)
    grad_upsilon_l = calculate_gradient(upsilon_l)
    term2 = kappa * (grad_rho_bar * grad_upsilon_l).sum(dim=index_dim)

    return term1 - term2
#---

#+++ Cross-scale APE flux
def calculate_cross_scale_ape_flux(rho, u_i, upsilon, filter, filter_dims=["x_caa", "y_aca"],
                                    filtered_density=None, filtered_velocity_vector=None,
                                    index_dim="i"):
    """
    Calculate the cross-scale APE flux Π = -τᵢ · ∇Υ

    The cross-scale APE flux quantifies the transfer of APE across the filter
    scale via the contraction of the subfilter tracer flux τᵢ with the gradient
    of the buoyancy displacement potential Υ = g(z - z₀)/ρ₀:

        Π = -(filtered(ρ uᵢ) - filtered(ρ) filtered(uᵢ)) · ∂Υ/∂xᵢ

    Parameters
    ----------
    rho : xr.DataArray
        Full (unfiltered) density field
    u_i : xr.DataArray
        Full (unfiltered) velocity vector with an "i" index dimension, as
        produced by condense_velocities()
    upsilon : xr.DataArray
        Buoyancy displacement potential Υ = g(z - z₀)/ρ₀, typically taken
        from the filtered potential energies dataset
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_density : xr.DataArray, optional
        Pre-computed filtered(ρ). Passed through to calculate_subfilter_tracer_flux.
    filtered_velocity_vector : xr.DataArray, optional
        Pre-computed filtered(uᵢ). Passed through to calculate_subfilter_tracer_flux.
    index_dim : str, optional
        Name of the vector index dimension, default "i"

    Returns
    -------
    xr.DataArray
        Cross-scale APE flux Π [J m⁻³ s⁻¹] with the same spatial dimensions
        as rho (the i dimension is summed over)
    """
    tau_i = calculate_subfilter_tracer_flux(
        rho, u_i, filter, filter_dims,
        filtered_density=filtered_density,
        filtered_velocity_vector=filtered_velocity_vector,
    )
    grad_upsilon = calculate_gradient(upsilon)
    return -(tau_i * grad_upsilon).sum(dim=index_dim)
#---
