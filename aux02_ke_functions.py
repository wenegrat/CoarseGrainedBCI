"""
Kinetic energy calculation functions

This module contains functions for calculating kinetic energy (KE).
"""

import numpy as np
import xarray as xr
from aux00_utils import integrate

# Physical constants
ρ0 = 1025  # reference density [kg/m^3]

#+++ KE from condensed velocity tensor
def local_KE_vector(u_i, index_dim="i"):
    """
    Calculate local KE density from the condensed velocity tensor

    Parameters
    ----------
    u_i : xr.DataArray
        Velocity vector with an index dimension (e.g. i=1,2,3 for u,v,w),
        as produced by condense_velocities().
    index_dim : str
        Name of the vector index dimension.

    Returns
    -------
    xr.DataArray
        Local KE density: (1/2)|u|²  [m² s⁻²]
    """
    return (u_i**2).sum(index_dim) / 2


def local_KE_l(u_i_bar, index_dim="i"):
    """
    Calculate large-scale KE density from the filtered velocity

    Following Aluie et al. (2018, JPO), the large-scale KE is the KE of the
    filtered velocity field, NOT the filtered KE:

        KE_l = (1/2)|ū|²

    Parameters
    ----------
    u_i_bar : xr.DataArray
        Filtered velocity vector (same shape as u_i).
    index_dim : str
        Name of the vector index dimension.

    Returns
    -------
    xr.DataArray
        Large-scale KE density: (1/2)|ū|²  [m² s⁻²]
    """
    return (u_i_bar**2).sum(index_dim) / 2


def local_KE_s(u_i, u_i_bar, filter, filter_dims=["x_caa", "y_aca"], index_dim="i"):
    """
    Calculate subfilter-scale (SFS) KE density

    The SFS KE equals half the trace of the sub-filter stress tensor τ(u,u)
    (Eyink & Aluie 2009, Eq. 6; Aluie et al. 2018):

        KE_s = (1/2)[ filter(|u|²) - |ū|² ]
             = filter(KE) - KE_l

    It is locally non-negative for non-negative filter kernels (Vreman et al. 1994).

    Parameters
    ----------
    u_i : xr.DataArray
        Full (unfiltered) velocity vector.
    u_i_bar : xr.DataArray
        Filtered velocity vector.
    filter : gcm_filters.Filter
        Filter object.
    filter_dims : list of str
        Spatial dimensions along which to apply the filter.
    index_dim : str
        Name of the vector index dimension.

    Returns
    -------
    xr.DataArray
        SFS KE density: filter(KE) - KE_l  [m² s⁻²]
    """
    KE_full = local_KE_vector(u_i, index_dim=index_dim)
    KE_bar  = filter.apply(KE_full, dims=filter_dims)
    KE_l    = local_KE_l(u_i_bar, index_dim=index_dim)
    return KE_bar - KE_l
#---

#+++ SFS stress tensor
def calculate_sfs_stress_tensor(u_i, filter, filter_dims=["x_caa", "y_aca"],
                                 filtered_u_i=None, index_dim="i"):
    """
    Calculate the full 3×3 SFS stress tensor τ̄ℓⁱʲ(u, u)

    Following Aluie et al. (2018, JPO) Eq. (5), each component is:

        τ̄ℓⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ

    The result is a symmetric tensor with a new dimension "j" (same index values
    as the input "i" dimension). The trace gives twice the SFS KE:

        (1/2) tr(τ̄ℓ) = (1/2)[ filter(|u|²) - |ū|² ] = KE_s  ≥ 0

    The filter is applied to the outer product uⁱ uʲ using xarray broadcasting,
    which evaluates all 9 components in a single pass — the same approach as
    calculate_sfs_flux_tensor() in aux01_pe_functions.py.

    Parameters
    ----------
    u_i : xr.DataArray
        Full (unfiltered) velocity tensor with index dimension (e.g. i=1,2,3).
    filter : gcm_filters.Filter
        Filter object used for the spatial filtering operation.
    filter_dims : list of str
        Spatial dimensions along which to apply the filter.
    filtered_u_i : xr.DataArray, optional
        Pre-computed filtered velocity ūⁱ. If None, it is computed from u_i.
    index_dim : str
        Name of the vector index dimension (default "i").

    Returns
    -------
    xr.DataArray
        Tensor τ̄ℓⁱʲ with dimensions (i, j, ...) where j carries the same
        coordinate values as i. Select a component via e.g.
        tau.sel(i=1, j=2) for τxy.
    """
    if filtered_u_i is None:
        filtered_u_i = filter.apply(u_i, dims=filter_dims)

    # Rename index dimension to "j" on the second operand so xarray broadcasts
    # u_i (i, ...) × u_j (j, ...) → outer product with shape (i, j, ...)
    u_j     = u_i.rename({index_dim: "j"})
    u_j_bar = filtered_u_i.rename({index_dim: "j"})

    # τ̄ℓⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ  (all components in one vectorised call)
    tau = filter.apply(u_i * u_j, dims=filter_dims) - filtered_u_i * u_j_bar

    return tau
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
        Local KE density: (u^2 + v^2 + w^2) / 2  [m² s⁻²]
    """
    return (u**2 + v**2 + w**2) / 2

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
