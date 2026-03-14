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
