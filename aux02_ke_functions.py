"""
Kinetic energy calculation functions

This module contains functions for calculating kinetic energy (KE).
"""

import numpy as np
import xarray as xr
from aux00_utils import integrate

# Physical constants
ρ0 = 1025  # reference density [kg/m^3]

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
