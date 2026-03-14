"""
Kinetic energy calculation functions

This module contains functions for calculating kinetic energy (KE).
"""

import numpy as np
import xarray as xr
from aux00_utils import integrate, calculate_gradient

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

#+++ Velocity gradient tensor
def calculate_velocity_gradient_tensor(u_i_bar, dimensions=("x_caa", "y_aca", "z_aac"),
                                        index_dim="i"):
    """
    Compute the large-scale velocity gradient tensor ∂ūⁱ/∂xʲ

    Each row i is the gradient of the i-th filtered velocity component, giving
    a tensor of shape (i, j, ...) where index j runs over spatial directions.

    Parameters
    ----------
    u_i_bar : xr.DataArray
        Filtered velocity vector with index dimension (e.g. i=1,2,3 for ū, v̄, w̄).
    dimensions : tuple of str
        Ordered spatial coordinate names matching index values 1, 2, 3.
    index_dim : str
        Name of the velocity index dimension (default "i").

    Returns
    -------
    xr.DataArray
        Tensor ∂ūⁱ/∂xʲ with dimensions (i, j, ...).
        Select a component via e.g. grad_u.sel(i=1, j=2) for ∂ū/∂y.
    """
    j_indices = list(u_i_bar[index_dim].values)
    grad_components = []
    for k in j_indices:
        u_k = u_i_bar.sel({index_dim: k}, drop=True)
        # calculate_gradient returns a vector along dimname="j"
        grad_k = calculate_gradient(u_k, dimensions=dimensions, dimname="j", indices=j_indices)
        grad_components.append(grad_k)

    return xr.concat(
        grad_components,
        dim=xr.DataArray(j_indices, dims=index_dim, name=index_dim),
    )
#---

#+++ Large-scale strain rate tensor
def calculate_large_scale_strain_tensor(u_i_bar, dimensions=("x_caa", "y_aca", "z_aac"),
                                         index_dim="i"):
    """
    Compute the large-scale strain rate tensor S̄ℓⁱʲ

    S̄ℓ is the symmetric part of the velocity gradient tensor:

        S̄ℓⁱʲ = (1/2)(∂ūⁱ/∂xʲ + ∂ūʲ/∂xⁱ)

    For incompressible flow tr(S̄ℓ) = ∇·ū = 0.

    Parameters
    ----------
    u_i_bar : xr.DataArray
        Filtered velocity vector with index dimension (e.g. i=1,2,3 for ū, v̄, w̄).
    dimensions : tuple of str
        Ordered spatial coordinate names matching index values 1, 2, 3.
    index_dim : str
        Name of the velocity index dimension (default "i").

    Returns
    -------
    xr.DataArray
        Symmetric tensor S̄ℓⁱʲ with dimensions (i, j, ...).
        Select a component via e.g. S.sel(i=1, j=2) for S̄ˣʸ.
    """
    grad_u = calculate_velocity_gradient_tensor(u_i_bar, dimensions=dimensions,
                                                 index_dim=index_dim)

    # Transpose: swap i ↔ j so that grad_u_T[i,j] = ∂ūʲ/∂xⁱ
    grad_u_T = grad_u.rename({index_dim: "j", "j": index_dim})

    return (grad_u + grad_u_T) / 2
#---

#+++ SFS KE dissipation
def calculate_sfs_ke_dissipation(S, nu, filter, filter_dims=["x_caa", "y_aca"],
                                  index_dims=("i", "j")):
    """
    Compute the SFS KE dissipation ε<ℓ = 2ρ₀ν τ(S, S)

    The τ operator applied to the strain rate tensor yields a scalar via
    double contraction:

        τ(S, S) = Σᵢⱼ [ filter(Sⁱʲ Sⁱʲ) - filter(Sⁱʲ)² ]

    which is the subfilter variance of the strain field.  Multiplied by
    2ρ₀ν this gives the rate at which viscosity dissipates subfilter KE.

    The structure mirrors calculate_sfs_ape_dissipation() in
    aux01_pe_functions.py: filter(S²) and filter(S)² are each evaluated
    across all (i,j) components in one call thanks to xarray broadcasting,
    then the index dimensions are summed away.

    Parameters
    ----------
    S : xr.DataArray
        Strain rate tensor Sⁱʲ with dimensions (i, j, ...).
        Should be computed from the *full* (unfiltered) velocity to capture
        all scales; typically from calculate_large_scale_strain_tensor()
        applied to the unfiltered velocity field.
    nu : xr.DataArray or float
        Kinematic viscosity ν [m² s⁻¹] (scalar or spatially varying field,
        e.g. the eddy viscosity ds.ν from SmagorinskyLilly).
    filter : gcm_filters.Filter
        Filter object used for the spatial filtering operation.
    filter_dims : list of str
        Spatial dimensions along which to apply the filter.
    index_dims : tuple of str
        Names of the two tensor index dimensions to contract over.

    Returns
    -------
    xr.DataArray
        SFS KE dissipation ε<ℓ [m² s⁻³], same spatial dimensions as S
        (the i and j index dimensions are contracted away).
    """
    S_bar   = filter.apply(S, dims=filter_dims)             # filter(Sⁱʲ)
    tau_S_S = filter.apply(S * S, dims=filter_dims) - S_bar * S_bar  # τ(Sⁱʲ, Sⁱʲ)
    return 2 * ρ0 * nu * tau_S_S.sum(list(index_dims))
#---

#+++ Cross-scale KE flux
def calculate_cross_scale_ke_flux(S, tau, index_dims=("i", "j")):
    """
    Compute the cross-scale KE flux Πℓ = -ρ₀ S̄ℓ : τ̄ℓ

    Following Aluie et al. (2018, JPO) Eq. (7).  Πℓ > 0 means forward
    (downscale) KE transfer; Πℓ < 0 means inverse (upscale) transfer.

    The double contraction of two symmetric 3×3 tensors is

        A : B = Σᵢⱼ Aⁱʲ Bⁱʲ
              = Aˣˣ Bˣˣ + Aʸʸ Bʸʸ + Aᶻᶻ Bᶻᶻ
                + 2 Aˣʸ Bˣʸ + 2 Aˣᶻ Bˣᶻ + 2 Aʸᶻ Bʸᶻ

    Because both tensors are stored as full (i, j, ...) DataArrays with all
    nine entries, the sum over both index dimensions gives the correct result
    (the off-diagonal factor-of-2 arises automatically from the (i,j) + (j,i)
    entries):

        Πℓ = -ρ₀ * (S * τ).sum(["i", "j"])

    Parameters
    ----------
    S : xr.DataArray
        Large-scale strain rate tensor S̄ℓⁱʲ with dimensions (i, j, ...).
        Typically from calculate_large_scale_strain_tensor().
    tau : xr.DataArray
        SFS stress tensor τ̄ℓⁱʲ with dimensions (i, j, ...).
        Typically from calculate_sfs_stress_tensor().
    index_dims : tuple of str
        Names of the two tensor index dimensions to sum over.

    Returns
    -------
    xr.DataArray
        Cross-scale KE flux Πℓ [m² s⁻³], same spatial dimensions as S / tau
        (the i and j dimensions are contracted away).
    """
    return -ρ0 * (S * tau).sum(list(index_dims))
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
