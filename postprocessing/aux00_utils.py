import os
from pathlib import Path
import numpy as np
import xarray as xr

PP_OUTPUT = Path(__file__).resolve().parent / "output"

#+++ Integrations and sums
def integrate(da, dV, dims=("x_caa", "y_aca", "z_aac")):
    """Integrate a DataArray over spatial dimensions"""
    return (da * dV).sum(dims)
#---

#+++ Load data
def load_dataset_and_grid(filename):
    """
    Load the simulation output and grid information

    Parameters
    ----------
    filename : str
        Path to the NetCDF file

    Returns
    -------
    ds : xr.Dataset
        Dataset with grid information added as attributes and variables
    """
    print(f"Loading data from {filename}...")
    ds = xr.open_dataset(filename, decode_times=False)
    grid = xr.open_dataset(filename, group="underlying_grid_reconstruction_kwargs")

    # Add grid extent as attributes
    ds.attrs["Lx"] = np.diff(grid.x)
    ds.attrs["Ly"] = np.diff(grid.y)
    ds.attrs["Lz"] = np.diff(grid.z)

    ds.attrs["x_min"] = grid.x.min()
    ds.attrs["x_max"] = grid.x.max()
    ds.attrs["y_min"] = grid.y.min()
    ds.attrs["y_max"] = grid.y.max()
    ds.attrs["z_min"] = grid.z.min()
    ds.attrs["z_max"] = grid.z.max()

    # Add volume and area variables
    ds["dV"] = ds.Δx_caa * ds.Δy_aca * ds.Δz_aac
    ds["LxLy"] = ds.Lx * ds.Ly

    return ds
#---

#+++ Condensing operations for Datasets
def condense(ds, vlist, varname, dimname="i", indices=(1, 2, 3)):
    """
    Condense variables in `vlist` into one variable named `varname`.
    In the process, individual variables in `vlist` are removed from `ds`.
    """
    ds[varname] = ds[vlist].to_array(dim=dimname).assign_coords({dimname : list(indices)})
    ds = ds.drop_vars(vlist)
    return ds

def condense_velocities(ds, dimname="i", indices=(1, 2, 3)):
    """Condense velocity components into tensor form"""
    return condense(ds, ["u", "v", "w"], "uᵢ", dimname=dimname, indices=indices)

def condense_uw_velocities(ds, dimname="i", indices=(1, 3)):
    """Condense u and w velocity components into tensor form (for 2D simulations)"""
    return condense(ds, ["u", "w"], "uᵢ", dimname=dimname, indices=indices)
#---

#+++ Spatial derivatives
def calculate_gradient(scalar, output_name="grad_scalar", dimensions=("x_caa", "y_aca", "z_aac"), dimname="i", indices=(1, 2, 3)):
    """
    Calculate the gradient of a scalar field

    Each component ∂scalar/∂xᵢ is computed via xr.DataArray.differentiate along
    the corresponding spatial coordinate, then all components are condensed into
    a single DataArray with an extra index dimension using condense().

    Parameters
    ----------
    scalar : xr.DataArray
        Scalar field to differentiate
    output_name : str, optional
        Name for the output DataArray. Defaults to "grad_scalar"
    dimname : str, optional
        Name of the new index dimension, default "i"
    indices : list, optional
        Index values along the new dimension. Defaults to [1, 2, ..., N].

    Returns
    -------
    xr.DataArray
        Gradient components stacked along a new `dimname` dimension,
        with the same spatial dimensions as scalar
    """
    vlist = []
    for dim in dimensions:
        if dim in scalar.dims and scalar.sizes[dim] > 1:
            vlist.append(scalar.differentiate(dim))
        else:
            vlist.append(xr.zeros_like(scalar))

    aux_ds = xr.Dataset()
    for i, da in enumerate(vlist):
        aux_ds[str(i+1)] = da
    aux_ds = condense(aux_ds, list(aux_ds.data_vars.keys()), output_name, dimname=dimname, indices=indices)
    return aux_ds[output_name]
#---

#+++ Gaussian filter (x: periodic, z: bounded)
class GaussianFilter:
    """Gaussian filter in x (periodic) and z (bounded) directions.

    Two sequential 1D scipy Gaussian convolutions:
      - x: mode='wrap'    — periodic BC
      - z: mode='reflect' — no-flux (Neumann) BC at domain boundaries

    sigma = ℓ / dx_min in grid units, so the physical-space sigma equals ℓ.
    """
    def __init__(self, ℓ, dx_min, dz_min):
        self._sigma_x = ℓ / dx_min
        self._sigma_z = ℓ / dz_min

    def apply(self, da, dims):
        """Apply filter in dims[0] (x, periodic) then dims[1] (z, bounded).

        Parameters
        ----------
        da : xr.DataArray
        dims : list of str
            [x_dim, z_dim], e.g. ['x_caa', 'z_aac']
        """
        from scipy.ndimage import gaussian_filter1d
        x_dim, z_dim = dims
        da_x = xr.apply_ufunc(
            gaussian_filter1d, da,
            input_core_dims=[[x_dim]],
            output_core_dims=[[x_dim]],
            kwargs={"sigma": self._sigma_x, "axis": -1, "mode": "wrap"},
            dask="parallelized",
            output_dtypes=[da.dtype],
        )
        return xr.apply_ufunc(
            gaussian_filter1d, da_x,
            input_core_dims=[[z_dim]],
            output_core_dims=[[z_dim]],
            kwargs={"sigma": self._sigma_z, "axis": -1, "mode": "reflect"},
            dask="parallelized",
            output_dtypes=[da_x.dtype],
        )


def make_gaussian_filter(ℓ, ds):
    """Return a GaussianFilter for length scale ℓ using grid spacing from ds.

    Parameters
    ----------
    ℓ : float
        Filter length scale in physical units.
    ds : xr.Dataset
        Simulation dataset (must contain Δx_caa and Δz_aac).
    """
    dx_min = float(ds.Δx_caa.min())
    dz_min = float(ds.Δz_aac.min())
    return GaussianFilter(ℓ, dx_min, dz_min)


def filter_fields(ds, filter_length_scales):
    """Filter velocity and buoyancy fields at each length scale in x and z.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with velocity components (u, w) and buoyancy b.
    filter_length_scales : array-like
        Physical length scales at which to apply the filter.

    Returns
    -------
    ds_filt : xr.Dataset
        Dataset with filtered fields ūᵢ and b̄ at each filter_length_scale,
        plus dV (scale-independent).
    """
    ds = condense_uw_velocities(ds, indices=(1, 3))

    ds_filt_list = []
    for ℓ in filter_length_scales:
        print(f"  filter_length_scale = {ℓ:.4f}...")
        gf = make_gaussian_filter(ℓ, ds)
        ds_filt_list.append(xr.Dataset({
            "ūᵢ": gf.apply(ds["uᵢ"], dims=["x_caa", "z_aac"]),
            "b̄":  gf.apply(ds["b"],  dims=["x_caa", "z_aac"]),
        }))

    scale_coord = xr.DataArray(filter_length_scales, dims="filter_length_scale",
                               name="filter_length_scale")
    ds_filt = xr.concat(ds_filt_list, dim=scale_coord)
    ds_filt["dV"] = ds["dV"]
    ds_filt.attrs.update(ds.attrs)
    ds_filt.attrs["filter_ndim"] = 2
    return ds_filt
#---

#+++ Dask-parallel filter wrapper
class DaskParallelFilter:
    """
    Thin proxy around a GaussianFilter that automatically chunks the input
    along the time dimension and computes with a thread pool, giving ~N×
    speedup where N is the number of available cores.

    All attributes other than `apply` are forwarded to the wrapped filter.
    """
    def __init__(self, filter_obj, chunk_size=1, n_workers=None):
        self._filter  = filter_obj
        self._chunk   = chunk_size
        self._workers = n_workers or os.cpu_count()
        print(f"  Using {self._workers} CPU workers")

    def apply(self, da, dims):
        if "time" in da.dims and da.sizes.get("time", 1) > 1:
            lazy = self._filter.apply(da.chunk({"time": self._chunk}), dims=dims)
            return lazy.compute(scheduler="threads", num_workers=self._workers)
        return self._filter.apply(da, dims=dims)

    def __getattr__(self, name):
        return getattr(self._filter, name)
#---

#+++ Pre-computed result loaders
def load_energy_transfer(filename):
    """Load the *_energy_transfer.nc file produced by 02_energy_transfer.py."""
    et_filename = str(PP_OUTPUT / (Path(filename).stem + "_energy_transfer.nc"))
    return xr.open_dataset(et_filename, decode_timedelta=False).chunk({"time": 1})
#---