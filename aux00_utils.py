import time
from functools import wraps
import numpy as np
import xarray as xr

#+++ Timing decorator
def timeit(func):
    """Decorator that prints the elapsed time of a function call"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n{func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Elapsed wall time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        return result
    return wrapper
#---

#+++ Integrations and sums
def volume_sum(da, dims=("x_caa", "y_aca", "z_aac")):
    """Sum a DataArray over spatial dimensions"""
    return da.sum(dims)

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
    ds = ds.drop(vlist)
    return ds

def condense_velocities(ds, dimname="i", indices=(1, 2, 3)):
    """Condense velocity components into tensor form"""
    return condense(ds, ["u", "v", "w"], "uᵢ", dimname=dimname, indices=indices)
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