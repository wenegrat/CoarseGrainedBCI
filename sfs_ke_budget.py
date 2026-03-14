#!/usr/bin/env python
"""
Calculate SFS KE budget from Kelvin-Helmholtz simulation output

Following the coarse-graining framework of Aluie et al. (2018, JPO) and
Eyink & Aluie (2009), the filtered total KE decomposes as:

    filter(KE) = KE_l + KE_s

where:
    KE     = (1/2)|u|²                 — local total KE
    KE_l   = (1/2)|ū|²                 — large-scale KE (KE of filtered velocity)
    KE_s   = filter(KE) - KE_l         — subfilter-scale (SFS) KE
           = (1/2)[ filter(|u|²) - |ū|² ]
           = (1/2) tr[τ(u,u)]          — half-trace of the SFS stress tensor

KE_s ≥ 0 pointwise for non-negative filter kernels (Vreman et al. 1994).
"""

#+++ Imports
import numpy as np
import xarray as xr
import gcm_filters
from aux00_utils import load_dataset_and_grid, condense_velocities, integrate
from aux02_ke_functions import local_KE_vector, local_KE_l, local_KE_s
#---

#+++ Configuration
# filename = "output/kelvin_helmholtz_instability_128x1x512.nc"
filename = "output/kelvin_helmholtz_instability_64x1x256.nc"
filter_length_scale = 0.8  # Length scale for filtering
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Filter velocity field
print("\n" + "="*60)
print("Filtering velocity field...")

filtered_dimensions = ["x_caa", "y_aca"]
filter_scale = filter_length_scale * np.sqrt(12)
gaussian_filter = gcm_filters.Filter(
    filter_scale=filter_scale,
    dx_min=float(min(ds.Δx_caa.min(), ds.Δy_aca.min())),
    filter_shape=gcm_filters.FilterShape.GAUSSIAN,
    grid_type=gcm_filters.GridType.REGULAR,
)

ds = condense_velocities(ds, indices=[1, 2, 3])  # uᵢ with i dimension
ds["ūᵢ"] = gaussian_filter.apply(ds["uᵢ"], dims=filtered_dimensions)

print(f"Velocities filtered with length scale: {filter_length_scale}")
#---

#+++ Calculate KE fields
print("\n" + "="*60)
print("Calculating KE fields...")

# Total KE at each point: KE = (1/2)|u|²
KE = local_KE_vector(ds["uᵢ"])
KE.name = "KE"

# Large-scale KE: KE_l = (1/2)|ū|²
KE_l = local_KE_l(ds["ūᵢ"])
KE_l.name = "KE_l"

# Filtered total KE: K̄E = filter((1/2)|u|²)
KE_bar = gaussian_filter.apply(KE, dims=filtered_dimensions)
KE_bar.name = "KE_bar"

# SFS KE: KE_s = K̄E - KE_l = filter(KE) - (1/2)|ū|²
KE_s = local_KE_s(ds["uᵢ"], ds["ūᵢ"], gaussian_filter, filter_dims=filtered_dimensions)
KE_s.name = "KE_s"

print("KE fields calculated: KE, KE_l, KE_bar, KE_s")
#---

#+++ Integrate
print("\n" + "="*60)
print("Integrating KE fields...")

dV = ds.Δx_caa * ds.Δy_aca * ds.Δz_aac

int_KE     = integrate(KE,     dV)
int_KE_l   = integrate(KE_l,   dV)
int_KE_bar = integrate(KE_bar, dV)
int_KE_s   = integrate(KE_s,   dV)

print("Done!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

output_ds = xr.Dataset({
    # Local fields
    "KE":     KE,
    "KE_l":   KE_l,
    "KE_bar": KE_bar,
    "KE_s":   KE_s,
    # Integrated scalars
    "∫KE dV":     int_KE,
    "∫KE_l dV":   int_KE_l,
    "∫KE_bar dV": int_KE_bar,
    "∫KE_s dV":   int_KE_s,
})

output_filename = filename.replace(".nc", "_sfs_ke_budget.nc")
output_ds.to_netcdf(output_filename)
print(f"\nResults saved to: {output_filename}")
#---

#+++ Plot integrated KE decomposition
print("\n" + "="*60)
print("Creating plots...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Panel 1: absolute values
ax = axes[0]
for var, label in [("∫KE dV", "KE (total)"), ("∫KE_bar dV", "K̄E (filtered)"),
                   ("∫KE_l dV", "KE_l (large-scale)"), ("∫KE_s dV", "KE_s (SFS)")]:
    output_ds[var].plot.line(ax=ax, x="time", label=label)
ax.set_ylabel("KE [m² s⁻² × m³]")
ax.set_title("Volume-integrated KE decomposition")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: SFS fraction KE_s / K̄E
ax = axes[1]
sfs_fraction = output_ds["∫KE_s dV"] / output_ds["∫KE_bar dV"]
sfs_fraction.plot.line(ax=ax, x="time")
ax.set_ylabel("KE_s / K̄E")
ax.set_title("SFS fraction of filtered KE")
ax.grid(True, alpha=0.3)

plot_filename = output_filename.replace(".nc", ".png")
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"KE decomposition plot saved to: {plot_filename}")
#---
