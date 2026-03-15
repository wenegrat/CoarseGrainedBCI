#!/usr/bin/env python
#+++ Imports
import numpy as np
import xarray as xr
import gcm_filters
from aux00_utils import load_dataset_and_grid, condense_velocities, integrate
from aux02_ke_functions import (
    calculate_ke_decomposition,
    calculate_sfs_stress_tensor,
    calculate_large_scale_strain_tensor,
    calculate_sfs_ke_dissipation,
    calculate_cross_scale_ke_flux,
)
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

ds_filt = ds[["dV", "ūᵢ"]].copy()
ds_full = ds[["dV", "uᵢ"]].copy()

print(f"Velocities filtered with length scale: {filter_length_scale}")
#---

#+++ Calculate SFS stress tensor
print("\n" + "="*60)
print("Calculating SFS stress tensor...")

# τⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ   shape: (i, j, time, z, y, x)
# Pre-pass filtered_u_i so the filter is not applied a second time
τ = calculate_sfs_stress_tensor(ds_full["uᵢ"], gaussian_filter,
                                filter_dims=filtered_dimensions,
                                filtered_u_i=ds_filt["ūᵢ"])
τ.name = "τ"

# Sanity check: trace/2 must equal KE_s pointwise
tau_trace = τ.sel(i=1, j=1) + τ.sel(i=2, j=2) + τ.sel(i=3, j=3)
KE_s = tau_trace / 2
print("Done!")
#---

#+++ Calculate large-scale strain rate tensor
print("\n" + "="*60)
print("Calculating large-scale strain rate tensor...")

# S̄ℓⁱʲ = (1/2)(∂ūⁱ/∂xʲ + ∂ūʲ/∂xⁱ)   shape: (i, j, time, z, y, x)
S = calculate_large_scale_strain_tensor(ds["ūᵢ"])
S.name = "S"

# Sanity check: incompressibility requires tr(S̄) = ∇·ū ≈ 0
S_trace = S.sel(i=1, j=1) + S.sel(i=2, j=2) + S.sel(i=3, j=3)
max_div = float(abs(S_trace).max())
mean_diag = float((abs(S.sel(i=1, j=1)) + abs(S.sel(i=2, j=2)) + abs(S.sel(i=3, j=3))).mean() / 3)
print(f"  Incompressibility check: max|∇·ū| = {max_div:.2e}  (mean diagonal magnitude: {mean_diag:.2e})")

print("Done!")
#---

#+++ Calculate SFS KE dissipation
print("\n" + "="*60)
print("Calculating SFS KE dissipation...")

# The dissipation uses the FULL (unfiltered) strain rate so all scales contribute
S_full = calculate_large_scale_strain_tensor(ds["uᵢ"])

# ε<ℓ = 2ρ₀ν τ(S, S) = 2ρ₀ν Σᵢⱼ [ filter(Sⁱʲ Sⁱʲ) - filter(Sⁱʲ)² ]   [m² s⁻³]
eps_sfs = calculate_sfs_ke_dissipation(S_full, ds.ν, gaussian_filter,
                                        filter_dims=filtered_dimensions)
eps_sfs.name = "ε<ℓ"

print("Done!")
#---

#+++ Calculate cross-scale KE flux
print("\n" + "="*60)
print("Calculating cross-scale KE flux...")

# Πℓ = -ρ₀ S̄ℓ : τ̄ℓ = -ρ₀ Σᵢⱼ Sⁱʲ τⁱʲ   [m² s⁻³]
Pi_ke = calculate_cross_scale_ke_flux(S, τ)
Pi_ke.name = "Π_KE"

print("Done!")
#---

#+++ Sanity check: (1/2) tr(τ) == KE_s
xr.testing.assert_allclose(tau_trace_half, KE_s, rtol=1e-4)
print("Sanity check passed: (1/2)tr(τ) = KE_s ✓")
#---

#+++ Integrate
print("\n" + "="*60)
print("Integrating KE fields...")

dV = ds.Δx_caa * ds.Δy_aca * ds.Δz_aac

int_KE_s   = integrate(ke_decomp.KE_s,   dV)
int_Pi_ke  = integrate(Pi_ke,   dV)
int_eps_sfs = integrate(eps_sfs, dV)

print("Done!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

output_ds = xr.Dataset({
    # Local fields
    "KE":     ke_decomp.KE,
    "KE_l":   ke_decomp.KE_l,
    "KE_bar": ke_decomp.KE_bar,
    "KE_s":   ke_decomp.KE_s,
    "τ":      τ,
    "S":      S,
    "Π_KE":   Pi_ke,
    "ε<ℓ":    eps_sfs,
    # Integrated scalars
    "∫KE dV":     int_KE,
    "∫KE_l dV":   int_KE_l,
    "∫KE_bar dV": int_KE_bar,
    "∫KE_s dV":   int_KE_s,
    "∫Π_KE dV":   int_Pi_ke,
    "∫ε<ℓ dV":    int_eps_sfs,
})

output_filename = filename.replace(".nc", "_sfs_ke_budget.nc")
output_ds.to_netcdf(output_filename)
print(f"\nResults saved to: {output_filename}")
#---

#+++ Plot integrated KE decomposition
print("\n" + "="*60)
print("Creating plots...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Panel 1: KE decomposition
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

# Panel 3: budget terms (cross-scale flux and SFS dissipation)
ax = axes[2]
output_ds["∫Π_KE dV"].plot.line(ax=ax, x="time", label="∫Π_KE dV  (+ = forward cascade)")
(-output_ds["∫ε<ℓ dV"]).plot.line(ax=ax, x="time", label="−∫ε<ℓ dV  (dissipation sink)")
ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.set_ylabel("[m² s⁻³ × m³]")
ax.set_title("SFS KE budget terms")
ax.legend()
ax.grid(True, alpha=0.3)

plot_filename = output_filename.replace(".nc", ".png")
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"KE decomposition plot saved to: {plot_filename}")
#---
