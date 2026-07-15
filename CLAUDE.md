# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BCI computes coarse-grained kinetic and available potential energy (APE) budgets for an idealized
baroclinic-instability channel, using the Winters et al. (1995) sorting method for APE and the
Aluie et al. (2018, JPO) coarse-graining framework for cross-scale KE/APE transfer. This is a fork of
[CoarseGrainedKHAPE](https://github.com/tomchor/CoarseGrainedKHAPE) (which targets a 2D x–z
Kelvin-Helmholtz instability), adapted to a 3D **double-front, doubly-periodic-horizontal** baroclinic
adjustment setup (following the [Oceananigans baroclinic_adjustment
example](https://clima.github.io/OceananigansDocumentation/stable/literated/baroclinic_adjustment)). The
pipeline is:
1. **Julia simulation** (Oceananigans.jl `NonhydrostaticModel`) -> NetCDF output
2. **Python post-processing** -> filter fields, sort density, compute energy transfer and SFS budgets, plot

GitHub remote: `git@github.com:wenegrat/CoarseGrainedBCI.git` (fork of `tomchor/CoarseGrainedKHAPE`,
tracked as the `upstream` remote)

## Running the Code

### Simulation
```bash
julia --project -t 8 baroclinic_adjustment.jl                     # default resolution (48x48x8), 20 days
julia --project -t 8 baroclinic_adjustment.jl --stop_time 1        # short run
julia --project -t 8 baroclinic_adjustment.jl --Nx 16 --Ny 16 --Nz 4 --stop_time 0.05 --progress_interval 1   # tiny smoke test
```
CLI args: `--Nx`, `--Ny`, `--Nz` (default 48, 48, 8), `--N2`, `--M2`, `--front_width`, `--perturbation_amplitude`,
`--latitude`, `--nu`, `--Pr`, `--stop_time` (days, default 20), `--filter_scales` (two horizontal FWHM scales in
km, default 50 100), `--progress_interval` (default 100; use a small value for short/smoke-test runs where the
default interval may never be reached).

**Note:** the `submit_*.sh`/`*.pbs` job scripts and `README.md`'s job-submission instructions still describe
the old KH pipeline (`kelvin_helmholtz_instability.jl`, `--Ri`/`--Re0`/`--U`/`--h`) and have not yet been
updated for `baroclinic_adjustment.jl` -- update them before submitting to the HPC.

### Post-processing
```bash
cd postprocessing
bash 00_get_budgets.sh output/bci_Nx48_Ny48_Nz8.nc --filter-scales 50000 100000
```
Set `N_WORKERS` env var to control parallelism (default 1): `N_WORKERS=4 bash 00_get_budgets.sh ...`

### Running tests
```bash
pytest tests/ -v -s
```
`test_budgets.py` checks SFS KE/APE budget closure (rms(residual)/min(rms(terms)) < 10%) against
`postprocessing/output/bci_Nx48_Ny48_Nz8_*` (run `00_get_budgets.sh` first). See Notes below for the current
state of this test -- it does not yet pass, and that looks like a resolution/duration limitation rather than
a code bug.

### Python environment
The repo's `environment.yml` is a Linux-only conda lockfile (built on an HPC). For local development
(e.g. macOS), create a plain venv instead:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r tests/requirements.txt
```

### Julia environment
Requires Julia 1.11.x (matching the HPC target). Use `juliaup` to manage multiple Julia versions
side by side; `juliaup override set 1.11.2` pins this directory to the right version so a bare `julia`
invocation picks it up automatically.

## Architecture

### Physical setup (`baroclinic_adjustment.jl`)
- Model: `NonhydrostaticModel` (no free surface at all -- see Notes for why this replaced the earlier
  `HydrostaticFreeSurfaceModel`+`ImplicitFreeSurface` setup), `BetaPlane` Coriolis, `BuoyancyTracer`,
  `Centered(order=4)` advection, `ScalarDiffusivity(ν, κ)` closure. w is a genuine prognostic variable
  here, with its own momentum equation and dissipative dynamics.
- Grid: `(Periodic, Periodic, Bounded)` -- a **double front** rather than a single front against channel
  walls: two opposite-signed buoyancy ramps (`double_ramp`) so the field closes periodically in y. This
  avoids side-wall boundary layers (an extra KE sink the budget would otherwise need to account for) and
  means the horizontal Gaussian filter is a pure periodic wrap in both directions, with no edge-extension
  boundary handling needed.
- Velocities are interpolated to `(Center, Center, Center)` (`u_center`/`v_center`/`w_center`) before being
  written to output or filtered. Writing the raw staggered (Face) velocities instead breaks the offline
  Python tensor math silently (products of fields at mismatched staggered locations broadcast across extra
  coordinate dimensions instead of multiplying pointwise) -- this bit us once during testing; see Notes.
- The coarse-graining filter is **horizontal-only** (x, y) -- it never touches z. Horizontal scales span the
  mesoscale/submesoscale range this budget targets; z has its own distinct structure (stratification,
  boundary layers) that shouldn't be smoothed over.
- Grid halo is sized explicitly (`halo=(Hx,Hy,3)`) from the largest requested filter scale's stencil radius
  (4σ truncation). The default halo Oceananigans picks is sized for the advection scheme, not for a wide
  Gaussian filter stencil -- an undersized halo causes silent memory corruption (a segfault at an unrelated
  *later* point, not a clean bounds-check error), not an immediate crash at the filter call site.
- **Πₖ (cross-scale KE flux) and ε_Kˢ (SFS KE dissipation) are computed online**, via Oceanostics'
  `KineticEnergyCrossScaleFlux`/`SubFilterKineticEnergyDissipationRate`, one field per filter scale
  (`Π_K_ℓ50km`, `ε_Kˢ_ℓ50km`, etc.). This used to be offline-only (an Oceanostics bug crashed the online
  multi-direction `GaussianFilter` for a periodic y-direction -- see Notes) but the fix landed in
  Oceanostics v0.17.3, and `SubFilterKineticEnergyDissipationRate` (the missing SFS-dissipation
  diagnostic) was added in the still-unmerged `tc/sfs-ke` branch (pinned in `Project.toml`/`Manifest.toml`
  via `Pkg.add(url=..., rev="tc/sfs-ke")` -- check whether tomchor/Oceanostics.jl#266 has merged and
  released before assuming this pin is still needed). Both were validated against the previous offline
  Python implementation before switching over (0.99 spatial correlation, rms agreement within ~1-10%).
- Πₖ is the **full 3D contraction** (`KineticEnergyCrossScaleFlux(model, filter; dims=(1,2,3))`): w is a
  genuine prognostic variable in this `NonhydrostaticModel`, with its own momentum equation and dissipative
  dynamics, so there's no reason to exclude it (unlike the earlier `HydrostaticFreeSurfaceModel` setup,
  where w was diagnostic and excluding it was necessary for the KE budget to close even in principle).
  ε_Kˢ's public API has no `dims` restriction at all -- it always includes w's full contribution via the
  model's actual per-direction viscous fluxes, which is simply correct now rather than the small "phantom"
  w-diffusion term it represented under the old hydrostatic setup (verified negligible there, ~1e-8
  relative magnitude, via a validation smoke test). The offline SFS KE budget pipeline
  (`04_sfs_ke_budget.py`, `calculate_energy_transfer()`) was updated to match -- the SFS KE density and
  offline Π_K (validation-only; Π_K is read from the online output in practice) are now full 3D too.
- `utils.jl` -- `closest_factor_number()` (FFT-friendly grid sizes), `show_gpu_status()` (unchanged from KH).

### Post-processing pipeline (`postprocessing/`)
Same 01-06 structure as KHAPE, adapted for horizontal (x,y) filtering instead of KH's (x,z):

| Script | Change from KHAPE |
|--------|--------|
| `01_filter_fields.py` | filters in (x,y) instead of (x,z); filter scales are free parameters again (no longer need to match an online `filter_ℓs`) |
| `02_sort_density.py` | unchanged -- the Winters sort is dimension-agnostic |
| `03_energy_transfer.py` | Πₖ is no longer computed here (`include_pi_k=False`) -- it's read straight from the simulation NetCDF now; still computes Π_A and the APE↔KE exchange offline |
| `04_sfs_ke_budget.py` | reads Πₖ and ε_Kˢ directly from the simulation output (`ds[f"Π_K_ℓ{ℓ_km}km"]`, `ds[f"ε_Kˢ_ℓ{ℓ_km}km"]`) instead of computing/loading them; still computes the SFS KE density (LHS) offline via the stress-tensor trace, full 3D (i,j ∈ {1,2,3}) to stay dimensionally consistent with the online Πₖ/ε_Kˢ now that w is prognostic |
| `05_sfs_ape_budget.py` | filters in (x,y); diffusivity κ now read from `nu`/`Pr` global attributes (a constant `ScalarDiffusivity`), not a `ds.κ` spatial field (which only exists for non-constant closures and was never actually populated here) |

`aux00_utils.py`'s `GaussianFilter` class filters (x,y), both periodic (`mode='wrap'` on both), replacing
KH's (x periodic, z bounded `mode='nearest'`). `condense_velocities` (u,v,w) is used throughout instead of
KH's `condense_uw_velocities` (u,w only, valid for the 2D x-z KH case); w is now included fully in the
KE cross-scale tensors too (see the Πₖ note above), not excluded.

`sweep*` scripts, `validation/`, and the standalone `plot*`/`anim1_panels.py` scripts still describe the KH
pipeline's online-vs-offline validation setup and have not been adapted -- there is no online Πₖ/ε_Kˢ to
validate against anymore. `anim2_surface_buoyancy.py` is new: a simple standalone script that animates the
surface buoyancy field to a GIF (no ffmpeg dependency, uses matplotlib's `PillowWriter`).

### Key dependencies
- **Python**: `numpy`, `xarray`, `scipy`, `matplotlib`, `dask`, `gcm_filters`, `netcdf4`
- **Julia**: `Oceananigans` v0.110.8, `Oceanostics` v0.18.0 (pinned to the `tc/sfs-ke` branch, not yet a
  tagged release -- see the Notes entry on the online Πₖ/ε_Kˢ switch), `NCDatasets`, `CairoMakie` (Julia
  1.11.2)

## Physics Reference

- **TPE** = integral of g*rho*z dV (total potential energy)
- **RPE** = minimum PE achievable by adiabatic rearrangement (from sorted reference state)
- **APE** = TPE - RPE (available for conversion to KE)
- **Πₖ**, **Π_A** -- cross-scale energy transfer (sub-filter to resolved). Both are full 3D/unrestricted:
  Π_A always was (density/APE has no analogous "diagnostic component" issue), and Πₖ is too now that w is
  prognostic (see Architecture) -- it was horizontal-only under the earlier hydrostatic setup.
- Physical constants: `g=9.81`, `rho_0=1025`

## Code Style

- Do not break a command/statement into multiple lines if it fits within 140 columns.
- Always delimit code sections with `#+++` on the opening line and `#---` on the closing line:
  ```python
  #+++ Section name
  ...code...
  #---
  ```

## Notes

- **Oceanostics bug (fixed)**: `GaussianFilter(; dims=(1,2), σ)` used to crash (heap corruption -> SIGILL)
  on a grid with real `Ny>1` and periodic y -- filed as
  [tomchor/Oceanostics.jl#262](https://github.com/tomchor/Oceanostics.jl/issues/262), with a minimal
  reproducer; fixed in v0.17.3 ([PR #263](https://github.com/tomchor/Oceanostics.jl/pull/263), root cause
  was a multi-direction filter's staged kernel launch being sized from the operand instead of the
  destination field, which broke specifically for *windowed* destinations like this repo's
  `indices=(:, :, grid.Nz)` surface output writer). The `SequentialGaussianFilter` workaround this repo used
  to carry (two sequential 1D passes instead of one `dims=(1,2)` call) has been removed now that the native
  filter works directly; Πₖ/ε_Kˢ are computed online (see Architecture) instead of deferred offline.
- **NonhydrostaticModel replaced HydrostaticFreeSurfaceModel+ImplicitFreeSurface.** Motivated by comparing
  against tomchor's own Eady baroclinic-instability example (Oceanostics PR #260,
  `docs/examples/eady_baroclinic_instability.jl`), which uses `NonhydrostaticModel` and closes its
  coarse-grained filtered-KE budget to ~11-15% residual/dominant -- much better than this repo's ~40-60%
  at the time. Switching just the model type (keeping our own closure/advection/resolution otherwise fixed)
  did *not* reproduce that improvement on its own (~40-45% either way), ruling out the free surface as the
  sole cause. A live, ongoing investigation into tomchor's example (run standalone, outside this repo) found
  that swapping his buoyancy-production-term convention (`w̄b̄`, using the raw filtered perturbation
  buoyancy) for this repo's own convention (`w̄b̄ᵣ`, using a Winters-sorted reference-state buoyancy)
  substantially degrades *his* closure too when done carelessly (dramatically, if the sort mistakenly
  includes the front's own horizontally-varying background buoyancy, which double-counts energy already
  captured by his separate mean-shear production term `Pu` -- only the horizontally-uniform, z-only part of
  a background field can be added to a buoyancy production term "for free", by an exact incompressibility
  argument: horizontal-mean w is exactly zero at every z in a periodic, impermeable-boundary domain). The
  corrected version of that test (background restricted to the z-only stratification) was in progress when
  the model switch was made permanent here; check conversation history for its outcome before assuming the
  buoyancy-convention question is resolved one way or the other. The NonhydrostaticModel switch itself is
  being kept regardless, since it removes the free surface's own complications (no η, no barotropic
  pressure-correction term, no dimension-inference limitation when trying to output η) and makes w a
  properly prognostic variable, matching tomchor's own validated approach -- but it should be treated as an
  ongoing architecture change, not a settled fix for the closure gap.
- **Budget closure is not yet validated at production resolution.** Historically (under the earlier
  hydrostatic setup) a 1-day toy run (16x16x4) and a 20-day run at default resolution (48x48x8) both failed
  the KHAPE-style closure test (`rms(residual)/min(rms(terms)) < 10%); comparing the residual against the
  *dominant* budget term instead gave ~25-87% depending on filter scale and resolution, and did not converge
  cleanly with resolution (see conversation history for a full resolution sweep and the many candidate
  causes tested -- output-sampling interval, gradient stencil order, SFS-vs-filtered budget choice, online
  vs. offline Πₖ/ε_Kˢ, and now model type -- none of which fully explained the gap on their own). tomchor's
  own Eady example achieving ~11-15% suggests the coarse-graining framework itself has a real, nonzero
  closure floor even under near-ideal conditions, but that our own ~40-60% is still meaningfully worse than
  that floor. Revisit once the buoyancy-convention investigation above concludes.
- `online_ke_transfer_validation.md` is a KH-era dev note about computing Πₖ online and validating it against
  the offline pipeline -- it predates both this fork's move to fully-offline Πₖ/ε_Kˢ and the subsequent move
  back to online (see above), so it still doesn't describe current behavior, though the general idea
  (validate online against offline before trusting it) is exactly what was done again for this switch.
- Output files (`.nc`, `.mp4`, `.pdf`, `.png`, `.gif`, `.jld2`) are excluded from git.
