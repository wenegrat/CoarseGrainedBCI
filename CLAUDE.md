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
1. **Julia simulation** (Oceananigans.jl `HydrostaticFreeSurfaceModel`) -> NetCDF output
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
- Model: `HydrostaticFreeSurfaceModel` + `ImplicitFreeSurface`, `BetaPlane` Coriolis, `BuoyancyTracer`,
  `Centered(order=4)` advection, `ScalarDiffusivity(ν, κ)` closure.
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
- **Unlike the KH setup, Πₖ (cross-scale KE flux) and ε_Kˢ (SFS KE dissipation) are NOT computed online.**
  An Oceanostics bug (see Notes) crashes the online multi-direction `GaussianFilter` for a periodic
  y-direction, so both are computed offline in Python instead (`03_energy_transfer.py`,
  `04_sfs_ke_budget.py`).
- The KE-side coarse-graining (Πₖ, ε_Kˢ, and the SFS KE density itself) is restricted to **horizontal
  velocity components only** (i,j ∈ {1,2}): w is diagnostic (not prognostic) in a
  `HydrostaticFreeSurfaceModel` and has no independent dissipative dynamics, so including it would leave the
  KE budget unable to close even in principle. w is still used, unrestricted, for the APE↔KE exchange term
  (a genuine physical buoyancy-flux conversion that needs the real vertical velocity).
- `SequentialGaussianFilter` (defined locally in `baroclinic_adjustment.jl`): does two sequential 1D Gaussian
  filter passes (x then y) instead of asking Oceanostics for one combined `dims=(1,2)` filter -- works
  around the Oceanostics bug for the plain filtered-field *outputs*. Not used for the composite
  `KineticEnergyCrossScaleFlux`/`CoarseGrainedKineticEnergyDissipationRate` functions -- threading it through
  those caused a severe compile-time blowup (LLVM choking on an already deeply-nested
  `KernelFunctionOperation` tree doubled in nesting depth), which is why Πₖ/ε_Kˢ are computed offline instead
  of patched online.
- `utils.jl` -- `closest_factor_number()` (FFT-friendly grid sizes), `show_gpu_status()` (unchanged from KH).

### Post-processing pipeline (`postprocessing/`)
Same 01-06 structure as KHAPE, adapted for horizontal (x,y) filtering instead of KH's (x,z):

| Script | Change from KHAPE |
|--------|--------|
| `01_filter_fields.py` | filters in (x,y) instead of (x,z); filter scales are free parameters again (no longer need to match an online `filter_ℓs`) |
| `02_sort_density.py` | unchanged -- the Winters sort is dimension-agnostic |
| `03_energy_transfer.py` | now computes Πₖ offline (`include_pi_k=True`, was `False` when Julia provided it) |
| `04_sfs_ke_budget.py` | biggest change -- reads Πₖ from `03`'s output (`load_energy_transfer`) instead of the now-nonexistent online field; computes ε_Kˢ offline via `calculate_sfs_ke_dissipation`; restricts the KE tensors to horizontal-only components so the budget's LHS (SFS KE) and RHS (Πₖ, ε_Kˢ) stay dimensionally consistent |
| `05_sfs_ape_budget.py` | filters in (x,y); diffusivity κ now read from `nu`/`Pr` global attributes (a constant `ScalarDiffusivity`), not a `ds.κ` spatial field (which only exists for non-constant closures and was never actually populated here) |

`aux00_utils.py`'s `GaussianFilter` class filters (x,y), both periodic (`mode='wrap'` on both), replacing
KH's (x periodic, z bounded `mode='nearest'`). `condense_velocities` (u,v,w) is used throughout instead of
KH's `condense_uw_velocities` (u,w only, valid for the 2D x-z KH case) -- though see the horizontal-only
restriction above for how w is still excluded specifically from the KE cross-scale tensors.

`sweep*` scripts, `validation/`, and the standalone `plot*`/`anim1_panels.py` scripts still describe the KH
pipeline's online-vs-offline validation setup and have not been adapted -- there is no online Πₖ/ε_Kˢ to
validate against anymore. `anim2_surface_buoyancy.py` is new: a simple standalone script that animates the
surface buoyancy field to a GIF (no ffmpeg dependency, uses matplotlib's `PillowWriter`).

### Key dependencies
- **Python**: `numpy`, `xarray`, `scipy`, `matplotlib`, `dask`, `gcm_filters`, `netcdf4`
- **Julia**: `Oceananigans` v0.110.8, `Oceanostics` v0.17.2, `NCDatasets`, `CairoMakie` (Julia 1.11.2)

## Physics Reference

- **TPE** = integral of g*rho*z dV (total potential energy)
- **RPE** = minimum PE achievable by adiabatic rearrangement (from sorted reference state)
- **APE** = TPE - RPE (available for conversion to KE)
- **Πₖ**, **Π_A** -- cross-scale energy transfer (sub-filter to resolved). Πₖ is horizontal-only here (see
  Architecture); Π_A is unrestricted (density/APE has no analogous "diagnostic component" issue).
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

- **Oceanostics bug**: `GaussianFilter(; dims=(1,2), σ)` crashes (heap corruption -> SIGILL) on a grid with
  real `Ny>1` and periodic y -- filed as
  [tomchor/Oceanostics.jl#262](https://github.com/tomchor/Oceanostics.jl/issues/262), with a minimal
  reproducer. `dims=(1,)` alone (periodic x only, matching KH's `Ny=1` case) does not crash -- this repo's
  Julia script works around it via `SequentialGaussianFilter` for filtered-field outputs, and avoids the
  composite KE-transfer functions entirely by moving Πₖ/ε_Kˢ offline (see Architecture).
- **Budget closure is not yet validated at production resolution.** A 1-day toy run (16x16x4) and a 20-day
  run at default resolution (48x48x8) both fail the KHAPE-style closure test (`rms(residual)/min(rms(terms))
  < 10%), though the 20-day run is substantially closer: the raw reported percentage is inflated by an
  anomalously tiny ε_Kˢ (little sub-filter-scale content has developed yet at this resolution/duration);
  comparing the residual against the *dominant* budget term instead gives ~25-87% depending on filter scale,
  improving with run length. This looks like a resolution/duration limitation (mesoscale eddies are barely
  resolved at 48x48, and even 20 days is only a handful of e-folding times of the Eady growth rate for this
  setup, ~1.2 days) rather than a code bug -- revisit with a finer-resolution, longer HPC run before trusting
  the closure numbers, or treating a failure as a real regression.
- `online_ke_transfer_validation.md` is a KH-era dev note about computing Πₖ online and validating it against
  the offline pipeline -- it predates this fork's move back to fully-offline Πₖ/ε_Kˢ and no longer describes
  current behavior.
- Output files (`.nc`, `.mp4`, `.pdf`, `.png`, `.gif`, `.jld2`) are excluded from git.
