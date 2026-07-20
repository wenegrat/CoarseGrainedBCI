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
`--latitude`, `--nu`, `--Pr`, `--stop_time` (days, default 20), `--filter_scales_m` (two horizontal FWHM scales
in meters, default 50000 100000 -- matches the units used throughout the offline post-processing pipeline;
renamed from the old km-based `--filter_scales` specifically so a stale invocation fails loudly instead of
silently applying scales 1000x too small), `--progress_interval` (default 100; use a small value for
short/smoke-test runs where the default interval may never be reached), `--advection_scheme` (`centered`
default or `weno`), `--closure` (`scale_aware` default, `constant`, or `smagorinsky`) with its `--Pe_cell_h`/
`--Pe_cell_v`/`--nu_h`/`--nu_v`/`--Pr` sub-parameters, `--architecture` (`auto` default -- uses a GPU if
`CUDA.functional()`, else `CPU()`; `cpu`/`gpu` to force one, `gpu` erroring loudly instead of silently
falling back if no GPU is found), `--implicit` (boolean flag, default false; forces `--closure=nothing` and
`--advection_scheme=WENO(order=9)` -- fully implicit/numerical dissipation, an implicit-LES configuration --
warning and overriding, not erroring, if `--closure`/`--advection_scheme`/`--Pe_cell_h`/`--Pe_cell_v`/
`--nu_h`/`--nu_v` were also explicitly set; see "Implicit-LES mode" below for the diagnostic implications).
See the file's own `--help` for full documentation of each.

**Implicit-LES mode (`--implicit`):** the online `ε_Kˢ`/`εˡ` SFS dissipation diagnostics and the offline APE
dissipation term both derive from an explicit closure's ν/κ, which is `nothing` under `--implicit` -- they
read ~0 rather than the real (but numerical, untracked) dissipation actually happening via WENO's own
implicit truncation error. `04_sfs_ke_budget.py`/`05_sfs_ape_budget.py` detect this via the simulation's own
`implicit` NetCDF attribute and substitute a residual-based estimate for the domain-*integrated* ε_Kˢ/ε_Aˢ
only (solving the budget for ε_Kˢ/ε_Aˢ assuming every other term is correct, which makes `residual_K`/
`residual_A` ≈0 by construction) -- the *local* (spatial) `ε_Kˢ`/`ε_Aˢ` fields are left at their
uninformative near-zero values, since a local residual would also absorb real spatial transport/
flux-divergence terms that only vanish upon domain integration, not pointwise. Both substituted variables
carry a `method` attribute in the output NetCDF recording this. This residual-based estimate can go
negative at some times (unlike a true dissipation rate), since it's absorbing whatever error/noise sits in
the other terms -- expected, not a bug. `06_plot_budgets.py`/`plot3_budgets.py`/`plot5`/`plot6`/`anim3` need
no changes, since they consume the same variable names either way. Also note: `NetCDFWriter`'s NCDatasets.jl
backend cannot write a raw Julia `Bool` as a NetCDF attribute at all (confirmed -- errors "KeyError: key Bool
not found"); `implicit` is written as `Int` (0/1) via a separate `netcdf_attributes` namedtuple, keeping
`params.implicit` a genuine `Bool` for use in conditionals earlier in the file.

### HPC job submission

`submit_*.sh`/`*.pbs` (repo root and `postprocessing/`) are adapted for `baroclinic_adjustment.jl`/BCI
naming (`bci_Nx${NX}_Ny${NY}_Nz${NZ}`), chained via `qsub -W depend=afterok`. Four entry points, depending
on how much of the pipeline you need:

| Script | Stages run | Use when |
|--------|-----------|----------|
| `bash submit_all_pbs.sh` | simulation → budgeting_filter → budgeting → plots (+ sweep_filter → sweep_transfer if `SWEEP=1`) | starting from scratch |
| `bash submit_simulation.sh` | simulation only | you only want the `.nc` output, no post-processing yet |
| `bash postprocessing/submit_budgeting.sh` | budgeting_filter → budgeting → plots | simulation already completed, (re)run analysis only (e.g. after changing filter scales, or after the simulation succeeded but post-processing failed) |
| `bash postprocessing/submit_sweep.sh` | sweep_filter → sweep_transfer | just the many-filter-scale transfer-spectrum sweep, independent of the fixed-2-scale budgeting above |

```bash
# Full pipeline, WENO advection on a GPU
bash submit_all_pbs.sh NX=128 NY=128 NZ=64 STOP_TIME=20 \
    EXTRA_ARGS='--advection_scheme weno --Pe_cell_h 50 --Pe_cell_v 50' GPU=1

# Simulation only
bash submit_simulation.sh NX=384 NY=384 NZ=128 STOP_TIME=16 GPU=1

# Post-processing only, against an already-completed simulation
cd postprocessing && bash submit_budgeting.sh NX=384 NY=384 NZ=128

# Many-filter-scale sweep only
cd postprocessing && bash submit_sweep.sh NX=384 NY=384 NZ=128
```

Shared flags across the scripts that take them: `NX`/`NY`/`NZ` (grid resolution), `STOP_TIME` (simulation
days, `submit_all_pbs.sh`/`submit_simulation.sh` only), `EXTRA_ARGS` (extra `baroclinic_adjustment.jl` CLI
args passed through verbatim -- quote multi-word values), `GPU=1` (requests an A100 for the simulation
stage only via a `qsub -l` override, since `#PBS` directives are static; post-processing stages are pure
CPU/numpy/dask regardless), `FILTER_SCALES_M` (offline post-processing filter scales in **meters**, passed
to `budgeting_filter.pbs`/`01_filter_fields.py` and `plots.pbs` -- **left unset by default**, which means
"use whatever the simulation actually used" rather than a separate hardcoded default; see the "Filter
scales: single source of truth" note below), `FIXED_REF=0|1|both` (fixed-in-time vs. recomputed reference
density profile; `both` submits both budgeting variants, sharing one filter-step run), `SWEEP=1`
(`submit_all_pbs.sh` only, adds the sweep branch after budgeting).

**Filter scales: single source of truth.** `baroclinic_adjustment.jl`'s `--filter_scales_m` (online
diagnostics) and the offline pipeline's `--filter-scales`/`FILTER_SCALES_M` used to be two fully independent
knobs with matching-but-separate hardcoded defaults (`50000 100000` in both places) -- easy to let drift
silently, since nothing checked they agreed. Fixed by making the simulation's own choice the source of
truth for the common case: `filter_scales_m` is now recorded as a NetCDF global attribute (confirmed
NCDatasets.jl writes/reads a `Vector{Float64}` attribute cleanly, unlike `Bool` -- see the `--implicit`
branch's NetCDF-attribute gotcha), and every offline script that previously hardcoded `[50000, 100000]` as
its `--filter-scales`/`--filter-scale` default now instead falls back to reading that attribute when the
flag isn't explicitly passed:
- `01_filter_fields.py` -- defaults to `ds.attrs["filter_scales_m"]` (falls back to `[50000, 100000]` only
  for older files that predate the attribute)
- `plot3_budgets.py` -- defaults to the first two scales actually present in the budget file's own
  `filter_scale` coordinate (which reflects whatever `01_filter_fields.py` actually used, so this is
  consistent with the point above by construction)
- `budgeting_filter.pbs`/`plots.pbs` -- `FILTER_SCALES_M` unset means "don't pass `--filter-scales` at
  all", letting the Python scripts' new defaults take over; `plots.pbs`'s per-scale loop (driving
  `plot5`/`plot6`/`anim3`) falls back to every scale in the budget file when `FILTER_SCALES_M` is unset

Passing `--filter-scales`/`FILTER_SCALES_M` explicitly still works exactly as before (deliberately using
different offline scales than the simulation's online ones is a real, intentional workflow -- re-exploring
offline without rerunning the simulation). The only thing that changed is what happens when you *don't*
specify it: previously a silent, independently-hardcoded guess; now derived from the one place that
actually knows what was used.

The `plots` stage runs `plot3_budgets.py`, `plot5_vorticity_strain_flux.py`/`plot6_middepth_snapshots.py`
(once per filter scale -- `FILTER_SCALES_M` if set, else every scale in the budget file),
`anim2_surface_buoyancy.py`, and `anim3_panels.py` (once per filter scale) -- the latter depends
specifically on the `FIXED_REF=0` budgeting output (no
`--fixed-reference` support in the plotting scripts themselves), so `submit_budgeting.sh` skips plots
automatically if only `FIXED_REF=1` was requested. `simulation.pbs`'s default `mem=64GB` is sized for a
modest resolution and is a *static* PBS resource request (doesn't scale with `NX*NY*NZ` automatically) --
bump it by hand in `simulation.pbs` for much larger grids. **Before first use**, every `*.pbs` file needs
its `#PBS -A`/`#PBS -M` placeholders (`CHANGE_ME`) replaced with your own account code and email (PBS
directives are parsed statically, so these can't be centralized), and `hpc_env.sh`'s `PYTHON` placeholder
needs to point at your own HPC Python environment (must have `postprocessing/tests/requirements.txt`
installed).

**Output storage on the HPC:** `output/` and `postprocessing/output/` are *not* plain directories on the
HPC -- they're symlinks into scratch space (e.g. `/glade/derecho/scratch/$USER/CoarseGrainedBCI/output/`
and `.../CoarseGrainedBCI/postprocessing/output/`, mirroring the repo's own layout), because HPC home
directories tend to have small quotas (100GB is common) that large-resolution runs blow through fast --
raw simulation NetCDFs and derived post-processing files (filtered velocities, energy transfer, SFS
budgets) both scale steeply with `Nx*Ny*Nz` and can reach 10s-100s of GB each at resolutions like
384x384x128. Both directories are fully gitignored (not just their `*.nc` contents) specifically because
git refuses to operate on tracked paths that sit behind a symlinked directory ("beyond a symbolic link"
fatal errors on `git stash`/checkout/etc if a `.gitkeep` is still tracked underneath one). A fresh
checkout needs these created manually before first use: a plain `mkdir output postprocessing/output` for
local (non-HPC) development, or the scratch-symlink setup above for the HPC. This is unrelated to (but
was investigated alongside) a separate large consumer of HPC home quota: the Julia package depot
(`~/.julia`, ~60GB with CUDA artifacts) defaults to the home directory unless `JULIA_DEPOT_PATH` is set.
On this HPC it's been migrated to `$WORK/.julia` (`mv ~/.julia $WORK/.julia`), with `export
JULIA_DEPOT_PATH="$WORK/.julia"` set in `~/.bash_profile` so interactive shells and any `#!/bin/bash -l`
(login-shell) PBS script -- including `simulation.pbs`, which also sets it explicitly as a version-controlled
safety net -- all agree on the same depot. Moving the depot invalidates Julia's precompiled cache (paths
are baked in), so expect a one-time recompile the first time each package is used afterward. A fresh
checkout/user on this HPC needs the same migration + `~/.bash_profile` line repeated -- it isn't captured
by the repo alone, since it's personal shell config.

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
`postprocessing/output/bci_Nx48_Ny48_Nz8_*` (run `00_get_budgets.sh` first). Historically failed; see the
Notes entry on the domain-padding bug fix below before assuming it still does -- that bug inflated every
budget term computed on this dataset too, and hasn't yet been re-checked against this specific test.

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

`validation/` and the standalone `plot*`/`anim1_panels.py` scripts still describe the KH pipeline's
online-vs-offline validation setup and have not been adapted -- there is no online Πₖ/ε_Kˢ to validate
against anymore. `anim2_surface_buoyancy.py` is new: a simple standalone script that animates the surface
buoyancy field to a GIF (no ffmpeg dependency, uses matplotlib's `PillowWriter`).

`sweep1_filter_fields.py` -> `sweep2_energy_transfer.py` -> `sweep3_plot_transfer_spectrum.py` (filter at
many log-spaced scales, compute Πₖ/Π_A at each, plot the resulting cross-scale transfer spectrum as a
Hovmöller of time vs. ℓ) have been adapted for BCI: `sweep1`'s filter-scale range is now data-driven
(`--scale-min`/`--scale-max` default to 2x the grid spacing and 40% of the domain width Lx respectively,
rather than the old hardcoded KH range in different units), `sweep2`'s log message now says "x and y" not
"x and z", and `sweep3`'s `SymLogNorm(linthresh=...)` scales with the data's own magnitude (`vmax*1e-3`)
instead of a fixed absolute value tuned for KH's much smaller Πₖ/Π_A magnitudes. Note `sweep2` calls
`calculate_energy_transfer` without `--fixed-reference`, so it redoes a full Winters sort per filter scale
(the sort itself doesn't depend on filter scale, but nothing shares it across scales) -- fine for a handful
of scales, expensive for `sweep1`'s default 30-scale sweep on a full-length run.

`sweep3` also gained a second row of panels that was previously missing: it already computed a `1/ℓ`
coordinate (`inv_scale`) specifically so a proper spectrum line plot could use it as the x-axis, but the
plotting code only ever drew the Hovmöllers -- the spectrum itself was never implemented. Now it also plots
the time-mean (±1 std across time, excluding the first `--min-time-days` as an initial-transient cutoff)
of ∫Π_K dV/∫Π_A dV vs. 1/ℓ, with a dashed vertical line at the theoretical Eady deformation radius
`Ld = N·Lz/|f0|` (computed from the run's own `N2`/`Lz`/`latitude` attrs). The shaded band is temporal
spread of the diagnostic itself, not a statistical confidence interval -- there's only one simulation
realization, so don't read it as sampling uncertainty on the mean.

`anim3_panels.py` is also new: a 6-panel GIF animation (`--filename ...`, `--filter-scale` in meters,
`--fps`, `--dpi`, `--clim-percentile`) combining surface buoyancy, surface Rossby number ζ/f, the SFS
APE→KE "conversion" term, cross-scale KE/APE fluxes Πₖ/Π_A, and their sum, all at the top z-level, plus a
full-width bottom row with the SFS KE and APE budget time series (each with a vertical marker tracking the
current frame). Two things worth knowing if extending it:
- Some offline APE-pipeline fields (`Π_A`, the KE↔APE exchange term) are stored with dims `(..., x, y)`
  instead of `(..., y, x)` like every other field (`b`, `ζ`, `Π_K`) -- a real bug in how those DataArrays
  get built upstream (`aux01_pe_functions.py`), not just a plotting quirk. Plotting `.values` directly
  against `(x_km, y_km)` renders them rotated 90° relative to everything else. `anim3_panels.py`'s
  `fix_orientation()` transposes any field to `(..., y_dim, x_dim)` before plotting regardless of its
  stored order, so this can't recur there -- but any *other* script plotting `Π_A` or an exchange term
  directly (e.g. a future `plot_middepth`-style script) needs the same treatment.
- `constrained_layout` cannot reconcile equal-aspect square map axes sharing one GridSpec with a wide,
  non-square row (it silently fails -- "axes sizes collapsed to zero" -- and produces uneven gaps).
  `anim3_panels.py` avoids this with explicit `wspace`/`hspace`/margins plus fixed-fraction colorbars
  (`fraction=0.046, pad=0.04`) instead of relying on the layout solver.

`plot5_vorticity_strain_flux.py` is new: conditions Πₖ, Π_A, and Πₖ+Π_A on the *filtered*-field vorticity
ζ̄/f0 and strain σ̄/|f0| (`--filename ...`, `--filter-scale` in meters, `--time` in days, `--z` in meters,
`--n-bins`, `--min-count`, `--clim-percentile`), following the joint-PDF/conditional-mean method of
[Balwada et al. (2021, JPO)](https://doi.org/10.1175/JPO-D-21-0016.1) but with our own cross-scale energy
fluxes in place of their vertical tracer flux. Produces, per filter scale: the JPDF, a conditional-mean
panel and a "net contribution" panel (conditional mean × JPDF) for each of the three flux quantities, plus
the flux fraction attributable to strain-dominated (SD) vs. vorticity-dominated (AVD/CVD) regions (the
σ=|ζ̄| partition from the paper). f0 is a single reference Coriolis value (evaluated at y=0), not local
f(y), to keep the JPDF axes free of an implicit y-dependence. Two gotchas hit while building it:
- `ūᵢ` (the filtered-velocity file) is *also* stored with `(..., x_caa, y_aca)` instead of `(..., y_aca,
  x_caa)` -- the same orientation bug as `Π_A`/the exchange term (see above), just in a different file.
  Uses the same `fix_orientation()` pattern.
- When overlaying the σ=|ζ| "V" boundary with `ax.plot(z, np.abs(z), ...)`, sampling `z` at only 2 points
  (the endpoints) draws a flat line at the max, not a V -- `np.abs` needs enough intermediate points to
  trace the actual piecewise-linear shape.

`plot3_budgets.py` had the same KH-era bug as the old `plot3_budgets_bci.py` scratch copy: a hardcoded
`ax.set_xlim(right=140)` (a leftover non-dimensional-time assumption from KH) that clipped almost the
entire BCI time axis, since our time coordinate is raw seconds up to ~10⁶. Fixed by plotting `time/86400`
(days) and dropping the `xlim` call entirely; also updated the default `--filter-scales` from KH's `[7,
1]` to BCI's `[50000.0, 100000.0]` (meters) and the per-panel `ℓ=` title to display km.

`plot6_middepth_snapshots.py` is new: a permanent version of the ad hoc mid-depth snapshot scripts used
earlier in this project's investigation (buoyancy, Rossby number ζ/f, cross-scale KE flux Πₖ, cross-scale
APE flux Π_A, single time/depth/filter-scale, 2x2 panel). Uses the same `fix_orientation()` and
`coriolis_f()` patterns as `anim3_panels.py`/`plot5_vorticity_strain_flux.py` (`--filename`,
`--filter-scale` in meters, `--time` in days, `--z` in meters, `--clim-percentile`).

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
- **Budget closure gap was (mostly) explained by a domain-padding bug -- fixed.** `aux00_utils.py`'s
  `load_dataset_and_grid()` used to call a `_pad_domain_in_z()` helper that doubled the z-domain via
  edge-value replication before recomputing `dV`, left over from the old KH pipeline where filtering also
  operated in z. It served no purpose once filtering became horizontal-only, but silently remained, so every
  volume integral computed via `integrate()` throughout this entire fork's history (everything downstream of
  `load_dataset_and_grid`: all of 01-05, `plot4_panels.py`, `S2_panels.py`, `inv06_total_mixing_check.py`,
  `S4_thumbnail.py`, the `sweep*` scripts) summed over roughly twice the physical domain, with the padded
  cells inflating each budget term by a *different*, term-specific factor (measured on one dataset/timestep:
  Πₖ 1.96x, εˡ 1.25x, ε_Kˢ 1.48x, raw w 9.21x) rather than a uniform scale that would cancel in a residual.
  Found during a code review requested specifically to look for bugs affecting the large-scale KE budget;
  confirmed via direct numerical test, then removed entirely (function and call site) per explicit
  instruction, since padding-in-z has no remaining purpose. Grep-confirmed no other code depends on it.
  Re-running the full 01->04 pipeline on `bci_Nx96_Ny96_Nz16_nonhydro` after the fix dropped the filtered
  (large-scale) KE budget residual/dominant from 39.6%/46.2% to **6.2%/4.6%** (ℓ=50/100km) -- right in the
  range of tomchor's own Eady-example floor (~11-15%, see above), essentially closing the multi-week
  "budgets don't converge with resolution" investigation. All closure percentages quoted anywhere earlier in
  this file or in conversation history predate this fix and should be treated as unreliable until
  regenerated; the buoyancy-convention investigation above became moot once this fix landed (see the
  "definitive, exact result" in conversation history: `w̄b̄` and `w̄b̄ᵣ` are provably identical for our own
  simulation regardless of the padding bug, since it never affected that particular identity). Still open:
  whether this fix changes closure at other resolutions (e.g. 192x192x32), and whether the SFS KE budget
  (as opposed to the filtered/large-scale one) improves comparably.
- **Minor, unresolved: Gaussian filter truncation-radius mismatch.** Oceanostics' online `GaussianFilter`
  defaults to a 2σ truncation radius (`ceil(Int, 2σ/Δ)` grid cells); this repo's offline
  `scipy.ndimage.gaussian_filter1d` defaults to 4σ (`truncate=4.0`). Verified numerically on a real w field:
  ~1.3% relative rms difference, 0.9999 correlation -- real but small, not yet fixed.
- `online_ke_transfer_validation.md` is a KH-era dev note about computing Πₖ online and validating it against
  the offline pipeline -- it predates both this fork's move to fully-offline Πₖ/ε_Kˢ and the subsequent move
  back to online (see above), so it still doesn't describe current behavior, though the general idea
  (validate online against offline before trusting it) is exactly what was done again for this switch.
- Output files (`.nc`, `.mp4`, `.pdf`, `.png`, `.gif`, `.jld2`) are excluded from git.
