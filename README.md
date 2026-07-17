# BCI — Baroclinic Coarse-Grained Instability

Computes coarse-grained kinetic energy (KE) and Available Potential Energy (APE) budgets for an idealized
baroclinic-instability channel, using the Winters et al. (1995) sorting method for APE and the Aluie et al.
(2018, JPO) coarse-graining framework for cross-scale KE/APE transfer.

This is a fork of [CoarseGrainedKHAPE](https://github.com/tomchor/CoarseGrainedKHAPE) (which targets a 2D
x–z Kelvin-Helmholtz instability), adapted to a 3D **double-front, doubly-periodic-horizontal** baroclinic
adjustment setup, following the [Oceananigans baroclinic_adjustment
example](https://clima.github.io/OceananigansDocumentation/stable/literated/baroclinic_adjustment).

## Pipeline overview

1. **Julia simulation** (`baroclinic_adjustment.jl`) — runs the baroclinic adjustment and writes NetCDF
   output. Filtered velocity/buoyancy fields are computed online; the cross-scale KE transfer (Πₖ) and SFS
   KE dissipation (ε_Kˢ) are **not** computed online (see [Known issues](#known-issues)) and are instead
   computed offline in the Python pipeline.
2. **Post-processing** (`postprocessing/`) — filters fields, sorts density, computes energy transfer and
   SFS budgets:
   - `01_filter_fields.py` — Gaussian-filter velocity (u,v,w) and buoyancy horizontally (x, y) at each
     requested length scale
   - `02_sort_density.py` — Winters (1995) density sort for the reference state
   - `03_energy_transfer.py` — cross-scale APE transfer Π_A, the APE↔KE exchange, and the cross-scale KE
     transfer Πₖ (restricted to horizontal velocity components)
   - `04_sfs_ke_budget.py` — sub-filter-scale KE budget (reads Πₖ from step 03; computes ε_Kˢ offline)
   - `05_sfs_ape_budget.py` — sub-filter-scale APE budget
   - `06_plot_budgets.py` — plot budget time series
   - `anim2_surface_buoyancy.py` — animate the surface buoyancy field to a GIF

## Setup

### Julia
Requires Julia 1.11.x. With [juliaup](https://github.com/JuliaLang/juliaup) installed:
```bash
juliaup add 1.11.2
juliaup override set 1.11.2   # pins this directory to 1.11.2
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Python
`environment.yml` is a Linux conda lockfile (built on an HPC) — on macOS or for quick local development, use
a plain venv instead:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r tests/requirements.txt
```

## Running the simulation

```bash
# Default resolution (48x48x8), 20-day run
julia --project -t 8 baroclinic_adjustment.jl

# Custom resolution / short run
julia --project -t 8 baroclinic_adjustment.jl --Nx 16 --Ny 16 --Nz 4 --stop_time 1

# Custom filter scales (horizontal FWHM, in meters) for the online filtered-field diagnostics
julia --project -t 8 baroclinic_adjustment.jl --filter_scales_m 50000 100000
```

Run with `--help` for the full list of CLI arguments (front width, N², M², latitude, viscosity, etc.).

Output is written to `output/<stem>.nc` (full 3D fields) and `output/<stem>_surface.nc` (surface slice),
where `<stem>` encodes the grid size, e.g. `bci_Nx48_Ny48_Nz8`.

### Running on an HPC cluster (PBS)

```bash
bash submit_all_pbs.sh NX=192 NY=192 NZ=32 STOP_TIME=16   # simulation -> budgeting_filter -> budgeting -> plots
```

This chains the simulation, post-processing, and final plots/animations as dependent PBS jobs (submit once,
each stage runs after the previous succeeds). Add `SWEEP=1` for the many-filter-scale transfer spectrum, or
`GPU=1` to run the simulation stage on an A100.

If the simulation's already done and you just want to (re)run analysis, use
`bash postprocessing/submit_budgeting.sh NX=192 NY=192 NZ=32` instead -- same
budgeting_filter → budgeting → plots chain, without resubmitting the simulation.

See `CLAUDE.md`'s "HPC job submission" section for the full reference (all four entry points, every flag),
and fill in each `*.pbs` file's `#PBS -A`/`#PBS -M` placeholders plus `hpc_env.sh`'s `PYTHON` path before
first use.

On the HPC, also symlink `output/` and `postprocessing/output/` into scratch space before first use --
home-directory quotas are typically far too small for raw simulation and derived post-processing NetCDF
files at production resolutions. See `CLAUDE.md`'s "Output storage on the HPC" note for the full rationale
and layout.

## Running the post-processing pipeline

```bash
cd postprocessing
bash 00_get_budgets.sh output/bci_Nx48_Ny48_Nz8.nc --filter-scales 50000 100000
```

Set `N_WORKERS` to control parallelism: `N_WORKERS=4 bash 00_get_budgets.sh ...`.

To animate the surface buoyancy field after a run:
```bash
python anim2_surface_buoyancy.py --filename output/bci_Nx48_Ny48_Nz8.nc
```

## Tests

```bash
pytest tests/ -v -s
```

`test_budgets.py` checks SFS KE/APE budget closure (rms(residual)/min(rms(terms)) < 10%) and requires
post-processing output for the resolution named in `tests/conftest.py`'s `STEM` (run `00_get_budgets.sh`
first). `test_filter.py`/`test_gaussian_filter.py` are self-contained filter unit tests needing no pipeline
output.

**Current status:** this test does not yet pass at the resolutions/durations tested so far (16×16×4 for 1
day, 48×48×8 for 20 days) — see [Known issues](#known-issues). This looks like a resolution/duration
limitation, not a code defect; validating it properly needs a finer, longer HPC run.

## Known issues

- **Oceanostics `GaussianFilter` bug**: a combined `dims=(1,2)` filter (horizontal, both directions) crashes
  with heap corruption on a grid with a real, periodic y-direction — filed as
  [tomchor/Oceanostics.jl#262](https://github.com/tomchor/Oceanostics.jl/issues/262). Worked around locally
  for the plain filtered-field outputs, but not for the composite cross-scale-flux functions (compile-time
  cost was prohibitive), which is why Πₖ/ε_Kˢ are computed offline in Python instead of online as in the
  original KH setup.
- **Budget closure not yet validated at production resolution/duration** — see the CLAUDE.md Notes section
  for details on what's been tried and why the current failure looks like a resolution issue.

## Logs

Output files (`.nc`, `.mp4`, `.pdf`, `.png`, `.gif`, `.jld2`) are excluded from git.
