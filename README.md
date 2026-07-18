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
   output. Filtered velocity/buoyancy fields, the cross-scale KE transfer (Πₖ), and SFS KE dissipation
   (ε_Kˢ) are all computed **online** (full 3D, via Oceanostics) at each of two requested filter scales.
2. **Post-processing** (`postprocessing/`) — filters fields (independently, offline), sorts density, and
   computes the APE-side energy transfer and SFS budgets:
   - `01_filter_fields.py` — Gaussian-filter velocity (u,v,w) and buoyancy horizontally (x, y) at each
     requested length scale (its own offline filter pass, decoupled from the simulation's online one --
     see CLAUDE.md for why)
   - `02_sort_density.py` — Winters (1995) density sort for the reference state
   - `03_energy_transfer.py` — cross-scale APE transfer Π_A and the APE↔KE exchange term (Πₖ is read
     directly from the simulation output here, not recomputed)
   - `04_sfs_ke_budget.py` — sub-filter-scale KE budget (Πₖ/ε_Kˢ read from the simulation output)
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
each stage runs after the previous succeeds).

If the simulation's already done and you just want to (re)run analysis, use
`bash postprocessing/submit_budgeting.sh NX=192 NY=192 NZ=32` instead -- same
budgeting_filter → budgeting → plots chain, without resubmitting the simulation.

**Common options**, passed as flags to `submit_all_pbs.sh`/`submit_simulation.sh`:

| Flag | What it does |
|------|-----|
| `GPU=1` | Run the simulation stage on an A100 (post-processing stages are always CPU) |
| `SWEEP=1` | Also run the many-filter-scale transfer-spectrum sweep after budgeting (`submit_all_pbs.sh` only) |
| `FIXED_REF=0\|1\|both` | Fixed-in-time vs. recomputed reference density profile |
| `FILTER_SCALES_M='30000 60000'` | Offline post-processing filter scales, in **meters** (default `"50000 100000"`) |
| `EXTRA_ARGS='...'` | Any `baroclinic_adjustment.jl` CLI flag, passed through verbatim -- this is how GPU-adjacent numerics options (advection scheme, closure, Péclet numbers) reach the simulation |

```bash
# WENO advection instead of the default centered scheme
bash submit_all_pbs.sh NX=192 NY=192 NZ=32 EXTRA_ARGS='--advection_scheme weno'

# Tune the scale-aware closure's Péclet numbers (lower = more dissipation; defaults are
# --Pe_cell_h 100 --Pe_cell_v 50)
bash submit_all_pbs.sh NX=192 NY=192 NZ=32 EXTRA_ARGS='--Pe_cell_h 50 --Pe_cell_v 20'

# Combine: WENO advection + custom Pe_cell + GPU + a 30-day run
bash submit_all_pbs.sh NX=384 NY=384 NZ=128 STOP_TIME=30 GPU=1 \
    EXTRA_ARGS='--advection_scheme weno --Pe_cell_h 50 --Pe_cell_v 50'

# Simulation only, on a GPU, then post-processing separately once you're ready
bash submit_simulation.sh NX=384 NY=384 NZ=128 STOP_TIME=16 GPU=1
cd postprocessing && bash submit_budgeting.sh NX=384 NY=384 NZ=128
```

Run `julia --project baroclinic_adjustment.jl --help` for the full list of simulation-level flags (closure
choice, front width, N², M², latitude, etc.) -- anything shown there can go in `EXTRA_ARGS`. Note units:
`EXTRA_ARGS`'s `--filter_scales_m` (online diagnostics) and `FILTER_SCALES_M` (offline post-processing) are
both in meters but are independent knobs -- set both to the same values if you want them to correspond.

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

**Current status:** budget closure was historically poor at the resolutions tested so far (16×16×4 for 1
day, 48×48×8 for 20 days), but the dominant cause (a domain-padding bug that inflated every volume-integrated
budget term) has since been fixed — see [Known issues](#known-issues). This specific test hasn't yet been
re-run against that fix.

## Known issues

- **Budget closure at production resolution**: after the domain-padding fix above, the filtered
  (large-scale) KE budget residual/dominant ratio dropped from 39.6%/46.2% to 6.2%/4.6% at 96×96×16. Still
  open: whether this holds at production resolution (e.g. 192×192×32 and up), and whether the SFS KE budget
  (as opposed to the filtered/large-scale one) improves comparably — see the CLAUDE.md Notes section for the
  full history.

## Logs

Output files (`.nc`, `.mp4`, `.pdf`, `.png`, `.gif`, `.jld2`) are excluded from git.
