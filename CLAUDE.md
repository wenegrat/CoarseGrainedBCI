# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KHAPE (Kelvin-Helmholtz Available Potential Energy) computes Available Potential Energy (APE) from Kelvin-Helmholtz instability simulations using the Winters et al. (1995) sorting method. The pipeline is:
1. **Julia simulation** (Oceananigans.jl on GPU) → NetCDF output
2. **Python post-processing** → filter fields, sort density, compute energy transfer and SFS budgets, plot

GitHub remote: `git@github.com:tomchor/APE_calculations.git`

## Running the Code

### Full pipeline (simulation + post-processing + sweep)
```bash
bash submit_all_pbs.sh                        # default Nz=2048, FIXED_REF=0
bash submit_all_pbs.sh NZ=1024 FIXED_REF=1   # custom resolution, fixed reference
```
Jobs are chained via PBS `afterok` dependencies. Always use `submit_*.sh` wrappers, never submit `*.pbs` files directly.

### Simulation only
```bash
bash submit_simulation.sh NZ=2048
```
Account: `UMCP0028`, queue: `casper`, 1× A100, 8 cores, 64 GB RAM.

### Post-processing only
```bash
cd postprocessing
bash submit_budgeting.sh NZ=2048 FIXED_REF=both   # runs both reference profile variants
bash submit_sweep.sh NZ=2048 FIXED_REF=both
```
`FIXED_REF=0` (default) recomputes the reference profile each timestep; `FIXED_REF=1` fixes it to t=0; `FIXED_REF=both` submits both variants sharing a single filter job.

### Local CPU simulation (development)
```bash
julia --project -t 8 kelvin_helmholtz_instability.jl
```

### Running tests
```bash
cd tests
pytest test_budgets.py -s                           # default (time-varying reference)
pytest test_budgets.py -s --ref-suffix _fixed_ref   # fixed reference variant
```
Tests check SFS KE and APE budget closure: rms(residual)/min(rms(terms)) < 10%. They expect post-processing output in `postprocessing/output/` for `khi_Nz512_Ri0.10`.

### Python environment
```bash
conda env create -f environment.yml   # creates env "py313"
conda activate py313
```

## Architecture

### Post-processing pipeline (`postprocessing/`)

Sequential numbered scripts, each reading the previous step's output:

| Script | Purpose |
|--------|---------|
| `01_filter_fields.py` | Gaussian-filter velocity and buoyancy at multiple length scales |
| `02_sort_density.py` | Sort density to compute reference state (Winters et al. 1995) |
| `03_energy_transfer.py` | Cross-scale KE and APE transfer terms (Π_KE, Π_APE) |
| `04_sfs_ke_budget.py` | Sub-filter-scale KE budget terms |
| `05_sfs_ape_budget.py` | Sub-filter-scale APE budget terms |
| `06_plot_budgets.py` | Plot budget time series |

`inv*` scripts are the sweep variant (parameter sweep over filter scales): `inv1` filters, `inv2` computes transfer, `inv3` plots spectra.

Shared utilities:
- `aux00_utils.py` — data loading (`load_dataset_and_grid`), filtering (`filter_fields`), domain padding
- `aux01_pe_functions.py` — density sorting, potential energy calculations
- `aux02_ke_functions.py` — kinetic energy transfer calculations
- `aux03_plotting.py` — plotting helpers

All post-processing scripts accept `--filename`, `--filter-scales`, `--n-workers`, `--fixed-reference` via argparse. Output goes to `postprocessing/output/`.

### Julia layer
- `kelvin_helmholtz_instability.jl` — main simulation (Oceananigans.jl, WENO(5), adaptive timestep)
- `utils.jl` — `closest_factor_number()` (FFT-friendly grid sizes), `show_gpu_status()`
- Setup: shear flow u(z)=tanh(z), stratification b(z)=0.025·tanh(z/0.25), Ri=0.1
- Domain: Lx=Lz=10, Ly=5 (periodic x,y; bounded z)
- Output: `output/kelvin_helmholtz_instability_NxNyNz.nc`

### Key dependencies
- **Python**: `numpy`, `xarray`, `scipy`, `matplotlib`, `dask`, `gcm_filters`, `netcdf4`
- **Julia**: `Oceananigans`, `Oceanostics`, `CUDA`, `NCDatasets`

## Physics Reference

- **TPE** = ∭ g·ρ·z dV  (total potential energy)
- **RPE** = minimum PE achievable by adiabatic rearrangement (from sorted reference state)
- **APE** = TPE − RPE  (available for conversion to KE)
- **Π_KE**, **Π_APE** — cross-scale energy transfer (sub-filter to resolved)
- Physical constants: `g=9.81`, `ρ0=1025`

## Code Style

- Always delimit code sections with `#+++` on the opening line and `#---` on the closing line:
  ```python
  #+++ Section name
  ...code...
  #---
  ```

## Maintenance Rules

- **Always update `README.md` when the job submission scheme changes.** This includes: adding/removing/renaming PBS scripts or wrapper scripts, changing argument names or defaults, adding new pipeline stages, or changing job dependency chains.

## Notes

- Output files are excluded from git (`.nc`, `.mp4`, `.pdf`, `.png`, `.jld2`).
- Simulation output can reach 650 GB; scratch directory is `/glade/derecho/scratch/tomasc/khape/output/`.
- Logs: `logs/<job_name>.log` (PBS), `logs/<job_name>.out` (Python stdout via tee). Job names follow `<stage>_Nz<NZ>_Ri0.10[_fixed_ref]`.
