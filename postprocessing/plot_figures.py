#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot local quantities from SFS budget output files")
parser.add_argument("--filename", default="output/kelvin_helmholtz_instability_128x1x256.nc",
                    help="Path to base simulation NetCDF file (budget files are derived from this)")
parser.add_argument("--time-stride", type=int, default=5,
                    help="Stride for time selection in plots")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
time_stride = args.time_stride
#---

#+++ Load budget datasets
ke_budget_file = filename.replace(".nc", "_sfs_ke_budget.nc")
ape_budget_file = filename.replace(".nc", "_sfs_ape_budget.nc")

print(f"Loading {ke_budget_file}...")
ds_ke = xr.open_dataset(ke_budget_file, decode_times=False)
print(f"Loading {ape_budget_file}...")
ds_ape = xr.open_dataset(ape_budget_file, decode_times=False)
#---

#+++ Plot local KE budget terms
print("\nPlotting local KE budget terms...")

ke_local_vars = ["KE_of_sfs_flow", "∂ₜ SFS KE", "Π_KE", "εₛ", "SFS APE->KE exchange"]

for var in ke_local_vars:
    ds_ke[var].isel(y_aca=0, time=slice(None, None, time_stride)).plot(
        x="x_caa", col="time", col_wrap=5)
    plot_filename = str(REPO_ROOT / "figures" / (os.path.basename(ke_budget_file).replace(".nc", f"_{var}.png").replace("/", "_")))
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_filename}")
#---

#+++ Plot local APE budget terms
print("\nPlotting local APE budget terms...")

ape_local_vars = ["Eaˢ(ρ, z)", "∂ₜ SFS APE", "Π_APE", "χₛ", "SFS KE->APE exchange", "Rˢ"]

for var in ape_local_vars:
    ds_ape[var].isel(y_aca=0, time=slice(None, None, time_stride)).plot(
        x="x_caa", col="time", col_wrap=5)
    plot_filename = str(REPO_ROOT / "figures" / (os.path.basename(ape_budget_file).replace(".nc", f"_{var}.png").replace("/", "_")))
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_filename}")
#---
