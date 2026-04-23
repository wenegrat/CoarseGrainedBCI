#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from aux03_plotting import budget_colors, run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot 2x2 panel of SFS KE and APE budgets at two filter scales")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc",
                    help="Path to simulation NetCDF file (used to derive budget filenames)")
parser.add_argument("--fixed-reference", action="store_true", default=False,
                    help="Load the fixed-in-time reference profile outputs")
parser.add_argument("--filter-scales", type=float, nargs=2, default=[0.4, 2.0],
                    help="Two filter length scales for left and right columns (default: 0.4 2.0)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES   = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
fixed_reference = args.fixed_reference
ref_suffix = "_fixed_ref" if fixed_reference else ""
ℓ_left, ℓ_right = args.filter_scales
#---

#+++ Load budget data
print("Loading budget data...")
ke_budget  = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ke_budget_integrated{ref_suffix}.nc"),  decode_timedelta=False)
ape_budget = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ape_budget_integrated{ref_suffix}.nc"), decode_timedelta=False)
print(f"  Filter scales available: {ke_budget.filter_length_scale.values}")
#---

#+++ Define budget terms (shared colors across all panels)
ke_terms = {
    r"$-\partial_t$ SFS KE":   ("∫-∂ₜ SFS KE dV",    budget_colors["tendency"]),
    r"$\Pi_{KE}$":             ("∫Π_KE dV",           budget_colors["flux"]),
    r"$-\varepsilon_s$":       ("∫-εₛ dV",            budget_colors["dissipation"]),
    r"SFS APE $\to$ KE":       ("∫(SFS APE->KE) dV",  budget_colors["exchange"]),
    r"residual":               ("residual_KE",         budget_colors["residual"]),
}
ape_terms = {
    r"$-\partial_t$ SFS APE":  ("∫-∂ₜ SFS APE dV",   budget_colors["tendency"]),
    r"$\Pi_{APE}$":            ("∫Π_APE dV",          budget_colors["flux"]),
    r"$-\chi_s$":              ("∫-χₛ dV",            budget_colors["dissipation"]),
    r"SFS KE $\to$ APE":       ("∫(SFS KE->APE) dV",  budget_colors["exchange"]),
    r"$R^s$":                  ("∫Rˢ dV",             "C4"),
    r"residual":               ("residual_APE",        budget_colors["residual"]),
}
#---

#+++ Plot 2×2 figure
print("Creating 2×2 budget panel plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

budget_configs = [
    (0, ke_budget,  ke_terms,  "SFS KE budget"),
    (1, ape_budget, ape_terms, "SFS APE budget"),
]

for row, budget, terms, row_title in budget_configs:
    for col, ℓ in enumerate([ℓ_left, ℓ_right]):
        ax = axes[row, col]
        for label, (var, color) in terms.items():
            data = budget[var].sel(filter_length_scale=ℓ, method="nearest").dropna("time")
            ls = "--" if "residual" in var else "-"
            lw = 1.0 if "residual" in var else 1.5
            ax.plot(data.time, data.values, label=label, color=color, ls=ls, lw=lw)

        ax.set_ylabel("Budget terms [W]")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3, lw=0.5)
        actual_ℓ = float(budget.filter_length_scale.sel(filter_length_scale=ℓ, method="nearest"))
        ax.set_title(f"{row_title}  ($\\ell = {actual_ℓ:.1f}$)")
#---

#+++ Share y-axis within each column
for col in range(2):
    ymin = min(axes[row, col].get_ylim()[0] for row in range(2))
    ymax = max(axes[row, col].get_ylim()[1] for row in range(2))
    for row in range(2):
        axes[row, col].set_ylim(ymin, ymax)
#---

#+++ Legend and labels
handles, labels = axes[0, 0].get_legend_handles_labels()
handles2, labels2 = axes[1, 0].get_legend_handles_labels()
for h, l in zip(handles2, labels2):
    if l not in labels:
        handles.append(h)
        labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=True, fancybox=True)

for ax, letter in zip(axes.flat, "abcd"):
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.8))

label = run_label(ke_budget.attrs)
suptitle_parts = []
if label:
    suptitle_parts.append(label)
if fixed_reference:
    suptitle_parts.append("(fixed reference)")
if suptitle_parts:
    fig.suptitle("  ".join(suptitle_parts), fontsize=11)
#---

#+++ Save
outfile = str(FIGURES / f"{stem}_sfs_budgets_2x2{ref_suffix}.png")
fig.savefig(outfile, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to: {outfile}")
#---
