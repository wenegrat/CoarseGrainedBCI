#!/usr/bin/env python
"""Animate the surface buoyancy field b(x, y, t) from the baroclinic adjustment simulation."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Animate the surface buoyancy field")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file (the _surface.nc variant is used)")
parser.add_argument("--fps", type=int, default=8, help="Frames per second")
parser.add_argument("--dpi", type=int, default=120, help="DPI for output GIF")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parent.parent
ANIMATIONS = REPO_ROOT / "animations"
ANIMATIONS.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
surface_filename = filename.replace(".nc", "_surface.nc")
stem = Path(filename).stem
#---

#+++ Load surface data
print(f"Loading surface data from {surface_filename}...")
ds = xr.open_dataset(surface_filename, decode_times=False)
b = ds["b"].squeeze()  # (time, y_aca, x_caa) after dropping the singleton z dim
x_km = ds.x_caa.values / 1e3
y_km = ds.y_aca.values / 1e3
times_days = ds.time.values / 86400
#---

#+++ Build animation
print(f"Building animation over {len(times_days)} frames...")
vmax = float(np.abs(b).max())

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.pcolormesh(x_km, y_km, b.isel(time=0).values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_aspect("equal")
cbar = fig.colorbar(im, ax=ax, label="Surface buoyancy b [m s⁻²]")
title = ax.set_title(f"t = {times_days[0]:.2f} days")


def update(frame):
    im.set_array(b.isel(time=frame).values.ravel())
    title.set_text(f"t = {times_days[frame]:.2f} days")
    return im, title


anim = FuncAnimation(fig, update, frames=len(times_days), blit=False)

output_filename = str(ANIMATIONS / f"{stem}_surface_buoyancy.gif")
anim.save(output_filename, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
print(f"Animation saved to: {output_filename}")
#---
