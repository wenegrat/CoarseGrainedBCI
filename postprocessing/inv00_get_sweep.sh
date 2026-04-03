#!/usr/bin/env bash
set -euo pipefail

FILENAME="${1:-output/khi_180x1x512.nc}"

python inv1_filter_fields_sweep.py    --filename "$FILENAME"
python inv2_energy_transfer_sweep.py  --filename "$FILENAME"
python inv3_plot_transfer_spectrum.py --filename "$FILENAME"
