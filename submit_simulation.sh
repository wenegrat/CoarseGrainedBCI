#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NX=192] [NY=192] [NZ=32] [STOP_TIME=16] [EXTRA_ARGS='--progress_interval 1'] [GPU=1]
#   NX/NY/NZ    grid resolution
#   STOP_TIME   simulation length, in days
#   EXTRA_ARGS  extra baroclinic_adjustment.jl CLI args, passed through verbatim (quote multi-word values)
#   GPU         1 requests an A100 (ngpus=1:gpu_type=a100) instead of simulation.pbs's default CPU-only
#               (ngpus=0) resource request, overridden on the qsub command line since #PBS directives are
#               static. baroclinic_adjustment.jl auto-detects the GPU (--architecture=auto, the default) --
#               no EXTRA_ARGS needed, though --architecture=gpu can be added to fail loudly instead of
#               silently falling back to CPU if the GPU somehow isn't functional.
NX=192; NY=192; NZ=32; STOP_TIME=16; EXTRA_ARGS=""; GPU=0
for arg in "$@"; do case $arg in
  NX=*)         NX="${arg#*=}";;
  NY=*)         NY="${arg#*=}";;
  NZ=*)         NZ="${arg#*=}";;
  STOP_TIME=*)  STOP_TIME="${arg#*=}";;
  EXTRA_ARGS=*) EXTRA_ARGS="${arg#*=}";;
  GPU=*)        GPU="${arg#*=}";;
esac; done
NAME="bci_Nx${NX}_Ny${NY}_Nz${NZ}"
GPU_FLAGS=()
[ "$GPU" = "1" ] && GPU_FLAGS=(-l select=1:ncpus=8:ngpus=1:gpu_type=a100:mem=64GB)
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     "${GPU_FLAGS[@]}" \
     -v "NX=$NX,NY=$NY,NZ=$NZ,STOP_TIME=$STOP_TIME,EXTRA_ARGS=$EXTRA_ARGS" \
     simulation.pbs
echo "Submitted simulation ($NAME, stop_time=${STOP_TIME}d${GPU:+, GPU=$GPU}): $NAME"
