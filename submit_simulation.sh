#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NX=192] [NY=192] [NZ=32] [STOP_TIME=16] [EXTRA_ARGS='--progress_interval 1']
#   NX/NY/NZ    grid resolution
#   STOP_TIME   simulation length, in days
#   EXTRA_ARGS  extra baroclinic_adjustment.jl CLI args, passed through verbatim (quote multi-word values)
NX=192; NY=192; NZ=32; STOP_TIME=16; EXTRA_ARGS=""
for arg in "$@"; do case $arg in
  NX=*)         NX="${arg#*=}";;
  NY=*)         NY="${arg#*=}";;
  NZ=*)         NZ="${arg#*=}";;
  STOP_TIME=*)  STOP_TIME="${arg#*=}";;
  EXTRA_ARGS=*) EXTRA_ARGS="${arg#*=}";;
esac; done
NAME="bci_Nx${NX}_Ny${NY}_Nz${NZ}"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v "NX=$NX,NY=$NY,NZ=$NZ,STOP_TIME=$STOP_TIME,EXTRA_ARGS=$EXTRA_ARGS" \
     simulation.pbs
echo "Submitted simulation ($NAME, stop_time=${STOP_TIME}d): $NAME"
