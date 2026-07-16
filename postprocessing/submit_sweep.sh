#!/usr/bin/env bash
# Submit sweep jobs: shared filter step (sweep1) once, then per-FIXED_REF transfer steps (sweep2+sweep3).
# Usage: bash submit_sweep.sh [NX=192] [NY=192] [NZ=32] [FIXED_REF=0|1|both] [N_TIME_SKIP=1]
#   FIXED_REF=both  submits transfer jobs for both 0 and 1 (filter runs only once)
NX=192; NY=192; NZ=32; FIXED_REF=0; N_TIME_SKIP=1
for arg in "$@"; do case $arg in
  NX=*)          NX="${arg#*=}";;
  NY=*)          NY="${arg#*=}";;
  NZ=*)          NZ="${arg#*=}";;
  FIXED_REF=*)   FIXED_REF="${arg#*=}";;
  N_TIME_SKIP=*) N_TIME_SKIP="${arg#*=}";;
esac; done
SIM="bci_Nx${NX}_Ny${NY}_Nz${NZ}"

FILTER_NAME="${SIM}_sweep_filter"
FILTER_JOB=$(qsub -N "$FILTER_NAME" \
                  -o "logs/${FILTER_NAME}.log" \
                  -e "logs/${FILTER_NAME}.log" \
                  -v NX=$NX,NY=$NY,NZ=$NZ,N_TIME_SKIP=$N_TIME_SKIP \
                  sweep_filter.pbs)
echo "Submitted filter job ($SIM): $FILTER_JOB"

submit_transfer() {
    local fr=$1
    [ "$fr" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
    local NAME="${SIM}_sweep_transfer${REF_SUFFIX}"
    local JOB=$(qsub -N "$NAME" \
                     -o "logs/${NAME}.log" \
                     -e "logs/${NAME}.log" \
                     -v NX=$NX,NY=$NY,NZ=$NZ,FIXED_REF=$fr \
                     -W depend=afterok:$FILTER_JOB \
                     sweep_transfer.pbs)
    echo "Submitted transfer job FIXED_REF=$fr (depends on $FILTER_JOB): $JOB"
}

if [ "$FIXED_REF" = "both" ]; then
    submit_transfer 0
    submit_transfer 1
else
    submit_transfer "$FIXED_REF"
fi
