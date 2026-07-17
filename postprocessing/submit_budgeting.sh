#!/usr/bin/env bash
# Submit budgeting jobs: shared filter step (01) once, then per-FIXED_REF steps (02-06), then plots (once).
# Usage: bash submit_budgeting.sh [NX=192] [NY=192] [NZ=32] [FIXED_REF=0|1|both] [FILTER_SCALES_M='50000 100000']
#   FIXED_REF=both       submits budget jobs for both 0 and 1 (filter runs only once)
#   FILTER_SCALES_M      offline post-processing filter scales, in meters, passed to budgeting_filter.pbs
#                        (01_filter_fields.py) and plots.pbs (default "50000 100000")
# Plots (plots.pbs) is chained after the FIXED_REF=0 budgeting job specifically -- that's the only variant
# its own scripts (anim3_panels.py etc.) actually read (they have no --fixed-reference support). It's
# submitted automatically whenever FIXED_REF is 0 or both; skipped (with a note) if FIXED_REF=1 only.
NX=192; NY=192; NZ=32; FIXED_REF=0; FILTER_SCALES_M="50000 100000"
for arg in "$@"; do case $arg in
  NX=*)        NX="${arg#*=}";;
  NY=*)        NY="${arg#*=}";;
  NZ=*)        NZ="${arg#*=}";;
  FIXED_REF=*) FIXED_REF="${arg#*=}";;
  FILTER_SCALES_M=*) FILTER_SCALES_M="${arg#*=}";;
esac; done
SIM="bci_Nx${NX}_Ny${NY}_Nz${NZ}"

FILTER_NAME="${SIM}_budgeting_filter"
FILTER_JOB=$(qsub -N "$FILTER_NAME" \
                  -o "logs/${FILTER_NAME}.log" \
                  -e "logs/${FILTER_NAME}.log" \
                  -v "NX=$NX,NY=$NY,NZ=$NZ,FILTER_SCALES_M=$FILTER_SCALES_M" \
                  budgeting_filter.pbs)
echo "Submitted filter job ($SIM): $FILTER_JOB"

PP_JOB_FIXED_REF_0=""

submit_budget() {
    local fr=$1
    [ "$fr" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
    local NAME="${SIM}_budgeting${REF_SUFFIX}"
    local JOB=$(qsub -N "$NAME" \
                     -o "logs/${NAME}.log" \
                     -e "logs/${NAME}.log" \
                     -v NX=$NX,NY=$NY,NZ=$NZ,FIXED_REF=$fr \
                     -W depend=afterok:$FILTER_JOB \
                     budgeting.pbs)
    echo "Submitted budget job FIXED_REF=$fr (depends on $FILTER_JOB): $JOB"
    [ "$fr" = "0" ] && PP_JOB_FIXED_REF_0=$JOB
}

if [ "$FIXED_REF" = "both" ]; then
    submit_budget 0
    submit_budget 1
else
    submit_budget "$FIXED_REF"
fi

if [ -n "$PP_JOB_FIXED_REF_0" ]; then
    PLOTS_NAME="${SIM}_plots"
    PLOTS_JOB=$(qsub -N "$PLOTS_NAME" \
                     -o "logs/${PLOTS_NAME}.log" \
                     -e "logs/${PLOTS_NAME}.log" \
                     -v "NX=$NX,NY=$NY,NZ=$NZ,FILTER_SCALES_M=$FILTER_SCALES_M" \
                     -W depend=afterok:$PP_JOB_FIXED_REF_0 \
                     plots.pbs)
    echo "Submitted plots+animations (depends on $PP_JOB_FIXED_REF_0): $PLOTS_JOB"
else
    echo "Skipping plots: plots.pbs reads the FIXED_REF=0 budgeting output, which wasn't submitted" \
         "(FIXED_REF=$FIXED_REF). Submit plots.pbs manually once that output exists."
fi
