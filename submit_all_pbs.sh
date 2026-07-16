#!/bin/bash
# Submit the simulation and full post-processing/plotting pipeline as chained PBS jobs (afterok
# dependencies); each stage only runs if the previous one succeeds.
#
#   simulation → budgeting_filter → budgeting → plots   (always: budgets, plot3/5/6, anim2/3)
#   + sweep_filter → sweep_transfer  (many-filter-scale transfer spectrum; parallel after budgeting)  (SWEEP=1)
#
# Usage: bash submit_all_pbs.sh [NX=192] [NY=192] [NZ=32] [STOP_TIME=16] [FIXED_REF=0] [SWEEP=0]
#   NX/NY/NZ    grid resolution
#   STOP_TIME   simulation length, in days
#   FIXED_REF   use fixed-in-time reference profile: 0 or 1
#   SWEEP       also run the many-filter-scale sweep (sweep1/2/3) after budgeting: 0 or 1
#
# To run post-processing alone (simulation already done):
#   bash postprocessing/submit_budgeting.sh [NX=192] [NY=192] [NZ=32] [FIXED_REF=0|1|both]

NX=192; NY=192; NZ=32; STOP_TIME=16; FIXED_REF=0; SWEEP=0
for arg in "$@"; do case $arg in
  NX=*)        NX="${arg#*=}";;
  NY=*)        NY="${arg#*=}";;
  NZ=*)        NZ="${arg#*=}";;
  STOP_TIME=*) STOP_TIME="${arg#*=}";;
  FIXED_REF=*) FIXED_REF="${arg#*=}";;
  SWEEP=*)     SWEEP="${arg#*=}";;
esac; done
[ "$FIXED_REF" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
SIM_NAME="bci_Nx${NX}_Ny${NY}_Nz${NZ}"

SIM_JOB=$(qsub -N "$SIM_NAME" \
               -o "logs/${SIM_NAME}.log" \
               -e "logs/${SIM_NAME}.log" \
               -v NX=$NX,NY=$NY,NZ=$NZ,STOP_TIME=$STOP_TIME \
               simulation.pbs)
echo "Submitted simulation ($SIM_NAME, stop_time=${STOP_TIME}d): $SIM_JOB"

cd postprocessing

BF_NAME="${SIM_NAME}_budgeting_filter"
BF_JOB=$(qsub -N "$BF_NAME" \
              -o "logs/${BF_NAME}.log" \
              -e "logs/${BF_NAME}.log" \
              -v NX=$NX,NY=$NY,NZ=$NZ \
              -W depend=afterok:$SIM_JOB \
              budgeting_filter.pbs)
echo "Submitted budgeting filter (depends on $SIM_JOB): $BF_JOB"

PP_NAME="${SIM_NAME}_budgeting${REF_SUFFIX}"
PP_JOB=$(qsub -N "$PP_NAME" \
              -o "logs/${PP_NAME}.log" \
              -e "logs/${PP_NAME}.log" \
              -v NX=$NX,NY=$NY,NZ=$NZ,FIXED_REF=$FIXED_REF \
              -W depend=afterok:$BF_JOB \
              budgeting.pbs)
echo "Submitted budgeting (depends on $BF_JOB): $PP_JOB"

PLOTS_NAME="${SIM_NAME}_plots"
PLOTS_JOB=$(qsub -N "$PLOTS_NAME" \
                 -o "logs/${PLOTS_NAME}.log" \
                 -e "logs/${PLOTS_NAME}.log" \
                 -v NX=$NX,NY=$NY,NZ=$NZ \
                 -W depend=afterok:$PP_JOB \
                 plots.pbs)
echo "Submitted plots+animations (depends on $PP_JOB): $PLOTS_JOB"

# Optional many-filter-scale sweep — parallel branch after budgeting, since sweep2 redoes its own
# Winters sort per scale rather than reusing budgeting's (see CLAUDE.md)
if [ "$SWEEP" = "1" ]; then
    SF_NAME="${SIM_NAME}_sweep_filter"
    SF_JOB=$(qsub -N "$SF_NAME" \
                  -o "logs/${SF_NAME}.log" \
                  -e "logs/${SF_NAME}.log" \
                  -v NX=$NX,NY=$NY,NZ=$NZ \
                  -W depend=afterok:$PP_JOB \
                  sweep_filter.pbs)
    echo "Submitted sweep filter (depends on $PP_JOB): $SF_JOB"

    ST_NAME="${SIM_NAME}_sweep_transfer${REF_SUFFIX}"
    ST_JOB=$(qsub -N "$ST_NAME" \
                  -o "logs/${ST_NAME}.log" \
                  -e "logs/${ST_NAME}.log" \
                  -v NX=$NX,NY=$NY,NZ=$NZ,FIXED_REF=$FIXED_REF \
                  -W depend=afterok:$SF_JOB \
                  sweep_transfer.pbs)
    echo "Submitted sweep transfer (depends on $SF_JOB): $ST_JOB"
fi
cd ..
