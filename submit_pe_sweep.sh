#!/usr/bin/env bash
# Pe_cell diagonal sweep, at each of one or more resolutions: for each (resolution, Pe) combination, runs
# a simulation with --Pe_cell_h=--Pe_cell_v=<value> (same value in both directions), then the minimum
# diagnostics needed to get the SFS KE/APE budget residual ratio -- 01_filter_fields -> 02_sort_density ->
# 03_energy_transfer -> 04_sfs_ke_budget -> 05_sfs_ape_budget, no plots/animations (see
# postprocessing/budgeting_notplot.pbs). Each (resolution, Pe) point is entirely its own job chain (own
# simulation -> own filter -> own budgeting), so memory requirements are unchanged from the existing
# budgeting_filter.pbs/budgeting_notplot.pbs defaults regardless of how many points are in the sweep.
#
# Usage: bash submit_pe_sweep.sh [RESOLUTIONS='256x256x64 192x192x64'] [STOP_TIME=15] [PE_VALUES='1 20 40 60']
#   RESOLUTIONS   space-separated NxYxZx triples (e.g. '256x256x64 192x192x64')
#
# After all jobs complete, aggregate results with:
#   python postprocessing/pe_sweep_results.py --resolutions 256x256x64 192x192x64 --pe-values 1 20 40 60 --min-time-days 5
RESOLUTIONS="256x256x64 192x192x64"; STOP_TIME=15; PE_VALUES="1 20 40 60"
for arg in "$@"; do case $arg in
  RESOLUTIONS=*) RESOLUTIONS="${arg#*=}";;
  STOP_TIME=*)   STOP_TIME="${arg#*=}";;
  PE_VALUES=*)   PE_VALUES="${arg#*=}";;
esac; done

for RES in $RESOLUTIONS; do
    NX=$(echo "$RES" | cut -dx -f1)
    NY=$(echo "$RES" | cut -dx -f2)
    NZ=$(echo "$RES" | cut -dx -f3)

    for PE in $PE_VALUES; do
        NAME_SUFFIX="pe${PE}"
        SIM_NAME="bci_Nx${NX}_Ny${NY}_Nz${NZ}_${NAME_SUFFIX}"
        EXTRA_ARGS="--Pe_cell_h $PE --Pe_cell_v $PE"

        SIM_JOB=$(qsub -N "$SIM_NAME" \
                       -o "logs/${SIM_NAME}.log" \
                       -e "logs/${SIM_NAME}.log" \
                       -v "NX=$NX,NY=$NY,NZ=$NZ,STOP_TIME=$STOP_TIME,NAME_SUFFIX=$NAME_SUFFIX,EXTRA_ARGS=$EXTRA_ARGS" \
                       simulation.pbs)
        echo "Submitted simulation (Nx=$NX,Ny=$NY,Nz=$NZ, Pe_cell_h=Pe_cell_v=$PE, $SIM_NAME): $SIM_JOB"

        cd postprocessing

        BF_NAME="${SIM_NAME}_budgeting_filter"
        BF_JOB=$(qsub -N "$BF_NAME" \
                      -o "logs/${BF_NAME}.log" \
                      -e "logs/${BF_NAME}.log" \
                      -v "NX=$NX,NY=$NY,NZ=$NZ,NAME_SUFFIX=$NAME_SUFFIX" \
                      -W depend=afterok:$SIM_JOB \
                      budgeting_filter.pbs)
        echo "  Submitted budgeting filter (depends on $SIM_JOB): $BF_JOB"

        BP_NAME="${SIM_NAME}_budgeting_notplot"
        BP_JOB=$(qsub -N "$BP_NAME" \
                      -o "logs/${BP_NAME}.log" \
                      -e "logs/${BP_NAME}.log" \
                      -v "NX=$NX,NY=$NY,NZ=$NZ,NAME_SUFFIX=$NAME_SUFFIX,FIXED_REF=0" \
                      -W depend=afterok:$BF_JOB \
                      budgeting_notplot.pbs)
        echo "  Submitted budgeting (no plots) (depends on $BF_JOB): $BP_JOB"

        cd ..
    done
done
