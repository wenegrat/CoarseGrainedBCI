#!/bin/bash
# Submit simulation and post-processing as a chained PBS job pair.
# Post-processing only runs if simulation succeeds (afterok dependency).
#
# Usage: bash submit_all_pbs.sh [Nz]
#   Nz  vertical resolution (default: 4096)
#
# To run post-processing alone:
#   qsub -v NZ=<Nz> postprocessing/submit_budgeting_pbs.sh

NZ=${1:-4096}

SIM_JOB=$(qsub -N kelvin_helmholtz_${NZ} \
               -o logs/kelvin_helmholtz_${NZ}.log \
               -e logs/kelvin_helmholtz_${NZ}.log \
               -v NZ=$NZ submit_simulation_pbs.sh)
echo "Submitted simulation (Nz=$NZ): $SIM_JOB"

POSTPROC_DIR=$(realpath postprocessing)
PP_JOB=$(qsub -N budgeting_Nz${NZ}_Ri0.10 \
              -o ${POSTPROC_DIR}/logs/budgeting_Nz${NZ}_Ri0.10.log \
              -e ${POSTPROC_DIR}/logs/budgeting_Nz${NZ}_Ri0.10.log \
              -v NZ=$NZ,POSTPROC_DIR=$POSTPROC_DIR \
              -W depend=afterok:$SIM_JOB postprocessing/submit_budgeting_pbs.sh)
echo "Submitted post-processing (depends on $SIM_JOB): $PP_JOB"
