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

SIM_JOB=$(qsub -v NZ=$NZ submit_simulation_pbs.sh)
echo "Submitted simulation (Nz=$NZ): $SIM_JOB"

PP_JOB=$(qsub -v NZ=$NZ -W depend=afterok:$SIM_JOB postprocessing/submit_budgeting_pbs.sh)
echo "Submitted post-processing (depends on $SIM_JOB): $PP_JOB"
