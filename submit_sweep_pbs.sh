#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N sweep_2916x1x4096
#PBS -o logs/sweep_2916x1x4096.log
#PBS -e logs/sweep_2916x1x4096.log
#PBS -l walltime=23:59:00
#PBS -q casper
#PBS -M tchor@umd.edu
#PBS -m ae
#PBS -r n
#PBS -l select=1:ncpus=18:mem=1400GB:ngpus=0
#PBS -l job_priority=premium

# Clear the environment from any previously loaded modules
module li
module --force purge
module load ncarenv/25.10
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

echo $CUDA_VISIBLE_DEVICES

export JULIA_DEPOT_PATH="$WORK/.julia"
export JULIA_CPU_TARGET="generic"
juliaup default 1.12

time /glade/u/home/tomasc/miniconda3/envs/py313/bin/python -u postprocessing/inv1_filter_fields_sweep.py --filename output/khi_2916x1x4096.nc 2>&1 | tee logs/inv1_filter_fields_sweep_2916x1x4096.out
qstat -f $PBS_JOBID >> logs/inv1_filter_fields_sweep_2916x1x4096.log
qstat -f $PBS_JOBID >> logs/inv1_filter_fields_sweep_2916x1x4096.out

time /glade/u/home/tomasc/miniconda3/envs/py313/bin/python -u postprocessing/inv2_energy_transfer_sweep.py --filename output/khi_2916x1x4096.nc --n-workers 18 2>&1 | tee logs/inv2_energy_transfer_sweep_2916x1x4096.out
qstat -f $PBS_JOBID >> logs/inv2_energy_transfer_sweep_2916x1x4096.log
qstat -f $PBS_JOBID >> logs/inv2_energy_transfer_sweep_2916x1x4096.out

