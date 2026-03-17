#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N kelvin_helmholtz
#PBS -o logs/kelvin_helmholtz.log
#PBS -e logs/kelvin_helmholtz.log
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

time /glade/u/home/tomasc/miniconda3/envs/py313/bin/python -u sfs_ape_budget.py --filename output/kelvin_helmholtz_instability_512x256x512.nc --n-workers 18 2>&1 | tee logs/sfs_ape_budget.out
qstat -f $PBS_JOBID >> logs/sfs_ape_budget.log
qstat -f $PBS_JOBID >> logs/sfs_ape_budget.out

time /glade/u/home/tomasc/miniconda3/envs/py313/bin/python -u sfs_ke_budget.py --filename output/kelvin_helmholtz_instability_512x256x512.nc 2>&1 | tee logs/sfs_ke_budget.out
qstat -f $PBS_JOBID >> logs/sfs_ke_budget.log
qstat -f $PBS_JOBID >> logs/sfs_ke_budget.out
