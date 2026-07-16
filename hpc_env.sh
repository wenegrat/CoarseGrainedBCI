#!/bin/bash
# Shared HPC environment setup, sourced by every *.pbs script in this repo.
PYTHON=${PYTHON:-/glade/work/wenegrat/conda-envs/bci/bin/python}

# Clear the environment from any previously loaded modules
module li
module --force purge
module load ncarenv/25.10
module li
