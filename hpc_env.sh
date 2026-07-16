#!/bin/bash
# Shared HPC environment setup, sourced by every *.pbs script in this repo.
#
# --- Fill in PYTHON below before submitting any job --- it must have the packages in
# postprocessing/tests/requirements.txt installed (a conda env or venv on this cluster; the plain
# `.venv` used for local macOS development in the README won't exist on the HPC filesystem).
PYTHON=${PYTHON:-/path/to/your/conda_or_venv/bin/python}

# Clear the environment from any previously loaded modules
module li
module --force purge
module load ncarenv/25.10
module li
