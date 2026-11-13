#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=work
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=pawsey0272

module load singularity/4.1.0-mpi
module load pytorch/2.7.1-rocm6.3.3

export VENV=/scratch/mwasci/giuse/venvs/torch_mwa_sys_v2
export SINGULARITYENV_PYTHONPATH=$VENV/lib/python3.12/site-packages

singularity exec "$SINGULARITY_CONTAINER" python3 -c "import torch, astropy, pandas, photutils; print('ok')"
singularity exec "$SINGULARITY_CONTAINER" python3 mask_extraction.py