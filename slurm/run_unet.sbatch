#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=work
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=pawsey0272

#module load singularity/4.1.0-mpi
module load singularity/4.3.5-mpi
#module load pytorch/2.7.1-rocm6.3.3
module load pytorch/2.13.0-rocm7.2.4

export VENV=/scratch/mwasci/giuse/venvs/torch_mwa_sys_v2
export SINGULARITYENV_PYTHONPATH=$VENV/lib/python3.12/site-packages

singularity exec "$SINGULARITY_CONTAINER" python3 -c "import torch, astropy, pandas, photutils; print('ok')"
# singularity exec "$SINGULARITY_CONTAINER" python3 unet_heatmap.py make-labels \
#     --root /scratch/mwasci/giuse/dip/MLtest/train_images \
#     --out /scratch/mwasci/giuse/dip/MLtest/labels \
#     --png /scratch/mwasci/giuse/dip/MLtest/label_review \
#     --cutout-size 48 \
#     --target "08h56m14.73s +02d23m59.6s"


singularity exec "$SINGULARITY_CONTAINER" python3 unet_heatmap.py train \
    --labels /scratch/mwasci/giuse/dip/MLtest/labels \
    --model-out /scratch/mwasci/giuse/dip/MLtest/models/source_unet.pt \
    --cutout-size 48 \
    --epochs 100 \
    --batch-size 32 \
    --patience 15    