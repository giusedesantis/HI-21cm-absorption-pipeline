#!/bin/bash -l
set -euo pipefail

# Repository directory
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Singularity container
CONTAINER="/software/projects/$PAWSEY_PROJECT/$USER/containers/mantaray.sif"

#module load singularity/4.1.0-slurm
module load singularity/4.3.5-slurm

# Make Python modules accessible inside Singularity
export SINGULARITYENV_PYTHONPATH="${REPO_DIR}/download:${REPO_DIR}/bin"

# Run from repository root
cd "$REPO_DIR"

# Download observations (default maximum: 120)
numberObs="${1:-120}"

singularity exec "$CONTAINER" python "${REPO_DIR}/download/manageReport.py" download "$numberObs"
