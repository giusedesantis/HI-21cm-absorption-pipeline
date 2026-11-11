#!/bin/bash -l
#SBATCH --job-name=post_imaging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=work
#SBATCH --time=19:30:00
#SBATCH --mem=50GB
#SBATCH --account=pawsey0272
#SBATCH --output=post_imaging_%j.out
#SBATCH --error=post_imaging_%j.err

# Run from the repository root:
#   cd /scratch/mwasci/giuse/github_pipeline
#   sbatch slurm/post_imaging.sbatch
# Optionally set REPO_DIR before sbatch if submitting elsewhere.
set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_DIR="$(cd "$REPO_DIR" && pwd)"
cd "$REPO_DIR"

# Use the same imaging/processing container as imaging_auto.sbatch.
: "${PAWSEY_PROJECT:?PAWSEY_PROJECT must be set on Setonix}"
: "${USER:?USER must be set}"
CONTAINER="/software/projects/${PAWSEY_PROJECT}/${USER}/containers/dip.sif"
CONFIG_FILE="${REPO_DIR}/config/config.yaml"

module load singularity/4.3.5-slurm

[[ -f "$CONTAINER" ]] || { echo "Missing container: $CONTAINER" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "Missing configuration: $CONFIG_FILE" >&2; exit 1; }

# Let scripts in post_imaging/ import config_utils.py in bin/.
# Also expose report.py if a processing script uses it.
export SINGULARITYENV_PYTHONPATH="${REPO_DIR}/bin:${REPO_DIR}/post_imaging:${REPO_DIR}/download"
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_MPLBACKEND=Agg

# IMPORTANT: These commands assume each script runs without required CLI arguments
# and gets its file paths from config/config.yaml. Verify this before a real job.
stages=(
    barycentric_correct.py
    resampling.py
    shift.py
    flux_fitting.py
    stacking.py
    back_resampling.py
    boxcar_filtering.py
    plotting.py
)

# Validate every script exists before running any processing stage.
for script in "${stages[@]}"; do
    [[ -f "${REPO_DIR}/post_imaging/${script}" ]] || {
        echo "Missing script: ${REPO_DIR}/post_imaging/${script}" >&2
        exit 1
    }
done

# Confirm that the configuration can be loaded in the selected container.
singularity exec "$CONTAINER" python -c \
    "from config_utils import load_config; load_config(); print('Configuration loaded successfully')"

echo "Repository: $REPO_DIR"
echo "Container: $CONTAINER"
echo "Started: $(date)"

for i in "${!stages[@]}"; do
    script="${stages[$i]}"
    echo "============================================================"
    echo "Stage $((i + 1))/${#stages[@]}: ${script} — $(date)"
    echo "============================================================"

    # Stop immediately if the script exits with a nonzero status.
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY RUN: would execute ${script}"
    else
        singularity exec "$CONTAINER" python "${REPO_DIR}/post_imaging/${script}"
    fi

    echo "Finished stage: ${script} — $(date)"
done

echo "All post-imaging stages completed — $(date)"
