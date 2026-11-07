#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=work
#SBATCH --time=19:30:00
#SBATCH --mem=50GB
#SBATCH --account=pawsey0272
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

REPO_DIR="/scratch/mwasci/giuse/github_pipeline"
CONFIG_FILE="${REPO_DIR}/config/config.yaml"

paramstukey="875"
paramsbriggs="0.0"

#module load singularity/4.1.0-slurm
module load singularity/4.3.5-slurm
#module load cray-python/3.11.7
module load cray-python/3.12.12
CONTAINER=/software/projects/$PAWSEY_PROJECT/$USER/containers/dip.sif

# Read shared paths from the YAML using the Python environment inside the container.
# Requires PyYAML (import yaml) to be available in that container.
config_values=$(singularity exec "$CONTAINER" python - "$CONFIG_FILE" <<'PYCONFIG'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
paths = config["paths"]
for field in ("project_dir", "images_dir", "report_csv", "obsids_file"):
    value = paths[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"paths.{field} must be a nonempty string")
    print(value)
PYCONFIG
) || { echo "Unable to read ${CONFIG_FILE} in the container" >&2; exit 1; }

mapfile -t config_lines <<< "$config_values"
if (( ${#config_lines[@]} != 4 )); then
    echo "Expected four paths from ${CONFIG_FILE}" >&2
    exit 1
fi
projectDir="${config_lines[0]}"
imagesDir="${config_lines[1]}"
reportCsv="${config_lines[2]}"
obsidsFile="${config_lines[3]}"

obsid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$obsidsFile")
[[ -n "$obsid" ]] || { echo "obsid empty for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

# The report module now lives in download/, not bin/. Expose it to Python
# running inside the Singularity container.
export SINGULARITYENV_PYTHONPATH="${REPO_DIR}/download:${REPO_DIR}/bin"

obsdir="${imagesDir}/${obsid}"
workdir="${obsdir}/wsclean_retry_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
state_dir="${obsdir}/.resume_state"


cd "$obsdir" || {
    echo "Failed to enter $obsdir"
    exit 1
}

mkdir -p "$state_dir"

cleanup_wsclean_products() {
    rm -f "${obsid}_deep-"*.fits "${obsid}_deep"*.fits "${obsid}_deep"*.txt \
          "${obsid}_deep"*.log "${obsid}.ms-part"*.tmp || return 1
}

step_done() {
    [[ -f "$state_dir/$1.done" ]]
}

mark_done() {
    touch "$state_dir/$1.done"
}

run_step() {
    local step_key="$1"
    local step_name="$2"
    shift 2
    local rc

    if step_done "$step_key"; then
        echo "Skipping $step_name (already completed earlier)."
        return 0
    fi

    echo "Starting $step_name..."
    "$@"
    rc=$?

    if (( rc != 0 )); then
        echo "$step_name failed with exit code $rc."
        exit "$rc"
    fi

    mark_done "$step_key"
    echo "$step_name completed successfully."
}

source_ms="${obsid}.ms"
copied_ms="${obsid}_sub_work.ms"
partial_copy="${copied_ms}.partial"

copy_measurement_set() {
    [[ -e "$source_ms" ]] || {
        echo "Source measurement set $source_ms does not exist."
        return 1
    }

    if [[ -d "$copied_ms" ]]; then
        echo "Destination $copied_ms already exists and looks valid; treating copy step as complete."
        return 0
    fi

    if [[ -e "$copied_ms" || -L "$copied_ms" ]]; then
        echo "Destination path $copied_ms exists but is not a directory. Please inspect it manually."
        return 1
    fi

    rm -rf "$partial_copy" || return 1

    # -H dereferences the command-line source if ${obsid}.ms is a symlink,
    # so the copy becomes a real directory rather than another symlink.
    cp -aH "$source_ms" "$partial_copy" || return 1
    mv "$partial_copy" "$copied_ms" || return 1

    [[ -d "$copied_ms" ]] || {
        echo "Copied measurement set $copied_ms was not created correctly."
        return 1
    }
}

# If the final step is already marked as done, skip this observation entirely.
if step_done "mini_cube_clean"; then
    echo "Observation $obsid already completed previously. Nothing to do."
    exit 0
fi

# Step 1: Create field template
if ! step_done "field_template"; then
    cleanup_wsclean_products || exit 1
fi
run_step "field_template" "field_template.py" \
    singularity exec "$CONTAINER" python ${REPO_DIR}/bin/field_template.py \
    "$projectDir" "$obsid" "$paramsbriggs" "$paramstukey" "$reportCsv"

# Step 2: Create mask
run_step "make_mask" "make_mask.py" \
    singularity exec "$CONTAINER" python ${REPO_DIR}/bin/make_mask.py "$obsid"

# Step 3: Image using the mask
if ! step_done "image_masking"; then
    cleanup_wsclean_products || exit 1
fi
run_step "image_masking" "image_masking.py" \
    singularity exec "$CONTAINER" python ${REPO_DIR}/bin/image_masking.py \
    "$projectDir" "$obsid" "$paramsbriggs" "$paramstukey" "$reportCsv"

# Step 4: Copy the modified measurement set
if step_done "ms_copy"; then
    echo "Skipping measurement-set copy (already completed earlier)."
elif [[ -d "$copied_ms" ]]; then
    echo "Existing $copied_ms found from a previous run; marking measurement-set copy as complete."
    mark_done "ms_copy"
else
    run_step "ms_copy" "measurement-set copy" copy_measurement_set
fi

# Step 5: Clean the mini cube
if ! step_done "mini_cube_clean"; then
    cleanup_wsclean_products || exit 1
fi
run_step "mini_cube_clean" "mini_cube_clean.py" \
    singularity exec "$CONTAINER" python ${REPO_DIR}/bin/mini_cube_clean.py \
    "$projectDir" "$obsid" "$paramsbriggs" "$paramstukey" "$reportCsv"