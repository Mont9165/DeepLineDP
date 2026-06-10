#!/bin/bash
#SBATCH --job-name=lopo-fold
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=lopo_logs/fold_%a_%j.log
#
# LOPO fold job array for RQ3.
#
# Each array task runs one fold from the LOPO manifest.
# The array range should be set when submitting:
#   sbatch --array=0-N run_lopo_slurm.sh
# where N = total_folds - 1 from the manifest.
#
# Environment variables (set by run_lopo_all.sh or manually):
#   LOPO_MANIFEST  - path to manifest.json
#   LOPO_OUTPUT    - LOPO output directory
#   LOPO_W2V       - path to shared Word2Vec model (optional)
#   LOPO_EPOCHS    - training epochs (default: 10)

set -e

# NOTE: under SLURM, $0 is the spooled copy (/var/spool/slurmd/jobNNN/slurm_script),
# so $0-relative paths break. Honor explicit overrides (set by the launcher), with a
# $0-relative fallback for direct (non-SLURM) invocation.
SCRIPT_DIR="${LOPO_SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
VENV="${LOPO_PYTHON:-${SCRIPT_DIR}/../../.venv/bin/python}"

# Defaults
MANIFEST=${LOPO_MANIFEST:-${SCRIPT_DIR}/../output/lopo/manifest.json}
OUTPUT_DIR=${LOPO_OUTPUT:-${SCRIPT_DIR}/../output/lopo}
W2V_MODEL=${LOPO_W2V:-""}
EPOCHS=${LOPO_EPOCHS:-10}

# Fold id = array task id + offset. The offset lets a chunk of folds with ids
# above SLURM's MaxArraySize (e.g. 2880-3866) be submitted as 0-based array
# indices. Defaults to 0, so direct (offset-less) submission is unchanged.
FOLD_ID=$(( SLURM_ARRAY_TASK_ID + ${LOPO_FOLD_OFFSET:-0} ))

echo "=== LOPO Fold ${FOLD_ID} ==="
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Manifest: ${MANIFEST}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
date

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Build command
CMD="${VENV} ${SCRIPT_DIR}/lopo_run_fold.py \
    --manifest ${MANIFEST} \
    --fold-id ${FOLD_ID} \
    --output-dir ${OUTPUT_DIR} \
    --num-epochs ${EPOCHS}"

if [ -n "${W2V_MODEL}" ]; then
  CMD="${CMD} --w2v-model ${W2V_MODEL}"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo ""
echo "=== Fold ${FOLD_ID} completed ==="
date
