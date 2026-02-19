#!/bin/bash
#SBATCH --job-name=lopo-fold
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/kosei-ho/LLM4SZZ/DeepLineDP/script/lopo_logs/fold_%a_%j.log
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

VENV=/work/kosei-ho/LLM4SZZ/DeepLineDP/.venv/bin/python
SCRIPT_DIR=/work/kosei-ho/LLM4SZZ/DeepLineDP/script

# Defaults
MANIFEST=${LOPO_MANIFEST:-${SCRIPT_DIR}/../output/lopo/manifest.json}
OUTPUT_DIR=${LOPO_OUTPUT:-${SCRIPT_DIR}/../output/lopo}
W2V_MODEL=${LOPO_W2V:-""}
EPOCHS=${LOPO_EPOCHS:-10}

FOLD_ID=${SLURM_ARRAY_TASK_ID}

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
