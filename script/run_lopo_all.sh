#!/bin/bash
#
# Top-level orchestrator for LOPO cross-validation.
#
# Steps:
#   1. Generate fold manifest
#   2. Train shared Word2Vec model
#   3. Submit SLURM array job for all folds
#
# Usage:
#   bash run_lopo_all.sh              # Full run
#   bash run_lopo_all.sh --aggregate  # Aggregate results only (after folds complete)

set -e

VENV=/work/kosei-ho/LLM4SZZ/DeepLineDP/.venv/bin/python
SCRIPT_DIR=/work/kosei-ho/LLM4SZZ/DeepLineDP/script
DATA_DIR=/work/kosei-ho/LLM4SZZ/DeepLineDP/datasets/preprocessed_data
OUTPUT_DIR=/work/kosei-ho/LLM4SZZ/DeepLineDP/output/lopo
EPOCHS=10

# Parse arguments
AGGREGATE_ONLY=false
if [ "$1" = "--aggregate" ]; then
    AGGREGATE_ONLY=true
fi

MANIFEST=${OUTPUT_DIR}/manifest.json
SHARED_W2V_DIR=${OUTPUT_DIR}/shared_w2v
LOG_DIR=${SCRIPT_DIR}/lopo_logs

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ "${AGGREGATE_ONLY}" = true ]; then
    echo "=== Step: Aggregate Results ==="
    ${VENV} ${SCRIPT_DIR}/lopo_aggregate.py \
        --manifest "${MANIFEST}" \
        --output-dir "${OUTPUT_DIR}" \
        --output "${OUTPUT_DIR}/lopo_results.json" \
        --pretty
    echo ""
    echo "Results saved to ${OUTPUT_DIR}/lopo_results.json"
    exit 0
fi

# ---- Step 1: Generate manifest ----
echo "=== Step 1: Generate fold manifest ==="
${VENV} ${SCRIPT_DIR}/lopo_generate_folds.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}"

# Read total folds from manifest
TOTAL_FOLDS=$(${VENV} -c "
import json
with open('${MANIFEST}') as f:
    m = json.load(f)
print(m['summary']['total_folds'])
")
MAX_ARRAY_ID=$((TOTAL_FOLDS - 1))

echo "Total folds: ${TOTAL_FOLDS}"
echo ""

# ---- Step 2: Train shared Word2Vec ----
echo "=== Step 2: Train shared Word2Vec model ==="
SHARED_W2V=$(${VENV} -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from train_pipeline import train_shared_word2vec
w2v_path = train_shared_word2vec(
    ['${DATA_DIR}/agent_data', '${DATA_DIR}/human_data', '${DATA_DIR}/mixed_data'],
    '${SHARED_W2V_DIR}',
)
print(w2v_path)
" | tail -1)

echo "Shared W2V: ${SHARED_W2V}"
echo ""

# ---- Step 3: Submit SLURM array ----
echo "=== Step 3: Submit SLURM array job ==="
echo "Array range: 0-${MAX_ARRAY_ID}"

# Export environment for the array job
export LOPO_MANIFEST="${MANIFEST}"
export LOPO_OUTPUT="${OUTPUT_DIR}"
export LOPO_W2V="${SHARED_W2V}"
export LOPO_EPOCHS="${EPOCHS}"

JOB_ID=$(sbatch \
    --array=0-${MAX_ARRAY_ID} \
    --parsable \
    ${SCRIPT_DIR}/run_lopo_slurm.sh)

echo "Submitted SLURM job array: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Logs in: ${LOG_DIR}/"
echo ""
echo "After all folds complete, run:"
echo "  bash ${SCRIPT_DIR}/run_lopo_all.sh --aggregate"
