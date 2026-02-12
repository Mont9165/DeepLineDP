#!/bin/bash
#SBATCH --job-name=deeplinedp-test
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=test_gpu_%j.log

set -e

VENV=/work/kosei-ho/LLM4SZZ/DeepLineDP/.venv/bin/python
SCRIPT_DIR=/work/kosei-ho/LLM4SZZ/DeepLineDP/script
PROJECT=activemq

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Step 1: Train Word2Vec (activemq) ==="
cd $SCRIPT_DIR
$VENV train_word2vec.py $PROJECT
echo "Word2Vec training done."

echo ""
echo "=== Step 2: Train Model (activemq, 2 epochs) ==="
$VENV train_model.py -dataset $PROJECT -num_epochs 2 -batch_size 32
echo "Model training done."

echo ""
echo "=== Step 3: Generate Prediction (activemq) ==="
$VENV generate_prediction.py -dataset $PROJECT -target_epochs 2
echo "Prediction generation done."

echo ""
echo "=== All tests passed! ==="
