#!/bin/bash
#SBATCH --job-name=gate5b_parammatch
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2  # 3 ablation modes: random, shuffled, zero
#SBATCH --output=logs/gate5b_parammatch_%A_%a.log
#
# Gate 5B — Test 2: Parameter-Matched Ablations
#
# Trains d4a4-shaped models with crippled descriptors to control
# that the improvement is causal (from ratio info, not just params).
#
# Usage:
#   sbatch experiments/bias_control/slurm/gate5b_param_matched.sh

set -euo pipefail

MODES=(random shuffled zero)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "=== Gate 5B Param-Matched Ablation: mode=${MODE} ==="

cd /home/mfmendez/Phideus
source venv/bin/activate

OUTPUT_DIR="data/gate5b_results/param_matched/${MODE}"
mkdir -p "${OUTPUT_DIR}"

# This requires a patched training script that applies the ablation mode.
# The patching logic is in test01_causal_ablation.py's ablate_descriptors().
# For training, we need a wrapper that patches BEFORE training starts.

echo "NOTE: This SLURM script requires a training wrapper with ablation support."
echo "Implementation pending — see test02_param_matched.py for details."
echo "Ablation mode: ${MODE}"

# TODO: Implement training with ablated descriptors
# python experiments/bias_control/gate5b/train_param_matched.py \
#     --ablation-mode "${MODE}" \
#     --descriptor d4a4 \
#     --maestro-dir data/maestro_v3/maestro-v3.0.0 \
#     --output "${OUTPUT_DIR}" \
#     --epochs 30 --batch-size 16 --num-workers 8 --seed 42

echo "=== DONE: mode=${MODE} ==="
