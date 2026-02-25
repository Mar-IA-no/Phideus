#!/bin/bash
#SBATCH --job-name=gate5b_multiseed
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-14  # 5 seeds × 3 descriptors = 15 jobs
#SBATCH --output=logs/gate5b_multiseed_%A_%a.log
#
# Gate 5B — Test 5: Multi-Seed Replication
#
# Trains D0, a4r, d4-a4r with 5 different seeds (30 epochs each).
# d4a4 already has 5-seed results (84.1% ± 2.3pp) from Gate 4.5.
#
# Usage:
#   sbatch experiments/bias_control/slurm/gate5b_multiseed.sh

set -euo pipefail

# Seeds and descriptors
SEEDS=(42 123 456 789 1337)
DESCRIPTORS=(d0 a4r d4-a4r)

# Parse array index → seed + descriptor
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
DESC_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_IDX]}
DESCRIPTOR=${DESCRIPTORS[$DESC_IDX]}

echo "=== Gate 5B Multi-Seed: descriptor=${DESCRIPTOR}, seed=${SEED} ==="
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

cd /home/mfmendez/Phideus
source venv/bin/activate

OUTPUT_DIR="data/gate5b_results/multiseed/${DESCRIPTOR}_seed${SEED}"
mkdir -p "${OUTPUT_DIR}"

python experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --mode train \
    --descriptor "${DESCRIPTOR}" \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output "${OUTPUT_DIR}" \
    --epochs 30 \
    --batch-size 16 \
    --num-workers 8 \
    --seed "${SEED}" \
    --from-scratch \
    --freeze-policy run-d \
    --lr 0.0001 \
    --lr-warmup-epochs 2 \
    --max-batches-per-epoch 1000 \
    --checkpoint-every 1 \
    --structured-eval-epochs 25 26 27 28 29 30

echo "=== DONE: ${DESCRIPTOR} seed=${SEED} ==="
