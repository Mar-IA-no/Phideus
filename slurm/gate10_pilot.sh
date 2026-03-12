#!/bin/bash
#SBATCH --job-name=g10_pilot
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/gate10_%A_%a.out
#SBATCH --error=slurm_logs/gate10_%A_%a.err
#SBATCH --array=0-8

# Gate 10: Mechanism Sweep — 3 descriptors × 3 mechanisms = 9 runs
# All same protocol: 30ep, from-scratch, run-d, seed 42
# Expected runtime: ~4-5h per run on A30

set -euo pipefail

# Define the 9 arms
DESCRIPTORS=(a7 a10a a10d a7-pca a10a-pca a10d-pca a7-ab a10a-ab a10d-ab)
BATCH_SIZES=(16 16 16 16 16 16 8 8 8)

DESC=${DESCRIPTORS[$SLURM_ARRAY_TASK_ID]}
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "=== Gate 10: Mechanism Sweep ==="
echo "Job ID: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Descriptor: ${DESC}"
echo "Batch size: ${BS}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo ""

# Environment
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate

# Paths
MAESTRO_DIR="${SCRATCH}/maestro-v3.0.0"
OUTDIR="data/gate10_results/${DESC}_seed42"

# Verify MAESTRO exists
if [ ! -f "${MAESTRO_DIR}/maestro-v3.0.0.json" ]; then
    echo "ERROR: MAESTRO not found at ${MAESTRO_DIR}"
    exit 1
fi

# Create dirs
mkdir -p slurm_logs
mkdir -p "$(dirname "${OUTDIR}")"

COMMON_ARGS="--from-scratch --freeze-policy run-d --gate 10 --epochs 30 \
    --seed 42 --max-batches-per-epoch 1000 --max-val-batches 846 \
    --structured-eval-epochs 5 10 15 20 25 28 29 30 \
    --num-workers 8 --embed-batch-size 16 \
    --maestro-dir ${MAESTRO_DIR}"

# Run training
srun python experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --descriptor "${DESC}" \
    --batch-size "${BS}" \
    --output "${OUTDIR}" \
    ${COMMON_ARGS}

echo ""
echo "=== Training complete ==="
echo "Descriptor: ${DESC}"
echo "Output: ${OUTDIR}"
echo "Date: $(date)"
