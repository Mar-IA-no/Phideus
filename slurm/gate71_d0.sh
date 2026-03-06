#!/bin/bash
#SBATCH --job-name=g71a_d0
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/gate71_d0_%j.out
#SBATCH --error=slurm_logs/gate71_d0_%j.err

# Gate 7.1a: MERT-330M Frozen Cross-Modal Probe (D0 Pilot)
# Expected runtime: ~24-36h for 30 epochs (1000 batches/epoch, batch_size=8)

set -euo pipefail

echo "=== Gate 7.1a D0 Pilot ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo ""

# Environment
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate

# Paths
MAESTRO_DIR="${SCRATCH}/maestro-v3.0.0"
OUTDIR="data/gate71_results/d0_mert330m_seed42"

# Verify MAESTRO exists
if [ ! -f "${MAESTRO_DIR}/maestro-v3.0.0.json" ]; then
    echo "ERROR: MAESTRO not found at ${MAESTRO_DIR}"
    exit 1
fi

# Create slurm_logs if needed
mkdir -p slurm_logs

# Run training
srun python experiments/bias_control/gate71/train_gate71.py \
    --descriptor d0 \
    --epochs 30 \
    --batch-size 8 \
    --max-batches-per-epoch 1000 \
    --seed 42 \
    --num-workers 8 \
    --lr-midi 5e-5 \
    --lr-proj 1e-4 \
    --warmup-steps 500 \
    --maestro-dir "${MAESTRO_DIR}" \
    --output "${OUTDIR}" \
    --structured-eval-epochs 5 10 15 20 25 28 29 30 \
    --embed-batch-size 16

echo ""
echo "=== Training complete ==="
echo "Output: ${OUTDIR}"
echo "Date: $(date)"
