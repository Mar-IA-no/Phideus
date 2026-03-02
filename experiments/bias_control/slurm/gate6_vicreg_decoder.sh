#!/bin/bash
#SBATCH --job-name=g6expC
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --array=0-3
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate6_expC_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate6_expC_%A_%a.err

# ── Gate 6 Exp C: AMT Decoder over VICReg Features ──
# Array: 4 arms (D0, d4a4, a4r, d4-a4r)
# ETA: ~4h/arm → ~4h total with 4 GPUs parallel

set -eo pipefail

# ── Environment ──
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ── Paths ──
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=/home/mfmendez/data/maestro_v3/maestro-v3.0.0

# ── Array task decoding ──
DESCRIPTORS=(d0 d4a4 a4r d4-a4r)
DESCRIPTOR=${DESCRIPTORS[$SLURM_ARRAY_TASK_ID]}
SEED=42

OUTDIR=$REPO/data/gate6_results/vicreg_decoder/${DESCRIPTOR}_seed${SEED}
mkdir -p $OUTDIR

echo "=== Gate 6 Exp C: ${DESCRIPTOR} (seed=${SEED}) ==="
echo "  Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Output: $OUTDIR"

# ── Data staging ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo "Staging MAESTRO to scratch..."
cp -r $MAESTRO_SRC $SCRATCH/maestro-v3.0.0
echo "Staging complete."

# ── Resume support ──
LAST_CKPT=$(ls -t $OUTDIR/checkpoints/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
RESUME_FLAG=""
if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
fi

# ── Run ──
srun python $REPO/experiments/bias_control/gate6/vicreg_amt_decoder.py \
    --descriptor "${DESCRIPTOR}" \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --epochs 80 \
    --batch-size 16 \
    --eval-every 5 \
    --checkpoint-every 1 \
    --seed ${SEED} \
    --device cuda \
    $RESUME_FLAG

echo "=== DONE: ${DESCRIPTOR} ==="

# ── Quick result ──
python -c "
import json
r = json.load(open('$OUTDIR/training_results.json'))
print(f'Best F1: {r[\"best_f1\"]:.4f} @ epoch {r[\"best_epoch\"]}')
print(f'Time: {r[\"total_time_minutes\"]:.1f} min')
"
