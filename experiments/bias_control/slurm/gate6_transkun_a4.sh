#!/bin/bash
#SBATCH --job-name=g6expA
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --array=0-14
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate6_expA_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate6_expA_%A_%a.err

# ── Gate 6 Exp A: Transkun + A4 Fine-tuning ──
# Array: 5 configs × 3 seeds = 15 jobs
# Configs: baseline, finetune-noA4, A4-event, A4-adapter, adapter-noA4
# Seeds: 42, 123, 456
# 50k iters @ 4.9s/iter = ~68h → needs checkpoint+resubmit (max 48h/slot)

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
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0

# ── Array task decoding: 5 configs × 3 seeds ──
CONFIGS=(baseline finetune-noA4 A4-event A4-adapter adapter-noA4)
SEEDS=(42 123 456)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

CONFIG=${CONFIGS[$CONFIG_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUTDIR=$REPO/data/gate6_results/transkun_a4/${CONFIG}_seed${SEED}
mkdir -p $OUTDIR

echo "=== Gate 6 Exp A: ${CONFIG} (seed=${SEED}) ==="
echo "  Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Output: $OUTDIR"

# ── Data staging ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo ""
echo "Staging MAESTRO to scratch..."
COPY_START=$(date +%s)
rsync -a --info=progress2 $MAESTRO_SRC/ $SCRATCH/maestro-v3.0.0/
COPY_END=$(date +%s)
echo "Staging complete in $((COPY_END - COPY_START)) seconds."

# ── Verify staging ──
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: MAESTRO metadata not found in scratch"
    exit 1
fi

# ── Resume support ──
RESUME_FLAG=""
if [ -f "$OUTDIR/checkpoint.pt" ]; then
    echo "  Resuming from: $OUTDIR/checkpoint.pt"
    RESUME_FLAG="--resume $OUTDIR/checkpoint.pt"
fi

# ── SIGTERM handler ──
trap 'echo "SIGTERM received at $(date), forwarding..."; kill -TERM $PID; wait $PID' SIGTERM

# ── Training ──
echo ""
echo "Starting training: ${CONFIG} seed=${SEED}"
TRAIN_START=$(date +%s)

srun python $REPO/experiments/bias_control/gate6/transkun_a4_finetune.py \
    --config "${CONFIG}" \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --iterations 50000 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed ${SEED} \
    --eval-every 5000 \
    --device cuda \
    $RESUME_FLAG &
PID=$!
wait $PID
EXIT_CODE=$?

TRAIN_END=$(date +%s)
TRAIN_TIME=$(( (TRAIN_END - TRAIN_START) / 60 ))

echo ""
echo "=== COMPLETED ==="
echo "  Config: ${CONFIG} seed=${SEED}"
echo "  Exit code: $EXIT_CODE"
echo "  Training time: ${TRAIN_TIME} min"

# ── Quick result ──
if [ -f "$OUTDIR/training_results.json" ]; then
    python -c "
import json
r = json.load(open('$OUTDIR/training_results.json'))
print(f'Best F1: {r[\"best_f1\"]:.4f} @ iter {r[\"best_iter\"]}')
print(f'Time: {r[\"total_time_minutes\"]:.1f} min')
"
fi

# ── Auto-resubmit if incomplete ──
if [ $EXIT_CODE -ne 0 ]; then
    if [ -f "$OUTDIR/checkpoint.pt" ]; then
        echo "Training incomplete, resubmitting task $SLURM_ARRAY_TASK_ID..."
        sbatch --array=$SLURM_ARRAY_TASK_ID $0
    else
        echo "Training failed with no checkpoint — not resubmitting."
    fi
fi

exit $EXIT_CODE
