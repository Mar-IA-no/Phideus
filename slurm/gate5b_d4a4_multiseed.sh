#!/bin/bash
#SBATCH --job-name=d4a4-ms
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --array=0-3
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/d4a4-ms_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/d4a4-ms_%A_%a.err
#
# Gate 5B — d4a4 Training Multi-Seed (4 new seeds)
# seed42 already exists. This adds seeds 123, 456, 789, 1337.
# Config identical to gate5b_multiseed D0/a4r/d4-a4r runs.
#
# Time estimate: d4a4 ~35 min/ep × 30 = ~17.5h + evals ~= 20h (fits in 48h)

set -eo pipefail

# --- Environment ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# --- Array task → seed ---
SEEDS=(123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Gate 5B d4a4 Multi-Seed: seed=${SEED} ==="
echo "  Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Date: $(date)"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/gate5b_multiseed/d4a4_seed${SEED}

# --- Data staging ---
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH $OUTDIR

echo "Staging MAESTRO to scratch..."
COPY_START=$(date +%s)
rsync -a --info=progress2 $MAESTRO_SRC/ $SCRATCH/maestro-v3.0.0/
COPY_END=$(date +%s)
echo "Staging complete in $((COPY_END - COPY_START)) seconds."

# --- Verify staging ---
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: MAESTRO metadata not found in scratch"
    exit 1
fi

# --- Resume support ---
RESUME_FLAG=""
LAST_CKPT=$(ls -t "$OUTDIR"/checkpoint_epoch*.pt 2>/dev/null | grep -v '_archive_' | head -1 || true)
if [ -n "$LAST_CKPT" ]; then
    echo "  Resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
fi

# --- SIGTERM handler ---
trap 'echo "SIGTERM received at $(date), forwarding..."; kill -TERM $PID; wait $PID' SIGTERM

# --- Training ---
echo ""
echo "Starting training: d4a4 seed=${SEED} 30ep from scratch"

srun python $REPO/experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --mode train \
    --descriptor d4a4 \
    --from-scratch \
    --output $OUTDIR \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --epochs 30 \
    --batch-size 16 \
    --freeze-policy run-d \
    --num-workers 8 \
    --embed-batch-size 16 \
    --max-batches-per-epoch 1000 \
    --max-val-batches 846 \
    --seed "${SEED}" \
    --device cuda \
    --structured-eval-epochs 25 26 27 28 29 30 \
    --gate 4.3-scratch \
    $RESUME_FLAG &
PID=$!
wait $PID
EXIT_CODE=$?

echo ""
echo "=== Training complete ==="
echo "  Descriptor: d4a4, seed: ${SEED}"
echo "  Exit code: $EXIT_CODE"
echo "  Date: $(date)"

# --- Quick result ---
if [ -f "$OUTDIR/final_results.json" ]; then
    python -c "
import json
r = json.load(open('$OUTDIR/final_results.json'))
eb = r['evaluation_best']
s = eb.get('gate_metrics', {}).get('S', '?')
ep = eb.get('epoch', '?')
print(f'  Best S: {s:.1%} (epoch {ep})' if isinstance(s, float) else f'  Best S: {s} (epoch {ep})')
"
else
    echo "Training incomplete (no final_results.json)."
fi

# --- Auto-resubmit if incomplete ---
if [ $EXIT_CODE -ne 0 ]; then
    CKPT_COUNT=$(ls "$OUTDIR"/checkpoint_epoch*.pt 2>/dev/null | grep -v '_archive_' | wc -l || true)
    if [ "$CKPT_COUNT" -gt 0 ]; then
        echo "Training incomplete with checkpoint, resubmitting task $SLURM_ARRAY_TASK_ID..."
        sbatch --array=$SLURM_ARRAY_TASK_ID $0
    fi
fi

exit $EXIT_CODE
