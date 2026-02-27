#!/bin/bash
#SBATCH --job-name=gate5b_t02
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --exclude=ivb03,ivb04,ivb10
#SBATCH --array=0-3
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate5b_t02_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate5b_t02_%A_%a.err
#
# Gate 5B — Test 02: Parameter-Matched Ablations (4 arms)
#
# Trains d4a4-architecture models with ablated descriptors to verify
# that the ~10pp improvement over D0 is causal (from ratio info).
#
# Array tasks: 0=real, 1=random, 2=shuffled, 3=zero
# Each arm: ~16h (30 epochs × ~30 min/ep + eval overhead)
#
# Usage:
#   sbatch experiments/bias_control/slurm/gate5b_param_matched.sh

set -eo pipefail

# --- Environment ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# --- Mode from array index ---
MODES=(real random shuffled zero)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "=== Gate 5B Test 02: Parameter-Matched Ablation ==="
echo "Mode: ${MODE} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=$REPO/results_unc/gate5b_param_matched/${MODE}

# --- Attempt counter (max 3 retries via scontrol requeue) ---
ATTEMPT_FILE=$OUTDIR/.attempt_count
mkdir -p $OUTDIR
if [ -f "$ATTEMPT_FILE" ]; then
    ATTEMPT=$(cat $ATTEMPT_FILE)
    ATTEMPT=$((ATTEMPT + 1))
else
    ATTEMPT=1
fi
echo $ATTEMPT > $ATTEMPT_FILE
echo "Attempt: $ATTEMPT / 3"

# --- Copy MAESTRO to scratch ---
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH

echo "Copying MAESTRO to scratch (~120GB, ~22 min)..."
cp -r $MAESTRO_SRC $SCRATCH/
echo "MAESTRO copied. $(date)"

# --- Verify data ---
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: maestro-v3.0.0.json not found in scratch"
    exit 1
fi

# --- Find latest checkpoint for resume ---
LAST_CKPT=$(ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
RESUME_FLAG=""
if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
fi

# --- Training ---
echo "Starting training: mode=${MODE}, 30 epochs, batch=16, 1000 bat/ep"
echo "Structured eval epochs: 5 10 15 20 25 28 29 30"

srun python $REPO/experiments/bias_control/gate5b/test02_param_matched.py \
    --mode $MODE \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --epochs 30 \
    --batch-size 16 \
    --max-batches-per-epoch 1000 \
    --seed 42 \
    --device cuda \
    --num-workers 8 \
    --structured-eval-epochs 5 10 15 20 25 28 29 30 \
    --stats-batches 50 \
    $RESUME_FLAG

EXIT_CODE=$?

echo "=== Training finished ==="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# --- Verify outputs ---
if [ -f "$OUTDIR/final_results.json" ]; then
    echo "final_results.json found. Extracting S metric:"
    python -c "
import json
r = json.load(open('$OUTDIR/final_results.json'))
best = r.get('evaluation_best', {})
if best:
    s = best.get('gate_metrics', {}).get('S', best.get('structured_S', 'N/A'))
    ep = best.get('epoch', '?')
    print(f'  Best S: {s} (epoch {ep})')
else:
    print('  (no evaluation_best in results)')
print(f'  Mode: ${MODE}')
print(f'  Call counts: {r.get(\"call_counts_final\", {})}')
"
    # Clean up attempt counter on success
    rm -f $ATTEMPT_FILE
else
    echo "Training incomplete (no final_results.json)."
    if [ $ATTEMPT -le 3 ]; then
        echo "Requeuing this task (attempt $ATTEMPT/3)..."
        scontrol requeue ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
    else
        echo "ERROR: Max retries ($ATTEMPT) exceeded for mode=${MODE}. Manual intervention needed."
        exit 1
    fi
fi

exit $EXIT_CODE
