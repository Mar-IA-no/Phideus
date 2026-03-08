#!/bin/bash
#SBATCH --job-name=g8_cond
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-06:00:00
#SBATCH --array=0-2
#SBATCH --signal=B:SIGTERM@595
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate8_cond_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate8_cond_%A_%a.err

# ── Gate 8: Conditioned Projection Heads (3 remaining arms) ──
# Array: 0=pcd-zero, 1=pcd, 2=pca
# Ref: ctrl S=79.2%, pcm S=80.0% (completed on LOCAL)

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
RESULTS_BASE=/home/mfmendez/Repos/Phideus/results_unc/gate8_conditioned_projections

# ── Array decode ──
ARMS=(a4r-pcd-zero a4r-pcd a4r-pca)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
OUTDIR=$RESULTS_BASE/${ARM}_seed42

echo "=== Gate 8 Conditioned Projections ==="
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Arm: $ARM (task $SLURM_ARRAY_TASK_ID of ${#ARMS[@]})"
echo "  Output: $OUTDIR"

# ── Data staging ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo ""
echo "Staging MAESTRO to scratch..."
COPY_START=$(date +%s)
cp -r $MAESTRO_SRC $SCRATCH/maestro-v3.0.0
COPY_END=$(date +%s)
echo "Staging complete in $((COPY_END - COPY_START)) seconds."

# ── Verify staging ──
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: MAESTRO metadata not found in scratch"
    exit 1
fi

# ── Resume support ──
RESUME_FLAG=""
LAST_CKPT=$(ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1)
if [ -n "$LAST_CKPT" ]; then
    echo "  Resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
fi

# ── SIGTERM handler (graceful checkpoint before SLURM kills us) ──
trap 'echo "SIGTERM received at $(date), forwarding..."; kill -TERM $PID; wait $PID' SIGTERM

# ── Training ──
echo ""
echo "Starting training: $ARM"
TRAIN_START=$(date +%s)

srun python $REPO/experiments/bias_control/gate5a_proj_cond.py \
    --arm $ARM \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --epochs 30 --batch-size 16 --num-workers 8 --seed 42 --device cuda \
    --structured-eval-epochs 5 10 15 20 25 28 29 30 \
    $RESUME_FLAG &
PID=$!
wait $PID
EXIT_CODE=$?

TRAIN_END=$(date +%s)
TRAIN_TIME=$(( (TRAIN_END - TRAIN_START) / 60 ))

echo ""
echo "=== COMPLETED ==="
echo "  Arm: $ARM"
echo "  Exit code: $EXIT_CODE"
echo "  Training time: ${TRAIN_TIME} min"
echo "  GPU VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"

# ── Auto-resubmit if incomplete ──
if [ $EXIT_CODE -ne 0 ] && [ -f "$OUTDIR/checkpoint_epoch"*.pt 2>/dev/null ]; then
    echo "Training incomplete, checking for auto-resubmit..."
    if [ -n "$(ls $OUTDIR/checkpoint_epoch*.pt 2>/dev/null)" ]; then
        echo "  Checkpoint found, resubmitting this array task..."
        sbatch --array=$SLURM_ARRAY_TASK_ID $0
    fi
fi

exit $EXIT_CODE
