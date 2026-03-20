#!/bin/bash
#SBATCH --job-name=g10sweep
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate10_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate10_%A_%a.err
#SBATCH --array=0-8

# Gate 10: Mechanism Sweep — 3 descriptors × 3 mechanisms = 9 runs
# Descriptors: a7(12d), a10a(12d), a10d(32d)
# Mechanisms: concat, pca (FiLM projection), attn_bias (attention bias)
# Protocol: 30ep from-scratch run-d seed42, ~4-5h/run on A30

set -eo pipefail

# ── Environment ──
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0

# ── Task mapping ──
DESCRIPTORS=(a7 a10a a10d a7-pca a10a-pca a10d-pca a7-ab a10a-ab a10d-ab)
BATCH_SIZES=(16 16 16 16 16 16 8 8 8)

DESC=${DESCRIPTORS[$SLURM_ARRAY_TASK_ID]}
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
OUTDIR=$REPO/data/gate10_results/${DESC}_seed42

echo "=== Gate 10: Mechanism Sweep ==="
echo "  Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "  Descriptor: ${DESC}, Batch size: ${BS}"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Output: $OUTDIR"
echo "  Date: $(date)"
echo ""

# ── Data staging ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH $OUTDIR

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
LAST_CKPT=$(ls -t "$OUTDIR"/checkpoint_epoch*.pt 2>/dev/null | grep -v '_archive_' | head -1 || true)
if [ -n "$LAST_CKPT" ]; then
    echo "  Resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
fi

# ── SIGTERM handler ──
trap 'echo "SIGTERM received at $(date), forwarding..."; kill -TERM $PID; wait $PID' SIGTERM

# ── Training ──
echo ""
echo "Starting training: ${DESC} (batch_size=${BS})"

srun python $REPO/experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --descriptor "${DESC}" \
    --batch-size "${BS}" \
    --output "${OUTDIR}" \
    --from-scratch --freeze-policy run-d --gate 10 --epochs 30 \
    --seed 42 --max-batches-per-epoch 1000 --max-val-batches 846 \
    --structured-eval-epochs 5 10 15 20 25 28 29 30 \
    --num-workers 8 --embed-batch-size 16 \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    $RESUME_FLAG &
PID=$!
wait $PID
EXIT_CODE=$?

echo ""
echo "=== Training complete ==="
echo "  Descriptor: ${DESC}"
echo "  Exit code: $EXIT_CODE"
echo "  Output: ${OUTDIR}"
echo "  Date: $(date)"

# ── Job metrics ──
echo ""
echo "Job Metrics:"
sacct -j $SLURM_JOB_ID --format=JobID,Cluster,User,Group,State,ExitCode,Nodes,NCPUS,CPUTime,Elapsed,MaxRSS,MaxVMSize -P 2>/dev/null || true

# ── Auto-resubmit if incomplete ──
if [ $EXIT_CODE -ne 0 ]; then
    CKPT_COUNT=$(ls "$OUTDIR"/checkpoint_epoch*.pt 2>/dev/null | grep -v '_archive_' | wc -l || true)
    if [ "$CKPT_COUNT" -gt 0 ]; then
        echo "Training incomplete with checkpoint, resubmitting task $SLURM_ARRAY_TASK_ID..."
        sbatch --array=$SLURM_ARRAY_TASK_ID $0
    fi
fi

exit $EXIT_CODE
