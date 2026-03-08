#!/bin/bash
#SBATCH --job-name=g5b-t13g
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@120
#SBATCH --array=0-2
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/g5b-t13g_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/g5b-t13g_%A_%a.err
#
# Gate 5B — Test 13G Phase B: Post-Hoc Pre-Pooling Decoder (3 arms)
#
# Array tasks: 0=a4r, 1=d4a4, 2=d0 (pool-to 188)
# Each arm: ~2h training + eval + generation. 12h is generous.
#
# Prerequisites:
#   - Encoder checkpoints in models/gate5b/{D0,d4a4,a4r}/best_model.pt
#   - PR targets precomputed (data/gate5b_results/pr_targets_*.npz)
#
# Usage:
#   # First: precompute PR targets (run once, no GPU needed)
#   python experiments/bias_control/gate5b/test13g_posthoc_decoder.py --phase precompute
#   # Then:
#   sbatch experiments/bias_control/slurm/gate5b_test13g.sh

set -eo pipefail

# --- Environment ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# --- Parse array index → arm ---
DESCRIPTORS=(a4r d4a4 d0)
POOL_FLAGS=("" "" "--pool-to 188")
DESC=${DESCRIPTORS[$SLURM_ARRAY_TASK_ID]}
POOL=${POOL_FLAGS[$SLURM_ARRAY_TASK_ID]}

echo "=== Gate 5B Test 13G Phase B: Post-Hoc Decoder ==="
echo "Descriptor: ${DESC} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Pool flags: ${POOL:-none}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0

# --- Copy MAESTRO to scratch ---
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH

echo "Copying MAESTRO to scratch (~120GB, ~22 min)..."
rsync -a --info=progress2 $MAESTRO_SRC $SCRATCH/
echo "MAESTRO copied. $(date)"

# --- Verify data ---
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: maestro-v3.0.0.json not found in scratch"
    exit 1
fi

# --- Verify checkpoint ---
CKPT_KEY=$DESC
if [ "$DESC" = "d0" ]; then CKPT_KEY="D0"; fi
if [ ! -f "$REPO/models/gate5b/${CKPT_KEY}/best_model.pt" ]; then
    echo "ERROR: Checkpoint not found: models/gate5b/${CKPT_KEY}/best_model.pt"
    exit 1
fi
echo "Checkpoint OK: models/gate5b/${CKPT_KEY}/best_model.pt"

# --- Verify PR targets ---
if [ ! -f "$REPO/data/gate5b_results/pr_targets_train_8k.npz" ]; then
    echo "ERROR: PR targets not precomputed. Run --phase precompute first."
    exit 1
fi
echo "PR targets OK."

# --- Train ---
echo "Starting training: descriptor=${DESC}, 40 epochs, eval_every=5, patience=4"

cd $REPO
srun python experiments/bias_control/gate5b/test13g_posthoc_decoder.py \
    --descriptor $DESC \
    --phase train \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --device cuda \
    $POOL

TRAIN_EXIT=$?
echo "Training exit code: $TRAIN_EXIT"

# --- Generate samples (if training succeeded) ---
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Running generation phase..."
    srun python experiments/bias_control/gate5b/test13g_posthoc_decoder.py \
        --descriptor $DESC \
        --phase generate \
        --maestro-dir $SCRATCH/maestro-v3.0.0 \
        --device cuda \
        $POOL
    echo "Generation exit code: $?"
fi

echo "=== Done ==="
echo "End: $(date)"

exit $TRAIN_EXIT
