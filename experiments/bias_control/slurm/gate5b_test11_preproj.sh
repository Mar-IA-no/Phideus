#!/bin/bash
#SBATCH --job-name=gate5b_t11
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate5b_t11_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate5b_t11_%A_%a.err
#
# Gate 5B — Test 11: Pre-Projection A/B Test (missing arms: d4a4 + d4-a4r)
#
# Diagnoses whether the projection bottleneck limits decoder quality.
# Array tasks: 0=d4a4, 1=d4-a4r
# Each arm: ~3.2h (extraction + 2 decoder trainings + eval + controls)
#
# Usage:
#   sbatch experiments/bias_control/slurm/gate5b_test11_preproj.sh

set -eo pipefail

# --- Environment ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# --- Arm from array index ---
ARMS=(d4a4 d4-a4r)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

echo "=== Gate 5B Test 11: Pre-Projection A/B ==="
echo "Arm: ${ARM} (array task ${SLURM_ARRAY_TASK_ID})"
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
cp -r $MAESTRO_SRC $SCRATCH/
echo "MAESTRO copied. $(date)"

# --- Verify data ---
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: maestro-v3.0.0.json not found in scratch"
    exit 1
fi

# --- Run Test 11 Pre-Proj A/B for this arm ---
echo "Starting Test 11 Pre-Proj A/B: arm=${ARM}"

cd $REPO
srun python experiments/bias_control/gate5b/test11_preproj_ab_test.py \
    --arms $ARM \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --device cuda \
    --num-workers 8 \
    --seed 42

EXIT_CODE=$?

echo "=== Test 11 finished ==="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# --- Verify outputs ---
OUTDIR=$REPO/data/gate5b_results/test11_preproj/${ARM}
if [ -d "$OUTDIR" ]; then
    echo "Output directory found: $OUTDIR"
    ls -la $OUTDIR/
else
    echo "WARNING: Output directory not found at $OUTDIR"
fi

exit $EXIT_CODE
