#!/bin/bash
#SBATCH --job-name=gate5b_t13g
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate5b_t13g_d4a4r_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate5b_t13g_d4a4r_%j.err
#
# Gate 5B — Test 13G-B: Post-Hoc Pre-Pooling Decoder (d4-a4r arm)
#
# Trains a PostHocPRDecoder on frozen encoder pre-pooling features.
# Phases: precompute targets → train decoder → generate → fullval
# Estimated: ~4h total
#
# Usage:
#   sbatch experiments/bias_control/slurm/gate5b_test13g_d4a4r.sh

set -eo pipefail

# --- Environment ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Gate 5B Test 13G-B: Post-Hoc Decoder (d4-a4r) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
SCRIPT=$REPO/experiments/bias_control/gate5b/test13g_posthoc_decoder.py

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

cd $REPO

# --- Common flags ---
COMMON="--descriptor d4-a4r --maestro-dir $SCRATCH/maestro-v3.0.0 --device cuda"

# --- Phase 1: Precompute PR targets ---
echo "--- Phase: precompute targets ---"
srun python $SCRIPT $COMMON --phase precompute

echo "Precompute done. $(date)"

# --- Phase 2: Train decoder ---
echo "--- Phase: train decoder ---"
srun python $SCRIPT $COMMON --phase train

echo "Training done. $(date)"

# --- Phase 3: Generate samples ---
echo "--- Phase: generate samples ---"
srun python $SCRIPT $COMMON --phase generate

echo "Generate done. $(date)"

# --- Phase 4: Full validation ---
echo "--- Phase: fullval ---"
srun python $SCRIPT $COMMON --phase fullval

echo "Full validation done. $(date)"

EXIT_CODE=$?

echo "=== Test 13G-B finished ==="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# --- Verify outputs ---
OUTDIR=$REPO/data/gate5b_results/test13g_posthoc/d4-a4r
if [ -d "$OUTDIR" ]; then
    echo "Output directory found:"
    ls -la $OUTDIR/
    if [ -f "$OUTDIR/test13g_posthoc_results.json" ]; then
        echo "Results:"
        python -c "
import json
r = json.load(open('$OUTDIR/test13g_posthoc_results.json'))
fv = r.get('fullval', {})
if fv:
    print(f'  F1: {fv.get(\"f1\", \"N/A\")}')
    print(f'  Precision: {fv.get(\"precision\", \"N/A\")}')
    print(f'  Recall: {fv.get(\"recall\", \"N/A\")}')
"
    fi
else
    echo "WARNING: Output directory not found at $OUTDIR"
fi

exit $EXIT_CODE
