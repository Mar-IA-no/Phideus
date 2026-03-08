#!/bin/bash
#SBATCH --job-name=g6_dryrun
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate6_dryrun_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate6_dryrun_%j.err

# ── Gate 6 Exp C: DRY RUN on short partition ──
# Validates: environment, MAESTRO staging, checkpoint loading, 1 epoch training+eval
# If this passes, the real job will work.
# Expected: ~30 min (22 min MAESTRO copy + ~8 min for 1 epoch)

set -eo pipefail

# ── Environment (identical to real job) ──
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ── Paths (identical to real job) ──
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0

# ── Validate paths before anything else ──
echo "=== PRE-FLIGHT CHECKS ==="
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'NO GPU')"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

FAIL=0

echo -n "  MAESTRO_SRC... "
if [ -d "$MAESTRO_SRC" ]; then echo "OK"; else echo "MISSING: $MAESTRO_SRC"; FAIL=1; fi

echo -n "  Python script... "
PYSCRIPT=$REPO/experiments/bias_control/gate6/vicreg_amt_decoder.py
if [ -f "$PYSCRIPT" ]; then echo "OK"; else echo "MISSING: $PYSCRIPT"; FAIL=1; fi

echo -n "  Checkpoints... "
for desc in D0 d4a4 a4r d4-a4r; do
    ckpt=$REPO/models/gate5b/$desc/best_model.pt
    if [ ! -f "$ckpt" ]; then echo "MISSING: $ckpt"; FAIL=1; fi
done
if [ $FAIL -eq 0 ]; then echo "OK (all 4)"; fi

echo -n "  Python imports... "
python -c "import torch; import mir_eval; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')" || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "=== PRE-FLIGHT FAILED — aborting ==="
    exit 1
fi
echo "=== PRE-FLIGHT PASSED ==="

# ── Data staging (identical to real job) ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
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

# ── Dry run: 1 epoch, 1 eval, d0 arm only ──
OUTDIR=$(mktemp -d $SCRATCH/dryrun_XXXXXX)
echo "=== DRY RUN: d0, 1 epoch, output: $OUTDIR ==="

srun python $PYSCRIPT \
    --descriptor d0 \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --epochs 1 \
    --batch-size 16 \
    --eval-every 1 \
    --checkpoint-every 1 \
    --seed 42 \
    --device cuda

echo ""
echo "==========================================="
echo "=== DRY RUN PASSED — safe to submit real job ==="
echo "==========================================="
echo "  Environment: OK"
echo "  MAESTRO staging: OK (${COPY_START}s)"
echo "  Checkpoint loading: OK"
echo "  Training (1 epoch): OK"
echo "  Evaluation: OK"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1)"
echo "==========================================="
