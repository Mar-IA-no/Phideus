#!/bin/bash
#SBATCH --job-name=g6B_pre
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate6_expB_preflight_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate6_expB_preflight_%j.err

# ── Gate 6 Exp B PREFLIGHT v6: Test checkpoint + resume ──
# v5 validated: training loop, torch.stack, requires_grad, degradation
# v6 tests: checkpoint save, resume from checkpoint, SIGTERM-safe workflow
#
# Plan:
#   Phase 1: 20 iters (eval_every=20 → checkpoint.pt saved at step 20)
#   Phase 2: resume from checkpoint.pt, 10 more iters (eval_every=10)
#   Verify: step continues from 20, F1/loss reported correctly

set -eo pipefail

# ── Environment (identical to real job) ──
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ── Paths ──
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0

echo "=== PREFLIGHT v6: Checkpoint + Resume Test ==="
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'NO GPU')"

# ── Quick checks ──
FAIL=0
echo -n "  MAESTRO... "
[ -d "$MAESTRO_SRC" ] && echo "OK" || { echo "MISSING"; FAIL=1; }
echo -n "  Pickles... "
[ -f "$MAESTRO_SRC/transkun_pickles/train.pickle" ] && echo "OK" || { echo "MISSING"; FAIL=1; }
echo -n "  Imports... "
python -c "import torch, transkun; print('OK')" || FAIL=1
[ $FAIL -ne 0 ] && { echo "=== PREFLIGHT FAILED ==="; exit 1; }

# ── Data staging ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo ""
echo "Staging MAESTRO to scratch..."
COPY_START=$(date +%s)
rsync -a --info=progress2 $MAESTRO_SRC/ $SCRATCH/maestro-v3.0.0/
COPY_END=$(date +%s)
echo "Staging complete in $((COPY_END - COPY_START)) seconds."

if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: MAESTRO metadata not found"; exit 1
fi

# ── Phase 1: Train 20 iters (checkpoint saves at step 20) ──
OUTDIR=$SCRATCH/preflight_resume_test
mkdir -p $OUTDIR

echo ""
echo "=========================================="
echo "=== PHASE 1: Train 20 iters (fresh) ==="
echo "=========================================="

srun python $REPO/experiments/bias_control/gate6/transkun_degraded.py \
    --degradation noise \
    --level 10 \
    --config finetune-degraded \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --iterations 20 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed 42 \
    --eval-every 20 \
    --device cuda

PHASE1_EXIT=$?
echo "  Phase 1 exit code: $PHASE1_EXIT"

# ── Verify checkpoint was saved ──
echo ""
echo "=== CHECKPOINT VERIFICATION ==="
if [ -f "$OUTDIR/checkpoint.pt" ]; then
    CKPT_SIZE=$(du -h "$OUTDIR/checkpoint.pt" | cut -f1)
    echo "  checkpoint.pt: EXISTS ($CKPT_SIZE)"
else
    echo "  checkpoint.pt: MISSING — CHECKPOINT SAVE FAILED"
    echo "  Files in outdir:"
    ls -la "$OUTDIR/"
    exit 1
fi

if [ -f "$OUTDIR/best_model.pt" ]; then
    echo "  best_model.pt: EXISTS ($(du -h "$OUTDIR/best_model.pt" | cut -f1))"
else
    echo "  best_model.pt: MISSING (no improvement — may be OK for 20 iters)"
fi

# Verify checkpoint contents
python -c "
import torch, json
ckpt = torch.load('$OUTDIR/checkpoint.pt', map_location='cpu', weights_only=False)
print(f'  Checkpoint step: {ckpt[\"step\"]}')
print(f'  Best F1: {ckpt[\"best_f1\"]:.4f} @ iter {ckpt[\"best_iter\"]}')
print(f'  State dict keys: {len(ckpt[\"state_dict\"])}')
print(f'  Optimizer state: {\"optimizer_state_dict\" in ckpt}')
print(f'  Scheduler state: {\"scheduler_state_dict\" in ckpt}')
print(f'  History entries: {len(ckpt.get(\"history\", []))}')
assert ckpt['step'] == 20, f'Expected step=20, got {ckpt[\"step\"]}'
assert 'optimizer_state_dict' in ckpt, 'Missing optimizer state'
assert 'scheduler_state_dict' in ckpt, 'Missing scheduler state'
print('  ALL CHECKPOINT ASSERTIONS PASSED')
" || { echo "CHECKPOINT VALIDATION FAILED"; exit 1; }

# ── Phase 2: Resume from checkpoint, train 10 more iters ──
echo ""
echo "============================================"
echo "=== PHASE 2: Resume + 10 more iters ==="
echo "============================================"

# We need iterations=30 total (resume at 20, run until 30)
srun python $REPO/experiments/bias_control/gate6/transkun_degraded.py \
    --degradation noise \
    --level 10 \
    --config finetune-degraded \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --iterations 30 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed 42 \
    --eval-every 10 \
    --resume "$OUTDIR/checkpoint.pt" \
    --device cuda

PHASE2_EXIT=$?
echo "  Phase 2 exit code: $PHASE2_EXIT"

# ── Verify resume worked ──
echo ""
echo "=== RESUME VERIFICATION ==="

# Check training_results.json (written when training completes)
if [ -f "$OUTDIR/training_results.json" ]; then
    python -c "
import json
r = json.load(open('$OUTDIR/training_results.json'))
print(f'  Config: {r.get(\"config_name\", \"?\")}')
print(f'  Best F1: {r[\"best_f1\"]:.4f} @ iter {r[\"best_iter\"]}')
print(f'  Time: {r[\"total_time_minutes\"]:.1f} min')
print(f'  History entries: {len(r.get(\"history\", []))}')
"
    echo "  training_results.json: OK"
else
    echo "  training_results.json: MISSING — training did not complete"
fi

# Check updated checkpoint
python -c "
import torch
ckpt = torch.load('$OUTDIR/checkpoint.pt', map_location='cpu', weights_only=False)
print(f'  Final checkpoint step: {ckpt[\"step\"]}')
print(f'  Final best F1: {ckpt[\"best_f1\"]:.4f}')
assert ckpt['step'] == 30, f'Expected step=30 after resume, got {ckpt[\"step\"]}'
print('  RESUME ASSERTION PASSED: step correctly advanced from 20 to 30')
" || { echo "RESUME VALIDATION FAILED"; exit 1; }

# ── Final report ──
echo ""
echo "==========================================="
echo "=== PREFLIGHT v6 RESULTS ==="
echo "==========================================="
echo "  Phase 1 (fresh 20 iters): exit $PHASE1_EXIT"
echo "  Phase 2 (resume +10 iters): exit $PHASE2_EXIT"
echo "  Checkpoint save: PASS"
echo "  Checkpoint load + resume: PASS"
echo "  Step continuity (20 → 30): PASS"
echo ""
echo "  VERDICT: READY for checkpoint+resubmit workflow"
echo "  Recommended: sbatch gate6_transkun_degraded.sh (27 jobs)"
echo "               sbatch gate6_transkun_a4.sh (15 jobs)"
echo "==========================================="
