#!/bin/bash
#SBATCH --job-name=g6expA
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --array=0-14
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate6_expA_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate6_expA_%A_%a.err

# ── Gate 6 Exp A: Transkun + A4 Fine-tuning ──
# Array: 5 configs × 3 seeds = 15 jobs
# Configs: baseline, finetune-noA4, A4-event, A4-adapter, adapter-noA4
# Seeds: 42, 123, 456
# ETA: ~1 day/run → ~5 days with 3 GPUs parallel

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
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Output: $OUTDIR"

# ── Data staging ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo "Staging MAESTRO to scratch..."
rsync -a --info=progress2 $MAESTRO_SRC/ $SCRATCH/maestro-v3.0.0/
echo "Staging complete."

# ── Run ──
srun python $REPO/experiments/bias_control/gate6/transkun_a4_finetune.py \
    --config "${CONFIG}" \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --iterations 50000 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed ${SEED} \
    --eval-every 5000 \
    --device cuda

echo "=== DONE: ${CONFIG} seed=${SEED} ==="

# ── Quick result ──
if [ -f "$OUTDIR/training_results.json" ]; then
    python -c "
import json
r = json.load(open('$OUTDIR/training_results.json'))
print(f'Config: {r.get(\"config_name\", \"?\")}')
print(f'Best F1: {r[\"best_f1\"]:.4f} @ iter {r[\"best_iter\"]}')
print(f'Time: {r[\"total_time_minutes\"]:.1f} min')
"
fi
