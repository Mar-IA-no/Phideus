#!/bin/bash
#SBATCH --job-name=g6expB
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --array=0-26
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate6_expB_%A_%a.out

# ── Gate 6 Exp B: Transkun Degraded Conditions ──
# Array: 3 degradations × 3 levels × 3 configs = 27 jobs
#
# Degradations:        noise(5,10,20 dB), lowpass(1000,2000,4000 Hz), data(0.1,0.25,0.5)
# Configs:             baseline-degraded, finetune-degraded, A4-degraded
# ETA: ~4h/run → ~4.5 days with 1 GPU, ~1.5 days with 3 GPUs
#
# Priority: If time limited, start with noise (array 0-8)

set -euo pipefail

# ── Environment ──
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ── Paths ──
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=/home/mfmendez/data/maestro-v3.0.0

# ── Array decoding: 9 degradation×level pairs × 3 configs ──
# Degradation × level pairs:
#   0: noise 5    1: noise 10   2: noise 20
#   3: lowpass 1000  4: lowpass 2000  5: lowpass 4000
#   6: data 0.1   7: data 0.25  8: data 0.5

DEGRADATIONS=(noise noise noise lowpass lowpass lowpass data_limit data_limit data_limit)
LEVELS=(5 10 20 1000 2000 4000 0.1 0.25 0.5)
CONFIGS=(baseline-degraded finetune-degraded A4-degraded)

DEG_IDX=$((SLURM_ARRAY_TASK_ID / 3))
CONFIG_IDX=$((SLURM_ARRAY_TASK_ID % 3))

DEGRADATION=${DEGRADATIONS[$DEG_IDX]}
LEVEL=${LEVELS[$DEG_IDX]}
CONFIG=${CONFIGS[$CONFIG_IDX]}

OUTDIR=$REPO/data/gate6_results/transkun_degraded/${DEGRADATION}_${LEVEL}_${CONFIG}
mkdir -p $OUTDIR

echo "=== Gate 6 Exp B: ${DEGRADATION}@${LEVEL} — ${CONFIG} ==="
echo "  Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "  Node: $(hostname)"
echo "  Output: $OUTDIR"

# ── Data staging ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo "Staging MAESTRO to scratch..."
cp -r $MAESTRO_SRC $SCRATCH/maestro-v3.0.0
echo "Staging complete."

# ── Run ──
srun python $REPO/experiments/bias_control/gate6/transkun_degraded.py \
    --degradation "${DEGRADATION}" \
    --level ${LEVEL} \
    --config "${CONFIG}" \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --iterations 50000 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed 42 \
    --eval-every 5000 \
    --device cuda

echo "=== DONE ==="
