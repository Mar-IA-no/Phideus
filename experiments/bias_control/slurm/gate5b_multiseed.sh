#!/bin/bash
#SBATCH --job-name=g5b-ms
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --array=0-14  # 5 seeds × 3 descriptors = 15 jobs
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/g5b-ms_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/g5b-ms_%A_%a.err
#
# Gate 5B — Test 5: Multi-Seed Replication
#
# Trains D0, a4r, d4-a4r with 5 different seeds (30 epochs each).
# d4a4 already has 5-seed results (84.1% ± 2.3pp) from Gate 4.5.
#
# Time estimates per job:
#   D0:     ~35 min/ep × 30 = ~17.5h + evals ~= 19h  (fits in 48h)
#   a4r:    ~13 min/ep × 30 = ~6.5h  + evals ~= 8h   (fits in 48h)
#   d4-a4r: ~13 min/ep × 30 = ~6.5h  + evals ~= 8h   (fits in 48h)
#
# Usage:
#   sbatch experiments/bias_control/slurm/gate5b_multiseed.sh

set -eo pipefail

# --- Entorno ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# --- Parse array index → seed + descriptor ---
SEEDS=(42 123 456 789 1337)
DESCRIPTORS=(d0 a4r d4-a4r)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
DESC_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_IDX]}
DESCRIPTOR=${DESCRIPTORS[$DESC_IDX]}

echo "=== Gate 5B Multi-Seed: descriptor=${DESCRIPTOR}, seed=${SEED} ==="
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/gate5b_multiseed/${DESCRIPTOR}_seed${SEED}

# --- Copiar datos a /scratch ---
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH

echo "Copiando MAESTRO a scratch (~120GB, esto tarda ~22 min)..."
cp -r $MAESTRO_SRC $SCRATCH/
echo "MAESTRO copiado. $(date)"

# --- Verificar datos ---
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: maestro-v3.0.0.json no encontrado en scratch"
    exit 1
fi

# --- Crear output dir y buscar checkpoint previo ---
mkdir -p $OUTDIR
LAST_CKPT=$(ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
RESUME_FLAG=""
if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
fi

# --- Training ---
echo "Iniciando training: ${DESCRIPTOR} seed=${SEED} 30ep from scratch"

srun python $REPO/experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --mode train \
    --descriptor "${DESCRIPTOR}" \
    --from-scratch \
    --output $OUTDIR \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --epochs 30 \
    --batch-size 16 \
    --freeze-policy run-d \
    --num-workers 8 \
    --seed "${SEED}" \
    --device cuda \
    --structured-eval-epochs 25 26 27 28 29 30 \
    $RESUME_FLAG

EXIT_CODE=$?

echo "=== Training finalizado ==="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# --- Verificar outputs ---
if [ -f "$OUTDIR/final_results.json" ]; then
    echo "final_results.json encontrado. Extrayendo S metric:"
    python -c "import json; r=json.load(open('$OUTDIR/final_results.json')); eb=r['evaluation_best']; s=eb.get('gate_metrics',{}).get('S',eb.get('structured_S','?')); ep=eb.get('epoch',eb.get('best_epoch','?')); print(f'  Best S: {s:.1%} (epoch {ep})' if isinstance(s,float) else f'  Best S: {s} (epoch {ep})')"
else
    echo "Training incompleto (sin final_results.json)."
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Exit code OK pero sin resultados finales — posible SIGTERM. Re-enviando..."
        sbatch $0
    fi
fi

exit $EXIT_CODE
