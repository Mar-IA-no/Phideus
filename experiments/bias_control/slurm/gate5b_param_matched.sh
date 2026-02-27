#!/bin/bash
#SBATCH --job-name=g5b-pm
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --array=0-2  # 3 ablation modes: random, shuffled, zero
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/g5b-pm_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/g5b-pm_%A_%a.err
#
# Gate 5B — Test 2: Parameter-Matched Ablations
#
# Trains d4a4-shaped models (~66.5M params) with crippled descriptors to control
# that the improvement is causal (from ratio info, not just params).
#
# d4a4 is STANDARD speed (~35 min/ep): 35 × 30 = ~17.5h + eval ~= 19h. Fits in 48h.
#
# Usage:
#   sbatch experiments/bias_control/slurm/gate5b_param_matched.sh

set -eo pipefail

# --- Entorno ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# --- Parse array index → ablation mode ---
MODES=(random shuffled zero)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "=== Gate 5B Param-Matched Ablation: mode=${MODE} ==="
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/gate5b_param_matched/${MODE}

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

# --- Training with ablated descriptors ---
echo "Iniciando training: d4a4 param-matched, ablation=${MODE}, 30ep from scratch"

srun python $REPO/experiments/bias_control/gate5b/train_param_matched.py \
    --ablation-mode "${MODE}" \
    -- \
    --mode train \
    --descriptor d4a4 \
    --from-scratch \
    --output $OUTDIR \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --epochs 30 \
    --batch-size 16 \
    --freeze-policy run-d \
    --num-workers 8 \
    --seed 42 \
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
