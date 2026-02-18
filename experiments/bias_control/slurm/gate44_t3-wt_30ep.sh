#!/bin/bash
#SBATCH --job-name=t3wt30
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --exclude=ivb03
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/t3wt30_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/t3wt30_%j.err

# Gate 4.4 — t3-wt scratch 30 epochs (Third Tower, weighted bridge)
# #3 en screening 5ep: S=67.6% (salto tardío 47.6%→67.6%)
# Scratch con freeze-policy run-d (igual que d4a4, a4r, d4a4r, d4-a4r)

set -eo pipefail

# --- Entorno ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== t3-wt scratch 30 epochs ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/gate44_t3-wt_scratch_30ep

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
echo "Iniciando training: t3-wt scratch 30ep"

srun python $REPO/experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --mode train \
    --descriptor t3-wt \
    --from-scratch \
    --freeze-policy run-d \
    --output $OUTDIR \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --epochs 30 \
    --batch-size 16 \
    --num-workers 8 \
    --seed 42 \
    --device cuda \
    --structured-eval-epochs 5 10 15 20 25 30 \
    $RESUME_FLAG

EXIT_CODE=$?

echo "=== Training finalizado ==="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# --- Verificar outputs ---
if [ -f "$OUTDIR/final_results.json" ]; then
    echo "final_results.json encontrado. Extrayendo S metric:"
    python -c "import json; r=json.load(open('$OUTDIR/final_results.json')); print(f\"  Best S: {r.get('best_structured_S', 'N/A')}\")" || true
else
    echo "Training incompleto (sin final_results.json)."
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Exit code OK pero sin resultados finales — posible SIGTERM. Re-enviando..."
        sbatch $0
    fi
fi

exit $EXIT_CODE
