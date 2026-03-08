#!/bin/bash
#SBATCH --job-name=moev
#SBATCH --partition=multi
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=ivb03
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/moev_%A_%a.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/moev_%A_%a.err

# Gate 4.4-MoE v2/v3/v4 Screening — 3 brazos, 5 epochs desde foundation
# Diagnóstico: moe-a4 sufrió inercia simétrica (zero-init + lb débil → MoE inerte)
# v2: non-zero init + router noise decay
# v3: v2 + entropy penalty
# v4: v2 + top-1 hard gating (Switch Transformer)
# Protocolo idéntico a Gate 4.4 original

set -eo pipefail

# --- Entorno ---
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# --- Mapeo array -> descriptor ---
DESCRIPTORS=(moe-a4-v2 moe-a4-v3 moe-a4-v4)
DESC=${DESCRIPTORS[$SLURM_ARRAY_TASK_ID]}

echo "=== Gate 4.4-MoE Screening: $DESC ==="
echo "Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
FOUNDATION=$REPO/data/bias_control_medium/training_outputs/foundation_locked_e25.pt
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/gate44/${DESC}

# --- Copiar datos a /scratch ---
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH

echo "Copiando foundation a scratch..."
cp $FOUNDATION $SCRATCH/
echo "Foundation copiado."

echo "Copiando MAESTRO a scratch (~120GB, esto tarda ~22 min)..."
rsync -a --info=progress2 $MAESTRO_SRC $SCRATCH/
echo "MAESTRO copiado. $(date)"

# --- Verificar datos ---
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: maestro-v3.0.0.json no encontrado en scratch"
    exit 1
fi
if [ ! -f "$SCRATCH/foundation_locked_e25.pt" ]; then
    echo "ERROR: foundation checkpoint no encontrado en scratch"
    exit 1
fi

# --- Training ---
echo "Iniciando training: $DESC"
mkdir -p $OUTDIR

srun python $REPO/experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --mode train \
    --descriptor $DESC \
    --checkpoint $SCRATCH/foundation_locked_e25.pt \
    --output $OUTDIR \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --epochs 5 \
    --batch-size 16 \
    --num-workers 8 \
    --max-batches-per-epoch 1000 \
    --max-val-batches 846 \
    --freeze-policy run-d \
    --lr-audio-unfreeze 1e-5 \
    --lr-audio-low 5e-6 \
    --lr-midi 5e-5 \
    --lr-proj 1e-4 \
    --lr-ratio 5e-4 \
    --warmup-steps 200 \
    --seed 42 \
    --device cuda \
    --structured-eval-epochs 3 5

EXIT_CODE=$?

echo "=== Training finalizado: $DESC ==="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# --- Verificar outputs ---
if [ -f "$OUTDIR/final_results.json" ]; then
    echo "final_results.json encontrado. Extrayendo S metric:"
    python -c "import json; r=json.load(open('$OUTDIR/final_results.json')); print(f\"  Best S: {r['evaluation_best']['structured_S']:.1%} (epoch {r['evaluation_best']['best_epoch']})\")" || true
else
    echo "WARNING: final_results.json NO encontrado"
fi

exit $EXIT_CODE
