#!/bin/bash
#SBATCH --job-name=d4a4r-60
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --exclude=ivb03,ivb10
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/d4a4r-60_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/d4a4r-60_%j.err

# Batch 60ep — d4-a4r (D4 concat MIDI + A4r reverse audio)
# S=79.8% a 30ep (3ro). Tiene +3.1M params vs d4a4 (reverse cross-att).
# Hipótesis: necesita más epochs para que los params extra converjan.

set -eo pipefail

. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== d4-a4r scratch 60 epochs ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/batch_60ep_d4-a4r

SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo "Copiando MAESTRO a scratch..."
cp -r $MAESTRO_SRC $SCRATCH/
echo "MAESTRO copiado. $(date)"

if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: maestro-v3.0.0.json no encontrado en scratch"; exit 1
fi

mkdir -p $OUTDIR
LAST_CKPT=$(ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
RESUME_FLAG=""
if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
fi

srun python $REPO/experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --mode train \
    --descriptor d4-a4r \
    --from-scratch \
    --freeze-policy run-d \
    --output $OUTDIR \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --epochs 60 \
    --batch-size 16 \
    --num-workers 8 \
    --seed 42 \
    --device cuda \
    --structured-eval-epochs 5 10 15 20 25 30 35 40 45 50 55 60 \
    $RESUME_FLAG

EXIT_CODE=$?
echo "=== Training finalizado (exit=$EXIT_CODE) ==="
echo "End: $(date)"

if [ -f "$OUTDIR/final_results.json" ]; then
    python -c "import json; r=json.load(open('$OUTDIR/final_results.json')); print(f\"  Best S: {r.get('best_structured_S', 'N/A')}\")" || true
else
    echo "Training incompleto."
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Posible SIGTERM — re-enviando..."
        sbatch $0
    fi
fi
exit $EXIT_CODE
