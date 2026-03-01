#!/bin/bash
#SBATCH --job-name=moed-60
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --exclude=ivb03,ivb10
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/moed-60_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/moed-60_%j.err

# Batch 60ep — moe-dual (MoE expert routing dual, S=72.6% a 30ep, 6to)
# NOTA: 60ep × ~54 min/ep ≈ 54h > 48h SLURM limit.
# El script se re-envía automáticamente si recibe SIGTERM.
# El --resume busca el último checkpoint para continuar.

set -eo pipefail

. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== moe-dual scratch 60 epochs ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/batch_60ep_moe-dual

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
    --descriptor moe-dual \
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
