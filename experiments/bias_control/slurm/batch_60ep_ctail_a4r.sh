#!/bin/bash
#SBATCH --job-name=a4r-ct60
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --exclude=ivb03
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/a4r-ct60_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/a4r-ct60_%j.err

# Cosine-tail 60ep — a4r (reverse cross-attention, S=82.0% at 30ep)
# LR schedule: cosine like 30ep until LR=0.10 (~e24), then linear tail 0.10→0.02 to e60.

set -eo pipefail

. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== a4r cosine-tail 60 epochs ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/batch_60ep_ctail_a4r

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
    --descriptor a4r \
    --from-scratch \
    --freeze-policy run-d \
    --output $OUTDIR \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --epochs 60 \
    --batch-size 16 \
    --num-workers 8 \
    --seed 42 \
    --device cuda \
    --lr-cosine-ref-epochs 30 \
    --lr-floor 0.10 \
    --lr-tail-end 0.02 \
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
