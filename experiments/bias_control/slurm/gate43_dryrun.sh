#!/bin/bash
#SBATCH --job-name=g43_dry
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/g43_dryrun_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/g43_dryrun_%j.err

# Dry run: 1 arm (a8), 1 epoch, 50 batches
# Verifica: VRAM, imports, I/O scratch, structured eval

set -euo pipefail

. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== DRY RUN Gate 4.3 Fase 5 ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# --- Verificar GPU ---
nvidia-smi -L
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.mem_get_info(0)[1]/1e9:.1f}GB')"

# --- Paths ---
REPO=/home/mfmendez/Repos/Phideus
FOUNDATION=$REPO/data/bias_control_medium/training_outputs/foundation_locked_e25.pt
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
OUTDIR=/home/mfmendez/results/gate43_fase5/dryrun_a8

# --- Copiar datos a /scratch ---
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH

echo "Copiando foundation a scratch..."
cp $FOUNDATION $SCRATCH/
echo "Foundation OK."

echo "Copiando MAESTRO a scratch..."
COPY_START=$(date +%s)
cp -r $MAESTRO_SRC $SCRATCH/
COPY_END=$(date +%s)
echo "MAESTRO copiado en $((COPY_END - COPY_START)) segundos."

# --- Verificar datos en scratch ---
ls $SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json > /dev/null
echo "Datos en scratch verificados."

# --- Dry run training: a8, 1 epoch, 50 batches ---
echo "Iniciando dry run: a8, 1ep, 50 batches..."
mkdir -p $OUTDIR

srun python $REPO/experiments/bias_control/gate42_training.py \
    --descriptor a8 \
    --checkpoint $SCRATCH/foundation_locked_e25.pt \
    --output $OUTDIR \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --epochs 1 \
    --batch-size 16 \
    --num-workers 8 \
    --max-batches-per-epoch 50 \
    --max-val-batches 50 \
    --freeze-policy run-d \
    --lr-audio-unfreeze 1e-5 \
    --lr-audio-low 5e-6 \
    --lr-midi 5e-5 \
    --lr-proj 1e-4 \
    --lr-ratio 5e-4 \
    --warmup-steps 200 \
    --seed 42 \
    --device cuda

echo ""
echo "=== DRY RUN COMPLETADO ==="
echo "Exit code: $?"
echo "End: $(date)"

# --- Verificar outputs ---
echo ""
echo "--- Outputs generados ---"
ls -lh $OUTDIR/
echo ""
echo "--- Eval epoch 1 ---"
if [ -f "$OUTDIR/eval_per_epoch/eval_epoch1.json" ]; then
    python -c "
import json
r = json.load(open('$OUTDIR/eval_per_epoch/eval_epoch1.json'))
gm = r.get('gate_metrics', r)
print(f\"  S: {gm['S']:.1%}\")
print(f\"  A2M R@10: {gm['a2m_r10']:.1%}\")
print(f\"  M2A R@10: {gm['m2a_r10']:.1%}\")
print(f\"  Hard neg: {gm.get('hard_neg', 'N/A')}\")
"
    echo "Structured eval: OK"
else
    echo "WARNING: eval_epoch1.json no encontrado"
fi

echo ""
echo "=== Si todo OK, lanzar array completo con: ==="
echo "sbatch $REPO/experiments/bias_control/slurm/gate43_fase5.sh"
