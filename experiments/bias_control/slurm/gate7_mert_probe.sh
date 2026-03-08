#!/bin/bash
#SBATCH --job-name=g7probe
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate7_probe_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate7_probe_%j.err

# ── Gate 7: MERT-large Linear Probe (Exp 7.0) ──
# Probes A4 accessibility in MERTLite (D0), MERT-95M, MERT-330M.
# First run downloads HF models (~1.3GB for 330M) to ~/.cache/huggingface/
# Subsequent runs use cache.
# ETA: ~2-3h (feature extraction dominates for 330M)

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
OUTDIR=$REPO/data/gate7_results

mkdir -p $OUTDIR/features $OUTDIR/probe_results
mkdir -p $REPO/logs

echo "=== Gate 7 MERT-large Linear Probe ==="
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Output: $OUTDIR"

# ── Data staging ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo "Staging MAESTRO to scratch..."
rsync -a --info=progress2 $MAESTRO_SRC/ $SCRATCH/maestro-v3.0.0/
echo "Staging complete."

# ── Run Exp 7.0: segment-level probe (primary) ──
srun python $REPO/experiments/bias_control/gate7/mert_large_probe.py \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --d0-checkpoint $REPO/models/gate5b/D0/best_model.pt \
    --output $OUTDIR \
    --encoders MERTLite MERT-v1-95M MERT-v1-330M \
    --n-splits 5 \
    --seed 42 \
    --device cuda \
    --batch-size 32

echo "=== Exp 7.0 complete ==="

# ── Quick result summary ──
python -c "
import json
r = json.load(open('$OUTDIR/probe_results/probe_results.json'))
seg = r['segment_level']
print('\\nGate 7 Summary:')
for enc, res in seg.items():
    print(f'  {enc:<22} R2={res[\"r2_global_mean\"]:.3f} ± {res[\"r2_global_std\"]:.3f}')
nulls = r['nulls']
print(f'  {\"Null (shuffled)\":<22} R2={nulls[\"shuffled_between\"][\"r2_global_mean\"]:.3f}')
print(f'  Time: {r[\"metadata\"][\"total_time_minutes\"]:.1f} min')
"

echo "=== DONE ==="
