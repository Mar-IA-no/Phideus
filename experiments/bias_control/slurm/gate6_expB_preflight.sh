#!/bin/bash
#SBATCH --job-name=g6B_pre
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=/home/mfmendez/Repos/Phideus/logs/gate6_expB_preflight_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/logs/gate6_expB_preflight_%j.err

# ── Gate 6 Exp B PREFLIGHT v4: Fixed torch.stack variable-length audio ──
# Changes from v3:
#   - Fix torch.stack: truncate to min length in batch (mixed sample rates)
# Changes from v2:
#   - Pickles precomputed on login node (no race condition)
#   - num_workers=2 (down from 4, should cut memory ~50%)
#   - --mem=48G to test if reduced workers fix the memory issue
#   - Added memory profiling at key stages

set -eo pipefail

# ── Environment (identical to real job) ──
. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ── Paths (identical to real job) ──
REPO=/home/mfmendez/Repos/Phideus
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0

mem_usage() {
    echo "  [MEM $(date +%H:%M:%S)] RSS=$(awk '/VmRSS/{print $2}' /proc/$$/status 2>/dev/null || echo '?')kB | free=$(free -g | awk '/Mem:/{print $4}')G"
}

# ── Pre-flight checks ──
echo "=== PREFLIGHT CHECKS ==="
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'NO GPU')"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Total RAM: $(free -g | awk '/Mem:/{print $2}')G"
mem_usage

FAIL=0

echo -n "  MAESTRO_SRC... "
if [ -d "$MAESTRO_SRC" ]; then echo "OK"; else echo "MISSING: $MAESTRO_SRC"; FAIL=1; fi

echo -n "  Transkun pickles (precomputed)... "
if [ -f "$MAESTRO_SRC/transkun_pickles/train.pickle" ] && [ -f "$MAESTRO_SRC/transkun_pickles/val.pickle" ]; then
    echo "OK ($(du -sh $MAESTRO_SRC/transkun_pickles/ | cut -f1))"
else
    echo "MISSING — run pickle prep first"; FAIL=1
fi

echo -n "  Python scripts... "
for f in transkun_degraded.py transkun_a4_finetune.py a4_descriptor_standalone.py evaluation.py; do
    if [ ! -f "$REPO/experiments/bias_control/gate6/$f" ]; then
        echo "MISSING: $f"; FAIL=1
    fi
done
if [ $FAIL -eq 0 ]; then echo "OK"; fi

echo -n "  Python imports... "
python -c "
import torch, transkun, mir_eval, pretty_midi, moduleconf
print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}, transkun=OK')
" || FAIL=1

echo -n "  Transkun pretrained weights... "
python -c "
import pkg_resources, os
p = pkg_resources.resource_filename('transkun', 'pretrained/2.0.pt')
print(f'OK ({os.path.getsize(p)/1e6:.0f}MB)' if os.path.exists(p) else 'MISSING')
" || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "=== PREFLIGHT FAILED — aborting ==="
    exit 1
fi
echo "=== PREFLIGHT PASSED ==="

# ── Data staging (identical to real job) ──
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
echo ""
echo "Staging MAESTRO to scratch..."
mem_usage
COPY_START=$(date +%s)
cp -r $MAESTRO_SRC $SCRATCH/maestro-v3.0.0
COPY_END=$(date +%s)
COPY_TIME=$((COPY_END - COPY_START))
echo "Staging complete in ${COPY_TIME} seconds."
mem_usage

# ── Verify staging ──
if [ ! -f "$SCRATCH/maestro-v3.0.0/maestro-v3.0.0.json" ]; then
    echo "ERROR: MAESTRO metadata not found in scratch"
    exit 1
fi
if [ ! -f "$SCRATCH/maestro-v3.0.0/transkun_pickles/train.pickle" ]; then
    echo "ERROR: Transkun pickles not found in scratch"
    exit 1
fi

# ── Benchmark: 100 iterations, noise 10dB, finetune-degraded ──
OUTDIR=$(mktemp -d $SCRATCH/preflight_XXXXXX)
echo ""
echo "=== BENCHMARK: 100 iters, noise@10dB, finetune-degraded ==="
echo "  Output: $OUTDIR"
echo "  Start: $(date)"
mem_usage

BENCH_START=$(date +%s)

srun python $REPO/experiments/bias_control/gate6/transkun_degraded.py \
    --degradation noise \
    --level 10 \
    --config finetune-degraded \
    --maestro-dir $SCRATCH/maestro-v3.0.0 \
    --output $OUTDIR \
    --iterations 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed 42 \
    --eval-every 50 \
    --device cuda

BENCH_END=$(date +%s)
BENCH_TIME=$((BENCH_END - BENCH_START))

echo ""
echo "==========================================="
echo "=== PREFLIGHT BENCHMARK RESULTS ==="
echo "==========================================="
echo "  MAESTRO staging: ${COPY_TIME}s"
echo "  100 iterations:  ${BENCH_TIME}s"
echo "  Per-iteration:   $(echo "scale=2; $BENCH_TIME / 100" | bc)s"
echo "  Extrapolation to 50,000 iters: $(echo "scale=1; $BENCH_TIME * 500 / 3600" | bc)h"
echo "  Suggested --time (1.5x margin): $(echo "scale=1; $BENCH_TIME * 500 * 1.5 / 3600" | bc)h"
echo "  GPU VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
mem_usage
echo "==========================================="
