#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <output-base-dir>"
  echo "Example: $0 data/bias_control_medium/training_outputs/gate42/screening_20260213_1700"
  exit 1
fi

BASE="$1"
FOUNDATION="data/bias_control_medium/training_outputs/foundation_locked_e25.pt"
MAESTRO="data/maestro_v3/maestro-v3.0.0"

echo "=== GATE 4.2 STAGE 1 SCREENING ==="
echo "Output directory: $BASE"
echo "Foundation: $FOUNDATION"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
mkdir -p "$BASE"

# Trazabilidad: loguear MD5 del foundation
echo "Foundation MD5: $(md5sum $FOUNDATION)" | tee -a "$BASE/stage1_meta.log"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$BASE/stage1_meta.log"

# TODOS los parametros canonicos explicitados — NINGUNO depende de defaults
COMMON="--checkpoint $FOUNDATION --maestro-dir $MAESTRO \
  --epochs 3 --batch-size 16 --num-workers 8 \
  --freeze-policy run-d --max-batches-per-epoch 1000 --max-val-batches 846 \
  --embed-batch-size 16 --seed 42 --warmup-steps 200 --device cuda \
  --lr-audio-unfreeze 1e-5 --lr-audio-low 5e-6 \
  --lr-midi 5e-5 --lr-proj 1e-4 --lr-ratio 5e-4"

echo ""
echo "=== D0 CONTROL (3 epochs) ==="
echo "Started D0: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
python experiments/bias_control/gate42_training.py \
  --descriptor d0 --output "$BASE/d0" $COMMON
echo "Finished D0: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

echo ""
echo "=== D1 PITCH RATIO HISTOGRAM (3 epochs) ==="
echo "Started D1: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
python experiments/bias_control/gate42_training.py \
  --descriptor d1 --output "$BASE/d1" --ratio-weight 0.1 $COMMON
echo "Finished D1: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

echo ""
echo "=== D4 INPUT AUGMENTATION (3 epochs) ==="
echo "Started D4: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
python experiments/bias_control/gate42_training.py \
  --descriptor d4 --output "$BASE/d4" $COMMON
echo "Finished D4: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

echo ""
echo "=== STAGE 1 COMPLETE ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$BASE/stage1_meta.log"
echo "Results in: $BASE"
