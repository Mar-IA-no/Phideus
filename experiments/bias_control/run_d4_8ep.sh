#!/bin/bash
set -euo pipefail

# Gate 4.2 — D4 extended: 8 epochs fresh from foundation
# Launched to explore D4's trajectory beyond 3 epochs before Gate 4.3

FOUNDATION="data/bias_control_medium/training_outputs/foundation_locked_e25.pt"
MAESTRO="data/maestro_v3/maestro-v3.0.0"
TIMESTAMP=$(date -u +%Y%m%d_%H%M)
OUTPUT="data/bias_control_medium/training_outputs/gate42/d4_8ep_${TIMESTAMP}"

mkdir -p "$OUTPUT"

echo "============================================="
echo "  D4 INPUT AUGMENTATION — 8 epochs fresh"
echo "  Output: $OUTPUT"
echo "  Foundation MD5: $(md5sum $FOUNDATION | cut -d' ' -f1)"
echo "  Start: $(date -u)"
echo "============================================="

python experiments/bias_control/gate42_training.py \
  --descriptor d4 \
  --checkpoint "$FOUNDATION" \
  --output "$OUTPUT" \
  --maestro-dir "$MAESTRO" \
  --epochs 8 \
  --batch-size 16 \
  --num-workers 8 \
  --freeze-policy run-d \
  --max-batches-per-epoch 1000 \
  --max-val-batches 846 \
  --embed-batch-size 16 \
  --seed 42 \
  --warmup-steps 200 \
  --device cuda \
  --lr-audio-unfreeze 1e-5 \
  --lr-audio-low 5e-6 \
  --lr-midi 5e-5 \
  --lr-proj 1e-4 \
  --lr-ratio 5e-4

echo ""
echo "============================================="
echo "  D4 8ep COMPLETE"
echo "  End: $(date -u)"
echo "============================================="
echo ""
echo "Results:"
for E in 1 2 3 4 5 6 7 8; do
  F="$OUTPUT/eval_per_epoch/eval_epoch${E}.json"
  [ -f "$F" ] && python3 -c "
import json, sys
d=json.load(open(sys.argv[1]))
m=d['gate_metrics']
print(f'  e{sys.argv[2]}: S={m[\"S\"]:.1%} A2M={m[\"a2m_r10\"]:.1%} M2A={m[\"m2a_r10\"]:.1%} hard={m[\"hard_neg\"]:.1%}')
" "$F" "$E"
done
