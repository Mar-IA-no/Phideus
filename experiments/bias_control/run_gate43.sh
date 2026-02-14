#!/bin/bash
set -euo pipefail

# Gate 4.3 — Audio Descriptors + Cross-Attention + Dual Injection (8 arms × 5ep fresh)
#
# Tests the Phideus hypothesis: do harmonic ratios from audio add signal?
#
# Full order (single-injection complete before duals):
#   d0     — baseline (no descriptor)                    [COMPLETE]
#   d4     — MIDI intervals (temperado, 12-TET)          [COMPLETE]
#   a4     — audio log-freq deltas (concat)              [COMPLETE]
#   a7     — rational attractor (concat)
#   a4x    — audio log-freq deltas (cross-attention)
#   a7x    — rational attractor (cross-attention)
#   d4a4   — dual: MIDI intervals + audio log-freq
#   d4a7   — dual: MIDI intervals + rational attractor

FOUNDATION="data/bias_control_medium/training_outputs/foundation_locked_e25.pt"
MAESTRO="data/maestro_v3/maestro-v3.0.0"
# Continuation: d0, d4, a4 already complete in gate43_20260214_1000
BASE="data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000"
mkdir -p "$BASE"

echo "============================================="
echo "  GATE 4.3 — continuation: a7 a4x a7x d4a4 d4a7 (5ep fresh each)"
echo "  Output: $BASE"
echo "  Foundation MD5: $(md5sum $FOUNDATION | cut -d' ' -f1)"
echo "  Start: $(date -u)"
echo "============================================="

COMMON_ARGS="--checkpoint $FOUNDATION --maestro-dir $MAESTRO \
  --epochs 5 --batch-size 16 --num-workers 8 \
  --freeze-policy run-d --max-batches-per-epoch 1000 --max-val-batches 846 \
  --embed-batch-size 16 --seed 42 --warmup-steps 200 --device cuda \
  --lr-audio-unfreeze 1e-5 --lr-audio-low 5e-6 \
  --lr-midi 5e-5 --lr-proj 1e-4 --lr-ratio 5e-4"

for ARM in a7 a4x a7x d4a4 d4a7; do
  echo ""
  echo "============================================="
  echo "  ARM: $ARM (5ep fresh from foundation)"
  echo "  Start: $(date -u)"
  echo "============================================="

  python experiments/bias_control/gate42_training.py \
    --descriptor $ARM --output "$BASE/$ARM" $COMMON_ARGS

  echo "  $ARM COMPLETE at $(date -u)"
done

echo ""
echo "============================================="
echo "  GATE 4.3 COMPLETE"
echo "  End: $(date -u)"
echo "============================================="
echo ""
echo "Summary table:"
for ARM in d0 d4 a4 a7 a4x a7x d4a4 d4a7; do
  echo "--- $ARM ---"
  for E in 1 2 3 4 5; do
    F="$BASE/$ARM/eval_per_epoch/eval_epoch${E}.json"
    [ -f "$F" ] && python3 -c "
import json, sys
d=json.load(open(sys.argv[1]))
m=d['gate_metrics']
print(f'  e{sys.argv[2]}: S={m[\"S\"]:.1%} A2M={m[\"a2m_r10\"]:.1%} M2A={m[\"m2a_r10\"]:.1%} hard={m[\"hard_neg\"]:.1%}')
" "$F" "$E"
  done
done
