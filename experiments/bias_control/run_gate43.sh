#!/bin/bash
set -euo pipefail

# Gate 4.3 — Multi-phase experiment
#
# Tests the Phideus hypothesis: do harmonic ratios from audio add signal?
#
# Gate 4.3 phases:
#   Fase 0 (COMPLETE): d0, d4, a4 — baselines + concat
#   Fase 1 (COMPLETE): a7, a4x, a7x — remaining single-injection
#   Fase 2 (COMPLETE): d4x — MIDI cross-attention
#   Fase 3 (COMPLETE): d4a4, d4a4cm — duals (same-mod + cross-modal)
#   Fase 5 (THIS RUN): a4r, d4r, a8, a9 — reverse cross-att + new descriptors
#
# Results Fases 0-3 (9 arms): d4a4=69.8% best, concat>cross-att, same-mod>cross-mod

FOUNDATION="data/bias_control_medium/training_outputs/foundation_locked_e25.pt"
MAESTRO="data/maestro_v3/maestro-v3.0.0"
BASE="data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000"
mkdir -p "$BASE"

COMMON_ARGS="--checkpoint $FOUNDATION --maestro-dir $MAESTRO \
  --epochs 5 --batch-size 16 --num-workers 8 \
  --freeze-policy run-d --max-batches-per-epoch 1000 --max-val-batches 846 \
  --embed-batch-size 16 --seed 42 --warmup-steps 200 --device cuda \
  --lr-audio-unfreeze 1e-5 --lr-audio-low 5e-6 \
  --lr-midi 5e-5 --lr-proj 1e-4 --lr-ratio 5e-4"

# =============================================
# FASE 5: Reverse cross-attention + new descriptors
# a4r: A4 reverse cross-att (descriptors query features)
# d4r: D4 reverse cross-att (intervals query embeddings)
# a8:  Onset-weighted chroma (Route A inspired)
# a9:  IDF-weighted attractor (Route B inspired)
# =============================================

echo "============================================="
echo "  GATE 4.3 Fase 5 — a4r d4r a8 a9 (5ep fresh each)"
echo "  Output: $BASE"
echo "  Foundation MD5: $(md5sum $FOUNDATION | cut -d' ' -f1)"
echo "  Start: $(date -u)"
echo "============================================="

for ARM in a4r d4r a8 a9; do
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
echo "  GATE 4.3 Fase 5 COMPLETE"
echo "  End: $(date -u)"
echo "============================================="
echo ""
echo "Summary table (all arms):"
for ARM in d0 d4 a4 a7 a4x a7x d4x d4a4 d4a4cm a4r d4r a8 a9; do
  DIR="$BASE/$ARM"
  [ -d "$DIR" ] || continue
  echo "--- $ARM ---"
  for E in 1 2 3 4 5; do
    F="$DIR/eval_per_epoch/eval_epoch${E}.json"
    [ -f "$F" ] && python3 -c "
import json, sys
d=json.load(open(sys.argv[1]))
m=d['gate_metrics']
print(f'  e{sys.argv[2]}: S={m[\"S\"]:.1%} A2M={m[\"a2m_r10\"]:.1%} M2A={m[\"m2a_r10\"]:.1%} hard={m[\"hard_neg\"]:.1%}')
" "$F" "$E"
  done
done
