# Gate 6 — AMT with Descriptor Conditioning

Gate 5B showed descriptors reorganize embedding geometry (causal +9.4pp, Test 02) but don't enrich feature decodability (Test 13G-B: F1~10% all arms). Gate 6 asks: does this geometric advantage translate to concrete musical tasks?

> **Note**: The previous "Gate 6" (RSA/CKA diagnostic, 2026-02) was absorbed by Gate 5B Test 06. Gate 6 is reassigned to AMT.

## Experiments

| Exp | Question | Method | Regime |
|-----|----------|--------|--------|
| **0** | Does Transkun transcribe our segments? | Pretrained inference | Both |
| **A** | Do descriptors add info SOTA doesn't have? | Inject A4 into Transkun | 44.1kHz/16s |
| **B** | More useful under degradation? | Transkun+A4 with noise/filtering | 44.1kHz/16s |
| **C** | Are our features musically decodifiable? | AMT decoder on VICReg features | 24kHz/4s |

## SOTA Model: Transkun v2

- **F1**: 92.94% on MAESTRO v3 (Note+Off+Vel)
- **Params**: 12.9M
- **Architecture**: CNN → 6-layer axial transformer → Semi-CRF
- **Input**: Mel spec 44.1kHz, 229 bins, hop=1024

## Files

```
experiments/bias_control/gate6/
├── README.md                      # This file
├── evaluation.py                  # Common mir_eval wrappers + conventions
├── test_transkun_baseline.py      # Exp 0: baseline verification
├── a4_descriptor_standalone.py    # A4 DSP wrapper for 44.1kHz
├── transkun_a4_finetune.py        # Exp A: Transkun + A4 fine-tuning
├── transkun_degraded.py           # Exp B: degraded conditions
├── amt_decoder_model.py           # AMTDecoder class (~38M params)
└── vicreg_amt_decoder.py          # Exp C: decoder over VICReg features

experiments/bias_control/slurm/
├── gate6_vicreg_decoder.sh        # Exp C SLURM (4 arms)
├── gate6_transkun_a4.sh           # Exp A SLURM (5 configs × 3 seeds)
└── gate6_transkun_degraded.sh     # Exp B SLURM (27 runs)
```

## Quick Start

```bash
cd /mnt/m2-1TB/Phideus
source venv/bin/activate

# Exp 0: Verify Transkun baseline (~2 min, local)
python experiments/bias_control/gate6/test_transkun_baseline.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/gate6_results/transkun_baseline \
    --device cuda

# Exp C: AMT decoder (local, ~4h/arm)
python experiments/bias_control/gate6/vicreg_amt_decoder.py \
    --descriptor d4a4 \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/gate6_results/vicreg_decoder/d4a4 \
    --epochs 80 --batch-size 16 --device cuda
```

## Evaluation Conventions

| Parameter | Value |
|-----------|-------|
| Onset tolerance | 50ms |
| Offset tolerance | 50ms or 20% (greater) |
| Pedal extension | No Ext |
| Note clipping | At segment borders |
| Velocity bins | 128 (MIDI standard) |
