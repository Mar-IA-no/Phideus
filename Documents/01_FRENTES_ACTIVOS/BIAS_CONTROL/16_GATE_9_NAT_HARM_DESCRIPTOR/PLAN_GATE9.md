# Plan: Gate 9 — Natural Harmony Retrospective Pilot (A7r / A9r)

## Context

Escalon 1 (Audio<>MIDI) demostro que descriptor-guided injection funciona (+9.4pp causal, d4a4=84.1%+/-2.3pp). Pero los descriptores ganadores NO testean la hipotesis de armonia natural:
- **A4**: envolvente espectral (Familia C, no-ratio)
- **D4**: intervalos MIDI en log2/semitonos (Familia D, armonia perceptual)

Los unicos descriptores que testean directamente HIT (ratios de just-intonation) fueron A7/A9, descartados en Gate 4.3 con mecanismos inferiores y horizonte corto:
- A7 concat (5ep): 58.8% (-1.4pp vs D0)
- A7x cross-att (5ep): 62.2% (+2.0pp vs D0) -- ya mostraba que mecanismo importa
- A9 concat (5ep): 58.8% (-1.4pp vs D0)

Reverse cross-att (a4r), el mecanismo ganador, nunca se probo con A7/A9. Este pilot lo completa.

**Framing**: Probe retrospectivo de alto valor narrativo (recomendacion Codex). No es experimento decisivo.

---

## Preregistro Interpretativo

**Antes de ver resultados**, estas son las lecturas posibles:

| Resultado | Interpretacion | Alcance |
|-----------|---------------|---------|
| a7r/a9r > D0 (75.2%) y > a4r (80.7%) | Armonia natural como descriptor privilegiado en Audio<>MIDI | Fuerte para paper, pero MAESTRO tiene confounds (temperamento) |
| a7r/a9r > D0 pero < a4r | Armonia natural aporta senal pero menos que envolvente espectral | Informativo: mecanismo funciona, pero A4 accede a mas info |
| a7r/a9r ~ D0 (+/-2pp) | Reverse cross-att no rescata A7/A9 con 30ep | Negativo acotado: esta operacionalizacion no funciona en este dominio |
| a7r/a9r < D0 | Descriptores de armonia natural interfieren | Negativo fuerte para A7/A9 en MAESTRO, no para HIT en general |

**Regla**: Comparaciones siempre mean+/-CI multi-seed vs mean+/-CI multi-seed. Stage 1 (single seed) es EXPLORATORIO -- no se hacen claims.

---

## Step 0: Code Changes -- COMPLETE (2026-03-10)

### 0.1 `experiments/bias_control/gate43_scratch/gate43_scratch_training.py` (10 edits)

1. `create_gate42_model()` -- added a7r/a9r branches (Gate42AudioReverseCrossAttModel, dim=12)
2. `create_gate42_optimizer()` -- expanded a4r condition to ('a4r', 'a7r', 'a9r')
3. `GATE42_PARAM_RANGES` run-b -- added a7r/a9r: (39M, 46M)
4. `GATE42_PARAM_RANGES` run-d -- added a7r/a9r: (64M, 72M)
5. Trainable prefixes -- expanded a4r to ('a4r', 'a7r', 'a9r')
6. Checkpoint eval_compatible -- added 'a7r', 'a9r'
7. Checkpoint archive_base -- added 'a7r', 'a9r'
8. `run_evaluate()` model reconstruction -- added a7r/a9r branch
9. Eval batch size clamp -- added 'a7r', 'a9r' to 16-batch set
10. Argparse choices -- added 'a7r', 'a9r'

### 0.2 `experiments/bias_control/gate5b/checkpoint_loader.py` (2 edits)

1. `_EVAL_BATCH_SIZES` -- added 'a7r': 16, 'a9r': 16
2. Model reconstruction -- added a7r/a9r branch with correct ad_type dispatch

### Verification

- a7r: 69,310,464 trainable params, forward pass OK
- a9r: 69,310,464 trainable params, forward pass OK
- Optimizer: 8 param groups (matching a4r structure)

---

## Step 1: Verification -- COMPLETE

Both arms pass forward check on GPU with synthetic data.

---

## Step 2: Training -- 2 arms x 30ep (~8h GPU)

Esperar a que termine el factorial de P2.5 (tmux `p25_factorial`).

```bash
mkdir -p data/gate9_results
tmux new-session -d -s gate9
tmux send-keys -t gate9 "cd /mnt/m2-1TB/Phideus && source venv/bin/activate && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
  --mode train --descriptor a7r --from-scratch --freeze-policy run-d \
  --output data/gate9_results/a7r_seed42 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --epochs 30 --batch-size 16 --num-workers 8 \
  --max-batches-per-epoch 1000 --max-val-batches 846 \
  --seed 42 --structured-eval-epochs 5 10 15 20 25 28 29 30 \
  2>&1 | tee data/gate9_results/a7r_seed42.log && \
echo '=== ARM 2: a9r ===' && \
python experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
  --mode train --descriptor a9r --from-scratch --freeze-policy run-d \
  --output data/gate9_results/a9r_seed42 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --epochs 30 --batch-size 16 --num-workers 8 \
  --max-batches-per-epoch 1000 --max-val-batches 846 \
  --seed 42 --structured-eval-epochs 5 10 15 20 25 28 29 30 \
  2>&1 | tee data/gate9_results/a9r_seed42.log" Enter
```

---

## Step 3: Evaluation Stage 1 -- EXPLORATORIO (~30 min)

Test 12 (Scoreboard) + Test 06 (RSA/CKA) para ambos arms.
Single-seed = PROVISIONAL. No se hacen claims.

Baselines (multi-seed, para contexto):
- D0: S=75.2%+/-2.3pp, CKA=0.435
- a4r: S=80.7%+/-1.9pp, CKA=0.766
- d4a4: S=84.1%+/-2.3pp, CKA=0.659

---

## Step 4: Multi-Seed -- PREREGISTRADO (~32h)

Independientemente del resultado de Stage 1, correr 5 seeds para AMBOS a7r y a9r.
Seeds: 42, 123, 456, 789, 1337 (mismas que Gate 5B Test 05).
Schedule: --structured-eval-epochs 25 26 27 28 29 30 (identico a Gate 5B multi-seed).

10 arms x ~4h = ~40h. Puede ejecutarse en UNC para paralelizar.

---

## Step 5: Results & Decision (USUARIO DECIDE)

Tabla comparativa multi-seed vs baselines locked. CKA single-seed contextual.

---

## Step 6: Test 01 Causal Ablation (CONDICIONAL)

Solo zero_audio. Los modos noise/shuffle requieren fix en collect_descriptor_stats().
Se corre para AMBOS arms si el usuario lo solicita.

---

## Output Directory Structure

```
data/gate9_results/
├── a7r_seed42/           # Stage 1
├── a7r_seed42.log
├── a9r_seed42/           # Stage 1
├── a9r_seed42.log
└── multiseed/            # Step 4
    ├── a7r_seed{42,123,456,789,1337}/
    └── a9r_seed{42,123,456,789,1337}/
```

---

## Files Modified

| File | Edits |
|------|-------|
| `experiments/bias_control/gate43_scratch/gate43_scratch_training.py` | 10 |
| `experiments/bias_control/gate5b/checkpoint_loader.py` | 2 |
