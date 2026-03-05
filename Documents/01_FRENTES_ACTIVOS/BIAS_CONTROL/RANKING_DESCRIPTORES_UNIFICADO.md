# Ranking Unificado de Descriptores — Phideus Bias Control

> Documento vivo. Se actualiza con cada nuevo screening.
> Última actualización: 2026-03-05 (Gate 5B COMPLETO; Test11 2/2; Test13G-B 4/4; Gate 6 Exp C RESUBMITTED Job 1144560)

---

## Snapshot Gate 5B (validación científica)

### Test12 Scoreboard (pool=256, queries=500, seed=42)

| Arm | S | A2M R@10 | M2A R@10 |
|---|---:|---:|---:|
| D0 | 73.4% | 74.8% | 73.4% |
| d4a4 | 83.8% | 84.4% | 83.8% |
| a4r | 82.0% | 82.6% | 82.0% |
| d4-a4r | 79.8% | 81.4% | 79.8% |

### Test05 Multi-Seed (CERRADO, 5 seeds × 4 descriptors)

| Descriptor | Media | ±Std | Δ vs D0 | t-stat | p<0.05 | Cohen d |
|-----------|-------|------|---------|--------|--------|---------|
| d4a4 | 84.1% | ±2.3pp | +8.9pp | 7.12 | SI | 4.50 |
| d4-a4r | 81.2% | ±2.5pp | +6.0pp | 3.95 | SI | 2.50 |
| a4r | 80.7% | ±1.9pp | +5.5pp | 4.16 | SI | 2.63 |
| D0 | 75.2% | ±2.3pp | — | — | — | — |

Cero overlap: peor descriptor-seed (a4r 79.4%) > mejor D0-seed (77.4%).

### Test01 Causal Ablation (lectura breve)

- A4/A4r: causal dominante (caídas grandes al ablacionar audio descriptor).
- D4 en duales (`d4a4`, `d4-a4r`): efecto marginal/casi nulo.
- `d4` puro: señal débil bajo ablación de descriptor.

### Test02 Param-Matched (CERRADO, 4/4)

Arquitectura d4a4 (~66.2M trainable, 75.5M total). Misma seed, schedule. Solo cambia el descriptor.

| Mode | S | A2M R@10 | M2A R@10 | vs real |
|------|---|---------|---------|---------|
| real | 83.0% (e25) | 83.2% | 83.0% | — |
| zero | 75.0% (e28) | 75.0% | 76.0% | -8.0pp |
| random | 73.6% (e30) | 74.4% | 73.6% | -9.4pp |
| shuffled | 73.6% (e20*) | 73.8% | 73.6% | -9.4pp |

Arms ablacionados caen a nivel D0 (75.2% multi-seed) con exactamente los mismos parámetros entrenables → mejora causal, no de capacidad.
\* Cierre operativo por convergencia clara a e20. Confirmado estable a e25 (73.6%).

### Test13G Phase B — Post-Hoc Pre-Pooling Decoder (COMPLETO 4/4)

Decoder 2.44M params, BCEWithLogitsLoss, 40ep, patience=4, eval_every=5.
Ningún arm hizo early stopping — los 4 mejoraron monotónicamente hasta e40.

| Arm | best_f1 | frame_precision | frame_recall | onset_f1 | bce | cosine | best_ep |
|-----|---------|----------------|-------------|----------|-----|--------|---------|
| D0 (pool-188) | 0.1089 | 0.0580 | 0.9215 | 0.0419 | 0.8310 | 0.2599 | 40 |
| d4a4 | 0.1037 | 0.0552 | 0.9069 | 0.0406 | 0.9042 | 0.2408 | 40 |
| a4r | 0.1024 | 0.0546 | 0.9141 | 0.0410 | 0.8948 | 0.2358 | 40 |
| d4-a4r | 0.1021 | 0.0543 | 0.9224 | 0.0415 | 0.8844 | 0.2363 | 40 |

F1 ~0.10 para todos (muy bajo), precision ~5.5%, recall ~92%. D0 ligeramente mejor que descriptor-arms.
Generación: 8 samples por arm.

### Test11 Pre-Proj AB (COMPLETO 2/2)

Pre-projection probing: decodificar MIDI events desde features pre-proyección (audio z=1024, midi z=512).
120ep max, patience=15, batch=48. Controls: shuffle, mean_z, zero_z.

| Arm | Task | best_ep | val_CE | token_acc | frame_f1 | Info Retention |
|-----|------|:---:|:---:|:---:|:---:|:---:|
| d4a4 | midi2events | 10 | 2.965 | 0.306 | 0.108 | — |
| d4a4 | audio2events | 8 | 3.069 | 0.289 | 0.051 | **0.770** |
| d4-a4r | midi2events | 11 | 2.971 | 0.307 | 0.111 | — |
| d4-a4r | audio2events | 10 | 3.073 | 0.289 | 0.045 | **0.748** |

Info retention ratio = (shuffle_CE - cross_CE) / (shuffle_CE - intra_CE). Ambos ~75% — features pre-proyección retienen información cross-modal.

### Estado de batería Gate 5B

- Cerrados local: `Test12`, `Test01`, `Test04`, `Test03`, `Test06`, `Test08`, `Test10`, `Test09`.
- Cerrados UNC: `Test05`, `Test02` (4/4), `Test13G-B` (4/4), `Test11` (2/2).
- **Gate 5B COMPLETO** — todos los tests cerrados.

---

## Gate 6 — AMT (Automatic Music Transcription)

Validación downstream: ¿la ventaja geométrica se traduce a tareas musicales concretas (AMT)?

### Exp 0: Transkun Baseline (COMPLETO — LOCAL)

Transkun v2 pretrained (12.9M params) sobre 100 segmentos MAESTRO validation.

| Régimen | onset_F1 | note+off_F1 | frame_F1 |
|---------|----------|-------------|----------|
| 4s (N=50) | 0.938 | 0.667 | 0.784 |
| 16s (N=50) | 0.972 | 0.729 | 0.814 |

### Exp C: AMT Decoder sobre VICReg Features (SUBMITTED — Job 1144325)

Decoder 34.3M params sobre features VICReg congeladas. 4 arms: D0, d4a4, a4r, d4-a4r.
80 epochs, batch 16, eval cada 5 epochs. ~4-6h/arm.

| Arm | Estado al envío | Job |
|-----|-----------------|-----|
| D0 | SUBMITTED | 1144325_0 |
| d4a4 | SUBMITTED | 1144325_1 |
| a4r | SUBMITTED | 1144325_2 |
| d4-a4r | SUBMITTED | 1144325_3 |

### Exp A: Transkun + A4 Fine-tuning (PENDIENTE)

5 configs × 3 seeds = 15 jobs. ~1 día/run.
Configs: baseline, finetune-noA4, A4-event, A4-adapter, adapter-noA4.

### Exp B: Transkun Degraded Conditions (PENDIENTE)

3 degradaciones × 3 niveles × 3 configs = 27 jobs. ~4h/run.
Degradaciones: noise, lowpass, data_limit.

---

## Screening @ 5 epochs (foundation + freeze-policy run-d)

Protocolo estándar: foundation_locked_e25.pt, freeze-policy run-d, batch-size 16, seed 42.
Métrica principal: **S = min(A2M_R@10, M2A_R@10)** sobre structured pool (13,532 segmentos, 500 piezas).

| # | Brazo | Familia | Mecanismo | Best S | Best Ep | A2M | M2A | hard_neg | vs D0 | Gate |
|---|-------|---------|-----------|--------|---------|-----|-----|----------|-------|------|
| 1 | **d4a4** | Dual (MIDI+Audio) | concat | **69.8%** | 5 | 69.8% | 70.6% | 91.6% | **+9.6pp** | 4.3 |
| 2 | **a4r** | Audio (log-freq) | reverse cross-att | **68.6%** | 5 | 68.6% | 69.0% | 91.6% | **+8.4pp** | 4.3-F5 |
| 3 | **t3-wt** | Third Tower | weighted bridge | **67.6%** | 5 | 71.4% | 67.6% | 91.2% | **+7.4pp** | 4.4 |
| 4 | **t3-tri** | Third Tower | trilinear bridge | **65.0%** | 5 | 65.4% | 65.0% | 90.6% | +4.8pp | 4.4 |
| 5 | d4r | MIDI (intervals) | reverse cross-att | 64.2% | 5 | 64.2% | 64.4% | 93.2% | +4.0pp | 4.3-F5 |
| 6 | D4 | MIDI (intervals) | concat | 63.6% | 5 | 63.6% | 64.4% | 91.2% | +3.4pp | 4.3 |
| 6 | A4 | Audio (log-freq) | concat | 63.6% | 5 | 65.8% | 63.6% | 92.4% | +3.4pp | 4.3 |
| 8 | A4x | Audio (log-freq) | cross-att | 62.6% | 5 | 64.0% | 62.6% | 92.4% | +2.4pp | 4.3 |
| 9 | A7x | Audio (attractor) | cross-att | 62.2% | 5 | 62.2% | 63.8% | 92.0% | +2.0pp | 4.3 |
| 10 | **D0** | — | **baseline** | **60.2%** | 3 | 60.4% | 60.2% | 90.0% | — | 4.3 |
| 11 | moe-a4-v2 | MoE v2 | non-zero init + noise decay | 60.2% | 5 | 60.4% | 60.2% | 90.8% | 0.0pp | 4.4-MoE |
| 12 | D4x | MIDI (intervals) | cross-att | 60.0% | 4 | 60.0% | 60.4% | 91.4% | -0.2pp | 4.3 |
| 13 | moe-a4-v4 | MoE v4 | top-1 hard gating | 59.4% | 5 | 60.6% | 59.4% | 91.2% | -0.8pp | 4.4-MoE |
| 13 | film-dual | FiLM | modulation (dual) | 59.4% | 5 | 60.2% | 59.4% | 91.4% | -0.8pp | 4.4 |
| 15 | film-a4 | FiLM | modulation (audio) | 59.2% | 3 | 60.8% | 59.2% | 89.8% | -1.0pp | 4.4 |
| 15 | moe-dual | MoE | expert routing (dual) | 59.2% | 5 | 61.2% | 59.2% | 91.6% | -1.0pp | 4.4 |
| 15 | moe-a4-v3 | MoE v3 | entropy penalty | 59.2% | 5 | 60.6% | 59.2% | 91.2% | -1.0pp | 4.4-MoE |
| 18 | a9 | Audio (IDF-attractor) | concat | 58.8% | 5 | 58.8% | 60.8% | 90.4% | -1.4pp | 4.3-F5 |
| 18 | A7 | Audio (attractor) | concat | 58.8% | 5 | 60.2% | 58.8% | 90.2% | -1.4pp | 4.3 |
| 20 | film-d4 | FiLM | modulation (MIDI) | 58.6% | 5 | 61.0% | 58.6% | 91.8% | -1.6pp | 4.4 |
| 21 | moe-a4 | MoE | expert routing | 58.2% | 3 | 58.8% | 60.2% | 89.6% | -2.0pp | 4.4 |
| 22 | a8 | Audio (onset-chroma) | concat | 57.4% | 5 | 60.4% | 57.4% | 90.6% | -2.8pp | 4.3-F5 |
| 23 | d4a4cm | Dual (cross-modal) | concat | 52.4% | 5 | 52.4% | 56.6% | 89.6% | -7.8pp | 4.3 |
| 24 | t3-anc | Third Tower | anchor bridge | 42.2% | 5 | 42.2% | 42.2% | 89.4% | -18.0pp | 4.4 |

**24 brazos finalizados** (21 originales + 3 MoE v2/v3/v4).

### MoE v2/v3/v4 — Resultado final

Variantes diseñadas para resolver la inercia simétrica de moe-a4.
Diagnóstico original: zero-init + lb_weight débil (0.01) → routing uniforme → expertos idénticos → MoE inerte.

| Brazo | Mecanismo | S@e3 | S@e5 | aux@e5 | Resultado |
|-------|-----------|------|------|--------|-----------|
| moe-a4-v2 | Non-zero init + router noise decay | 58.6% | **60.2%** | 0.001 | Mejor MoE, empata D0 |
| moe-a4-v3 | v2 + entropy penalty | 59.8% | 59.2% | 0.157 | Bajó de e3→e5, aux activo pero insuficiente |
| moe-a4-v4 | v2 + top-1 hard gating | 59.2% | 59.4% | 0.001 | Hard gating no rompió simetría |

Conclusión: ninguno supera D0. Familia MoE no competitiva en screening 5ep.

---

## Runs largos (30 epochs, scratch)

| Descriptor | Protocolo | Best S | Best Ep | A2M | M2A | hard_neg | Tiempo total |
|-----------|-----------|--------|---------|-----|-----|----------|-------------|
| **d4a4** | scratch, run-d, seed 42 | **83.6%** | 30 | 83.6% | 84.2% | 95.2% | ~15.5h |
| d4a4 | scratch, multi-seed (5) | **84.1% ±2.3pp** | 30 | — | — | — | ~78h total |
| **a4r** | scratch, run-d, seed 42 | **82.0%** | 29 | 82.6% | 82.0% | 94.4% | 12.3h |
| **d4-a4r** | scratch, run-d, seed 42 | **79.8%** | 30 | 81.4% | 79.8% | 94.2% | 12.1h |
| **t3-wt** | scratch, run-d, seed 42 | **79.8%** | 30 | 82.4% | 79.8% | 94.8% | 24.8h |
| **d4a4r** | scratch, run-d, seed 42 | **74.4%** | 30 | 74.4% | 74.8% | 92.0% | 12.4h |
| **moe-dual** | scratch, run-d, seed 42 | **72.6%** | 30 | 72.8% | 72.6% | 93.4% | 26.8h |

### Curvas epoch-by-epoch (runs scratch 30ep)

#### d4a4 (benchmark)
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 10 | 74.6% | — | — | 93.0% |
| 15 | 65.8% | — | — | 91.0% |
| 20 | 75.6% | — | — | 93.6% |
| 25 | 82.2% | — | — | 95.4% |
| 28 | 82.8% | — | — | 94.8% |
| 29 | 82.6% | — | — | 95.2% |
| 30 | **83.6%** | 83.6% | 84.2% | 95.2% |

#### a4r
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 61.8% | 61.8% | 62.2% | 91.0% |
| 10 | 69.0% | 71.8% | 69.0% | 90.2% |
| 15 | 77.2% | 77.2% | 77.2% | 94.0% |
| 20 | 77.6% | 77.6% | 77.8% | 94.8% |
| 25 | 80.4% | 81.4% | 80.4% | 94.6% |
| 28 | 81.8% | 83.2% | 81.8% | 94.4% |
| 29 | **82.0%** | 82.6% | 82.0% | 94.4% |
| 30 | 80.2% | 82.2% | 80.2% | 94.6% |

#### d4-a4r
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 62.2% | 62.2% | 62.6% | 90.8% |
| 10 | 58.8% | 58.8% | 60.6% | 89.0% |
| 15 | 72.2% | 72.2% | 72.2% | 91.0% |
| 20 | 77.6% | 77.6% | 77.6% | 94.2% |
| 25 | 79.2% | 80.4% | 79.2% | 94.2% |
| 30 | **79.8%** | 81.4% | 79.8% | 94.2% |

#### d4a4r
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 43.8% | 43.8% | 47.8% | 84.0% |
| 10 | 56.0% | 56.0% | 61.6% | 90.2% |
| 15 | 67.8% | 68.8% | 67.8% | 90.8% |
| 20 | 71.4% | 72.4% | 71.4% | 92.6% |
| 25 | 74.2% | 76.4% | 74.2% | 92.0% |
| 28 | 74.2% | 76.0% | 74.2% | 93.0% |
| 29 | 73.6% | 75.8% | 73.6% | 92.4% |
| 30 | **74.4%** | 74.4% | 74.8% | 92.0% |

#### t3-wt
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 40.0% | 40.0% | 46.6% | 86.2% |
| 10 | 57.6% | 57.6% | 58.0% | 92.0% |
| 15 | 66.2% | 66.2% | 68.2% | 92.2% |
| 20 | 77.6% | 79.2% | 77.6% | 92.6% |
| 25 | 79.4% | 81.0% | 79.4% | 93.8% |
| 30 | **79.8%** | 82.4% | 79.8% | 94.8% |

#### moe-dual
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 42.4% | 42.4% | 49.6% | 87.2% |
| 10 | 63.6% | 63.6% | 65.0% | 91.0% |
| 15 | 67.8% | 68.8% | 67.8% | 93.4% |
| 20 | 69.8% | 71.2% | 69.8% | 92.8% |
| 25 | 71.2% | 71.2% | 71.4% | 92.8% |
| 30 | **72.6%** | 72.8% | 72.6% | 93.4% |

### Comparativa lado a lado (S por epoch)

| Epoch | d4a4 | a4r | d4-a4r | t3-wt | d4a4r | moe-dual |
|-------|------|-----|--------|-------|-------|----------|
| 5 | — | 61.8% | 62.2% | 40.0% | 43.8% | 42.4% |
| 10 | 74.6% | 69.0% | 58.8% | 57.6% | 56.0% | 63.6% |
| 15 | 65.8% | 77.2% | 72.2% | 66.2% | 67.8% | 67.8% |
| 20 | 75.6% | 77.6% | 77.6% | 77.6% | 71.4% | 69.8% |
| 25 | 82.2% | 80.4% | 79.2% | 79.4% | 74.2% | 71.2% |
| 30 | **83.6%** | 80.2% | **79.8%** | **79.8%** | **74.4%** | **72.6%** |

---

## Runs extendidos (50ep / 60ep)

### a4r 60ep — cosine estirado (COMPLETO)

Cosine LR estándar estirado a 60 epochs. Hipótesis: más epochs con LR residual permite seguir aprendiendo.

| Epoch | S | A2M | M2A | hard_neg | lr_mult (est.) |
|-------|---|-----|-----|----------|---------------|
| 5 | 50.8% | 50.8% | 52.0% | 86.8% | 0.93 |
| 10 | 53.2% | 53.8% | 53.2% | 89.6% | 0.75 |
| 15 | 62.8% | 62.8% | 65.4% | 90.0% | 0.50 |
| 20 | 67.6% | 68.4% | 67.6% | 92.8% | 0.25 |
| 25 | 72.8% | 74.0% | 72.8% | 92.4% | 0.09 |
| 30 | 74.0% | 74.0% | 75.2% | 92.4% | 0.02 |
| 35 | 75.2% | 76.8% | 75.2% | 92.6% | ~0 |
| 40 | 78.6% | 78.6% | 81.6% | 94.4% | ~0 |
| 45 | 77.0% | 77.0% | 78.8% | 92.6% | ~0 |
| 50 | 77.4% | 77.4% | 79.2% | 94.0% | ~0 |
| 55 | 78.4% | 78.4% | 79.2% | 93.8% | ~0 |
| **60** | **79.4%** | 79.4% | 79.8% | 94.4% | 0 |

**Best S=79.4% (e60)** — NO superó a4r 30ep (82.0% e29). Cosine estirado retrasa convergencia: a e25 del 60ep (LR=0.09), a4r tiene 72.8% vs 80.4% del 30ep al mismo epoch.

### D0 60ep — cosine estirado (TERMINADO por time limit, e55/60)

Control: si D0 mejora mucho con más epochs, la ganancia es del training extra, no del descriptor.

| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 58.4% | 58.4% | 61.2% | 91.0% |
| 10 | 67.6% | 67.6% | 70.2% | 91.8% |
| 15 | 72.0% | 73.6% | 72.0% | 92.4% |
| 20 | 68.8% | 68.8% | 69.2% | 92.4% |
| 25 | 69.0% | 69.0% | 69.2% | 91.6% |
| 30 | 72.0% | 74.2% | 72.0% | 92.8% |
| 35 | 70.6% | 70.8% | 70.6% | 92.6% |
| 40 | 72.4% | 72.6% | 72.4% | 93.4% |
| 45 | 71.2% | 72.8% | 71.2% | 92.8% |
| **50** | **72.8%** | 72.8% | 73.2% | 93.2% |
| 55 | 72.2% | 74.0% | 72.2% | 93.2% |

**Best S=72.8% (e50)**. Murió por time limit (48h) a e55. Oscila 68-73% desde e15 — sin tendencia ascendente. Control confirma que ganancias son del descriptor.

### d4a4 60ep — cosine estirado (TERMINADO por time limit, e55/60)

| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 51.8% | 51.8% | 57.0% | 88.0% |
| 10 | 63.2% | 64.0% | 63.2% | 90.6% |
| 15 | 59.2% | 59.2% | 62.6% | 91.2% |
| 20 | 71.6% | 71.6% | 73.8% | 92.2% |
| 25 | 79.0% | 79.0% | 80.4% | 94.2% |
| 30 | 76.8% | 76.8% | 78.4% | 93.8% |
| 35 | 75.6% | 77.4% | 75.6% | 93.8% |
| 40 | 82.6% | 82.6% | 82.8% | 95.0% |
| 45 | 82.4% | 83.4% | 82.4% | 94.6% |
| **50** | **83.8%** | 84.4% | 83.8% | 95.4% |
| 55 | 83.4% | 83.4% | 84.4% | 95.2% |

**Best S=83.8% (e50)** — **NUEVO RECORD ABSOLUTO**, supera d4a4 30ep (83.6%) por +0.2pp. Murió por time limit (48h) a e55. El cosine estirado produjo un "segundo e25" alrededor de e40-e50 donde el LR cae lo suficiente para refinar.

### t3-wt 50ep — trapezoidal LR (COMPLETO)

LR schedule: warmup → hold pleno e1-25 → cosine decay e26-50 (`--lr-hold-fraction 0.5`).

| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 49.6% | 49.6% | 60.8% | 88.8% |
| 10 | 54.0% | 54.0% | 55.8% | 88.6% |
| 15 | 73.6% | 76.0% | 73.6% | 94.0% |
| 20 | 68.6% | 68.6% | 71.0% | 92.8% |
| 25 | 77.0% | 77.0% | 77.6% | 92.0% |
| 30 | 74.2% | 74.2% | 75.0% | 92.4% |
| 35 | 74.6% | 75.0% | 74.6% | 92.8% |
| 40 | 80.6% | 80.6% | 80.8% | 94.0% |
| 45 | 80.4% | 81.2% | 80.4% | 92.6% |
| **50** | **81.2%** | 81.4% | 81.2% | 93.8% |

**Best S=81.2% (e50)** — superó t3-wt 30ep (79.8%) por **+1.4pp**. Subió en el último epoch. Training time: 2446.8 min (~40.8h).

### d4-a4r 60ep — cosine estirado (COMPLETO, Job 1143088)

| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 58.6% | 58.6% | 64.0% | 90.2% |
| 10 | 56.8% | 56.8% | 59.0% | 90.8% |
| 15 | 65.8% | 65.8% | 69.4% | 91.4% |
| 20 | 72.2% | 72.8% | 72.2% | 93.6% |
| 25 | 70.4% | 71.8% | 70.4% | 91.8% |
| 30 | 71.8% | 72.0% | 71.8% | 93.6% |
| 35 | 75.4% | 77.0% | 75.4% | 93.2% |
| 40 | 76.0% | 76.0% | 77.6% | 93.8% |
| 45 | 77.8% | 78.6% | 77.8% | 93.0% |
| 50 | 78.8% | 79.4% | 78.8% | 94.8% |
| **55** | **79.8%** | 79.8% | 79.8% | 94.0% |
| 60 | 79.2% | 79.2% | 79.4% | 93.2% |

**Best S=79.8% (e55)** — igualó d4-a4r 30ep (79.8% e30). Regresión leve a e60 (79.2%). Ascenso monotónico e10→e55.

### moe-dual 60ep — cosine estirado (MUERTO por time limit, e50/60)

| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 50.0% | 50.0% | 55.8% | 88.4% |
| 10 | 57.2% | 57.2% | 60.8% | 91.6% |
| 15 | 65.8% | 67.4% | 65.8% | 93.6% |
| 20 | 67.4% | 68.0% | 67.4% | 92.8% |
| 25 | 68.0% | 68.0% | 70.4% | 92.8% |
| **30** | **73.0%** | 73.0% | 73.4% | 92.8% |
| 35 | 70.2% | 70.2% | 71.6% | 93.8% |
| 40 | 69.6% | 71.8% | 69.6% | 92.6% |
| 45 | 69.4% | 71.2% | 69.4% | 93.4% |
| 50 | 72.6% | 73.4% | 72.6% | 93.8% |

**Best S=73.0% (e30)** — superó moe-dual 30ep (72.6%) por +0.4pp. Pero cayó a 69-70% en e35-e45, rebote parcial a e50 (72.6%). Ganancia no sostenida. Murió por time limit (48h) a e50.

### Cosine-tail 60ep (nuevo scheduler)

Cosine-tail: replica curva del 30ep hasta LR=0.10 (~e24), luego cola lineal 0.10→0.02 hasta e60.
Flags: `--lr-cosine-ref-epochs 30 --lr-floor 0.10 --lr-tail-end 0.02`

#### D0 ctail 60ep (MUERTO por time limit, e59/60, Job 1143105)

| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 54.8% | 54.8% | 57.8% | 92.0% |
| 10 | 70.2% | 70.4% | 70.2% | 92.6% |
| 15 | 72.4% | 73.2% | 72.4% | 93.6% |
| 20 | 67.6% | 67.6% | 69.0% | 93.2% |
| 25 | 71.4% | 74.0% | 71.4% | 93.8% |
| 30 | 71.2% | 76.2% | 71.2% | 93.8% |
| 35 | 73.0% | 75.6% | 73.0% | 94.2% |
| 40 | 72.4% | 75.4% | 72.4% | 93.8% |
| 45 | 72.2% | 74.8% | 72.2% | 94.2% |
| **50** | **73.4%** | 74.8% | 73.4% | 94.6% |
| 55 | 73.2% | 76.4% | 73.2% | 94.6% |

**Best S=73.4% (e50)** — all-time best D0 (+0.6pp sobre cosine D0 72.8%). Murió por time limit a e59. Control. Oscila 67-73% típico.

#### d4a4 ctail 60ep (MUERTO por time limit, e58/60, Job 1143106)

| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 34.4% | 34.4% | 43.2% | 84.8% |
| 10 | 36.6% | 36.6% | 46.6% | 85.2% |
| 15 | 64.8% | 64.8% | 69.0% | 90.8% |
| 20 | 80.2% | 81.0% | 80.2% | 93.6% |
| 25 | 80.4% | 81.6% | 80.4% | 94.6% |
| **30** | **83.4%** | 83.8% | 83.4% | 95.6% |
| 35 | 83.2% | 84.0% | 83.2% | 94.6% |
| 40 | 82.6% | 82.8% | 82.6% | 94.2% |
| 45 | 82.4% | 82.4% | 82.6% | 94.2% |
| 50 | 82.8% | 83.6% | 82.8% | 95.0% |
| 55 | 81.2% | 84.0% | 81.2% | 94.0% |

**Best S=83.4% (e30)** — a -0.4pp del RECORD (83.8% cosine e50). Explosión e10→e30: +46.8pp en 20 epochs. Regresión leve e35-e55 (83.2→81.2%). Murió por time limit a e58.

#### a4r ctail 60ep (COMPLETO, Job 1143107)

| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 31.4% | 31.4% | 38.4% | 78.0% |
| 10 | 69.0% | 69.0% | 72.6% | 91.8% |
| 15 | 73.8% | 73.8% | 75.0% | 93.4% |
| 20 | 73.6% | 73.6% | 75.4% | 94.0% |
| 25 | 78.4% | 78.6% | 78.4% | 96.0% |
| 30 | 77.8% | 80.0% | 77.8% | 94.8% |
| 35 | 78.2% | 80.6% | 78.2% | 94.4% |
| 40 | 79.2% | 79.2% | 80.4% | 95.0% |
| 45 | 79.8% | 79.8% | 81.2% | 94.0% |
| 50 | 79.8% | 79.8% | 82.0% | 95.2% |
| 55 | 79.6% | 79.6% | 79.8% | 94.6% |
| **60** | **80.6%** | 81.6% | 80.6% | 94.6% |

**Best S=80.6% (e60)** — NO superó 30ep (82.0%), -1.4pp. Ascenso sostenido e30→e60 en la zona de cola lineal. Dip e30 (77.8%) → recovery e35-e50 → kick final e60 (80.6%). Training time: ~25h.

#### d4-a4r ctail 60ep (PENDIENTE — pospuesto tras Gate 5B)

Job original (1143108) cancelado por nodo degradado. Re-envíos sucesivos (1143330, 1143406) cancelados. Pendiente de lanzar tras completar Gate 5B (Tests 02 + 05).

---

## Gate 5B — Validación científica (en curso)

### Test 05: Multi-Seed Replication (30ep, scratch, 5 seeds)

Seeds: 42, 123, 456, 789, 1337. Protocolo idéntico a runs originales.

| Descriptor | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1337 | **Media** | **±Std** |
|-----------|---------|----------|----------|----------|-----------|-----------|----------|
| **d4a4** (Gate 4.5) | 83.6% | 86.4% | 84.0% | 82.0% | 84.4% | **84.1%** | **±2.3pp** |
| **d4-a4r** | 83.2% (e29) | 83.4% (e27) | 78.4% (e25) | 78.6% (e29) | 82.2% (e27) | **81.2%** | **±2.4pp** |
| **a4r** | 80.2% (e26) | 84.0% (e30) | 80.4% (e29) | 79.6% (e26) | 79.4% (e29) | **80.7%** | **±1.8pp** |
| **D0** | en curso | en curso | PENDING | PENDING | PENDING | — | — |

### Test 02: Parameter-Matched Ablations (PENDING)

3 jobs d4a4-architecture (~66.5M params) con descriptores saboteados: random, shuffled, zero.
Si d4a4 gana solo por parámetros extra, estas ablaciones deberían igualar a d4a4 (~84%).
Si la ganancia es causal (del descriptor), deberían caer a nivel D0 (~73%).

### Resumen all-time best por descriptor

| Descriptor | 5ep | 30ep | Cosine 60ep | Ctail 60ep | Trapezoidal | **All-time Best** | Nota |
|-----------|----:|-----:|------------:|-----------:|------------:|------------------:|------|
| **d4a4** | 69.8% | 83.6% | **83.8%@e50** | 83.4%@e30 | — | **83.8%** (60ep cos e50) | RECORD |
| **a4r** | 68.6% | **82.0%** | 79.4%@e60 | 80.6%@e60 | — | **82.0%** (30ep e29) | |
| **t3-wt** | 67.6% | 79.8% | — | — | **81.2%@e50** | **81.2%** (50ep trap) | |
| **d4-a4r** | — | 79.8% | 79.8%@e55 | —* | — | **79.8%** (30ep=cos60) | ctail pendiente |
| d4a4r | — | **74.4%** | — | — | — | 74.4% (30ep) | |
| **D0** | 60.2% | — | 72.8%@e50 | **73.4%@e50** | — | **73.4%** (ctail e50) | |
| moe-dual | 59.2% | 72.6% | 73.0%@e30 | — | — | **73.0%** (60ep cos e30) | MUERTO, peak no sostenido |

\* = pospuesto tras Gate 5B

---

## Observaciones empíricas

Patrones observados en los datos. No constituyen juicio GO/NO-GO — las decisiones las toma el equipo.

1. **Concat > Cross-attention** para descriptores fuertes (D4, A4)
2. **Reverse cross-att > Standard cross-att**: a4r=68.6% vs A4x=62.6% (+6.0pp)
3. **Same-modality > Cross-modal**: d4a4=69.8% vs d4a4cm=52.4% (+17.4pp)
4. **Efecto superaditivo en d4a4**: D4(+3.4) + A4(+3.4) = d4a4(+9.6), no 6.8
5. **Log-freq > Attractor**: A4 supera A7 en todos los mecanismos
6. **d4a4 late bloomer a 30ep**: dip en e15 (65.8%) pero sube fuerte e20→e30 (+8pp). Único que mejora hasta e30 sin regresión
7. **a4r converge rápido**: lidera e10-e20 pero techo en e29 (82.0%) con regresión a 80.2% en e30
8. **d4-a4r intermedio**: empata a4r en e5 y e20, pero se estanca ~79-80% — no tiene la subida tardía de d4a4
9. **d4a4r (dual reverse) no competitivo**: -9.2pp vs d4a4 a 30ep. Reverse en ambas modalidades perjudica
10. **FiLM y MoE (Gate 4.4)**: todos en franja 58-60%, en/por debajo de D0=60.2%
11. **moe-a4 inercia simétrica**: lb→0 = routing uniforme (no colapso a 1 experto). Zero-init + lb_weight=0.01 insuficiente → expertos nunca se especializan → MoE inerte. Diagnóstico confirmado por Codex
12. **MoE v2/v3/v4**: ninguno supera D0. v2 empata (60.2%). Familia MoE agotada
13. **Third Tower**: t3-wt (#3, 67.6%) y t3-tri (#4, 65.0%) son los mejores brazos de Gate 4.4
14. **t3-wt 30ep**: S@e5=40.0% → S@e30=79.8%. Empata d4-a4r en 3er lugar. Crecimiento sostenido sin regresión
15. **moe-dual 30ep**: S@e30=72.6%. Plateau desde e20 (+2.8pp en 10 epochs). 6to de 6 runs largos
16. **t3-wt = d4-a4r a 30ep**: ambos 79.8%, pero t3-wt arranca mucho peor (40% vs 62% a e5) y recupera. Curvas muy diferentes, mismo destino
17. **Cosine estirado retrasa convergencia**: a4r 60ep termina en 79.4% vs 82.0% del 30ep. El LR más alto a cada epoch retrasa la entrada a la zona de explotación
18. **D0 control sin tendencia ascendente**: oscila 68-72% desde e15. Confirma que ganancias de descriptores son reales y no artefacto de más epochs
19. **d4a4 60ep rebota a e40**: salto de 75.6%→82.6% entre e35→e40, acercándose a su 30ep peak (83.6%). Cosine estirado le da un "segundo e25"
20. **t3-wt trapezoidal supera su 30ep**: 80.6% (e40) vs 79.8% (30ep). Único descriptor que mejora con run extendido hasta ahora
21. **Velocidad por arquitectura**: a4r/d4a4r/d4-a4r ~13 min/ep (2.6x más rápido que D0 ~34 min/ep). Causa: reverse cross-att comprime audio de 2400→188 tokens antes del Transformer (O(N²) self-attention)
22. **d4a4 60ep nuevo record**: S=83.8% (e50), +0.2pp sobre 30ep (83.6%). El cosine estirado produce un "segundo e25" donde el LR baja lo suficiente (~0.05) para refinar. Curva: dip e15 (59.2%) → recovery e25 (79.0%) → dip e35 (75.6%) → peak e50 (83.8%)
23. **t3-wt trapezoidal completo**: S=81.2% (e50), +1.4pp sobre 30ep (79.8%). Subió en el último epoch (80.4→81.2%). El hold phase e1-25 con LR pleno seguido de decay agresivo e26-50 funciona
24. **D0 60ep y d4a4 60ep murieron por time limit (48h)**: ambos llegaron hasta e55 pero no completaron e60. d4a4 ya había hecho su peak a e50, D0 oscilaba sin tendencia — resubmit no justificado
25. **d4-a4r cosine 60ep iguala 30ep**: ascenso monotónico desde e10 hasta S=79.8% (e55), igualando exactamente el 30ep best. Patrón similar a a4r: el cosine estirado no produce ganancia neta
26. **ctail d4a4 explosión tardía**: e10=36.6% → e20=80.2% (+43.6pp en 10 epochs). La curva ctail replica el patrón late-bloomer del 30ep con timing similar
27. **ctail a4r COMPLETO**: e60=80.6%, -1.4pp vs 30ep (82.0%). Dip e30 (77.8%) → recovery sostenida e35-e50 → kick final e60 (+0.8pp). Cola lineal 0.10→0.02 produjo ascenso continuo pero no alcanzó el 30ep
28. **ctail D0 nuevo all-time best**: 73.4% (e50), +0.6pp sobre cosine D0 (72.8%). Oscila 67-73% típico, pero la cola lineal empujó un poco más arriba. Control sigue sin tendencia clara
29. **moe-dual cosine 60ep MUERTO**: peak e30=73.0% (+0.4pp sobre 30ep), pero cayó a 69-70% en e35-e45, rebote parcial a 72.6% (e50). Ganancia de e30 no sostenida. Murió por time limit a e50
30. **d4-a4r cosine 60ep COMPLETO**: S=79.8% (e55), regresión a 79.2% (e60). Igualó 30ep pero no lo superó. Patrón similar a a4r: cosine estirado no da ganancia neta
31. **ctail d4a4 peak a e30=83.4%**: a -0.4pp del RECORD (83.8% cosine e50). Explosión e10→e30 (+46.8pp). Regresión leve e35-e45 (83.2→82.4%), rebote a 82.8% (e50). La curva ctail replica el patrón late-bloomer con timing ligeramente adelantado vs cosine
32. **ctail d4a4 vs cosine d4a4**: ctail peak e30=83.4% vs cosine peak e50=83.8%. ctail converge ~20 epochs antes pero llega -0.4pp abajo. Trade-off: convergencia rápida vs refinamiento máximo
33. **a4r: ningún schedule extendido supera 30ep**: cosine=79.4%, ctail=80.6%, ambos <82.0% (30ep). a4r parece óptimo con el schedule agresivo original de 30ep
34. **D0 all-time best actualizado a ctail**: 73.4% (ctail e50) > 72.8% (cosine e50). La cola lineal benefició ligeramente al control, sugiriendo que el efecto no es exclusivo de descriptores
35. **d4-a4r multi-seed sorpresa**: media 81.2% ±2.4pp, supera su single-seed best (79.8% seed 42) por +1.4pp. Seeds 123 y 42 dan 83.2-83.4%, rivalizando con d4a4. Alta varianza (78.4-83.4%, rango 5pp)
36. **a4r multi-seed estable**: media 80.7% ±1.8pp. Seed 42 (82.0%) fue su mejor caso; la media cae 1.3pp. Menor varianza que d4-a4r
37. **d4a4 sigue líder en multi-seed**: 84.1% ±2.3pp vs d4-a4r 81.2% ±2.4pp. Diferencia de 2.9pp pero dispersiones solapan. d4-a4r es competitivo con 2.6x menos tiempo de cómputo

---

## Referencia histórica

- **Gate 4.0/4.1**: Audio encoder 100% frozen. Descriptores ratio sin efecto.
- **Gate 4.2**: Primer resultado positivo con D4 al descongelar audio encoder layers 2-3.

---

## Glosario de mecanismos

| Mecanismo | Código | Descripción |
|-----------|--------|-------------|
| concat | d0-d4, a4, a7 | Descriptor concatenado a features antes de proyección |
| cross-att | A4x, A7x, D4x | Q=features, K/V=descriptor (standard) |
| reverse cross-att | a4r, d4r | Q=descriptor, K/V=features (invertido) |
| dual concat | d4a4 | D4 en MIDI + A4 en audio, ambos concat |
| dual reverse | d4a4r | A4r en audio + D4r en MIDI, ambos reverse |
| dual mixed | d4-a4r | D4 concat en MIDI + A4r reverse en audio |
| cross-modal | d4a4cm | D4→audio, A4→MIDI (cruzado) |
| trilinear bridge | t3-tri | Third tower con producto trilineal audio×midi×ratio |
| anchor bridge | t3-anc | Third tower con anchor points ratio→(audio,midi) |
| weighted bridge | t3-wt | Third tower con weighted sum ratio-conditioned |
| FiLM | film-* | Feature-wise Linear Modulation (γ,β from descriptor) |
| MoE | moe-* | Mixture of Experts con routing condicionado por descriptor |
| MoE v2 | moe-*-v2 | MoE con non-zero init + router noise decay |
| MoE v3 | moe-*-v3 | v2 + entropy penalty (castiga routing uniforme) |
| MoE v4 | moe-*-v4 | v2 + top-1 hard gating (Switch Transformer) |
