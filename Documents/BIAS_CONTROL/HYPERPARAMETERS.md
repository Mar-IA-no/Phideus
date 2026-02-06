# BIAS_CONTROL — Hiperparámetros Completos

**Fecha**: 2026-02-06
**Proyecto**: Cross-Modal Learning con Control de Sesgo (Audio MIDI)

---

## Arquitectura (compartida Gates 2-3)

### Audio Path

| Componente | Valor |
|------------|-------|
| Encoder | MERT-lite (frozen) |
| Output dim | 1024 |
| Trainable | No |

### MIDI Path

| Componente | Valor |
|------------|-------|
| Encoder | Transformer |
| Embed dim | 512 |
| Layers | 4 |
| Heads | 8 |
| Dropout | 0.1 |
| Positional enc | Sinusoidal |
| Event embeddings | pitch=256, velocity=128, duration=128 |
| Output dim | 512 |

### Projection Heads (Audio + MIDI)

| Componente | Valor |
|------------|-------|
| Layers | 3 (Linear-BN-ReLU x2 + Linear) |
| Hidden dim | 512 |
| Output dim | 256 |
| Normalization | BatchNorm1d |

### VICReg Loss

| Peso | Valor |
|------|-------|
| Invariance (MSE) | 10.0 |
| Variance (anti-collapse) | 10.0 |
| Covariance (decorrelation) | 1.0 |
| Variance epsilon | 1e-4 |
| Variance threshold | 1.0 |

### DANN Domain Classifier (Gate 3)

| Componente | Valor |
|------------|-------|
| Input dim | 256 (= proj output) |
| Hidden dim | 64 |
| Architecture | Linear-ReLU-Dropout-Linear-ReLU-Dropout-Linear(2) |
| Gradient Reversal | GRL con lambda schedule |

---

## Datos

| Parámetro | Valor |
|-----------|-------|
| Dataset | MAESTRO v3.0.0 |
| Piezas | 1,276 (WAV + MIDI) |
| Segment length | 4.0 s |
| Hop | 1.0 s |
| Total segments (train) | ~96K (5,994 batches x16) |
| Total segments (val) | ~13.5K (846 batches x16) |
| Batch size | 16 |
| Num workers | 8 |

---

## Gate 2: Foundation Training

| Parámetro | Valor |
|-----------|-------|
| **Optimizer** | AdamW |
| **LR projection** | 1e-3 |
| **LR MIDI encoder** | 1e-4 |
| **Weight decay** | 1e-4 |
| **Warmup steps** | 500 (linear) |
| **Scheduler** | CosineAnnealingLR |
| **Grad clip** | max_norm=1.0 |
| **Epochs** | 61 (2 fases) |
| **Phase 1 (ep1-31)** | 200 batches/epoch |
| **Phase 2 (ep32-61)** | 1000 batches/epoch |
| **Max val batches** | 200 |
| **Val shuffle** | False |
| **DANN** | No |
| **Embedding normalization** | No |
| **Best checkpoint** | Epoch 45 |

### Resultados Gate 2

| Metrica | Valor |
|---------|-------|
| Gap | 0.478 |
| R@10 pool256 (a2m) | 34.4% |
| R@10 pool256 (m2a) | 37.6% |
| Hard neg accuracy | 80.4% |
| Domain probe | 92.7% (shortcut detectado) |

---

## Gate 3 Run A: DANN sin normalización

| Parámetro | Valor |
|-----------|-------|
| **Base** | Gate 2 epoch 45 |
| **Optimizer** | AdamW (2 grupos) |
| **LR projection** | 5e-4 |
| **LR MIDI encoder** | 5e-5 |
| **Weight decay** | 1e-4 |
| **Warmup steps** | 500 (linear) |
| **Scheduler** | CosineAnnealingLR |
| **Grad clip** | max_norm=1.0 |
| **Epochs** | 10 (detenido) |
| **Batches/epoch** | 1000 |
| **Max val batches** | 200 |
| **DANN weight** | 0.01 |
| **Lambda schedule** | linear_0_to_1 (total 30K steps) |
| **Domain dropout** | 0.1 |
| **F.normalize** | **No** |
| **Best criterion** | recall_avg - 0.5*\|domain_acc - 0.5\| |
| **Checkpoint every** | 5 |

### Resultados Run A

| Metrica | Best (ep7) | ep10 |
|---------|-----------|------|
| Domain acc | 62.7% | 65.9% |
| R@10 (a2m) | 6.3% | 5.7% |
| Gap | 0.364 | 0.376 |
| Lambda | 0.23 | 0.33 |

---

## Gate 3 Run B: DANN con F.normalize

| Parámetro | Valor | Delta vs Run A |
|-----------|-------|----------------|
| **Base** | Gate 2 epoch 45 | = |
| **Optimizer** | AdamW (2 grupos) | = |
| **LR projection** | 5e-4 | = |
| **LR MIDI encoder** | 5e-5 | = |
| **Weight decay** | 1e-4 | = |
| **Warmup steps** | 500 | = |
| **Scheduler** | CosineAnnealingLR | = |
| **Grad clip** | max_norm=1.0 | = |
| **Epochs** | 10 (detenido) | = |
| **Batches/epoch** | 1000 | = |
| **Max val batches** | 200 | = |
| **DANN weight** | 0.01 | = |
| **Lambda schedule** | linear_0_to_1 (30K steps) | = |
| **Domain dropout** | 0.1 | = |
| **F.normalize** | **Si** | **CAMBIO** |
| **Best criterion** | recall_avg - 0.5*\|domain_acc - 0.5\| | = |
| **Checkpoint every** | 5 | = |

### Resultados Run B

| Metrica | Best (ep6) | ep9 |
|---------|-----------|-----|
| Domain acc | 76.8% | 73.2% |
| R@10 (a2m) | **9.4%** | 8.1% |
| Gap | **0.482** | 0.419 |
| Lambda | 0.20 | 0.30 |

---

## Gate 3 Run C: Configuración optimizada (Claude + ChatGPT)

| Parámetro | Valor | Delta vs Run B |
|-----------|-------|----------------|
| **Base** | Gate 2 epoch 45 | = |
| **Optimizer** | AdamW (**3 grupos**) | **+1 grupo** |
| **LR projection** | 5e-4 | = |
| **LR MIDI encoder** | **1e-4** | **x2** |
| **LR domain head** | **2e-4** | **NUEVO** (era =proj) |
| **Weight decay** | **1e-3** | **x10** |
| **Warmup steps** | 500 | = |
| **Scheduler** | CosineAnnealingLR | = |
| **Grad clip** | max_norm=1.0 | = |
| **Epochs** | 30 | = |
| **Batches/epoch** | 1000 | = |
| **Max val batches** | **None (846)** | **x4.2** (era 200) |
| **DANN weight** | 0.01 | = |
| **Lambda schedule** | **warmup_ramp_cap** | **CAMBIO** |
| **Lambda max** | **0.8** | **NUEVO** (era 1.0 implícito) |
| **DANN warmup** | **2000 steps** | **NUEVO** (λ=0, sin domain loss) |
| **DANN ramp** | **6000 steps** | **NUEVO** (ramp lineal 0→0.8) |
| **Domain dropout** | **0.3** | **x3** |
| **F.normalize** | Si | = |
| **Best criterion** | **recall_avg puro** | **CAMBIO** |
| **Best saves** | **3** (recall, gap, invariant) | **NUEVO** |
| **Checkpoint every** | **1** | **CAMBIO** |

### Lambda Schedule Run C

```
λ
0.8 ──────────────────────────────── cap (step 8000+)
    |           /
    |         /   ramp lineal (steps 2000-8000)
    |       /
    |     /
0.0 ────|
    warmup (steps 0-2000, λ=0, solo VICReg)
```

### Razones de cada cambio

| Cambio | Razón |
|--------|-------|
| LR MIDI encoder x2 | Run B mostró que mayor agresividad es viable |
| LR domain separado | Estabilizar clasificador independientemente del encoder |
| Weight decay x10 | Regularización más fuerte para prevenir overfitting |
| warmup_ramp_cap | VICReg primero establece representaciones, DANN gradual después |
| Lambda max 0.8 | No sobreregularizar (λ=1.0 puede destruir features útiles) |
| DANN warmup 2000 | ~2 epochs sin domain loss para estabilizar VICReg |
| Domain dropout x3 | Dificultar memorización del clasificador de dominio |
| Val all batches | Eliminar sesgo por shuffle=False + subset fijo |
| Best = recall puro | Viejo criterio con penalidad de dominio saboteaba selección |
| 3 best saves | Capturar diferentes optimas (recall vs gap vs invariance) |
| Checkpoint every 1 | Granularidad fina para análisis post-training |

---

## Tabla Comparativa Resumen

| Parámetro | Gate 2 | Run A | Run B | Run C |
|-----------|--------|-------|-------|-------|
| LR proj | 1e-3 | 5e-4 | 5e-4 | 5e-4 |
| LR MIDI | 1e-4 | 5e-5 | 5e-5 | 1e-4 |
| LR domain | — | =proj | =proj | 2e-4 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 | 1e-3 |
| DANN | No | Si | Si | Si |
| F.normalize | No | No | Si | Si |
| Lambda schedule | — | linear 0→1 | linear 0→1 | warmup_ramp_cap |
| Lambda max | — | 1.0 | 1.0 | 0.8 |
| DANN warmup | — | 0 | 0 | 2000 steps |
| Domain dropout | — | 0.1 | 0.1 | 0.3 |
| Val batches | 200 | 200 | 200 | 846 (all) |
| Best criterion | loss | recall-penalty | recall-penalty | recall puro |
| Epochs | 61 | 10 | 10 | 30 |

---

## Resultados Comparativos

| Metrica | Gate 2 | Run A best | Run B best | Run C |
|---------|--------|-----------|-----------|-------|
| Domain acc | 92.7% | 62.7% (ep7) | 76.8% (ep6) | — |
| R@10 (a2m) | 2.6% | 6.3% (ep7) | **9.4%** (ep6) | — |
| Gap | **0.478** | 0.364 | **0.482** | — |
| R@10 pool256 | 34.4% | pending | pending | — |
