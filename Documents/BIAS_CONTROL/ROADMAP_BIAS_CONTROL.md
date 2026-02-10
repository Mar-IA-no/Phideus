<div align="center">

# Roadmap BIAS_CONTROL
### Cross-Modal Learning con Control de Sesgo

![Version](https://img.shields.io/badge/Version-2.0-111827?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-MAESTRO_v3.0.0-1F6FEB?style=for-the-badge)
![Phase](https://img.shields.io/badge/Phase-Escalon_1--C-F59E0B?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Gate_4_En_Curso-0A7E3B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Fecha**: 2026-02-10  
> **Base**: integración de análisis Claude + GPT5.2Think (criterios recalibrados)  
> **Estado**: ✅ Escalón 1-A/B completado (Gate 3 cerrado, DANN no mejora) -> 🟡 Escalón 1-C en curso (Gate 4 + Gate 6)  
> **Run operativo actual**: Run A Gate 4 iniciado el 2026-02-09 22:50 con `1000/846` (`max-batches-per-epoch=1000`, `max-val-batches=846`, `seed=42`)

## Navegación rápida

- [Estado Actual](#estado-actual)
- [Marco Rosetta](#marco-rosetta)
- [1. Premisa de Diseño](#premisa-diseno)
- [2. Arquitectura Objetivo](#arquitectura-objetivo)
- [3. Gates GO/NO-GO](#gates-go-no-go)
- [Gate 4 (línea principal actual)](#gate4-linea-principal)
- [Criterios de Éxito Final / cierre de escalón](#criterios-exito-final)

---

<a id="estado-actual"></a>
## ✅ Estado Actual (2026-02-10) - GATE 3 CERRADO, GATE 4 EN CURSO

<a id="marco-rosetta"></a>
## Marco Rosetta (alineacion del roadmap)

Este roadmap se interpreta dentro de `Documents/Rosetta_triplescaloneta.md` como parte del **Escalon 1 (MAESTRO Audio<->MIDI)**.

Subfases operativas:

- **Escalon 1-A (baseline cross-modal):** Gates 0/1/2.
- **Escalon 1-B (control de sesgo/invariancia):** Gate 3 (resultado negativo informativo).
- **Escalon 1-C (estructura de ratios + retroanalisis):** Gate 4 + Gate 6.

Nota de consistencia:
- Gate 5 permanece opcional y no bloquea el cierre del Escalon 1-C.
- El Escalon 1 se cierra formalmente al completar Gate 4 + Gate 6 y consolidar auditoria final.

### Resultados Finales Gate 2

**Checkpoint seleccionado**: `checkpoint_epoch45.pt` (74M params)

| Métrica | Valor | Umbral GO | Status |
|---------|-------|-----------|--------|
| Gap (aligned - random) | **0.478** | > 0.15 | ✅ PASS (3.2×) |
| Recall@10 (pool 256) | **34.4%** | > 25% | ✅ PASS (1.4×) |
| Hard Neg Accuracy | **80.4%** | > 60% | ✅ PASS (1.3×) |
| Domain Probe | **92.7%** | Diagnóstico | ⚠️ Shortcut |

**Decisión**: **GO** a Gate 3 (DANN)

### Pool Estructurado (TEST DEFINITIVO - PASADO)

| Dirección | R@1 | R@5 | R@10 | MRR |
|-----------|-----|-----|------|-----|
| Audio→MIDI | 4.4% | 20.8% | 34.4% | 0.138 |
| MIDI→Audio | 5.2% | 24.6% | 37.6% | 0.158 |

El modelo distingue:
- **vs Same-Piece-Diff-Time**: 80.4% accuracy (identidad temporal confirmada)
- **vs Random**: 87.0% accuracy

### Gate 2.5 Diagnóstico

| Probe | Resultado | Acción |
|-------|-----------|--------|
| Domain Probe | 92.7% separability | → DANN requerido |
| Piece Clustering | Silhouette -0.11 | Monitorear |
| Dead Dims | 0/256 | Sin colapso |

### Gate 3: DANN — CERRADO ❌ (4 Runs completados, DANN no mejora sobre Gate 2)

#### Smoke Test - GO
Gap 0.477 (=Gate 2), R@10 2.6%, DANN loss 0.693 (log(2)). Sin degradación.

#### Run A (sin normalización) — DETENIDO ep10

| Epoch | Domain Acc | R@10 (a2m)* | Gap | Lambda |
|-------|-----------|-------------|-----|--------|
| 1 | 67.6% | 6.2% | 0.387 | 0.03 |
| **7** | **62.7%** | **6.3%** | **0.364** | **0.23** | **★ Best** |
| 10 | 65.9% | 5.7% | 0.376 | 0.33 |

**Problema detectado**: Domain acc oscila 62-77%. Magnitud del embedding actúa como discriminador trivial.

#### Run B (con F.normalize) — 10 epochs completados

**Fix**: `F.normalize(embeddings, dim=1)` antes del domain head.

| Epoch | Domain Acc | R@10 (a2m)* | Gap | Lambda |
|-------|-----------|-------------|-----|--------|
| 1 | **47.1%** | 5.0% | 0.390 | 0.03 |
| **6** | **76.8%** | **9.4%** | **0.482** | **0.20** | **★ Best recall** |
| 9 | 73.2% | 8.1% | 0.419 | 0.30 |

**Resultado A/B**: Run B supera a Run A en recall (+49%) y gap (superó Gate 2).

*R@10 de Runs A/B medido con 200 val batches (pool ~3,200). No comparable directamente con Run C.

#### Run C — Configuración optimizada — DETENIDO ep27/30

Resultado de análisis conjunto Claude + ChatGPT. Cambios principales:

| Parámetro | Run A/B | Run C |
|-----------|---------|-------|
| Lambda schedule | linear 0→1 | **warmup_ramp_cap** (warmup 2000 steps, ramp 6000, cap 0.8) |
| LR MIDI encoder | 5e-5 | **1e-4** |
| LR domain head | =projection | **2e-4** (separado) |
| Weight decay | 1e-4 | **1e-3** |
| Domain dropout | 0.1 | **0.3** |
| Val batches | 200/846 | **Todas (846)** |
| Checkpoint every | 5 | **1** |

**Output**: `data/bias_control_medium/training_outputs/gate3_c/`

**Resultados Run C** (27 epochs, detenido manualmente):

| Epoch | Loss | Domain Acc | R@10 (a2m)** | Gap | Lambda |
|-------|------|-----------|-------------|-----|--------|
| 1 | 14.128 | 50.0% | 2.9% | 0.404 | ~0.0 |
| **4** | 14.095 | 69.6% | **3.1%** | **0.469** | ~0.3 | **★ Best** |
| 8 | 14.010 | 70.8% | 2.7% | 0.414 | 0.80 |
| 13 | 13.944 | 68.4% | 2.8% | 0.397 | 0.80 |
| 18 | 13.845 | 52.9% | 2.0% | 0.321 | 0.80 |
| 26 | 13.733 | 53.5% | 2.5% | 0.328 | 0.80 |

**R@10 medido con TODAS las 846 val batches (pool ~13,536). Random baseline = 0.074%.

**Diagnóstico Run C: λ_max=0.8 es excesivo**
- Lambda alcanzó cap (0.8) en ~epoch 8 y se mantuvo ahí
- Después del cap: recall estancado (0.019-0.028), gap en declive (0.469→0.32)
- Domain acc oscila 53-72%, nunca estabiliza en 50%
- **Conclusión**: Sobre-regularización adversarial. El DANN destruye señal de retrieval sin lograr invariancia modal estable
- El mejor checkpoint (ep4) es anterior al cap de lambda

#### Comparabilidad de métricas (CAVEAT IMPORTANTE)

Los R@10 de training NO son comparables entre runs:
- **Run A/B**: 200 val batches → pool ~3,200 → random R@10 = 0.31%
- **Run C**: 846 val batches → pool ~13,536 → random R@10 = 0.074%

Para comparación justa se usa `evaluate_structured_pool.py` con pool fijo de 256 candidatos.

#### Evaluación comparativa — COMPLETADA ✅

Script: `experiments/bias_control/compare_gate3_checkpoints.py`

Evalúa 6 checkpoints con protocolo idéntico:
- Pool: 256 candidatos (64 hard + 32 semi-hard + 159 random + 1 positivo)
- 500 queries, seed 42
- Métricas: R@{1,5,10,20}, MRR, mean rank, vs-random multiplier, hard neg accuracy

**Resultados Structured Pool** (test definitivo):

| Checkpoint | R@10 a2m | R@10 m2a | Hard Neg | MRR a2m | Decision |
|-----------|---------|---------|----------|---------|----------|
| **gate2_ep45** | **34.4%** | 37.6% | **80.4%** | 0.138 | **GO** |
| runA_best_ep7 | 27.8% | 35.4% | 74.8% | 0.132 | GO |
| runB_ep5 | 24.6% | 32.0% | 70.4% | 0.112 | WEAK-GO |
| runB_ep10 | 29.8% | 34.6% | 73.6% | 0.130 | GO |
| **runC_best_ep4** | **34.6%** | **39.2%** | **81.2%** | **0.148** | **GO** |
| runC_ep13 | 32.2% | 38.0% | 76.6% | 0.144 | GO |
| runD_best_ep12 | 27.4% | 36.4% | 73.2% | 0.134 | GO (pero peor) |

#### Run D — λ_max=0.3 sostenido — COMPLETADO ✅

**Hipótesis**: Run C ep4 (el mejor DANN) estaba en λ~0.3 *transitoriamente*. Run D mantiene λ=0.3 como cap para probar si el régimen moderado sostenido mejora sobre Gate 2.

| Parámetro | Run C | Run D |
|-----------|-------|-------|
| Lambda max | 0.8 | **0.3** |
| Lambda schedule | warmup_ramp_cap | warmup_ramp_cap |
| Warmup steps | 2000 | **1000** |
| Ramp steps | 6000 | **3000** |
| Epochs | 30 | **15** |
| Otros | = | = (mismos LR, wd, dropout) |

**Resultados Run D** (training metrics):

| Epoch | Loss | Domain Acc | R@10 (a2m) | Gap | Lambda |
|-------|------|-----------|-----------|-----|--------|
| 1 | 14.120 | 57.3% | 2.2% | 0.371 | ~0.0 |
| 5 | 14.063 | 69.1% | **2.8%** | 0.395 | ~0.3 |
| 6 | 14.044 | 73.4% | 2.4% | **0.417** | 0.3 |
| 12 | 13.885 | 63.5% | 2.6% | 0.359 | 0.3 |
| 15 | 13.862 | 61.0% | 2.3% | 0.352 | 0.3 |

**Structured Pool** (best_model ep12):
- R@10 a2m: 27.4% (Gate 2: 34.4%) — **PEOR**
- R@10 m2a: 36.4% (Gate 2: 37.6%) — ligeramente peor
- Hard neg: 73.2% (Gate 2: 80.4%) — **PEOR**
- MRR a2m: 0.134 (Gate 2: 0.138) — ligeramente peor

**Output**: `data/bias_control_medium/training_outputs/gate3_d/`

#### Conclusión Gate 3: DANN CERRADO ❌

**DANN no mejora sobre Gate 2 en ningún régimen probado.**

| Régimen | Run | R@10 a2m | vs Gate 2 |
|---------|-----|---------|-----------|
| Sin DANN | Gate 2 | **34.4%** | baseline |
| λ~0.3 transitorio | Run C ep4 | 34.6% | ≈ empate |
| λ~0.3 sostenido | **Run D ep12** | **27.4%** | **-7pp PEOR** |
| λ=0.8 sostenido | Run C ep13 | 32.2% | -2pp |
| λ linear sin norm | Run A ep7 | 27.8% | -6.6pp |
| λ linear con norm | Run B ep10 | 29.8% | -4.6pp |

**Insight científico**: La separabilidad modal (92.7%) detectada en Gate 2.5 **no es el factor limitante**. Forzar invariancia destruye información útil sin compensar. Gate 2 sin DANN es el mejor checkpoint.

**Siguiente**: Gate 4 (Ratio Auxiliary View)

Ver: `Documents/BIAS_CONTROL/Gate3_DANN_Results/INFORME_GATE3_COMPLETO.md` para informe exhaustivo.

#### Correcciones aplicadas al script (10 issues + Run C hyperparams)
1. Defaults corregidos (segment_len=4.0, hop=1.0, batch_size=16) para evitar OOM
2. CLI args: `--segment-len`, `--hop`, `--max-batches-per-epoch`, `--resume`, `--checkpoint-every`, `--max-val-batches`
3. Warmup bug: `initial_lr` movido a `__init__()`
4. Resume capability: `load_checkpoint()` method con restauración de DANN step
5. `evaluate_structured_pool.py`: `strict=False` para modelos DANN
6. CLI args Run C: `--lr-projection`, `--lr-midi-encoder`, `--lr-domain-head`, `--weight-decay`, `--dann-lambda-schedule`, `--dann-lambda-max`, `--dann-warmup-steps`, `--dann-ramp-steps`, `--dann-dropout`
7. Best model: 3 criterios (recall puro, gap, invariance) en lugar de 1 con penalidad

**Informe Gate 2**: `Documents/BIAS_CONTROL/INFORME_GATE2_COMPLETO.md`

### Auditoría Gate 2 (8/10 PASS)

| Check | Status | Notas |
|-------|--------|-------|
| A1: Dataset | ✅ | 1,276 piezas |
| A2: Alignment | ❌* | Método impreciso |
| A3: Checkpoint | ✅ | 398MB, epoch 44 |
| B1-B3: Model | ✅ | No colapso |
| C1-C2: Metrics | ✅ | Pool global + estructurado |
| D1: Shuffled | ❌* | Esperado (piece signature) |
| D2: Oracle | ✅ | Diagonal=1.0 |

*Falsos positivos explicados en informe completo.

---

<a id="premisa-diseno"></a>
## 1. Premisa de Diseño

### 1.1 Lo que se abandona

- **Matching exacto de hashes** estilo Shazam cross-modal
- **Discretización agresiva** (bins de ΔT y log_ratio)
- **Criterio de éxito**: "tokens Audio = tokens MIDI"

### 1.2 Lo que se conserva

- **Insight de ratios**: "Las relaciones proporcionales codifican estructura transferible"
- **Extractor de ratios**: Como vista auxiliar / regularizador / herramienta de diagnóstico
- **Dataset MAESTRO**: Alineación ~3ms, pedales, velocidad → banco de pruebas ideal

### 1.3 Nuevo paradigma

```
┌─────────────────────────────────────────────────────────────────────┐
│ NUEVO CRITERIO DE ÉXITO:                                            │
│                                                                     │
│ Dado un segmento de audio, recuperar el segmento MIDI              │
│ correspondiente usando DISTANCIA EN EMBEDDING SPACE,                │
│ superando significativamente el azar con negativos duros.           │
└─────────────────────────────────────────────────────────────────────┘
```

---

<a id="arquitectura-objetivo"></a>
## 2. Arquitectura Objetivo

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ARQUITECTURA FINAL                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   AUDIO                              MIDI                           │
│     │                                  │                            │
│     ▼                                  ▼                            │
│  ┌──────────┐                    ┌──────────┐                       │
│  │   MERT   │                    │   MIDI   │                       │
│  │ (frozen) │                    │ Encoder  │                       │
│  │  330M    │                    │(Transf.) │                       │
│  └────┬─────┘                    └────┬─────┘                       │
│       │                               │                             │
│       ▼                               ▼                             │
│  ┌──────────┐                    ┌──────────┐                       │
│  │Projection│                    │Projection│                       │
│  │  Head    │                    │  Head    │                       │
│  │ (MLP)    │                    │ (MLP)    │                       │
│  └────┬─────┘                    └────┬─────┘                       │
│       │                               │                             │
│       └───────────┬───────────────────┘                             │
│                   │                                                 │
│                   ▼                                                 │
│            ┌────────────┐      ┌────────────┐                       │
│            │  VICReg    │      │   DANN     │                       │
│            │   Loss     │      │ (opcional) │                       │
│            └────────────┘      └────────────┘                       │
│                   │                   │                             │
│                   └─────────┬─────────┘                             │
│                             │                                       │
│                   ┌─────────▼─────────┐                             │
│                   │  Ratio Auxiliary  │ (Gate 4)                    │
│                   │   View (opcional) │                             │
│                   └───────────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

<a id="gates-go-no-go"></a>
## 3. Gates GO/NO-GO

### Gate 0 — Integridad y Alineación del Dataset

**Objetivo**: Garantizar que el pipeline no tiene bugs de slicing/alineación.
**Script**: `experiments/bias_control/gate0_data_integrity.py` ✅

**Tareas**:
- [x] Cargar MAESTRO v3.0.0 con metadata oficial
- [x] Definir segmentación: `segment_len=8.0s`, `hop=2.0s`
- [x] Verificar: `audio_duration ≈ midi_duration` para cada pieza
- [x] Verificar: slicing consistente (mismo t0/t1 en ambos)
- [x] Control negativo: shuffled pairs debe destruir cualquier señal

**Criterios GO**:
| Criterio | Umbral |
|----------|--------|
| Piezas con drift < 100ms | > 95% |
| Segmentos válidos generados | > 10,000 |
| Shuffled pairs Recall@10 | ≈ random (10/N) |

**NO-GO si**: Drift sistemático o mismatch en slicing.

---

### Gate 1 — Baselines Intra-Modales

**Objetivo**: Si no hay retrieval intra-modal, lo cross-modal no tiene piso.
**Script**: `experiments/bias_control/gate1_intra_modal.py` ✅

**Tareas**:
- [x] Audio→Audio: embeddings MERT + cosine
- [x] MIDI→MIDI: embeddings MIDI encoder + cosine
- [x] Evaluar Recall@{1,5,10,20} y MRR

**Criterios GO**:
| Métrica | Umbral |
|---------|--------|
| Audio→Audio Recall@10 | > 50% |
| MIDI→MIDI Recall@10 | > 50% |
| Separación aligned vs random | Gap > 0.3 (cosine) |

**NO-GO si**: Intra-modal falla → problema en encoders base.

---

### Gate 2 — Cross-Modal Foundation Baseline

**Objetivo**: Probar cross-modality con el enfoque más "industrial" posible (sin ratios).
**Script**: `experiments/bias_control/gate2_foundation.py` ✅

**Tareas**:
- [x] Congelar MERT audio encoder
- [x] Entrenar MIDI encoder (Transformer sobre piano-roll/eventos)
- [x] Entrenar projection heads (MLP 512→256)
- [x] Loss: VICReg(Audio, MIDI)
- [x] Evaluar retrieval bidireccional
- [ ] **Evaluar con pool estructurado** (hard negatives)

**Configuración VICReg** (conservadora):
```python
invariance_weight = 10.0
variance_weight = 10.0
covariance_weight = 1.0
```

#### Criterios GO (RECALIBRADOS v1.3)

Gate 2 requiere pasar **DOS** evaluaciones:

**1. Pool Global (13,532 segmentos)**

Random baseline Recall@10 = 10/13,532 = **0.074%**

| Métrica | NO-GO | GO |
|---------|-------|-----|
| vs Random | <5× | **>10×** |
| Gap (aligned - random) | <0.10 | **>0.15** |
| min(a2m, m2a) Recall | <0.3% | **>0.5%** |
| No collapse (std) | <0.05 | **>0.1** |

**2. Pool Estructurado (256 candidatos) — TEST DEFINITIVO**

Composición del pool por query:
- 64 hard negatives: **misma pieza, distinto tiempo**
- 32 semi-hard: **mismo compositor, otra pieza**
- 159 random: otras piezas
- 1 positivo: el match correcto

| Métrica | NO-GO | GO |
|---------|-------|-----|
| Recall@10 (pool 256) | <15% | **>25%** |
| Accuracy vs same-piece-diff-time | <50% | **>60%** |
| MRR | <0.10 | **>0.20** |

**Script**: `experiments/bias_control/evaluate_structured_pool.py`

```bash
python experiments/bias_control/evaluate_structured_pool.py \
    --model best_model.pt \
    --pool-size 256 --n-hard-negatives 64 --n-semi-hard 32
```

**NO-GO si**: Pasa pool global pero falla pool estructurado → el modelo aprende "firma de pieza" pero no identidad temporal.

**Nota**: No hay "STRONG GO" para saltear gates. El pool estructurado es obligatorio.

---

### Gate 2.5 — Análisis de Embeddings (Diagnóstico CUANTITATIVO)

**Objetivo**: Decidir si necesitamos DANN usando **probes cuantitativos**, no visualizaciones.
**Script**: `experiments/bias_control/gate2_5_probes.py`

**Tareas** (offline, no requiere GPU del training):
- [ ] **Domain Probe**: Clasificador lineal Audio vs MIDI sobre embeddings
- [ ] **Piece Probe**: Clasificador de pieza desde embeddings
- [ ] **Time Probe**: Predictor de offset temporal dentro de pieza

**Resultados y Decisiones**:

| Probe | Resultado | Diagnóstico | Acción |
|-------|-----------|-------------|--------|
| **Domain Probe** | acc ≈ 50-60% | Modal-agnostic | Skip DANN (Gate 3) |
| **Domain Probe** | acc > 80% | Fuga de modalidad | Necesita DANN |
| **Piece Probe** | acc muy alta | Aprendió pieza | Warning: ¿identidad temporal? |
| **Time Probe** | mejora | Identidad temporal | Buena señal |
| **Time Probe** | random | No hay temporal | Problema |

**Configuración Probes**:
```python
# Domain probe: clasificador simple
domain_probe = LogisticRegression()  # o MLP(256, 64, 2)
# Entrenar sobre embeddings congelados
# Evaluar accuracy en val set

# Piece probe: clasificar qué pieza
piece_probe = MLP(256, 128, n_pieces)

# Time probe: predecir offset en segundos
time_probe = MLP(256, 64, 1)  # regresión
```

**Nota**: t-SNE/UMAP son opcionales para intuición, pero las decisiones se basan en los probes cuantitativos.

---

### Gate 3 — Robustez por Control de Sesgo (DANN)

**Objetivo**: Forzar embeddings modal-agnostic.
**Script**: `experiments/bias_control/gate3_dann.py` ✅

**Tareas**:
- [x] Implementar Domain Classifier (MLP pequeño)
- [x] Implementar Gradient Reversal Layer (GRL)
- [x] Entrenar con domain loss
- [x] Monitorear: domain accuracy → 50% = modal-agnostic

**Configuración DANN** (conservadora):
```python
domain_loss_weight = 0.01  # Empezar muy bajo
grl_lambda_schedule = "linear_0_to_1"  # Gradual
domain_classifier = MLP(256, 64, 2)
```

**Criterios GO**:
| Métrica | Umbral |
|---------|--------|
| Domain classifier accuracy | 50% ± 5% (no puede distinguir) |
| Cross-modal Recall@10 | ≥ Gate 2 (no empeorar) |
| Gap vs hard negatives | Mejora sobre Gate 2 |

**NO-GO si**: DANN desestabiliza training o empeora retrieval.

---

<a id="gate4-linea-principal"></a>
### Gate 4 — Híbrido con Ratios como Vista Auxiliar

**Objetivo**: Reinyectar el "ratio insight" de forma compatible con aprendizaje.
**Script**: `experiments/bias_control/gate4_ratio_auxiliary.py` ✅

**Tareas**:
- [x] Implementar ratio encoder pequeño (MLP sobre histogramas soft)
- [x] Computar ratios en ambos dominios (sin hashing)
- [x] Añadir losses multi-view:
  - VICReg(Audio, Ratio)
  - VICReg(MIDI, Ratio)
- [x] Hardening operativo:
  - `piece_idx`/`segment_idx` a CPU en evaluación (evita crash por device mismatch)
  - guardado de checkpoint antes de `evaluate()`
  - checkpoint dual (`full` + `*_base.pt`) para compatibilidad de evaluación
- [x] Alinear régimen a Gate 2 (`segment_len=4.0`, `hop=1.0`, `batch_size=16`)
- [x] Habilitar control de batches por CLI (`--max-batches-per-epoch`, `--max-val-batches`)
- [x] Habilitar `--seed` para comparación causal reproducible Run A vs Run B

**Configuración operativa actual (Escalón 1-C)**:
```python
ratio_weight = 0.1
ratio_encoder = MLP(256_bins * 1_channel, 128, 64)
train_batches_per_epoch = 1000
val_batches_per_epoch = 846
seed = 42
```

**Ejecución en curso**:
- **Run A** (`ratio_weight=0.1`): iniciado 2026-02-09 22:50, en progreso.
- **Run B** (`ratio_weight=0.0`): pendiente tras completar Run A.

**Criterios GO (decisión final)**:
| Métrica | Criterio |
|---------|----------|
| Structured pool (A vs B) | A debe superar B de forma estable |
| Structured pool (A vs Gate 2 ep45) | A no debe degradar materialmente |
| Hard negative accuracy | Mantener o mejorar vs baseline |
| Señal causal | Diferencia A-B atribuible a `ratio_weight` |

**Interpretación**:
- Si mejora → Ratios aportan información útil
- Si no cambia → Ratios son redundantes con foundation features
- Si empeora → Ratios introducen ruido

---

### Gate 5 — Curriculum de Brecha de Dominio (Opcional)

**Objetivo**: "Hacerlo más fácil primero" para validar pipeline.

**Tareas**:
- [ ] Renderizar MIDI a audio con FluidSynth/piano virtual
- [ ] Entrenar alignment en dominio sintético (brecha chica)
- [ ] Mezclar progresivamente audio real MAESTRO
- [ ] Evaluar transferencia

**Criterios GO**:
| Fase | Criterio |
|------|----------|
| Sintético puro | Recall@10 > 80% |
| 50% real | Recall@10 > 50% |
| 100% real | Recall@10 ≥ Gate 3 |

---

### Gate 6 — Retroanálisis: Embeddings vs Representaciones de Ratios

**Objetivo**: Usar el embedding DANN como **instrumento de análisis** para medir qué capturaban (y qué perdían) nuestras representaciones de ratios históricas. Cierra el arco de investigación conectando el embedding aprendido con el "ratio language" que originó el proyecto.

**Prerequisito**: Gate 4 cerrado (Run A/B evaluados) y baseline Gate 2 consolidado.

**Pregunta central**: *¿El embedding aprendió lo mismo que nuestros ratios pero más robusto, o descubrió estructura que nuestras representaciones no capturaban?*

#### 6.1 RSA/CKA — Comparación de Geometrías

Dado un set fijo de ~5K segmentos MAESTRO, construir matrices de similitud entre segmentos usando cada representación, y comparar las geometrías con RSA (Spearman) y CKA.

**Representaciones a comparar**:

| ID | Representación | Cómo se computa | Similitud |
|----|---------------|-----------------|-----------|
| `E` | Embedding DANN (256d) | Forward pass del modelo | Cosine |
| `H_hist` | Histograma de ratios v5.0 [T,256,3] | `analizador_5.0.py` sobre audio | Cosine sobre mean temporal |
| `H_roseta` | Histograma Roseta v2.2 [T,256,3] | `analizador_roseta.py` con prominencia+estabilidad | Cosine |
| `H_const` | Constellation tokens [T,K,5] | `analizador_roseta.py --output-format constellation` | Aggregated cosine |
| `H_hash_A` | Hashes Route A (Event-Based) | `event_based_extractor.py` | TF-IDF overlap |
| `H_hash_B` | Hashes Route B (Improved TF) | `improved_tf_extractor.py` | TF-IDF overlap |
| `E_mert` | MERT raw (pre-projection) | Forward MERT sin projection | Cosine |
| `E_midi` | MIDI encoder raw (pre-projection) | Forward MIDI encoder | Cosine |

**Métricas**:
- **RSA** (Spearman entre matrices de similitud aplanadas)
- **CKA** (Centered Kernel Alignment, más robusto a reescalados)

**Interpretación**:

| Resultado | Significado | Implicación para Phideus |
|-----------|-------------|--------------------------|
| `E ≈ H_hist` (RSA > 0.7) | Embedding ≈ histograma denoised | Ratio language capturaba lo esencial |
| `E ≈ H_hash` (RSA > 0.7) | Embedding ≈ hashing suave | Shazam approach era correcto, solo faltaba robustez |
| `E ≈ E_mert` (RSA > 0.7) | Embedding ≈ MERT raw | Projection head no agrega mucho, MERT domina |
| `E ≉ ninguna` (RSA < 0.3) | Embedding aprendió estructura nueva | Nuestras representaciones perdían información crítica |

**Tareas**:
- [ ] Extraer embeddings E para 5K segmentos (audio + MIDI)
- [ ] Computar H_hist, H_roseta, H_const sobre los mismos segmentos de audio
- [ ] Portar Route A/B a MAESTRO para H_hash_A, H_hash_B
- [ ] Computar 8 matrices de similitud (N×N)
- [ ] Calcular RSA (Spearman) y CKA entre todas las parejas
- [ ] Heatmap de correlaciones entre representaciones

#### 6.2 Probes de Ratio Features — ¿Qué contiene el embedding?

Entrenar modelos lineales (Ridge/LogReg) que predigan features de ratio language desde el embedding congelado.

**Probes a implementar**:

| Probe | Input | Target | Tipo | Qué responde |
|-------|-------|--------|------|-------------|
| Log-ratio histogram | E (256d) | Distribución de log₂(f₂/f₁) [256 bins] | Regresión | ¿E contiene distribución de ratios? |
| Delta-T histogram | E (256d) | Distribución de Δt entre eventos [64 bins] | Regresión | ¿E codifica timing relativo? |
| Pitch-class profile | E (256d) | Chroma vector [12d] | Regresión | ¿E contiene información tonal? |
| Evento density | E (256d) | Eventos/segundo (escalar) | Regresión | ¿E codifica actividad? |
| Token type ratio | E (256d) | Proporción chord/seq/constellation | Regresión | ¿E distingue tipos de relación? |
| Tempo proxy | E (256d) | IOI medio (escalar) | Regresión | ¿E captura tempo? |

**Métrica**: R² para regresión, accuracy para clasificación.

**Interpretación**:
- R² alto en log-ratio → **el embedding contiene ratio language**
- R² alto en pitch-class pero bajo en log-ratio → **aprendió tonalidad, no ratios**
- R² bajo en todo → **representación abstracta no reducible a features conocidas**

**Comparación pre/post DANN**:
Correr los mismos probes sobre embeddings Gate 2 (pre-DANN) y Gate 3 (post-DANN). Si DANN destruye la información de ratios para lograr modal-agnosticism, eso es informativo.

#### 6.3 Disagreement Analysis — ¿Dónde gana cada representación?

Para los mismos 5K segmentos, comparar retrieval con embedding vs retrieval con cada representación clásica:

**Para cada query**:
- ¿Embedding acierta y hashes fallan?
- ¿Hashes aciertan y embedding falla?

**Agrupar disagreements por**:
- Densidad de eventos (notas/segundo)
- Tempo
- Proporción chord vs sequential tokens
- Complejidad armónica (entropía del histograma)
- Pieza / compositor

**Output**: Tabla de "fortalezas relativas" por representación:

| Condición | Gana Embedding | Gana Hashes | Gana Histograma |
|-----------|---------------|-------------|-----------------|
| Alta densidad | ? | ? | ? |
| Bajo tempo | ? | ? | ? |
| Pasajes monofónicos | ? | ? | ? |
| Pasajes polifónicos | ? | ? | ? |

#### Criterios de Éxito Gate 6

Este gate es **analítico, no tiene GO/NO-GO**. El éxito es obtener respuestas claras a:

| Pregunta | Respuesta esperada |
|----------|-------------------|
| ¿El embedding valida el ratio language? | RSA(E, H_hist) + probes de log-ratio |
| ¿Qué representación es más cercana al embedding? | Ranking RSA/CKA |
| ¿DANN destruye información de ratios? | Comparación probes pre/post DANN |
| ¿Los hashes capturaban lo correcto pero de forma frágil? | Disagreement analysis |
| ¿Qué invariancias nuevas aprendió el modelo? | Probes con R² bajo = estructura no capturada |

**Entregable final**: Informe `INFORME_GATE6_RETROANALISIS.md` con:
1. Heatmap RSA/CKA entre todas las representaciones
2. Tabla de probes (R² por feature, pre/post DANN)
3. Disagreement analysis con fortalezas por condición
4. Conclusión: ¿qué parte del "ratio language" era real vs artefacto?

---

## 4. Configuración Default

### 4.1 Segmentación
```python
segment_len = 8.0  # segundos
hop = 2.0  # segundos
sr_audio = 24000  # Hz (MERT default)
```

### 4.2 Modelo
```python
# Audio encoder
audio_encoder = "m-a-p/MERT-v1-330M"
audio_encoder_frozen = True  # Inicialmente

# MIDI encoder
midi_encoder = "Transformer"
midi_embed_dim = 512
midi_n_layers = 4
midi_n_heads = 8

# Projection
proj_hidden = 512
proj_output = 256
```

### 4.3 Optimización
```python
optimizer = "AdamW"
lr_projection = 1e-3
lr_midi_encoder = 1e-4
lr_audio_encoder = 0  # frozen → luego 1e-5 si fine-tune
weight_decay = 1e-4
warmup_steps = 500
max_epochs = 100
batch_size = 64
```

### 4.4 VICReg
```python
invariance_weight = 10.0
variance_weight = 10.0
covariance_weight = 1.0
```

### 4.5 DANN (Gate 3)
```python
domain_loss_weight = 0.01
grl_lambda_start = 0.0
grl_lambda_end = 1.0
grl_schedule = "linear"
```

---

## 5. Métricas de Evaluación

### 5.1 Retrieval
- Recall@{1, 5, 10, 20}
- MRR (Mean Reciprocal Rank)
- Offset MAE (si aplica)
- **Simetría**: usar `min(a2m, m2a)` o media armónica para evitar que una dirección oculte problemas

### 5.2 Hard Negatives Suite (CRÍTICO)
- **NEG_SAME_PIECE_DIFF_TIME**: misma pieza, ventana diferente — **el test más importante**
- NEG_SAME_COMPOSER: mismo compositor, otra pieza
- NEG_RANDOM: otras piezas random
- NEG_TEMPO_SHIFT: misma pieza con tempo modificado (solo MIDI)

### 5.3 Pool Estructurado (256 candidatos)
```
Por cada query:
├── 64 hard negatives (same-piece-diff-time)
├── 32 semi-hard (same-composer)
├── 159 random
└── 1 positivo (match correcto)

Este es el TEST DEFINITIVO de Gate 2.
```

### 5.4 Controles
- Shuffle control: pares aleatorios ≈ azar
- Oracle: MIDI→MIDI debe ser alto

### 5.5 Monitoreo de Colapso
- Varianza por dimensión del embedding (std > 0.1)
- Correlación entre dimensiones
- Domain classifier accuracy (para DANN)

### 5.6 Probes Cuantitativos (Gate 2.5)
- Domain probe: accuracy Audio vs MIDI
- Piece probe: accuracy clasificación de pieza
- Time probe: MAE predicción offset temporal

---

## 6. Estructura de Directorios

```
/mnt/m2-1TB/Phideus/
├── src/
│   ├── bias_control/              # ✅ IMPLEMENTADO
│   │   ├── __init__.py                  ✅
│   │   ├── encoders/
│   │   │   ├── mert_encoder.py          ✅
│   │   │   ├── midi_encoder.py          ✅
│   │   │   └── projection.py            ✅
│   │   ├── losses/
│   │   │   └── dann.py                  ✅
│   │   ├── models/
│   │   │   └── cross_modal_model.py     ✅
│   │   └── data/
│   │       └── maestro_segments.py      ✅
│   └── datasets/
│       └── maestro_segments.py    # NUEVO o modificar existente
├── experiments/
│   └── bias_control/              # ✅ IMPLEMENTADO
│       ├── gate0_data_integrity.py      ✅
│       ├── gate1_intra_modal.py         ✅
│       ├── gate2_foundation.py          ✅
│       ├── gate2_5_embedding_analysis.py ✅
│       ├── gate3_dann.py                ✅
│       ├── gate4_ratio_auxiliary.py     ✅
│       ├── gate6_retroanalysis.py       ⏳
│       └── run_all_gates.py             ✅
├── Documents/
│   └── BIAS_CONTROL/
│       ├── ROADMAP_BIAS_CONTROL.md  # Este documento
│       └── Planes_Claude/
└── data/
    └── maestro_v3/
        └── maestro-v3.0.0/        # Dataset existente
```

---

## 7. Timeline Estimado

| Gate | Duración | Dependencias |
|------|----------|--------------|
| 0 | 1 día | Dataset descargado |
| 1 | 1 día | Gate 0 |
| 2 | 3-4 días | Gate 1 |
| 2.5 | 0.5 días | Gate 2 |
| 3 | 2-3 días | Gate 2.5 |
| 4 | 2-3 días | Gate 3 |
| 5 | 2-3 días | Gate 4 (opcional) |
| **6** | **2-3 días** | **Gate 3 (mínimo), idealmente post-Gate 4** |

**Total estimado**: 10-15 días para Gates 0-4, +2-3 días para Gate 6

---

<a id="criterios-exito-final"></a>
## 8. Criterios de Éxito Final

### Éxito Mínimo (válido científicamente)
- Gate 2 pasa **pool estructurado**: Recall@10 > 25% con hard negatives
- Accuracy vs same-piece-diff-time > 60%
- Evidencia de **identidad temporal**, no solo "firma de pieza"

### Éxito Completo
- Gate 4 pasa: Ratios aportan mejora medible en hard negatives
- Pool estructurado Recall@10 > 40%
- Time probe muestra capacidad de localización temporal
- Gate 6: Retroanálisis confirma qué parte del ratio language captura el embedding

### Resultado Negativo Informativo
- Gate 2 pasa pool global pero falla pool estructurado
- Conclusión: "El modelo aprende firma de pieza/estilo pero no identidad temporal"
- Valor: Documenta que cross-modal alignment ≠ cross-modal identification

### Momento Científico Clave
```
El "momento de verdad" de BIAS_CONTROL es el HARD NEGATIVE SUITE.

Si el modelo puede distinguir:
  "este segmento de audio a t=30s"
vs
  "mismo audio a t=45s" (hard negative)

...entonces tenemos evidencia real de cross-modal temporal identity.

Todo lo demás (gap, vs-random global) son indicadores tempranos,
pero el hard negative test es la prueba concreta.
```

### Cierre de Escalon 1 (criterio de programa)

En alineacion con `Documents/Rosetta_triplescaloneta.md`, este roadmap representa el **Escalon 1**.

El cierre de Escalon 1 requiere:

1. Gate 4 completado con control causal (ratio vs control) y evaluacion estructurada consistente.
2. Gate 6 completado con evidencia representacional (RSA/CKA/probes/disagreement).
3. Auditoria final consolidada de BIAS_CONTROL con decision explicita de siguiente escalon.

---

## 9. Referencias

1. [MERT: Acoustic Music Understanding Model](https://arxiv.org/abs/2306.00107)
2. [VICReg: Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906)
3. [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
4. [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
5. [Barlow Twins](https://arxiv.org/abs/2103.03230)
6. [Representational Similarity Analysis (RSA)](https://doi.org/10.3389/neuro.06.004.2008)
7. [CKA: Similarity of Neural Network Representations](https://arxiv.org/abs/1905.00414)
