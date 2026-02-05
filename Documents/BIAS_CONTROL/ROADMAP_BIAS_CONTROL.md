# Roadmap: Cross-Modal Learning con Control de Sesgo

**Fecha**: 2026-02-05
**Versión**: 1.3
**Base**: Integración análisis Claude + GPT5.2Think (criterios recalibrados)
**Dataset**: MAESTRO v3.0.0 (Audio ↔ MIDI)
**Estado**: 🔄 **GATE 2 EN EJECUCIÓN** (Epoch 54/61, 1000 bat/ep, Gap: 0.478 best)

---

## 🔄 Estado Actual (2026-02-05 12:30)

### Test 1000 batches/epoch en Progreso (tmux)

| Epoch | Loss | Gap | a2m R@10 | m2a R@10 | Tendencia |
|-------|------|-----|----------|----------|-----------|
| 32 | 14.63 | 0.365 | 1.4% | 1.7% | baseline |
| 38 | 14.37 | 0.475 | 2.5% | 3.7% | ↑ peak |
| 45 | 14.22 | **0.478** | 2.5% | 2.7% | ★ best gap |
| 50 | 14.12 | 0.437 | 2.8% | 2.8% | estable |
| 53 | 14.09 | 0.388 | 2.3% | 2.7% | actual |

**ETA**: ~4 horas (8 epochs × ~30 min)

**Observación**: El modelo ha plateaued en Gap ~0.4 con alta varianza (0.35-0.48). Loss sigue bajando lentamente. Recalls estables en ~2.5% (≈34× random con pool 13,532).

**Próximo paso**: Al terminar epoch 61, ejecutar **evaluación con pool estructurado** (hard negatives) para determinar GO/NO-GO real.

### Sanity Checks Completados

- ✅ Alineación Audio-MIDI: 30-50ms (excelente)
- ✅ Segmentos válidos: 127,092
- ✅ Fórmula de recall: correcta
- ✅ No hay bugs críticos en pipeline

### Nota sobre Recalibración de Criterios (v1.3)

Los criterios originales de Gate 2 (Recall@10 > 20%) estaban mal calibrados para un pool de 13,532 segmentos. Con ese tamaño, random baseline = 0.074%. Ver sección Gate 2 para criterios corregidos basados en:
1. Pool global (vs random)
2. Pool estructurado con hard negatives (test definitivo)

---

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

### Gate 4 — Híbrido con Ratios como Vista Auxiliar

**Objetivo**: Reinyectar el "ratio insight" de forma compatible con aprendizaje.
**Script**: `experiments/bias_control/gate4_ratio_auxiliary.py` ✅

**Tareas**:
- [x] Implementar ratio encoder pequeño (MLP sobre histogramas soft)
- [x] Computar ratios en ambos dominios (sin hashing)
- [x] Añadir losses multi-view:
  - VICReg(Audio, Ratio)
  - VICReg(MIDI, Ratio)
  - Opcional: predict(histogram_ratio) desde embeddings

**Configuración Ratio-Aux**:
```python
ratio_loss_weight = 0.05  # Empezar bajo
ratio_encoder = MLP(256_bins * 3_channels, 128, 64)
```

**Criterios GO**:
| Métrica | vs Gate 3 |
|---------|-----------|
| Gap vs same-piece-diff-time | Mejora |
| Offset MAE | Reduce |
| Recall@10 | No empeora |

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

**Total estimado**: 10-15 días para Gates 0-4

---

## 8. Criterios de Éxito Final

### Éxito Mínimo (válido científicamente)
- Gate 2 pasa **pool estructurado**: Recall@10 > 25% con hard negatives
- Accuracy vs same-piece-diff-time > 60%
- Evidencia de **identidad temporal**, no solo "firma de pieza"

### Éxito Completo
- Gate 4 pasa: Ratios aportan mejora medible en hard negatives
- Pool estructurado Recall@10 > 40%
- Time probe muestra capacidad de localización temporal

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

---

## 9. Referencias

1. [MERT: Acoustic Music Understanding Model](https://arxiv.org/abs/2306.00107)
2. [VICReg: Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906)
3. [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
4. [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
5. [Barlow Twins](https://arxiv.org/abs/2103.03230)
