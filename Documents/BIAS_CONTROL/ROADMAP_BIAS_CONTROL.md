# Roadmap: Cross-Modal Learning con Control de Sesgo

**Fecha**: 2026-02-04
**Versión**: 1.2
**Base**: Integración análisis Claude + GPT5.2Think
**Dataset**: MAESTRO v3.0.0 (Audio ↔ MIDI)
**Estado**: 🔄 **MEDIUM TEST EN EJECUCIÓN** (Epoch 11/30, Gap: 0.398, tmux)

---

## 🔄 Estado Actual (2026-02-04)

### Medium Test en Progreso (migrado a tmux)

| Epoch | Loss | Gap | Tendencia |
|-------|------|-----|-----------|
| 1 | 19.59 | 0.054 | baseline |
| 3 | 15.86 | 0.175 | ↑ +224% |
| 6 | 15.35 | 0.302 | ↑ peak |
| 10 | 15.18 | **0.398** | ↑ **15.3× baseline** |
| 11+ | ... | ... | (en tmux) |

**Señal MUY positiva**: Gap 0.398 supera criterio GO (>0.10) por 4×. Migrado a tmux con `--checkpoint-every 1`.

### Sanity Checks Completados

- ✅ Alineación Audio-MIDI: 30-50ms (excelente)
- ✅ Segmentos válidos: 127,092
- ✅ Fórmula de recall: correcta
- ✅ No hay bugs críticos en pipeline

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

**Configuración VICReg** (conservadora):
```python
invariance_weight = 10.0
variance_weight = 10.0
covariance_weight = 1.0
```

**Criterios GO**:
| Métrica | Umbral GO | Umbral STRONG GO |
|---------|-----------|------------------|
| Audio→MIDI Recall@10 | > 20% | > 40% |
| MIDI→Audio Recall@10 | > 20% | > 40% |
| vs Random | > 5x | > 10x |
| Gap aligned vs same-piece-diff-time | > 0.1 | > 0.2 |

**NO-GO si**: Recall@10 < 2x random → revisar encoders o datos.

---

### Gate 2.5 — Análisis de Embeddings (Diagnóstico)

**Objetivo**: Entender qué aprendió el modelo antes de añadir DANN.
**Script**: `experiments/bias_control/gate2_5_embedding_analysis.py` ✅

**Tareas**:
- [x] Extraer embeddings de validación (500+ segmentos)
- [x] t-SNE/UMAP visualización
- [x] Medir separabilidad por modalidad (Audio vs MIDI)
- [x] Medir separabilidad por pieza
- [x] Detectar colapso (varianza por dimensión)

**Interpretación**:
| Observación | Diagnóstico | Acción |
|-------------|-------------|--------|
| Clusters por modalidad | Modal shortcut | → Gate 3 (DANN) |
| Clusters por pieza | Señal útil | → Quizás skip Gate 3 |
| Colapso total | Problema de loss | → Revisar VICReg weights |
| Mixto | Normal | → Gate 3 |

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

### 5.2 Hard Negatives Suite
- NEG_RANDOM: otras piezas random
- NEG_SAME_PIECE_DIFF_TIME: misma pieza, ventana diferente
- NEG_SAME_COMPOSER: mismo compositor, otra pieza
- NEG_TEMPO_SHIFT: misma pieza con tempo modificado (solo MIDI)

### 5.3 Controles
- Shuffle control: pares aleatorios ≈ azar
- Oracle: MIDI→MIDI debe ser alto

### 5.4 Monitoreo de Colapso
- Varianza por dimensión del embedding
- Correlación entre dimensiones
- Domain classifier accuracy (para DANN)

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
- Gate 2 pasa: Cross-modal retrieval > 5x random
- Negativos duros separados de positivos

### Éxito Completo
- Gate 4 pasa: Ratios aportan mejora medible
- Retrieval > 10x random
- Gap vs hard negatives > 0.2

### Resultado Negativo Informativo
- Gate 2 falla después de debugging exhaustivo
- Conclusión: "Cross-modal Audio↔MIDI requiere más que alignment de embeddings"
- Valor: Documenta límites del enfoque

---

## 9. Referencias

1. [MERT: Acoustic Music Understanding Model](https://arxiv.org/abs/2306.00107)
2. [VICReg: Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906)
3. [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
4. [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
5. [Barlow Twins](https://arxiv.org/abs/2103.03230)
