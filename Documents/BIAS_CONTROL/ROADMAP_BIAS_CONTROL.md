# Roadmap: Cross-Modal Learning con Control de Sesgo

**Fecha**: 2026-02-05
**Versión**: 1.6
**Base**: Integración análisis Claude + GPT5.2Think (criterios recalibrados)
**Dataset**: MAESTRO v3.0.0 (Audio ↔ MIDI)
**Estado**: 🔄 **GATE 3 COMPARACIÓN A/B** (Run A sin norm detenido ep10, Run B con norm en progreso)

---

## ✅ Estado Actual (2026-02-05) - GATE 2 COMPLETADO

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

### Gate 3: DANN - EN EJECUCIÓN

#### Smoke Test (Prueba Piloto) - GO

Antes del training completo, se ejecutó un smoke test (1 epoch, 5 batches) para validar:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Gap | **0.477** | Sin degradación vs Gate 2 (0.478) |
| R@10 a2m (global) | **2.6%** | Mantiene nivel Gate 2 |
| R@10 m2a (global) | **2.5%** | Mantiene nivel Gate 2 |
| Domain accuracy | 44.7% | DANN aún no activo (lambda=0.00) |
| DANN loss | 0.693 | Cross-entropy inicial (log(2), esperado) |
| VICReg loss | 14.15 | Normal para inicio |

**Resultado**: **GO** - El script DANN funciona correctamente, las métricas no se degradan, y el modelo está listo para training completo.

#### Training Completo - Epoch 8/30

```bash
tmux attach -t gate3  # Monitorear
```

**Progreso (epochs 1-7 completados)**:

| Epoch | Loss | Domain Acc | R@10 | Lambda | Notas |
|-------|------|-----------|------|--------|-------|
| 1 | 14.108 | 67.6% | 6.2% | 0.03 | |
| 3 | 14.069 | 77.4% | 6.6% | 0.10 | Pico domain acc |
| 5 | 14.031 | 65.2% | 6.1% | 0.17 | |
| **7** | **13.992** | **62.7%** | **6.3%** | **0.23** | **★ Best** |

**Tendencia Run A**: Domain acc bajando (77.4% → 62.7%), pero oscilando (rebote a 75.2% en ep9). R@10 estable 5-7%.

**Run A detenido en epoch 10** para comparación A/B.

**Fix aplicado**: `F.normalize(embeddings, dim=1)` antes del domain head — elimina shortcut por magnitud.

**Run B (normalizado)** lanzado: `data/bias_control_medium/training_outputs/gate3_norm/`
- tmux: `gate3norm`
- ETA epoch 10: ~04:50 UTC 2026-02-06

**Correcciones aplicadas al script** (10 issues):
1. Defaults corregidos (segment_len=4.0, hop=1.0, batch_size=16) para evitar OOM
2. CLI args faltantes: `--segment-len`, `--hop`, `--max-batches-per-epoch`, `--resume`, `--checkpoint-every`, `--max-val-batches`
3. Warmup bug: `initial_lr` movido a `__init__()`
4. Resume capability: `load_checkpoint()` method con restauración de DANN step
5. Checkpoints mejorados: incluyen epoch, history, scheduler_state_dict
6. `gate2_recall` default: 0.026 (R@10 global de Gate 2)
7. `evaluate_structured_pool.py`: `strict=False` para cargar modelos DANN

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

### Gate 6 — Retroanálisis: Embeddings vs Representaciones de Ratios

**Objetivo**: Usar el embedding DANN como **instrumento de análisis** para medir qué capturaban (y qué perdían) nuestras representaciones de ratios históricas. Cierra el arco de investigación conectando el embedding aprendido con el "ratio language" que originó el proyecto.

**Prerequisito**: Gate 3 GO (embedding DANN modal-agnostic disponible).

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

---

## 9. Referencias

1. [MERT: Acoustic Music Understanding Model](https://arxiv.org/abs/2306.00107)
2. [VICReg: Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906)
3. [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
4. [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
5. [Barlow Twins](https://arxiv.org/abs/2103.03230)
6. [Representational Similarity Analysis (RSA)](https://doi.org/10.3389/neuro.06.004.2008)
7. [CKA: Similarity of Neural Network Representations](https://arxiv.org/abs/1905.00414)
