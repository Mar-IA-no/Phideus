# BIAS_CONTROL: Sistema de Control de Sesgos Cross-Modal

**Fecha**: 2026-02-04
**Branch**: `feature/extractor-v22`
**Estado**: ✅ Implementado y Auditado - Listo para Ejecución

---

## Resumen Ejecutivo

BIAS_CONTROL es un sistema experimental para aprendizaje cross-modal entre Audio y MIDI usando el dataset MAESTRO. El objetivo es demostrar que representaciones aprendidas pueden capturar correspondencias semánticas entre modalidades.

### Arquitectura

```
Audio (WAV) ─────┬──► MERT Encoder ──► Projection ──► Audio Embedding
                 │                                          │
                 │                                          ▼
                 │                                    VICReg Loss
                 │                                          ▲
                 │                                          │
MIDI ────────────┴──► Transformer ──► Projection ──► MIDI Embedding
                              │
                              ▼
                    Optional: DANN Loss
                    (domain adversarial)
```

---

## Estructura de Archivos

### Módulos Principales (`src/bias_control/`)

```
src/bias_control/
├── __init__.py                     # Exports principales
├── encoders/
│   ├── __init__.py
│   ├── mert_encoder.py             # MERT (HuggingFace) + MERTEncoderLite
│   ├── midi_encoder.py             # Transformer encoder para MIDI
│   └── projection.py               # ProjectionHead para embedding space
├── losses/
│   ├── __init__.py
│   └── dann.py                     # DANN + GradientReversalLayer
├── architectures/
│   ├── __init__.py
│   └── cross_modal_model.py        # CrossModalModel + VICReg
└── datasets/
    ├── __init__.py
    └── maestro_segments.py         # MaestroSegmentDataset + collate
```

### Scripts de Experimento (`experiments/bias_control/`)

```
experiments/bias_control/
├── gate0_data_integrity.py         # Verificación de datos
├── gate1_intra_modal.py            # Baselines intra-modal
├── gate2_foundation.py             # Training cross-modal (VICReg)
├── gate2_5_embedding_analysis.py   # Análisis de embeddings
├── gate3_dann.py                   # Training con DANN
├── gate4_ratio_auxiliary.py        # Auxiliary loss con ratios
├── run_all_gates.py                # Orquestador principal
├── audit_dependencies.py           # Auditoría de dependencias
├── audit_components.py             # Auditoría de componentes
└── audit_dataset.py                # Auditoría de dataset
```

---

## Componentes Detallados

### 1. MERTEncoder / MERTEncoderLite

**Archivo**: `src/bias_control/encoders/mert_encoder.py`

- **MERTEncoder**: Wrapper para MERT pre-entrenado de HuggingFace (m-a-p/MERT-v1-330M)
  - Input: Waveform [B, T] a 24kHz
  - Output: Embedding [B, 1024]
  - Soporte para freeze/fine-tune
  - Lazy loading del modelo

- **MERTEncoderLite**: Alternativa ligera sin HuggingFace
  - CNN + Transformer (4 layers, 8 heads)
  - Para testing y prototyping rápido
  - Position embeddings: 6000 tokens máx

### 2. MIDIEncoder

**Archivo**: `src/bias_control/encoders/midi_encoder.py`

- Transformer encoder para secuencias MIDI
- Input: pitch [B, N], velocity [B, N], duration [B, N]
- Output: Embedding [B, 512]
- Aggregation: mean, cls, o attention pooling
- Duration buckets: 32 buckets en escala logarítmica

### 3. ProjectionHead

**Archivo**: `src/bias_control/encoders/projection.py`

- MLP para proyectar a espacio compartido
- Configurable: input_dim, hidden_dim, output_dim, n_layers
- BatchNorm + ReLU entre capas

### 4. DANNLoss

**Archivo**: `src/bias_control/losses/dann.py`

- **GradientReversalLayer**: Invierte gradientes durante backprop
- **DomainClassifier**: Predice modalidad (Audio vs MIDI)
- **DANNLoss**: Loss completa con scheduling de lambda
- **ConditionalDANNLoss**: Variante condicionada por pieza

### 5. CrossModalModel

**Archivo**: `src/bias_control/architectures/cross_modal_model.py`

- Combina audio encoder + MIDI encoder + projections
- VICReg loss integrada:
  - Invariance: MSE entre pares
  - Variance: Evita colapso
  - Covariance: Decorrelación de dimensiones
- Soporte opcional para DANN

### 6. MaestroSegmentDataset

**Archivo**: `src/bias_control/datasets/maestro_segments.py`

- Carga segmentos alineados Audio-MIDI de MAESTRO v3.0.0
- Segmentación configurable (segment_len, hop)
- Soporte para shuffled pairs (control negativo)
- Lazy loading de audio/MIDI
- Collate function con padding automático

---

## Pipeline de Gates

### Gate 0: Data Integrity
- **Objetivo**: Verificar integridad de datos
- **Criterios GO**:
  - Alignment rate > 90%
  - Segments > 10,000
  - Shuffling funcionando

### Gate 1: Intra-Modal Baselines
- **Objetivo**: Establecer baselines
- **Criterios GO**:
  - Audio→Audio Recall@10 > 50%
  - MIDI→MIDI Recall@10 > 50%

### Gate 2: Cross-Modal Foundation
- **Objetivo**: Training inicial con VICReg
- **Criterios GO**:
  - Audio→MIDI Recall@10 > 20%
  - vs Random > 5x

### Gate 2.5: Embedding Analysis
- **Objetivo**: Análisis de representaciones
- **Métricas**: t-SNE, UMAP, cluster analysis

### Gate 3: DANN Training
- **Objetivo**: Domain-adversarial training
- **Criterios GO**:
  - Domain accuracy → 50% (random)
  - Recall mantiene o mejora

### Gate 4: Ratio Auxiliary
- **Objetivo**: Auxiliary loss con ratios armónicos
- **Criterios GO**:
  - Mejora vs Gate 3

---

## Dataset MAESTRO

### Estadísticas

| Métrica | Valor |
|---------|-------|
| Total Piezas | 1,276 |
| Duración Total | ~200 horas |
| Compositores | 10 |
| Años | 2004-2018 |

### Splits

| Split | Piezas | Segmentos (8s, hop=2s) |
|-------|--------|------------------------|
| Train | 962 | 92,252 |
| Validation | 137 | 12,843 |
| Test | 177 | 16,845 |
| **Total** | **1,276** | **121,940** |

### Formato JSON (v3.0.0)

MAESTRO v3.0.0 usa formato **columnar** (dict of dicts):
```python
{
    "canonical_composer": {"0": "Alban Berg", "1": "..."},
    "canonical_title": {"0": "Sonata Op. 1", "1": "..."},
    "split": {"0": "train", "1": "..."},
    "audio_filename": {"0": "2004/...", "1": "..."},
    "midi_filename": {"0": "2004/...", "1": "..."},
    "duration": {"0": 234.5, "1": ...}
}
```

---

## Bugs Corregidos en Auditoría

### 1. JSON Format Bug (CRÍTICO)
- **Problema**: Código asumía lista, pero MAESTRO usa dict columnar
- **Fix**: Conversión automática columnar → row format

### 2. Position Embedding Overflow (ALTO)
- **Problema**: max_pos_len=1000 < tokens reales (~4800)
- **Fix**: max_pos_len=6000 + interpolación fallback

### 3. Shuffle Verification (MEDIO)
- **Problema**: Comparaba piece_idx en vez de contenido MIDI
- **Fix**: Comparar tensores de pitch directamente

### 4. Alignment Tolerance (BAJO)
- **Problema**: 100ms muy estricto para duración audio vs MIDI
- **Fix**: 2000ms (diferencia es por decay, no desalineación)

---

## Comandos de Ejecución

### Ejecutar Pipeline Completo

```bash
cd <repo-root>
source venv/bin/activate

python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control \
    --gates 0 1 2 2.5 3 4 \
    --device cuda \
    --batch-size 64 \
    --num-workers 8
```

### Ejecutar Gates Individuales

```bash
# Gate 0: Data Integrity
python experiments/bias_control/gate0_data_integrity.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control/gate0

# Gate 1: Intra-modal baselines
python experiments/bias_control/gate1_intra_modal.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control/gate1 \
    --n-samples 1000

# Gate 2: Foundation training
python experiments/bias_control/gate2_foundation.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control/gate2 \
    --epochs 100 --batch-size 64

# Gate 3: DANN training
python experiments/bias_control/gate3_dann.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --checkpoint data/bias_control/gate2/best_model.pt \
    --output data/bias_control/gate3 \
    --epochs 50
```

### Ejecutar Auditorías

```bash
# Dependencias
python experiments/bias_control/audit_dependencies.py

# Componentes
python experiments/bias_control/audit_components.py

# Dataset
python experiments/bias_control/audit_dataset.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0
```

---

## Dependencias

### Críticas
- torch >= 2.0
- transformers >= 4.0
- librosa >= 0.10
- pretty_midi >= 0.2
- numpy
- tqdm

### Opcionales
- sklearn (métricas)
- umap-learn (visualización)
- matplotlib (plots)

### Instalación

```bash
pip install torch transformers librosa pretty_midi numpy tqdm
pip install scikit-learn matplotlib  # Opcionales
```

---

## Resultados Gate 0 (Auditoría)

```
============================================================
GATE 0: DATA INTEGRITY - RESULTS
============================================================

1. Audio-MIDI Alignment:
   Rate: 100.0% (threshold: 90%)
   Status: PASS
   Max drift: 1075.0 ms

2. Valid Segments:
   Total: 121,940 (threshold: 10,000)
   Status: PASS
   By split: {train: 92252, validation: 12843, test: 16845}

3. Shuffled Control:
   Shuffling working: True

============================================================
DECISION: GO
============================================================
```

---

## Próximos Pasos

1. **Gate 1**: Ejecutar baselines intra-modal
2. **Gate 2**: Training foundation con VICReg
3. **Gate 2.5**: Análisis de embeddings
4. **Gate 3**: Training con DANN
5. **Gate 4**: Auxiliary loss con ratios
6. **Evaluación Final**: Decidir GO/NO-GO para H3
