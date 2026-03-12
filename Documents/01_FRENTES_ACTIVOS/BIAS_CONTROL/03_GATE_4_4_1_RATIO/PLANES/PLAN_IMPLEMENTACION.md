# Plan de Implementación: BIAS_CONTROL

**Fecha**: 2026-02-04
**Estado**: ✅ IMPLEMENTACIÓN COMPLETADA
**Commit**: `f501039`

> [!NOTE]
> Addendum de vigencia (2026-02-17): este plan describe la implementación base histórica del frente.
> El ciclo activo avanzó por Bloque A, cerró Gate 4.2 y hoy ejecuta Gate 4.3 (`D0`/`D4` completados, `A4` en curso).
> Estado vigente en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`.

---

## Resumen Ejecutivo

Sistema de cross-modal learning Audio ↔ MIDI usando embeddings aprendidos con soft matching, abandonando el enfoque de hash matching exacto.

**Cambio de paradigma**:
- ❌ OLD: Tokens Audio = Tokens MIDI (hash identity)
- ✅ NEW: Distancia en embedding space < threshold (soft matching)

---

## Arquitectura Implementada

```
   AUDIO                              MIDI
     │                                  │
     ▼                                  ▼
┌──────────┐                    ┌──────────┐
│   MERT   │                    │   MIDI   │
│ (frozen) │                    │ Encoder  │
│  330M    │                    │(Transf.) │
└────┬─────┘                    └────┬─────┘
     │ [B, T, 1024]                  │ [B, T, 512]
     ▼                               ▼
┌──────────┐                    ┌──────────┐
│Projection│                    │Projection│
│  Head    │                    │  Head    │
│ (MLP)    │                    │ (MLP)    │
└────┬─────┘                    └────┬─────┘
     │ [B, 256]                      │ [B, 256]
     └───────────┬───────────────────┘
                 │
                 ▼
          ┌────────────┐      ┌────────────┐
          │  VICReg    │      │   DANN     │
          │   Loss     │      │ (Gate 3)   │
          └────────────┘      └────────────┘
```

---

## Estructura de Archivos Implementados

### Módulo Principal: `src/bias_control/`

```
src/bias_control/
├── __init__.py                      # 26 líneas
├── encoders/
│   ├── __init__.py                  # 7 líneas
│   ├── mert_encoder.py              # 283 líneas - MERT + MERTEncoderLite
│   ├── midi_encoder.py              # 393 líneas - Transformer MIDI
│   └── projection.py                # 258 líneas - ProjectionHead variantes
├── losses/
│   ├── __init__.py                  # 5 líneas
│   └── dann.py                      # 348 líneas - DANN + GRL
├── architectures/
│   ├── __init__.py                  # 5 líneas
│   └── cross_modal_model.py         # 421 líneas - CrossModalModel
└── datasets/
    ├── __init__.py                  # 5 líneas
    └── maestro_segments.py          # 409 líneas - MaestroSegmentDataset
```

### Experimentos: `experiments/bias_control/`

```
experiments/bias_control/
├── gate0_data_integrity.py          # 439 líneas
├── gate1_intra_modal.py             # 442 líneas
├── gate2_foundation.py              # 567 líneas
├── gate2_5_embedding_analysis.py    # 595 líneas
├── gate3_dann.py                    # 584 líneas
├── gate4_ratio_auxiliary.py         # 669 líneas
└── run_all_gates.py                 # 415 líneas
```

**Total**: ~6,300 líneas de código en 19 archivos

---

## Componentes Clave

### 1. Encoders

#### MERTEncoder (`mert_encoder.py`)
- Wrapper para modelo MERT de HuggingFace (`m-a-p/MERT-v1-330M`)
- Lazy loading para eficiencia
- Soporte para freeze/unfreeze
- Agregación: mean, cls, last
- `MERTEncoderLite`: versión simplificada para testing rápido

#### MIDIEncoder (`midi_encoder.py`)
- Transformer encoder para secuencias MIDI
- Event embedding: pitch + velocity + duration
- Positional encoding sinusoidal
- Agregación: mean, cls, attention pooling
- Funciones auxiliares: `duration_to_bucket()`, `piano_roll_to_events()`

#### ProjectionHead (`projection.py`)
- MLP projection head estándar (SimCLR/VICReg style)
- `DualProjectionHead`: para retrieval + auxiliary tasks
- `ExpanderProjection`: VICReg expander (8192d → 256d)

### 2. Losses

#### DANNLoss (`dann.py`)
- `GradientReversalLayer`: Reversa gradientes con factor λ
- `DomainClassifier`: MLP para clasificar modalidad
- `DANNLoss`: Módulo completo con scheduling de λ
- `ConditionalDANNLoss`: Variante condicionada por pieza

### 3. Modelo Principal

#### CrossModalModel (`cross_modal_model.py`)
- Integra audio encoder, MIDI encoder, projections
- `compute_vicreg_loss()`: Invariance + Variance + Covariance
- `compute_loss()`: VICReg + DANN opcional
- `compute_retrieval_metrics()`: Recall@K, MRR, gap

### 4. Dataset

#### MaestroSegmentDataset (`maestro_segments.py`)
- Carga MAESTRO v3.0.0 con metadata JSON
- Segmentación configurable (8s segments, 2s hop)
- Lazy loading de audio (librosa) y MIDI (pretty_midi)
- Soporte para shuffled pairs (control negativo)
- `collate_segments()`: Padding de secuencias MIDI
- `create_dataloaders()`: Factory para train/val/test

---

## Gates Implementados

### Gate 0: Data Integrity
**Script**: `gate0_data_integrity.py`
- Verifica alignment audio-MIDI (drift < 100ms)
- Cuenta segmentos válidos (> 10,000)
- Control negativo con shuffled pairs
- Genera metadata de segmentos

### Gate 1: Intra-Modal Baselines
**Script**: `gate1_intra_modal.py`
- Audio→Audio retrieval con MERT
- MIDI→MIDI retrieval con Transformer
- Métricas: Recall@K, MRR, gap

### Gate 2: Cross-Modal Foundation
**Script**: `gate2_foundation.py`
- Training loop completo con VICReg
- Warmup + cosine annealing LR
- Checkpointing (best + periodic)
- Evaluación de retrieval bidireccional

### Gate 2.5: Embedding Analysis
**Script**: `gate2_5_embedding_analysis.py`
- t-SNE/UMAP visualización
- Análisis de separabilidad modal vs pieza
- Detección de colapso (varianza por dimensión)
- Recomendaciones automáticas

### Gate 3: DANN
**Script**: `gate3_dann.py`
- Fine-tuning con DANN loss
- Scheduling de λ (linear 0→1)
- Monitor: domain accuracy → 50%
- Carga checkpoint de Gate 2

### Gate 4: Ratio Auxiliary
**Script**: `gate4_ratio_auxiliary.py`
- `RatioEncoder`: MLP sobre histogramas soft
- `MultiViewModel`: Audio + MIDI + Ratios
- `compute_ratio_histogram()`: Soft binning
- Análisis time discrimination

### Orquestador
**Script**: `run_all_gates.py`
- Ejecuta gates en secuencia
- Maneja dependencias y checkpoints
- Soporta ejecución parcial (--gates)
- Genera resumen final

---

## Configuración Default

```python
# Segmentación
segment_len = 8.0  # segundos
hop = 2.0  # segundos
sr_audio = 24000  # Hz (MERT default)

# Modelo
audio_encoder = "m-a-p/MERT-v1-330M"
audio_encoder_frozen = True
midi_embed_dim = 512
midi_n_layers = 4
midi_n_heads = 8
proj_hidden = 512
proj_output = 256

# Optimización
optimizer = "AdamW"
lr_projection = 1e-3
lr_midi_encoder = 1e-4
weight_decay = 1e-4
warmup_steps = 500
max_epochs = 100
batch_size = 64

# VICReg
invariance_weight = 10.0
variance_weight = 10.0
covariance_weight = 1.0

# DANN
domain_loss_weight = 0.01
grl_lambda_schedule = "linear_0_to_1"
```

---

## Criterios GO/NO-GO

| Gate | Criterio | Umbral GO |
|------|----------|-----------|
| 0 | Drift < 100ms | > 95% piezas |
| 0 | Segmentos válidos | > 10,000 |
| 1 | Audio→Audio Recall@10 | > 50% |
| 1 | MIDI→MIDI Recall@10 | > 50% |
| 2 | Cross-modal Recall@10 | > 20% |
| 2 | vs Random | > 5x |
| 3 | Domain accuracy | 50% ± 5% |
| 3 | Recall@10 | ≥ Gate 2 |
| 4 | Recall@10 | ≥ Gate 3 |

---

## Comandos de Ejecución

```bash
cd <repo-root>
source venv/bin/activate

# Dependencias
pip install transformers pretty_midi

# Pipeline completo
python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control

# Gates individuales
python experiments/bias_control/gate0_data_integrity.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control/segments

python experiments/bias_control/gate2_foundation.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/training_outputs/bias_control/gate2 \
    --epochs 100 --batch-size 64
```

---

## Dependencias

```
torch >= 2.0
transformers >= 4.30  # Para MERT
pretty_midi >= 0.2.10
librosa >= 0.10
numpy
tqdm
scikit-learn  # Opcional: para t-SNE, silhouette
umap-learn    # Opcional: para UMAP
matplotlib    # Opcional: para visualizaciones
```

---

## Próximos Pasos

1. **Auditoría de implementación** (antes de ejecutar)
2. **Instalar dependencias** faltantes
3. **Ejecutar Gate 0** para verificar datos
4. **Ejecutar pipeline completo**
5. **Documentar resultados** por gate

---

*Documento generado: 2026-02-04*
