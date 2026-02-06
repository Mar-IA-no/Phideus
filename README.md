# Phideus v5.0 - Harmonic Information Theory Research

**Estado**: Programa de investigación activo | **Última actualización**: 2026-02-05

---

## Resumen

Phideus investiga si las **relaciones armónicas (ratios de frecuencia)** constituyen un lenguaje universal que puede transferirse entre modalidades sensoriales.

### Estado de Hipótesis (Febrero 2026)

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| **H1: Estructura** | ✅ VALIDADA | Distribuciones de ratios no aleatorias |
| **H2: Aprendibilidad** | ✅ VALIDADA | VAE/HRM val_loss < 0.5 |
| **H3: Cross-modality** | 🟢 **PROMETEDOR** | BIAS_CONTROL Gap: 0.478, Hard neg acc: 80.4% |

### Experimento Actual: BIAS_CONTROL - Gate 3 (DANN) - Epoch 8/30

Enfoque de **soft matching con embeddings** (VICReg + MERT + MIDI Transformer + DANN) sobre dataset MAESTRO.

**Gate 2 completado - GO**:

| Métrica | Valor | Umbral GO | Status |
|---------|-------|-----------|--------|
| Gap (aligned - random) | **0.478** | > 0.15 | ✅ PASS (3.2x) |
| Recall@10 (pool 256, hard neg) | **34.4%** | > 25% | ✅ PASS (1.4x) |
| Hard Negative Accuracy | **80.4%** | > 60% | ✅ PASS (1.3x) |
| Domain Probe | **92.7%** | Diagnóstico | ⚠️ Modal shortcut |

**Gate 3 (DANN) epoch 8/30** — Domain Adversarial training para forzar embeddings modal-agnostic:

| Métrica | Gate 2 | Gate 3 (best, ep7) | Tendencia |
|---------|--------|---------------------|-----------|
| Domain accuracy | 92.7% | **62.7%** | ↓ Bajando (objetivo ~50%) |
| R@10 (global) | 2.6% | **6.3%** | ↑ Mejorando |
| Loss | 14.09 | **13.99** | ↓ Convergiendo |

---

## Concepto Central

Los paisajes sonoros contienen relaciones de frecuencia significativas (3:2, 5:4, φ, √2). Phideus detecta y aprende estos patrones usando representaciones físicas puras, evitando sesgos musicales temperados.

**Hipótesis Central**: Los ratios armónicos son unidades fundamentales de información que se preservan across modalidades sensoriales (audio, vibración, MIDI, etc.).

---

## Hallazgos Principales

| Hito | Resultado | Significado |
|------|-----------|-------------|
| **BIAS_CONTROL** | Gap 0.478, 34× random | Primera señal real de cross-modality |
| **Analizador 5.0** | VAE = HRM (val_loss ~0.46) | Representación > Arquitectura |
| **Extractor v2.2** | Gap pre-red: 0.691 | Histogramas pueden ser discriminativos |
| **UOEMD** | NO-GO (128 muestras) | Dataset pequeño insuficiente para H3 |

---

## Estructura del Repositorio

```
Phideus/
├── src/
│   ├── analizador/
│   │   ├── analizador_5.0.py          # Principal - escala lineal + temporal
│   │   └── analizador_roseta.py       # Dual-domain con estabilidad temporal
│   ├── bias_control/                  # Módulo BIAS_CONTROL
│   │   ├── encoders/                  # MERT, MIDI Transformer, Projections
│   │   ├── losses/                    # VICReg, DANN
│   │   └── models/                    # CrossModalModel
│   ├── extractors/                    # Route A/B (Escalón 1)
│   ├── datasets/                      # Data loaders
│   ├── RNA/                           # VAE, JEPA, VICReg, Barlow
│   └── hrm/                           # Hierarchical Reasoning Model
│
├── experiments/
│   ├── bias_control/                  # Gates 0-6 de BIAS_CONTROL
│   ├── un_audio_un_midi/              # Scripts Escalón 1 (pausado)
│   └── *.py                           # Experimentos generales
│
├── Documents/
│   ├── INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md  # Historia completa
│   ├── INDICE_DOCUMENTACION.md        # Índice de documentación
│   ├── Proyecto_Estado_Actual.md      # Estado actual
│   ├── BIAS_CONTROL/                  # Documentación BIAS_CONTROL
│   ├── ESCALON_1/                     # Documentación Escalón 1
│   ├── UOEMD/                         # Documentación UOEMD (histórico)
│   └── Experimentos/                  # Reportes de experimentos
│
└── data/                              # Datasets (no en git)
```

---

## Quick Start

### 1. Configurar Entorno

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generar Dataset (Analizador 5.0)

```bash
python src/analizador/analizador_5.0.py \
    --input-dir train/synthetic_dataset_500 \
    --output data/datasets/temporal_5.0.npz \
    --format npz --workers 14
```

### 3. Ejecutar BIAS_CONTROL

```bash
python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control
```

---

## Documentación

### Documentos Principales

| Documento | Descripción |
|-----------|-------------|
| [INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md](Documents/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md) | **Historia completa** de representaciones de ratios |
| [INDICE_DOCUMENTACION.md](Documents/INDICE_DOCUMENTACION.md) | Índice de toda la documentación |
| [Proyecto_Estado_Actual.md](Documents/Proyecto_Estado_Actual.md) | Estado actual del proyecto |

### Por Experimento

| Experimento | Documentación | Estado |
|-------------|---------------|--------|
| BIAS_CONTROL | [ROADMAP_BIAS_CONTROL.md](Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) | 🔄 Gate 3 DANN epoch 8/30 |
| Escalón 1 | [Plan_implementacion.md](Documents/ESCALON_1/Plan_implementacion.md) | ⏸️ Pausado |
| UOEMD | [ROADMAP.md](Documents/UOEMD/UOEMD_Revisionismo/ROADMAP.md) | 🔴 NO-GO |

### Resultados Técnicos

| Documento | Descripción |
|-----------|-------------|
| [REPORTE_COMPARATIVO_4.1_vs_5.0.md](Documents/Experimentos/REPORTE_COMPARATIVO_4.1_vs_5.0.md) | Cambio de paradigma v4.1 → v5.0 |
| [RESULTADOS_HRM_VS_VAE_MASIVO.md](Documents/Experimentos/RESULTADOS_HRM_VS_VAE_MASIVO.md) | Comparación HRM vs VAE |

---

## Hipótesis de Investigación

### H1: Estructura de Ratios ✅ Validada
Las señales naturales contienen distribuciones de ratios armónicos estructuradas y no aleatorias.

### H2: Aprendibilidad ✅ Validada
Redes neuronales pueden aprender representaciones compactas de estas distribuciones (val_loss < 0.5).

### H3: Cross-modality 🟢 Prometedor
Las representaciones aprendidas en un dominio se alinean con las de otro dominio cuando ambos capturan el mismo fenómeno físico.

**Estado actual**: BIAS_CONTROL Gate 3 (DANN) en epoch 8/30. Domain accuracy bajando de 92.7% → 62.7% (objetivo ~50%). R@10 mejorando de 2.6% → 6.3%. Nuevo best model en epoch 7.

---

## Arquitectura BIAS_CONTROL

```
Audio (waveform) → MERT (frozen, 330M) → Projection → Embedding (256d)
MIDI (piano-roll) → Transformer (4L, 8H) → Projection → Embedding (256d)
                              ↓
              VICReg Loss + DANN (Gradient Reversal Layer)
```

---

## Licencia

MIT License - Ver [LICENSE.md](LICENSE.md)

---

*"El bosque ya canta. Nuestra tarea es entender su afinación."*

**Los ratios armónicos son el lenguaje universal de la naturaleza.**
