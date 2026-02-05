# Phideus v5.0 - Harmonic Information Theory Research

**Estado**: Programa de investigación activo | **Última actualización**: 2026-02-05

---

## Resumen

Phideus investiga si las **relaciones armónicas (ratios de frecuencia)** constituyen un lenguaje universal que puede transferirse entre modalidades sensoriales.

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| **H1: Estructura** | ✅ VALIDADA | Distribuciones de ratios no aleatorias |
| **H2: Aprendibilidad** | ✅ VALIDADA | VAE/HRM val_loss < 0.5 |
| **H3: Cross-modality** | 🟡 **EN EVALUACIÓN** | BIAS_CONTROL Gap: 0.478 (prometedor) |

### Estado Actual: BIAS_CONTROL Medium Test

**BIAS_CONTROL en ejecución**: Enfoque de soft matching con embeddings (VICReg + MERT + MIDI encoder).

| Métrica | Valor | Umbral GO | Status |
|---------|-------|-----------|--------|
| Gap (aligned - random) | **0.478** | > 0.15 | ✅ PASS |
| vs Random | 34× | > 10× | ✅ PASS |
| No collapse (std) | ~0.35 | > 0.1 | ✅ PASS |

**Pendiente**: Evaluación con **pool estructurado** (hard negatives) — test definitivo.

### Hallazgos Principales

| Hito | Resultado | Significado |
|------|-----------|-------------|
| **BIAS_CONTROL (activo)** | Gap 0.478, 34× random | Señal prometedora, pendiente hard neg |
| **Escalón 1 (pausado)** | 27% accuracy, 5.4× random | Rendimientos decrecientes |
| **UOEMD (NO-GO)** | Gap post-red: 0.007 | Dataset insuficiente para H3 |
| **Extractor v2.2** | Gap pre-red: 0.691 | Histogramas discriminativos (172× mejor) |

---

## Concepto Central

Los paisajes sonoros contienen relaciones de frecuencia significativas (3:2, 5:4, φ, √2). Phideus detecta y aprende estos patrones usando representaciones físicas puras, evitando sesgos musicales temperados.

**Hipótesis Central**: Los ratios armónicos son unidades fundamentales de información que se preservan across modalidades sensoriales (audio, vibración, temperatura, etc.).

---

## Resultados Experimentales

### Extractor v2.2: Sweep de Configuraciones (Enero 2026)

El Extractor v2.2 implementa filtrado de picos por prominencia y estabilidad temporal:

| Configuración | Parámetros | Score | Gap |
|---------------|------------|-------|-----|
| **config_002** (óptima) | K=8, prom=0.1, stab=0.7 | **0.621** | 0.691 |
| config_014 | K=12, prom=0.1, stab=0.7 | 0.617 | 0.694 |
| config_026 | K=16, prom=0.1, stab=0.7 | 0.612 | 0.688 |

**Conclusión**: La estabilidad temporal (0.7) es la mejora más crítica. Las 36/36 configuraciones pasan GO/NO-GO.

### Rosetta1 2.0: Diagnóstico (Enero 2026)

El experimento Rosetta1 2.0 reveló que el problema era el extractor, no la arquitectura:

| Criterio | Resultado | Estado |
|----------|-----------|--------|
| aligned vs shuffled | Δ = 0.004 | ❌ Indistinguible de random |
| Retrieval Top-1 | 0.78% | ❌ Equivalente a azar |
| **Diagnóstico** | Histogramas uniformes | ⚠️ Problema en extractor |

**Solución**: Extractor v2.2 con filtrado temporal → Gap 172× mejor.

### Comparación 4 Arquitecturas (Analizador 5.0)

| Rank | Arquitectura | Val Loss | Parámetros |
|------|--------------|----------|------------|
| 1 | VAE Temporal | **0.4560** | 1.82M |
| 2 | HRM Temporal | 0.4607 | 2.27M |
| 3 | HRM Estático | 0.5906 | 0.85M |
| 4 | VAE Estático | 0.5997 | 0.84M |

### Cambio de Paradigma: Analizador 4.1 → 5.0

| Métrica | v4.1 | v5.0 | Cambio |
|---------|------|------|--------|
| HRM val_loss | 2.74 | 0.4607 | **-83%** |
| VAE val_loss | 4212.58 | 0.4560 | **-99.99%** |
| Ventaja HRM | 153,500% | -1% | **VAE ahora gana** |

---

## Estructura del Repositorio

```
Phideus/
├── src/
│   ├── analizador/
│   │   ├── analizador_5.0.py          # Principal - escala lineal + temporal
│   │   ├── analizador_roseta.py       # Dual-domain para Roseta (v2.2)
│   │   ├── analizador_maestro.py      # ★ NUEVO: Extractor MAESTRO
│   │   └── analizador_4.1_Enriched.py # Legacy - escala log
│   ├── datasets/
│   │   ├── temporal_dataset_5.py      # Loader NPZ/JSON
│   │   ├── roseta_dataset.py          # Loader dual-domain
│   │   └── maestro_dataset.py         # ★ NUEVO: Loader MAESTRO
│   ├── RNA/
│   │   ├── roseta_vae.py              # VAE con InfoNCE loss
│   │   ├── constellation_vae.py       # ConstellationVAE modular
│   │   ├── jepa_lite.py               # JEPA-lite sin decoder
│   │   ├── vicreg.py                  # ★ NUEVO: VICReg loss
│   │   └── barlow_twins.py            # ★ NUEVO: Barlow Twins loss
│   ├── utils/
│   │   └── midi_utils.py              # ★ NUEVO: Parseo MIDI
│   ├── hrm/                           # Hierarchical Reasoning Model
│   ├── generador/                     # Generación de WAVs sintéticos
│   └── auditor/                       # Auditoría de ratios
│
├── experiments/
│   ├── maestro/                       # ★ NUEVO: Experimento MAESTRO
│   │   ├── gate0_harness.py           # Métricas + controles negativos
│   │   ├── gate1_ingest.py            # Descarga + segmentación
│   │   ├── gate2_baselines.py         # Chroma + CCA baselines
│   │   ├── gate3_cross_modal.py       # Training VICReg/Barlow
│   │   ├── gate4_ratio_tokens.py      # Training constellation
│   │   ├── gate5_moco.py              # MoCo + hard negatives
│   │   └── run_maestro_experiment.py  # Script orquestador
│   ├── run_experiments_5.0.py         # Comparación 4 arquitecturas
│   ├── run_roseta_experiment.py       # Experimento Roseta
│   ├── evaluate_cross_reconstruction.py  # Evaluación con controles
│   ├── evaluate_retrieval.py          # Retrieval metrics
│   └── evaluate_regime_separation.py  # Separation metrics
│
├── Documents/
│   ├── ESCALON_1/                     # ★ NUEVO: Plan MAESTRO
│   │   ├── Plan_implementacion.md
│   │   └── AUDITORIA_IMPLEMENTACION.md
│   ├── UOEMD/                         # Documentación UOEMD (NO-GO)
│   ├── Proyecto_Estado_Actual.md      # Estado actual del proyecto
│   ├── bitacora_desarrollo.md         # Log de desarrollo
│   └── Legacy/                        # Documentación histórica
│
├── config/                            # Configuraciones
└── data/                              # Datasets y outputs (no en git)
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

### 3. Ejecutar Comparación de Arquitecturas

```bash
python experiments/run_experiments_5.0.py \
    --data data/datasets/temporal_5.0_full.npz \
    --output data/training_outputs/experiments_5.0 \
    --epochs 50 --batch-size 32
```

### 4. Ejecutar Experimento Roseta

```bash
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_full.npz \
    --output data/training_outputs/roseta \
    --epochs 100
```

---

## Componentes Principales

### Analizador 5.0

Pipeline estándar para convertir señales 1D en secuencias temporales de histogramas de ratios:

- **Escala lineal** para ratios (no log₂)
- **Datos temporales** [T, B, 3] por archivo
- **3 canales**: proporción, momento, entropía
- **Formato binario** NPZ (12x más eficiente)

### RosetaVAE

VAE dual-encoder para alineación cross-modal:

- **Arquitectura**: BiLSTM encoders + factorized latent space
- **Loss**: Reconstruction + KL + InfoNCE contrastive
- **Parámetros**: 3.16M
- **Innovación**: z_shared (cross-modal) + z_private (específico)

### HRM (Hierarchical Reasoning Model)

Modelo de razonamiento jerárquico para análisis armónico:

- **Dual-timescale**: H-Module (slow) + L-Module (fast)
- **Attention**: Multi-head para dependencias de largo alcance
- **Eficiencia**: Mejor ratio rendimiento/parámetros

---

## Documentación

### Documentos Principales

| Documento | Descripción |
|-----------|-------------|
| [Proyecto_Estado_Actual.md](Documents/Proyecto_Estado_Actual.md) | Estado actual del proyecto |
| [bitacora_desarrollo.md](Documents/bitacora_desarrollo.md) | Log de desarrollo |
| [ROADMAP.md](Documents/Revisionismo/ROADMAP.md) | Roadmap del Revisionismo |

### Revisionismo de Extracción de Ratios

| Fase | Documento | Estado |
|------|-----------|--------|
| Fase 0 | [Fase_0_results.md](Documents/Revisionismo/Fase_0/Fase_0_results.md) | ✅ Completada |
| Fase 1 | [Fase_1_results.md](Documents/Revisionismo/Fase_1/Fase_1_results.md) | ✅ Completada (GO) |
| Fase 2 | [Fase_2_results.md](Documents/Revisionismo/Fase_2/Fase_2_results.md) | ✅ Completada (NO-GO) |
| Fase 3A | [Fase_3A.md](Documents/Revisionismo/Fase_3A/Fase_3A.md) | 🔄 Próxima |

### Experimentos

| Documento | Descripción |
|-----------|-------------|
| [REPORTE_COMPARATIVO_4.1_vs_5.0.md](Documents/Experimentos/REPORTE_COMPARATIVO_4.1_vs_5.0.md) | Análisis del cambio de paradigma |
| [RESULTADOS_HRM_VS_VAE_MASIVO.md](Documents/Experimentos/RESULTADOS_HRM_VS_VAE_MASIVO.md) | Comparación HRM vs VAE |

### Analizador

| Documento | Descripción |
|-----------|-------------|
| [SPEC_ANALIZADOR_5.0.md](Documents/Revisionismo/Analizador/SPEC_ANALIZADOR_5.0.md) | Especificación técnica |
| [INFORME_REVISIONISMO_EXTRACCION_RATIOS.md](Documents/Revisionismo/Analizador/INFORME_REVISIONISMO_EXTRACCION_RATIOS.md) | Diagnóstico y propuesta |

---

## Hipótesis de Investigación

### H1: Estructura de Ratios (Validada)
Las señales naturales contienen distribuciones de ratios armónicos estructuradas y no aleatorias.

### H2: Aprendibilidad (Validada)
Redes neuronales pueden aprender representaciones compactas de estas distribuciones.

### H3: Transferencia Cross-Modal (En Revisión)
Las representaciones aprendidas en un dominio se alinean con las de otro dominio cuando ambos capturan el mismo fenómeno físico.

**Estado**: Rosetta1 2.0 falló por histogramas uniformes. Extractor v2.2 muestra potencial (gap 0.69 vs 0.004). Pendiente re-evaluación con Fase 2.

---

## Próximos Pasos

### BIAS_CONTROL (EN EJECUCIÓN)

Enfoque de soft matching con embeddings para cross-modal Audio↔MIDI:

```bash
# Monitorear training actual
grep -E "^2026.*INFO.*Epoch" data/bias_control_medium/gate2_1000batches.log | tail -5

# Evaluar con pool estructurado (al terminar epoch 61)
python experiments/bias_control/evaluate_structured_pool.py \
    --model data/bias_control_medium/training_outputs/gate2/best_model.pt \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --pool-size 256 --n-hard-negatives 64 --n-semi-hard 32
```

**Criterios GO/NO-GO**:
- Pool global: Gap > 0.15, vs random > 10× ✅ PASS
- **Pool estructurado** (test definitivo): Recall@10 > 25% con hard negatives
- Gate 2.5: Probes cuantitativos (domain/piece/time)

### Siguiente: Gate 2.5 → Gate 3 (si hard negatives pasa)

Si pool estructurado pasa → probes cuantitativos → DANN (si domain leakage).

### Fallback: Publicar H1/H2

Si hard negatives falla → el modelo aprende "firma de pieza" pero no identidad temporal.
Documentar como resultado negativo informativo.

---

## Citación

```bibtex
@software{phideus2026,
  title={Phideus: Harmonic Information Theory Research},
  author={PHIDEUS Research Team},
  year={2026},
  url={https://github.com/AlterMundi/Phideus},
  note={Cross-modal alignment via harmonic ratios}
}
```

---

## Licencia

MIT License - Ver [LICENSE.md](LICENSE.md)

---

*"El bosque ya canta. Nuestra tarea es entender su afinación."*

**Los ratios armónicos son el lenguaje universal de la naturaleza.**
