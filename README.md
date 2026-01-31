# Phideus v5.0 - Harmonic Information Theory Research

**Estado**: Programa de investigación activo | **Última actualización**: 2026-01-30

---

## Resumen

Phideus investiga si las **relaciones armónicas (ratios de frecuencia)** constituyen un lenguaje universal que puede transferirse entre modalidades sensoriales.

### Estado Actual: Extractor v2.2 Validado

El Extractor v2.2 resuelve el problema de histogramas uniformes que causó el fallo de Rosetta1 2.0.

| Métrica | Rosetta1 2.0 | Extractor v2.2 | Mejora |
|---------|--------------|----------------|--------|
| Gap aligned-shuffled | 0.004 | **0.691** | **172×** |
| Entropía | ~0.95 | 0.51 | -46% |
| GO/NO-GO | NO-GO | **GO (36/36)** | ✅ |

### Hallazgos Principales

| Hito | Resultado | Significado |
|------|-----------|-------------|
| **Extractor v2.2** | Gap 0.691 vs 0.004 | Histogramas discriminativos (172× mejor) |
| **Analizador 5.0** | VAE val_loss: 0.456 vs 4212 (v4.1) | La representación importa más que la arquitectura |
| **Comparación 4 Arquitecturas** | VAE ≈ HRM con datos óptimos | Ambas arquitecturas son válidas |

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
│   │   ├── analizador_4.1_Enriched.py # Legacy - escala log
│   │   └── analizador_roseta.py       # Dual-domain para Roseta
│   ├── datasets/
│   │   ├── temporal_dataset_5.py      # Loader NPZ/JSON
│   │   └── roseta_dataset.py          # Loader dual-domain
│   ├── RNA/
│   │   └── roseta_vae.py              # VAE con InfoNCE loss
│   ├── hrm/                           # Hierarchical Reasoning Model
│   ├── generador/                     # Generación de WAVs sintéticos
│   └── auditor/                       # Auditoría de ratios
│
├── experiments/
│   ├── run_experiments_5.0.py         # Comparación 4 arquitecturas
│   └── run_roseta_experiment.py       # Experimento Roseta 1
│
├── Documents/
│   ├── PHIDEUS_RESEARCH_PROGRAM_2026.md  # Paper principal (47 refs)
│   ├── Proyecto_Estado_Actual.md         # Estado actual
│   ├── Analizador/
│   │   └── SPEC_ANALIZADOR_5.0.md        # Especificación técnica
│   ├── Experimentos/
│   │   ├── REPORTE_COMPARATIVO_4.1_vs_5.0.md
│   │   ├── RESULTADOS_HRM_VS_VAE_MASIVO.md
│   │   └── RESULTADOS_HRM_TRAINING.md
│   └── Roseta/
│       ├── INFORME_ROSETA_1_PARA_PUBLICACION.md
│       ├── INFORME_ROSETA_1_HARMONIC_INFORMATION_THEORY.md
│       └── PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md
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
| [PHIDEUS_RESEARCH_PROGRAM_2026.md](Documents/PHIDEUS_RESEARCH_PROGRAM_2026.md) | Paper principal del programa de investigación |
| [Proyecto_Estado_Actual.md](Documents/Proyecto_Estado_Actual.md) | Estado actual del proyecto |
| [SPEC_ANALIZADOR_5.0.md](Documents/Analizador/SPEC_ANALIZADOR_5.0.md) | Especificación técnica del analizador |

### Experimentos

| Documento | Descripción |
|-----------|-------------|
| [REPORTE_COMPARATIVO_4.1_vs_5.0.md](Documents/Experimentos/REPORTE_COMPARATIVO_4.1_vs_5.0.md) | Análisis del cambio de paradigma |
| [RESULTADOS_HRM_VS_VAE_MASIVO.md](Documents/Experimentos/RESULTADOS_HRM_VS_VAE_MASIVO.md) | Comparación HRM vs VAE (848 samples) |
| [RESULTADOS_HRM_TRAINING.md](Documents/Experimentos/RESULTADOS_HRM_TRAINING.md) | Resultados de entrenamiento HRM |

### Extractor v2.2 y Revisionismo

| Documento | Descripción |
|-----------|-------------|
| [Fase_1_results.md](Documents/Analizador/Fase_1_results.md) | **Resultados Fase 1 (sweep 36 configs)** |
| [INFORME_REVISIONISMO_EXTRACCION_RATIOS.md](Documents/Analizador/Recursos/INFORME_REVISIONISMO_EXTRACCION_RATIOS.md) | Diagnóstico y roadmap |

### Experimento Roseta

| Documento | Descripción |
|-----------|-------------|
| [INFORME_ROSETA_1_PARA_PUBLICACION.md](Documents/Roseta/INFORME_ROSETA_1_PARA_PUBLICACION.md) | Informe técnico Roseta 1 |
| [INFORME_ROSETA_1_HARMONIC_INFORMATION_THEORY.md](Documents/Roseta/INFORME_ROSETA_1_HARMONIC_INFORMATION_THEORY.md) | Marco teórico HIT |
| [PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md](Documents/Roseta/PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md) | Propuesta Roseta 2 |

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

### Fase 2: Re-entrenar Rosetta con Extractor v2.2
1. Regenerar dataset con config_002 (configuración óptima)
2. Re-entrenar RosetaVAE con histogramas discriminativos
3. Evaluar con controles negativos (aligned vs shuffled)
4. Criterio de éxito: gap del modelo > 0.15

### Configuración Óptima (config_002)
```python
extractor_params = {
    'top_k_peaks': 8,
    'min_prominence': 0.1,
    'temporal_stability_threshold': 0.7,
    'use_warped_bins': False,
}
```

### Extensiones Futuras
- Roseta 2: Audio → Visual (Lissajous)
- Más dominios sensoriales (temperatura, corriente)
- Arquitecturas híbridas HRM-VAE

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
