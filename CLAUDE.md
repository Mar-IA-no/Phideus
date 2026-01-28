# CLAUDE.md

Guía para Claude Code cuando trabaje con código en este repositorio.

## Resumen del Proyecto

Phideus v5.0 es un programa de investigación sobre **Harmonic Information Theory** - la hipótesis de que los ratios de frecuencia constituyen un lenguaje universal cross-modal.

**Estado (Enero 2026)**:
1. **H1 - Estructura**: ✅ VALIDADA - Las señales contienen distribuciones de ratios estructuradas
2. **H2 - Aprendibilidad**: ✅ VALIDADA - Redes neuronales pueden aprenderlas (val_loss < 0.5)
3. **H3 - Cross-modality**: ❌ NO VALIDADA - Rosetta1 2.0 demostró que aligned ≈ shuffled

## Arquitectura Actual

### Modelos Neuronales

| Modelo | Ubicación | Uso |
|--------|-----------|-----|
| **RosetaVAE** | `src/RNA/roseta_vae.py` | Cross-modal (Audio ↔ Vibración) |
| **VAE/HRM Temporal** | `experiments/run_experiments_5.0.py` | Comparación arquitecturas |
| **HRM Module** | `src/hrm/` | Hierarchical Reasoning Model |

### Resultados Clave

| Experimento | Resultado | Significado |
|-------------|-----------|-------------|
| VAE Temporal (5.0) | val_loss: 0.4560 | Mejor absoluto |
| HRM Temporal (5.0) | val_loss: 0.4607 | Mejor eficiencia |
| Rosetta1 2.0 | aligned ≈ shuffled | ❌ Cross-modal NO demostrado |

## Estructura del Repositorio

```
/root/Phideus/
├── src/
│   ├── analizador/
│   │   ├── analizador_5.0.py          # PRINCIPAL - escala lineal + temporal
│   │   ├── analizador_4.1_Enriched.py # Legacy - escala log (para referencia)
│   │   └── analizador_roseta.py       # Dual-domain para Roseta
│   ├── datasets/
│   │   ├── temporal_dataset_5.py      # Loader NPZ/JSON
│   │   └── roseta_dataset.py          # Loader dual-domain
│   ├── RNA/
│   │   └── roseta_vae.py              # VAE con InfoNCE loss
│   ├── hrm/                           # Hierarchical Reasoning Model
│   │   ├── models/                    # H-Module, L-Module, ACT
│   │   ├── training/
│   │   └── examples/
│   ├── generador/                     # Generación WAVs sintéticos
│   └── auditor/                       # Auditoría de ratios
│
├── experiments/
│   ├── run_experiments_5.0.py         # Comparación 4 arquitecturas
│   ├── run_roseta_experiment.py       # Experimento Roseta (2.0)
│   ├── freeze_baseline.py             # WP1: Congela baseline
│   ├── evaluate_cross_reconstruction.py  # Con controles negativos
│   ├── evaluate_retrieval.py          # WP4: Retrieval extendido
│   ├── evaluate_regime_separation.py  # WP5: Silhouette, AUC
│   ├── run_ablations.py               # WP6: Ablations A/B/C/D
│   └── roseta_v1_archived/            # Scripts visualización v1.0 (archivados)
│
├── Documents/
│   ├── PHIDEUS_RESEARCH_PROGRAM_2026.md  # Paper principal (47 refs)
│   ├── Proyecto_Estado_Actual.md
│   ├── bitacora_desarrollo.md
│   ├── Analizador/
│   │   └── SPEC_ANALIZADOR_5.0.md
│   ├── Experimentos/
│   │   ├── REPORTE_COMPARATIVO_4.1_vs_5.0.md
│   │   ├── RESULTADOS_HRM_VS_VAE_MASIVO.md
│   │   └── RESULTADOS_HRM_TRAINING.md
│   ├── Roseta/
│   │   ├── README.md                      # Índice de documentación Roseta
│   │   ├── ROSETTA1_2.0_IMPLEMENTATION_PLAN.md  # ★ Plan actual
│   │   ├── DIAGNOSTICO_ROSETTA1_ENERO2026.md
│   │   ├── Rosetta1_2.0_-_Roadmap_GTP5.2Pro.md
│   │   ├── Rosetta1_consistence_evaluation_GPT5.2Pro.md
│   │   └── v1.0_archived/             # Documentación v1.0 (archivada)
│   └── Legacy/                        # NO RASTREADO - histórico
│
├── config/                            # Configuraciones
├── data/                              # Datasets y outputs (NO en git)
├── train/                             # WAVs de entrenamiento (NO en git)
└── models/                            # Modelos guardados (NO en git)
```

## Comandos de Desarrollo

### Workflow Principal (Analizador 5.0)

```bash
# 1. Activar entorno
source venv/bin/activate

# 2. Generar dataset temporal
python src/analizador/analizador_5.0.py \
    --input-dir train/synthetic_dataset_500 \
    --output data/datasets/temporal_5.0.npz \
    --format npz --workers 14

# 3. Ejecutar comparación de 4 arquitecturas
python experiments/run_experiments_5.0.py \
    --data data/datasets/temporal_5.0_full.npz \
    --output data/training_outputs/experiments_5.0 \
    --epochs 50 --batch-size 32

# 4. Generar WAVs sintéticos (si es necesario)
python src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py
```

### Workflow Roseta (Cross-Modal)

```bash
# 1. Procesar dataset dual-domain
python src/analizador/analizador_roseta.py \
    --input-dir data/datasets/UOEMD/raw \
    --output data/datasets/roseta_full.npz

# 2. Ejecutar experimento Roseta
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_full.npz \
    --output data/training_outputs/roseta \
    --epochs 100
```

## Reglas de Organización

### Archivos que NUNCA se commitean
- Audio (WAV, MP3, FLAC)
- Datasets (JSON, NPZ > 1MB)
- Modelos (.pt, .pth, .h5)
- Virtual environments (venv/)
- `Documents/Legacy/`

### Carpeta Legacy - RESTRICCIÓN IMPORTANTE
**NUNCA revisar ni acceder al contenido de `Documents/Legacy/`** a menos que el usuario lo solicite explícitamente. Esta carpeta contiene documentación histórica que no es relevante para el desarrollo actual y solo debe consultarse bajo petición directa.

### Prioridades de Desarrollo

**Para Análisis de Datos**:
1. Analizador 5.0 (escala lineal + temporal)
2. Analizador 4.1 Enriched (solo legacy/referencia)

**Para Modelos**:
1. RosetaVAE (cross-modal)
2. VAE/HRM Temporal (comparación)
3. HRM módulo (investigación)

### Protocolo de Documentación

Cuando el usuario pide "actualizar documentación":
1. `README.md` - Overview del proyecto
2. `Documents/Proyecto_Estado_Actual.md` - Estado actual
3. `Documents/bitacora_desarrollo.md` - Entrada de log
4. Otros documentos relevantes según cambios

## Hallazgos Científicos

### Validados

1. **H1 - Estructura**: Las señales contienen distribuciones de ratios estructuradas
2. **H2 - Aprendibilidad**: Redes neuronales pueden aprenderlas (val_loss < 0.5)
3. **H3 - Transferencia**: ⚠️ Pendiente validación robusta (Roseta 1: cos_sim = 0.766)

### Descubrimientos Clave

- **Representación > Arquitectura**: Escala lineal + temporal habilita tanto VAE como HRM
- **VAE Rehabilitado**: De catastrófico (4212) a excelente (0.456)
- **Cross-modal funciona**: Audio ↔ Vibración comparten estructura latente (pendiente validación)

## Estado Actual: Rosetta1 2.0

**Documentación completa**: `Documents/Roseta/ROSETTA1_2.0_IMPLEMENTATION_PLAN.md`

### Scripts Nuevos (Rosetta1 2.0)

| Script | Propósito | WP |
|--------|-----------|-----|
| `freeze_baseline.py` | Congela artefactos baseline | WP1 |
| `evaluate_retrieval.py` | Retrieval global/intra/cross | WP4 |
| `evaluate_regime_separation.py` | Silhouette, AUC, Fisher | WP5 |
| `run_ablations.py` | Ablations A/B/C/D | WP6 |

### Workflow Rosetta1 2.0

```bash
# 1. Congelar baseline
python experiments/freeze_baseline.py \
    --checkpoint models/roseta_vae_best.pt \
    --data data/datasets/roseta_full.npz

# 2. Re-entrenar con fix z_private
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_full.npz \
    --output data/training_outputs/roseta_v2 \
    --beta-kl-private 0.01 \
    --dropout-shared 0.5 \
    --lambda-diff 0.1 \
    --all-data --epochs 100

# 3. Evaluación completa
python experiments/evaluate_cross_reconstruction.py \
    --model data/training_outputs/roseta_v2/best_model.pt \
    --run-all-controls

python experiments/evaluate_retrieval.py \
    --model data/training_outputs/roseta_v2/best_model.pt

python experiments/evaluate_regime_separation.py \
    --model data/training_outputs/roseta_v2/best_model.pt
```

## Próximos Pasos

### Rosetta1 2.0: COMPLETADO ✅
- ✅ Baseline congelado
- ✅ Re-entrenamiento con fix z_private (100% dataset)
- ✅ Evaluación con controles negativos
- ❌ Resultado: NO-GO (H3 no validada)

### Opciones Futuras
1. **Enfocarse en H1/H2**: Abandonar claim de cross-modality
2. **Cambiar arquitectura**: Probar sin VAE, con transformer
3. **Cambiar representación**: Los ratio-histograms pueden ser insuficientes
4. **Buscar más datos**: Datasets con pares audio-vib más diversos

---

## CONTEXTO CRÍTICO (Para recuperación post-compactación)

### Estado al 2026-01-28
**Rosetta1 2.0**: EJECUTADO - Resultado **NO-GO**

### Resultado Principal
El modelo **NO demuestra cross-modality real**:
- aligned vs shuffled: Δcorr = 0.002 (necesario > 0.15)
- Retrieval Top-1: 0.78% (= random)
- El modelo genera "histograma promedio", no aprende correspondencia

### Qué se ejecutó
1. Baseline congelado: `artifacts/baseline/`
2. Entrenamiento full (128 archivos): `data/training_outputs/roseta_v2_full/`
3. Evaluación con controles negativos: aligned ≈ shuffled ❌

### Criterios Go/No-Go (Resultados)
| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| var(z_private) | > 0.1 | ~0 | ❌ FAIL |
| z_priv diff | > 0.5 | 0.61 | ✅ PASS |
| Cross-recon | > 0.75 | 0.70 | ⚠️ CLOSE |
| **aligned >> shuffled** | **Δ > 0.15** | **0.002** | ❌ **CRITICAL** |
| Retrieval Top-1 | > 15% | 0.78% | ❌ FAIL |

### Lecciones Aprendidas
1. cos_sim alto no garantiza cross-modality
2. Controles negativos detectaron el problema
3. El problema no es cantidad de datos, es arquitectura/representación

### Documentación
- `Documents/Roseta/ROSETTA1_2.0_RESULTADOS_FULL.md` - Resultados finales
- `Documents/Proyecto_Estado_Actual.md` - Estado del proyecto
