# CLAUDE.md

Guía para Claude Code cuando trabaje con código en este repositorio.

## Resumen del Proyecto

Phideus v5.0 es un programa de investigación sobre **Harmonic Information Theory** - la hipótesis de que los ratios de frecuencia constituyen un lenguaje universal cross-modal.

**Hitos Validados (Enero 2026)**:
1. **Analizador 5.0**: La representación de datos importa más que la arquitectura (VAE ≈ HRM)
2. **Experimento Roseta 1**: Cross-modal alignment validado (cos_sim = 0.766)
3. **Hipótesis H1-H3**: Estructura, aprendibilidad y transferencia demostradas

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
| Roseta 1 | cos_sim: 0.766 | Cross-modal funciona |

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
│   └── run_roseta_experiment.py       # Experimento Roseta 1
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
│   │   ├── INFORME_ROSETA_1_PARA_PUBLICACION.md
│   │   ├── INFORME_ROSETA_1_HARMONIC_INFORMATION_THEORY.md
│   │   └── PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md
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
3. **H3 - Transferencia**: Se preservan cross-modalmente (Roseta: cos_sim = 0.766)

### Descubrimientos Clave

- **Representación > Arquitectura**: Escala lineal + temporal habilita tanto VAE como HRM
- **VAE Rehabilitado**: De catastrófico (4212) a excelente (0.456)
- **Cross-modal funciona**: Audio ↔ Vibración comparten estructura latente

## Próximos Pasos

### Roseta 2: Audio → Visual (Lissajous)
- Validar H3 en dominio visual
- Cross-modal Audio → Imagen

### Investigación
- Arquitecturas híbridas HRM-VAE
- Más dominios sensoriales
