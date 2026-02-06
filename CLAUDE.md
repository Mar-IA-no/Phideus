# CLAUDE.md

Guía para Claude Code cuando trabaje con código en este repositorio.

## 📍 UBICACIÓN DEL PROYECTO (Febrero 2026)

**Directorio de trabajo**: `/mnt/m2-1TB/Phideus`

El proyecto fue migrado desde `/root/Phideus` a un disco M.2 de 1TB para:
- Mayor espacio disponible (726GB vs 37GB)
- Preparación para descargar datasets grandes

**NO trabajar desde `/root/Phideus`** - ese directorio está obsoleto.

---

## 🚀 DIRECTIVA CRÍTICA: Optimización de Hardware

**Hardware disponible:**
- **CPU**: Intel i5-12600K - **16 cores**
- **GPU**: NVIDIA RTX 3090 - **24GB VRAM**

### Reglas de Optimización

**SIEMPRE maximizar el uso del hardware disponible:**

1. **Procesamiento paralelo CPU**:
   - Usar `--workers 14` (dejar 2 cores para sistema) en scripts de procesamiento
   - Usar `multiprocessing.Pool(14)` o `ProcessPoolExecutor(max_workers=14)`
   - Para operaciones I/O-bound, usar `ThreadPoolExecutor`

2. **GPU (PyTorch)**:
   - Batch size máximo que quepa en VRAM (RTX 3090 = 24GB)
   - Usar `torch.cuda.amp` (mixed precision) cuando sea apropiado
   - DataLoader con `num_workers=8, pin_memory=True, prefetch_factor=2`
   - Para training: `batch_size=64` o mayor si el modelo lo permite

3. **Ejemplos de comandos optimizados**:
   ```bash
   # Procesamiento de datos (14 workers)
   python src/analizador/analizador_roseta.py \
       --input-dir data/datasets/UOEMD/raw \
       --output data/datasets/roseta.npz \
       --workers 14

   # Training con GPU optimizado
   python experiments/run_roseta_experiment.py \
       --data data/datasets/roseta.npz \
       --batch-size 64 \
       --num-workers 8
   ```

4. **Nunca usar**:
   - `--workers 1` (a menos que sea para debugging)
   - Batch sizes pequeños sin justificación
   - Procesamiento secuencial cuando se puede paralelizar

---

## 📁 DIRECTIVA: Organización de Documentación

### Estructura Actual (Febrero 2026)

**ÍNDICE COMPLETO**: `Documents/INDICE_DOCUMENTACION.md`

```
Documents/
├── INDICE_DOCUMENTACION.md       # ★ ÍNDICE DE TODA LA DOCUMENTACIÓN
├── Proyecto_Estado_Actual.md     # Estado global del proyecto
├── BIAS_CONTROL/                 # ★ EXPERIMENTO ACTIVO (Febrero 2026)
│   ├── ROADMAP_BIAS_CONTROL.md           # Arquitectura y gates
│   ├── BIAS_CONTROL_FAST_TEST_RESULTS.md # Fast test (3 epochs)
│   ├── BIAS_CONTROL_MEDIUM_TEST_RESULTS.md # Medium test (30 epochs)
│   └── fast_test/                        # Resultados JSON
├── ESCALON_1/                    # Escalón 1 original (pausado)
│   ├── Plan_implementacion.md
│   ├── PLAN_VALIDACION_H3.md
│   └── INFORME_ANALISIS_ERRORES.md
├── UOEMD/                        # UOEMD/Rosetta (histórico - NO-GO)
└── Legacy/                       # NO USAR
```

### Reglas
1. **Documentación UOEMD** → `Documents/UOEMD/`
2. **Nuevos planes** → `Documents/ESCALON_1/` o carpeta del proyecto
3. **Legacy** → NO acceder a menos que se solicite explícitamente

---

## Resumen del Proyecto

Phideus v5.0 es un programa de investigación sobre **Harmonic Information Theory** - la hipótesis de que los ratios de frecuencia constituyen un lenguaje universal cross-modal.

**Estado (Febrero 2026)** - BIAS_CONTROL Gate 3 (DANN) EN EJECUCIÓN - Epoch 8/30:
1. **H1 - Estructura**: ✅ VALIDADA - Las señales contienen distribuciones de ratios estructuradas
2. **H2 - Aprendibilidad**: ✅ VALIDADA - Redes neuronales pueden aprenderlas (val_loss < 0.5)
3. **H3 - Cross-modality**: 🟢 **PROMETEDOR** - BIAS_CONTROL: Gap 0.478, Hard neg acc 80.4%

### BIAS_CONTROL Gate 3 DANN (2026-02-05) - EN EJECUCIÓN 🔄

**Base**: Gate 2 checkpoint_epoch45.pt (Gap 0.478, R@10 34.4%, Hard neg 80.4%)

**Gate 3 training epoch 8/30** (nuevo best en epoch 7):

| Métrica | Gate 2 | Gate 3 (ep7 best) | Tendencia |
|---------|--------|-------------------|-----------|
| Domain acc | 92.7% | **62.7%** | ↓ Objetivo ~50% |
| R@10 (global) | 2.6% | **6.3%** | ↑ 2.4× Gate 2 |
| Loss | 14.09 | **13.99** | ↓ Convergiendo |

**Decisión**: Gate 3 en progreso. Tendencia positiva — domain acc bajando, R@10 mejorando.

**Documentación**: `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` (v1.6, incluye Gate 6)

### Escalón 1 MAESTRO (2026-02-04) - ANÁLISIS COMPLETADO

Fases A (Auditoría) y B (Replicación) completadas + Análisis de errores exhaustivo:

| Experimento | Route A | Route B | vs Random |
|-------------|---------|---------|-----------|
| N=10 (con bug) | 71.4% | 80.0% | - |
| N=10 (corregido) | 42.5% | 32.9% | 4.2x / 3.3x |
| N=20 (replicación) | 26.6% | 21.4% | 5.3x / 4.3x |
| N=20 (post-mejoras) | **27.0%** | 21.4% | **5.4x** / 4.3x |

**Diagnóstico**: El problema es la **resolución temporal del onset detector**, no el algoritmo de hashing.
- Tokens chord: 72% overlap cross-modal (funcionan)
- Tokens sequential/constellation: 3-15% overlap (no funcionan)
- Mejoras incrementales: +8pp overlap → +0.4pp accuracy (rendimientos decrecientes)

**Decisión**: Pausado. BIAS_CONTROL ofrece mejor perspectiva para H3.

Ver: `Documents/ESCALON_1/INFORME_ANALISIS_ERRORES.md`

### Revisionismo UOEMD (Histórico - NO-GO)

El experimento con dataset UOEMD produjo NO-GO (128 muestras insuficientes).

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
/mnt/m2-1TB/Phideus/
├── src/
│   ├── analizador/
│   │   ├── analizador_5.0.py          # PRINCIPAL - escala lineal + temporal
│   │   ├── analizador_4.1_Enriched.py # Legacy - escala log (para referencia)
│   │   └── analizador_roseta.py       # Dual-domain para Roseta
│   ├── extractors/                    # ★ NUEVOS EXTRACTORES (Escalón 1)
│   │   ├── event_based_extractor.py   # Route A: Event-Based (71.4%)
│   │   └── improved_tf_extractor.py   # Route B: Improved TF (80.0%)
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

### En Investigación

3. **H3 - Cross-modality**: 🟡 PENDIENTE - Resultados preliminares prometedores (N=10)

### Descubrimientos Clave

- **Representación > Arquitectura**: Escala lineal + temporal habilita tanto VAE como HRM
- **VAE Rehabilitado**: De catastrófico (4212) a excelente (0.456)
- **Nuevos extractores prometedores**: 71-80% accuracy en experimento piloto (N=10)
- **Onset anchoring parece crítico**: Reduce hashes genéricos (pendiente validación)

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

### Estado al 2026-01-30
**Fase 2 Completada**: Resultado **NO-GO** - H3 no validada

### Resultado Principal
El Extractor v2.2 mejoró 172× la discriminabilidad pre-red, pero el modelo **sigue sin demostrar cross-modality**:
- aligned vs shuffled: Δcorr = 0.007 (necesario > 0.15)
- Retrieval Top-1: 10.94% (mejoró pero insuficiente)
- El modelo genera embeddings genéricos, no aprende correspondencia de pares

### Qué se ejecutó (Fase 2)
1. Dataset regenerado con Extractor v2.2: `data/datasets/roseta_v22_full.npz`
2. Entrenamiento RosetaVAE (100 epochs): `data/training_outputs/roseta_v22/`
3. Evaluación completa: cross-reconstruction, retrieval, regime separation

### Criterios Go/No-Go (Resultados Fase 2)
| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| **Gap aligned-shuffled** | **> 0.15** | **0.007** | **FAIL (CRÍTICO)** |
| Retrieval Top-1 | > 10× random | 10.94% vs 0.78% (14×) | ✅ PASS |
| Silhouette score | > 0.3 | -0.14 | ❌ FAIL |
| var(z_private) | > 0.1 | 0.0043 | ❌ FAIL |

### Lecciones Aprendidas (Adicionales)
1. Mejorar el extractor (172×) no garantiza mejora del modelo (solo 3.5×)
2. El VAE colapsa la información discriminativa del histograma
3. El problema puede ser arquitectural (VAE+InfoNCE) o de hipótesis (H3 falsa)

### Documentación Actualizada
- `Documents/Roseta/ROSETTA_V22_RESULTS.md` - **Resultados Fase 2**
- `Documents/Proyecto_Estado_Actual.md` - Estado del proyecto actualizado
- `data/evaluations/retrieval/REPORT_RETRIEVAL.md` - Métricas retrieval
- `data/evaluations/regime_separation/REPORT_REGIME_SEPARATION.md` - Métricas separación

---

## 🔴 ESTADO FINAL: Revisionismo de Extracción de Ratios (Febrero 2026)

### Resumen del Revisionismo

**Fase 1 (Completada)**: Extractor v2.2 implementado
- Gap pre-red: 0.691 (172× mejor que v1)
- Config óptima: K=8, prom=0.1, stab=0.7

**Fase 2 (Completada)**: Re-entrenamiento con histogramas discriminativos
- Gap post-red: 0.007 (solo 3.5× mejor que v1)
- **Conclusión**: El modelo no capitaliza la mejora del extractor

**Fase 3A (Completada)**: Tokens sparse estilo Shazam
- Gap pre-red: 0.029 (cerca de random)
- Top-1 Retrieval: 0.78% (= random)
- **Conclusión**: Tokens sparse pierden información discriminativa

**Fase 3A-1b (Completada)**: Corrección de extractor y training config
- Extractor mejorado: linear bands, más tokens, constraints relajados
- Training corregido: dropout=0.5, beta_kl_private=0.01
- **Top-1 Retrieval: 0.78%** - aún en nivel random
- **Conclusión FINAL**: El problema es la REPRESENTACIÓN, no la arquitectura

### Hipótesis Estado Final (Actualizado 2026-02-04)

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | val_loss < 0.5 |
| H3: Cross-modality | 🟡 **PENDIENTE** | Resultados preliminares prometedores (N=10, 80% acc) |

### Decisión Tomada: Fase 3A - Ratio Constellations

**Fecha decisión**: 2026-01-31
**Plan completo**: `Documents/Planes Claude/Fase_3A.md`

#### Concepto Principal
Cambiar de histograma denso [T, 256, 3] a **tokens sparse** estilo Shazam:
```python
token = {
    'log_ratio': np.log2(target.freq / anchor.freq),
    'delta_t': target.time - anchor.time,
    'weight': np.sqrt(anchor.amp * target.amp),
    'anchor_band': get_band_id(anchor.freq),
    'target_band': get_band_id(target.freq)
}
# Output: [T, 48, 5] en lugar de [T, 256, 3]
```

#### 6 Configuraciones a Probar
| Config | Encoder | Decoder |
|--------|---------|---------|
| C1 | MLP+Attention | Histograma |
| C2 | MLP+Attention | Tokens |
| C3 | Transformer | Histograma |
| C4 | Transformer | Tokens |
| C5 | MLP+Attention | **JEPA-lite (sin decoder)** |
| C6 | Transformer | **JEPA-lite (sin decoder)** |

#### Fases de Implementación (TODAS COMPLETADAS)
1. **3A-0**: ✅ Auditoría de evaluación (consistencia)
2. **3A-1**: ✅ Extractor de constellations
3. **3A-2**: ✅ Dataset loader con máscaras
4. **3A-3**: ✅ Modelos ConstellationVAE + JEPA-lite
5. **3A-4**: ✅ Training loop
6. **3A-5**: ✅ Sweep de 6 configs + evaluación

#### RESULTADOS FASE 3A: NO-GO → AUDITORÍA REVELÓ BUGS

| Config | Encoder | Decoder | Top-1 | Status |
|--------|---------|---------|-------|--------|
| C1 | MLP | Histogram | 0.78% | FAIL |
| C2 | MLP | Token | 0.78% | FAIL |
| C3 | Transformer | Histogram | 0.78% | FAIL |
| C4 | Transformer | Token | 0.78% | FAIL |
| **C5** | MLP | JEPA-lite | **1.56%** | FAIL |
| C6 | Transformer | JEPA-lite | 0.78% | FAIL |

**Random baseline**: 0.78% (1/128). Todos los modelos en nivel random o muy cerca.

#### ✅ FASE 3A-1b: CORRECCIONES APLICADAS (2026-02-01)

**Correcciones implementadas**:
1. Extractor mejorado: linear bands, n_bands=16, más tokens (avg 27/frame)
2. Training corregido: `dropout_shared=0.5`, `beta_kl_private=0.01`, `lambda_diff=0.1`

**Resultados post-corrección**:
| Métrica | Antes | Después | Umbral |
|---------|-------|---------|--------|
| Gap pre-red | -0.0019 | 0.029 | > 0.05 |
| Top-1 Retrieval | 0.78% | 0.78% | > 15% |

**Conclusión FINAL**: Las correcciones **no mejoraron retrieval**. El problema es estructural:
- Tokens sparse pierden información discriminativa presente en histogramas densos
- Histograma v2.2 tiene gap = 0.691, pero tokens solo tienen gap = 0.029

Ver: `Documents/UOEMD/UOEMD_Revisionismo/Fase_3A/FASE_3A_1b_RESULTS.md`

#### Scripts de Auditoría

```
experiments/audits/
├── checkpoint_1_data_integrity.py
├── checkpoint_2_collate_verification.py
├── checkpoint_3_forward_pass.py
├── checkpoint_4_metrics.py
├── checkpoint_5_training.py
├── checkpoint_6_e2e.py
└── run_all_audits.py
```

#### Opciones Futuras
1. **Aceptar NO-GO**: Publicar resultados negativos (valor científico)
2. **Cambiar representación**: Abandonar ratios, probar spectrograms + contrastive
3. **Cambiar hipótesis**: H3' = "Audio y vibración NO comparten estructura armónica"

### Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `Documents/UOEMD/UOEMD_Revisionismo/Fase_3A/FASE_3A_1b_RESULTS.md` | **Resultados finales 3A-1b** |
| `Documents/UOEMD/Planes Claude/Fase_3A.md` | Plan original Fase 3A |
| `Documents/UOEMD/UOEMD_Revisionismo/ROADMAP.md` | Roadmap del revisionismo |
| `data/datasets/roseta_constellation_v3.npz` | Dataset tokens (mejor config) |
| `data/training_outputs/constellation_v3_jepa_mlp/` | Modelo JEPA-lite entrenado |
| `data/evaluations/retrieval_v3_jepa/` | Métricas de retrieval |

### Estructura de Documentos (Actualizada 2026-02-04)

```
Documents/
├── UOEMD/                          # Documentación del proyecto UOEMD/Rosetta
│   ├── Planes Claude/              # Planes de implementación
│   │   ├── Fase_3A.md              # Plan tokens sparse
│   │   └── fase_2.md
│   ├── UOEMD_Revisionismo/         # Documentación del Revisionismo
│   │   ├── Analizador/             # Specs, informes GPT5.2Think
│   │   ├── Fase_0/                 # Tests sintéticos
│   │   ├── Fase_1/                 # Extractor v2.2
│   │   ├── Fase_2/                 # Re-entrenamiento (NO-GO)
│   │   ├── Fase_3A/                # Tokens sparse (NO-GO)
│   │   └── ROADMAP.md              # Roadmap del revisionismo
│   ├── UOEMD_Roseta_v2.2/          # Resultados v2.2
│   └── UOEMD_Rosetta_v1_y_v2/      # Archivos históricos
├── ESCALON_1/                      # Nuevo experimento (pendiente)
├── Experimentos/                   # Resultados de experimentos
├── Legacy/                         # NO usar
├── Proyecto_Estado_Actual.md
└── bitacora_desarrollo.md
```

---

## Histórico: Fases 1-2 del Revisionismo

### Fase 1: Extractor v2.2 (COMPLETADA - GO)
- Sweep de 36 configuraciones
- Config óptima: K=8, prom=0.1, stab=0.7
- Gap pre-red: 0.691 (172× mejor que v1)

### Fase 2: Re-entrenamiento (COMPLETADA - NO-GO)
- VAE con dataset discriminativo
- Gap post-red: 0.007 (solo 3.5× mejor)
- El modelo colapsa la información discriminativa

### Archivos Clave para Fase 3A

| Archivo | Propósito |
|---------|-----------|
| `Documents/UOEMD/Planes Claude/Fase_3A.md` | **Plan completo de implementación** |
| `src/analizador/analizador_roseta.py` | Extractor con `extract_constellation()` |
| `src/datasets/roseta_dataset.py` | Loader de tokens |
| `src/RNA/constellation_vae.py` | Modelo modular (C1-C4) |
| `src/RNA/jepa_lite.py` | Modelo sin decoder (C5-C6) |
| `experiments/run_roseta_experiment.py` | Training con flags para constellation |
| `experiments/evaluate_retrieval.py` | Evaluación con hard negatives |

### Comandos para Fase 3A

```bash
# Activar entorno
cd /mnt/m2-1TB/Phideus
source venv/bin/activate
git checkout feature/extractor-v22

# Fase 3A-0: Auditoría de evaluación
python experiments/evaluate_retrieval.py \
    --model data/training_outputs/roseta_v22/best_model.pt \
    --seed 42

# Fase 3A-1: Extractor de constellations
python src/analizador/analizador_roseta.py \
    --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files \
    --output data/datasets/roseta_constellation.npz \
    --output-format constellation

# Fase 3A-5: Entrenar configuraciones
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --model constellation \
    --encoder-type mlp \
    --decoder-type histogram
```

### Recursos de Referencia
- `Documents/UOEMD/UOEMD_Revisionismo/Analizador/` - Informes GPT5.2Think y Claude
- `Documents/UOEMD/UOEMD_Revisionismo/Fase_3A/` - Resultados y auditorías
- SOTA: Shazam, PeakNetFP, Audio-JEPA, SparseVLM

---

## 🟢 PROGRESO FASE 3A (2026-02-01) - IMPLEMENTACIÓN COMPLETADA

### Estado de Implementación

| Fase | Commit | Estado | Descripción |
|------|--------|--------|-------------|
| **3A-0** | `3ce4b4b` | ✅ | Reproducibilidad en scripts de evaluación |
| **3A-1** | `601280d` | ✅ | Extractor de constellations |
| **3A-2** | `baaa349` | ✅ | Dataset loader para tokens |
| **3A-3** | `09c5229` | ✅ | ConstellationVAE + JEPA-lite |
| **3A-4** | `94fcb3e` | ✅ | Training loop actualizado |
| **3A-5** | - | 🟡 **PENDIENTE** | Sweep de 6 configuraciones |

### Archivos Creados/Modificados

**Nuevos archivos**:
- `src/RNA/constellation_vae.py` - ConstellationVAE modular (C1-C4)
- `src/RNA/jepa_lite.py` - JEPA-lite sin decoder (C5-C6)

**Modificados**:
- `src/analizador/analizador_roseta.py` - `--output-format constellation`
- `src/datasets/roseta_dataset.py` - `RosetaConstellationDataset`, auto-detect
- `experiments/run_roseta_experiment.py` - `--model`, `--encoder-type`, `--decoder-type`
- `experiments/evaluate_retrieval.py` - Reproducibilidad
- `experiments/evaluate_cross_reconstruction.py` - Reproducibilidad

### Arquitectura de Modelos (6 Configuraciones)

| Config | Encoder | Decoder | Params |
|--------|---------|---------|--------|
| C1 | MLP+Attention | Histogram | ~460K |
| C2 | MLP+Attention | Token | ~398K |
| C3 | Transformer | Histogram | ~523K |
| C4 | Transformer | Token | ~461K |
| C5 | MLP+Attention | JEPA (sin decoder) | ~196K |
| C6 | Transformer | JEPA (sin decoder) | ~258K |

### Formato de Datos Constellation

```python
# NPZ keys:
audio_tokens_{idx}: [T, 48, 5]      # tokens sparse
audio_mask_{idx}: [T, 48]           # 1=válido, 0=padding
vibration_tokens_{idx}: [T, 48, 5]
vibration_mask_{idx}: [T, 48]

# Token format: [log_ratio, delta_t, weight, anchor_band, target_band]
```

### Dataset de Test

```
Archivo: /tmp/test_constellation.npz
- 128 archivos, 52,096 frames
- Avg tokens/frame: Audio=11.2, Vib=13.2
- Testeado con ConstellationVAE y JEPA-lite ✓
```

### PRÓXIMO PASO: Fase 3A-5 - Sweep de 6 Configuraciones

**1. Generar dataset completo** (14 workers):
```bash
python src/analizador/analizador_roseta.py \
    --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files \
    --output data/datasets/roseta_constellation.npz \
    --output-format constellation \
    --workers 14
```

**2. Entrenar 6 configuraciones** (ejemplo C1):
```bash
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --output data/training_outputs/constellation_C1_mlp_hist \
    --model constellation --encoder-type mlp --decoder-type histogram \
    --epochs 100 --batch-size 64 --num-workers 8
```

**3. Configuraciones a entrenar**:
- C1: `--model constellation --encoder-type mlp --decoder-type histogram`
- C2: `--model constellation --encoder-type mlp --decoder-type token`
- C3: `--model constellation --encoder-type transformer --decoder-type histogram`
- C4: `--model constellation --encoder-type transformer --decoder-type token`
- C5: `--model jepa-lite --encoder-type mlp`
- C6: `--model jepa-lite --encoder-type transformer`

**4. Evaluar con** `experiments/evaluate_retrieval.py`

### Criterios GO/NO-GO (Fase 3A)

| Criterio | Umbral |
|----------|--------|
| **Gap aligned-shuffled (intra-cond)** | **> 0.10** |
| Gap aligned-shuffled (global) | > 0.15 |
| Retrieval Top-1 (intra-cond) | > 2× random |

3. Crear factory function `create_roseta_dataloaders()` que auto-detecte formato:
   - Si `output_format == 'constellation'` → usar constellation loader
   - Si no → usar histogram loader existente

**Test**: Usar `/tmp/test_constellation.npz` para verificar

### Comandos para Continuar

```bash
cd /mnt/m2-1TB/Phideus
source venv/bin/activate
git checkout feature/extractor-v22

# Ver estado actual
git log --oneline -5
```

---

## 🎹 ESCALON 1: MAESTRO (Audio ↔ MIDI) - IMPLEMENTACIÓN COMPLETADA

**Fecha**: 2026-02-04
**Estado**: ✅ TODOS LOS 6 GATES IMPLEMENTADOS

**Objetivo**: Demostrar cross-modal learning entre Audio y MIDI usando ratio constellations.

**Dataset**: MAESTRO v3.0.0 (120GB, ~200h piano)

### Archivos Implementados

```
src/utils/midi_utils.py              ✅ Parseo MIDI, piano roll, constellation tokens
src/RNA/vicreg.py                    ✅ VICReg loss + encoder
src/RNA/barlow_twins.py              ✅ Barlow Twins loss + encoder
src/analizador/analizador_maestro.py ✅ Extracción constellation audio+MIDI
src/datasets/maestro_dataset.py      ✅ DataLoader para tokens MAESTRO

experiments/maestro/
├── gate0_harness.py                 ✅ Métricas + controles negativos
├── gate1_ingest.py                  ✅ Descarga + segmentación MAESTRO
├── gate2_baselines.py               ✅ Chroma + CCA baselines
├── gate3_cross_modal.py             ✅ Training VICReg/Barlow
├── gate4_ratio_tokens.py            ✅ Training constellation + baseline matching
├── gate5_moco.py                    ✅ MoCo queue + hard negatives
└── run_maestro_experiment.py        ✅ Script orquestador principal
```

### Comandos para Ejecutar el Experimento

```bash
cd /mnt/m2-1TB/Phideus
source venv/bin/activate

# Instalar dependencias
pip install pretty_midi mido

# Descargar MAESTRO (101GB)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip \
    -O data/maestro_v3/maestro-v3.0.0.zip
cd data/maestro_v3 && unzip maestro-v3.0.0.zip && cd ../..

# Ejecutar pipeline completo (6 Gates)
python experiments/maestro/run_maestro_experiment.py \
    --mode full \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/training_outputs/maestro_experiment \
    --epochs 100 --batch-size 64 --num-workers 8

# O ejecutar gates individuales:
python experiments/maestro/gate4_ratio_tokens.py \
    --data data/maestro_v3/constellations/tokens.npz \
    --output data/training_outputs/maestro_constellation \
    --model constellation --encoder-type mlp \
    --epochs 100 --batch-size 64
```

### Criterios GO/NO-GO por Gate

| Gate | Descripción | Criterio GO |
|------|-------------|-------------|
| 0 | Setup harness | Oracle > 90%, random ~ 1/N |
| 1 | Ingesta datos | Correlación energia-densidad > 0.7 |
| 2 | Baselines | Piece Top-1 > 10x random |
| 3 | VICReg/Barlow | No colapso + Top-1 > baselines |
| 4 | Ratio tokens | Matching > random, modelo comparable |
| 5 | MoCo | Mejora NEG-SAME-COMPOSER |

### Plan Original

Ver: `Documents/ESCALON_1/Plan_implementacion.md`

### ✅ IMPLEMENTACIÓN COMPLETADA (2026-02-04)

**Informe de auditoría**: `Documents/ESCALON_1/AUDITORIA_IMPLEMENTACION.md`

**Correcciones aplicadas**:
1. ✅ `gate4_ratio_tokens.py`: Lee `max_tokens` del NPZ
2. ✅ `gate5_moco.py`: Lee `max_tokens` del NPZ
3. ✅ `gate4_ratio_tokens.py`: Añadido VICReg/Barlow según plan original

**Gate 4 - Modelos disponibles** (commit `9c2906a`):

| Modelo | Comando | Descripción |
|--------|---------|-------------|
| `vicreg` | `--model vicreg` | **Plan original**: Token encoder + VICReg loss |
| `barlow` | `--model barlow` | **Plan original**: Token encoder + Barlow loss |
| `constellation` | `--model constellation` | UOEMD: ConstellationVAE con decoder |
| `jepa-lite` | `--model jepa-lite` | UOEMD: JEPA sin decoder |

**Estado**: Implementación 100% completa. Descarga MAESTRO en progreso.

### ✅ MAESTRO DESCARGADO Y DESCOMPRIMIDO (2026-02-04)

```
data/maestro_v3/maestro-v3.0.0/   # 121GB, 1276 WAV + 1276 MIDI
```

---

## 🟢 ESCALÓN 1: RESULTADOS FINALES (2026-02-04) - GO

### Estado: ✓ **GO** con nuevos extractores (Route A: 71.4%, Route B: 80.0%)

El experimento Escalón 1 (Audio ↔ MIDI) concluyó con resultado **GO** después de implementar nuevos extractores.

### Evolución de Resultados

| Fase | Extractor | Piece Accuracy | Status |
|------|-----------|---------------|--------|
| V2 original | TF-Constellations | 15.5% | ✗ NO-GO |
| **Route A** | **Event-Based** | **71.4%** | **✓ GO** |
| **Route B** | **Improved TF** | **80.0%** | **✓ GO** |

### Diagnóstico del Problema (V2)

`diagnose_hash_collision.py` reveló **COLISIÓN GENÉRICA**:
- overlap_aligned: 66.23%, overlap_random: 65.13%
- Gap: 1.10% → hashes coincidían pero igual para cualquier par

### Soluciones Implementadas

**Route A: Event-Based** (`src/extractors/event_based_extractor.py`)
- Audio → eventos via CQT + onset detection
- Ratio language sobre intervalos semánticos
- 1,800 tokens/pieza, 71.4% accuracy

**Route B: Improved TF** (`src/extractors/improved_tf_extractor.py`)
- Onset anchoring + Harmonic folding + IDF agresivo
- 52,000 tokens/pieza, **80.0% accuracy**

### Resultados Finales

| Test | Métrica | V2 | Route A | Route B |
|------|---------|-----|---------|---------|
| Token Compatibility | Cosine | 0.957 | - | - |
| Oracle (MIDI vs MIDI) | Piece Acc | 90.9% | - | - |
| **Cross-Modal** | **Piece Acc** | 15.5% | **71.4%** | **80.0%** |
| Cross-Modal | Recall@5 | 50.9% | **100%** | **100%** |

### Estado del Experimento Piloto (N=10)

| Hipótesis | Estado | Nota |
|-----------|--------|------|
| H1: Distribuciones compatibles | ✓ Verificada | cosine > 0.95 |
| H2: Shazam voting funciona | ✓ Verificada | Oracle 90.9% |
| H3: Cross-modal identification | 🟡 **PENDIENTE** | Resultados prometedores, falta validación |

**⚠️ N=10 pares es insuficiente para validar H3.**

### Scripts de Prueba

```
src/extractors/
├── event_based_extractor.py        # Route A
└── improved_tf_extractor.py        # Route B

experiments/un_audio_un_midi/
├── diagnose_hash_collision.py      # Diagnóstico COLISIÓN GENÉRICA
├── compare_routes.py               # Comparación overlap
├── test_retrieval_routes.py        # Retrieval (N=10)
└── Varios_pares/                   # Muestra piloto
```

### Documentación

- **Recomendaciones GPT**: `Documents/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md`
- **Resultados preliminares**: `Documents/ESCALON_1/RESULTADOS_NUEVOS_ENFOQUES.md`
- **Plan de validación**: `Documents/ESCALON_1/PLAN_VALIDACION_H3.md`

### Observaciones Preliminares (Pendiente Validación)

1. **El extractor parece importar más que el algoritmo**: 15.5% → 80% (N=10)
2. **Onset anchoring parece crítico**: Reduce hashes genéricos
3. **Harmonic folding**: Potencialmente esencial para música tonal
4. **IDF agresivo**: Stoplist 30% parece reducir ruido

### Próximos Pasos REQUERIDOS

1. **Auditoría del experimento** (verificar correctitud)
2. **Replicación con muestra independiente** (10-20 pares nuevos)
3. **Validación a escala** (100+ piezas)
4. **Pipeline completo Escalón 1** (Gates 0-5)
