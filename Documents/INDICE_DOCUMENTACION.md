# Índice de Documentación - Proyecto Phideus v5.0

**Actualizado**: 2026-02-06
**Propósito**: Referencia rápida de todos los documentos importantes del proyecto

---

## Documentos Principales

| Documento | Ubicación | Descripción |
|-----------|-----------|-------------|
| **Estado Actual** | `Documents/Proyecto_Estado_Actual.md` | Estado global del proyecto |
| **Este índice** | `Documents/INDICE_DOCUMENTACION.md` | Mapa de documentación |
| **CLAUDE.md** | `CLAUDE.md` | Instrucciones para Claude Code |
| **Bitácora** | `Documents/bitacora_desarrollo.md` | Log de desarrollo |
| **Paper** | `Documents/PHIDEUS_RESEARCH_PROGRAM_2026.md` | Paper principal (47 refs) |
| **★ Informe Histórico** | `Documents/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | **NUEVO**: Historia completa de representaciones de ratios |

---

## Escalón 1: MAESTRO (Audio ↔ MIDI)

### Estado: 🟡 Análisis completado, decisión pendiente

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| Plan de implementación | `Documents/ESCALON_1/Plan_implementacion.md` | 6 Gates del experimento |
| Plan de validación H3 | `Documents/ESCALON_1/PLAN_VALIDACION_H3.md` | 4 fases: Auditoría, Replicación, Escala, Pipeline |
| Recomendaciones GPT | `Documents/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md` | Specs de Route A y Route B |
| Resultados preliminares | `Documents/ESCALON_1/RESULTADOS_NUEVOS_ENFOQUES.md` | Resultados N=10 (preliminares) |
| **Auditoría Fase A** | `Documents/ESCALON_1/AUDITORIA_FASE_A.md` | Bug t_anchor encontrado |
| **Informe Fases A-B** | `Documents/ESCALON_1/INFORME_FASES_A_B.md` | Resultados corregidos + replicación |
| **Plan análisis errores** | `Documents/ESCALON_1/PLAN_ANALISIS_ERRORES.md` | 5 fases de análisis |
| **Informe análisis errores** | `Documents/ESCALON_1/INFORME_ANALISIS_ERRORES.md` | **Diagnóstico completo, causa raíz** |

### Scripts Principales

| Script | Propósito | Uso |
|--------|-----------|-----|
| `test_retrieval_routes.py` | Test Shazam-style retrieval | `python test_retrieval_routes.py --input-dir <dir>` |
| `analyze_errors.py` | Análisis de errores | `python analyze_errors.py --route A` |
| `analyze_overlap_deep.py` | Análisis de componentes | `python analyze_overlap_deep.py` |
| `ablation_chord_only.py` | Ablation por tipo token | `python ablation_chord_only.py` |
| `diagnose_hash_collision.py` | Diagnóstico colisiones | `python diagnose_hash_collision.py` |
| `compare_routes.py` | Comparación overlap | `python compare_routes.py` |

### Extractores

| Archivo | Descripción | Config actual |
|---------|-------------|---------------|
| `src/extractors/event_based_extractor.py` | **Route A**: Event-Based | DT_BIN=10, CHORD_TOL=5, BOOST=2.0 |
| `src/extractors/improved_tf_extractor.py` | **Route B**: Improved TF | Original |

### Datos

| Directorio | Contenido | Tamaño |
|------------|-----------|--------|
| `experiments/un_audio_un_midi/Varios_pares/` | 10 pares originales | ~2GB |
| `experiments/un_audio_un_midi/muestra_replicacion/` | 20 pares replicación | ~4GB |
| `data/maestro_v3/maestro-v3.0.0/` | Dataset completo | 121GB |

### Resultados Clave

| Fase | Métricas | Documento |
|------|----------|-----------|
| N=10 original (con bug) | 71-80% accuracy | `RESULTADOS_NUEVOS_ENFOQUES.md` |
| N=10 corregido | 32-42% accuracy | `AUDITORIA_FASE_A.md` |
| N=20 replicación | 21-27% accuracy | `INFORME_FASES_A_B.md` |
| N=20 post-mejoras | 27% accuracy, 5.4x random | `INFORME_ANALISIS_ERRORES.md` |

---

## BIAS_CONTROL: Cross-Modal Learning con Control de Sesgo

### Estado: ✅ **Gate 3 CERRADO** — 4 Runs DANN, ninguno mejora sobre Gate 2. Próximo: Gate 4

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Roadmap** | `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan completo y criterios GO/NO-GO (v2.0) |
| **★ Informe Gate 3 completo** | `Documents/BIAS_CONTROL/Gate3_DANN_Results/INFORME_GATE3_COMPLETO.md` | **Evaluación comparativa 4 Runs + decisión** |
| **Comparación Gate 3** | `Documents/BIAS_CONTROL/Gate3_DANN_Results/COMPARISON_GATE3.md` | Tabla comparativa (6 checkpoints) |
| **Informe Runs A/B** | `Documents/BIAS_CONTROL/INFORME_GATE3_DANN_SIN_NORM.md` | Runs A (sin norm) y B (F.normalize) |
| **Informe Gate 2** | `Documents/BIAS_CONTROL/INFORME_GATE2_COMPLETO.md` | Informe exhaustivo Gate 2 |
| **Fast test results** | `Documents/BIAS_CONTROL/BIAS_CONTROL_FAST_TEST_RESULTS.md` | 3 epochs, Gap: 0.026 |
| **Medium test results** | `Documents/BIAS_CONTROL/BIAS_CONTROL_MEDIUM_TEST_RESULTS.md` | 61 epochs, Gap: 0.478 best |
| **Plan implementación** | `Documents/BIAS_CONTROL/Planes_Claude/PLAN_IMPLEMENTACION.md` | Detalles técnicos |

### Módulo Principal: `src/bias_control/`

| Componente | Archivos | Descripción |
|------------|----------|-------------|
| Encoders | `encoders/mert_encoder.py`, `midi_encoder.py`, `projection.py` | MERT, Transformer MIDI, MLPs |
| Losses | `losses/dann.py` | DANN + Gradient Reversal Layer |
| Modelos | `architectures/cross_modal_model.py` | CrossModalModel con VICReg |
| Datos | `datasets/maestro_segments.py` | Dataset MAESTRO segmentado |

### Experimentos: `experiments/bias_control/`

| Script | Gate | Descripción |
|--------|------|-------------|
| `gate0_data_integrity.py` | 0 | Verificación datos y alignment |
| `gate1_intra_modal.py` | 1 | Baselines Audio→Audio, MIDI→MIDI |
| `gate2_foundation.py` | 2 | VICReg cross-modal |
| `gate2_5_embedding_analysis.py` | 2.5 | t-SNE/UMAP diagnóstico |
| `gate3_dann.py` | 3 | Domain adversarial training |
| `gate4_ratio_auxiliary.py` | 4 | Multi-view con ratios |
| `evaluate_structured_pool.py` | - | Pool estructurado (test definitivo) |
| `compare_gate3_checkpoints.py` | 3 | **Comparación 6+ checkpoints Gate 3** |
| `gate6_retroanalysis.py` | 6 | RSA/CKA embeddings vs ratios (pendiente) |
| `run_all_gates.py` | - | Orquestador completo |

### Arquitectura

```
Audio → MERT (frozen, 330M) → Projection → Embedding (256d)
MIDI  → Transformer (4L, 8H) → Projection → Embedding (256d)
       └──────────────────────────────────────────────┘
                              │
                    VICReg Loss + DANN (opcional)
```

### Comandos

```bash
# Pipeline completo
python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control
```

---

## UOEMD / Rosetta (Histórico - NO-GO)

### Estado: 🔴 Cerrado - Dataset insuficiente

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| Roadmap revisionismo | `Documents/UOEMD/UOEMD_Revisionismo/ROADMAP.md` | Plan de 4 fases |
| Fase 0 | `Documents/UOEMD/UOEMD_Revisionismo/Fase_0/` | Tests sintéticos |
| Fase 1 | `Documents/UOEMD/UOEMD_Revisionismo/Fase_1/` | Extractor v2.2 |
| Fase 2 | `Documents/UOEMD/UOEMD_Revisionismo/Fase_2/` | Re-entrenamiento |
| Fase 3A | `Documents/UOEMD/UOEMD_Revisionismo/Fase_3A/` | Constellation tokens |
| Resultados v2.2 | `Documents/UOEMD/UOEMD_Roseta_v2.2/` | Métricas extractor |
| Planes Claude | `Documents/UOEMD/Planes Claude/` | Planes de implementación |

### Conclusión UOEMD

El dataset UOEMD (128 muestras de motor diésel) no demostró cross-modality:
- Gap pre-red: 0.691 (extractor v2.2)
- Gap post-red: 0.007 (modelo no aprende)
- Top-1 retrieval: 0.78% (= random)

---

## Experimentos Generales

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| Comparativo 4.1 vs 5.0 | `Documents/Experimentos/REPORTE_COMPARATIVO_4.1_vs_5.0.md` | Analizadores |
| HRM vs VAE masivo | `Documents/Experimentos/RESULTADOS_HRM_VS_VAE_MASIVO.md` | 4 arquitecturas |
| HRM training | `Documents/Experimentos/RESULTADOS_HRM_TRAINING.md` | Hierarchical model |

### Scripts

| Script | Propósito |
|--------|-----------|
| `experiments/run_experiments_5.0.py` | Comparación 4 arquitecturas |
| `experiments/run_roseta_experiment.py` | Experimento Roseta |
| `experiments/evaluate_retrieval.py` | Evaluación retrieval |
| `experiments/evaluate_cross_reconstruction.py` | Cross-reconstruction |

---

## Código Fuente Principal

### Analizadores

| Archivo | Descripción |
|---------|-------------|
| `src/analizador/analizador_5.0.py` | Principal - escala lineal + temporal |
| `src/analizador/analizador_roseta.py` | Dual-domain para Roseta |
| `src/analizador/analizador_maestro.py` | Extracción MAESTRO (no usado) |

### Modelos RNA

| Archivo | Descripción |
|---------|-------------|
| `src/RNA/roseta_vae.py` | VAE cross-modal |
| `src/RNA/constellation_vae.py` | ConstellationVAE |
| `src/RNA/jepa_lite.py` | JEPA sin decoder |
| `src/RNA/vicreg.py` | VICReg loss |
| `src/RNA/barlow_twins.py` | Barlow Twins loss |

### Datasets

| Archivo | Descripción |
|---------|-------------|
| `src/datasets/temporal_dataset_5.py` | Loader NPZ/JSON |
| `src/datasets/roseta_dataset.py` | Loader dual-domain |
| `src/datasets/maestro_dataset.py` | Loader MAESTRO |

### Utilidades

| Archivo | Descripción |
|---------|-------------|
| `src/utils/midi_utils.py` | Parseo MIDI, piano roll |

---

## Estructura de Directorios

```
/mnt/m2-1TB/Phideus/
├── Documents/
│   ├── INDICE_DOCUMENTACION.md      # ← Este archivo
│   ├── Proyecto_Estado_Actual.md    # Estado global
│   ├── ESCALON_1/                   # Documentación Escalón 1
│   ├── UOEMD/                       # Documentación UOEMD (histórico)
│   ├── Experimentos/                # Reportes de experimentos
│   ├── Analizador/                  # Specs analizadores
│   ├── Roseta/                      # Docs Roseta (histórico)
│   └── Legacy/                      # NO USAR
├── src/
│   ├── analizador/                  # Extractores de features
│   ├── extractors/                  # Route A/B extractors
│   ├── datasets/                    # Data loaders
│   ├── RNA/                         # Modelos neuronales
│   └── utils/                       # Utilidades
├── experiments/
│   ├── maestro/                     # Scripts MAESTRO (Gates)
│   ├── un_audio_un_midi/            # Scripts Escalón 1
│   └── *.py                         # Scripts generales
├── data/
│   ├── maestro_v3/                  # Dataset MAESTRO (121GB)
│   └── datasets/                    # Datasets procesados
└── models/                          # Modelos guardados
```

---

## Comandos Frecuentes

### Setup

```bash
cd /mnt/m2-1TB/Phideus
source venv/bin/activate
```

### Escalón 1

```bash
# Test retrieval
python experiments/un_audio_un_midi/test_retrieval_routes.py \
    --input-dir experiments/un_audio_un_midi/muestra_replicacion

# Análisis de errores
python experiments/un_audio_un_midi/analyze_errors.py --route A

# Ablation
python experiments/un_audio_un_midi/ablation_chord_only.py
```

### Git

```bash
git status
git log --oneline -10
git diff
```

---

## Histórico de Decisiones

| Fecha | Decisión | Razón |
|-------|----------|-------|
| 2026-01 | NO-GO UOEMD | Dataset muy pequeño (128 muestras) |
| 2026-02-04 | Auditar experimento N=10 | Resultados sospechosamente altos |
| 2026-02-04 | Bug t_anchor encontrado | 71% → 42% accuracy |
| 2026-02-04 | Replicar con N=20 | Confirmar resultados |
| 2026-02-04 | Análisis de errores | Accuracy baja (27%) |
| 2026-02-04 | Mejoras A+B | Overlap +8pp, accuracy +0.4pp |
| 2026-02-04 | **Pausa Escalón 1** | Rendimientos decrecientes |
| 2026-02-04 | **BIAS_CONTROL** | Nuevo enfoque: soft matching con embeddings |
| 2026-02-04 | Fast test BIAS_CONTROL | Gap: 0.026 (3 epochs) |
| 2026-02-04 | Medium test inicio | 30 epochs, 200 bat/ep |
| 2026-02-04 | Migración tmux | Resume capability añadida |
| 2026-02-05 | Escalar a 1000 bat/ep | Más data coverage |
| 2026-02-05 | Recalibrar criterios (v1.3) | Pool estructurado como test definitivo |
| 2026-02-05 | **Gate 2 completado - GO** | Gap 0.478, R@10 34.4%, Hard neg 80.4% |
| 2026-02-05 | **Gate 3 smoke test - GO** | Script validado, métricas sin degradación |
| 2026-02-05 | **Gate 3 DANN training** | 30 epochs lanzado en tmux |
| 2026-02-05 | **Gate 6 añadido al roadmap** | Retroanálisis embeddings vs ratios (v1.6) |
| 2026-02-05 | Gate 3 epoch 7 **nuevo best** | Domain acc 62.7%, R@10 7.3% |
| 2026-02-06 | **Gate 3 Run A detenido ep10** | Fix normalización, lanzar Run B |
| 2026-02-06 | **Gate 3 Run B (norm) completado** | F.normalize antes domain head |
| 2026-02-06 | **Gate 3 Run C (λ=0.8) detenido ep27** | Sobre-regularización, no mejora |
| 2026-02-06 | **Evaluación comparativa completada** | 6 checkpoints, pool estructurado: Gate 2 ≈ Run C ep4 |
| 2026-02-06 | **Gate 3 Run D (λ=0.3) lanzado** | Último experimento DANN |
| 2026-02-07 | **Gate 3 Run D completado** | R@10 27.4% — peor que Gate 2 |
| 2026-02-07 | **Gate 3 CERRADO** | DANN no mejora en ningún régimen → Gate 4 |
