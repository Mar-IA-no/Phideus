# Documentos del Revisionismo de Extracción de Ratios

**Actualizado**: 2026-02-01

Lista de todos los documentos relacionados con la implementación y ejecución de las Fases 0, 1, 2 y 3A del Revisionismo de Extracción de Ratios.

---

## Documentos Principales (Fases 0-3A)

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **ROADMAP_FINAL_EXTRACCION_RATIOS.md** | `Documents/Analizador/` | Plan maestro con árbol de decisiones, Protocolo P0, especificación v2.2 |
| **Fase_0_results.md** | `Documents/Analizador/` | Resultados preparación: tests sintéticos, branch, backups |
| **Fase_1_results.md** | `Documents/Analizador/` | Resultados sweep 36 configs, config óptima (K=8, prom=0.1, stab=0.7) |
| **Fase_2_results.md** | `Documents/Analizador/` | Resultados re-entrenamiento, controles P0, decisión NO-GO |
| **FASE_3A_SWEEP_RESULTS.md** | `data/evaluations/` | Sweep 6 configuraciones constellation, resultado NO-GO |

---

## Fase 3A - Ratio Constellations (NO-GO)

### Documentos de Diseño

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Fase_3A.md** | `Documents/Planes Claude/` | Plan completo de implementación |
| **zazzy-petting-valiant.md** | `/root/.claude/plans/` | Plan mode file |

### Código Implementado

| Archivo | Ubicación | Contenido |
|---------|-----------|-----------|
| **constellation_vae.py** | `src/RNA/` | MLPConstellationEncoder, TransformerConstellationEncoder, decoders |
| **jepa_lite.py** | `src/RNA/` | JEPAPredictor, JEPALite (sin decoder) |
| **analizador_roseta.py** | `src/analizador/` | `extract_constellation()`, `--output-format constellation` |
| **roseta_dataset.py** | `src/datasets/` | `RosetaConstellationDataset`, `detect_npz_format()` |
| **run_roseta_experiment.py** | `experiments/` | Soporte para `--model constellation/jepa-lite` |
| **evaluate_retrieval.py** | `experiments/` | Soporte para modelos constellation |

### Resultados del Sweep

| Reporte | Ubicación | Contenido |
|---------|-----------|-----------|
| **FASE_3A_SWEEP_RESULTS.md** | `data/evaluations/` | Resumen sweep 6 configs |
| **REPORT_RETRIEVAL.md** | `data/evaluations/constellation_C[1-6]/` | Retrieval por config |
| **roseta_experiment_report.md** | `data/training_outputs/constellation_*/` | Training reports |

### Datasets

| Dataset | Ubicación | Contenido |
|---------|-----------|-----------|
| **roseta_constellation.npz** | `data/datasets/` | 128 archivos, tokens [T,48,5] + masks |

### Resultado: NO-GO

Todas las 6 configuraciones FAIL (Top-1 ≤ 1.56%, umbral 15%).
5/6 tienen exactamente 0.78% (nivel random) - **sospechoso, requiere auditoría**.

---

## Documentos de Soporte (Análisis Previo)

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **INFORME_REVISIONISMO_EXTRACCION_RATIOS.md** | `Documents/Analizador/Recursos/` | Síntesis inicial del diagnóstico |
| **PROPUESTA_DOCTORAL_EXTRACCION_RATIOS.md** | `Documents/Analizador/Recursos/` | Alternativas y fases propuestas |

---

## Documentos Rosetta v2.2 (Fase 2)

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **ROSETTA_V22_RESULTS.md** | `Documents/Roseta_v2.2/` | Resultados y decisión Fase 3A/3B |

---

## Documentos Rosetta v2.0 (Histórico - Pre-Revisionismo)

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **ROSETTA1_2.0_IMPLEMENTATION_PLAN.md** | `Documents/Rosetta_v1_y_v2/` | Plan original Rosetta1 2.0 |
| **ROSETTA1_2.0_RESULTADOS_EJECUCION.md** | `Documents/Rosetta_v1_y_v2/v2.0_archived/` | Resultados originales (NO-GO) |
| **Rosetta1_2.0_-_Roadmap_GTP5.2Pro.md** | `Documents/Rosetta_v1_y_v2/v2.0_archived/` | Roadmap GPT5.2Pro |
| **Rosetta1_consistence_evaluation_GPT5.2Pro.md** | `Documents/Rosetta_v1_y_v2/v2.0_archived/` | Evaluación de consistencia |

---

## Estado General del Proyecto

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Proyecto_Estado_Actual.md** | `Documents/` | Estado actualizado con decisión Fase 3 |
| **bitacora_desarrollo.md** | `Documents/` | Log de desarrollo |

---

## Reportes Generados (data/)

### Fase 2 - Evaluaciones

| Reporte | Ubicación | Contenido |
|---------|-----------|-----------|
| **REPORT_RETRIEVAL.md** | `data/evaluations/retrieval/` | Métricas retrieval Fase 2 |
| **REPORT_REGIME_SEPARATION.md** | `data/evaluations/regime_separation/` | Métricas separación Fase 2 |
| **roseta_experiment_report.md** | `data/training_outputs/roseta_v22/` | Reporte automático entrenamiento |

### Histórico (Rosetta v2.0)

| Reporte | Ubicación | Contenido |
|---------|-----------|-----------|
| **roseta_experiment_report.md** | `data/training_outputs/roseta_v2_full/` | Reporte Rosetta1 2.0 |
| **REPORT_RETRIEVAL.md** | `data/evaluations/retrieval_v2_full/` | Retrieval Rosetta1 2.0 |
| **REPORT_REGIME_SEPARATION.md** | `data/evaluations/regime_separation_v2_full/` | Separación Rosetta1 2.0 |

---

## Resumen por Fase

| Fase | Estado | Documentos Clave |
|------|--------|------------------|
| **Fase 0** | GO | `Fase_0_results.md`, `ROADMAP_FINAL_EXTRACCION_RATIOS.md` |
| **Fase 1** | GO | `Fase_1_results.md`, `ROADMAP_FINAL_EXTRACCION_RATIOS.md` |
| **Fase 2** | NO-GO | `Fase_2_results.md`, `ROSETTA_V22_RESULTS.md`, reportes en `data/evaluations/` |
| **Fase 3A** | NO-GO | `FASE_3A_SWEEP_RESULTS.md`, `data/evaluations/constellation_*/` |

---

## Rutas Absolutas

```
Documents/Analizador/
├── documentos_importantes.md      # Este documento
├── Fase_0_results.md
├── Fase_1_results.md
├── Fase_2_results.md
├── ROADMAP_FINAL_EXTRACCION_RATIOS.md
└── Recursos/
    ├── INFORME_REVISIONISMO_EXTRACCION_RATIOS.md
    └── PROPUESTA_DOCTORAL_EXTRACCION_RATIOS.md

Documents/Roseta_v2.2/
└── ROSETTA_V22_RESULTS.md

Documents/Rosetta_v1_y_v2/
├── ROSETTA1_2.0_IMPLEMENTATION_PLAN.md
└── v2.0_archived/
    ├── ROSETTA1_2.0_RESULTADOS_EJECUCION.md
    ├── Rosetta1_2.0_-_Roadmap_GTP5.2Pro.md
    └── Rosetta1_consistence_evaluation_GPT5.2Pro.md

Documents/
├── Proyecto_Estado_Actual.md
└── bitacora_desarrollo.md

data/evaluations/
├── retrieval/REPORT_RETRIEVAL.md              # Fase 2
├── regime_separation/REPORT_REGIME_SEPARATION.md
├── FASE_3A_SWEEP_RESULTS.md                   # Fase 3A resumen
├── constellation_C1/REPORT_RETRIEVAL.md       # Fase 3A por config
├── constellation_C2_mlp_token/
├── constellation_C3_trans_hist/
├── constellation_C4_trans_token/
├── constellation_C5_mlp_jepa/
└── constellation_C6_trans_jepa/

data/training_outputs/
├── roseta_v22/roseta_experiment_report.md
├── roseta_v2_full/roseta_experiment_report.md
├── constellation_C1_mlp_hist/                 # Fase 3A
├── constellation_C2_mlp_token/
├── constellation_C3_trans_hist/
├── constellation_C4_trans_token/
├── constellation_C5_mlp_jepa/
└── constellation_C6_trans_jepa/

src/RNA/                                       # Fase 3A modelos
├── constellation_vae.py
└── jepa_lite.py
```
