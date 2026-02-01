# Documentos del Revisionismo de Extracción de Ratios

**Actualizado**: 2026-01-31

Lista de todos los documentos relacionados con la implementación y ejecución de las Fases 0, 1 y 2 del Revisionismo de Extracción de Ratios.

---

## Documentos Principales (Fases 0-2)

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **ROADMAP_FINAL_EXTRACCION_RATIOS.md** | `Documents/Analizador/` | Plan maestro con árbol de decisiones, Protocolo P0, especificación v2.2 |
| **Fase_0_results.md** | `Documents/Analizador/` | Resultados preparación: tests sintéticos, branch, backups |
| **Fase_1_results.md** | `Documents/Analizador/` | Resultados sweep 36 configs, config óptima (K=8, prom=0.1, stab=0.7) |
| **Fase_2_results.md** | `Documents/Analizador/` | Resultados re-entrenamiento, controles P0, decisión NO-GO |

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

| Fase | Documentos Clave |
|------|------------------|
| **Fase 0** | `Fase_0_results.md`, `ROADMAP_FINAL_EXTRACCION_RATIOS.md` |
| **Fase 1** | `Fase_1_results.md`, `ROADMAP_FINAL_EXTRACCION_RATIOS.md` |
| **Fase 2** | `Fase_2_results.md`, `ROSETTA_V22_RESULTS.md`, `Proyecto_Estado_Actual.md`, reportes en `data/evaluations/` |

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
├── retrieval/REPORT_RETRIEVAL.md
├── regime_separation/REPORT_REGIME_SEPARATION.md
└── ...

data/training_outputs/
├── roseta_v22/roseta_experiment_report.md
└── roseta_v2_full/roseta_experiment_report.md
```
