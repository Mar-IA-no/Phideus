# Scripts Rosetta1 v1.0 - Archivados

**Estado**: Archivado (Enero 2026)
**Razón**: Scripts de visualización/extracción del experimento original

---

## Contenido

Scripts de análisis y visualización del experimento Rosetta1 original.
Estos scripts fueron creados para explorar los resultados iniciales pero
no forman parte del pipeline de validación 2.0.

| Script | Propósito |
|--------|-----------|
| `extract_roseta_latents.py` | Extracción de representaciones latentes |
| `visualize_roseta_latents.py` | Visualización 2D (UMAP/t-SNE) |
| `visualize_roseta_latents_3d.py` | Visualización 3D |
| `visualize_roseta_intuitive_3d.py` | Visualización 3D interactiva |

## Scripts actuales (2.0)

Los scripts de evaluación actualizados están en `experiments/`:
- `freeze_baseline.py` - Congela artefactos baseline
- `evaluate_cross_reconstruction.py` - Con controles negativos
- `evaluate_retrieval.py` - Retrieval extendido
- `evaluate_regime_separation.py` - Silhouette, AUC, Fisher
- `run_ablations.py` - Estudios de ablación

## Nota

Estos scripts podrían reactivarse o adaptarse si se necesitan
visualizaciones después de validar Rosetta1 2.0.
