# Resultados Gate 2 - BIAS_CONTROL

**Fecha**: 2026-02-05
**Estado**: COMPLETADO - GO

Este directorio contiene copias de todos los resultados de evaluación del Gate 2.

## Estructura

```
resultados_gate2/
├── README.md                          # Este archivo
├── structured_pool_epoch45.json       # Evaluación pool estructurado (TEST DEFINITIVO)
├── eval_log.txt                       # Log de evaluaciones
├── audit/
│   └── audit_gate2_results.json       # Auditoría completa (8/10 PASS)
└── gate2_5/
    ├── gate2_5_results.json           # Análisis de embeddings
    ├── tsne_visualization.png         # t-SNE coloreado por modalidad
    └── variance_analysis.png          # Distribución de varianza por dimensión
```

## Archivos Principales

### structured_pool_epoch45.json
Evaluación con pool estructurado (256 candidatos):
- 64 hard negatives (misma pieza, distinto tiempo)
- 32 semi-hard (mismo compositor)
- 159 random
- 1 positivo

**Resultados clave**:
- Recall@10 A2M: 34.4%
- Recall@10 M2A: 37.6%
- Hard Negative Accuracy: 80.4%
- Decision: **GO**

### audit_gate2_results.json
Auditoría completa con 10 checks:
- 8/10 PASS
- 2 falsos positivos explicados (A2: método impreciso, D1: comportamiento esperado)

### gate2_5_results.json
Análisis de embeddings:
- Domain Probe: 92.7% separabilidad (necesita DANN)
- Dead Dimensions: 0/256 (sin colapso)
- Piece Clustering: Silhouette -0.11 (pobre, pero esperado)

## Visualizaciones

### tsne_visualization.png
t-SNE de embeddings coloreados por modalidad (Audio vs MIDI).
Muestra clara separación modal → justifica Gate 3 (DANN).

### variance_analysis.png
Distribución de varianza por dimensión del embedding.
Confirma que no hay dimensiones "muertas".

## Origen de los Datos

Los archivos originales se encuentran en:
```
data/bias_control_medium/evaluations/
```

Esta copia en Documents/ sirve para:
1. Respaldo de resultados importantes
2. Documentación autónoma del experimento
3. Referencia histórica del proyecto

## Checkpoint Asociado

`data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt`
- 398 MB
- 74M parámetros
- Epoch 44 (guardado como epoch45)
