# INFORME + PLAN DE CURADURÍA VISUAL BIAS_CONTROL (Post DEC-005)

Fecha: 2026-02-11  
Autor: Codex  
Contexto: cierre de ejecución diagnóstica Gate 6 + Gate 4.2 (DEC-005)

---

## 1) Objetivo de este documento

Este documento integra dos cosas en un solo entregable:

1. Auditoría técnica actualizada de resultados/logs de `BIAS_CONTROL` (gates ejecutados).
2. Plan detallado para dejar todos los resultados relevantes duplicados, ordenados y presentables dentro de `Documents/BIAS_CONTROL`, con navegación pensada tanto para trabajo técnico como para exhibición.

Además, incorpora y contrasta explícitamente el informe de Claude:

- `Documents/BIAS_CONTROL/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`.

---

## 2) Fuentes revisadas (re-auditoría)

### Resultados estructurados

- `data/bias_control_medium/evaluations/structured_pool_epoch45.json`
- `data/bias_control_medium/evaluations/gate4/RA5_ep5.json`
- `data/bias_control_medium/evaluations/gate4/RB0_ep5.json`
- `data/bias_control_medium/evaluations/gate4/R1rescue_ep5.json`
- `data/bias_control_medium/evaluations/gate4/R1rescue_best.json`
- `data/bias_control_medium/evaluations/gate3_comparison/comparison_summary.json`
- `data/bias_control_medium/evaluations/gate6/layer_drift.json`
- `data/bias_control_medium/evaluations/gate6/hubness_analysis.json`
- `data/bias_control_medium/evaluations/gate42/h426_prered_results.json`

### Artefactos visuales ya generados

- `data/bias_control_medium/evaluations/gate6/fig_umap_multigate.png`
- `data/bias_control_medium/evaluations/gate6/fig_bridges_multigate.png`
- `data/bias_control_medium/evaluations/gate6/fig_heatmaps_multigate.png`
- `data/bias_control_medium/evaluations/gate6/fig_hubness_distribution.png`
- `data/bias_control_medium/evaluations/gate6/fig_similarity_distributions.png`
- `data/bias_control_medium/evaluations/gate42/fig_histogram_overlay.png`
- `data/bias_control_medium/evaluations/gate42/fig_roc_p0_p1.png`
- `data/bias_control_medium/evaluations/gate42/fig_similarity_scatter.png`

### Logs / historiales

- `data/bias_control_medium/gate2_1000batches.log`
- `data/bias_control_medium/gate3c_training.log`
- `data/bias_control_medium/gate3d_training.log`
- `data/bias_control_medium/training_outputs/gate2/training_history.json`
- `data/bias_control_medium/training_outputs/gate4_runA/training_history.json`
- `data/bias_control_medium/training_outputs/gate4_RB0/training_history.json`
- `data/bias_control_medium/training_outputs/gate4_R1rescue/training_history.json`

### Documentación de contexto

- `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/BIAS_CONTROL/AUDITORIA_BIAS_CONTROL_CODEX.md`
- `Documents/BIAS_CONTROL/BIAS_CONTROL_MEDIUM_TEST_RESULTS.md`
- `Documents/BIAS_CONTROL/BIAS_CONTROL_FAST_TEST_RESULTS.md`
- `Documents/BIAS_CONTROL/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md` (Claude)

---

## 3) Estado por gate (síntesis ejecutiva)

## Gate 0

Estado: completado y sano.  
Lectura: no hay evidencia de problema de integridad que invalide el pipeline.

## Gate 1

Estado: resultado bajo en fast test (esperable por régimen inicial), no bloqueante en práctica.  
Lectura: útil como sanity check, no como criterio principal de calidad cross-modal.

## Gate 2 (baseline operativo)

Estado: checkpoint de referencia confirmado (`epoch45`).  
Structured pool canónico:

- A2M R@10 = `0.344`
- M2A R@10 = `0.376`
- Hard neg (same-piece) = `0.804`
- Hard neg (random) = `0.870`

Logs/historial: pico de `val_gap` en epoch 45, luego degradación progresiva pese a menor loss.

Conclusión: baseline robusto para comparar todo lo posterior.

## Gate 3 (DANN)

Estado: cerrado por no mejorar robustamente al baseline Gate 2.  
Comparativa estructurada (6 checkpoints en `comparison_summary.json`) consistente con esta decisión.

Dato relevante:

- `runC_best_ep4` logra buen punto local (A2M 0.346, M2A 0.392, hard_neg 0.812),
- pero no hay mejora estable/sostenida del régimen DANN completo.

Conclusión: línea DANN no justifica inversión principal ahora.

## Gate 4 / 4.1

Resultado estructurado (canónico):

- Gate 2: A2M `0.344`, M2A `0.376`, hard_neg `0.804`
- RB0: A2M `0.302`, M2A `0.382`, hard_neg `0.776`
- RA5: A2M `0.314`, M2A `0.406`, hard_neg `0.790`
- R1rescue ep5: A2M `0.310`, M2A `0.402`, hard_neg `0.788`

Lectura:

- Señal marginal en dirección M2A, pero pérdida clara de A2M vs Gate 2.
- No se alcanza umbral fuerte para promoción de rama 4.1.

Conclusión: cierre disciplinado de Gate 4.1 fue correcto.

## Gate 6 (DEC-005, diagnóstico)

Artefactos completos presentes:

- `layer_drift.json`
- `multigate_embeddings.npz`
- `hubness_analysis.json`
- 5 figuras (UMAP, bridges, heatmaps, hubness, similarity)

Hallazgos cuantitativos críticos:

1. `Audio Encoder` efectivamente congelado en fine-tuning (drift 0 en backbone).
2. Drift dominante en proyecciones + lado MIDI (asimetría de adaptación).
3. Separación de similitud cae respecto a Gate 2:
   - Gate2: ~0.479
   - RB0: ~0.396
   - RA5: ~0.419
   - R1: ~0.395

Conclusión: explicación causal de la degradación A2M queda bien sustentada.

## Gate 4.2 pre-red (H4.2-6)

Resultado en `h426_prered_results.json`:

- P0_oracle: AUC `0.5592`, delta_sim `0.0341` -> NO-GO
- P1_real: AUC `0.5018`, delta_sim `-0.0041`, p `0.7155` -> NO-GO

Conclusión: hipótesis dual-domain por este extractor queda descartada en su forma actual.

## Gate 5

No aparece como línea ejecutada/cerrada en este corte. Sigue en modo opcional/hold.

---

## 4) Conclusiones propias (Codex) tras revisar todo + informe de Claude

1. Coincido con Claude en el núcleo causal: el problema no es únicamente “ratios sí/no”, sino el régimen de fine-tuning asimétrico (audio backbone no adaptando).
2. El cierre de Gate 4.1 fue correcto metodológicamente.
3. DEC-005 produjo evidencia de calidad y suficiente para decidir siguiente iteración sin “seguir probando a ciegas”.
4. El frente de visualizaciones YA tiene material fuerte; el problema ahora no es generar más por generar, sino ordenar, curar y convertir en narrativa reproducible.
5. Para comunicación externa/interna, hoy falta principalmente “producto documental visual” (curaduría + navegación + snapshots estables), no compute adicional inmediato.

---

## 5) Qué falta para dar por “cerrado y mostrable” este ciclo

Aunque los artefactos técnicos existen, falta empaquetado curado dentro de `Documents/BIAS_CONTROL`.

Pendientes funcionales:

1. Duplicado consistente de resultados clave desde `data/.../evaluations/*`.
2. Índice visual único con narrativa por etapa.
3. Estructura clara para doble uso:
   - trabajo técnico diario,
   - presentación/demo.
4. Regla de sincronización que evite divergencia entre `data/` y `Documents/`.

---

## 6) Plan detallado de curaduría visual y duplicado en `Documents/BIAS_CONTROL`

## 6.1 Principios de diseño

1. Navegación corta (máximo 2 niveles hasta llegar a figura/dato clave).
2. Separar “fuente cruda” de “curado para lectura”.
3. Mantener nombres estables para links desde README/estado/roadmap.
4. Incluir piezas visuales “impactantes” sin perder trazabilidad técnica.

## 6.2 Estructura objetivo (propuesta)

```text
Documents/BIAS_CONTROL/
  RESULTADOS_CURADOS/
    _INDEX_VISUAL.md
    _SNAPSHOT_DEC005.md

    GATE2_BASELINE/
      datos/
        structured_pool_epoch45.json
      visuales/
        (opcional: tabla-resumen renderizada en md)

    GATE3_DANN_CIERRE/
      datos/
        comparison_summary.json
      visuales/
        (figuras comparativas si aplica)

    GATE4_41_CAUSAL/
      datos/
        RA5_ep5.json
        RB0_ep5.json
        R1rescue_ep5.json
        R1rescue_best.json
      visuales/
        (tabla causal y delta S/H en markdown)

    GATE6_RETROANALYSIS/
      datos/
        layer_drift.json
        hubness_analysis.json
      visuales/
        fig_umap_multigate.png
        fig_bridges_multigate.png
        fig_heatmaps_multigate.png
        fig_hubness_distribution.png
        fig_similarity_distributions.png

    GATE42_PRERED/
      datos/
        h426_prered_results.json
      visuales/
        fig_histogram_overlay.png
        fig_roc_p0_p1.png
        fig_similarity_scatter.png

    EXHIBIT/
      HERO_01_umap_bridges.png
      HERO_02_hubness_similarity.png
      HERO_03_prered_roc_scatter.png
```

## 6.3 Narrativa visual propuesta

### `_INDEX_VISUAL.md`

Debe incluir:

1. “Estado en 30 segundos” (tabla de 6-8 métricas críticas).
2. “Recorrido recomendado” en 5 pasos.
3. Links directos a visuales y JSON fuente.
4. Glosario corto (A2M, M2A, hard-neg, separation, hubness).

### `_SNAPSHOT_DEC005.md`

Documento de corte para decisión:

1. Qué hipótesis cayeron.
2. Qué hipótesis sobreviven.
3. Qué ejecutar después (sin ambigüedad).

### `EXHIBIT/`

Composiciones visuales “showcase” con captions breves y contundentes:

1. Antes/después Gate2 vs fine-tuned (UMAP + bridges).
2. Hubness + separación en una sola lámina.
3. P0/P1 pre-red con veredicto NO-GO visual.

---

## 7) Flujo operativo para mantenerlo dinámico

## 7.1 Pipeline de sincronización (recomendado)

Crear un script de sync (propuesto):

- `scripts/sync_bias_control_curated_results.py`

Responsabilidades:

1. Crear árbol `RESULTADOS_CURADOS` si no existe.
2. Copiar whitelist de artefactos desde `data/bias_control_medium/evaluations/*`.
3. Regenerar `_INDEX_VISUAL.md` y `_SNAPSHOT_DEC005.md`.
4. Verificar que no haya links rotos.
5. Emitir resumen de sincronización con timestamp.

## 7.2 Cuándo correrlo

1. Al cerrar cualquier run/eval importante.
2. Antes de actualizar README/Estado Actual.
3. Antes de commit de documentación de milestone.

---

## 8) Criterios de aceptación (Definition of Done)

1. `RESULTADOS_CURADOS` existe y contiene Gate2, Gate3, Gate4.1, Gate6, Gate4.2.
2. Cada bloque tiene `datos/` y `visuales/` navegables.
3. `_INDEX_VISUAL.md` permite recorrer todo sin abrir 20 archivos sueltos.
4. README y `Proyecto_Estado_Actual.md` linkean al índice curado.
5. La historia DEC-005 queda entendible sin leer logs crudos.

---

## 9) Riesgos y mitigaciones

1. Riesgo: divergencia entre `data/` y `Documents/`.
   - Mitigación: sync script + whitelist + timestamp.
2. Riesgo: sobrecargar docs con duplicados enormes.
   - Mitigación: copiar solo artefactos de decisión; mantener `multigate_embeddings.npz` fuera del curado visual salvo referencia.
3. Riesgo: estetizar de más y perder rigor.
   - Mitigación: toda lámina visual linkea al JSON fuente.

---

## 10) Recomendación final

Recomendación ejecutiva:

1. No correr más training ahora.
2. Cerrar primero la curaduría visual/documental de DEC-005 en `Documents/BIAS_CONTROL/RESULTADOS_CURADOS`.
3. Con ese paquete consolidado, abrir la próxima DEC con foco en hipótesis que sobrevivieron (adapter/unfreezing controlado), no en repetir variantes descartadas.

Este orden maximiza claridad estratégica y evita iteración técnica desanclada de evidencia.

---

## 11) Anexo breve de métricas canónicas (snapshot)

| Checkpoint | A2M R@10 | M2A R@10 | HardNeg same-piece |
|---|---:|---:|---:|
| Gate2 | 0.344 | 0.376 | 0.804 |
| RB0 | 0.302 | 0.382 | 0.776 |
| RA5 | 0.314 | 0.406 | 0.790 |
| R1rescue ep5 | 0.310 | 0.402 | 0.788 |

Gate 4.2 pre-red:

- P0: AUC 0.5592, delta 0.0341 (NO-GO)
- P1: AUC 0.5018, delta -0.0041, p=0.7155 (NO-GO)

Gate 6 separación (correct vs incorrect):

- Gate2: ~0.479
- RB0: ~0.396
- RA5: ~0.419
- R1: ~0.395

