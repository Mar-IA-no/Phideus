# INFORME + PLAN DE CURADURÍA VISUAL BIAS_CONTROL (Post DEC-005)

Fecha: 2026-02-11
Autor: Codex (secciones 1-5, 11) + Claude (secciones 6-10 revisadas)
Contexto: cierre de ejecución diagnóstica Gate 6 + Gate 4.2 (DEC-005)

> [!NOTE]
> Addendum de vigencia (2026-02-17): la curaduría sigue vigente como base de orden documental.
> El estado experimental actual cerró Gate 4.3 y consolidó resultados canónicos completos del bloque.
> Documento de estado activo: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`.

---

## 1) Objetivo de este documento

Este documento integra dos cosas en un solo entregable:

1. Auditoría técnica actualizada de resultados/logs de `BIAS_CONTROL` (gates ejecutados).
2. Plan detallado para dejar todos los resultados relevantes duplicados, ordenados y presentables dentro de `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL`, con navegación pensada tanto para trabajo técnico como para exhibición.

Además, incorpora y contrasta explícitamente el informe de Claude:

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`.

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

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/AUDITORIA_BIAS_CONTROL_CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/BIAS_CONTROL_MEDIUM_TEST_RESULTS.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/BIAS_CONTROL_FAST_TEST_RESULTS.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md` (Claude)

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

Aunque los artefactos técnicos existen, falta empaquetado curado dentro de `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL`.

Pendientes funcionales:

1. Duplicado consistente de resultados clave desde `data/.../evaluations/*`.
2. Índice visual único con narrativa por etapa.
3. Estructura clara para doble uso:
   - trabajo técnico diario,
   - presentación/demo.
4. Regla de sincronización que evite divergencia entre `data/` y `Documents/`.

---

## 6) Plan de curaduría visual — REVISADO (Claude, 2026-02-11)

### 6.0 Cambios respecto a la propuesta original de Codex

La propuesta original (secciones 6-10 pre-revisión) planteaba:
- Duplicar JSONs desde `data/` a `Documents/` en subdirectorios por gate
- Crear un script de sincronización (`sync_bias_control_curated_results.py`)
- Estructura de 5 subdirectorios con `datos/` + `visuales/` cada uno

**Problemas identificados:**
1. **Duplicación de JSONs = fuente de divergencia.** Si alguien edita el JSON en `data/` y no corre el sync, `Documents/` queda desactualizado. Peor aún: ¿cuál es la fuente de verdad?
2. **Sync script = sobreingeniería.** Para un proyecto con un equipo de 2 agentes + 1 humano, un script de sync es mantenimiento gratuito sin beneficio real.
3. **Subdirectorios por gate redundantes.** Los datos ya están organizados en `data/bias_control_medium/evaluations/{gate2,gate3,gate4,gate6,gate42}/`. Duplicar esa estructura en `Documents/` solo agrega ruido.

**Decisión: ENLAZAR en vez de COPIAR.**

Los markdown (INDEX, SNAPSHOT) contendrán links relativos a los archivos en `data/`. Las únicas imágenes nuevas son los 3 héroes compuestos en `EXHIBIT/`, que son artefactos nuevos (no copias).

### 6.1 Principios de diseño (revisados)

1. **Cero duplicación de datos.** Los JSONs y figuras viven en `data/`. Los docs apuntan a ellos.
2. **Estructura plana.** Máximo 1 nivel de profundidad (`CURADURIA_VISUAL/EXHIBIT/`).
3. **3 héroes compuestos = artefactos NUEVOS.** No son copias: son composiciones de múltiples figuras con datos incrustados, captions, y narrativa visual.
4. **Links estables.** Los paths relativos funcionan desde cualquier viewer markdown.

### 6.2 Estructura objetivo (implementada)

```text
Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/
  INDEX_VISUAL.md                      ← navegación + "30 segundos" + links a data/
  SNAPSHOT_DEC005.md                   ← decisiones: qué cayó, qué sobrevive, qué sigue
  EXHIBIT/
    HERO_01_drift_frozen.png           ← drift table + UMAP Gate2 vs R1
    HERO_02_bridges_separation.png     ← bridges + similarity distributions
    HERO_03_prered_nogo.png            ← ROC + scatter con veredicto NO-GO
```

**Total: 2 archivos markdown + 3 imágenes compuestas. Sin JSONs duplicados. Sin scripts de sync.**

### 6.3 Especificación de los 3 héroes

**Estilo unificado:** fondo oscuro `#0a0a0a`, audio `#00e5ff`, MIDI `#ff1493`, texto con glow.

#### HERO_01: "Audio Encoder Frozen"

| Panel izquierdo | Panel derecho |
|-----------------|---------------|
| Tabla de drift renderizada visualmente | UMAP Gate2 vs R1 (2 paneles, no 4) |
| Audio CNN: 0%, Audio Transformer: 0% | Comparación directa del espacio |
| MIDI Transformer: 8-12%, Projections: 11-14% | Baseline vs peor fine-tuned |
| Color: verde=0%, rojo=alto drift | Audio cyan, MIDI magenta |

**Caption:** *"Audio encoder frozen. Fine-tuning only moves MIDI side."*

#### HERO_02: "Bridges + Separation"

| Panel superior | Panel inferior |
|----------------|----------------|
| Bridges Gate2 (mean=3.27) vs R1 (mean=4.68) | Similarity distributions Gate2 (sep=0.479) vs R1 (sep=0.395) |
| 2 paneles lado a lado | 2 paneles: correct vs incorrect histograms |
| Líneas coloreadas por distancia | Overlap visual del deterioro |

**Caption:** *"Fine-tuning lengthens cross-modal bridges and degrades separation."*

#### HERO_03: "Pre-Red NO-GO"

| Panel izquierdo | Panel derecho |
|-----------------|---------------|
| ROC P0/P1 con diagonal de chance | Scatter aligned vs random |
| AUC P0=0.559, P1=0.502 anotados | delta_sim anotado |
| Línea roja de threshold GO (0.80) | Veredicto NO-GO prominente |

**Caption:** *"CQT cannot discriminate, even under oracle conditions."*

### 6.4 Contenido de INDEX_VISUAL.md

1. **"Estado en 30 segundos"**: tabla con 8 métricas críticas (A2M, M2A, hard_neg, separation, bridge dist, drift, AUC P0/P1)
2. **"Recorrido recomendado"**: 5 pasos con links directos
3. **Links a todos los JSONs fuente** en `data/bias_control_medium/evaluations/`
4. **Links a todas las figuras** existentes
5. **Glosario**: A2M, M2A, hard-neg, separation, hubness, drift, pre-red

### 6.5 Contenido de SNAPSHOT_DEC005.md

1. Qué hipótesis cayeron (H4.2-6, con números)
2. Qué hipótesis sobreviven (H4.2-2 adapter, H4.2-1 audio-only)
3. Qué hallazgo cambió el panorama (audio encoder frozen)
4. Qué ejecutar después (sin ambigüedad)
5. Tabla de decisión explícita

---

## 7) Criterios de aceptación (Definition of Done)

1. `CURADURIA_VISUAL/` existe con INDEX_VISUAL.md, SNAPSHOT_DEC005.md, y 3 héroes en EXHIBIT/.
2. INDEX_VISUAL.md permite recorrer todo DEC-005 sin abrir más de 2 archivos.
3. Los 3 héroes son composiciones nuevas que cuentan la historia visual completa.
4. Todos los links en los markdown apuntan a archivos existentes.
5. No hay JSONs ni PNGs duplicados desde `data/`.

---

## 8) Riesgos y mitigaciones (revisados)

1. **Riesgo: links rotos si se mueven archivos en `data/`.**
   - Mitigación: paths relativos consistentes. Si se reorganiza `data/`, actualizar INDEX_VISUAL.md (un solo archivo).
2. **Riesgo: estetizar de más y perder rigor.**
   - Mitigación: toda lámina visual incluye referencia al JSON fuente en su caption.
3. **Riesgo: héroes desactualizados tras nuevos experimentos.**
   - Mitigación: los héroes son DEC-005-specific. Nuevos ciclos producen nuevos héroes.

---

## 9) Recomendación final

Recomendación ejecutiva (sin cambios respecto a Codex):

1. No correr más training ahora.
2. Cerrar primero la curaduría visual/documental de DEC-005.
3. Con ese paquete consolidado, abrir la próxima DEC con foco en hipótesis que sobrevivieron (adapter/unfreezing controlado).

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
