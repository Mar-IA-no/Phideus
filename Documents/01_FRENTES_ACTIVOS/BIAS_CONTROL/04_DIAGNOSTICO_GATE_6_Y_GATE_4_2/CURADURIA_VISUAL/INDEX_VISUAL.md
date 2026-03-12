# INDEX VISUAL — BIAS_CONTROL DEC-005

> Curaduría visual del ciclo diagnóstico DEC-005 (2026-02-11).
> Todo lo que necesitas para entender qué pasó, qué aprendimos, y qué sigue.

---

## Estado en 30 segundos

| Aspecto | Metric | Gate 2 (baseline) | Mejor fine-tuned | Veredicto |
|---------|--------|-----------------:|------------------:|-----------|
| Retrieval A2M | R@10 | **0.344** | 0.314 (RA5) | Fine-tuning degrada A2M |
| Retrieval M2A | R@10 | 0.376 | **0.406** (RA5) | Mejora marginal M2A |
| Hard negatives | same-piece | **0.804** | 0.790 (RA5) | Pérdida neta |
| Separación sim | correct-incorrect | **0.479** | 0.419 (RA5) | -12.5% de separación |
| Bridge distance | UMAP mean dist | **3.27** | 4.68 (R1) | Puentes +43% más largos |
| Audio drift | rel_change | 0% | 0% | Audio encoder CONGELADO |
| MIDI drift | rel_change | — | 8-12% | Solo MIDI cambia |
| Pre-red H4.2-6 | AUC P0/P1 | — | 0.559 / 0.502 | NO-GO (chance level) |

**Lectura rápida:** El fine-tuning solo mueve el lado MIDI del modelo. El encoder de audio está completamente congelado. Esto causa que los puentes cross-modal se alarguen y la separación se degrade. La extracción de ratios desde audio (CQT) no funciona ni bajo condiciones oracle.

---

## Recorrido recomendado (5 pasos)

### Paso 1: Entiende la asimetría de drift

El hallazgo más importante de DEC-005: el audio encoder tiene **cero drift** en todos los checkpoints fine-tuned.

- **Datos:** [`layer_drift.json`](../../../data/bias_control_medium/evaluations/gate6/layer_drift.json)
- **Visual compuesto:** [HERO_01 — Drift + UMAP](EXHIBIT/HERO_01_drift_frozen.png)

| Módulo | RB0 drift | RA5 drift | R1 drift |
|--------|----------:|----------:|---------:|
| Audio CNN | 0.0% | 0.0% | 0.0% |
| Audio Transformer | 0.0% | 0.0% | 0.0% |
| Audio Projection | 11.3% | 13.9% | 11.3% |
| MIDI Embedding | 3.5% | 5.0% | 3.5% |
| MIDI Transformer | 8.3% | 12.1% | 8.5% |
| MIDI Projection | 11.8% | 13.1% | 10.8% |

### Paso 2: Observa el efecto en el espacio de embeddings

Los puentes cross-modal se alargan con el fine-tuning. La separación entre pares correctos e incorrectos se degrada.

- **Datos:** [`hubness_analysis.json`](../../../data/bias_control_medium/evaluations/gate6/hubness_analysis.json)
- **Visual compuesto:** [HERO_02 — Bridges + Separation](EXHIBIT/HERO_02_bridges_separation.png)
- **Figuras individuales:**
  - [UMAP 2x2](../../../data/bias_control_medium/evaluations/gate6/fig_umap_multigate.png)
  - [Bridges 2x2](../../../data/bias_control_medium/evaluations/gate6/fig_bridges_multigate.png)
  - [Heatmaps 1x4](../../../data/bias_control_medium/evaluations/gate6/fig_heatmaps_multigate.png)
  - [Hubness distribution](../../../data/bias_control_medium/evaluations/gate6/fig_hubness_distribution.png)
  - [Similarity distributions](../../../data/bias_control_medium/evaluations/gate6/fig_similarity_distributions.png)

| Checkpoint | Bridge mean dist | Separation (correct-incorrect) |
|------------|----------------:|-------------------------------:|
| Gate 2 | **3.27** | **0.479** |
| RB0 | 4.50 (+37%) | 0.396 (-17%) |
| RA5 | 4.47 (+37%) | 0.419 (-13%) |
| R1 | 4.68 (+43%) | 0.395 (-18%) |

### Paso 3: Verifica que H4.2-6 es inviable

La extracción de ratios armónicos desde audio real via CQT no puede discriminar pares alineados de pares random. Ni siquiera bajo condiciones oracle (audio sintetizado desde MIDI).

- **Datos:** [`h426_prered_results.json`](../../../data/bias_control_medium/evaluations/gate42/h426_prered_results.json)
- **Visual compuesto:** [HERO_03 — Pre-Red NO-GO](EXHIBIT/HERO_03_prered_nogo.png)
- **Figuras individuales:**
  - [ROC P0/P1](../../../data/bias_control_medium/evaluations/gate42/fig_roc_p0_p1.png)
  - [Scatter aligned vs random](../../../data/bias_control_medium/evaluations/gate42/fig_similarity_scatter.png)
  - [Histogram overlay](../../../data/bias_control_medium/evaluations/gate42/fig_histogram_overlay.png)

| Fase | AUC | CI 95% | delta_sim | Umbral GO | Veredicto |
|------|----:|-------:|----------:|----------:|-----------|
| P0 (oracle) | 0.559 | [0.480, 0.631] | +0.034 | >= 0.80 | **NO-GO** |
| P1 (real) | 0.502 | [0.422, 0.588] | -0.004 | >= 0.70 | **NO-GO** |

### Paso 4: Revisa las métricas de Gate 4.1

Gate 4.1 se cerró por no superar al baseline. Los números confirman: el fine-tuning con ratios auxiliares no mejora el rendimiento neto.

- **Datos:**
  - [`RA5_ep5.json`](../../../data/bias_control_medium/evaluations/gate4/RA5_ep5.json)
  - [`RB0_ep5.json`](../../../data/bias_control_medium/evaluations/gate4/RB0_ep5.json)
  - [`R1rescue_ep5.json`](../../../data/bias_control_medium/evaluations/gate4/R1rescue_ep5.json)

| Checkpoint | A2M R@10 | M2A R@10 | Hard neg | dS vs Gate 2 |
|------------|--------:|---------:|---------:|-------------:|
| Gate 2 | **0.344** | 0.376 | **0.804** | — |
| RB0 | 0.302 | 0.382 | 0.776 | -0.8pp |
| RA5 | 0.314 | **0.406** | 0.790 | +0.8pp |
| R1 rescue | 0.310 | 0.402 | 0.788 | +0.6pp |

### Paso 5: Lee el snapshot de decisión

El documento de corte que explica qué cayó, qué sobrevive, y qué se ejecuta a continuación.

- [SNAPSHOT_DEC005.md](SNAPSHOT_DEC005.md)
- [Informe completo (1067 líneas)](../INFORME_DEC005_DIAGNOSTICO_COMPLETO.md)

---

## Inventario completo de artefactos

### JSONs (datos fuente, NO duplicados)

| Archivo | Ubicación | Contenido |
|---------|-----------|-----------|
| `structured_pool_epoch45.json` | `data/bias_control_medium/evaluations/` | Gate 2 baseline |
| `RA5_ep5.json` | `evaluations/gate4/` | Gate 4 Run A |
| `RB0_ep5.json` | `evaluations/gate4/` | Gate 4 control |
| `R1rescue_ep5.json` | `evaluations/gate4/` | Gate 4 R1 rescue |
| `layer_drift.json` | `evaluations/gate6/` | Drift por capa |
| `hubness_analysis.json` | `evaluations/gate6/` | Hubness + per-piece |
| `h426_prered_results.json` | `evaluations/gate42/` | P0/P1 pre-red |

### Figuras (generadas por scripts DEC-005)

| Figura | Ubicación | Script |
|--------|-----------|--------|
| `fig_umap_multigate.png` | `evaluations/gate6/` | visualize_embeddings_multigate.py |
| `fig_bridges_multigate.png` | `evaluations/gate6/` | visualize_embeddings_multigate.py |
| `fig_heatmaps_multigate.png` | `evaluations/gate6/` | visualize_embeddings_multigate.py |
| `fig_hubness_distribution.png` | `evaluations/gate6/` | analyze_hubness.py |
| `fig_similarity_distributions.png` | `evaluations/gate6/` | analyze_hubness.py |
| `fig_histogram_overlay.png` | `evaluations/gate42/` | h426_prered_test.py |
| `fig_roc_p0_p1.png` | `evaluations/gate42/` | h426_prered_test.py |
| `fig_similarity_scatter.png` | `evaluations/gate42/` | h426_prered_test.py |

### Héroes compuestos (nuevos, en EXHIBIT/)

| Héroe | Archivo | Composición |
|-------|---------|-------------|
| HERO_01 | `EXHIBIT/HERO_01_drift_frozen.png` | Drift table + UMAP Gate2 vs R1 |
| HERO_02 | `EXHIBIT/HERO_02_bridges_separation.png` | Bridges + similarity distributions |
| HERO_03 | `EXHIBIT/HERO_03_prered_nogo.png` | ROC + scatter + NO-GO verdict |

---

## Glosario

| Término | Significado |
|---------|-------------|
| **A2M** | Audio-to-MIDI retrieval. Dado un query de audio, buscar el segmento MIDI correspondiente. |
| **M2A** | MIDI-to-Audio retrieval. Dirección inversa. |
| **R@10** | Recall at 10. Proporción de queries cuyo match correcto aparece en el top-10 del ranking. |
| **Hard neg** | Hard negative accuracy. Capacidad de distinguir el segmento correcto de otros segmentos de la misma pieza. |
| **Separation** | Diferencia entre la similitud coseno media de pares correctos vs incorrectos. Mayor = mejor. |
| **Bridge dist** | Distancia euclidiana en espacio UMAP entre pares audio-MIDI del mismo segmento. Mayor = peor alineación. |
| **Drift** | Cambio relativo en los pesos del modelo respecto al baseline Gate 2. 0% = sin cambio. |
| **Hubness** | Fenómeno donde algunos embeddings MIDI son vecinos cercanos de muchos queries audio (hubs). Skewness alta = más hubs. |
| **Pre-red** | Test previo a reducción completa. Evalúa si una hipótesis tiene señal mínima antes de invertir en training. |
| **AUC** | Area Under the ROC Curve. 0.5 = chance, 1.0 = perfecto. |

---

> Todos los paths son relativos a la raíz pública del repositorio (`<repo-root>/`).
> Generado: 2026-02-11 | DEC-005 | Claude + Codex
