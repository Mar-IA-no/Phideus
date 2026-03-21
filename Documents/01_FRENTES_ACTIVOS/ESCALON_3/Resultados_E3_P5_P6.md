# Resultados E3-P5 / E3-P6

Fecha de consolidacion: `2026-03-21`

## Proposito

Este documento fija la lectura canónica de la primera pasada geométrica completa de Escalón 3.

La pregunta ya no es si `P5` y `P6` "merecían correrse". Esa decisión ya quedó absorbida por el frente. La pregunta correcta ahora es otra:

- qué dejó `P5` cuando la geometría no plana se introduce de forma mixta;
- qué dejó `P6` cuando el espacio pasa a ser toroidal puro;
- y cómo queda reordenada la línea `P2 -> P5 -> P6` una vez corregida la selección de checkpoints por evaluación estructurada.

## Artefactos auditados

### Baselines `L0`

- `data/escalon3/p2_baseline_seed42/final_results.json`
- `data/escalon3/p2_cqtshift_seed42/final_results.json`

### Línea geométrica mixta

- `data/escalon3/p5_mixed_flat_seed42/final_results_e40_structured.json`
- `data/escalon3/p5_mixed_cqtshift_seed42/final_results.json`
- `data/escalon3/p5_mixed_flat_seed42/checkpoint_sweep.json`
- `data/escalon3/p5_mixed_cqtshift_seed42/checkpoint_sweep.json`
- `data/escalon3/p5_mixed_flat_seed42/euclidean_weight_sweep.json`
- `data/escalon3/p5_mixed_cqtshift_seed42/euclidean_weight_sweep.json`
- `data/escalon3/p5_mixed_flat_seed42/torus_ablation.json`
- `data/escalon3/p5_mixed_cqtshift_seed42/torus_ablation.json`

### Línea toroidal pura

- `data/escalon3/p6_tvicreg_flat_seed42/final_results_e40_structured.json`
- `data/escalon3/p6_tvicreg_flat_seed42/final_results_e50_trainselect.json`
- `data/escalon3/p6_tvicreg_flat_seed42/checkpoint_sweep.json`
- `data/escalon3/p6_tvicreg_cqtshift_seed42/final_results_e30_structured.json`
- `data/escalon3/p6_tvicreg_cqtshift_seed42/final_results_e50_trainselect.json`
- `data/escalon3/p6_tvicreg_cqtshift_seed42/checkpoint_sweep.json`

## Tabla corta

Checkpoint estructuralmente correcto por brazo:

- `P2-flat` = `E40`
- `P2-cqtshift` = `E30`
- `P5-flat` = `E40`
- `P5-cqtshift` = `E30`
- `P6-flat` = `E40`
- `P6-cqtshift` = `E30`

| Brazo | IID `S` | `scale_ood S` | `equiv_ood S` | `render_noisy` | `render_thick` | `sil_test` | `sil_equiv_ood` |
|------|---------|---------------|---------------|----------------|----------------|------------|-----------------|
| `P2-flat` | `0.583` | `0.096` | `0.240` | `0.585` | `0.506` | `0.960` | `-` |
| `P2-cqtshift` | `0.515` | `0.476` | `0.458` | `0.515` | `0.344` | `1.000` | `-` |
| `P5-flat` | `0.552` | `0.114` | `0.278` | `0.581` | `0.498` | `0.928` | `0.104` |
| `P5-cqtshift` | `0.510` | `0.508` | `0.472` | `0.515` | `0.356` | `0.993` | `0.965` |
| `P6-flat` | `0.477` | `0.068` | `0.228` | `0.463` | `0.367` | `0.615` | `0.052` |
| `P6-cqtshift` | `0.515` | `0.438` | `0.434` | `0.506` | `0.319` | `0.985` | `0.975` |

## Observaciones

### 1. `P2-flat` sigue siendo el mejor baseline general de retrieval

`P2-flat` conserva la mejor referencia `IID` del frente:

- `IID S = 0.583`
- `render_thick = 0.506`
- `silhouette_combined = 0.960`

La lectura vigente no cambia: el baseline plano sigue siendo la referencia general más fuerte de Escalón 3.

### 2. `P5-flat` no reemplaza a `P2-flat`, pero tampoco fue vacío

`P5-flat` no gana la línea general del frente:

- cae en `IID` frente a `P2-flat` (`0.552 < 0.583`);
- no mejora `render_thick`;
- no se vuelve el nuevo baseline canónico.

Pero la corrida deja dos señales reales:

- sube `scale_ood` y `equiv_ood` frente a `P2-flat`;
- y la ablation mostró que la rama toroidal sí aporta señal causal, porque `torus_shuffle` degrada de manera visible el retrieval.

La lectura correcta no es "ganó" ni "fue inútil". La lectura correcta es:

- negativo como reemplazo de baseline general;
- positivo como evidencia de que la rama toroidal puede contribuir.

### 3. `P5-cqtshift` es el mejor brazo geométrico/OOD del corte

En la comparación estructuralmente correcta, `P5-cqtshift` queda arriba de todos los brazos geométricos en las métricas OOD primarias:

- `scale_ood S = 0.508`
- `equiv_ood S = 0.472`

Además:

- mantiene `IID` competitivo (`0.510`, muy cerca de `P2-cqtshift`);
- conserva `render_noisy` en la misma banda del baseline `cqtshift`;
- y organiza el toro con mucha claridad (`sil_test = 0.993`, `sil_equiv_ood = 0.965`).

Eso no lo convierte en "ganador universal" del frente. `P2-flat` sigue siendo mejor en `IID` y `render_thick`. Pero sí lo convierte en el mejor brazo geométrico/OOD disponible bajo la receta actual.

### 4. `P6-flat` es un negativo claro

`P6-flat` falla en casi todo lo que importaba sostener:

- `IID` peor que `P2-flat` y `P5-flat`;
- `scale_ood = 0.068`
- `equiv_ood = 0.228`
- `sil_equiv_ood = 0.052`

La hipótesis fuerte del toro puro no recibió apoyo en el brazo plano.

### 5. `P6-cqtshift` organiza muy bien el toro, pero no gana retrieval OOD

`P6-cqtshift` deja una señal geométrica fuerte:

- `sil_test = 0.985`
- `sil_equiv_ood = 0.975`

Pero esa organización no se traduce en una mejora del retrieval OOD sobre `P5-cqtshift`:

- `scale_ood 0.438 < 0.508`
- `equiv_ood 0.434 < 0.472`

Y el ajuste de checkpoint estructurado no rescata la línea. Al contrario: el checkpoint estructuralmente correcto (`E30`) queda todavía peor en OOD que el `E50` elegido por trainselect.

## Hipotesis compatibles

- La geometría mixta parece ofrecer una forma más útil de acoplar periodicidad y residuo que el toro puro, al menos bajo esta receta.
- El encoder `cqtshift` sigue siendo el factor más favorable cuando la pregunta principal es invariancia de ratio del lado audio.
- Una organización toroidal más "limpia" no garantiza por sí sola mejor retrieval cross-modal.

## Inferencias válidas

1. `P2-flat` queda confirmado como baseline canónico general de Escalón 3.
2. `P5-cqtshift` pasa a ser el mejor brazo geométrico/OOD del frente en esta primera pasada completa.
3. `P6` no queda refutado en sentido lógico, pero **no es el ganador bajo la receta actual**.
4. La hipótesis fuerte "toro puro > mixed" no recibe apoyo en este corte.

## Decisión operativa vigente

- `P2-flat` sigue siendo `L0-Flat Canonical`.
- `P2-cqtshift` sigue siendo `L0-Shift Ratio-Aware`.
- `P5-cqtshift` queda como mejor brazo geométrico/OOD actual.
- `P6` queda documentado como hipótesis pura interesante, pero no ganadora en esta receta.

## Lo que no corresponde decir

- no corresponde decir que `P5` "ganó Escalón 3" en general;
- no corresponde decir que `P6` quedó definitivamente refutado en todas sus variantes posibles;
- no corresponde confundir estructura toroidal muy limpia con mejor retrieval;
- no corresponde volver a leer `P4` como si hubiera podido resolver por sí solo esta discusión.

La lectura correcta es más fina: la línea geométrica sí produjo información útil, pero esa información favorece hoy a la geometría mixta con encoder `cqtshift`, no al toro puro.
