# Resultados E3-P4

Fecha de corte: `2026-03-21`

## Propósito

Este documento fija la lectura operativa de `E3-P4` después de correr el régimen de probes sobre los dos baselines `L0` ya congelados:

- `P2-flat` como baseline canónico de retrieval general;
- `P2-cqtshift` como baseline alternativo ratio-aware del lado audio.

La conclusión importante no es solo qué probe rindió mejor. La conclusión importante es qué parte de la hipótesis de `activation` queda realmente resuelta por una lectura post-hoc sobre un latente plano y qué parte sigue abierta para geometrías no planas.

## Artefactos auditados

- `data/escalon3/p4_flat_seed42/`
- `data/escalon3/p4_cqtshift_seed42/`

Archivos principales:

- `probe_results.json`
- `probe_family_summary.csv`
- `probe_exemplars.json`

## Resultado resumido

### `L0-Flat`

Métrica primaria `scale_ood a2i ratio_hit@10`:

- `cosine = 0.138`
- `phi = 0.174` (`+0.036`)
- `noble = 0.168` (`+0.030`)
- `rational_simple_mean = 0.1655`
- `rational_complex_mean = 0.1693`
- `irrational_random_1 = 0.172`
- `irrational_random_2 = 0.168`
- `irrational_random_3 = 0.166`

Métrica primaria `equiv_ood a2i equiv_hit@10`:

- `cosine = 0.426`
- `phi = 0.420`
- `noble = 0.420`
- `rational_simple_mean = 0.422`

Lectura:

- sí aparece una mejora marginal de traversal sobre `cosine` en `scale_ood a2i`;
- pero `phi` no queda separado de manera robusta de otros irracionales o de la familia racional compleja;
- y esa mejora no se sostiene en `equiv_ood a2i`.

### `L0-CQTShift`

En las métricas primarias:

- `scale_ood a2i = 1.000` para todas las familias;
- `equiv_ood a2i = 1.000` para todas las familias.

Lectura:

- el espacio ya está tan alineado con el target que la familia de probe deja de ser discriminativa en `hit@10`;
- el frente no obtiene de `cqtshift` una respuesta sobre “qué probe gana”, sino un control donde el problema primario queda saturado.

## Lectura metodológica

### Observación

- sobre el latente plano, algunos traversals mejoran un poco a `cosine` en un slice concreto;
- `phi` no se vuelve claramente especial frente a otros irracionales;
- sobre `cqtshift`, las métricas primarias saturan.

### Hipótesis compatible

- una lectura por traversal puede ayudar más que `cosine` en ciertos regímenes;
- pero la primera corrida de `P4` no muestra evidencia suficiente para decir que `phi` tenga una ventaja diferencial fuerte por sí solo.

### Inferencia válida

`P4` deja un resultado informativo, pero **no decisivo contra la línea geométrica**.  
Lo que quedó evaluado acá fue la lectura sobre embeddings entrenados en geometrías planas. Eso no agota la hipótesis de que una geometría de storage no plana pueda cambiar la respuesta de los probes.

## Decisión operativa vigente

- `P4` queda documentado como resultado útil y honesto, pero no como veto suficiente sobre `P5/P6`.
- El frente sigue con `P5` y `P6` completos, por decisión explícita del programa.
- La lectura correcta del corte ya no es “`phi` fracasó, por lo tanto no vale la pena una geometría no plana”, sino esta:
  - `P4` no mostró una ventaja diferencial fuerte de `phi` en lectura post-hoc sobre latente plano;
  - por eso la hipótesis geométrica fuerte ya no puede juzgarse solo desde `P4`;
  - y pasa a requerir evaluación directa en `P5/P6`.

## Postscriptum despues de `P5/P6`

La primera pasada geométrica completa ya fue corrida y deja una lectura útil sobre este documento:

- `P5-cqtshift` terminó siendo el mejor brazo geométrico/OOD del corte;
- `P6` no desplazó a `P5` bajo la receta actual;
- y por eso `P4` queda mejor ubicado exactamente en el lugar que este documento ya le asignaba:
  - resultado informativo sobre lectura en `L0`;
  - insuficiente por sí solo para cerrar o abrir toda la discusión geométrica.
