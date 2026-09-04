# Plan CPU — ablación mean-only del ranking de cola

**Fecha:** 2026-09-04
**Estado:** diseño congelado; implementación y ejecución pendientes
**Régimen:** diagnóstico post hoc sobre R366 y sus dos cohortes abiertas
**Autoridad:** adjudica si la corrección de cola debe participar del orden de acción; no valida no-daño, no confirma arquitectura ni decide GO/NO-GO

## Pregunta

R366 mostró que `mu + u_topology` no ordena de forma general mejor que
`mu + u_permuted` o `mu + u_public_base` bajo igual cantidad de acciones. Sin
embargo, todos esos scorers comparten `mu`, el predictor topology de media
congelado desde R360. Falta el control causal inmediato: preguntar si sumar la
cola topology mejora o empeora el ranking que ya entrega `mu`.

R367 no ajusta una cabeza nueva. Repite exactamente la propuesta de alpha y los
seis presupuestos R366, y añade dos ablations:

```text
score_mean_only    = mu_alpha_common
score_constant_tail = mu_alpha_common + u_constant_alpha_common
```

Se comparan con `score_topology_tail`, `score_public_base_tail` y los dieciséis
`score_permuted_tail` ya definidos. La pregunta primaria es
`topology_tail - mean_only`: un valor positivo favorece mean-only.

## Fuentes y freeze

Fuente inmediata: R366, manifest
`56fe363ca0c353e142673b8496502f5ccd974d6bec7b7881d0a5d3412166f28c`.
R366 ata R365 y las cohortes R361/R362 y R363/R364. La reconstrucción B debe
volver a igualar exactamente R364.

Se conserva sin cambios:

- `alpha_common = argmin(mu + u_public_base)` por vista;
- primer alpha como tie-break;
- ranking global sin labels IID/grouped;
- presupuestos `1%, 2%, 5%, 10%, 20%, 40%` con
  `ceil(fracción × n_views)` acciones;
- tie-break de score por índice de vista;
- bootstrap original por cohorte y slices IID/grouped/balanceado;
- ausencia de threshold, firewall y selección de presupuesto ganador.

## Estimandos

Por cohorte, brazo, presupuesto y slice se preservan efectos por master de:

1. `topology_tail - mean_only` — primario, aporte de cola topology al ranking;
2. `constant_tail - mean_only` — control de añadir una cola sin features;
3. `topology_tail - constant_tail` — aporte topology sobre el control mínimo;
4. `mean_only - identity` — utilidad del ranking común, sin atribución causal;
5. `mean_only - public_base_tail` y `mean_only - mean(permuted_tail)` — posición
   de la media frente a correcciones alternativas.

Los intervalos son pointwise. Se clasifica transporte con tolerancia `1e-12` y
se reporta el promedio descriptivo no ponderado entre los seis presupuestos.
No se corrige multiplicidad porque no se selecciona una celda para promoción;
los conteos describen la curva completa.

## Decisión arquitectónica que informa

- Si `topology_tail - mean_only` es positivo de forma transportable en una
  región amplia, la cola mejora calibración pero perjudica ranking: debe quedar
  en incertidumbre/abstención y `mu` ordenar la acción.
- Si es negativo y estable, la cola conserva un papel dual que necesitaría un
  freeze futuro.
- Si cambia de signo por presupuesto o cohorte, mezclar ambos objetivos en una
  suma escalar no está identificado; el siguiente candidato es separar cabeza
  de ranking y cabeza de riesgo.
- El resultado no atribuye por sí solo topology a `mu`: para eso harían falta
  controles de media permutada ya preservados desde R360.

## Artefactos y recursos

El output canónico será
`data/geometria_proporcional/proportional_graph_mean_only_ranking_ablation_v1/`.
Conservará scores, acciones, efectos por master, bootstraps, overlaps, análisis,
manifest, entorno y replay.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.
No consulta ni usa GPU; todo trabajo GPU permanece en cola hasta nueva orden.
