# Plan CPU — interfaz two-stage de ranking y elegibilidad

**Fecha:** 2026-09-04
**Estado:** diseño congelado; implementación y ejecución pendientes
**Régimen:** diagnóstico post hoc sobre dos cohortes abiertas
**Autoridad:** evalúa una distribución de responsabilidades; no acredita no-daño prospectivo, no promueve arquitectura ni decide GO/NO-GO

## Pregunta

R367 mostró que ordenar por `mu` supera a `mu + u_topology` en `50/72` celdas
de dos cohortes, mientras la cola gana `2/72`. La suma escalar obliga a un
estimador de incertidumbre a reordenar una media ya útil. R368 separa las dos
funciones:

```text
alpha_common = argmin(mu + u_public_base)
rank_score   = mu_alpha_common
eligible_f   = mu_alpha_common + u_f,alpha_common + q_f < 0
```

El ranking es idéntico para todas las familias. Constant, public-base, topology
y dieciséis topology-permuted sólo pueden filtrar vistas; no alteran el orden.

## Freeze y fuentes

Fuente inmediata: R367, manifest
`814aced29af86ddb0d0f4e39611ab2fc31058304b7397c27d9319327485e281c`,
que ata R366, R365 y las cohortes R361/R362 y R363/R364. Se reconstruyen las
predicciones preservadas sin fit, vistas ni solves nuevos.

Para aislar el filtro se conserva `alpha_common` public-base de R366/R367. No se
cambia todavía a `argmin(mu)`. El orden usa sólo `mu` y tie-break por índice.
Los presupuestos máximos son `1%, 2%, 5%, 10%, 20%, 40%` del total de vistas,
sin labels IID/grouped. Cada política toma, en orden, hasta
`ceil(fracción × n_views)` vistas que su filtro declara elegibles. Puede actuar
menos que el presupuesto; esa diferencia es parte del estimando de
elegibilidad, no una comparación de ranking.

## Calibración del filtro

En risk-calibration de cada cohorte y familia:

```text
score_f = (delta_real - mu)_alpha_common - u_f,alpha_common
q_f = conformal_order_0.90(score_f en IID)
```

La propuesta de alpha se congela antes de calibrar. La garantía permanece
marginal selected-action IID; no es cobertura condicional entre elegibles.
Se verifica `q_f <= q_simultaneous_f` sólo cuando ambas cantidades cubren el
mismo predictor y score; no se fuerza una comparación si la propuesta difiere.

## Policy selection y adjudicación

Policy-selection materializa cada familia×presupuesto y aplica el bootstrap
original sobre el daño medio IID. El firewall `upper95 <= 0` decide si esa
política completa pasa a adjudicación; rechazo produce identidad. Los controles
permutados se calibran y filtran individualmente antes de promediar outcomes.

Se preservan dos stages:

- `calibrated`: ranking fijo más filtro conformal, sin firewall agregado;
- `deployed`: la misma política después del firewall de policy-selection.

## Estimandos

Por cohorte, brazo, presupuesto, stage y slice IID/grouped/balanceado:

1. topology-filter menos identidad;
2. topology-filter menos mean-only sin filtro, con el mismo presupuesto máximo;
3. topology-filter menos constant-filter;
4. topology-filter menos public-base-filter;
5. topology-filter menos promedio topology-permuted-filter.

El quinto es el control incremental principal. Se reportan efectos por master,
intervalos pointwise con bootstrap original, transporte entre cohortes,
acciones solicitadas/realizadas, cobertura marginal, alphas, daño entre
actuadas y firewalls. Tolerancia de igualdad `1e-12`.

No se selecciona presupuesto ganador. El promedio no ponderado de los seis
puntos es descriptivo. Los múltiples intervalos no reciben lectura familiar.

## Lecturas admisibles

- Si topology filtra mejor que constant/public/permuted con ranking fijo, la
  cola conserva valor como módulo de elegibilidad.
- Si todas las colas actúan igual o topology no supera controles, pinball OOF
  no se traduce a un filtro atribuible.
- Si calibrated ayuda pero firewall bloquea o no transporta, el cuello sigue en
  soporte/policy-selection y no se arregla declarando seguridad marginal.
- Mean-only favorable con filtros adversos indica sobreabstención.
- Ningún resultado post hoc concede garantía prospectiva, autoridad física,
  transferencia externa, promoción ni GO/NO-GO.

## Artefactos y recursos

Output canónico:
`data/geometria_proporcional/proportional_graph_two_stage_eligibility_diagnostic_v1/`.
Conservará source hashes, cuantiles, scores, elegibilidad, acciones, firewalls,
efectos por master, bootstraps, análisis, entorno, manifest y replay.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.
No consulta ni usa GPU; todo trabajo GPU continúa en cola.
