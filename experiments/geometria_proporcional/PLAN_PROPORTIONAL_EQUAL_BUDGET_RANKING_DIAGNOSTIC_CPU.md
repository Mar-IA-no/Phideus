# Plan CPU — ranking topology/control con presupuesto igualado

**Fecha:** 2026-09-04
**Estado:** diseño congelado; implementación y ejecución pendientes
**Régimen:** diagnóstico post hoc sobre dos cohortes ya abiertas
**Autoridad:** separa valor de ranking y cantidad de acción; no produce una política segura, no confirma arquitectura ni decide GO/NO-GO

## Motivación

R365 mostró que selected-action mezcla tres operaciones: proponer alpha, decidir
cuántas vistas cruzan el umbral y ordenar cuáles actúan. El firewall topology
cambió `1/4→4/4` entre cohortes y siete de doce signos contra el control matched
fueron inestables. Comparar políticas con números de acciones distintos no
localiza si la asignación topológica ordena mejor las oportunidades.

Fijar exactamente las mismas vistas tampoco sirve: topology y los controles
son scorers de riesgo, no executors. Si reciben la misma vista y el mismo alpha,
el resultado downstream es idéntico y el estimando se anula.

## Fuentes

La fuente inmediata es R365, manifest
`ad983efb5ee2b62f9581f8e0ac9495c087d155762d5458c398bf0df4d05024d6`,
que ata las cohortes R361/R362 y R363/R364 y verificó la reproducción exacta de
R364. El diagnóstico reconstruye sus predicciones preservadas; no ajusta
modelos, no abre roles y no ejecuta solves.

## Propuesta de alpha compartida

Para cada cohorte, brazo y vista de adjudicación se fija:

```text
alpha_common = argmin_alpha(mu_alpha + u_public_base_alpha)
```

El tie-break es el primer alpha, como en R364. Esta propuesta es idéntica para
topology, public-base y las dieciséis permutaciones. Sobre ese alpha, cada
scorer produce:

```text
rank_score_family = mu_alpha_common + u_family_alpha_common
```

Los cuantiles conformales son escalares por familia y no cambian este orden;
no se usan umbrales ni firewall. La salida no es una política de no-daño, sino
un diagnóstico de ranking deployable desde observables públicos.

## Presupuesto igualado

Cada scorer ordena el mismo conjunto de vistas de adjudicación, sin usar labels
IID/grouped ni outcomes. Para cada fracción predeclarada
`1%, 2%, 5%, 10%, 20%, 40%`, actúa en exactamente
`ceil(fracción × n_views)` vistas. Los empates se resuelven por índice de vista.
Topology, public-base y cada control reciben el mismo número total de acciones,
pero pueden elegir vistas distintas. El alpha ejecutado es siempre
`alpha_common` de esa vista.

No se selecciona una fracción ganadora. La grilla completa es el resultado y
se conserva también el promedio no ponderado entre sus seis puntos como
resumen descriptivo, no como test múltiple corregido.

## Estimandos

Por cohorte, brazo, presupuesto y slice IID/grouped/balanceado se construyen
efectos por master de:

1. topology menos identidad;
2. topology menos public-base con igual presupuesto;
3. topology menos el promedio de dieciséis topology-permuted, cada uno con el
   mismo presupuesto.

El tercero es principal. Los intervalos pointwise usan el bootstrap original
de cada cohorte, sin pooling. El balanceado empareja IID/grouped por master. Se
preservan conteos de acción por slice, overlap/Jaccard entre rankings y el alpha
compartido.

Para cada celda se clasifica transporte como `FAVORABLE_BOTH`, `ADVERSE_BOTH`,
`SIGN_UNSTABLE` o `IDENTITY_OR_NUMERICAL_ZERO` con tolerancia `1e-12`. No se
elige presupuesto por el signo observado. Una ventaja sólo sería prometedora
si ocupa una región contigua de presupuestos, transporta entre cohortes y no
depende de una acción efectiva de pocos casos.

## Lecturas admisibles

- Una curva topology−control favorable y transportable localiza valor de
  ranking bajo oportunidad igualada; todavía no acredita abstención segura.
- Una ventaja sólo en `1%` conserva deuda de soporte y potencia.
- Cambio de signo al crecer el presupuesto indica que el ranking sólo ordena
  una cola estrecha, no una política general.
- Null o adversidad frente a controles desplaza el cuello desde firewall hacia
  el scorer/proposer; no niega la señal OOF de R363.
- Ninguna lectura concede autoridad física, transferencia externa, promoción
  arquitectónica o GO/NO-GO.

## Artefactos y recursos

El output canónico será
`data/geometria_proporcional/proportional_graph_equal_budget_ranking_diagnostic_v1/`.
Conservará source hashes, scores, rankings, acciones, efectos por master,
bootstraps, curvas, overlap, manifest, entorno y replay.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.
No consulta, reserva ni consume GPU. Todo trabajo GPU sigue en cola hasta nueva
orden explícita de Mariano.
