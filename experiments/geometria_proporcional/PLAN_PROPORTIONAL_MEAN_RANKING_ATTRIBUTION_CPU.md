# Plan CPU — atribución de la señal del ranking de media

**Fecha:** 2026-09-04
**Estado:** diseño congelado; implementación y ejecución pendientes
**Régimen:** diagnóstico post hoc sobre dos cohortes abiertas
**Autoridad:** atribuye un ranker congelado; no valida arquitectura, transferencia externa, no-daño prospectivo ni GO/NO-GO

## Pregunta

R367 mostró que `mu` ordena mejor que `mu + u_topology`. R368 mantuvo ese orden
pero la cola topology sobreabstuvo y no superó los controles. Falta adjudicar
qué información produce la señal útil de `mu`: features públicas generales,
localización topológica o capacidad extra sin asignación correcta.

R369 reaplica los modelos de media congelados de R360 sobre las cohortes A y B
de R368. No ajusta modelos, colas, thresholds ni features.

## Fuentes y reproducción

Fuente inmediata: R368, manifest
`bffbf9fefa915b6dfd00df63e35bacf9084ec68d71831471b5c0537b11b2536e`,
que ata R367–R361 y R364–R363. Los modelos de media proceden de R360, manifest
`5d7659fe74cfe18ec79609935568d63bdcb90da19f96224b2c182f673d4cbaed`.

Antes de comparar, la predicción `topology_augmented` reconstruida debe igualar
el tensor `mu` preservado por R368 con tolerancia `1e-12` en cada rol, brazo y
cohorte. Toda divergencia invalida la ejecución.

## Familias de ranking

Los cinco objetos comparados son:

1. `reduced_mean`: dos features de escala de la corrección;
2. `public_base_mean`: las quince features públicas generales;
3. `topology_mean`: base más nueve features de localización topology;
4. `topology_permuted_mean`: dieciséis controles de igual dimensión cuya
   localización se permuta preservando el multiconjunto;
5. `target_shuffled_topology_mean`: dieciséis controles entrenados con target
   permutado dentro del contrato de R360, como control secundario.

Los modelos, folds, lambdas, outputs y datos de entrenamiento no cambian.
Topology-permuted es el control incremental primario; target-shuffled separa
asignación informativa de mera capacidad, pero no reemplaza al primero.

## Propuesta y presupuesto

El estimando primario fija `alpha=0,25` para todas las vistas y familias. R366
mostró que la propuesta common de R368 ya elegía ese alpha en `93,9–99,6%` de
las vistas; fijarlo elimina del contraste la pequeña contribución del proposer.

Como sensibilidad predeclarada se conserva exactamente
`alpha_common = argmin(mu_topology + u_public_base)` de R368. En ambos regímenes
cada familia puntúa el alpha común con su propia media y ordena de menor a mayor
predicción, con tie-break por índice. Los presupuestos son
`1%, 2%, 5%, 10%, 20%, 40%`; cada ranker actúa exactamente sobre
`ceil(fracción × n_views)` vistas antes del firewall. No usa labels IID/grouped
para ordenar ni cambia el executor.

## Policy selection y adjudicación

Para cada cohorte, propuesta, brazo, presupuesto y ranker, policy-selection
usa su bootstrap original sobre el daño medio IID. La regla congelada
`upper95 <= 0` despliega la política completa; el rechazo produce identidad.
Cada control permutado se selecciona individualmente y sólo después se promedia
su outcome.

Se preservan dos stages:

- `ranked`: igual presupuesto, sin firewall;
- `deployed`: misma política después del firewall.

## Estimandos

Por cohorte, propuesta, brazo, presupuesto, stage y slice IID/grouped/balanced:

1. topology-mean menos identidad;
2. topology-mean menos reduced-mean;
3. topology-mean menos public-base-mean;
4. topology-mean menos promedio topology-permuted-mean — primario;
5. topology-mean menos promedio target-shuffled-topology-mean — secundario.

Se conservan efectos por master, intervalos pointwise con los bootstraps
originales, transporte entre cohortes, overlap/Jaccard de vistas, alphas,
acciones y firewalls. El promedio no ponderado entre presupuestos es sólo
descriptivo; no se elige un presupuesto ganador ni se da lectura familiar a
los intervalos. Tolerancia de cero `1e-12`.

## Lecturas admisibles

- Topology favorable frente a public y permuted bajo ambas propuestas sostiene
  valor de localización para ranking dentro del generador.
- Topology similar a public indica que la señal útil ya está en las features
  generales; no acredita primitive topológica.
- Topology similar a permuted indica que dimensión/capacidad, no asignación,
  explica el ranking.
- Ranked favorable pero deployed inestable localiza el cuello en selección y
  transporte, no autoriza a llamar seguro al ranker.
- Ningún resultado post hoc promueve arquitectura, justifica un freeze o
  decide GO/NO-GO.

## Artefactos y recursos

Output canónico:
`data/geometria_proporcional/proportional_graph_mean_ranking_attribution_v1/`.
Conservará hashes de fuentes, predicciones, rankings, acciones, overlaps,
firewalls, efectos por master, bootstraps, análisis, entorno, manifest y replay.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.
No consulta ni usa GPU. Cualquier contraste CUDA permanece en cola hasta nueva
orden explícita del usuario.
