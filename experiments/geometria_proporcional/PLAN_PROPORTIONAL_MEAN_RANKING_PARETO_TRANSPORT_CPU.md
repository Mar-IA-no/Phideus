# Plan CPU — transporte del frente Pareto de políticas de media

**Fecha:** 2026-09-04
**Estado:** diseño congelado; implementación y ejecución pendientes
**Régimen:** diagnóstico post hoc set-valued sobre policy-selection y adjudication abiertas
**Autoridad:** evalúa estabilidad de un conjunto de políticas; no elige utilidad, no valida deployment prospectivo, no promueve arquitectura ni decide GO/NO-GO

## Pregunta

R370 mostró que el firewall binario borra `26/37` señales favorables del ranker
topology. El beneficio se concentra en grouped mientras la restricción se
formula sobre IID. R371 no resuelve esa tensión con un peso inventado: devuelve
el conjunto de políticas no dominadas en las dos coordenadas observables

```text
(mean_delta_IID, mean_delta_grouped)
```

donde menor es mejor en ambas. Identity y los seis presupuestos ranked de R369
son candidatos. No se usan las políticas deployed ni sus firewalls para formar
el frente.

## Fuentes y freeze

Fuente metodológica inmediata: R370, manifest
`c6734871d03e2404d6cd726ad75bdcd28890921147a6952fb98a3a2bf8c4d7e8`.
Fuente de acciones: R369, manifest
`bc898c97ed3730e8719e344dc51d102a61637324eb7c3d79cc3c7e677941f994`.
Ambas atan R368 y los modelos R360.

Se preservan cohortes A/B, roles policy-selection/adjudication, alphas,
rankings, presupuestos `1/2/5/10/20/40%`, bootstraps y quotient RMSE. No hay
fit, vistas, solves, thresholds nuevos ni acceso a test para seleccionar el
frente.

## Familias y regímenes

Primario: `fixed_alpha_0.25`. Sensibilidad: `r368_common`.

Se construyen frentes separados para:

- `topology_mean`;
- `public_base_mean`;
- `reduced_mean`;
- cada una de las dieciséis réplicas `topology_permuted_mean`.

Cada frente contiene siete políticas identificables: identity y los seis
presupuestos. Separar réplicas evita conceder a los controles dieciséis veces
más candidatos en una única arena.

## Dominancia y bootstrap

Una política `q` domina a `p` si no es peor en ninguna coordenada dentro de
`1e-12` y es estrictamente mejor en al menos una. Los empates permanecen en el
frente; no se resuelven por presupuesto, sparsity ni preferencia implícita.

Policy-selection produce el frente desplegable. Sus índices se congelan antes
de leer adjudication. Los `2.000` bootstraps originales generan, sin threshold,
la frecuencia de inclusión de cada política en el frente. Esa frecuencia mide
inestabilidad; no se convierte en probabilidad calibrada ni criterio de corte.

## Transporte selection → adjudication

Para cada cohorte, propuesta, brazo y familia se reporta:

1. Jaccard entre el frente seleccionado y el frente oracle de adjudication;
2. retención: fracción del frente seleccionado que sigue no dominada;
3. recall diagnóstico: fracción del frente oracle que había sido seleccionada;
4. políticas añadidas/perdidas y sus coordenadas;
5. frecuencia bootstrap de inclusión en ambos roles.

El frente oracle de adjudication sólo adjudica transporte; nunca re-selecciona
la política. Entre cohortes A/B se compara por Jaccard el conjunto de IDs
seleccionados, porque ambas comparten exactamente las siete políticas.

## Atribución contra controles

En adjudication se comparan conjuntos seleccionados sin utilidad mediante
cobertura por dominancia:

- `TOPOLOGY_DOMINATES`: todo punto del frente comparador es dominado por algún
  punto topology y no ocurre la cobertura inversa;
- `CONTROL_DOMINATES`: condición inversa;
- `EQUIVALENT`: ambas coberturas;
- `INCOMPARABLE`: ninguna cobertura total.

Se aplica a public-base, reduced y cada réplica permutada. Para permuted se
reporta la distribución de los dieciséis estados, sin votar una verdad por
mayoría. También se conserva la comparación oracle-adjudication como techo
diagnóstico separado del frente seleccionado.

## Lecturas admisibles

- Un frente selection estable entre cohortes y retenido en adjudication
  sostiene una interfaz set-valued aun sin utilidad.
- Alta inclusión bootstrap con baja retención indica incertidumbre de
  transporte, no multiplicidad de soluciones válidas.
- `INCOMPARABLE` frente a controles significa tradeoffs distintos; no equivale
  a ventaja topology.
- Si identity domina o los frentes no transportan, el ranker queda descriptivo
  bajo este régimen.
- Ningún resultado abierto autoriza elegir un punto, fijar utilidad, llamar
  segura a una política, promover arquitectura o decidir GO/NO-GO.

## Artefactos y recursos

Output canónico:
`data/geometria_proporcional/proportional_graph_mean_ranking_pareto_transport_v1/`.
Conservará objetivos por master, masks de frente, frecuencias bootstrap,
comparaciones de dominancia, hashes, entorno, manifest y replay.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.
No consulta ni usa GPU; cualquier etapa CUDA permanece en cola.
