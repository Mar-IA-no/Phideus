# R371 — Transporte del frente Pareto del ranking de media

**Fecha:** 2026-09-04
**Estado:** oficial y replay exacto completados
**Régimen:** diagnóstico post hoc set-valued sobre políticas R369
**Autoridad:** caracteriza conjuntos no dominados; no elige utilidad, política, arquitectura ni GO/NO-GO

## Pregunta

R370 localizó una tensión que el firewall binario ocultaba: el ranker de media
conservaba beneficio sobre todo en grouped, mientras la elegibilidad exigía
no-daño IID. R371 no suma esas coordenadas ni les asigna un peso. Para cada
cohorte, brazo y familia construye el frente no dominado de siete políticas:
identity y los presupuestos ranked `1/2/5/10/20/40%`. Menor
`mean_delta_IID` y menor `mean_delta_grouped` son mejores.

El frente de policy-selection queda congelado antes de adjudication. La
comparación con su frente oracle posterior mide transporte, no re-selección.
Las dieciséis permutaciones topology se procesan como arenas separadas para no
multiplicar artificialmente sus candidatos.

## Integridad

Diseño `3600701`, implementación `5bca9b0`. Oficial y replay terminaron en
`84,100/83,627 s`, con `0,730/0,731 GiB`, un thread y CUDA invisible. No hubo
fit, vistas ni solves nuevos; `gpu_queried: false`. El manifest byte-idéntico
es:

```text
88333aed65a78444a92b4236979981ccfedc1d695fe0c86fd8a26ab89625a34c
```

Coinciden `8/8` deterministas. Los dos NPZ contienen `610 + 610` arrays y
`4.181.792` valores finitos. La regresión proporcional cerró `184/184`.

## El frente topology es pequeño, pero no idéntico entre cohortes

Con alpha fijo, los ocho frentes seleccionados contienen entre una y tres
políticas, media `2,125`. `budget_0.40` aparece en `8/8`, `budget_0.20` en
`6/8` y `budget_0.10` en `3/8`; identity y los presupuestos `1/2/5%` no
aparecen. El frente oracle conserva `40%` en `8/8` y `20%` en `6/8`.

Que `40%` aparezca siempre no lo convierte en política elegida: en varias
celdas comparte el frente con presupuestos que intercambian daño IID por
beneficio grouped. El Jaccard del frente seleccionado entre cohortes A/B va de
`0,333` a `0,667`, media `0,542`. La identidad de conjunto, por tanto, sólo
transporta parcialmente.

Las frecuencias bootstrap de los miembros seleccionados topology van de
`0,598` a `0,998`, mediana `0,943`; un candidato que no pertenece al frente
puntual llega a `0,480`. Esto describe comparaciones cercanas en parte de la
frontera. No son probabilidades calibradas ni justifican un cutoff.

## Transporte selection → adjudication

La retención topology primaria por cohorte×brazo es:

```text
A: 1,000 · 0,500 · 1,000 · 1,000
B: 0,667 · 0,333 · 0,667 · 1,000
```

Su media es `0,771`, con mínimo `0,333`. Los Jaccard individuales van de
`0,333` a `1,000`: dos frentes transportan sin cambios, mientras los restantes
ganan o pierden al menos una política. El proposer común mantiene el tamaño y
el Jaccard entre cohortes; eleva la retención media a `0,833`, pero no elimina
el mínimo `0,333`.

Los controles permutados tampoco son estables por construcción: en el régimen
primario promedian Jaccard `0,676`, retención `0,751` y recall oracle `0,904`.
La inestabilidad no es exclusiva de topology, aunque tampoco acredita su valor
incremental.

## Atribución sin utilidad

Sobre los frentes seleccionados evaluados en adjudication, topology domina por
cobertura a reduced en `8/8` celdas. Frente a public-base domina en `4/8` y
queda incomparable en `4/8`; public-base no domina ninguna. Contra las 128
arenas permutadas, el resultado es:

| Estado | Frentes seleccionados | Frentes oracle |
|---|---:|---:|
| topology domina | 35 | 29 |
| control domina | 7 | 6 |
| incomparables | 86 | 93 |
| equivalentes | 0 | 0 |

La sensibilidad con proposer común conserva la lectura: `31/7/90/0` para
topology domina/control domina/incomparables/equivalentes. La categoría modal
es incomparabilidad, no ventaja topology. Un punto topology puede mejorar una
coordenada y perder la otra sin que exista una preferencia autorizada para
resolver el intercambio.

## Lectura

**Observación.** El ranker topology elimina identity y los presupuestos menores
del frente puntual, y conserva una envolvente de presupuestos altos. Esa
envolvente transporta mejor que una política binaria bloqueada por firewall,
pero cambia entre cohortes y no cubre de manera uniforme a public-base ni a los
controles permutados.

**Hipótesis.** La señal recuperable no es un selector escalar sino una ruta
anidada de acciones con tradeoff. Parte de la variación del frente puede venir
de comparaciones pareadas cercanas entre `10/20/40%`, no de tres soluciones
estructuralmente distintas.

**Inferencia acotada.** R371 sostiene estudiar una interfaz que devuelva un
conjunto o una curva de políticas junto con sus dos coordenadas. No autoriza
elegir `40%`, declarar seguro el frente ni promover el ranker. El siguiente
diagnóstico CPU con poder discriminante es abrir la matriz pareada de
dominancia entre presupuestos y su cambio selection→adjudication: puede separar
una envolvente estable de intercambios sostenidos por ruido sin introducir una
utilidad. Otra realización histórica corta no resolvería la potencia observada
en R370. Toda GPU permanece en cola.

Artefactos: plan
`experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_RANKING_PARETO_TRANSPORT_CPU.md`,
runner
`experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pareto_transport.py`,
oficial
`data/geometria_proporcional/proportional_graph_mean_ranking_pareto_transport_v1/`
y replay con sufijo `_replay`.
