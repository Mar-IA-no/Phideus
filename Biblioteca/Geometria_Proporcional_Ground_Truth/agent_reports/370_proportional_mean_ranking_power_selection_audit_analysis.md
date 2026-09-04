# R370 — Potencia y selección del ranking de media

**Fecha:** 2026-09-04  
**Estado:** oficial y replay exacto completados  
**Régimen:** auditoría post hoc sobre efectos y acciones R369  
**Autoridad:** planifica evidencia y descompone selección; no valida política, arquitectura ni GO/NO-GO

## Pregunta

R369 dejó `37/72` puntos favorables para
`topology_mean - mean(topology_permuted_mean)`, sin intervalos resueltos en
ambas cohortes, y una inversión después del firewall. R370 preserva rankings,
alpha, presupuestos y bootstraps para preguntar cuánto costaría cerrar esos
efectos y qué parte de la inversión introduce cada firewall.

La proyección condicionada usa
`n × ((upper95-mean)/abs(mean))²`. También reconstruye el efecto contra cada una
de las dieciséis permutaciones y remuestrea dos ejes: masters con los índices
originales y controles con seed congelada. Este segundo bootstrap es
descriptivo; las permutaciones no se declaran IID.

## Integridad

Diseño `e78beb6`, implementación `dd41cb7`. Oficial y replay terminaron en
`26,067/26,371 s`, con `0,791/0,790 GiB`, un thread y CUDA invisible. No hubo
refit, vistas ni solves nuevos; `gpu_queried: false`. El manifest byte-idéntico
es:

```text
c6734871d03e2404d6cd726ad75bdcd28890921147a6952fb98a3a2bf8c4d7e8
```

Coinciden `8/8` deterministas, los `2.356` arrays canónicos son finitos y la
regresión proporcional cerró `180/180`. Los `72` efectos ranked reproducen
R369 y la identidad aditiva del firewall cierra a `1e-12`.

## Potencia condicionada

Las `37` celdas con punto favorable en ambas cohortes proyectan:

- mínimo `522` masters;
- mediana `5.552`;
- media `121.320`, dominada por una cola larga;
- máximo `1.541.364`.

Ninguna queda en `<=300` o `<=500`; cinco quedan en `<=1.000` y doce en
`<=2.000`. Los dos mínimos son raw-typed al `1%`: grouped proyecta `522` y
balanced `569`. Luego aparecen closure-typed al `20%`: balanced `645`, IID
`809` y grouped `890`.

Estas cifras son optimistas en un sentido preciso: mantienen fijos efecto y
varianza por master y condicionan en el promedio de dieciséis controles. No son
promesas de potencia ni techos. El bootstrap conjunto masters×controles deja
`7/72` intervalos favorables en A y `0/72` en B; por tanto `0/72` se resuelve en
ambas. Remuestrear sólo las medias de control deja `19/72` resueltas en ambas,
pero ese cálculo todavía condiciona en las cohortes observadas y no representa
masters infinitos.

## La reversión viene del firewall topology

Con alpha fijo, la descomposición transportada es:

| Combinación | Favorable | Adversa | Inestable | Cero |
|---|---:|---:|---:|---:|
| ambos ranked (`E00`) | 37 | 11 | 24 | 0 |
| sólo topology con firewall (`E10`) | 11 | 31 | 30 | 0 |
| sólo controles con firewall (`E01`) | 44 | 10 | 18 | 0 |
| ambos deployed (`E11`) | 11 | 20 | 29 | 12 |

De las `37` celdas ranked favorables, `26` dejan de serlo después del
deployment. En el promedio entre presupuestos, `E10` es adverso en `11/12` y
no favorable en ninguna; `E01` es favorable en `10/12`. La inversión no nace
de que los controles reciban un firewall ventajoso: aparece cuando la política
topology queda bloqueada.

La sensibilidad con el proposer común reproduce la separación. `E00` queda
`39` favorable y `7` adverso; `E10`, `19` favorable y `31` adverso; `E01`, `47`
favorable y `4` adverso; `E11`, `19/19`. El cuello es policy-selection, no la
elección casi constante de alpha.

## Lectura

**Observación.** El ranker topology contiene una señal pequeña y transportada
por punto, pero ninguna celda queda confirmable con una realización histórica
de aproximadamente 250 masters. La incertidumbre de controles empeora la
lectura y el firewall topology elimina la mayoría de los puntos favorables.

**Hipótesis.** La restricción de no-daño medio IID y el beneficio concentrado
en grouped describen objetivos en tensión. Un selector binario por presupuesto
colapsa esa tensión a identidad; aumentar muestra sin cambiar señal no resuelve
el costo de selección.

**Inferencia acotada.** No se justifica un freeze del ranker actual ni otra cola.
La siguiente arquitectura de decisión debería exponer el frente
IID-daño/grouped-beneficio como conjunto de políticas, sin inventar un peso de
utilidad. Un diagnóstico Pareto CPU puede medir si ese frente transporta antes
de pedir al usuario una elección de utilidad o diseñar una confirmación. Esto
no promueve el ranker ni decide GO/NO-GO.

Artefactos: plan
`experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_RANKING_POWER_SELECTION_AUDIT_CPU.md`,
runner
`experiments/geometria_proporcional/run_proportional_graph_mean_ranking_power_selection_audit.py`,
oficial
`data/geometria_proporcional/proportional_graph_mean_ranking_power_selection_audit_v1/`
y replay con sufijo `_replay`. Toda GPU permanece en cola.
