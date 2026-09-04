# R372 — Dominancia pareada en la ruta de presupuestos

**Fecha:** 2026-09-04
**Estado:** oficial y replay exacto completados
**Régimen:** auditoría post hoc sobre objetivos y acciones congeladas R371/R369
**Autoridad:** explica la geometría interna del frente; no fija utilidad, cutoff, política, arquitectura ni GO/NO-GO

## Pregunta

R371 mostró que `40%` pertenece a los ocho frentes topology y que `20%` o
`10%` a veces lo acompañan. R372 abre los 21 pares entre identity y los seis
presupuestos para distinguir dos causas: una ampliación puede dominar ambas
coordenadas o puede comprar beneficio grouped a costa de IID. El mismo cálculo
se hace en policy-selection y adjudication con los 2.000 índices bootstrap
originales.

Antes de interpretar la ruta, se verificaron las acciones R369: un presupuesto
mayor conserva cada acción no nula del menor con el mismo alpha. Las `3.040`
comprobaciones sobre cohortes, roles, regímenes, brazos, familias y controles
cerraron sin violaciones.

## Integridad

Diseño `3122f10`, implementación `7b3d537`. Oficial y replay terminaron en
`124,402/123,172 s`, con `0,751/0,751 GiB`, un thread y CUDA invisible. No hubo
fit, vistas, solves ni remuestreo nuevo; `gpu_queried: false`. El manifest
byte-idéntico es:

```text
2e21aab7b3b7b5f1cea49bb91add2fc939033addb6432572cc637915e1d03b63
```

Coinciden `8/8` deterministas. Los dos NPZ contienen `12.768` arrays y
`6.460.608` valores finitos. La regresión proporcional cerró `189/189`.

## La ruta primero domina y luego intercambia

En las 168 comparaciones topology primarias de selección —dos cohortes por
cuatro brazos por 21 pares— la expansión domina `117`, paga costo IID por
ganancia grouped en `48`, invierte ese intercambio en `2` y queda dominada por
la base en `1`. En adjudication la expansión domina `135`, el tradeoff
IID-costo/grouped-ganancia ocupa `32` y la base domina `1`.

Los seis pasos adyacentes muestran dónde nace la frontera:

| Expansión | Selection: domina / costo IID | Adjudication: domina / costo IID |
|---|---:|---:|
| identity → 1% | 6 / 0 | 4 / 4 |
| 1% → 2% | 7 / 0 | 6 / 1 |
| 2% → 5% | 4 / 4 | 7 / 1 |
| 5% → 10% | 8 / 0 | 7 / 1 |
| 10% → 20% | 5 / 3 | 7 / 1 |
| 20% → 40% | 2 / 6 | 2 / 6 |

La tabla omite para brevedad los pocos estados inversos de los dos primeros
pasos, preservados en los crudos. `20→40%` nunca favorece a la base: o mejora
ambas coordenadas o mejora grouped pagando IID. Esa es la razón geométrica por
la que `40%` no es dominado en R371. No implica que sea la política preferida.

## La pertenencia al frente oculta cambios de mecanismo

Aunque `20→40%` conserva el mismo conteo agregado `2/6` en selection y
adjudication, sólo cuatro de ocho celdas mantienen su estado, todas como
tradeoff. Dos pasan de dominancia a tradeoff y otras dos hacen el recorrido
inverso. `10→20%` también conserva sólo cuatro estados; `5→10%` conserva siete.

Sobre los 168 pares, `101` mantienen el estado puntual entre roles: `93`
dominancias de expansión y `8` tradeoffs. El acuerdo es `60,1%`. Entre los 48
pares adyacentes permanecen `26`, o `54,2%`. La distancia L1 media entre los
vectores bootstrap es `0,603` para todos los pares y `0,693` para los
adyacentes, en una escala `0–2`.

El acuerdo A/B del estado pareado promedia `0,679` en selection y `0,702` en
adjudication. Los controles permutados transportan algo mejor en conjunto:
`1.838/2.688` estados primarios quedan en la diagonal (`68,4%`) y su distancia
bootstrap L1 media es `0,569`. Esto no establece significación, pero contradice
una lectura en la que la inestabilidad fuera un costo exclusivo de los shams.

## Bootstrap sin umbral

La frecuencia del estado puntual no es uniforme. Para `20→40%` va de `0,520`
a `0,959` en selection, mediana `0,812`; en adjudication cae a un rango
`0,439–0,962`, mediana `0,629`. En `5→10%`, que parece más monotónico, la
dominancia seleccionada va de `0,501` a `1,000`.

Estas frecuencias describen cercanía de comparaciones. No son probabilidades
calibradas y R372 no inventa un cutoff para convertirlas en certificados.

## Lectura

**Observación.** La ruta topology es anidada y la mayor parte de sus expansiones
mejora ambos objetivos medios. La frontera aparece en el tramo alto, donde
`40%` intercambia daño IID por beneficio grouped frente a `20%`. La etiqueta
agregada de frente transporta mejor que la relación pareada que la produce.

**Hipótesis.** La primitive recuperable es un objeto `BudgetPath`: secuencia
anidada de acciones, coordenadas IID/grouped e incertidumbre por comparación.
Un firewall escalar elimina la ruta; elegir el extremo por recurrencia oculta
la utilidad requerida.

**Inferencia acotada.** El slicing histórico llegó a su límite útil. R372
explica por qué `40%` permanece en el frente, pero el acuerdo de estado
selection→adjudication (`60,1%`) y A/B (`67,9–70,2%`) no acreditan una política
ni ventaja topológica robusta. El siguiente trabajo CPU razonable es diseñar y
chequear la interfaz tipada `BudgetPath` como candidata separada del baseline.
Una confirmación empírica exigiría un freeze prospectivo dimensionado por R370,
no otra realización corta. La eventual ejecución de ese freeze y toda variante
neuronal/GPU quedan en cola; no hay promoción ni GO/NO-GO.

Artefactos: plan
`experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_RANKING_PAIRWISE_DOMINANCE_CPU.md`,
runner
`experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pairwise_dominance.py`,
oficial
`data/geometria_proporcional/proportional_graph_mean_ranking_pairwise_dominance_v1/`
y replay con sufijo `_replay`.
