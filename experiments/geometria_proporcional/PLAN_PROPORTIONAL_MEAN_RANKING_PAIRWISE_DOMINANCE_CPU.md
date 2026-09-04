# Plan CPU — dominancia pareada en la ruta de presupuestos

**Fecha:** 2026-09-04
**Estado:** oficial y replay byte-exacto completados
**Régimen:** auditoría post hoc de la geometría interna del frente R371
**Autoridad:** explica comparaciones entre políticas; no fija utilidad, cutoff, política, arquitectura ni GO/NO-GO

## Pregunta

R371 conservó frentes topology de una a tres políticas, casi siempre entre los
presupuestos `10/20/40%`. Su membresía transportó sólo parcialmente. R372 abre
las comparaciones pareadas que producen ese frente: pregunta si ampliar el
presupuesto domina al nivel anterior, si introduce un intercambio IID/grouped
o si la dirección cambia entre policy-selection y adjudication.

No se elige un punto del frente. La salida es una matriz de relaciones y sus
frecuencias bootstrap.

## Fuentes y freeze

Fuente inmediata: R371, manifest
`88333aed65a78444a92b4236979981ccfedc1d695fe0c86fd8a26ab89625a34c`.
Las acciones se contrastan contra R369, manifest
`bc898c97ed3730e8719e344dc51d102a61637324eb7c3d79cc3c7e677941f994`,
únicamente para verificar que los soportes ranked son anidados.

Se preservan cohortes A/B, roles, brazos, regímenes fixed/common, objetivos,
presupuestos, familias, dieciséis controles separados e índices bootstrap. No
hay fit, vistas, solves, remuestreo nuevo ni GPU.

## Relaciones entre dos presupuestos

Para cada par no ordenado `base < expansion` entre identity y
`1/2/5/10/20/40%`, se calcula

```text
incremento = objetivo(expansion) - objetivo(base)
```

en IID y grouped. Con tolerancia `1e-12`, el estado es exclusivo:

- `EXPANSION_DOMINATES`: la expansión no empeora ninguna coordenada y mejora
  al menos una;
- `BASE_DOMINATES`: relación inversa;
- `IID_COST_GROUPED_GAIN`: empeora IID y mejora grouped;
- `IID_GAIN_GROUPED_COST`: mejora IID y empeora grouped;
- `EQUIVALENT`: ambas diferencias son numéricamente cero.

Se guardan los 21 pares. Los seis pares adyacentes reciben un resumen propio
porque describen la ruta incremental, pero no más autoridad.

## Bootstrap y transporte

Los `2.000` índices originales se reaplican a cada diferencia pareada. Se
reporta la frecuencia continua de los cinco estados; no se transforma en
probabilidad calibrada ni se aplica umbral.

Entre selection y adjudication se conserva:

1. transición del estado puntual;
2. cambio de los dos incrementos medios;
3. distancia L1 entre vectores de frecuencias bootstrap;
4. acuerdo de estado entre cohortes A/B.

Los controles permutados se resumen como distribución sobre dieciséis
réplicas, sin voto mayoritario ni arena conjunta.

## Contratos negativos

- Una expansión dominante no es una política segura: usa medias abiertas.
- Un tradeoff no prueba multiplicidad ontológica: puede ser pequeño o
  inestable.
- Una frecuencia bootstrap alta no sustituye confirmación ni utility.
- La repetición de `40%` en R371 no autoriza elegirlo.
- El diagnóstico no vuelve rentable otra cohorte histórica corta; R370 conserva
  autoridad sobre potencia.

## Artefactos y recursos

Output canónico:
`data/geometria_proporcional/proportional_graph_mean_ranking_pairwise_dominance_v1/`.
Conservará matrices por cohorte, incrementos por master, frecuencias,
transiciones, config, entorno, manifest y replay.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.
No consulta ni usa GPU; cualquier etapa CUDA permanece en cola.

## Ejecución

Diseño `3122f10`, implementación `7b3d537`. Oficial y replay terminaron en
`124,402/123,172 s`, con `0,751/0,751 GiB`. Coincidieron los ocho artefactos
deterministas y el manifest fue byte-idéntico:

```text
2e21aab7b3b7b5f1cea49bb91add2fc939033addb6432572cc637915e1d03b63
```

Los NPZ preservaron `12.768` arrays y `6.460.608` valores finitos. Las `3.040`
comprobaciones de anidamiento cerraron sin fallos y la suite proporcional pasó
`189/189`; `gpu_queried: false`.
