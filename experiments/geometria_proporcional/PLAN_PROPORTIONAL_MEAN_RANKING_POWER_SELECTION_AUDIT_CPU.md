# Plan CPU — potencia y selección del ranking de media

**Fecha:** 2026-09-04
**Estado:** diseño congelado; implementación y ejecución pendientes
**Régimen:** auditoría post hoc sobre los efectos y acciones R369
**Autoridad:** planifica evidencia y localiza inestabilidad; no acredita una política, no promueve arquitectura ni decide GO/NO-GO

## Pregunta

R369 encontró señal favorable de `topology_mean - topology_permuted_mean` en el
ranking puro, pero ningún intervalo primario se resolvió en ambas cohortes y el
firewall revirtió la comparación. R370 pregunta dos cosas separadas:

1. ¿qué tamaño exigirían los efectos observados si media y varianza por master
   permanecieran fijas?;
2. ¿la incertidumbre omitida por promediar dieciséis controles y la no
   linealidad del firewall alteran esa planificación?

## Fuente y freeze

Fuente única inmediata: R369, manifest
`bc898c97ed3730e8719e344dc51d102a61637324eb7c3d79cc3c7e677941f994`.
El manifest ata R368 y los modelos R360. La auditoría reconstruye outcomes desde
acciones, quotient RMSE y bootstraps preservados; no ajusta modelos, no cambia
rankings, alpha, presupuestos, gates ni vistas.

El estimando primario permanece `fixed_alpha_0.25`, stage `ranked`,
`topology_mean - mean(topology_permuted_mean)`, sobre los `72` puntos
brazo×presupuesto×slice. La propuesta R368-common es sensibilidad. Los promedios
entre presupuestos se conservan como descripción, nunca como presupuesto
seleccionado.

## Proyección condicionada en masters

Para cada cohorte y celda con media negativa:

```text
radius = upper95 - mean
n_projected = ceil(n_current * (radius / abs(mean))^2)
```

Si el intervalo ya es favorable, `n_projected=n_current`; si la media no es
negativa, no existe proyección favorable finita bajo este supuesto. Una celda
transportada usa el máximo de A y B. Se reporta la distribución completa y los
conteos bajo referencias descriptivas `300/500/1.000/2.000`, sin convertirlas
en umbrales de éxito.

La fórmula mantiene efecto y varianza por master constantes, ignora cambio de
generador y no incorpora incertidumbre de entrenamiento ni de controles. Es una
herramienta de planificación, no una promesa de potencia.

## Eje de controles

R369 promedió dieciséis permutaciones congeladas. R370 reconstruye, para cada
celda, el efecto por réplica y conserva:

- fracción de réplicas donde topology tiene mejor punto;
- mínimo, mediana, máximo y dispersión de esas medias;
- bootstrap descriptivo conjunto que reusa los `2.000` índices originales de
  masters y remuestrea con reemplazo las dieciséis réplicas mediante seed fijo
  `2026091103`.

También calcula un intervalo del eje control a master infinito aproximado,
remuestreando sólo las medias por réplica. Si su upper95 no es negativo, aumentar
únicamente masters no cierra la variabilidad del conjunto actual de controles.
Esta lectura trata las dieciséis permutaciones como muestra descriptiva; no les
concede estatuto IID ni reemplaza nuevos trainings.

## Descomposición del firewall

Para cada efecto por master se materializan cuatro combinaciones:

```text
E00 = topology_ranked   - controls_ranked
E10 = topology_deployed - controls_ranked
E01 = topology_ranked   - controls_deployed
E11 = topology_deployed - controls_deployed
```

La contribución de bloquear topology es `E10-E00`; la de los firewalls de
control es `E01-E00`. Se verifica la identidad aditiva
`E11 = E00 + (E10-E00) + (E01-E00)` a `1e-12`. Se reportan transporte, intervalos
y reversión de signo; no se inventa un gate común contrafáctico.

Además se preservan tasas de pass topology y distribución pass de controles por
cohorte, brazo y presupuesto. El stage deployed no se usa para proyectar una
confirmación si su signo no transporta.

## Lecturas admisibles

- Una proyección moderada master-only con eje control no resuelto sigue siendo
  insuficiente para justificar un freeze.
- Proyecciones grandes o signos inestables desplazan el próximo experimento
  hacia una señal mayor o un diseño menos variable, no demuestran un techo.
- Si la reversión proviene de firewalls distintos, ranking y policy selection
  requieren módulos y evaluaciones separados.
- Ningún resultado abierto concede confirmación, seguridad, transferencia,
  promoción ni GO/NO-GO.

## Artefactos y recursos

Output canónico:
`data/geometria_proporcional/proportional_graph_mean_ranking_power_selection_audit_v1/`.
Conservará efectos por réplica y master, bootstraps, proyecciones, descomposición
del firewall, hashes, entorno, manifest y replay.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.
No consulta ni usa GPU; cualquier etapa CUDA permanece en cola.
