# Lector aprendido de particiones: contraste completo

Fecha: 2026-09-08. Estado: 36 entrenamientos y cuatro tests nuevos con
evaluación y replay completos. Auditorías independientes de evidencia y
alineación cerradas sin hallazgos materiales abiertos; no hay promoción
arquitectónica ni decisión GO/NO-GO.

## Resultado

Aprender a puntuar particiones candidatas completas dentro del pool fijo no produjo una ventaja general del
factor de fuente compartida. En el primario de mayor polifonía supera a
Pares/estructura y a Fuente desacoplada, pero no muestra una ventaja clara
frente a Compatibilidad local. Bajo la familia deformada, Compartida pierde
ARI frente a los tres controles aprendidos. Es evidencia mixta sobre este
mecanismo y esta receta, no una refutación de la geometría armónica ni una
demostración de que las frecuencias sean insuficientes.

## Qué se comparó

El [protocolo congelado](PROTOCOL_LEARNED_PARTITION_READER.md) entrena cuatro
lectores sobre tres backbones descriptor-guided congelados, con tres
inicializaciones por lector: 36 corridas de 50 épocas. Comparten algoritmo
de candidatos, datos, normalización train-only, presupuesto y receta. Los
controles distinguen relaciones de pares, compatibilidad local, costo conjunto
de fuente y un costo conjunto desacoplado dentro de tamaños comparables.
El witness conjunto ajusta la familia con `gamma=0`; el test deformado genera
`gamma≠0` sin ampliar ese fitter. El desajuste de familia es una explicación
posible del daño observado, no una causa aislada por este contraste.
Las épocas comunes seleccionadas por calibración fueron, respectivamente,
40, 50, 40 y 40; no se ajustaron con los tests.

La loss estima por MSE las entropías condicionales normalizadas de separación
y mezcla de fuentes. Los costos no son probabilidades. La representación del
backbone no se vuelve a entrenar: se prueba un lector sobre ella, no una nueva
geometría latente. Frente a lectores fijos cambian conjuntamente cabeza y loss;
esa comparación no permite atribuir el efecto a una sola de las dos.

Train y calibración contienen 4.096 y 512 escenas; cada test nuevo contiene
512. El puerto de entrada es el vector permutado de log-frecuencias centradas
en float32 (`q32`), del cual derivan las features y los backbones congelados.
El fenómeno sintético sigue la ley
`f_n = f_0 n sqrt(1 + beta n² + gamma n⁴)` y ruido de dos cents. No expone
amplitud, tiempo, fase, régimen ni identificadores verdaderos. Mayor beta,
cuatro fuentes simultáneas y gamma no nulo definen los tres cambios de
distribución. Las semillas y rangos exactos están en el protocolo.

## ARI por escenario

Cada media promedia primero las nueve predicciones de una escena y después
las 512 escenas. Esas nueve celdas no son nueve réplicas poblacionales
independientes. Los nombres breves de la tabla conservan el orden anterior.

| Lector | IID | Mayor beta | Mayor polifonía | Familia deformada |
|---|---:|---:|---:|---:|
| Pares/estructura | 0.960128 | 0.695120 | 0.901010 | 0.743220 |
| Compatibilidad local | 0.960655 | 0.737457 | 0.903530 | 0.744268 |
| Fuente compartida | 0.961765 | 0.741959 | 0.903916 | 0.737577 |
| Fuente desacoplada | 0.960574 | 0.700793 | 0.901492 | 0.742366 |
| Pares fijo | 0.961502 | 0.720754 | 0.896462 | 0.736326 |
| Fuente compartida fija | 0.961365 | 0.714162 | 0.897015 | 0.730739 |
| Histórico | 0.959843 | 0.795027 | 0.853707 | 0.751273 |

El primario predeclarado es Compartida menos cada control aprendido en ARI
de mayor polifonía. Usa 2.000 remuestreos pareados por escena, con intervalos
percentiles nominales de 98,3333% por contraste —Bonferroni para tres
comparaciones—, condicionales a los checkpoints y lectores entrenados.

| Contraste primario | Delta ARI | Intervalo 98,3333% |
|---|---:|---:|
| Compartida − Pares/estructura | +0.002906 | [0.001815, 0.003984] |
| Compartida − Compatibilidad local | +0.000387 | [-0.000544, 0.001364] |
| Compartida − Fuente desacoplada | +0.002424 | [0.001419, 0.003572] |

El intervalo frente a Local cruza cero: no demuestra equivalencia. El signo
medio por celda checkpoint × lector es favorable a Compartida en 7/9, 6/9 y
8/9 celdas, respectivamente; esos conteos son descriptivos, no otro test.

Las comparaciones restantes son descriptivas al 95%. En mayor beta,
Compartida supera a Local en ARI medio por +0.004502 [0.001055, 0.008009],
pero sólo 4/9 celdas tienen signo positivo y queda por debajo del Histórico
por −0.053068 [-0.061891, -0.044708]. En familia deformada pierde frente a
Pares por −0.005643 [-0.007929, -0.003259], frente a Local por −0.006691
[-0.009369, -0.004303] y frente a Desacoplada por −0.004789
[-0.007044, -0.002549]. No corresponde resumir estos escenarios como una
única ganancia OOD.

## Partición y fragmentación

ARI no sustituye las demás métricas. Esta tabla muestra los dos lectores
físicos aprendidos y el Histórico; las fracciones no son porcentajes.

| Escenario / lector | Partición exacta | Error absoluto de k | Masa sub-3 |
|---|---:|---:|---:|
| IID / Local | 0.845052 | 0.081814 | 0.006011 |
| IID / Compartida | 0.851345 | 0.083116 | 0.006261 |
| IID / Histórico | 0.852865 | 0.037109 | 0.003463 |
| Mayor beta / Local | 0.219618 | 1.137804 | 0.093060 |
| Mayor beta / Compartida | 0.209201 | 1.196832 | 0.099559 |
| Mayor beta / Histórico | 0.298177 | 0.810547 | 0.086022 |
| Polifonía / Local | 0.442708 | 0.415365 | 0.022635 |
| Polifonía / Compartida | 0.448568 | 0.422526 | 0.023695 |
| Polifonía / Histórico | 0.333984 | 0.337240 | 0.007565 |
| Deformada / Local | 0.290799 | 0.819661 | 0.061572 |
| Deformada / Compartida | 0.268663 | 0.920790 | 0.071049 |
| Deformada / Histórico | 0.326823 | 0.696615 | 0.058955 |

Local tiene menor error absoluto de k y menor masa en grupos de menos de
tres miembros que Compartida en los cuatro escenarios. El Histórico conserva
menor error de k en los cuatro y mayor partición exacta salvo en polifonía.
No se introdujo un umbral posterior para declarar un ganador global. Las
[trece métricas completas](RESULTS_LEARNED_PARTITION_MEANS.csv) quedan visibles
para los siete lectores y cuatro escenarios, incluidas discrepancia de pares,
VI y entropías split/merge. Los artefactos conservan además sus contrastes e
intervalos por lector.

## Uso del factor y cobertura de candidatos

Las siete intervenciones de canales modifican decisiones en los cuatro
escenarios: los conteos por intervención van de 39 a 171 en IID, de 225 a
1.156 en mayor beta, de 157 a 613 en polifonía y de 187 a 1.195 en deformada,
sobre 4.608 casos entrenados cada una. No se excluyó ningún caso del primario.
Los cambios máximos absolutos de componente, según intervención, abarcan
0.036894–0.139385, 0.052485–0.170075, 0.035689–0.105394 y
0.046101–0.179363, respectivamente. Son cambios en costos del lector, no
tensiones físicas medidas.

La intervención no siempre altera la entrada: las rotaciones y el sham
original cambian, cada una, 1.530/1.536 entradas en beta y 1.535/1.536 en deformada,
con 18 y 3 casos entrenados de delta constante, respectivamente. Los ceros
cambian todas; IID y polifonía cambian todas las entradas y componentes.
Esto acredita dependencia funcional del canal, no que su interpretación de
fuente física sea correcta ni que toda intervención destruya información.

El oracle elige con etiquetas dentro del pool; **no es un lector desplegable**.
Sobre 1.536 pares escena × checkpoint por escenario, su ARI máximo medio es
0.978804, 0.947371, 0.931297 y 0.897675. Sus brechas frente a Compartida
son 0.017039, 0.205411, 0.027380 y 0.160098. Los mínimos de VI medios,
calculados como oracles separados, son 0.037330, 0.093860, 0.145716 y
0.171473. Esa cobertura no demuestra que las particiones sean reconocibles
desde los inputs ni identifica la causa de la brecha.

## Alcance geométrico y siguiente decisión

El experimento interroga la correspondencia entre relaciones espectrales,
compatibilidad de fuentes y selección de una partición global. El costo
conjunto puede ayudar bajo algunos cambios y perjudicar bajo otros: imponer
una familia geométrica no asegura que sea una buena guía fuera de ella.
La partición plantada por el generador tampoco equivale automáticamente a
una identidad física identificable desde la observación.

No se atribuye este resultado a audio medido, coherencia temporal o fase:
esos canales no están presentes. Quedan abiertas operaciones relacionales,
objetivos de aprendizaje, representación de incertidumbre y medición real.
La auditoría de alineación recomienda examinar márgenes y fuentes rivales
bajo el mismo puerto observado. El [siguiente diseño](PLAN_OBSERVABLE_SOURCE_RIVALS.md)
acota ese diagnóstico, sin otra ronda de tuning del test ni una campaña
bibliográfica sin discriminante experimental. Encontrar fits rivales bajo
ruido gaussiano no certifica por sí solo no-identificabilidad estadística;
no encontrarlos en una búsqueda finita tampoco demuestra unicidad.

## Evidencia recuperable

Los cuatro replays reproducen byte a byte los ocho payloads de cada
evaluación, sin reentrenar ni nuevos forwards. Se conservan checkpoints,
predicciones por candidato, métricas por unidad, índices bootstrap, soporte,
configs y manifestaciones de ejecución. Las correcciones de validación y el
límite operativo de RAM se documentan en el [perfil de recursos](RESULTS_LEARNED_PARTITION_RESOURCE_PROFILE.md);
los intentos incompletos permanecen separados de los resultados completos.

Raíz de artefactos: `data/atencion_armonica/learned_partition_reader_v1/`.
El roster `evaluation_release_recovery_v2/tests_01.json`, SHA256
`7f4f368cf22ea60a2d02faa92099933cdbec2dcb13a69dc1df2a603f45c66153`,
enlaza por contenido datos, modelos, selección, evaluaciones y replays.

| Escenario | Ruta de summary.json bajo la raíz | SHA256 |
|---|---|---|
| IID | `test_memory_recovery_v1/outputs/iid/evaluation_01/summary.json` | `b835a9849c69159842192620400e022011ffc8da7f06557bc8a5c4a2787bc39b` |
| Mayor beta | `test_memory_recovery_v1/outputs/ood_beta/evaluation_01/summary.json` | `860f4686501c425817fd996728060c0ace3d5fb2ba3db018c5bea739715146f1` |
| Polifonía | `evaluation_release_recovery_v2/outputs/ood_polyphony/evaluation_01/summary.json` | `fb51ac102e99b5c36416136171d36f5544e1583b2e1f8a0f69de36a8c4e71c62` |
| Deformada | `evaluation_release_recovery_v2/outputs/deformed_family/evaluation_01/summary.json` | `1bcb4345fefbfbe45df87a71afe8760f30b4db1e365ef004eaa740010a460da0` |
