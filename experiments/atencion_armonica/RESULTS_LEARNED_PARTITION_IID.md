# Lector aprendido de particiones: resultado IID parcial

Fecha: 2026-09-08. Estado: inferencia, evaluación y replay IID completos;
auditoría independiente de evidencia completada sin hallazgos materiales en
este alcance parcial. Los tres tests fuera de distribución siguen
pendientes por suspensión de GPU. No es el cierre del contraste ni una
promoción arquitectónica.

## Qué se puso a prueba

El [protocolo](PROTOCOL_LEARNED_PARTITION_READER.md) compara cuatro lectores
aprendidos sobre el mismo algoritmo de candidatos y tres redes congeladas.
Las 36 corridas completaron 50 épocas; la selección común por brazo quedó
fijada exclusivamente desde calibración y pasó revisión independiente.
Este test contiene 512 escenas IID nuevas, seed2026090882. Cada escena tiene
predicciones de tres checkpoints y tres inicializaciones de lector: esas nueve
predicciones no son nueve escenas independientes.

Fuente compartida obtiene una mejora pequeña de ARI medio frente a los tres
controles aprendidos en IID. La comparación frente al lector fijo de pares
no muestra una ventaja clara con el intervalo descriptivo empleado. La pregunta
primaria —generalización a mayor polifonía— todavía no tiene resultado.

## Medias IID

Se promedian primero las nueve predicciones de cada escena y luego las escenas.
Partición exacta y masa sub-3 se expresan como fracciones, no porcentajes.

| Lector | ARI | Partición exacta | Error absoluto de k | Masa sub-3 |
|---|---:|---:|---:|---:|
| Pares/estructura aprendido | 0.960128 | 0.841146 | 0.095052 | 0.007130 |
| Compatibilidad local aprendida | 0.960655 | 0.845052 | 0.081814 | 0.006011 |
| Fuente compartida aprendida | 0.961765 | 0.851345 | 0.083116 | 0.006261 |
| Fuente desacoplada aprendida | 0.960574 | 0.843316 | 0.090061 | 0.006603 |
| Pares fijo | 0.961502 | 0.844401 | 0.125651 | 0.009737 |
| Fuente compartida fija | 0.961365 | 0.839193 | 0.156250 | 0.012229 |
| Histórico | 0.959843 | 0.852865 | 0.037109 | 0.003463 |

La mejora no es uniforme entre métricas: Local tiene menor error absoluto de k
y menor masa sub-3 que Compartida; el Histórico conserva mayor partición exacta
y menor error absoluto de k. El ARI mayor no sustituye esas diferencias.

| Compartida aprendida menos control | Delta ARI | Intervalo descriptivo 95% |
|---|---:|---:|
| Pares/estructura aprendido | +0.001638 | [0.000831, 0.002538] |
| Compatibilidad local aprendida | +0.001110 | [0.000460, 0.001838] |
| Fuente desacoplada aprendida | +0.001191 | [0.000595, 0.001843] |
| Pares fijo | +0.000264 | [-0.001771, 0.002126] |

Son intervalos percentiles con 2.000 remuestreos pareados por escena,
condicionales a los checkpoints e inicializaciones seleccionados. En IID son
descriptivos al 95%, sin la corrección del primario de polifonía. No convierten
esta lectura parcial en decisión confirmatoria del goal.

## Geometría, aprendizaje y límites

El factor físico no fue completamente ignorado: las intervenciones del canal
modificaron costos y algunas decisiones preservadas. Eso demuestra dependencia
del lector, no que su interpretación física sea correcta. El resultado compara
acceso a ese factor dentro de una receta común; no aísla cabeza y loss frente
al lector fijo ni demuestra una nueva geometría latente del backbone congelado.

La evidencia sigue siendo sintética y sólo de frecuencia. No certifica
identificabilidad de fuentes, validez general de HIT ni comportamiento con picos
detectados en audio real. Mayor inarmonicidad, mayor polifonía y familia
deformada permanecen pendientes; no se cambian receta, selección ni muestras
para acomodar este resultado IID.

## Reproducibilidad y continuidad

Las 99 predicciones y los 63 diagnósticos quedaron sellados antes de leer
etiquetas. La evaluación y su replay reprodujeron exactamente los ocho payloads,
sin reentrenamiento ni nuevos forwards. Se conservan métricas por escena,
checkpoint e inicialización, costos por candidato, índices bootstrap, soporte
y las métricas de errores split/merge, pares, VI y k completas.
Los [recursos y la recuperación](RESULTS_LEARNED_PARTITION_RESOURCE_PROFILE.md)
se documentan aparte. La suspensión vigente impide ejecutar los forwards GPU
restantes; el goal completo queda abierto.

Fuentes bajo `data/atencion_armonica/learned_partition_reader_v1/test_memory_recovery_v1/`:

- `outputs/iid/evaluation_01/manifest.json`: `f582792ea90a073fecbbef369196748499d7c8e3ec4882c3e8c7786646102dd0`.
- `outputs/iid/evaluation_01/summary.json`: `b835a9849c69159842192620400e022011ffc8da7f06557bc8a5c4a2787bc39b`.
- `outputs/iid/replay_01/manifest.json`: `0da511880bdd3bf369f6e9465c4b7cc28af87a3ef55989647779f7d09f0438d0`.
