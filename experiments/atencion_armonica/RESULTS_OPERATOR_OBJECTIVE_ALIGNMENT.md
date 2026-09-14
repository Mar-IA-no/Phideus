# Orden geométrico, aprendizaje y decisión

2026-09-14. Diagnóstico retrospectivo completo, replay exacto y auditorías
independientes técnica y de alineación cerradas sin hallazgos materiales
abiertos dentro de sus alcances. No hubo entrenamiento, forward, refit,
nuevas escenas ni uso de GPU.

## Resultado principal

En la familia deformada, en media por escena, las cabezas aprendidas ordenan
mejor el conjunto de candidatos según la VI normalizada matemática (target
float64), pero la regla geométrica clásica elige mejor su mínimo.
La correspondencia relevante no puede reducirse a
correlación global: preservar la decisión es una exigencia distinta.

El [protocolo](PROTOCOL_OPERATOR_OBJECTIVE_ALIGNMENT.md) compara el mínimo de
cota superior de ajuste conjunto de **Extendida** con la suma de entropías
condicionales normalizadas, una VI normalizada. **Generativa** recibe evidencia
de ese ajuste; **Local** no la recibe; **Desacoplada** recibe la intervención
aprendida ya fijada. Las cabezas y predicciones son las del
[contraste cerrado](RESULTS_GENERATIVE_EVIDENCE_READER.md). El mínimo UB no
representa todo el vector de evidencia entregado a la cabeza.

Cada escenario conserva sus 512 escenas. Se promedian las nueve celdas de
cada brazo dentro de escena y luego escenas; no son nueve réplicas
independientes. Las escenas sin candidatos permanecen registradas con métricas
indefinidas, no cero. La siguiente tabla usa las 484 escenas válidas del
primario deformado; otras 28 no tienen candidatos.

| Método | Kendall τ-b ↑ | Regret VI ↓ | Elección óptima VI ↑ | ARI ↑ |
|---|---:|---:|---:|---:|
| Extendida UB | 0.240753 | 0.030361 | 0.811983 | 0.883872 |
| Local | 0.432132 | 0.083863 | 0.533976 | 0.783519 |
| Generativa | 0.434221 | 0.082377 | 0.543388 | 0.786475 |
| Desacoplada | 0.428900 | 0.086137 | 0.524564 | 0.779618 |

Regret es la diferencia entre el target del candidato elegido y el mínimo
dentro del mismo universo observable. Elección óptima admite todos los
coóptimos exactos, no sólo su representante canónico. Generativa−Extendida da
τ +0.193467, regret +0.052016 y ARI −0.097396. Generativa mejora modestamente
las medias de Local y Desacoplada en esas tres medidas del primario; esto no
añade una prueba confirmatoria ni convierte la comparación con Extendida en
atribución causal de un módulo. Exigir soporte común a las nueve celdas no
cambia estos contrastes ni sus 484 escenas.

## Qué explicaciones delimita

Los oracles de mínimo VI y máximo ARI eligen la misma firma en 476/484 escenas
y sus conjuntos óptimos se intersectan en 479/484. Elegir por VI pierde sólo
0.000442 de ARI medio respecto del oracle ARI. El mínimo float32 cambia la
firma frente a float64 en una escena; permanece dentro del vecindario
predeclarado de 1e-12. Estas discrepancias observadas no bastan para explicar
la gran diferencia de decisión entre Generativa y Extendida. No demuestran
equivalencia universal de objetivos ni que la regresión MSE sea adecuada.

La ausencia de candidatos tampoco explica toda la diferencia: cuando la
partición plantada está en el pool, Extendida y Generativa alcanzan ARI
0.978048 y 0.882552 en 351 escenas; cuando está entre los vecinos, 0.913634 y
0.604328 en 34. El slice ausente conserva 127 escenas, 99 con candidatos y
28 sin ellos. Estos slices usan información privilegiada sólo para evaluar.

Al restringir la competencia entre candidatos cambia la comparación:

| Universo dentro de escena | Regret pareado Generativa−Extendida |
|---|---:|
| Completo, primario | +0.052016 |
| Estratos por número de grupos | +0.045873 |
| Tamaños y disponibilidad de ramas | +0.009694 |
| Además, rama minimizadora de Extendida | −0.009746 |

Todos estos regrets tienen soporte de 484 escenas, pero son estimandos
distintos: cada esquema recalcula mínimos y promedia uniformemente sus
estratos definidos. La última clave depende del propio score clásico.
La inversión sugiere investigar la comparabilidad entre estratos; no prueba
que una descalibración sea la causa, ni permite sumar diferencias como una
descomposición causal. Tampoco justifica ajustar una corrección sobre los
tests abiertos ni usar los slices privilegiados de presencia como inputs.

## Otros escenarios y regresión

No hay una conclusión única para todos los escenarios. Diferencias pareadas
Generativa−Extendida en el universo completo:

| Escenario | Escenas válidas / total | Δτ | Δregret VI | ΔARI |
|---|---:|---:|---:|---:|
| IID | 505/512 | +0.331951 | +0.001813 | −0.002254 |
| Mayor beta | 497/512 | +0.208048 | +0.052192 | −0.094729 |
| Polifonía | 451/512 | +0.315916 | −0.000452 | −0.002079 |

En polifonía, mejorar ligeramente el regret frente a Extendida no mejora ARI
ni frecuencia de elección óptima. Frente a Desacoplada, Generativa tiene
mayor regret (+0.004071) y menor ARI (−0.006387), aunque su error medio por
componente es menor: 0.002847801 frente a 0.002877057. Por tanto, reducir esa
MSE agregada tampoco garantiza mejorar la decisión observada.

En el primario, el error medio por componente de Generativa es 0.003245948
frente a 0.003325678 de Local; el error de la suma, en cambio, es 0.010591688
frente a 0.010547261. Son reducciones diferentes. La MSE diagnóstica respeta
la ponderación del entrenamiento, pero usa operandos float64 y no promete
reproducir bit a bit una reducción Torch float32. La época conservada fue
elegida por ARI de calibración, no por MSE.

## Balance geométrico y siguiente pregunta

El diagnóstico distingue ajuste geométrico, orden global, regresión de
componentes y decisión de partición. No identifica todavía representación,
optimización, selección de época o función de pérdida como causa. La ley
sintética conocida tampoco aporta ground truth físico independiente ni
validación de Harmonic Information Theory.

La clase de experimento que merece seguir es prospectiva: contrastar si una
interfaz de energía y un aprendizaje orientados a conservar el mínimo que
decide la partición aprovechan mejor la estructura geométrica que la regresión puntual
de componentes. Debe separar el cambio de operación/representación del cambio
de loss, mantener controles Local/descriptores y la referencia clásica, y
evaluar sobre nuevas escenas con selección sólo TRAIN/calibración. No se
selecciona todavía una arquitectura ganadora ni se atribuye causalidad a la
MSE. Este banco de agrupamiento sigue siendo un medio para estudiar geometría
computable, no el horizonte completo del proyecto.

## Evidencia y alcance de verificación

Raíz de artefactos: `data/atencion_armonica/operator_objective_alignment_v1/`.
El barrido y replay completaron 2048 bundles y cuatro resúmenes exactos, con
1350.970249 s acumulados, incluida la auditoría técnica final. La
[enmienda operativa](AMENDMENT_OPERATOR_OBJECTIVE_EXECUTION.md) conserva el
ledger previo y fija explícitamente 7200 s; no modifica el protocolo científico.

Las tablas proceden de `summaries/<scenario>/summary.json`,
`slices.all.metrics`: claves `classical/extended_ub`, `arm/<arm>` y
`paired/<contrast>`, seguidas de esquema y métrica. Los oracles están bajo
`oracle/full`; los slices de presencia bajo `slices.pool`, `neighbor` y
`absent`. Cada métrica conserva `mean`, `defined`, `undefined`, cuantiles y
soportes. Identidades SHA256:

| Artefacto relativo a esa raíz | SHA256 |
|---|---|
| `manifest.json` | `c85c0a64267526218e37c9ee6c20618cc4c66732691480c973252d494484f9b9` |
| `execution/complete.json` | `167a35133ec6a20ecc9718daf642918805c90b9c311d953b587cb8fa846cb184` |
| `execution/replayed.json` | `ba2fe2141fac07f2763fb4e02c4a0aa6412b1bf61e183326db697397b68a8ae4` |
| `attempts/0008/final_technical_audit.json` | `041605c7677cb0e13da94b789e4442b4538cbf1102762e4756d59e9fc6ff8aa4` |
| `attempts/0008/finish.json` | `5f4d2d17c37f42dcd178407c42b93aa37a6028862c2de6c1b90865b7698dff24` |
| `summaries/deformed_family/summary.json` | `a71a1c858e656cdeac2080f180673456d132861fa90120f4071fbf6ab4201d72` |
| `summaries/iid/summary.json` | `7dee2f8a5c94ffded4851c69abdcea563618af17efb78d300eb0d860e074ae7f` |
| `summaries/ood_beta/summary.json` | `b9197ad9d11e538d82a93cce062bbae97e00eb1e99ddedf7c3a4110f08c5f716` |
| `summaries/ood_polyphony/summary.json` | `41bffbe6b576fab348eeac8ec050d7ee949be1950d53cf405cdf2fc8dd6952b0` |

El replay prueba reproducibilidad desde compactos, no independencia de la
implementación. La auditoría técnica autenticó el cierre completo y reextrajo
16 escenas seleccionadas de antemano por posición, ausencia y máximo número
de candidatos. La extracción reutiliza puertos autenticados; las entropías,
elecciones, regrets y tau tienen comprobaciones independientes. Pasaron 496
controles de método, incluidos casos vacíos. No es una reextracción
independiente de las 2048 fuentes ni una muestra representativa.

Una segunda instancia leyó íntegramente el informe y sus contratos y verificó
cifras, soportes e interpretación geométrica. El relevo exige separar
operación/interfaz y loss, no sólo cambiar una pérdida de ranking ni optimizar
tau. Queda respondida la pregunta local del diagnóstico y justificada esa
clase de experimento prospectivo; su diseño concreto es el siguiente trabajo.
No hay promoción ni GO/NO-GO.
