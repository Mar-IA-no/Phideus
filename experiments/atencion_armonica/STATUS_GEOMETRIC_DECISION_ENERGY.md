# Energía geométrica para la decisión — estado de implementación

2026-09-15. Los 72 entrenamientos, la selección, los 144 estados archivados,
las 2048 escenas prospectivas, evaluación, recuperación y reporte de datos
están completos. Los resultados ya fueron consultados después del sello y
del replay. La verificación técnica ejecutable, las revisiones independientes
y su integración documental están completas.

El supervisor integrado y la enmienda de contabilidad cerraron su auditoría
técnica antes de ejecutar. La suite CPU completa pasó 285 pruebas. La admisión
real terminó en 5,178484 s y congeló cuatro escenarios de 512 escenas, los
144 estados iniciales/seleccionados, fuentes y recursos. El recorrido posterior
completó observables, sello, evaluación y ambos replays sin reentrenamiento.
Los cierres operativos no sustituyen la comprobación independiente de la
procedencia, los cálculos y la interpretación.

| Etapa terminada | Recibo bajo `control/attempts/` | Tiempo del intento (s) |
|---|---|---:|
| Observables prospectivos | `0013/finish.json` | 12227.353831 |
| Evaluación | `0015/finish.json` | 1212.697706 |
| Recuperación observable | `0017/finish.json` | 4140.501281 |
| Replay de métricas | `0018/finish.json` | 1197.365631 |
| Reporte reproducible | `0019/finish.json` | 125.177389 |

Son tiempos medidos de esos intentos, no el coste acumulado completo. La
[auditoría final](PLAN_GEOMETRIC_DECISION_FINAL_AUDIT.md) distingue autenticación
integral, reconstrucción matemática de un corte estructural y recomputación
de los cuatro primarios. Las correcciones del verificador conservan los
cortes y los cargos anteriores; no modifican los artefactos experimentales.
La revisión técnica del cierre real no encontró findings materiales. La
revisión de horizonte y las correcciones de cobertura del informe están
integradas. La [lectura de resultados](RESULTS_GEOMETRIC_DECISION_ENERGY.md)
distingue ventaja relativa, comparación con el clásico inicial y diferencias
entre escenarios. La revisión de propagación documental resolvió sus findings;
el contraste queda cerrado dentro de ese alcance, sin arquitectura promovida.

El intento final `0025` completó VERIFY en 614.107661 s de CPU. Su resultado
`audit-final-coverage-verify/result.json` tiene SHA256
`1e51936b53709034f9ee47994dbd16825cb16884c10907309a90a46c19794e28`;
el recibo `control/attempts/0025/finish.json`,
`9b012d71375260e7bc8498bb5a65f6fc1e7581fa2fa0d703e32410f2ae95b103`.
El cargo acumulado de auditoría es 2483.276911 s y el total contable de etapas,
26579.544657 s: incluyen cargos anteriores y una reserva imputada por un intento
sin finish, no son exclusivamente tiempo medido de corridas completas.

Fuentes bajo `data/atencion_armonica/geometric_decision_energy_v1/`:
`control/freezes/prospective.json`, SHA256
`febc010fa6c88651ecf684d97e7fc0d2b8707c4223357df6c81fe7e61f9dd3fe`;
`control/attempts/0012/finish.json`, SHA256
`09126495a11ec36969b27e43ccf8fcb9934dde23cd05277ab6fcbc257ed730a2`;
`control/attempts/0013/finish.json`, SHA256
`1765cbaa04191bdfa1083df726aaaf339489d9369dc94a662de9b381225c838f`;
`report/complete.json`, SHA256
`c213d59be4f8d805efa6e150784bb8d391fb8387601d2a91dd81d7406a51cd49`.

## Diseño y ejecución conservados

Los párrafos siguientes distinguen las pruebas de interfaces previas a la
campaña de su ejecución posterior; sus costes parciales no deben sumarse como
si fueran todos intentos diferentes ni leerse como un estado pendiente actual.

El [protocolo](PROTOCOL_GEOMETRIC_DECISION_ENERGY.md), auditado antes de la
campaña, fija cuatro rutas por dos pérdidas: Inyección, Geométrica,
Desacoplada y Local, cada una con MSE o surrogate de decisión. Son 72
entrenamientos en el roster de la campaña activa. La comparación separa la ruta
geométrica en la decisión del objetivo de aprendizaje.

La [cabeza común](../../src/atencion_armonica/geometric_decision_model.py)
implementa 2258 parámetros, salida firmada y suma float64. Los tres brazos
no Local reciben los mismos ocho canales float32; el bypass usa exactamente
la columna entregada, sin precisión ni parámetros extra. La corrección
inicial nula reproduce el bypass por construcción: no es aprendizaje.

El [núcleo observable](../../src/atencion_armonica/geometric_decision_core.py)
implementa mínimo de ramas disponibles, escala RMS por escena, donantes,
selector de empates por firma y roundtrip de cuantización/recentrado. Las
[pruebas del núcleo](test_geometric_decision_core.py) incluyen coóptimos,
aritmética del target, gradientes, padding, donantes singleton, universo vacío
y transporte. Su auditoría independiente cerró sin findings abiertos en ese
alcance; esas pruebas aisladas no constituyen evidencia de rendimiento
experimental.

El [adaptador OPEN](../../src/atencion_armonica/geometric_decision_open.py)
lee los agregados ya autenticados sin modificar sus stores. Sus
[pruebas de frontera](test_geometric_decision_open.py) cubren recibos,
rutas, formatos y exclusión de roles test. La auditoría del adaptador también
comprobó el roster completo mediante fixtures, el peso igual por escena de la
escala y la conservación del diagnóstico desacoplado. Una lectura sobre
un shard TRAIN y los tres backbones pasó sin abrir targets. Esto no equivale
a adaptar todo el corpus ni a congelar la nueva escala. La ejecución posterior
del [operador de preparación](prepare_geometric_decision_open.py), ya auditado,
completó las 27 entradas sobre las 4096 escenas TRAIN y 512 de calibración en
50,708750 s. El cierre local autenticado tiene SHA256
`70f5996408403ade1e83f67fd84441cd983001dd9ba789ff3f3ab29047adcb6f`.
La escala TRAIN es `7,5314913344363035`, con 4036 escenas elegibles y 60 sin
candidatos; estas últimas permanecen en el roster sin recibir peso artificial.
El recibo de escala tiene SHA256
`25da9bd505efb0b594e88755f0e20c0018919646a4149e8d1b6faae19e26bd8e`.
La ejecución no abrió targets ni repitió ajustes o forwards. El consumo OPEN
acumulado, incluido el smoke anterior, es 54,502751 s de su tope de 1800 s.

El [runner de una celda](../../src/atencion_armonica/geometric_decision_cell.py),
su kernel y su store ya fueron auditados. Conservan estados de modelo,
optimizador y generadores aleatorios, checkpoints recuperables y salidas
firmadas float64 de calibración al inicio y cada cinco épocas. Las
[pruebas de recuperación](test_geometric_decision_cell.py) completan 50 épocas
con fixtures de soporte elegible pequeño y comprueban igualdad exacta frente
a una ejecución interrumpida. No son entrenamientos científicos. Los
[perfiles CPU/CUDA](RESULTS_GEOMETRIC_DECISION_PROFILE.md) ya completaron la
prueba positiva de recuperación exacta CUDA en cuatro casos mecánicos,
además de medir cabeza, fitter y carga completa. Su consumo acumulado es
71,594984 s de los 600 s de perfil; no se repitieron los ajustes de todo OPEN.

La [preparación durable](../../src/atencion_armonica/geometric_decision_corpus.py)
y el [selector de época](../../src/atencion_armonica/geometric_decision_selection.py)
están implementados y auditados con pruebas mecánicas de sus puertos.
El selector exige las nueve celdas por brazo y las diez épocas elegibles;
no elige una semilla o backbone ganador. El
[operador de selección](select_geometric_decision.py) y su
[perfil CPU específico](../../src/atencion_armonica/geometric_decision_selection_profile.py)
también quedaron implementados y auditados. Reautentican el cierre integral
de cada celda, su estado final y los once checkpoints de calibración;
no aceptan un output parcial como finalización. Su ejecución real completó
la selección sobre las 720 calibraciones elegibles, sin forward ni fitting.
El perfil CPU específico consumió 4,909168 s y la selección 109,324164 s;
el ledger acumula 76,504152 s de perfil y 109,324164 s de evaluación/replay.
La selección conserva una época común por brazo, no una semilla ganadora:

| Ruta | MSE | Decisión |
|---|---:|---:|
| Inyección | 45 | 30 |
| Geométrica | 40 | 45 |
| Desacoplada | 45 | 40 |
| Local | 50 | 45 |

La fuente es `data/atencion_armonica/geometric_decision_energy_v1/selection/selection.json`,
SHA256 `a0e8912bab71ed7b79745d802477962b6fcb472e44eb2cf53ead54c726d35abc`.
El cierre del operador, `control/attempts/0008/finish.json` bajo la raíz
experimental, tiene SHA256
`b1933a5822291892595154a885f325292067214ba095f12c63dd397d7d5d5700`.
Estas épocas son decisiones de calibración, no evidencia de generalización.

El [supervisor completo](train_geometric_decision.py), auditado junto con la
admisión y proyección de recursos, completó las 72 celdas en CUDA. Reutiliza
una carga por backbone y conserva las celdas completas sin reentrenarlas al
recuperar. El [plan de ejecución](PLAN_GEOMETRIC_DECISION_CAMPAIGN.md) mantiene
las 50 épocas, semillas, pérdidas y controles del protocolo. El entrenamiento
completo consumió 4912,186859 s, dentro de la reserva de 6046 s y del tope
acumulado de 14400 s. El cierre `training/complete.json` tiene SHA256
`deb1b7c6b7059255fa08a313f3b45278ddd525077e41d89a3dab3aaa18c8ca81`;
su recibo `control/attempts/0006/finish.json` tiene SHA256
`bda2550d6cd90521e59a2f33c3a9f22b7e49197648303f86357a12a21ccd6378`.
Ambos pertenecen a `data/atencion_armonica/geometric_decision_energy_v1/`.
Completar entrenamiento y selección no permite interpretar ventajas entre brazos.

La admisión integral, el freeze y el recorrido prospectivo ya están completos.
La interpretación de generalización está contrastada con la verificación final;
no depende únicamente de terminar entrenamiento y selección.

El [ensamblado observable](../../src/atencion_armonica/geometric_decision_observables.py)
y el [almacén de escenas](../../src/atencion_armonica/geometric_decision_scene_store.py)
ya están implementados y auditados con 25 pruebas CPU. Conservan features,
logits de tres backbones, factores de ajuste completos y entradas raw/delivered;
los fits se reconstruyen desde sus factores para verificar coherencia sin
repetir la grilla. Los probes enlazan una fuente original autenticada y
preservan el linaje por evento y las coordenadas del roundtrip. La auditoría
cerró sus findings dentro de ese alcance. Estos puertos no autorizan draws:
el freeze y el operador prospectivo deben establecer la procedencia de las
observaciones y de la normalización. Su ejecución posterior forma parte del
recorrido prospectivo terminado que se resume al inicio.

El [ensamblador recuperable](../../src/atencion_armonica/geometric_decision_pipeline.py),
la extensión de exclusiones y el archivo numérico de cabezas pasaron auditoría
de sus interfaces. Dos defectos de identidad y recuperación se corrigieron
antes de usarlos: JSON no canónico se rechaza antes del cálculo y los callbacks
no pueden modificar los metadatos autenticados por aliasing. Las pruebas de
estas tres interfaces son mecánicas. La ejecución posterior del
[operador CPU de archivo](prepare_geometric_decision_archive.py), auditado y
admitido desde los cierres COMPLETE, conservó los 144 estados initial/selected
y extendió las exclusiones con las 2048 observaciones históricas del contraste
anterior. El inventario reúne 24467 huellas únicas, incluidos antecedentes y
fixtures; no es un conteo de nuevas escenas experimentales. Consumió
9,938720 s y no abrió tests nuevos. El cierre `archive/complete.json` bajo la
raíz experimental tiene SHA256
`f36a55dcc9dd5c3c25a7429fe044f74511ebd07cc4bde60b873b9ba3c89d7f75`;
`control/attempts/0009/finish.json`,
`158fe1bc539f0c31e4baaa6191ba963d3cb6573531f8865dd146c1a933625e1c`.
El archivo no autorizaba draws por sí solo: la admisión integral y el freeze
se completaron después, antes de la ejecución prospectiva.

Los puertos de producción única de observaciones y conservación de predicciones
también pasaron auditoría y 36 pruebas CPU. La recuperación de los transportes
revalida arrays, permutaciones, energía y diagnósticos sin ejecutar el modelo;
el roster de primeras cuatro escenas elegibles se recompone antes de usarse.
Las correcciones mantienen la frontera observable sin campos de supervisión.
La integración de esos puertos y la referencia clásica raw UB pasaron auditoría
antes del perfil real. El [perfil observable](RESULTS_GEOMETRIC_DECISION_PROFILE.md)
completó las 16 escenas TRAIN fijadas y cuatro roundtrips, incluidos los 144
estados y recuperación exacta sin forward ni fitting. Consumió 113,436136 s;
el ledger acumula 189,940288 / 600 s de perfiles. Sus proyecciones siguen
siendo parciales: faltan generación, sello y métricas en la admisión completa.
La extensión de exclusiones de sus coordenadas pasó auditoría como interfaz
y después quedó incorporada a la admisión prospectiva. El perfil no constituye
por sí mismo un resultado de generalización ni un cambio de arquitectura.

El [plan de métricas](PLAN_GEOMETRIC_DECISION_METRICS.md) implementa el
estimando ya fijado: targets matemáticos y entregados separados, puntuaciones
firmadas float64 y bootstrap pareado por escena. Su núcleo y el adaptador de
admisión del perfil pasaron auditoría de sus interfaces y 18 pruebas CPU;
la suite conjunta de ese corte completó 247 pruebas. La integración posterior
del sello global y de la evaluación por identidad de evento pasó auditoría
y pruebas mecánicas. Exige los cuatro batches completos antes de leer respuestas
y conserva replay exacto sin reparar resultados ausentes.

El [perfil CPU de cierre](RESULTS_GEOMETRIC_DECISION_PROFILE.md) ya ejecutó
la evaluación y su replay sobre 16 TRAIN conocidas y cuatro probes, incluidos
los 144 estados; terminó en 25,693755 s sin nuevos tests. El coste acumulado de
perfiles es 215,634043/600 s. La proyección original de evaluación/replay excede
su reserva restante; motivó una revisión operativa explícita, sin reducir
controles ni muestras. La revisión y el supervisor integral se auditaron
después; la admisión real y el freeze están enlazados al inicio de este estado.
El contraste terminó evaluación, replay y verificación independiente;
su integración documental se resume al inicio.

El protocolo conserva la corrección anterior al freeze de una seed IID
abierta durante una comprobación de diseño: queda retirada y excluida,
sin presentarla como test prospectivo. El reemplazo no se eligió por
resultados de modelos. No hubo entrenamiento ni comparación de arquitecturas
en ese incidente.

No se promueve una arquitectura. El operador geométrico sigue siendo externo
y usa una ley sintética conocida; la pregunta del experimento es si conservar
esa operación en la decisión aporta algo más que entregarla como descriptor.
