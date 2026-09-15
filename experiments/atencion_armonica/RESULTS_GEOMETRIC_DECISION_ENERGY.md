# Geometría en la decisión: ventaja relativa, no ganador universal

2026-09-15. Campaña, replay y auditorías técnica, de interpretación y de
alineación completos, con findings materiales resueltos. No hay promoción
de arquitectura ni decisión de GO/NO-GO.

## Pregunta y contraste

El [diagnóstico anterior](RESULTS_OPERATOR_OBJECTIVE_ALIGNMENT.md) distinguió
ordenar alternativas y elegir su mínimo: una cabeza podía mejorar el orden
global y elegir peor que el ajuste geométrico clásico. Este experimento
pregunta si la estructura ayuda más cuando participa directamente en la
decisión que cuando entra sólo como descriptor, y cómo cambia esa comparación
con el objetivo de aprendizaje.

El [protocolo predeclarado](PROTOCOL_GEOMETRIC_DECISION_ENERGY.md) cruza cuatro
rutas con dos pérdidas:

| Ruta | Intervención |
|---|---|
| Inyección | Evidencia geométrica como input de la cabeza |
| Geométrica | El ajuste conjunto participa directamente en la energía; la cabeza aprende una corrección |
| Desacoplada | El término directo procede de otro candidato del mismo estrato; conserva inputs alineados |
| Local | No recibe esos canales geométricos |

MSE aproxima dos componentes de error de partición. Decisión usa un surrogate
de margen sensible al regret entre candidatos. Sólo supervisa la suma de
las salidas: sus componentes no son entropías calibradas ni energías físicas.
La cabeza tiene 2258 parámetros; los tres brazos no Local reciben los mismos
ocho canales. Local tiene menos información y capacidad activa, por lo que
no constituye un control de acceso idéntico. El bypass geométrico introduce
un prior inicial y una trayectoria de aprendizaje distintos; el contraste
no identifica por separado ambos mecanismos.

Se completaron 72 entrenamientos de 50 épocas: ocho variantes, tres backbones
congelados y tres semillas de lector. La selección usó calibración, no tests,
y conservó una época común por variante. Se evaluaron 144 estados iniciales
y seleccionados sobre cuatro escenarios de 512 escenas nuevas cada uno.
Los checkpoints, predicciones, configuraciones, selección y replay se preservan.

La entrada es sintética: logfrecuencias con ruido gaussiano de 2 cents,
centradas, convertidas a float32 y permutadas. No incluye amplitudes ni fases
y no procede de detección sobre audio. La identidad de los componentes se
conserva para evaluar; las familias conocidas forman parte del diseño del fitter.

## Resultado primario

El primario es regret matemático de partición (`regret_tM`) en familia
deformada, sobre 489 escenas elegibles; las otras 23 permanecen registradas
como universos vacíos. Menor regret es mejor. Primero se promedian las nueve
celdas dentro de escena y después las escenas. Los intervalos percentiles
del 98.75% usan 10000 remuestras pareadas por escena; están condicionados a
esas celdas, no son incertidumbre sobre nuevas semillas o backbones ni un
umbral de decisión científica.

| Contraste predeclarado | Diferencia de regret | IC 98.75% |
|---|---:|---|
| Geométrica − Inyección, MSE | −.034946 | [−.042795, −.027610] |
| Geométrica − Inyección, Decisión | −.033541 | [−.041439, −.025869] |
| Interacción Decisión − MSE | +.001405 | [−.004363, +.007390] |
| Geométrica − Desacoplada, Decisión | −.032555 | [−.040393, −.024923] |

La ruta geométrica tiene menor regret que Inyección bajo ambas losses y que
Desacoplada bajo Decisión en este primario. El intervalo de interacción
incluye cero: no acredita una interacción especial, pero tampoco demuestra
equivalencia de efectos.

La referencia clásica impide confundir esa ventaja relativa con una mejora
del punto de partida. Geométrica-Decisión obtiene regret .037770 y ARI .870531;
la regla Extendida y el estado geométrico inicial obtienen .031161 y .884263.
La media seleccionada queda peor que la inicial en este escenario bajo la
receta fijada. Es una comparación descriptiva, sin nuevo intervalo pareado
selected–initial, y no mide fidelidad a las elecciones del solver.

## Loss y cambios de escenario

Las ocho variantes en familia deformada muestran una mejora descriptiva con
Decisión respecto de MSE; no es exclusiva de la ruta geométrica:

| Ruta | Initial MSE / Decisión | Selected MSE | Selected Decisión | Selected Decisión − MSE |
|---|---:|---:|---:|---:|
| Inyección | .311185 / .311185 | .084822 | .071312 | −.013510 |
| Geométrica | .031161 / .031161 | .049876 | .037770 | −.012106 |
| Desacoplada | .298733 / .298733 | .084292 | .070325 | −.013967 |
| Local | .311185 / .311185 | .086303 | .066060 | −.020243 |

Las otras tres rutas mejoran desde reglas iniciales con mucho mayor regret.
Inyección y Local parten constantes; Geométrica y Desacoplada heredan bypass
distintos. Comparar los estados seleccionados no aísla el aprendizaje de
esa asimetría inicial. Las diferencias de loss son descriptivas.

La comparación cambia según el escenario. Esta tabla es descriptiva, no una
ampliación de los contrastes primarios:

| Escenario | Elegibles /512 | Inyección-Decisión | Geométrica-Decisión | Local-Decisión | Extendida clásica |
|---|---:|---:|---:|---:|---:|
| IID | 499 | .009686 | .009453 | .009196 | .010982 |
| Mayor beta | 498 | .019150 | .014635 | .011502 | .021435 |
| Mayor polifonía | 460 | .012805 | .014192 | .011025 | .017996 |
| Familia deformada | 489 | .071312 | .037770 | .066060 | .031161 |

Geométrica-Decisión mejora su media inicial en los otros tres escenarios,
pero Local-Decisión tiene menor regret medio en ellos pese a recibir menos
información. No hay ganador universal ni evidencia de que añadir geometría
siempre ayude.

En deformada también cambia la comparación según la presencia de la verdad:

| Presencia | Escenas elegibles | Inyección-Decisión | Geométrica-Decisión | Extendida |
|---|---:|---:|---:|---:|
| En pool | 343 | .053224 | .016772 | .011936 |
| En vecinos | 41 | .199260 | .113991 | .030264 |
| Ausente | 105 | .080437 | .076604 | .094310 |

El slice ausente tiene 128 escenas en total, incluidas las 23 vacías. Su
regret compara con el mejor candidato disponible, no con recuperar una verdad
que falta del universo. Presencia es información privilegiada de evaluación,
no un gate operativo ni una explicación causal de la diferencia.

## Qué geometría se comprobó

Los probes cubren 16 escenas y 144 estados por escena: 2304 filas, no 2304
escenas independientes. No cambian las elecciones por transporte,
singleton/batch o roundtrip, ni el universo por identidad de evento en ese
corte. El máximo error de componentes por transporte es 2.384186e−7 y el
de energía por roundtrip, 1.835823e−5. Es estabilidad numérica; la escala global
ya estaba eliminada de la entrada. No demuestra invariancia física aprendida.

Todos los transportes satisfacen `atol=1e-6, rtol=1e-5`; estas tolerancias no
definen empates ni sustituyen la comparación exacta de elecciones. Los márgenes
originales y transportados están guardados: hay 144 márgenes originales cero
por escenario, por lo que no existe separación positiva universal. El máximo
cambio de margen es 3.874302e−7. No se observan colisiones de coordenadas en las
16 escenas. El roundtrip compara canales y energías sólo entre candidatos
correspondientes por identidad de evento; el máximo error de canal es
4.768372e−6. El resultado pertenece al efecto conjunto de cuantización y
recentrado, sin atribuir por separado sus contribuciones ni forzar un universo
idéntico cuando no exista.

El control desacoplado cambia el escalar en 28387/28417 candidatos IID,
28268/28333 beta, 29856/30246 polifonía y 27788/27859 deformada. Preserva
estratos y mantiene disponibles inputs alineados: no destruye toda la señal
geométrica. En estos datos, los no modificados son exactamente los candidatos
de estratos singleton: 30, 65, 390 y 71 por escenario y por backbone. El único
candidato de esos estratos queda sin cambio por construcción; todos los
no-singleton sí cambian en el corpus observado. Los totales se repiten en los
tres backbones, no son tres réplicas independientes. El detalle por escena y
checkpoint está en `summary.sham.scene_checkpoint_rows`, incluidos denominadores,
`scalar_changed_mask`, `six_channel_sham.donors` y soporte por estrato.

La geometría de este ciclo reside en el ajuste externo y su interfaz, no en
una nueva ley descubierta por la red. La mejora del selector no valida HIT,
identificabilidad física ni transferencia a medición real.

## Alcance de la verificación y evidencia

La [verificación técnica](PLAN_GEOMETRIC_DECISION_FINAL_AUDIT.md) terminó sin
discrepancias en su alcance predeclarado: autenticación de la cadena y sus
artefactos, 72 entrenamientos, 720 calibraciones, 144 estados y 2048 escenas;
reconstrucción matemática independiente en 27 originales y los 16 derivados
previstos; comprobación de los cuatro primarios sobre las matrices completas,
sus índices bootstrap y sus intervalos. Autenticar todo el corpus no equivale
a recalcular independientemente cada métrica de cada escena. No se repitieron
entrenamientos, forwards ni fitting para esta auditoría.

Los intentos anteriores del verificador y su contabilidad permanecen
conservados. La ejecución final consumió 614.107661 s en CPU, sin alterar el
corte fijado antes de la lectura de resultados ni reducir cobertura.

Raíz de artefactos: `data/atencion_armonica/geometric_decision_energy_v1/`.

| Artefacto | SHA-256 |
|---|---|
| `report/complete.json` | `c213d59be4f8d805efa6e150784bb8d391fb8387601d2a91dd81d7406a51cd49` |
| `report/deformed_family/summary.json` | `0c7a6c8d2aa75157a3b40ada775e202a62ebc6cab8b6fc2f28d638041865377f` |
| `audit-final-coverage-verify/result.json` | `1e51936b53709034f9ee47994dbd16825cb16884c10907309a90a46c19794e28` |
| `control/attempts/0025/finish.json` | `9b012d71375260e7bc8498bb5a65f6fc1e7581fa2fa0d703e32410f2ae95b103` |

`report/complete.json` enlaza primarios y resúmenes de los cuatro escenarios;
éstos enlazan probes y estratos. `summary.arm_scene_first` conserva el orden
de etapas, rutas y métricas; `summary.decision_minus_mse_by_route` contiene
las diferencias de loss. Los datos crudos no se sustituyen por estas tablas.

## Respuesta local y siguiente pregunta

La respuesta local es una ventaja relativa de la ruta geométrica en el
primario, sin mejora frente a su referencia inicial ni interacción acreditada
con la loss. Globalmente, esto justifica conservar el operador clásico como
adversario fuerte y abandonar la expectativa de que otra corrección aprendida
sea el siguiente paso obligatorio.

El siguiente contraste propuesto es **operador geométrico bajo medición**:
mantener fijos operador, candidatos y lectores en su definición, y comparar
entradas canónicas con render de audio y detección de picos. Debe separar
errores de observación, cobertura de alternativas y decisión; cuando cambien
los eventos detectados, no comparar particiones como si compartieran el mismo
soporte. Seguiría siendo evidencia sintética. El diseño deberá fijar si el
render usa frecuencias ideales o perturbadas y evitar duplicar el ruido.

**Ajuste geométrico amortizado** permanece como alternativa: aprender variables
u operaciones del ajuste en vez de corregir sólo su puntuación. Su factibilidad
y ventaja de coste no están demostradas. El diagnóstico de medición y los
costes observados deben orientar esa decisión, no una preferencia arquitectónica.
