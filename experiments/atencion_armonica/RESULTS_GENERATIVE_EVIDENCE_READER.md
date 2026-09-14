# Evidencia generativa: alineación útil, ventaja no resuelta frente a Local

2026-09-14. Contraste completo: 27 entrenamientos, cuatro evaluaciones y sus
replays; auditorías técnica y de alineación cerradas sin hallazgos materiales
abiertos dentro del alcance revisado.
No hay promoción arquitectónica ni decisión GO/NO-GO.

## Resultado y alcance

En familia deformada, el primario predeclarado, el canal generativo alineado
mejora ARI frente al canal desacoplado, pero no establece una ventaja clara
frente a Local con el intervalo predeclarado. Esto distingue valor de la
correspondencia entre evidencia y candidato de utilidad incremental frente a
descriptores locales fuertes. No demuestra equivalencia con Local ni que la
geometría sea innecesaria.

La referencia clásica Extendida alcanza mayor ARI en ese escenario que las
tres cabezas aprendidas. La operación explícita conserva por tanto interés
experimental, pero la comparación es entre sistemas con objetivos y acceso
efectivo diferentes, no una atribución causal de superioridad arquitectónica.
En polifonía, Generativa pierde frente a Desacoplada: el efecto no se puede
resumir como una ventaja OOD general.

## Qué se puso a prueba

El [protocolo](PROTOCOL_GENERATIVE_EVIDENCE_READER.md) fijó una misma cabeza
invariante de particiones, con 2194 parámetros y pérdida MSE de las entropías
normalizadas de separación y fusión. Se cruzaron tres checkpoints congelados
con tres inicializaciones del lector y tres brazos: Local, Generativa y
Desacoplada. Los 27 entrenamientos completaron 50 épocas; la selección sobre
calibración abierta eligió la época 50 para cada brazo, sin elegir semilla o
checkpoint ganador.

Todos comparten candidatos, metadatos de soporte, inicialización pareada y
receta. Generativa recibe seis cotas de ajuste conjunto, Local seis ceros y
Desacoplada los vectores generativos permutados conjuntamente dentro de
estratos de tamaños. Un canal activo no garantiza la misma capacidad efectiva
que uno nulo; el sham conserva el multiset conjunto del canal dentro de estratos
observables, sin eliminar todas sus correlaciones con el target. Ninguno de
los dos controles elimina por sí solo todas las explicaciones alternativas.

Se produjeron 512 escenas nuevas por escenario y se sellaron 45 salidas antes
de abrir su supervisión: 27 originales y dos intervenciones sobre cada una de
las nueve cabezas Generativa. Los fallos de integración y la pausa recuperable
se conservan en el [estado operativo](STATUS_GENERATIVE_EVIDENCE_READER.md).
La enmienda JSON ocurrió después del primer draw y no se presenta como una
ejecución inalterada del código inicial.
Los draws son nuevos, pero pertenecen a la misma ontología generativa; no
aportan una ley independiente ni autoridad física externa.

## ARI por escenario

Se promedian primero las nueve celdas dentro de cada escena y después las
escenas con salida común. No se ensamblan logits ni se cuentan nueve celdas
como nueve observaciones independientes. Los intervalos son bootstrap pareado
de 2000 réplicas, condicionado a checkpoints e inicializaciones entrenados.
En los dos contrastes ARI de familia deformada son nominalmente 97.5%; en
los otros escenarios, 95% descriptivos.

| Escenario | Con salida / 512 | Local | Generativa | Desacoplada | Generativa−Local [IC] | Generativa−Desacoplada [IC] |
|---|---:|---:|---:|---:|---|---|
| IID | 505 | 0.969636 | 0.972077 | 0.970569 | +0.002442 [0.000913, 0.004022] | +0.001508 [0.000458, 0.002736] |
| Mayor beta | 497 | 0.846677 | 0.850673 | 0.837040 | +0.003996 [−0.000741, 0.008394] | +0.013633 [0.008755, 0.018706] |
| Mayor polifonía | 451 | 0.925779 | 0.924814 | 0.931201 | −0.000966 [−0.002914, 0.000983] | −0.006387 [−0.008496, −0.004446] |
| Familia deformada — primario | 484 | 0.783519 | 0.786475 | 0.779618 | +0.002957 [−0.000668, 0.007134] | +0.006858 [0.002706, 0.011171] |

Por celda, el contraste ARI del primario es positivo en 9/9 para
Generativa−Desacoplada y en 6/9 para Generativa−Local. En polifonía,
Generativa−Desacoplada es negativo en 9/9, mientras Generativa−Local es positivo
en 4/9. Estos signos describen consistencia entre celdas, no nueve réplicas
independientes ni un segundo criterio de significación.

Las escenas sin salida conservan su lugar en el denominador: 7, 15, 61 y 28,
respectivamente. No se les imputa ARI cero. La cobertura condiciona el alcance
de las medias; no es una métrica de identidad correcta.

Otras métricas impiden reducir la lectura al ARI favorable. En el primario,
la exactitud media de partición es 0.480946/0.492883/0.471534 para
Local/Generativa/Desacoplada. Generativa mejora descriptivamente esa exactitud
frente a Local, pero aumenta su error absoluto de cantidad de fuentes:
0.082415 frente a 0.074151, diferencia +0.008264 e IC95% [0.003214, 0.013545].
VI es 0.333982/0.330606/0.340380; su contraste Generativa−Local no establece
una ventaja clara: −0.003375 [−0.008279, 0.001451]. La masa en grupos menores
de tres es cero por la máscara de candidatos, no por un aprendizaje demostrado
de ausencia de fragmentación. Las trece métricas se conservan en los artefactos
de evaluación y en la [tabla completa](RESULTS_GENERATIVE_EVIDENCE_METRICS.csv);
no se reemplaza el primario por una secundaria favorable.

## Universo candidato y referencias de sistema

| Escenario | Plantada en pool | Añadida por vecinos | Plantada ausente | Oracle ARI | Base ARI | Extendida ARI |
|---|---:|---:|---:|---:|---:|---:|
| IID | 480 | 14 | 18 | 0.993412 | 0.975222 | 0.974331 |
| Mayor beta | 436 | 16 | 60 | 0.975359 | 0.945402 | 0.945402 |
| Mayor polifonía | 310 | 22 | 180 | 0.952886 | 0.926892 | 0.926892 |
| Familia deformada | 351 | 34 | 127 | 0.931162 | 0.841835 | 0.883872 |

Las tres columnas de presencia suman 512 por escenario; no equivalen a las
escenas sin ninguna salida. Oracle, Base y Extendida se promedian sobre
el soporte común con candidatos. El oracle usa etiquetas para seleccionar
el mejor candidato: mide una posibilidad privilegiada, no aprendibilidad.

Base y Extendida eligen por mínimo de la cota superior de ajuste, sin cabeza
aprendida. Extendida añade una rama deformada explícitamente conocida. Su
ventaja descriptiva sobre Generativa en mayor beta y familia deformada sugiere
examinar qué se pierde entre operación, representación y aprendizaje; no
identifica cuál de esas mediaciones es responsable. Histórico conserva sus
tres checkpoints y se evalúa tanto sobre soporte común como sobre las 512
escenas; tampoco es un brazo de capacidad igualada.

## Intervenciones con cabeza fija y soporte del sham

Estas intervenciones no reentrenan ni sustituyen a los brazos originales.
Retiran o desacoplan el canal en las mismas nueve cabezas Generativa.

| Escenario | Canal original: ARI | Canal cero: ARI | Canal desacoplado: ARI |
|---|---:|---:|---:|
| IID | 0.972077 | 0.971066 | 0.968788 |
| Mayor beta | 0.850673 | 0.843079 | 0.836098 |
| Mayor polifonía | 0.924814 | 0.931204 | 0.930848 |
| Familia deformada | 0.786475 | 0.781582 | 0.779899 |

Son medias descriptivas, sin nuevos intervalos de intervención. En polifonía,
retirar el canal mejora la media de las mismas cabezas; en los otros tres
escenarios la reduce. El canal afecta la decisión, pero su utilidad no es
uniforme. Los diagnósticos crudos preservan cambios de componentes, costos y
elecciones, distinguiendo desplazamientos constantes de cambios de ranking.

Al poner el canal en cero cambian 43/4545, 636/4473, 320/4059 y 232/4356
elecciones; al desacoplarlo, 86/4545, 716/4473, 409/4059 y 301/4356, en el
mismo orden de escenarios. Los denominadores son celdas por escenas con
salida, no observaciones independientes. En todas esas unidades cambia de
forma no constante la suma de costos entre candidatos: alterar costos no
implica necesariamente alterar el mínimo elegido.

En los cuatro tests todas las escenas con salida tienen al menos un vector
generativo alterado por el sham: 505, 497, 451 y 484. Eso no implica que todos
los candidatos o estratos cambien: se preservan los singleton, donantes,
máscaras de cambio efectivo y vectores sin cambio. El subconjunto de escenas
con salida totalmente intacta está vacío y no habilita estimar allí un efecto.
Por candidato, cambian 27848/27891, 27916/27981, 29267/29620 y 27606/27675
vectores; los 43, 65, 353 y 69 estratos singleton explican los restantes.

## Balance geométrico y límites

El operador traduce razones módulo escala global y realizabilidad conjunta
en un ajuste computable de particiones. La cabeza aprende a estimar errores
de partición respecto de la plantada, no a minimizar el residual de la ley. Geometría, representación
y loss están relacionadas, pero no son cantidades intercambiables.

La campaña aporta evidencia sobre el uso de esa operación bajo una interfaz
común. No incorpora una nueva operación geométrica aprendida al backbone.
La ley, el ruido, los índices 1–8, las cardinalidades, las ramas y los rangos
de fundamentales son priors del banco sintético. La deformación ya pertenece
al fitter; no se descubre una ley ni se valida transferencia a audio observado.
Los errores no prueban insuficiencia de la frecuencia ni necesidad de agregar
tiempo o fase. Tampoco una mejora del sistema valida HIT.

La pregunta local obtiene una respuesta mixta: alinear el canal aporta frente
al desacople en el primario, sin establecer allí una ventaja clara de ARI
frente a Local. Globalmente, el operador explícito sigue siendo pertinente,
pero su utilidad como referencia no se traslada automáticamente a esta cabeza
y esta pérdida. No se ha probado cuál de esas mediaciones explica la brecha.

El [siguiente diseño](PLAN_OPERATOR_OBJECTIVE_ALIGNMENT.md) propone comparar
el orden geométrico con el target verdadero, el error de cada componente de
entropía y el ranking producido por su suma. Separará además cobertura y
desacuerdo entre oracles de ARI y de target. Es un diagnóstico retrospectivo
CPU sobre artefactos preservados, no otro entrenamiento adaptado a estos tests.
Su resultado debe justificar un experimento discriminante o explicitar qué
sigue indeterminado. La campaña cierra su pregunta local y deja ese relevo
justificado por evidencia; no hay promoción arquitectónica.

## Evidencia recuperable

Raíz: `data/atencion_armonica/generative_evidence_reader_v1/`. La selección
y los estados de entrenamiento están ligados al freeze; cada test preserva
draws, inputs, factores, predicciones por candidato, métricas por escena y
bootstrap. Los siguientes hashes identifican las evaluaciones leídas para
el informe; no sustituyen la auditoría desde los outputs originales.

| Escenario | SHA256 de `fresh/<split>/evaluation/summary.json` |
|---|---|
| IID | `3a6345420b8485f486c66f61336a7c68c7b0904fcf09822f52f8e3e1128be053` |
| Mayor beta | `a721f1b4c3e18771bb258722562ba37165b20b70acfa3d5decd8e2c06fcb265e` |
| Mayor polifonía | `f265c2d81982b2c40afe04208375e8ef351cca24c18bd112426faacfd20249b5` |
| Familia deformada | `09535ff5fc208255ec354065ccbea1b1a4534c667e211baf1f99c5ea51c277c0` |

El cierre `fresh/test_completion.json`, SHA256
`160473a7c88ab9e9a296a9e37a3145546a6ccc7019dbec473394ccd966cae3f9`,
liga las trece etapas: freeze y predicción/evaluación/replay de cada test.
El supervisor terminó con salida 0 y 7101.919 segundos acumulados, incluidos
los intentos fallidos y la pausa recuperable; el límite era de cuatro horas.
El replay del primario conservó el índice y el resumen de su evaluación.

El [exportador](export_generative_evidence_report.py) ya copió los resúmenes
autenticados después del cierre. No recalcula estadísticas ni confiere por sí
solo autoridad científica. El CSV conserva 676 filas de las trece métricas,
brazos, contrastes y referencias con su soporte, fuente y hash. La copia
publicable sólo normaliza finales de línea; sus 676 registros coinciden
exactamente con la exportación. El JSON local conserva además celdas,
intervenciones y máscaras del sham; no se presenta el CSV como reemplazo de
todos los artefactos crudos.

La auditoría técnica independiente autenticó 10837 archivos y recalculó
selección, normalizadores, decisiones, trece métricas, intervalos, sham,
intervenciones, referencias y oracles. No encontró discrepancias materiales;
84 pruebas focales y seis del exportador pasaron. Su alcance no incluye
repetir entrenamiento o forwards, recalcular las cotas del fitter ni rederivar
las elecciones clásicas desde esas cotas y logits: autentica las elecciones
selladas y comprueba su evaluación. La auditoría independiente de alineación
leyó el informe y el diseño sucesor completos, contrastó su propagación y
cerró tres inconsistencias de estado documental. No quedan hallazgos
materiales abiertos; ese cierre no amplía la autoridad física del experimento.
