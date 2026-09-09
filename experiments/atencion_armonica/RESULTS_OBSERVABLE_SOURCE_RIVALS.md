# Fuentes rivales: ajuste, cobertura y error de partición

2026-09-08. Diagnóstico retrospectivo de 96 escenas: búsqueda, evaluación y
replays completos. Auditorías finales de evidencia y alineación completadas;
la precisión documental sobre replicaciones quedó corregida.
No hay promoción arquitectónica ni decisión GO/NO-GO.

## Resultado

El diagnóstico distingue tres límites que no conviene reunir bajo la palabra
«ambigüedad». Algunas particiones verdaderas faltan en el universo candidato;
algunas rivales ajustan mejor la muestra ruidosa; y una familia física mal
especificada puede dejar un residual grande incluso sobre la partición
correcta. En estas escenas no aparece una explicación uniforme del error de
los lectores anteriores basada sólo en un margen pequeño entre fuentes.

El cambio más visible está en el escenario deformado: admitir la deformación
reduce el RMS mediano del mejor ajuste observable de 10.376 a 1.454 cents,
pero las particiones exactas sólo pasan de 18 a 19 entre las 23 escenas con
candidatos admisibles. Ajustar mejor la observación e identificar mejor su
partición son resultados diferentes. La escena restante conserva su ausencia
de candidatos y sigue en el denominador de 24.

## Qué se ejecutó

El [protocolo](PROTOCOL_OBSERVABLE_SOURCE_RIVALS.md) seleccionó 24 escenas de
cada uno de cuatro tests ya abiertos, sin nuevos datos, entrenamientos ni
forwards. La entrada fue el vector q32 de log-frecuencias centradas y la unión
de los pools observables de tres checkpoints. Se añadieron hasta 64 vecinos
de un paso, elegidos por una regla hash fijada antes del ajuste, sin etiquetas.

Se exigieron 2–4 fuentes, 4–8 miembros por grupo e índices parciales distintos
de 1 a 8. Una misma rama de parámetros gobierna toda la escena; debe existir
una escala global que sitúe todas las fundamentales entre 100 y 500 Hz.
La familia Base reúne beta baja/alta con gamma cero; Extendida incorpora una
rama deformada. Las mismas opciones se aplican en todos los escenarios, sin
dar al buscador el régimen verdadero. Extendida contiene Base y tiene mayor
flexibilidad: un menor costo no acredita por sí solo una mejor explicación.

El costo es `J = ||Pq32 − Pμ||²/(2σ²)`, con ruido de dos cents y proyección
de centrado P. No es una probabilidad calibrada, un posterior ni la likelihood
integrada de las celdas float32. Se preservan la cota de cuantización, las
grillas anidadas, todos los mínimos por asignación de índices y los factores
que reconstruyen los productos conjuntos evaluados. Las cotas inferior y
superior corresponden a la grilla discreta y tolerancia numérica declaradas,
no al óptimo continuo ni a una prueba de identificabilidad.

La búsqueda completa se selló antes de abrir las etiquetas. La partición
plantada se ajustó después con el mismo procedimiento como referencia
privilegiada; nunca se utilizó para completar el universo observable.

## Ajustes y separación en la grilla

«Plantada/rival» cuenta escenas donde las cotas numéricas separan el costo
mínimo de la partición plantada y el del mejor rival del universo observable,
con tolerancia J de 1e-7. No significa que la plantada esté disponible para
el buscador, ni establece separación estadística de distribuciones.

| Escenario | Familia | Con candidatos | Partición exacta | ARI medio | RMS mediano (cents) | Plantada/rival |
|---|---|---:|---:|---:|---:|---:|
| IID | Base | 24/24 | 21/24 | 0.967881 | 1.642177 | 21/3 |
| IID | Extendida | 24/24 | 21/24 | 0.967881 | 1.642177 | 21/3 |
| Mayor beta | Base | 24/24 | 22/24 | 0.944341 | 1.583027 | 24/0 |
| Mayor beta | Extendida | 24/24 | 22/24 | 0.944341 | 1.583027 | 24/0 |
| Mayor polifonía | Base | 24/24 | 18/24 | 0.943543 | 1.714244 | 23/1 |
| Mayor polifonía | Extendida | 24/24 | 18/24 | 0.943543 | 1.714244 | 23/1 |
| Familia deformada | Base | 23/24 | 18/23 | 0.908680 | 10.376163 | 19/4 |
| Familia deformada | Extendida | 23/24 | 19/23 | 0.915183 | 1.454075 | 21/2 |

ARI y RMS están condicionados a que haya candidatos; no se imputó éxito ni
fracaso de partición al caso sin salida. No hubo comparaciones indeterminadas
por las cotas en los casos con rivales. Son 24 escenas por escenario, no un
nuevo benchmark confirmatorio ni una estimación de prevalencia en audio real.

En Extendida, los rivales con menor costo corresponden a IID 391/398/472,
polifonía 181 y deformada 90/472. Sus diferencias encontradas respecto de la
plantada son −0.287, −1.923, −0.442, −1.198, −1.226 y −3.524 unidades de J.
La búsqueda no construye certificados de colisión noiseless: ese estado es
`NOT_TESTED`. Una muestra ruidosa mejor explicada por otra partición no prueba
no-identificabilidad estadística.

## Cobertura y límites del buscador

| Escenario | Plantada en pool original | Añadida por vecinos observables | Ausente del universo final |
|---|---:|---:|---:|
| IID | 22/24 | 2/24 | 0/24 |
| Mayor beta | 20/24 | 2/24 | 2/24 |
| Mayor polifonía | 17/24 | 1/24 | 6/24 |
| Familia deformada | 18/24 | 3/24 | 3/24 |

En mayor beta, la referencia plantada queda por debajo de todos los rivales
buscados en las 24 escenas, pero falta del universo observable en dos. En
polifonía faltan seis particiones plantadas. Un lector perfecto dentro de ese
universo seguiría sin poder elegirlas. La escena deformada 6 no tiene ninguna
partición que respete las cardinalidades, por lo que tampoco genera vecinos.
La ausencia de rivales adicionales fuera de esta búsqueda no está probada.

## Relación con los lectores anteriores

Las asociaciones promedian primero las nueve celdas checkpoint × semilla del
lector, por escena y por lector, y relacionan su error `1−ARI` con los márgenes. Se conservan los
siete lectores y los cuatro escenarios por separado. Para el margen
segunda−mejor de Extendida, los coeficientes Spearman abarcan −0.671 a −0.626
en IID, +0.126 a +0.421 en mayor beta, −0.376 a −0.294 en polifonía y +0.289
a +0.384 en deformada; n es 24, 24, 24 y 23. El signo cambia entre escenarios:
el margen no funciona aquí como una explicación uniforme del error neuronal.
Son asociaciones descriptivas, no causalidad ni probabilidades de error.

Este diagnóstico no constituye una comparación arquitectónica igualada con
los lectores anteriores. Cambian conjuntamente unión de pools, vecindad,
restricciones de cardinalidad, soporte físico y presupuesto de búsqueda.
Las buenas particiones encontradas justifican conservar un lector generativo
explícito como alternativa experimental; no permiten atribuir una ventaja
a la nueva familia, a la geometría o al solver por separado.

## Balance geométrico y próxima pregunta

La acción realizada tradujo una ley de fuentes en plantillas, restricciones
conjuntas, un costo y decisiones sobre particiones. No cambió una geometría
latente neuronal. El resultado cuestiona tanto «un buen fit identifica la
fuente» como «el error de la red demuestra que faltan observaciones».

El fitter comparte ecuación, rangos, índices, cardinalidades, f0 y ruido con
el generador. En particular, k=4 sólo admite la rama base de beta baja. Son
priors del sampler, no invariantes físicos descubiertos: la operación valida
una traducción bajo mundo conocido, no armonía natural ni la tesis de HIT.

La auditoría de alineación descartó como próximo contraste causal una
comparación directa de `argmin J` con cabezas preservadas: mezcla información,
flexibilidad paramétrica y objetivos de aprendizaje. El [diseño siguiente](PLAN_GENERATIVE_EVIDENCE_READER.md)
propone una sola ablación de evidencia generativa correcta, ausente o
desacoplada sobre cabeza, candidatos, priors y loss comunes, con Local como
baseline fuerte. El ajuste clásico queda como referencia de sistema. No se
predecide una arquitectura ni se justifica otra modalidad por estos errores.

## Reproducción

Raíz: `data/atencion_armonica/observable_source_rivals_v1/`. Los factores
guardados reproducen exactamente los fits y elecciones de las 96 escenas;
el replay privilegiado reproduce la evaluación completa, sin barridos ni
redes nuevas. La búsqueda tardó 44.760 s; replay observable, 29.650 s;
evaluación, 6.375 s; replay de evaluación, 5.055 s, según su ledger. Estos
tiempos no incluyen todo el arranque de procesos ni el trabajo de auditoría.

La [tabla por escena](RESULTS_OBSERVABLE_SOURCE_RIVALS_SCENES.csv) conserva las
192 filas de evaluación (96 escenas × dos familias), incluidos estados sin
salida y métricas de los lectores anteriores. Los campos vacíos representan
valores no definidos, no ceros.

| Artefacto | SHA256 |
|---|---|
| `campaign.json` | `a60213c6fb6fa3911bdc6c32733ab779b4b3895db5aaba519fe01ddae00c485c` |
| `observable_seal.json` | `86031fa51294aaca4bd83aa8be97b91cd8c4fa584f4375d75ca55b7817471a8c` |
| `evaluation.json` | `5808c3291105bf92bc93d59f78d829dcfbf8727949d078e1c282805e86f87c6f` |

La implementación del núcleo está en `e0aadc6`; campaña, evaluación y CLI,
en `18c094e`. Se preservan grillas, versiones, referencias de entrada,
semillas, IDs, parámetros, predicciones centradas, cotas, estados de soporte,
costos y métricas por escena, además de los replays y recursos separados.
