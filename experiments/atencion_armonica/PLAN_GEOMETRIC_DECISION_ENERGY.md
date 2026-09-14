# Energía geométrica para la decisión

2026-09-14. Plan inicial del siguiente goal finito. Diseño autorizado bajo el
programa de investigación-acción; protocolo ejecutable y auditoría pendientes.
No autoriza todavía correr una campaña con fórmulas o controles sin resolver.

## Pregunta

¿Conservar una operación de ajuste geométrico explícita en la energía que
decide la partición aporta algo distinguible de cambiar la función de pérdida?

El [diagnóstico cerrado](RESULTS_OPERATOR_OBJECTIVE_ALIGNMENT.md) mostró que
mejorar el orden global de candidatos no garantiza preservar el mínimo. En el
primario deformado, la regla clásica elige mejor que las cabezas aunque tenga
menor tau. Los oracles VI/ARI, la cobertura y los estratos delimitan explicaciones,
pero no atribuyen el error a representación, optimización o loss.

## Contraste y modelo relacional

La observación induce un universo de particiones y un ajuste conjunto por
ramas de ley. La candidata mantiene esa operación en la energía final y aprende
una corrección, en lugar de exigir que una cabeza reconstruya su uso solamente
desde features. Se cruza esa intervención con el objetivo de aprendizaje.
La geometría sigue siendo una relación explícita bajo ley sintética conocida;
no se supone descubierta ni aprendida en el backbone.

El diseño ejecutable debe permitir estos contrastes, factorialmente o mediante
brazos puente equivalentes:

| Operación/interfaz | MSE de componentes | Loss sensible al mínimo/regret |
|---|---|---|
| Inyección convencional de evidencia | Control | Efecto de loss sin nueva operación |
| Vía geométrica explícita en energía | Efecto de operación | Combinación e interacción |
| Vía desacoplada del candidato | Control de correspondencia | Correspondencia bajo otra loss |

Local/descriptores sin canal generativo permanece como control aprendido y
Extendida como referencia clásica, no brazo de capacidad igualada. La vía
desacoplada debe conservar acceso, distribución y capacidad relevantes sin
convertirse en una tarea diferente. No se presupone que anule toda señal.

Mantener comunes observaciones, proposer/pool, backbones, capacidad relevante,
inicializaciones, receta, selección y presupuesto. Si la vía requiere cambiar
la salida de la cabeza, ese cambio debe ser común al control o tener un brazo
puente; no atribuir al operador una modificación simultánea sin control.
Conservar el baseline anterior como evidencia histórica, no fingir igualdad
entre una receta nueva y los resultados previos.

## Diseño matemático e investigación puntual

Releer las fuentes locales de cotas, features, normalizadores y loss. La
consulta dirigida de pérdidas estructuradas debe distinguir energía de
inferencia, pérdida de tarea y surrogate. El margen estructurado es un
antecedente, no una garantía transferible a redes no convexas ni una teoría
física ([Tsochantaridis et al., §2](https://www.jmlr.org/papers/volume6/tsochantaridis05a/tsochantaridis05a.pdf)).
El protocolo debe especificar:

- operación de ramas disponibles, unidades y escala; normalización sólo TRAIN;
- corrección nula y relación con la elección clásica, con empates y precisión;
- loss de decisión que trate coóptimos y no sólo una etiqueta canónica;
- contraste con MSE sin imponer que dos salidas bajo otra loss sean entropías;
- estados vacíos, máscaras, degeneraciones, invariancias y soporte;
- estimandos de operación, loss e interacción, sin optimizar tau por inercia.

Una forma candidata de surrogate usa el regret verdadero dentro del pool y
la energía media del conjunto coóptimo como referencia de margen; debe probarse
su cota de regret y explicitar las restricciones adicionales que introduce.
No se congela aquí la fórmula final ni se sustituye el diseño por una analogía.

## Ejecución y evidencia

Reutilizar TRAIN/calibración y estados autorizados cuando mantengan el contrato,
sin refits/forwards innecesarios. Los cuatro tests abiertos del diagnóstico
anterior sólo motivan hipótesis: no ajustan parámetros, selección o umbrales.
El protocolo fija roster, escenarios, métricas primarias/secundarias, unidades,
soporte e incertidumbre antes de generar y evaluar escenas nuevas. La selección
usa sólo TRAIN/calibración y no escoge checkpoint o semilla ganadores.

Evaluar mínimo/regret y calidad de partición, además de pruebas de relaciones
y transformaciones geométricas dentro de las guardas correctas. Distinguir
transformar la observación de transportar un universo candidato ya fijado;
permutación, escala global y cuantización no pueden mezclarse silenciosamente.
Separar propiedad impuesta por construcción, preservada numéricamente y
aprendida. Una invariancia exacta no valida por sí sola la geometría del fenómeno.

Implementación en módulos nuevos y puertos reutilizables, sin modificar fuentes
congeladas. Conservar checkpoints inicial/last_epoch/intermedios pertinentes,
outputs por escena/candidato/celda, configuraciones, semillas, splits y replay.
Auditoría independiente de plan y mecanismos antes de campaña; cortes técnicos
por riesgo y auditoría final de horizonte, sin iteraciones cosméticas.

GPU local habilitada por el usuario. Informar alcance, duración y VRAM antes
del uso y comprobar ownership; no pedir autorización por corrida. Perfilar
cabeza y fitting por separado, elegir CPU sólo cuando sea proporcionada y
fijar un presupuesto acumulado con reserva de auditoría antes del roster.
No usar una corrida CPU muy larga como sustitución de GPU ni infraestructura
remota por rutina. Recuperaciones conservan muestras, estados y consumo.

## Cierre y bifurcaciones

El goal exige protocolo auditado, implementación, contraste completo, replay,
informe, auditorías técnica y de alineación, docs/wiki y commit/push. No termina
con un preflight. Una respuesta negativa o mixta también reduce incertidumbre;
un impedimento operativo mantiene explícitamente el objetivo incompleto.

Una mejora común de la nueva loss en todos los controles no basta para atribuir
un aporte geométrico. Una ventaja de la operación alineada sobre controles
comparables, o su interacción reproducible con la loss, sostendría esa vía
concreta. Un resultado adverso debe permitir cambiarla; no elimina el horizonte
geométrico ni demuestra insuficiencia de frecuencia. La siguiente pregunta
dependerá de evidencia, no de una arquitectura preferida de antemano.
Promoción y GO/NO-GO corresponden al usuario.
