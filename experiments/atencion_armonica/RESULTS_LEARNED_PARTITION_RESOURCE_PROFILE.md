# Lector aprendido: perfiles de recursos antes del contraste

Fecha: 2026-09-08. Perfiles mecánicos CPU y GPU, no entrenamiento de campaña ni resultado de
generalización. El protocolo científico permanece sin cambios.

El primer perfil geométrico terminó en 4,211 s y preservó sus mediciones.
La proyección de validación por celda fue de 4.397,012 s, superior al límite
operativo de 600 s aun sin añadir entrenamiento. En ese corte se postergaron
los perfiles de entrenamiento CPU/GPU y no se autorizaron datos prospectivos.

| Cantidad | Primer perfil |
|---|---:|
| RSS máximo del perfil | 386.740.224 bytes |
| Preparación de pools por shard, proyectada | 1.065,344 s |
| Validación por celda, proyectada | 4.397,012 s |
| Artefactos preservados, proyección de disco | 38.102.749.184 bytes |

El término dominante fue la lectura y comprobación de pools/filas. Una de
sus repeticiones tardó 22,763 ms; las otras dos del mismo caso, 2,412 y
2,151 ms. La fórmula extrapolaba ese máximo aislado, escalado por tamaño, a
cada lectura. No se estableció la causa de la pausa ni se descartó la medición.

La corrección operativa auditada agrupa comprobaciones numéricas conservando
su semántica y mide tres bloques de 32 lecturas. El máximo por bloque dividido
por su denominador estima throughput de fixtures cacheadas, no latencia máxima
por archivo. Conserva el margen y los límites del supervisor. El perfil original
queda como evidencia del corte anterior; no se reescribe ni se elige el mejor
de corridas repetidas.

Fuente preservada: `data/atencion_armonica/learned_partition_reader_v1/profiles/geometry_01/report.json`.
Su manifest tiene SHA-256 `039ffa5f4bb7c3fa89acee6c6cee7ac0eb4ee14913e16360d99b223e8f640cb4`.
Este resultado identifica una dificultad operativa, no refuta ni confirma la
hipótesis del lector aprendido.

## Segundo corte medido

El código corregido completó geometría y ambos perfiles de entrenamiento.
Estos últimos se ejecutaron una vez para estimar un presupuesto viable,
no porque la proyección de validación hubiera entrado ya en 600 s.

| Perfil | Tiempo medido | Proyección de celda completa |
|---|---:|---:|
| Geometría CPU | 6,465 s | 694,791 s, sólo validación |
| Entrenamiento CPU | 5,083 s | 771,565 s |
| Entrenamiento 3090 | 6,256 s | 785,076 s |

La 3090 reservó como máximo 331.350.016 bytes (316 MiB) y quedó libre al
terminar. El criterio predeclarado de costo por celda favorece CPU en este
perfil de cabezas pequeñas; no es una comparación universal entre dispositivos.
Los forwards descriptor-guided permanecen como etapa GPU separada.

La autorización de datos fue rechazada antes de producir observaciones: la
identidad de runtime comparaba la versión de distribución `2.10.0` con la
versión completa del módulo `2.10.0+cu128`. La corrección conserva ambas
identidades, sin suprimir el sufijo de build. Independientemente de ese error,
las proyecciones exceden varios límites operativos: 771,565 s por celda,
1.890,168 s para score y más de 3.000 s para inferencia/evaluación de test.
No se incrementaron los límites ni se generaron datos.

La revisión técnica siguiente también elimina una asignación NumPy repetida
en la comprobación de soporte, conservando sus multiconjuntos y resultados.
No sustituye este perfil por una estimación editada: exige una medición nueva
bajo las fuentes corregidas antes de proponer el presupuesto pendiente.

Manifests preservados bajo `data/atencion_armonica/learned_partition_reader_v1/profiles/`:

- `geometry_02/manifest.json`: `9ca54731ceb5dda0d102195e85835c8378fcf9c7cf896924ea8b0967d1a9e9e1`.
- `training_cpu_02/manifest.json`: `84e48a4241595ebe92eeeac5241721213f816488061d4567f5d88af30ca743a4`.
- `training_gpu_02/manifest.json`: `6c94e48cba4e97e29b8e944fd4f4af5648a8c31f4e2ad44705e73343244cc074`.

Los tiempos proyectados llevan margen y no son walltime observado ni garantía.

## Tercer corte: correcciones verificadas y presupuesto insuficiente

La auditoría focal verificó la identidad completa de Torch y la equivalencia
del soporte. El tercer corte completó los tres perfiles sin reutilizar bindings
anteriores; la autorización rechazó ahora el exceso de presupuesto por celda,
después de validar perfiles y runtime. No se produjo autorización ni dato nuevo.

| Cantidad | Tercer corte |
|---|---:|
| Perfil geométrico medido | 5,455 s |
| Perfil training CPU medido | 5,105 s |
| Perfil training 3090 medido | 6,229 s / 316 MiB VRAM |
| Celda CPU proyectada, incluida validación | 801,598 s |
| Celda GPU proyectada, incluida validación | 825,492 s |
| 36 celdas CPU proyectadas | 28.857,517 s |
| Score por shard proyectado | 1.935,454 s |
| Inferencia/evaluación test proyectadas | 1.024,348 / 1.088,121 s |

El soporte mide ahora 0,085237 s por bloque de 32 casos. La regla predeclarada
continúa eligiendo CPU para las cabezas; no se reserva la 3090 durante ese
trabajo. Los forwards congelados conservan su etapa GPU. Los perfiles terminados
no demuestran tiempos de campaña y los topes siguen sin modificarse.

La continuidad requiere resolver el presupuesto operativo, no elegir una
arquitectura ganadora ni retirar controles. Las 36 corridas, cuatro brazos,
datos frescos y tests permanecen pendientes. La suite integrada conserva
98 pruebas CPU aprobadas; no equivale a evidencia de aprendizaje.

Manifests del tercer corte, bajo el mismo directorio de perfiles:

- `geometry_03/manifest.json`: `6df05efad031a36742d9be0e63549bb61bd64ab5a4bd545f10ee715a388152b9`.
- `training_cpu_03/manifest.json`: `655e3ee65987358b718bec9b30b9a426e1b48311172098682d8fcaca5fec5a15`.
- `training_gpu_03/manifest.json`: `ae5625fa660d637531456cdf3144f7ee4c2dc7b4f0cd98b74d814e72fe709193`.

## Enmienda de tiempos antes de datos

El [protocolo](PROTOCOL_LEARNED_PARTITION_READER.md) incorpora una ampliación
operativa aprobada después del tercer corte: 1.200 s por entrenamiento,
forward e inferencia, 2.400 s por score y 43.200 s de entrenamiento acumulado,
incluidos intentos fallidos. Los demás stages CPU conservan 1.200 s y los
perfiles 120 s. RAM, VRAM, muestras, semillas, controles y receta no cambian.
Las doce horas limitan entrenamiento acumulado; no reservan doce horas de GPU
ni describen la duración total del experimento.

La implementación conserva los checks internos y del supervisor, con un
medidor de score local para no ampliar productores históricos. La suite
actual reúne 100 pruebas CPU aprobadas. La nueva autorización requiere
perfiles coherentes con este protocolo y código, sin reutilizar los bindings
anteriores. La ampliación por sí sola no acredita aprendizaje ni generalización.
