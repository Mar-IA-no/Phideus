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

## Cuarto corte: preparación completa y entrenamiento interrumpido

La auditoría de implementación cerró sin hallazgos materiales. Los tres
perfiles nuevos y la autorización de train/calibración terminaron bajo las
fuentes enmendadas. La regla de costo seleccionó CPU para las cabezas y
conservó los forwards congelados en 3090.

| Cantidad | Cuarto corte |
|---|---:|
| Perfil geométrico medido | 5,402 s |
| Perfil training CPU medido | 5,105 s |
| Perfil training 3090 medido | 6,211 s / 316 MiB VRAM |
| Celda CPU/GPU proyectada, incluida validación | 776,255 / 793,304 s |
| Forward por shard proyectado, con validación | 829,084 s |
| Score por shard proyectado, con validación | 1.846,373 s |
| Inferencia/evaluación test proyectadas | 985,172 / 1.052,739 s |

Estas proyecciones tienen margen: no son tiempos medidos de campaña. La
preparación completó 4.096 escenas de entrenamiento y 512 de calibración,
sus nueve forwards y bloques normalizados. El normalizador se ajustó sólo
con entrenamiento. El primer intento de entrenamiento terminó con error
después de 53,077 s; ese tiempo permanece en el presupuesto acumulado.

El control de suma de incidencias rechazó una partición válida de catorce
componentes: la suma `float32` de catorce pesos `1/14` acumula un error de
2,384186e-7, por encima de su tolerancia de 2e-7. La revisión de esa entrada
no encontró valores no finitos ni padding inválido. Se conserva el intento
fallido y su snapshot inicial; la corrección del control requiere una
enmienda trazable antes de reanudar. No hay entrenamiento completo, selección
ni test fresco autorizado. El objetivo conserva las 36 celdas y cuatro tests.

Referencias bajo `data/atencion_armonica/learned_partition_reader_v1/`:

- `profiles/geometry_04/manifest.json`: `6ddd86e6799dac51925de2a21e53430b6624de77f3d86df0306dcc5af5dcb5e4`.
- `profiles/training_cpu_04/manifest.json`: `9f1813d0aa71672ed25ad337e127136af8d92909f1b20d2d58f9414b4326c56d`.
- `profiles/training_gpu_04/manifest.json`: `8a1c92db57f3c6a990d5e6a673a9d08dad2f60d3bc00b267b8884a4218acf402`.
- `authorization/train_calibration_04.json`: `ca99fcf035e823d180fb54d23e0c8187fd9d2296d7a2614e44bfe59187b62614`.
- `train/shard_00/data/manifest.json`: `7de0474941bfd9f1084288f146c39f9a497821d800fd4e3a64faccfbd5ae01ed`.
- `train/shard_00/logits/manifest.json`: `be2c5b91bb9de1daad59c1aa5aed9e34d8bf9e1eef3c77cad48b87b36cf4c1ec`.
- `train/aggregate/manifest.json`: `4fe5bf20f35635bb0a945f155f2d093a7d77d39b4de9ce60081566fa38043dac`.
- `calibration/aggregate/manifest.json`: `c9e25e8807b9ff8e03a55245e47a2887377cf6908a4d801bca76128365c109b6`.
- `normalizers/manifest.json`: `5298840426a078e2c038df5edd0f21877b8344b80d9e657034743ef916c35285`.
- `supervision/supervisor-tvyvi5j0/terminal.json` y `stderr.log`: cierre y traceback del primer intento.

## Quinto corte: guard corregido y cohorte reutilizada

La [enmienda numérica](AMENDMENT_LEARNED_PARTITION_NUMERICAL_GUARD.md)
cambia sólo la acumulación del control de suma a `float64`. La aritmética
del modelo sigue en `float32`; protocolo, receta y presupuesto conservan
su identidad. La auditoría independiente verificó 112 pruebas CPU y cerró
un finding que exigía fijar también los hashes de los tests originales.

Los perfiles nuevos midieron 5,421 s de geometría, 5,116 s de training CPU
y 6,341 s de training GPU, con 316 MiB de VRAM reservada. Las proyecciones
por celda, incluida validación, fueron 777,507 s CPU y 796,171 s GPU.
La regla conservó CPU; la 3090 quedó libre después del perfil.

La importación explícita terminó en 31,863 s de supervisor. Preserva la
cohorte preparada: copia los payloads científicos y reescribe sólo los
cuatro índices de shards, además de manifests y recursos que distinguen
productor original y ejecutor actual. No repite draws, forwards, targets
ni ajuste del normalizador. El cierre independiente de esa copia sigue
siendo requisito antes de entrenar. El débito anterior de 53,077 s no se
reinicia. Todavía no hay una corrida completa, selección ni tests nuevos.

Referencias bajo el mismo árbol experimental:

- `profiles/geometry_05/manifest.json`: `8d91679bfabe77e1bf1a94bbdba69b5a17f618d7e8306707d1f0e91ddeba4c9c`.
- `profiles/training_cpu_05/manifest.json`: `cb6744b1118e0617f3fc51078bfcc36db78945497ec3bfb54e1bc44719f13157`.
- `profiles/training_gpu_05/manifest.json`: `3ddcddc5632ef5dfbc686bb8112d457c9f307794bdf11b933d270f021ef3ad8e`.
- `authorization/train_calibration_05.json`: `687350ef99beb1d3621fee6bd97092d408969a7e7f21a32486cc56cca0832811`.
- `reuse_05/manifest.json`: `3e9cfaf428a8bea67f3ddaff0c939cc5a0b7ce917739cd64d1b429674c76dc6a`.
- `supervision/supervisor-8ty8jh8v/terminal.json`: `3a30389a837076a8d162b0b58551ae903cb27c2d55f39f0545e546a64e08b087`.
