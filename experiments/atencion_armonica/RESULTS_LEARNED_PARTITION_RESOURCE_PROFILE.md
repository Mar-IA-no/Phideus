# Lector aprendido: perfiles de recursos antes del contraste

Fecha: 2026-09-08. Perfiles mecánicos CPU/GPU y continuidad operativa del contraste.
Los perfiles no acreditan aprendizaje; las corridas reales se registran por separado.
El protocolo científico permanece sin cambios.

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

La auditoría posterior comprobó integralmente la copia: 60.098 payloads
científicos con SHA idéntico, cuatro índices reescritos, sin hardlinks ni
symlinks. Antes de entrenar se detectó otro error de composición: la
recuperación consultaba el contador global desde una reserva aún activa y
por ello rechazaba su propio intento. Se reprodujo en fixtures, sin lanzar
training 05 ni añadir otro débito real.

La corrección auditada mantiene el contador estricto en el supervisor y
verifica localmente el asiento original desde su terminal inmutable. Un
proceso hijo real atraviesa permiso y recuperación dentro de una reserva
activa; el contador global sigue rechazando correctamente ese registro
abierto. La suite conserva 112 pruebas CPU aprobadas. El corte 06 usa una
identidad y destino nuevos, con el mismo productor 04; no reetiqueta la
copia 05 ni cambia guard, modelo, protocolo o presupuesto.

## Sexto corte: bootstrap corregido, perfiles e importación completos

Los perfiles midieron 5,407 s de geometría, 5,068 s de training CPU y
6,163 s de training GPU, con 316 MiB de VRAM. La proyección por celda
fue 769,650 s CPU frente a 791,798 s GPU; la autorización seleccionó CPU.
La copia nueva terminó en 32,076 s de supervisor. El cierre independiente
de sus bytes precede al primer request de training de este corte.

Referencias bajo el mismo árbol experimental:

- `profiles/geometry_06/manifest.json`: `28e4d1402518bcbea1493452bbf0bb7bf523a3437338fee60a92176df718fdfb`.
- `profiles/training_cpu_06/manifest.json`: `b7dbe9e95eea413d36269bc2fe43d13a60efccf3b95cada1d7f657b369bff6a4`.
- `profiles/training_gpu_06/manifest.json`: `c6e33d402ae7a3c70f05973d20f3362df73cf193c742a5c8b1356a3cc7e955a6`.
- `authorization/train_calibration_06.json`: `7740d564c748389f47ce7a818675ed5331e81ae4e0c7e50e6576a2e8df8568b5`.
- `reuse_06/manifest.json`: `f09c96b700accb834bb99233f9b01278e3b30d99d4fc3e5ea3ae74c5b9813ceb`.
- `supervision/supervisor-37y4009a/terminal.json`: `c652fd06e1adc0707081fca910871d2c56b8ac5ba3ad740b99dd3324a27ebbda`.

La revisión integral cerró la copia 06 y el entrenamiento se reanudó. La
primera celda, Pares/estructura con checkpoint `2026090721` y lector
`2026090891`, completó 50 épocas y 6.400 updates en 139,198 s de supervisor,
con diez bloques de calibración preservados. Su RSS pico fue 1.989.263.360
bytes, sin GPU. El débito previo permanece separado y acumulado. Es una
corrida terminada, no el contraste completo ni evidencia de generalización.
La campaña continúa; selección y tests siguen sin abrirse.

- `cells_06/pairs_structure__checkpoint_2026090721__reader_2026090891/manifest.json`:
  `1188650701541cf6d180f7f2447e1049ecd825bae6364bde1d7342f58d5fb535`.
- `supervision/supervisor-j3d1hp15/terminal.json`:
  `80c98cf65e88f6ce41fdb8979db7de5b0e48b78152412bcb186776608abc5f50`.

El corte siguiente completó las nueve corridas de Pares/estructura y las
nueve de Compatibilidad local: 18/36 celdas, con 2.455,712 s de entrenamiento
supervisado completo, además del débito anterior. La comprobación de ambos
bloques verificó sus bundles y preservación de 198 snapshots y 180 bloques
de calibración. Fuente compartida comenzó a ejecutarse. Estas cantidades
proceden de los manifests bajo `cells_06/` y sus terminales ligados, no de
proyecciones de recursos. No hay selección ni tests del lector aprendido aún.

El entrenamiento terminó después sus 36 celdas, todas con 50 épocas y
6.400 updates, sin nuevos fallos en este corte. Sus terminales suman
4.901,189 s; el contador acumulativo registra 4.954,266 s al incluir el
intento original fallido. La verificación de los bundles completos encontró
396 snapshots y 360 calibraciones. El registro tiene 37 intentos cerrados,
36 completos y uno fallido; la celda de mayor consumo acumulado suma
192,276 s, dentro del límite operativo. Son tiempos supervisados, no
benchmarks de kernels ni una comparación de calidad entre los brazos.
La selección se ejecuta sólo después de este cierre y exige su propia
auditoría antes de abrir tests.

Fuentes: manifests y cadenas bajo `cells_06/`, terminales enlazados por sus
requests y contador de `learned_partition_budget.accounting()`. El manifest
de la última celda, `decoupled_source__checkpoint_2026090723__reader_2026090893`,
tiene SHA `e7d4e81b41f5466c327d1804502ea14c682fc301842d023a3ed9f5fc06e35553`;
su terminal `supervision/supervisor-sj1db8qo/terminal.json` conserva SHA
`1b1419f41c41b5545c221057ee6860780926c95320e2297080f8b2bc0428cb4a`.

El freeze terminó en 62,702 s de supervisor y seleccionó época40 para
Pares/estructura, Fuente compartida y Fuente desacoplada, y época50 para
Compatibilidad local. Son decisiones de calibración, no comparaciones de test.
Su auditoría independiente permanece pendiente en este hito.

- `selection/freeze_06.json`: `08bef072cfce37b3cda30be7e0278d5452f460cd194a265e783144cdcf9644cc`.
- `supervision/supervisor-_mbh336q/terminal.json`: `516a0895c64d5553e1cf14436f2fbfb12b70d4cb36e65f10af7e7f7e777046f8`.

## Selección auditada y evaluación interrumpida por memoria

La revisión independiente reconstruyó la selección desde las 360 calibraciones,
sin discrepancias, y verificó las 36 cadenas de entrenamiento completas. La
autorización de test se produjo después de ese cierre. El conjunto IID completó
observaciones, agregación, forwards congelados y scoring. El forward reservó
184 MiB de VRAM; no se repitieron entrenamientos ni se cambió la selección.

La normalización IID se detuvo a los 117,406 s al alcanzar 2.147.901.440 bytes
de RSS, sobre el límite de 2 GiB. El supervisor confirmó la terminación del
trabajador y conservó el intento incompleto. Sus doce NPZ no constituyen un
bundle válido: no hay manifest completo ni predicciones o métricas de test.
Los otros tres conjuntos todavía no se generaron. El siguiente paso es una
recuperación de memoria con procedencia explícita, sin cambiar muestras,
pesos, transformaciones ni límites; no es una conclusión sobre generalización.

Referencias bajo el mismo árbol experimental:

- `authorization/test_06.json`: `d48e5f0b758965dd673f2d152501de2f885fb9be8b5f4087ae0dd9c76900da36`.
- `iid/shard_00/logits/manifest.json`: `a30b096e8872da11d18378b570302b544634d9f9a3d14c38f88b9e5ee003352d`.
- `iid/shard_00/scored/manifest.json`: `91bd8f93e879e4b00d774a42d6298af9d08960976e49d5ad5bf16bbac6a2ec62`.
- `supervision/supervisor-vi6ekkdu/terminal.json`: `95bd60679810638bbbb941334d06c05a1091becc28a2db650cc2dd95dc318fbc`.

## Normalización recuperada con ejecutor versionado

La recuperación libera objetos ya serializados antes de volver a validar las
entradas. Conserva las operaciones, dtypes, muestras, pesos, selección y límites;
los nuevos productores declaran su propia identidad y consumen la base original
sin reasignarle código nuevo. La revisión independiente de implementación cerró
los controles de rutas y de igualdad obligatoria entre evaluación y replay.
La [suite focal](test_partition_test_recovery.py) pasó sus nueve pruebas CPU.

La única normalización IID ejecutada con ese contrato terminó en 173,402 s,
con pico supervisado de 1.758.867.456 bytes RSS y sin GPU. El proceso hijo
terminó confirmado con código0. Sus doce archivos normalizados coinciden
exactamente en arrays y dtypes con los candidatos conservados del intento
fallido; éste permanece incompleto y no fue reemplazado. La salida nueva está
sellada. Su auditoría de ejecución precede a la inferencia: todavía no hay
predicciones ni métricas de test, ni se generaron los otros tres conjuntos.

Referencias bajo `test_memory_recovery_v1/` del mismo árbol experimental:

- `contracts/contract_02.json`: `75242f36a2b2b29610e8944f55eb5b1eba9ccf141e90e5930f0a059bc711de7b`.
- `authorization/recovery_02.json`: `fb6c9229329a158a5b43d2622a017bdc53314b3d1a047475f860ae1334a9c7a3`.
- `outputs/iid/normalized_01/manifest.json`: `8cfbc1a498e87c26f1fd7705d83459455ffbae48e2ecf2277f241f2c265e20d1`.
- `supervision/supervisor-x89ef7p1/terminal.json`: `6629c7442608ff82eb137765a9067f4f70526faa6d518035c67ac3c8fab7aeef`.

## Inferencia, evaluación y replay IID exclusivamente CPU

La auditoría del normalizado cerró sin hallazgos materiales. Bajo la suspensión
posterior de GPU, tres requests separados completaron únicamente IID sobre
los logits ya conservados; no se ejecutó el operador que continúa a CUDA.

| Etapa | Tiempo supervisado (s) | Pico RSS (bytes) | GPU |
|---|---:|---:|---:|
| Inferencia | 184.798 | 1.761.214.464 | 0 |
| Evaluación | 200.004 | 2.053.509.120 | 0 |
| Replay | 200.813 | 2.036.256.768 | 0 |

Los tres hijos terminaron confirmados con código0. La inferencia conservó
99 NPZ, 63 diagnósticos y su índice. Evaluación y replay reprodujeron los
ocho payloads byte-exactos. La evaluación quedó sólo 93.974.528 bytes por
debajo de su cap de 2 GiB; ese margen no se extrapola a los tests restantes.
El [balance IID parcial](RESULTS_LEARNED_PARTITION_IID.md) separa las métricas
de este cierre operativo. Los otros tres tests y el primario siguen pendientes.

Terminales bajo `test_memory_recovery_v1/supervision/`:

- `supervisor-w8aywgob/terminal.json`: `b4103fd5ed7215d8460430fc18b86f6dd7887d186f7e40d4a4086268f47f25a7`.
- `supervisor-aefx528l/terminal.json`: `64ecdb9964784a4ae789812f1fa66dd00147e5379281c6dd0af796137084cfea`.
- `supervisor-3xybvh68/terminal.json`: `3e432fe514c690c84d1cfb73097b0c2db9c1c9a538f6441c8515db84d5b7f9b5`.

## Continuación fuera de distribución

La secuencia CPU/GPU se reanudó con IID reutilizado y sin cambiar selección
ni receta. Mayor inarmonicidad completó preparación, agregado y forwards;
después completó puntuación, inferencia, evaluación y replay. Los ocho payloads
científicos coinciden exactamente; la auditoría global sigue pendiente.
El forward terminó confirmado con código0 en 128.005 s de supervisor,
con pico GPU observado de 432.013.312 bytes; esta lectura del supervisor
no se confunde con memoria reservada reportada por PyTorch.

Terminal bajo `data/atencion_armonica/learned_partition_reader_v1/`:
`supervision/supervisor-zn5wpni7/terminal.json`, SHA-256
`347ef3dab726ab9c5e88f1672cd12cae035a8eb70b9de79e021a90f974b8b8fb`.

Mayor polifonía completó los forwards y la normalización. Su inferencia se
detuvo al construir el soporte de una intervención: la comprobación de masa
sumó pesos válidos en float32 y produjo un falso rechazo por redondeo, incluso
comparando la entrada consigo misma. No fallaron GPU ni límite de memoria.
Se conservan once NPZ parciales y el marcador de fallo, sin declararlos
COMPLETE. No hay todavía evaluación de polifonía ni datos de familia deformada.

La recuperación propuesta limita el cambio a acumular esa validación en
float64, con el mismo `atol=2e-7`; no modifica pesos ni cómputo del modelo.
La ruta separada pasó su auditoría de implementación previa a ejecución; las dieciocho
pruebas mecánicas pasaron, incluida la igualdad estructural de los recorridos
científicos salvo el delta declarado. La reconstrucción del soporte IID/beta
coincidió exactamente en los 63 reportes de cada test, con 512 escenas por
reporte, sin nuevos modelos ni etiquetas. El cierre del registro final verifica
también los payloads de predicciones y los agregados encadenados de datos.
Terminal del intento fallido bajo
`test_memory_recovery_v1/supervision/supervisor-zqijy7bc/terminal.json`, SHA-256
`fed094a5a614dad6445e4672917effe075b30a502354620ea0a05a34685d3f40`.

La inferencia recuperada de polifonía completó las 99 predicciones y 63 reportes
de soporte en 200.236 s, con pico RSS observado de 1.947.774.976 bytes y sin GPU.
Los once NPZ previos coincidieron en bytes y arrays antes del seal. La evaluación
posterior se detuvo por RAM; no hay aún resultado del primario ni cierre del contraste.
Terminal bajo `support_validation_recovery_v1/supervision/supervisor-35aeny1c/terminal.json`,
SHA-256 `bb63bb09731190ecf6d4d4801c72addc5ba5c6839dc370f11948888beb7df72d`;
manifest bajo `support_validation_recovery_v1/outputs/ood_polyphony/predictions_01/manifest.json`,
SHA-256 `d8e48e2c0250d5248e2e93a344fa2f953a5c2c31f94ec78aad676a9a05ce8356`.

La evaluación posterior llegó a 2.151.206.912 bytes de RSS frente al límite de
2.147.483.648, a los 131.021 s. El supervisor confirmó su terminación y preservó
seis payloads científicos parciales más el marcador de fallo, sin soporte ni
manifest de finalización. Es un exceso de memoria, no de tiempo o GPU. El
diagnóstico revisa estructuras ya serializadas que siguen vivas durante el
soporte; no cambia el límite ni da por recuperada la evaluación antes de medirlo.
Terminal bajo `support_validation_recovery_v1/supervision/supervisor-fdjeu3ar/terminal.json`,
SHA-256 `33dd6f6d8d17a436f4ecc36e630c4dbd2a6920f9ba929027655121f8c682f6d9`.
