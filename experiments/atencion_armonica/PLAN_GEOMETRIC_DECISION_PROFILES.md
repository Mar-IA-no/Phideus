# Perfiles de ejecución del contraste geométrico

2026-09-14. Suplemento de implementación del protocolo vigente, sin cambiar
roster, arquitectura, pérdidas, selección ni topes. La preparación OPEN ya
está completa y no se repite. Este perfil no produce resultados científicos.

## Entradas y medición

Autenticar primero el cierre COMPLETE del operador OPEN, su manifiesto,
binding y recibo de preparación. Extraer una vez el primer batch TRAIN de la
primera seed y backbone del protocolo, siguiendo el orden aleatorio ya fijado.
Reusar los ocho canales entregados y la escala congelada; los updates
mecánicos utilizan sólo targets de esas 32 escenas TRAIN. Los puertos
existentes autentican supervisión por shard de 512 escenas, no por fila:
no afirmar que sólo se leen 32 targets. Guardar el subset con recibos,
identidades y orden originales. En el mismo operador medir una carga completa
de CellData para el primer backbone, incluidos targets OPEN y validación,
y cotejar sus filas con el subset. La carga completa abre TRAIN y calibración
OPEN por el puerto ya auditado, pero no evalúa la calibración, no selecciona
épocas y no computa métricas sobre sus targets. Ese costo alimenta la proyección de carga
de los tres backbones; no se sustituye por la latencia de un subset. No llamar
al sampler ni abrir tests nuevos. Ambas lecturas caben en la reserva de
extracción y no recalculan ajustes ni forwards.

La cabeza se mide separadamente para CPU y CUDA, con ambos objetivos y dos
entradas: ese batch real y una envolvente aritmética densa de 32 escenas,
328 grupos y 82 candidatos. La envolvente no pretende ser un fenómeno
realizable. Cada caso hace 25 actualizaciones mecánicas con el kernel real,
collation, transferencias, validaciones y diagnósticos completos. Conservar
estado inicial, tras 10 updates y final; restaurar el intermedio y repetir
los 15 restantes para comprobar igualdad exacta del estado. Esa repetición
se cobra al perfil, no al costo normal por update. Medir además tres
evaluaciones con escritura/lectura de outputs y serialización de checkpoints.
Los IDs mecánicos 0..31 sólo indexan el subset: nunca son una celda científica.

El fitter usa por separado CPU y CUDA, Grid(257,65,4), assignment_batch=8.
Medir cada rama y tamaños 4..8 sobre grupos aritméticos y grupos del primer
batch, elegidos determinísticamente por orden de escena y firma, hasta ocho
grupos por tamaño. Conservar factores completos, procedencia y tiempos.
Son mediciones de la primitiva por forma, no refits de todo OPEN. La admisión
de tests deberá agregar composición de particiones, backbone, pool e I/O;
no presentar el costo aislado del fitter como costo de la pipeline completa.

## Control y elección

Todos los operadores conservan el ledger acumulado existente. La extracción
tiene reserva de 100 s; cada combinación cabeza/fitter × CPU/GPU, 120 s;
el total de perfil sigue limitado a 600 s, incluyendo startup y repeticiones.
Mantener RSS/VRAM 8 GiB y guardas de disco del protocolo, lock exclusivo,
manifiestos de código/runtime y recibos inmutables de intento y cierre.
Una interrupción no habilita repetir silenciosamente una medición.

Antes de CUDA, informar alcance y estimación de recursos, comprobar dispositivo
y ausencia de otro owner. Una hebra CPU, determinismo y sin TF32/AMP.
Elegir por costo proyectado completo, no por latencia de forward aislado.
El entrenamiento cuenta 72 × 50 × ceil(4036/32) updates, además de snapshots,
calibraciones y carga autenticada. Aplicar reserva de 25%; si no cabe, revisar
recursos explícitamente, no reducir el experimento. Una auditoría independiente
comprueba los puertos y la medición antes de ejecutar sobre datos reales.
