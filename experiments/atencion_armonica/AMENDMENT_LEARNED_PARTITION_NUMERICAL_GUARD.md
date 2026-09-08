# Enmienda operativa: suma de validación y reutilización de datos

Fecha: 2026-09-08. El [protocolo científico](PROTOCOL_LEARNED_PARTITION_READER.md)
conserva sus bytes y SHA `466ec717cde290277782dd720ebd97667fe2e2786f627f55c64a7c3ce4110138`.
La enmienda no cambia arquitectura, loss, seeds, selección, estimando ni presupuesto.

El primer entrenamiento rechazó una incidencia válida por redondeo de suma
`float32`. Sólo esa comprobación acumula ahora en `float64`, con el mismo
`atol=2e-7`. Los tensores del modelo, agregación, gradientes y optimizer siguen
en `float32`. La reparación ocurre después de observar train/calibración;
no se presenta como una decisión anterior a esos datos. Los tests siguen cerrados.

La declaración de reutilización fija el productor original, sus 50 bundles,
el intento fallido y el destino nuevo. Después de auditar la implementación,
se realizan los tres perfiles mecánicos requeridos. Una autorización reuse
vincula el ejecutor nuevo con la declaración auditada; no permite nuevos draws.

La importación copia payloads científicos byte por byte. Sólo cambia bindings,
resources y los cuatro índices `shards.json` que contienen referencias a bundles.
Los manifiestos nuevos declaran importación, productor original y ejecutor actual.
No se reetiquetan originales, ni se repiten forwards, score, targets o ajuste de
normalizadores. El mapping completo y el índice preparado se sellan y auditan
independientemente antes de entrenar. Cada request de training y selección
incluye `reuse_audit` exacto, o `null` si no es una autorización reuse.

La comprobación integral original→copia ocurre durante importación/auditoría.
Los consumidores posteriores verifican manifests/mapping/auditoría y conservan
sus comprobaciones transitivas sobre los bytes nuevos. No se permite equivalencia
global de identidades de fuente ni usar hashes antiguos para ejecutar código nuevo.

La primera celda conserva request, terminal y snapshot originales. Sólo se
admite continuidad desde su initial `0/0/0`, sin calibraciones ni estados
aprendidos posteriores. El estado completo se conserva, salvo su binding de
procedencia. Se publica `imported_initial` como raíz, sin inventar una arista
de entrenamiento entre dos snapshots con cero updates. El origen queda ligado
al resume y a la importación. Las interrupciones posteriores usan recuperación
ordinaria. El débito de 53,077151232995675 s permanece en el registro único.

La revisión del bootstrap posterior a la importación 05 detectó que la
recuperación consultaba un registro completamente terminal desde una reserva
todavía activa. No se lanzó training 05. En el corte 06 el supervisor conserva
su comprobación global previa a reservar; la recuperación verifica el asiento
original exacto ligado al terminal, sin exigir un cierre del intento actual.
Se preservan perfiles y copia 05; la nueva identidad usa destino `reuse_06`
y el mismo origen 04, sin nuevos draws ni reinicio del presupuesto.

Pruebas requeridas: soportes positivos 1..32 con padding hasta 94, incidencias
inválidas, equivalencia exacta de outputs/gradientes/updates admitidos por ambos
guards, importación completa y alteraciones rechazadas, cierre obligatorio,
continuidad initial y presupuesto acumulado. La enmienda termina al reanudar
la campaña; no sustituye sus 36 entrenamientos, selección, cuatro tests y replay.
