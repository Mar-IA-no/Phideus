# RAID1 como archivo principal; M.2 para el frente CASMI

4/11/2026. El retiro de datos y resultados terminó y quedó comprobado. Se retiraron de M.2 **1,388,989 archivos y enlaces netos**, equivalentes a **176.54 GiB lógicos**. El archivo principal permanece en RAID1; código, documentación y conjunto activo CASMI siguen disponibles en M.2. El registro final de la operación quedó archivado y comprobado en RAID.

| Proyecto | Archivos y enlaces retirados netos | Bytes regulares lógicos |
|---|---:|---:|
| Phideus | 1.256.292 | 42.719.772.374 |
| Phideus-CASMI | 132.697 | 146.842.261.440 |
| Total | 1.388.989 | 189.562.033.814 |

**Espacio libre al cierre: 296.10 GiB.** Además de los datos y resultados, se archivaron en RAID los registros voluminosos de esta operación y se retiraron4.076copias locales de control, con5.568.223.133bytes lógicos.

Justo antes del retiro había 117.47 GiB libres en el volumen; después del retiro, poda y recuperación de las dependencias necesarias se midieron 290.90 GiB. Son mediciones del volumen compartido, no una atribución exclusiva de toda su variación: hay hardlinks, archivos de control y actividad concurrente. Los bytes lógicos retirados tampoco equivalen a bloques físicos liberados.

## Conservación y comprobación

Cada archivo retirado tiene una contraparte directa comprobada o un miembro indexado en un tar PAX nuevo. Se cotejaron tipo, tamaño, fecha y destino de enlaces; no se repitieron hashes completos para verificar el backup. Los 1.341 lotes terminaron y su cotejo posterior comprobó ausencia local, respaldo vigente y ledger sin retiros pendientes. Se conservaron los originales científicos de selección, checkpoints, curvas, predicciones y cierres, incluidos los fallos anteriores y sus recuperaciones.

Los faltantes excluidos del backup habitual quedaron bajo `_project_archives/<proyecto>/20261104/` del archivo RAID, con índices de miembros y metadatos. Las copias se completaron fuera del árbol que se sube a Drive y se publicaron cerradas. Selección, recibos, herramientas y atributos de directorio se respaldaron en RAID antes del primer retiro; el archivo final conserva los ledgers, ambas pasadas de poda y la auditoría independiente. Su condición final quedó corroborada, sin hallazgos materiales abiertos.

Se quitaron 214.908 directorios vacíos del conjunto previamente registrado. La comprobación final identificó cuatro enlaces del runtime y treinta informes pequeños requeridos por la wiki que debían permanecer locales: se recuperaron desde RAID, sin regenerar ciencia, con 6.122.568 bytes regulares y quince directorios recreados. Los conteos netos de la tabla ya descuentan esas treinta y cuatro recuperaciones. La selección efectiva incorpora esta corrección; los registros del retiro original se conservan.

El intérprete CASMI conservado importó NumPy2.5.3, PyArrow25.0.1, threadpoolctl3.6.0, Torch2.7.1+cu126, la clase MIST y el consumidor general, sin inicializar CUDA. La wiki validó22páginas y91fuentes. No se cargaron pesos para otra evaluación ni se ejecutaron entrenamientos. Código versionado y cambios concurrentes quedaron conservados.

## Recuperación y continuidad

El mapa de restauración distingue contraparte directa de archivo tar y miembro. La recuperación se realiza primero en un destino separado y comprueba metadatos y enlaces antes de usar el entorno. Una extracción real de un miembro de las dependencias conservó tamaño, permisos, propietario y fecha. Los índices mantienen los atributos originales de directorio; los hardlinks entre archivos distintos pueden requerir reconstrucción coordinada de su topología.

La [política de almacenamiento](../../../Phideus-CASMI/docs/STORAGE_POLICY.md) rige las corridas siguientes. Los estados finales de CASMI276, operandos preparados, roster, banco, química, predictor final y dependencias necesarias permanecen accesibles. Los demás frentes mantienen sus pausas. [CASMI276](../../../Phideus-CASMI/Biblioteca/Competition_Delivery/CASMI276_VIEW_DIVERSITY_RESULTS.md) quedó cerrado;277 requiere rebasar sus rutas y presupuesto después del archivo. Esta limpieza habilita espacio, sin constituir otra mejora molecular.

El script de Mariano copia acumulativamente M.2 al RAID y luego al destino Drive configurado; ninguna etapa propaga los borrados locales. No se modificó ni detuvo ese proceso. La presencia en RAID no acredita la llegada a Drive: los complementos pueden requerir otra ejecución del backup del usuario.
