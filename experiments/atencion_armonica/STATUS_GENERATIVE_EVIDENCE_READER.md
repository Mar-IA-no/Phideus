# Lector con evidencia generativa: estado de ejecución

2026-09-09. La campaña completó sus 27 entrenamientos y la selección por
calibración. Los cuatro tests nuevos están en ejecución; todavía no hay
resultados comparativos completos de sus cabezas aprendidas.

El [protocolo](PROTOCOL_GENERATIVE_EVIDENCE_READER.md) mantiene tres brazos
—Local, Generativa y Desacoplada— con cabeza y pérdida comunes, 27
entrenamientos y cuatro tests nuevos. La pregunta es si la evidencia de ajuste
conjunto ayuda a elegir particiones más allá de los descriptores locales y
de su versión desacoplada, no si un residual pequeño identifica una fuente.

La implementación publicada incluye la
[cabeza](../../src/atencion_armonica/generative_evidence_model.py), el
[entrenamiento recuperable por celda](../../src/atencion_armonica/generative_evidence_cell.py),
el [corpus compacto](../../src/atencion_armonica/generative_evidence_corpus.py),
los [estimandos](../../src/atencion_armonica/generative_evidence_evaluation.py),
las [intervenciones de inferencia](../../src/atencion_armonica/generative_evidence_inference.py)
y las [referencias de sistema](../../src/atencion_armonica/generative_evidence_references.py).
Sus pruebas mecánicas no demuestran una ventaja neuronal.

El perfil CPU/GPU y de almacenamiento ya se ejecutó. La
[preparación de train y calibración](prepare_generative_evidence.py) terminó,
reutilizando observaciones y redes históricas congeladas. También se completaron
la entrega del corpus compacto, la medición de su carga real y la comprobación
de la proyección de recursos. Los ajustes guardados se reabren durante la
recuperación; no se sustituyen escenas.

Los supervisores de entrega, entrenamiento y selección están integrados y
auditados. El [ejecutor](run_generative_training.py) cerró los 27
entrenamientos de 50 épocas. Conservó 51 snapshots por celda y 270 estados de
calibración. Los rosters elegibles coinciden: 4036/4096 escenas de train y
503/512 de calibración; las escenas sin candidatos no se reemplazan.
El índice `data/atencion_armonica/generative_evidence_reader_v1/training/index.json`
registra este cierre con SHA256
`2d495bc107d875e7572670ca39b1db237c6524e0fe56d1f74a5ed9da1298013d`.
La [selección real](run_generative_selection.py) terminó en CPU y su auditoría
independiente confirmó la época 50 para los tres brazos, recalculando las
curvas desde los 270 resultados guardados sobre soporte común 503/512.
También verificó los 27 estados seleccionados y las inicializaciones
idénticas entre brazos/checkpoints a semilla de lector común. El índice
`data/atencion_armonica/generative_evidence_reader_v1/selection/index.json`
tiene SHA256 `02b21884343693ac745b7dc37db6bc1f18ff9185a32d65ba2f4c8d31828997f1`.
La calibración es desarrollo abierto, no evidencia de generalización.
Las revisiones de inferencia y supervisión cerraron sin hallazgos materiales
abiertos. El [supervisor de tests](run_generative_tests.py) completó el freeze
antes del primer draw y comenzó por IID. El archivo
`data/atencion_armonica/generative_evidence_reader_v1/test_freeze.json`
quedó fijado con SHA256
`8b1030ca466ef289eeef738149b878886a640fbc598774cb6fabd01147ee7937`.
Cada test conserva 45 predicciones antes de abrir su supervisión; luego se
evalúa y se verifica por replay CPU. Las pruebas previas fueron mecánicas:
el cierre de los cuatro tests y sus auditorías de evidencia siguen pendientes.
Base, Extendida e Histórico son referencias de sistema, no brazos de capacidad
igualada. No hay promoción arquitectónica ni una conclusión nueva sobre HIT.
