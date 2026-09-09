# Lector con evidencia generativa: estado de ejecución

2026-09-09. La campaña comenzó sus entrenamientos; todavía no hay resultados
comparativos de sus cabezas aprendidas sobre los tests nuevos.

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
auditados. El [ejecutor](run_generative_training.py) avanza por los 27
entrenamientos; la selección real aún no se ejecutó.
Los tests nuevos permanecen sin generar: requieren el cierre de los 27
entrenamientos, la selección fijada y el inventario de exclusiones sellado.
Base, Extendida e Histórico son referencias de sistema, no brazos de capacidad
igualada. No hay promoción arquitectónica ni una conclusión nueva sobre HIT.
