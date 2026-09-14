# Inventario de carga para revisar la admisión del diagnóstico

2026-09-14. El [último perfil](RESULTS_OPERATOR_OBJECTIVE_FAST_PROFILE.md)
no habilita el barrido. Antes de cambiar la proyección, medir qué parte de
la carga máxima supuesta corresponde al roster efectivamente preservado.
No cambia la hipótesis, el experimento, el presupuesto ni la fórmula vigente.

## Operación autorizada tras auditoría de este corte

Implementar `experiments/atencion_armonica/inventory_diagnostic_workload.py`
y `test_inventory_diagnostic_workload.py`. Auditar conjuntamente plan,
implementación y pruebas con instancia Sol/high independiente. Un review
inmutable `WORKLOAD_REVIEW_COMPLETE` debe fijar sus tres archivos y el informe;
también validar el review fast previo, que conserva los snapshots anteriores.
No usar los prefijos de archivos congelados ni modificar sus bytes.

Leer el inventario v1 autenticado y usar `ClosedCorpus.load_split` para los
cuatro escenarios, uno por vez. Este puerto autentica las fuentes, los tres
checkpoints y los offsets de las 27 predicciones originales. Puede leer
metadatos de evaluación por autenticación heredada, pero el inventario de
carga no utiliza métricas, elecciones, etiquetas ni rendimiento para contar.
No abrir factores gzip, sidecars externos o modelos; no llamar extract_scene,
fit, draw, forward ni diagnose_scene. No GPU ni infraestructura remota.

Por cada una de las 512 escenas, comprobar que los tres checkpoints tienen
los mismos offsets/cantidad y particiones observadas; conservar identidad,
cantidad de candidatos y grupos, tamaños y número de eventos por candidato.
La cantidad debe estar entre 0 y 82 y coincidir con el número de particiones
de los metadatos. Comprobar particiones canónicas, índices enteros, grupos
disjuntos, cobertura de eventos y tamaño; no reconstruir la verdad plantada.
La escena sin candidatos permanece en el inventario.

Guardar las referencias originales de entradas, el hash de offsets y una
huella de particiones por escena. Emitir histogramas exactos de C, suma de C,
suma de choose(C,2) y suma de choose(max(C,31),2). El piso31 corresponde al
menor C de los cuatro perfiles fijados y hace visible el trabajo de escenas
pequeñas: estas sumas son descriptores de carga, no una fórmula habilitante.
No elegir piso, escenas o categorías después de ver su rendimiento.

## Continuidad y presupuesto

Mismo lock y `AttemptBudget(store,"profile")`, desde16.400343163s totales y
10.224412733s de perfil; límites v1 intactos. Start y runtime_revision fijan
manifest/review. Sólo finish COMPLETE sella el informe `WORKLOAD_COMPLETE`;
PAUSED/FAILED/BUDGET_EXHAUSTED conservan sus costos sin dar autoridad al
candidato. Las señales se terminalizan como en los operadores previos.
Sólo el primer COMPLETE del snapshot se reutiliza: cambiar filename de review
no produce otra ejecución. Salidas nuevas bajo el directorio del intento,
sin sobrescribir v1 ni publicar complete.json/replayed.json del roster.

## Pruebas y condición de avance

Fixtures completos de 512 IDs por split, pero arrays pequeños: C0/1/2,
offsets inconsistentes, discrepancia de checkpoint/particiones, índices,
cobertura y tamaños inválidos, agregados exactos y metadatos inmutables.
Probar review/first-success, fallo/pausa y finish sin autoridad parcial.
El operador necesita ese review antes de abrir los arrays reales.

El resultado debe decidir una enmienda de costo y runtime, no una nueva
optimización local. Las alternativas a evaluar son usar carga auténtica con
costos fijos explícitos, o separar preparación compacta reutilizable con
presupuesto acumulado. Ninguna queda autorizada por este inventario. La
enmienda debe incluir lecturas de validación/replay y guardas cuyo costo crece
con archivos publicados, mantener margen/reserva y no escoger promedios
favorables. El cierre sigue exigiendo las 2048 escenas, replay, informe de
mecanismo y auditorías científica/técnica; este inventario no lo sustituye.
