# Enmienda de ejecución completa del diagnóstico

2026-09-14. Diseño pendiente de auditoría independiente de plan e
implementación. Desarrolla el protocolo operador–objetivo sin alterar su
pregunta, roster, fórmulas científicas, fuentes, oracles o agregación.

## Corrección de rumbo

El inventario real ya está completo y las optimizaciones conservan los bytes
de las cuatro escenas perfiladas. Seguir encadenando perfiles para satisfacer
la admisión inicial posterga la pregunta científica. El límite de 1800 s era
una decisión operativa, no una condición de validez del diagnóstico. Esta
enmienda lo reemplaza explícitamente por 7200 s acumulados de operadores,
incluyendo los 23.820174567 s consumidos. No se reinicia el contador ni se
modifican los recibos, el manifiesto o los módulos congelados.

Se mantienen un proceso/hilo CPU, 6 GiB RSS, 4 GiB de salidas propias y
temporales, 80 GiB libres, margen de proyección ×2 y reserva final de 600 s
para auditoría. El subtotal histórico de perfiles permanece registrado y no
se autoriza otro perfil real. Dos horas son un límite duro, no una duración
esperada ni una autorización para reintentar tras agotarlo. Lectura,
autenticación, operaciones pequeñas y publicación son trabajo CPU; CUDA no
resuelve ese costo de manera directa. No hay entrenamiento ni forward.

## Proyección explícita antes de ejecutar

Usar exclusivamente el primer perfil fast exitoso y el inventario completo
autenticados. Mantener las tasas máximas de los cuatro perfiles, no seleccionar
escenas favorables. E es el máximo tiempo de extracción por byte decodificado
por el total de bytes fuente. D es el máximo tiempo diagnóstico por par entre
esos perfiles, multiplicado por la suma observada de choose(max(C,31),2).
Las 27 celdas y cuatro esquemas permanecen presentes en ambas mediciones;
no se añade ni quita ese factor. El piso31 es el fijado antes del inventario.
S es dos veces la suma de los cuatro tiempos de preparación originales.

Proyectar `consumido + 2*(E + 3*D + S + 900) + 600` y exigir que quepa en
7200 s. Dos D corresponden a cálculo/publicación inicial y replay; el tercero
reserva una pasada equivalente para carga, validación y cotejos no separados
en el perfil. Los 900 s adicionales son una provisión operativa explícita para
agregación, recibos, relecturas y guardas sobre el árbol creciente, no un costo
medido ni una cota demostrada. El margen ×2 se aplica también a esa provisión.
Es una previsión conservadora imperfecta: el límite real instrumentado decide
si la ejecución debe detenerse. Publicar todos los términos y su procedencia.
Mantener la proyección de bytes original, que no depende de una extrapolación
optimista del inventario. No reducir la reserva para hacer caber una corrida.

La admisión del primer barrido cuenta el trabajo completo, incluidos los cuatro
prefijos existentes. En continuaciones se exige el mismo recibo de admisión
inicial y se aplica el residual del límite duro, sin rehacer una proyección de
trabajo completo que duplique lo ya consumido. El replay conserva la reserva
y el mismo contador. Un fallo real por recursos exige rediseño explícito;
una pausa normal permite reutilizar sólo unidades completas y autenticadas.

## Autoridad y cambios de implementación

Añadir módulos con nombres distintos de los ocho originales congelados:
`src/atencion_armonica/diagnostic_execution_revision.py` y el CLI/pruebas en
`experiments/atencion_armonica/`. Reutilizar los puertos científicos revisados
y el kernel/codec fast sin monkeypatch global ni cambiar su aritmética.

Conservar la raíz v1 y su manifiesto como identidad científica e historial del
presupuesto original. Un review inmutable `EXECUTION_REVISION_REVIEW_COMPLETE`
fija esta enmienda, implementación, pruebas, reviews ancestrales y los seis
intentos históricos exactos. Declara `supersedes_operational_limits` y los
límites efectivos. Cada nuevo start referencia ese review como
`execution_revision`; por tanto no presenta 7200 s como presupuesto original.
Nuevos recibos de escena, admisión y cierres globales enlazan la misma revisión.
Los cuatro prefijos originales se autentican y reutilizan sin modificación.

El presupuesto revisado conserva el ledger contiguo original y sus reglas:
cada intento cobra su tiempo observado; un start sin finish cobra toda su
reserva; un BUDGET_EXHAUSTED impide reintento. Valida las referencias exactas
de los seis intentos históricos y exige la revisión actual en los posteriores.
No admite un límite por argumento CLI ni un nuevo review para reiniciar tiempo.
Conserva lock, alarmas, chequeos de memoria/disco/tiempo, señales y estados
terminales. No relaja la inspección completa de salidas para ganar velocidad.

El ejecutor recorre los cuatro tests y las 512 escenas mediante los mismos
puertos. Mantiene transacciones de unidad, compactos reutilizables, validación
completa, resumen por escenario y publicación sólo después de un finish
COMPLETE que selle su candidato. El replay comprueba bytes de cada bundle y
resumen. Una recuperación sólo publica un candidato ya sellado, no transforma
un prefijo o intento fallido en éxito. La revisión de código se comprueba antes
de recuperación y de cada nueva operación.

Los nuevos starts usan operaciones `revision_run`, `revision_replay` y
`revision_audit`; sus marcadores globales son `execution/complete.json` y
`execution/replayed.json`, no los paths legacy. El recovery revisado valida
review, operación, candidato y sello antes de publicar. El recovery original
no debe reconocer estos finishes ni su CLI aceptar estos cierres como propios.
Esto evita que una caída entre finish y commit permita recuperar la nueva
ejecución mediante un camino que sólo comprueba el manifiesto histórico.

## Pruebas, auditoría y cierre

Auditar el plan antes de implementar. Luego auditar el delta y sus fixtures
independientemente antes del barrido: continuidad de consumo, rechazo de
historial/review alterado, cotas de recursos, fallo/pausa/señales, publicación
y recuperación sellada, ejecución y replay de fixtures, equivalencia con
kernel original, tratamiento de escenas sin candidatos y pruebas negativas
de no recuperación/aceptación por el CLI legacy. La revisión sólo
autoriza ejecución cuando no queden findings materiales abiertos.

Tras ejecución completa y replay, obtener auditorías técnica y de alineación
geométrica sobre artefactos reales. Publicar el diagnóstico por escenario y
soporte, sus límites y la siguiente pregunta discriminante. Este cambio de
presupuesto no promueve arquitectura, declara GO/NO-GO ni cierra el goal.
