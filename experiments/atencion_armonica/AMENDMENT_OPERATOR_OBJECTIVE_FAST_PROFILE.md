# Perfil conjunto de medias y codec: operador separado

2026-09-14. Implementa la medición posterior al
[plan de optimización](PLAN_OPERATOR_OBJECTIVE_FAST_PATH.md). No modifica
los snapshots v1/cache, el manifiesto, sus límites ni el diagnóstico científico.

## Revisión previa

Entry point nuevo `profile_fast_diagnostic.py`. Su review exige estado
`FAST_PROFILE_REVIEW_COMPLETE`, informes independientes del corte puro y de
este operador, y hashes de seis archivos: módulo `operator_diagnostic_fast.py`,
sus tests, entry point, `test_profile_fast_diagnostic.py`, plan FAST_PATH y
esta enmienda. El snapshot valida además el review v1 y el review cacheado,
y liga el finish del primer perfil cacheado, que sella su informe. No toca
los cinco archivos fijados por ese review ni los ocho módulos originales.

La misma implementación operativa de R746 se copia a un archivo separado,
con cambios explícitos: kernel/codec rápidos, snapshot ampliado y nombres de
estado propios. No se sustituye globalmente ninguna función importada.
La revisión independiente debe auditar ese delta y sus pruebas antes de medir.

## Ejecución exacta y presupuesto

Usar mismo lock y `AttemptBudget(store,"profile")`; punto inicial observado
15.821776066 s totales y 9.645845636 s de perfil. Mantener180s perfil,
1800s total,600s reserva, un hilo,6GiBRSS,4GiB outputs y80GiB libres.
Start→runtime_revision→report→finish se conservan. Sólo finish COMPLETE
sella el informe; fallos/pausas/exceso consumen tiempo y no acreditan perfil.

Se autentican y recomputan únicamente las cuatro escenas0 preservadas,
en orden fijo, sin factores externos, nuevas escenas, fit, draw o forward.
Se usa el kernel nuevo, cotejo original y codec nuevo; se exigen iguales
bytes y recibo codec contra v1 antes de publicar cada prueba no-replacing.
La carga del compacto integra tiempo acumulado; cálculo, cotejo, codec y
publicación/guardas se registran por separado dentro de la pierna medida.

La primera ejecución exitosa del mismo snapshot es la única elegible; una
segunda llamada autentica y reutiliza ese resultado sin crear un intento,
incluso si cambia el filename del review. No seleccionar el timing más rápido.
La nueva fórmula sigue siendo v1: sustituye sólo la pierna diagnóstica y
conserva extracción/setup previos, máximos, replay, margen y reserva.

No se lanzará barrido por existir este perfil. Si no cabe, este corte decide
revisar explícitamente el modelo de costo por términos de trabajo, no seguir
encadenando optimizaciones locales sin examinar su suficiencia. Si cabe,
corresponde implementar y auditar la continuidad del runtime completo.
Ambas bifurcaciones mantienen el roster y el cierre científico del goal.
