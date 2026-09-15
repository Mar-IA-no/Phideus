# Sello global y evaluación prospectiva

2026-09-14. Integración de PLAN_GEOMETRIC_DECISION_FRESH y del protocolo
vigente. No cambia escenarios, semillas, modelos, pérdidas ni estimandos.

El operador observable prepara cuatro batches de512escenas y sus probes.
Antes de publicar un sello, reabre los cuatro índices de draws y ejecuta
`run_observed(recovery_only=True)` sobre cada batch. Debe obtener exactamente
los cuatro recibos completos conservados, sin modelo ni fitting. Verifica
rosters originales, cabezas y probes; conserva en el control un inventario
por path/tamaño/SHA de todos los archivos del store observable. Sidecars
entran al hash como bytes, nunca como JSON. El sello vive fuera de ese store,
evitando autorreferencia. No se modifica el store después del sello.

El puerto privilegiado sólo acepta un finish COMPLETE del operador
`prospective-observables`, enlazado al freeze y al sello. Revalida el inventario
completo antes de llegar al primer parser de sidecars. Un batch incompleto,
hash cambiado, archivo adicional, ausencia de probe o finish parcial impide
leer cualquier respuesta. El freeze fija el roster de2048escenas, los144
estados del archivo admitido, normalización, escala, runtime, código y recursos.

Evaluación: reconstruir la verdad original una sola vez por escena con el
validador existente, sin regeneración. En los probes, mapear labels desde
rango original a identidad de evento y de allí al rango del probe; no validar
el qprobe como reconstrucción física del sidecar original. Calcular una vez
los targets de cada universo. Reabrir predicciones firmadas guardadas y
compararlas con esos targets mediante el núcleo de métricas auditado.

Preservar targets/labels por escena, métricas y diagnósticos por cabeza,
referencias clásicas, estados iniciales/seleccionados, estratos observables,
presencia de la plantada y correspondencia efectiva del sham. Reportar los
cuatro escenarios por separado. El primario exige512deformed_family y
regret_tM, nueve celdas por brazo dentro de escena, bootstrap predeclarado.
Conservar índices/distribuciones. Los secundarios y probes no redefinen el
primario ni habilitan selección posterior de modelos.

La recuperación post-sello vuelve a verificar predicciones y recalcula
métricas desde los mismos bytes, sin training/forward/fitting. Los artefactos
de evaluación se publican en otro store y sólo aceptan igualdad exacta en
replay. No reparar resultados ausentes en el modo replay.

Presupuesto: el replay observable previo al sello agrega su proyección medida
a fresh, además del recorrido original. El replay posterior sigue en
evaluación/replay. Incluir ambos explícitamente antes del freeze, más
producción/IO, inventario, métricas/bootstrap y verificación final. Perfilar
las operaciones restantes sólo sobre OPEN/fixtures autorizados. Si la suma
no cabe en los topes, registrar revisión de recursos antes de los nuevos
draws; no reducir silenciosamente muestras o controles. Auditar integración
y supervisor antes de usarlos con datos nuevos.
