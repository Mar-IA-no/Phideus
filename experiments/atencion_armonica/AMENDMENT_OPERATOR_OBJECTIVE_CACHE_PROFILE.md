# Medición del cache: revisión de operador, no cambio científico

2026-09-14. Detalla sólo la medición acotada prevista en el
[plan de costo](PLAN_OPERATOR_OBJECTIVE_COST_REVISION.md). El runtime de barrido
optimizado sigue fuera de alcance. No modifica protocolo, manifiesto o código v1.

## Identidad y entrada

Entry point nuevo: `experiments/atencion_armonica/profile_cached_diagnostic.py`.
Opera sobre los cuatro bundles autenticados de escena0, sin leer factores,
draws, sidecars externos, modelos o escenas adicionales. Usa el cache del
módulo `operator_diagnostic_cached.py` y el codec/cotejo v1 intactos.

Antes de computar exige review canónico con estado `CACHE_PROFILE_REVIEW_COMPLETE`:
referencias al manifiesto y perfil v1, hashes de módulo nuevo, sus tests,
entry point y ambos documentos de diseño, e informes independientes completos.
La revisión fija exactamente esos cinco archivos. Se autentican sus bytes y
los informes; además se vuelve a validar el review original de los ocho módulos.
Cambios posteriores impiden arrancar. El coordinador sólo publica ese review
después de integrar la auditoría del módulo y de este operador.

Sólo la primera ejecución con finish COMPLETE del mismo snapshot tiene
autoridad de perfil. Una llamada posterior autentica y devuelve ese primer
informe, sin abrir otro AttemptBudget ni medir otra vez. La identidad usa el
snapshot (incluidos perfil fuente y cinco archivos), no el nombre del review:
duplicar el archivo de revisión no habilita repetir hasta obtener menor tiempo.
Un snapshot distinto exige justificar y auditar la revisión antes de medir;
no se comparan repeticiones de la misma versión para escoger una favorable.

## Presupuesto y recibos

El operador toma el mismo lock v1 y construye `AttemptBudget(store,"profile")`.
Continúa desde los intentos existentes; el subtotal observado es15.225378603s
y9.049448173s de perfil. Conserva caps180s perfil/1800s total, reserva600s,
un hilo,6GiBRSS,4GiBoutputs y80GiB libres. No tiene flags de ampliación/roster.

Antes de cargar datos, cada intento publica `runtime_revision.json`, ligado al
hash de su start, al manifiesto y al review nuevo. Salidas nuevas viven bajo
ese mismo directorio de intento; no reemplazan `profile.complete.json` v1.
La señal de timeout y SIGINT/SIGTERM terminalizan el intento; no hay autoretry.

Por cada split en el orden original: autenticar bundle0, recomputar diagnóstico,
cotejar métricas/elecciones archivadas, producir bundle con codec v1 y exigir
igualdad completa de bytes. Guardar una copia de prueba por publicación atómica
no-replacing y medir diagnóstico+cotejo+codec+IO. El tiempo de cargar el compacto
no integra esa pierna, pero sí el tiempo acumulado del operador.
Registrar también por separado cálculo cacheado, cotejo, codec y publicación/
guardas; sus tiempos no deben confundirse con una aceleración atribuible al cache.

La proyección conserva extracción por byte y setup del perfil original, puesto
que esos puertos no se optimizaron. Sustituye únicamente los cuatro tiempos de
diagnóstico por los medidos del cache, y conserva exactamente la fórmula v1,
max82,2048escenas,27celdas,cuatro esquemas,replay,margen2,reserva600s.

El informe es un candidato no autoritativo hasta terminar: `finish COMPLETE`
debe sellar su ref y la ref de runtime_revision. Un candidato sin finish,
PAUSED/FAILED/BUDGET_EXHAUSTED no acredita perfil válido. Fallos y estados sin
cierre se contabilizan según v1. No hay marcador global del roster, ni recovery
que promueva resultados parciales. Una inspección del perfil usa siempre la
cadena finish→informe→runtime_revision→review, no la mera existencia del JSON.

## Interpretación

Una mejora de velocidad no es mejora científica: los bytes deben coincidir.
Incluso si la proyección cabe, todavía falta diseñar e implementar la continuidad
del runtime completo, auditarla y efectuar barrido/replay. Si no cabe, conservar
el resultado y revisar el siguiente término de costo, sin cambiar el presupuesto
ni utilizar el perfil de otra manera para declarar cerrado el goal.
