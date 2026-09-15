# Continuación de la auditoría tras fallo de apertura

2026-09-15. Enmienda operativa del plan de auditoría final. No cambia el
experimento, los resultados ni el alcance de las comprobaciones independientes.

## Incidente y frontera preservada

El replay y el informe completaron sus intentos0018 y0019. La admisión de
auditoría0020 también terminó COMPLETE, con19.62826516793575 segundos.
La creación posterior del store de verificación falló: el preflight anidado
ya había creado el directorio padre sin binding y ArtifactStore rechazó
adoptarlo. No se creó el intento final ni PRECOMMIT ni comenzó VERIFY.

El checker publicado en281a384 queda intacto. Un operador nuevo de continuación
reutiliza la admisión y proyección autenticadas, sin repetir el preflight, el
replay, el informe, entrenamientos, forward ni fitting. El directorio existente
no se adopta, mueve ni elimina. La verificación usa la raíz hermana
`data/atencion_armonica/geometric_decision_energy_v1/audit-final-verify`.

## Contabilidad explícita

No existe un reloj preservado del tramo posterior al preflight hasta el fallo.
No se presenta un tiempo estimado como medición. Una enmienda con manifiesto
propio conserva el incidente y representa una obligación contable conservadora
por toda la reserva inicial pendiente:1500−19.62826516793575 =
1480.3717348320642 segundos.

La enmienda publica un start estándar sin finish, deliberadamente y documentado
como cargo de reserva, no como una ejecución científica histórica. El ledger
vigente consume íntegra esa reserva. No se fabrica un elapsed ni se cambia la
implementación presupuestaria. Se rechaza duplicar la enmienda o reinterpretar
su ausencia de finish como autorización para repetir trabajo.

Con el informe y esa enmienda, el cargo audit es1625.1773888189346 segundos;
queda1974.8226111810654 bajo el tope3600. La continuación recibe como máximo
1500 segundos adicionales, siempre limitada por los saldos actuales audit y
global49200. Esto amplía explícitamente la reserva inicial de auditoría, no
el presupuesto global ni su cobertura. La proyección autenticada de
327.0271509163656 segundos debe caber en el saldo y reserva actualizados.
Se separan los tiempos medidos de los cargos conservadores al informar costos.

## Secuencia única de recuperación

1. Fijar fuentes del checker, wrapper y esta enmienda. CPU de una hebra,
   CUDA oculta, mismo lock exclusivo; nada de ejecución concurrente.
2. Autenticar el preflight0020 y sus refs exactas, informe y condiciones
   previas. Rechazar un intento final/PRECOMMIT/result ya existente. El
   acceso preliminar se limita a metadatos, nunca a resultados científicos.
3. Publicar la enmienda contable una sola vez con cargo conservador explícito
   y reconstruir el ledger acumulado. No modificar recibos existentes.
4. Crear manifiesto e intento de continuación antes del store de salida y
   antes de operaciones que puedan fallar. Contabilizar desde el lanzamiento;
   errores conservan un finish o la reserva íntegra si no puede cerrarse.
5. Reautenticar admisión y proyección; ligar wrapper, enmienda y preflight al
   binding y al resultado. Crear la raíz hermana sólo bajo ese intento.
6. Ejecutar PRECOMMIT y VERIFY del checker original con exactamente el
   alcance aprobado. Verificar fuentes de wrapper y checker antes/después.
   El resultado distingue continuidad operativa de evidencia matemática.
7. Publicar cierre o discrepancia y preservar toda salida. Ningún fallo
   habilita un relanzamiento implícito de esta continuación.

## Pruebas antes de ejecutar

La revisión del wrapper incluye un fixture de arranque que utilice realmente
ArtifactStore y StageBudget sobre un árbol abstracto propio: preflight completo,
padre sin binding, raíz hermana y costo acumulado. No basta buscar strings en
el código ni reemplazar esos componentes por mocks que oculten el fallo.
Los cálculos científicos pueden sustituirse por fixtures abstractos en esta
prueba operativa. Verificar también fallo al crear store, repetición rechazada,
recibos mutados, cargo duplicado, falta de saldo y cambio de fuentes.

La implementación recibe revisión independiente antes de su única ejecución
real. El cierre del goal conserva auditoría técnica completa, revisión de
horizonte, informe humano, documentación/wiki y publicación. Ninguna enmienda
operativa constituye promoción arquitectónica ni GO/NO-GO.
