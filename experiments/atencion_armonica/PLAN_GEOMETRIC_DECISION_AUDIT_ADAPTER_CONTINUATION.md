# Continuación técnica tras corregir los adapters del auditor

2026-09-15. Diseño previo con requisitos R780/R781 integrados; ejecución
pendiente de revisión independiente de la implementación.

## Alcance y motivo

La campaña, evaluación, replay e informe de datos están completos. El intento
de verificación final 0022 terminó FAILED al buscar `manifest.root` en OPEN:
el productor publica esa ruta en `output.root`. La corrección pertenece al
auditor, no al experimento. Además se comprueban bindings internos ya emitidos
por los productores y la igualdad de raíces del informe. No cambian core
matemático, targets, selección, protocolo, datos ni resultados.

Los fixtures anteriores fabricaban un manifest OPEN con `root`, ocultando la
incompatibilidad. Los nuevos reproducen el manifest sin ese campo y verifican
raíces e identidad interna de terminales autenticados, incluyendo rechazos.
No se presentan estos tests de contrato como verificación científica.

## Historia inmutable y autoridad de continuación

- Preservar el preflight 0020, la responsabilidad contable 0021 sin finish y
  el finish FAILED 0022. No publicar otra responsabilidad por 0022: su tiempo
  de 21.753944049938582 segundos está medido.
- Preservar `audit-final/preflight` y `audit-final-verify`, incluido el primer
  PRECOMMIT, SHA `4af416c714bf2475056af0ac06127d50e55d428ea900647ac00a7fbf599a5fa0`.
  Sus referencias al checker anterior describen historia, no código vigente.
- Exigir el finish 0022, SHA
  `d72866e2d75c2d2a67634582695cc34e59346becef2c1f7c15bd02a94c083d4c`,
  como cola terminal antes del nuevo intento y autenticar su start/manifest,
  discrepancia y vínculo con el PRECOMMIT histórico.
  La raíz anterior debe contener exactamente binding y PRECOMMIT. Autenticar
  también el start 0021 sin finish y comprobar que el nuevo charged_before sea
  idéntico al charged_after de 0022, sin sumar nuevamente aquella reserva.
- Usar una entrada nueva, `continue_geometric_decision_audit.py`, y una raíz
  hermana nueva `audit-final-adapter-verify`. Nunca reintentar el wrapper
  anterior ni adoptar stores históricos bajo un binding nuevo.
  Los nombres de control son `final-technical-audit-adapter-continuation.json`
  bajo manifests, outputs y discrepancies; todos deben estar ausentes. Cualquier
  intento posterior a 0022 o raíz nueva existente rechaza el lanzamiento.
- Vincular separadamente versiones históricas y revisadas del checker.
  El código histórico permanece recuperable en Git; no se exige que sus hashes
  antiguos coincidan con archivos que precisamente se han corregido.
  No reutilizar `validate_initial_state` del wrapper anterior: exige ausencia
  de la recuperación que ahora debe preservarse y liga preflight a código actual.

## Presupuesto y ejecución

La reserva nueva no supera 1500 segundos ni los saldos comunes. El cargo de
audit observado tras 0022 es 1646.9313328688731 segundos: quedan
1953.0686671311269 del cap de 3600. El cap global de 49200 no cambia.
El lanzamiento revalida saldos, no confía en esta fotografía contable.

Lock común y reloj desde lanzamiento. Publicar manifest e intento antes de
abrir el nuevo store o realizar lecturas pesadas. Mantener guards de tiempo,
RSS y disco y cierre medido de fallos. Una interrupción no autoriza retry.
CPU de un hilo, CUDA oculta. No perfiles, entrenamiento, fitter, forward,
evaluación, replay o informe nuevos.

Reutilizar admisión y proyección autenticadas del preflight: mismo universo
y mismos kernels. Las comprobaciones nuevas comparan metadata ya cargada;
no añaden recorridos de payloads. Si aparece un cambio material de costo,
resolverlo antes de ejecutar, no recortar alcance para entrar en el saldo.

## Corte y verificación

Revalidar admisión contra terminales sellados. El PRECOMMIT nuevo debe ligar
al anterior y conservar exactamente sus cuatro cortes: 27 escenas originales
y 16 derivadas. La igualdad incluye IDs, razones y referencias, no sólo
conteos. También se conservan terminales científicos, freeze, seal, contrato
post-sello y origen del informe. No elegir otro corte después de leer métricas.
El coordinador ya leyó métricas después del primer PRECOMMIT. El nuevo registra
`result_payloads_opened=true` con ese alcance, sin fingir una segunda ceguera;
la protección del corte reside en su identidad exacta con el documento previo.

Publicar PRECOMMIT nuevo antes de VERIFY. Ejecutar el checker revisado con
todo el alcance del plan final original. Antes de publicar COMPLETE, volver
a comprobar los hashes de checker, core, wrapper, plan y protocolo.
La autoridad de continuación liga las revisiones R780/R781 y el registro del
acceso posterior a resultados. El enlace a root/binding/PRECOMMIT históricos
forma parte del binding nuevo, no de un campo informativo sin validación.

## Aceptación y cierre

Revisión independiente del diseño y de la implementación, fixtures con
ArtifactStore/StageBudget reales para arranque/cierre, rechazo de historia
mutada, cut alterado, source pin erróneo y segundo intento. Publicar código
revisado antes de la corrida real y conservar su único handle hasta terminar.
COMPLETE técnico no sustituye auditoría de horizonte, informe humano,
actualización documental ni cierre íntegro del goal.
