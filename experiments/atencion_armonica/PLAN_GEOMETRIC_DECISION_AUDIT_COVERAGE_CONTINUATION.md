# Continuación por corrección de nombres de cobertura

2026-09-15. Diseño previo, no autorización de retry de los operadores anteriores.

El intento0024 cerró FAILED en104.41553306300193s: el auditor confundía los
ocho almacenes de casos de perfiles de cabeza al usar un mismo label para
todos. El fix conserva Coverage y los hashes: distingue cada caso mediante
el label del perfil padre (incluye dispositivo), case y objetivo. La revisión
del mecanismo debe cubrir todos los sitios AuditStore/open_bound, no sólo
la primera colisión. Las fuentes y los resultados científicos no cambian.

## Entrada finita y procedencia

1. Crear una entrada fija `verify_geometric_decision_coverage.py`, root
   `audit-final-coverage-verify` y nombres control
   `final-technical-audit-coverage.json` en manifests/outputs/discrepancies.
   No hacer configurable por CLI el root, el historial ni la selección.
   No volver a ejecutar ninguno de los tres operadores anteriores.
2. Bajo lock común, exigir exactamente25starts, tail0024FAILED con completion
   nulo, y autenticar sus cuatro registros de control. Finish435B
   SHAddf77779bbbe9d390f5c62d4c82d5b34f7926c2667471c5488b4a1b275131bd5;
   start1320B SHAcee4a7113942ddf0af01c153afe61c8ce5e4302b8bc6a699d2daddfa8694f8a2;
   manifest4432B SHA0d4691db8baa541ac3b7a9cec57c47010d8dd961a832306cdcae788eb91dcafd;
   discrepancy395B SHAa588da17b078edf8674764ed07562170f2499367569de029194ea6a90fb72646.
   Exigir ValueError y mensaje exacto de conflicto profile-head-case:binding.json.
3. Exigir los roots0024,0023,0022 sin symlinks ni entradas adicionales:
   `{binding.json, precommit.json}`. Autenticar0024binding7471B
   SHA42e71604dfd2633646f67d3ad17d9e239d895fd31a48b60940cebd3621eb54a5
   y PRECOMMIT21046B SHAbd5368a55749b58bd31c3765c14e913e58ed28dbd073076f8bbe2103bce80aeb.
   Recorrer las referencias autenticadas de su autoridad hacia0023 y0022,
   cotejando manifest/start/finish/discrepancy de cada salto con el ledger.
   Conservar la liability0021 sin finish y el preflight0020 originales.
4. Separar explícitamente checker actual corregido, checker59d872a de0024,
   checker7b1622f de0023 y checker25e81f de0022. No exigir a fuentes corregidas
   que vuelvan a coincidir con hashes históricos. Ligarlos por los receipts
   anteriores y autenticar los sources actuales por hashes revisados.
5. El binding nuevo liga wrapper/plan nuevos, checker/core, plan científico y
   protocolo, revisiones causales/diseño pertinentes, helpers importados y
   registro de acceso posterior. Ligar root/binding/PRECOMMIT de los tres
   antecedentes. Revalidar fuentes antes de start y después del VERIFY.
   Pins concretos: checker aada6600410874e68af3797bd00c525368568a0d2293fa94eb718691a7ec05a9;
   core ae09edc9507e89e73df74e03b5345c95572601032e966f83de283d44f2d55dd9;
   R787 f9ddd704d4b0491d3a347b1505bba34bd223fb8183da3bf57a64447e99cc2b52;
   R788 una vez publicado. Ligar por hash este plan y los dos módulos helpers:
   continue_geometric_decision_audit.py (2518c217cb6a80ef2c9ecbe715753cb5a97ff942e83228c035bc4ac73e852244)
   y verify_geometric_decision_profile_order.py (692fcc7e43be85fc58ea3adb2ef37cf58f02a027b9e1648574604b0cc8f35c1d).

## Presupuesto, selección y efectos

Audit acumulado1869.1692501478828s; saldo1730.8307498521172s. Reserva nueva≤1500s,
proyección preservada327.0271509163656s y saldo global revalidado. Los caps
audit3600/global49200 no cambian. El fallo0024 ya tiene cargo medido; no otra
liability ni segundo cobro de0021. `charged_before` debe coincidir exactamente
con `charged_after` de0024 y con el constructor StageBudget real.

Reloj nuevo antes de imports, CPU1thread/CUDAhidden, guards RSS/disco y alarma
restando todo el tiempo desde lanzamiento. Manifest/start preceden store nuevo
y lectura pesada. Fallos cobran elapsed; crashes sin finish consumen reserva.
La presencia de attempt0025, root o cualquiera de los destinos nuevos impide
otro lanzamiento, también ante fallo del constructor. No editar globals de
operadores anteriores. Sólo reutilizar helpers puros con source pin: por
allowlist exacta approved_source/current_checker/fixed_precommit e
historical_precommit. validate_history/start/execute/close/publicación y reloj
son locales: el gate anterior sólo modela tail0023 y dos PRECOMMIT.

Comparar canónicamente los tres PRECOMMIT para cuts, terminal_receipts, freeze,
seal, contract, report_result_ref, preflight y plan. Volver a comparar la
admisión vigente contra cada corte dentro del intento, antes de publicar el
nuevo PRECOMMIT. Conservar acceso a resultados declarado true, sin fingir
nueva ceguera. La igualdad de27+16 conteos no reemplaza igualdad de identidades.

No repetir entrenamiento, fitting, forwards, observaciones, evaluación ni
informe. El único trabajo de campaña pendiente de esta entrada es el VERIFY
íntegro ya definido, sin convertir etapas atravesadas en certificados parciales.

## Antes de ejecutar

Revisión independiente del fix/namespaces y del camino concreto. Fixtures con
stores/budgets reales: ocho casos CPU/CUDA nominales sin ejecución CUDA,
alteración rechazada, historia y selección mutadas con iguales conteos,
presupuesto insuficiente, constructor fallido, no-retry y ausencia de writes
bajo nombres viejos. Publicar código revisado antes de una sola corrida y
seguir su handle hasta terminal. Un eventual COMPLETE no cierra por sí solo
informe humano, auditorías, documentación/wiki ni el goal completo.
