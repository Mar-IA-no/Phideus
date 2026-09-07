# R569 — Auditoría independiente del plan del paquete físico set-valued

**Fecha:** 2026-09-07  
**Commit auditado:** `691b3f2b668bb460a665fc086b71b689de1d8a07`  
**Objeto:** `experiments/geometria_proporcional/PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md`  
**Régimen:** lectura completa y probes read-only CPU; sin consultar, inicializar ni usar GPU/CUDA  
**Antecedentes contrastados:** runner y checker cerrados por R564, sus artefactos primario/replay, fuentes históricas abiertas W54/W59 y patrones físicos de Ola 59

## Veredicto

**REVISE — 2 HIGH / 5 MEDIUM / 1 LOW.**

El plan mejora sustancialmente la separación lógica de R564: introduce un cuarto rol disjunto, divide propuesta, evaluación, freeze y aplicación en procesos distintos, conserva estados portables, exige replay desde vacío y mantiene correctamente el estatuto `OPENED_DATA_PHYSICAL_PREFLIGHT`. Sin embargo, todavía no determina una implementación segura y unívoca. Los dos defectos altos están en la frontera prospectiva: la ruta `FRESH_PROSPECTIVE` puede quedar aceptable antes de que exista una autoridad criptográfica y cronológica congelada, y los schemas no autentican relaciones semánticas entre arrays que sí afectan folds, modelos y features. Cinco ambigüedades adicionales afectan el selector, la referencia R564, restart/journals, evidencia durable y runtime staging.

El preflight abierto no adquiere autoridad prospectiva por pasar estos checks. Este veredicto es técnico; no promueve una arquitectura ni constituye una decisión científica `GO/NO-GO`.

## Findings

### H1 — `FRESH_PROSPECTIVE` queda consumible antes de existir una autoridad prospectiva completa

**Evidencia.** El plan afirma que el binario podrá validar y consumir un paquete `FRESH_PROSPECTIVE` si presenta seis bundles, un commitment, un freeze y una atestación detached (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:145-167`). Pero no fija:

- path y SHA-256 de la clave pública confiable;
- algoritmo, representación canónica y payload exacto de firma;
- compromiso pre-generación/pre-oráculo;
- evidencia de que los logits de los cuatro roles fueron producidos antes de materializar truth y sin selección adaptativa;
- identidad y hashes de generador, inference code, checkpoints y software productor;
- conteos mínimos, stopping rule y contrato prospectivo que el propio plan difiere al goal siguiente (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:726-733`).

La máquina tampoco distingue una salida futura en su resultado: `EVALUATION_TRUTH` declara **siempre** `status = OPENED_DATA_PHYSICAL_PREFLIGHT` y `prospective_evidence = false` (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:331-346`). Por tanto, el texto combina dos contratos incompatibles: la ruta fresca sería consumible en v1, pero los elementos que le darían autoridad y su status quedan deliberadamente para después.

Ola 59 muestra por qué esos campos no son ornamentales. Su plan exige generación sellada, acciones congeladas antes de truth y una cadena de fases físicamente observable (`WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md:256-305`); la implementación liga una clave pública confiable por path, verifica su fingerprint y autentica un payload exacto que declara blind inference antes de cualquier oracle (`run_wave59_hgb_guard_bracket.py:70-73`, `:552-665`). La config histórica liga además el SHA-256 de esa clave (`configs/wave59_fresh_hgb_guard_bracket.json:321`). El nuevo plan no hereda un equivalente.

**Impacto.** Un preparador puede autofirmar con una clave aportada por el propio paquete, firmar sólo hashes post hoc o producir logits después de ver truth. Cumpliría la lista nominal sin acreditar virginidad, cronología ni independencia. La frase que prohíbe sustituir la firma fresca por evidencia abierta (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:165-167`) no basta para que `P1_AUTHORITY`/`P2_PREPARATION` lo detecten.

**Corrección requerida.** Para este goal, hacer que `FRESH_PROSPECTIVE` sea una variante de schema reconocida pero **terminalmente rechazada antes de staging**, con reason code propio, hasta que el gate de readiness congele el protocolo faltante. Alternativamente, si se desea ejecutarla ya, fijar ahora trust root, algoritmo, payload canónico, pre-generation freeze, identidad/hash de generador e inference/checkpoints, orden pre-oráculo, conteos, stopping rule, inventario físico y cadena de receipts. En ambos casos, ningún receipt, status ni flag derivado del modo abierto puede satisfacer una condición fresca.

### H2 — Los seis schemas fijan campos y shapes, pero no sus invariantes semánticos cruzados

**Evidencia.** La sección de schemas exige keys exactas, arrays no-object, finitud, targets no vacíos, identidad de `pair_token` en los pares y disjunción entre roles (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:169-227`). No exige, entre otras relaciones:

- `cardinality == target.sum(axis=1)` y rango `1..4`;
- `ensemble_logits == per_seed_logits.mean(axis=0)`;
- shape/dtype exactos de cada array más allá de la notación descriptiva;
- vocabulario y derivación admitidos para `design_stratum`;
- asociación del eje `per_seed_logits[3,...]` con el roster/checkpoints congelados;
- consistencia de `cardinality` público con el target de su truth companion.

Estas relaciones no son decorativas. `cardinality` interviene en folds y derangements del posterior (`run_proportional_set_valued_native_preflight.py:349-366`), y los logits por checkpoint alimentan features y sensitivities. R564 podía reconstruir cada bundle byte a byte desde fuentes históricas ligadas (`check_proportional_set_valued_native_preflight.py:721-775`); un paquete prospectivo externo no dispone de esa referencia de contenido y necesita validación semántica propia.

Un probe CPU read-only confirmó que las fuentes actuales sí cumplen `cardinality == sum(target)`, rango `1..4`, `ensemble_logits == mean(per_seed_logits)` bit a bit y targets no vacíos en W54 `fit_select_bundle.npz` y W59 `gate_fit_bundle.npz`/`gate_select_truth_bundle.npz`. El defecto está en el contrato futuro, no en esos artefactos abiertos.

**Impacto.** Un paquete firmado pero internamente inconsistente podría alterar estratificación, folds, control shuffled, features o sensitivities y aun pasar el P2 descrito. La firma sólo autenticaría bytes defectuosos; no les daría validez científica.

**Corrección requerida.** Convertir todas las relaciones cross-array en predicados exactos de schema, con dtypes, shapes, vocabularios, roster/axis order, tolerancias —preferiblemente igualdad bit a bit cuando la derivación lo permite— y reason codes. El checker debe recomponerlas sin confiar en el preparador ni en la atestación.

### M1 — El contrato del selector no determina `authorized_rows` y deja abierta metadata outcome-aware

**Evidencia.** `SELECTION_PROPOSE` dice publicar metadata, acciones y masks de override (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:276-287`). Sin embargo, `SELECTION_EVALUATE` recibe sólo truth, acciones candidatas, HARD y metadata, no masks ni scores (`:289-303`), y la matriz de stages vuelve a omitir los masks (`:354-362`). La clave R564 que se pretende conservar incluye `-authorized_rows`, donde `authorized_rows` cuenta overrides efectivos (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:233-245`; `proportional_set_valued_native.py:1101-1149`). El evaluador no tiene una regla congelada para obtener ese valor.

Hay una segunda indeterminación ligada. En R564, la metadata de cada candidata contiene cuantiles **y thresholds numéricos**, y el objeto `selected` fusiona esa metadata con las métricas outcome-aware (`proportional_set_valued_native.py:1061-1098`, `:1151-1160`). El nuevo plan permite que el evaluador que vio truth devuelva “metadata seleccionada” (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:295-303`) y luego dice que `SELECTION_FREEZE` deriva thresholds de nuevo sin truth (`:305-315`), pero no fija qué campos puede devolver el evaluador ni cuál copia prevalece.

**Impacto.** Dos implementaciones conformes pueden: stagear masks, derivarlos como `candidate_actions != hard_actions`, confiar en un conteo dentro de metadata o no disponer de `authorized_rows`. Peor, una implementación puede transportar thresholds escritos por el proceso outcome-aware y usarlos después. La reaplicación bit-exacta reduce el riesgo, pero el schema actual no obliga a ignorar y contrastar esos campos.

**Corrección requerida.** Elegir una sola ruta. La opción más austera es que el evaluator reciba actions, override masks y hashes ya congelados; calcule la clave; y publique únicamente `selected_index`, métricas seleccionadas y hashes. `SELECTION_FREEZE` debe recuperar cuantiles/thresholds desde el candidate freeze del proposer, nunca desde la decisión outcome-aware, y recomponer masks/actions bit a bit. Si se deriva `authorized_rows` como `actions != HARD`, congelar y probar la equivalencia con `override` bajo el invariante `disagreement = HARD != posterior_action`.

### M2 — La paridad con R564 no liga la referencia ni fija comparadores por campo

**Evidencia.** El source freeze propuesto liga plan, auditoría del plan, config, código y fuentes históricas, pero no enumera los manifests/artifact roots primario y replay de R564 (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:412-436`). No obstante, la aceptación exige comparar contra R564 estados, 344 candidatas, scores, acciones y thresholds, admitiendo que sean “numérica o bit-exactamente equivalentes” sin asignar comparator/tolerancia por artefacto (`:552-557`).

Los roots vigentes son mutables y no versionados. En este corte sus manifests tienen SHA-256 `a0401834c3958680ef687ad264b8b56a017a8996eaade904873b342319528a39` (primario) y `2752c103f80ae8655747cf92709fe9462ef753dd282c20f449694a90b6e44039` (replay); R564 documenta 32 archivos comparables y cero diferencias (`564_proportional_set_valued_runner_final_reaudit.md:43-71`). El plan no congela esas identidades como referencia de paridad.

**Impacto.** La referencia puede cambiar entre implementación y auditoría, o una tolerancia elegida por el implementador puede ocultar una deriva que cambie un threshold/tie-break. El checker podría dar PASS contra un R564 distinto del auditado.

**Corrección requerida.** Ligar en source freeze ambos artifact manifests —o un manifest de referencia inmutable con todos los digests relevantes—, el SHA-256 de R564 y el commit auditado. Declarar por path/key si la comparación es byte-exacta, array-exacta o `rtol=0/atol=X`; thresholds, metadata, índices y acciones deben ser exactos.

### M3 — Journal y recovery no tienen aún un commit protocol durable ni un layout recuperable unívoco

**Evidencia.** El plan exige journals durables, transiciones monotónicas, resume desde el último journal y archivo recuperable si aparece un output sin journal (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:100-115`, `:485-528`). Pero no define:

- escritura de journal mediante temporal, `fsync`, rename atómico y `fsync` del directorio;
- que el `.pending` promovido viva en el mismo filesystem que su destino;
- vocabulario/orden exacto de `maximum_truth_materialized`;
- journal y recovery de la propia preparación `INITIALIZED → PREPARED`;
- inventario esperado para cada terminal `NOT_EVALUABLE_*`;
- ubicación de la preparación inmutable cuando un run completo se archiva y se exige recomenzar desde output vacío.

La ambigüedad final es especialmente material para un paquete fresco: el runner analítico no puede regenerar el draw ni abrir generation escrow (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:145-167`, `:438-464`), de modo que debe existir un input package externo, inmutable y separable de los output roots primario/replay. El árbol actual mezcla `preparation/`, `prepared/` y outputs bajo una única presentación (`:438-483`). Ola 59 implementa el patrón durable con temporal, `fsync`, `os.replace` y `fsync` del padre (`run_wave59_hgb_guard_bracket.py:192-235`) y promueve desde un sibling `.pending` (`:862-880`).

**Impacto.** Un crash puede dejar un journal visible pero no durable, una promoción cross-filesystem no atómica o un fresh input archivado junto con el output que debía poder reconstruirse. Dos coordinadores conformes pueden tomar decisiones distintas ante el mismo estado parcial.

**Corrección requerida.** Separar explícitamente `immutable_input_package`, `primary_output` y `replay_output`; ambos outputs referencian por hash el mismo input. Congelar el algoritmo de commit durable, los enums de truth, tablas `state × required artifacts × next states`, el tratamiento de PREPARED parcial y todos los crash points. Recovery nunca reconstruye fresh input desde escrow ni desde bytes no autenticados.

### M4 — La aceptación de mutaciones y recovery carece de receipts durables con paths cerrados

**Evidencia.** El plan exige fallos inyectados, mutaciones, checks de primaria/replay y auditoría final (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:526-528`, `:591-647`, `:702-724`). El manifest final contempla categorías genéricas, pero no nombra artefactos canónicos para resultados de unit tests, checker primario/replay, mutation suite ni recovery campaign (`:466-483`). `P12_RESTART_REPLAY` no puede reconstruir desde un único run que efectivamente se inyectaron todos los crash points; esos intentos ocurren fuera del artefacto validado (`:559-589`).

**Impacto.** La condición “todas las mutaciones detectadas” puede descansar sólo en stdout o en una afirmación documental. Se pierde trazabilidad de conteos, reason codes, comandos, versiones, budgets, hashes de inputs y correspondencia entre cada crash point y su resultado recuperado.

**Corrección requerida.** Fijar schemas y paths para `unit_test_receipt`, `primary_check_receipt`, `replay_check_receipt`, `mutation_receipt` y `recovery_receipt`; incluir caso, expected/observed reason code, exit, hashes, wall/RSS y referencia a los outputs/failure inventories. Ligarlos en un evidence manifest no autorreferencial y preservarlos antes de la auditoría final.

### M5 — La frontera de runtime del worker no queda materializada en la allowlist

**Evidencia.** La matriz afirma que cada stage contiene `phase_request.json` y los inputs enumerados, y que el worker no recibe root ni paths originales (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:348-367`). La sección de permisos fija `setpriv`, UID/GID, grupos, NNP y threads (`:369-390`), pero no define de dónde se ejecuta/importa el worker, qué runtime source tree se copia, qué `PYTHONPATH`/cwd/env recibe ni cómo el receipt demuestra que ejecutó exactamente los blobs ligados. También sólo exige capabilities **efectivas** cero, pese a describir workers “sin capacidades” (`:100-115`, `:388-390`).

Ola 59 materializa un runtime mínimo, copia y hashea módulos/worker, aplica modos cerrados y después lanza con env explícito (`run_wave59_hgb_guard_bracket.py:706-722`, `:761-858`). Ese patrón no aparece aún como objeto del nuevo stage ni en P3.

**Impacto.** El proceso puede resolver imports desde el worktree, user/site packages o una ruta no ligada aunque el archivo principal tenga hash correcto. Además, un receipt self-reported sobre capability efectiva no prueba por sí solo el set completo ni el runtime realmente cargado.

**Corrección requerida.** Congelar un `runtime/` staged por fase o común, con inventario exacto, hashes, modos, cwd y env mínimo; deshabilitar user-site/import paths no allowlisted; registrar módulos cargados relevantes y verificarlos desde el coordinador/checker. Definir y comprobar effective/permitted/inheritable/ambient/bounding capabilities, o acotar honestamente el claim a effective caps + NNP.

### L1 — Los budgets no cubren toda la campaña que la aceptación obliga a ejecutar

**Evidencia.** Los límites fijan primaria+replay, checker+mutaciones, RSS y disco de los dos runs (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:649-669`). El orden de aceptación agrega unit tests, fixtures de permisos y una campaña de recovery con fallos tras cada promoción y en promoción/journal (`:526-528`, `:702-713`), pero no les asigna wall/RSS total. El disco de `failures/` tampoco entra en los 512 MiB aunque los intentos fallidos deben preservarse.

**Impacto.** El riesgo práctico es bajo con el tamaño actual —primario+replay R564 suman 26.290.616 bytes y el archivo mayor mide 7.549.459 bytes—, pero `P15_COST` no tiene una cobertura cerrada y una campaña de fallos puede pasar aun excediendo costo o disco no contabilizado.

**Corrección requerida.** Añadir límites agregados para unit/permission/recovery suites, disco temporal y `failures/`; declarar qué se mide por proceso y por campaña y qué cleanup conserva receipts sin acumular copias regenerables.

## Aspectos verificados que sí están bien resueltos

- El commit `691b3f2` es plan-only: agrega exactamente el archivo auditado. Su SHA-256 es `7f25209c9f4cfd6fc2003ee6c20763424752f50f03fc867842e9b1fe5c3aca92`.
- R564 cerró el antecedente real: 14/14 checks en primario y replay, 32 archivos comparables sin diferencias, 54/54 mutaciones y ausencia de findings (`564_proportional_set_valued_runner_final_reaudit.md:8-14`, `:43-93`).
- El diagnóstico del defecto R564 es exacto: `select_and_apply()` recibe simultáneamente public y truth, y la evaluación vuelve sobre esa población (`run_proportional_set_valued_native_preflight.py:581-630`, `:1372-1394`).
- El bundle W54 observado contiene `192 calibration_fit + 192 decision_select`. Los cuatro roles activos propuestos tienen `192/768/768/192` tokens únicos y son disjuntos dos a dos. El `decision_select` W54 tampoco intersecta con el monitor histórico W59. Esto acredita identidad y topología del fixture, no virginidad científica.
- La secuencia `PROPOSE → EVALUATE → FREEZE → APPLY → TRUTH` elimina la reutilización directa de selection para evaluación y, una vez cerrada M1, permite que ningún proceso outcome-aware regenere acciones (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:229-346`).
- La serialización canónica, las allowlists por fase, los outputs privados root-only, el replay desde vacío y la independencia declarada del checker están bien orientados (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:348-410`, `:530-589`).
- El plan conserva correctamente `NOT_EVALUABLE_CONTROL_SUPPORT` como estado de celda y no lo convierte en refutación (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:305-315`).
- La frontera de claims es correcta para la única corrida autorizada ahora: no fresh draw, no monitor/lockbox nuevo, no promoción y decisión científica nula (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:31-56`, `:715-736`).

## Comprobaciones realizadas

1. Lectura completa de las 736 líneas del plan vigente y verificación del diff/parent del commit.
2. Lectura de R564 y contraste focal con runner, checker, config y core NumPy vigentes.
3. Inspección de schemas, roles, conteos, keys y tokens de los artefactos históricos ligados; comprobación CPU de disjunción y relaciones cross-array.
4. Contraste con separación física, atestación, staging, permisos, promoción y journals de Ola 59.
5. Revisión de autoridad Git/source-freeze, fases y handoffs, replay, restart, mutation coverage, artefactos y costos.

No se ejecutó el futuro runner —todavía no existe—, no se editó plan ni implementación y no se consultó ni utilizó GPU/CUDA.

## Condición para reauditoría

La siguiente versión debe cerrar H1 y H2 antes de implementar. En particular, debe elegir entre hard-disable prospectivo o autoridad prospectiva completa; congelar invariantes semánticos de bundles; hacer unívoco el transporte de `authorized_rows` y thresholds; ligar la referencia R564; definir commit/recovery durable y layout input/output; materializar receipts de suites; y cerrar el runtime staging. L1 puede resolverse en la misma pasada sin alterar la pregunta científica.
