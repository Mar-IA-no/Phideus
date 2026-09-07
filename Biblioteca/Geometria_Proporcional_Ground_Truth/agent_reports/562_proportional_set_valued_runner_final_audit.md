# R562 — Auditoría final independiente del runner set-valued nativo

Fecha: 2026-09-07  
Alcance: plan, config, primitives, runner, checker, tests, artefacto primario y replay de `proportional_set_valued_native_preflight_v1`  
Régimen: sólo CPU, `CUDA_VISIBLE_DEVICES=''`, un thread BLAS/OpenMP, sin consulta ni inicialización de GPU/CUDA  
Escrituras: únicamente este informe; los probes negativos se ejecutaron sobre copias temporales autodescartables

## Dictamen de auditoría

**REVISE — auditoría final no limpia.** Conteo: **1 HIGH, 4 MEDIUM, 1 LOW**.

No encontré corrupción efectiva en los dos artefactos canónicos examinados: los cuatro bundles preparados coinciden exactamente con sus fuentes históricas abiertas; el refit MARGINAL real y shuffled reproduce coeficiente, intercept y `n_iter`; una muestra independiente de 37 tokens reconstruida desde raw dio error absoluto máximo `0.0`; sensibilidad, duplicaciones y digests de controles también coincidieron exactamente en una recomputación adicional completa. Sin embargo, el checker certifica superficies que en realidad no verifica y seis mutaciones independientes de un solo elemento atravesaron sus 14 predicados con `PASS`. Esto incumple el contrato de detección y la suite mínima fijados por el plan.

Este dictamen se refiere a validez de implementación y cumplimiento del plan. No es un `GO/NO-GO`, no promueve arquitectura y no interpreta el diagnóstico histórico como evidencia prospectiva.

## Verificaciones ejecutadas

- Lectura completa de los archivos exigidos: `AGENTS.md`, `CODEX.md`, plan, config, módulo de primitives, runner, checker, dos archivos de tests y los 35 archivos no archivados de cada artefacto.
- Inventario por artefacto: 35 archivos; 17 NPZ; 585 arrays; `35,701,634` valores. Los 34 archivos distintos del manifest están inventariados. Los `1,166,958` valores no finitos son exclusivamente `NaN` de scores fuera de `disagreement`; sus máscaras coinciden exactamente con el complemento del soporte activo.
- Unit tests: **7/7 PASS** en `0.795 s` (`1.281 s` de proceso envolvente).
- Checker primario sin referencia: **14/14 PASS**, `3.3744732327759266 s`.
- Checker replay con referencia al primario: **14/14 PASS**, `3.4323733374476433 s`.
- Suite implementada de mutaciones: **28/28 PASS**.
- Replay: 32 archivos comparables, inventario idéntico y cero diferencias SHA-256. Receipt: `byte_exact:true`; tiempo primario+replay `56.25880528986454 s`.
- Costo de los runners: primario `28.156469065696 s`, replay `28.10233622416854 s`; RSS pico `982,118,400` y `982,102,016` bytes. Quedan debajo de `1800 s` y `1.5 GiB`.
- Fases: 192 tokens posterior-fit, 768 policy-fit y 768 decision-select; cero intersecciones entre las tres poblaciones. Public y truth de selección tienen identidad alineada y el bundle público carece de campos target/truth.
- Recomputación independiente desde fuentes: `posterior_fit`, `policy_fit`, `decision_select_public` y `decision_select_truth` coinciden array por array con su extracción declarada.
- Refit MARGINAL independiente: real `|delta coef|=0`, `|delta intercept|=0`, `n_iter=5/5`; shuffled `0`, `0`, `8/8`.
- Muestra independiente desde raw: seed `90562026`, 37 tokens, digest de tokens `ea7735ae3de34a4b51a76e8a98a7524684a9152aa91fce99f0c3c35a60f6614a`; masas, métricas set-valued, acciones y métricas HARD/CONTEXTUAL para MARGINAL/JOINT dieron error máximo `0.0` contra `diagnostic_arrays.npz`.
- Recomputación adicional de superficies no cubiertas por el checker: las 36 arrays de sensibilidad, 48 filas agregadas por checkpoint/cardinalidad, seis comparaciones de duplicación y los diez pares de digests de controles coinciden exactamente con el artefacto actual.

Nota operativa: la dirección canónica de comparación es replay como artefacto auditado y primario como `--reference`. La invocación inversa falla P12 porque el receipt primario tiene `byte_exact:null`; no contradice la igualdad observada.

## Findings

### F-01 — HIGH — El checker no establece procedencia de los bundles preparados ni integridad semántica completa del raw

**Evidencia.** El plan exige recomputación independiente y detección de raw coherentemente corrupto (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:434-441`), además de integridad raw/replay en P12 (`:455-456`). P2 sólo comprueba conjuntos de keys, disjunción de tokens y ausencia nominal de truth en public (`check_proportional_set_valued_native_preflight.py:663-688`); no reextrae ni compara los cuatro bundles contra las fuentes ligadas, no comprueba conteos/roles/uniqueness completos y no autentica cada array de preparación. P12 sólo verifica que los hashes y tamaños actuales coincidan con un manifest que se puede regenerar junto con la mutación (`check_proportional_set_valued_native_preflight.py:1121-1139`). P4 recompone NLL/Brier y theta final, pero no confronta `fold_theta`, objetivos, gradientes, iteraciones y evaluaciones preservadas (`check_proportional_set_valued_native_preflight.py:706-752`). Tampoco valida la serialización canónica de los NPZ.

Dos probes independientes de un solo cambio, con manifest actualizado, terminaron `status:PASS`, 14/14:

1. `prepared/posterior_fit_truth.npz:cluster_id[0]` fue sustituido por un valor corrupto, rompiendo la igualdad con la fuente congelada.
2. `posterior_fit/oof_arrays.npz:joint_real__fold_theta[0,0,0] += 0.125`; la reescritura además usó el ZIP no canónico de `numpy.savez_compressed`.

**Impacto.** Una copia autoconsistente con el manifest puede ser certificada aun cuando ya no representa las fuentes ligadas o cuando un estado raw preservado fue alterado. El replay demuestra determinismo entre dos ejecuciones, pero por sí solo no prueba procedencia ni exactitud de todas las superficies. Esto contradice una propiedad central del checker y vuelve insuficiente su `PASS` como cierre independiente.

**Corrección requerida.** En P2, reextraer desde cada fuente congelada los cuatro bundles y exigir igualdad de inventario, dtype, shape y bytes por key, junto con filas, roles, unicidad y disjunción. En P3/P4/P5 y P12, verificar todos los estados/arrays preservados contra la recomputación, incluidos metadata de optimización y mapas. Validar también orden de keys, `allow_pickle=false`, timestamps ZIP fijos, compresión y atributos canónicos. Agregar mutaciones negativas específicas para procedencia, cada raw auxiliar y serialización no canónica.

**Estado del artefacto actual.** La recomputación adicional de esta auditoría encontró los cuatro bundles actuales exactamente iguales a sus fuentes y el `fold_theta` no mostró corrupción. El finding es una brecha de certificación reproducida, no una afirmación de corrupción actual.

### F-02 — MEDIUM — P11 no verifica el contenido de sensibilidad ni de duplicaciones

**Evidencia.** El contrato P11 exige sensitivities y duplicaciones exactas (`PLAN...:455`). La implementación sólo comprueba que el NPZ tenga 36 keys y que el JSON tenga seis filas con un boolean global (`check_proportional_set_valued_native_preflight.py:1095-1100`). Dos probes, con manifest actualizado, pasaron 14/14:

- cambio de una acción en `evaluate_fixture/sensitivity_arrays.npz:checkpoint_17__marginal__hard__actions[0,0]`;
- inversión de `evaluate_fixture/cell_duplications.json:comparisons[0].actions_exact`.

**Impacto.** El checker puede certificar sensibilidad por checkpoint o declaraciones de duplicación incorrectas. No cambia los estimandos primarios, pero sí invalida dos capas de preservación y diagnóstico exigidas por el plan.

**Corrección requerida.** Reaplicar cada checkpoint a estados y thresholds congelados; confrontar exactamente las 36 arrays, las 48 filas por cardinalidad y sus métricas. Reconstruir las seis comparaciones de acciones y regret, incluidos `action_position_equal_fraction`, `actions_exact` y `regret_exact`. Añadir mutaciones de valor, key, cardinalidad y duplicación.

**Estado del artefacto actual.** La recomputación completa adicional dio error `0.0` en sensibilidad y coincidencia exacta en las seis duplicaciones.

### F-03 — MEDIUM — La diversidad de controles se decide con digests declarados, no recalculados

**Evidencia.** P10 exige cinco seeds, mapping exacto, soporte y matching (`PLAN...:454`). El checker recompone los mappings y targets transportados, pero agrega a sets los strings `mapping_sha256` y `target_triplet_sha256` tomados de `states.json` (`check_proportional_set_valued_native_preflight.py:939-971`) y sólo exige cardinalidad cinco (`:992-993`). Al reemplazar `policy_fit/states.json:marginal.controls[0].diagnostics.mapping_sha256` por 64 ceros y actualizar el manifest, el checker devolvió 14/14 PASS.

**Impacto.** Los digests y el predicado de diversidad pueden divergir de los mapas/targets reales sin detección. En un caso de colisión real entre seeds, cinco strings distintos podrían sostener falsamente `CONTROL_DIVERSITY`.

**Corrección requerida.** Recalcular ambos digests desde tokens, mapping, máscara activa y tripletes transportados con la serialización exacta; comparar cada digest declarado y aplicar diversidad sobre los digests recalculados o directamente sobre los arrays. Verificar también `permutable_fraction`, singleton rows y resumen por estrato contra la recomputación.

**Estado del artefacto actual.** Los diez digests actuales fueron recalculados y coinciden; hay cinco mappings y cinco tripletes distintos por posterior.

### F-04 — MEDIUM — P13 es una blacklist literal que acepta promoción equivalente

**Evidencia.** El plan exige ausencia de promoción o decisión científica (`PLAN...:457`) y una mutación de cualquier frase/campo promocional (`:485-486`). P13 sólo busca cuatro substrings exactos en `REPORT.md`: `fresh draw authorized`, `architecture promoted`, `scientific go`, `scientific no-go` (`check_proportional_set_valued_native_preflight.py:1143-1154`). Tras añadir `The JOINT architecture is recommended for promotion.` y actualizar el manifest, el checker terminó 14/14 PASS.

**Impacto.** Un reporte puede emitir una recomendación/promoción materialmente equivalente y conservar certificación de frontera de claims. El riesgo es documental y de gobernanza, no numérico, pero contradice el alcance negativo central.

**Corrección requerida.** Hacer autoritativos campos estructurados allowlisted de status, autoridad, evidencia y siguiente paso; exigir que el reporte sea generado exactamente desde esos campos o comparar su hash con una regeneración canónica. Si se mantiene análisis textual, cubrir formas equivalentes y usarlo sólo como defensa adicional. Añadir mutaciones para recomendación, selección, autorización y decisión en inglés y español, tanto en REPORT como en JSON.

**Estado del artefacto actual.** La lectura completa del REPORT y JSON actuales no encontró promoción, autorización de draw ni decisión científica.

### F-05 — MEDIUM — La suite no implementa la cobertura mínima ni el presupuesto auxiliar agregado del plan

**Evidencia.** El plan enumera como mínimo mutaciones de leakage, receta, folds, shuffle, clipping, modelos, thresholds, controles, matching, estimandos, raw/replay, imports y claims (`PLAN...:460-486`) y fixtures de empates, portabilidad, assignment y common support (`:488-491`). La suite contiene 28 casos (`run_proportional_set_valued_mutations.py:97-127`) y los unit tests contienen siete métodos (`test_proportional_set_valued_native.py:29-154`). Faltan, entre otras exigencias literales, path prohibido, lectura de truth por applier, utilidad en posterior, fold dependiente de target, tie-break JOINT invertido, shuffle distinto por representación/cruce de estrato, clipping de advantage, pseudoinversa/intercept penalizado, signos/clases de guards, método/estrictitud/grid/HARD_ONLY, identidad/diversidad/soporte de controles, uso de target/otro `k`/sort/unión en matching, pérdida o utility divergente, bootstrap por policy-row/seed/support, NPZ no canónico, replay divergente, imports prohibidos y varias fixtures de empate/common support. Los seis probes de F-01 a F-04 demuestran que no es una omisión sólo nominal.

Además, el plan fija `900 s` para checker doble más mutaciones (`PLAN...:511-514`). P14 mide el tiempo de cada proceso checker aislado (`check_proportional_set_valued_native_preflight.py:1191-1234`); el runner de mutaciones no mide wall/RSS agregado ni produce por defecto un receipt canónico (`run_proportional_set_valued_mutations.py:130-169`).

**Impacto.** `28/28 PASS` no satisface “suite completa” ni prueba los invariantes negativos fijados. El costo observado fue holgadamente bajo, pero el límite agregado no está enforceado ni preservado.

**Corrección requerida.** Completar una matriz trazable plan→test con al menos un cambio único por subcaso obligatorio y reason code esperado; incluir los seis bypass reproducidos. Envolver checker primario, checker replay y mutaciones en un harness que mida wall agregado y RSS por proceso, falle sobre `900 s`/`1.5 GiB` y preserve un receipt inventariado.

### F-06 — LOW — `advantage` aplica clipping contrario al contrato literal

**Evidencia.** El plan define `advantage = hard_risk-minimum_risk` “sin clipping adicional” (`PLAN...:203-205`). La primitive calcula la resta, tolera hasta `-1e-12` y luego aplica `np.maximum(advantage, 0.0)` (`proportional_set_valued_native.py:489-494`). El checker reproduce el mismo clipping (`check_proportional_set_valued_native_preflight.py:320`), por lo que no es independiente frente a este desvío. La mutación explícita de clipping requerida por el plan tampoco existe.

**Impacto.** En los artefactos actuales no hay efecto: mínimo raw `0.0`, cero valores negativos, cero valores cambiados por clipping para MARGINAL y JOINT. Queda, no obstante, una divergencia contractual que podría ocultar diferencias numéricas pequeñas en otros fixtures.

**Corrección requerida.** Eliminar el clipping y conservar la resta exacta después del guard de tolerancia, o enmendar explícitamente el contrato si se desea canonizar la estabilización. El checker debe calcular la resta sin compartir el mismo desvío y la suite debe forzar un caso discriminante.

## Lectura metodológica del artefacto actual

Los resultados actuales permanecen correctamente rotulados como `OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC`. Ambos readers seleccionan la candidata 48; MARGINAL conserva `67/215` tokens en soporte común (`0.3116279069767442`) y JOINT `74/235` (`0.3148936170212766`), por debajo del mínimo 0.8, por lo que `READER_CONTROL` es `NOT_EVALUABLE` en ambos. `READER_WORST` es `ADVERSE` para ambos; `SET_JOINT_BRIER` queda `NOT_RESOLVED`; `JOINT_PATTERN_PRESENT` y ambos `CONTEXTUAL_PATTERN_PRESENT` son falsos. Esos estados son observaciones sobre datos abiertos y no autorizan extrapolación prospectiva.

## Cumplimiento de la propia auditoría

Se cumplieron lectura, reejecución de tests/checkers, replay, muestra independiente desde raw, revisión de costo, preservación, claims y archivo del informe. No se usó ni consultó GPU/CUDA, no se abrieron fuentes externas y no se lanzaron subagentes. No quedó ningún requisito de esta auditoría sin ejecutar.

