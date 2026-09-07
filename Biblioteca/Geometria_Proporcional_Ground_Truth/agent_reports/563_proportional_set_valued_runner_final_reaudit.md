# R563 — Reauditoría final independiente del runner set-valued nativo

Fecha: 2026-09-07  
Base auditada: `366bbc4d29e0aaee88c30df2b2f82f19a42fe54e`  
Alcance: resolución de R562 F-01…F-06, estado vigente completo de plan, config, primitives, runner, checker, tests, artefacto primario y replay  
Régimen: sólo CPU, `CUDA_VISIBLE_DEVICES=''`, un thread BLAS/OpenMP; no se consultó ni inicializó GPU/CUDA  
Escrituras: únicamente este informe; los probes adversariales usaron copias temporales autodescartables

## Dictamen

**PASS_WITH_LOW — 0 HIGH, 0 MEDIUM, 1 LOW.**

Los seis findings de R562 están resueltos en el estado vigente. Los dos artefactos canónicos pasan los 14 predicados; la campaña completa pasa 53/53 mutaciones con checker primario y replay incluidos y presupuesto agregado válido; nueve probes negativos nuevos dirigidos a F-01…F-04 fueron rechazados por el reason code esperado; la fixture numérica de `advantage` confirma que ya no existe clipping. El único hallazgo residual es que P12 todavía no verifica la representación canónica de los archivos JSON distintos del reporte, aunque todos los JSON canónicos actuales sí tienen el encoding exigido y el replay es byte-exacto.

Este dictamen es exclusivamente técnico y metodológico sobre el preflight. No constituye `GO/NO-GO`, promoción arquitectónica ni interpretación prospectiva.

## Estado leído y verificaciones positivas

- Se releyeron completos los archivos vigentes exigidos y todos los archivos no archivados de ambos artefactos. Cada árbol contiene 35 archivos, 17 NPZ, 585 arrays y `35,701,634` valores.
- Todos los NPZ actuales tienen members ordenados, timestamp `1980-01-01`, DEFLATE, atributos `0600` y arrays cargables con `allow_pickle=false`.
- Todos los JSON actuales coinciden byte por byte con UTF-8, `ensure_ascii=false`, keys ordenadas, separadores compactos y newline final.
- Los `1,166,958` no finitos por artefacto siguen limitados a `NaN` intencionales de scores fuera del soporte `disagreement`.
- Los 32 archivos comparables entre primario y replay tienen inventario idéntico y cero diferencias SHA-256.
- Runtime primario: `28.14092292636633 s`, RSS `984,887,296` bytes.
- Runtime replay: `28.498577643185854 s`, RSS `984,788,992` bytes.
- Primario + replay: `56.63950056955218 s`, `within_hard_budget:true`.
- Unit tests: **9/9 PASS**, `0.775 s` internos.
- Checker primario: **14/14 PASS**, `3.7753884941339493 s`.
- Checker replay contra primario: **14/14 PASS**, `3.8728136643767357 s`.
- Campaña completa: **53/53 mutaciones PASS**; los dos valid checks internos también fueron PASS; el proceso terminó exit 0 y declaró `within_budget:true`, lo que requiere simultáneamente wall agregado `<=900 s`, RSS hijo máximo `<=1.5 GiB` y ambos checkers válidos (`tests/run_proportional_set_valued_mutations.py:194-246`).

## Revisión dirigida de R562

### F-01 HIGH anterior — RESUELTO

El checker ahora reconstruye los cuatro bundles desde las fuentes congeladas y exige igualdad de keys, dtype, shape y contenido, además de conteos, unicidad y disjunción (`check_proportional_set_valued_native_preflight.py:702-775`). MARGINAL se reajusta independientemente desde posterior-fit (`:777-809`). JOINT confronta folds, NLL, Brier, `fold_theta`, objetivo, gradiente, iteraciones, evaluaciones, theta final, interacciones y metadata de refit (`:811-893`). P12 verifica metadata canónica de cada NPZ (`:1447-1483`).

Probes nuevos:

- corrupción canónica de `prepared/decision_select_public.npz:design_stratum[0]` → `PHASE_BUNDLE_INVALID`;
- corrupción canónica de `posterior_fit/oof_arrays.npz:joint_target_shuffled__fold_objective[0,1]` → `JOINT_RECIPE_INVALID`;
- cambio exclusivo de `external_attr` en un member de `prepared/decision_select_truth.npz` → `RAW_OR_REPLAY_INVALID`.

Los tres probes terminaron exit 1 con el reason code esperado aun después de refrescar el manifest.

### F-02 MEDIUM anterior — RESUELTO

P11 recompone el inventario raw completo, métricas de controles, orden y digest de bootstrap; luego reaplica los tres checkpoints a ambos posteriors/readers, confronta exactamente las 36 arrays y las 48 filas de resumen por cardinalidad, y reconstruye las seis comparaciones de duplicación (`check_proportional_set_valued_native_preflight.py:1258-1445`).

Probes nuevos:

- `diagnostic_metrics.json:checkpoint_sensitivity[0].mean_regret += 0.1` → `CELL_ESTIMAND_MISMATCH`;
- `cell_duplications.json:comparisons[1].action_position_equal_fraction = 0.123` → `CELL_ESTIMAND_MISMATCH`.

Ambos fueron rechazados con manifest refrescado.

### F-03 MEDIUM anterior — RESUELTO

P10 recalcula los digests semánticos del mapping y del triplete transportado, reconstruye `permutable`, singletons, Hamming por estrato, Hamming total y el documento completo de diagnostics; la diversidad se decide sobre los digests recalculados (`check_proportional_set_valued_native_preflight.py:1128-1256`).

Probes nuevos:

- sustitución de `joint.controls[2].diagnostics.target_triplet_sha256` → `MATCHED_CONTROL_INVALID`;
- incremento de `marginal.controls[3].diagnostics.maximum_hamming` → `MATCHED_CONTROL_INVALID`.

Ambos fueron rechazados con manifest refrescado.

### F-04 MEDIUM anterior — RESUELTO

P13 ya no depende de una blacklist léxica: vuelve a renderizar canónicamente el reporte desde `estimand_table`, exige igualdad textual completa, comprueba status estructurado en cuatro documentos y exige `truth_keys_received_by_applier:[]` (`check_proportional_set_valued_native_preflight.py:1485-1506`).

Probes nuevos:

- agregado `Se recomienda promover JOINT como arquitectura.` al reporte → `CLAIM_BOUNDARY_INVALID`;
- `estimand_table.json:status = FRESH_DRAW_AUTHORIZED` → `CLAIM_BOUNDARY_INVALID`.

Ambos fueron rechazados con manifest refrescado.

### F-05 MEDIUM anterior — RESUELTO para el contrato de cierre auditado

La suite vigente contiene 53 casos (`tests/run_proportional_set_valued_mutations.py:113-179`) y cubre explícitamente los bypass de R562: procedencia preparada, fold state, NPZ no canónico, sensibilidad, duplicación, digest declarado, promoción equivalente y recibo de truth en applier. También amplía receta, guards, selección, matching, raw y soporte común. La corrida completa dio 53/53 y los reason codes esperados.

El harness mide desde antes de los valid checks hasta después de la última mutación, incluye checker primario y replay, recoge RSS máximo de procesos hijos y sólo retorna cero si casos, checkers y límites agregado/RSS son válidos (`tests/run_proportional_set_valued_mutations.py:194-246`). La ejecución completa observada retornó cero y `within_budget:true`.

### F-06 LOW anterior — RESUELTO

La primitive conserva `advantage = hard_risk - minimum_risk` después del guard de tolerancia, sin `np.maximum` (`proportional_set_valued_native.py:489-494`). El checker aplica la misma resta sin clipping pero con guard independiente (`check_proportional_set_valued_native_preflight.py:320-323`). La nueva fixture fuerza `-5e-13`, comprueba que el valor permanezca negativo y que la feature 0 sea byte-idéntica a `advantage` (`tests/test_proportional_set_valued_native.py:127-150`). Pasó tanto individualmente como dentro de 9/9.

## Finding residual

### R563-F01 — LOW — P12 no exige encoding canónico para JSON no-REPORT

**Evidencia.** P12 valida inventario, SHA-256/tamaño autorreferentes en el manifest y canonicalidad de NPZ, pero no compara los bytes de cada `.json` con una serialización canónica (`check_proportional_set_valued_native_preflight.py:1447-1483`). En una copia temporal se reescribió `source_bindings.json` con indentación de dos espacios, preservando exactamente su semántica, y se refrescó su entrada de manifest. El checker terminó exit 0, 14/14 PASS. El reporte no tiene este bypass porque P13 exige render exacto.

**Impacto.** No altera valores, fases, estimandos ni claims. El replay con referencia detectaría una diferencia unilateral, y los JSON de ambos artefactos actuales sí son canónicos. La brecha sólo permite que un artefacto primario aislado conserve semántica válida con una representación JSON fuera del formato congelado.

**Corrección sugerida.** Para cada `.json`, cargar con rechazo de constantes no finitas y exigir que los bytes originales sean iguales a `json.dumps(..., sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"`. Añadir un caso `json_noncanonical` a la suite.

## Cierre de la reauditoría

No quedan findings HIGH o MEDIUM abiertos. Los requisitos solicitados para esta reauditoría pudieron ejecutarse completos. La única deuda LOW no cambia la validez numérica ni la separación de claims del artefacto vigente y no contradice la igualdad de replay observada.
