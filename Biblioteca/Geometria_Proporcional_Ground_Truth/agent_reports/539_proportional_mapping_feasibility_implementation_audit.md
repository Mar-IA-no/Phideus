# R539 — Auditoría técnica independiente de la implementación `MAPPING-FEASIBILITY`

## Dictamen técnico: REVISE

La implementación nominal ejecuta dos runs CPU deterministas dentro del
presupuesto y materializa correctamente una parte sustantiva de los adapters
set-valued y relacionales. Sin embargo, **no autoriza todavía ninguna de las
tres hojas semánticas**. El checker puede aceptar candidato, evidencia, árbol
público y replay alterados sin detectar la ruptura; los controles shuffled y
matched no llegan a ejecutarse como estimands; el factorial relacional sólo
recompone la primera vista de cada estado; el evaluator abre fuentes fuera de
sus fases; y un fallo presupuestario posterior conserva una adjudicación no
nula.

Conteo final: **5 HIGH, 2 MEDIUM y 1 LOW**. El artefacto observado
`BIFURCATE_NATIVE_CONTRASTS` no es válido mientras persistan estos findings.
Esto es un dictamen de implementación; no declara `GO/NO-GO`, no promueve una
arquitectura y no interpreta el resultado científico del gate.

## Identidad, independencia y alcance

- Configuración de la instancia: **GPT-5.6-Sol**, esfuerzo **high**.
- Target exacto: commit
  `4205a0407ff5d372e308a8fb307d34a0b0f1feef`.
- Parent directo: `85e8e7165a2257b46a8be4df0bebc2e045791414`.
- Tree del target: `3944babac8c1f4c31b088d3f94ae934cf30fcaa7`.
- Plan aprobado: commit
  `c1b9b45a8d015e6e147328e2e395402cd4930d5b`; SHA-256 vigente
  `ef61081fcbd6917368d62da065843e3b397abec0149dfed5c0cc99e77fbdc6d2`;
  blob Git `29969b938050a815bba5ee5dd7a1db6d0f23adcb`.
- Auditoría preexperimental R538 leída completa: SHA-256
  `243ea848f51cd76448aedc178796684bc8715a9162e1a44c5a612692f919fbb6`;
  blob Git `6d288eef49d817a3fa0b14cafa214e0b683d6a39`.
- Se leyeron completos el plan, R538 y los nueve archivos del commit. Se auditó
  el contenido vigente, que coincide con `4205a04` en esos nueve paths.
- No se abrió lockbox, `sealed_monitor_bundle.npz` ni monitor histórico. No se
  usó ni consultó GPU. No se modificaron código, plan ni datos canónicos.

### Pathset exacto y SHA-256

Digest del pathset ordenado, con newline por path:
`d1f384fbfbba038fcc730d4726756e6ae2ae320ff371af6fcd08c8ebe0b8287d`.
Digest del patch binario de `git show --format= --binary 4205a04`:
`fd0c60a62e243203809f9506234af944c6cce504368a9d4e6cbdc8bfba4cc7c0`.

| Path | SHA-256 |
|---|---|
| `experiments/geometria_proporcional/build_proportional_mapping_candidate.py` | `223edf5c5526ab835c02e3b556c85b16897490d2d661a52af78887630fbe39de` |
| `experiments/geometria_proporcional/check_proportional_mapping_feasibility.py` | `04722644144173b30c70a5388a1970eaae07bed91585ede13a86003a0bb83754` |
| `experiments/geometria_proporcional/configs/proportional_mapping_feasibility_v1.json` | `0eda5cc2b92f41d375b785f4680bfe6cb65257d83a6502de5a170eee48687507` |
| `experiments/geometria_proporcional/evaluate_proportional_mapping_feasibility.py` | `00fb69f452c425d0cc44c221ff6e55fee0036d8f50a7dc4764585325b7719acf` |
| `experiments/geometria_proporcional/prepare_proportional_mapping_feasibility.py` | `ed336dc23da5a067b35e3833c9630d0cb809aed82a2e3e139506ab11ed71a184` |
| `experiments/geometria_proporcional/run_proportional_mapping_feasibility.py` | `c823ee762792c1b5dd6f7ef1f59c833392d0319947dbd970f3f003f880d24414` |
| `tests/fixtures/proportional_mapping_private_invariance/a/fixture_manifest.json` | `32d9ce95ea083c549eccb9f017cae2bdc38e084796f064496b528e55356a1b01` |
| `tests/fixtures/proportional_mapping_private_invariance/b/fixture_manifest.json` | `dac2f473176ce804b37bc5d66bac6ee903336ec7445bf0ada00549d6d9b46a16` |
| `tests/test_proportional_mapping_feasibility.py` | `3a381e2fbf52264f9e9d24d514dd02a5e83c77903b70aea4c2b4e1037c7392d7` |

## Findings

### HIGH-01 — El checker acepta una hoja forjada y no verifica el cierre criptográfico

El contrato exige que el checker recalcule directamente desde fuentes ligadas
y outputs congelados, sea la única autoridad de adjudicación y rechace
corrupciones de hash, keyset, shape, join, predicado y decisión
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:123-129,357-359`). La
implementación no cumple esa frontera:

- `verify_tree_manifest` sólo verifica archivos enumerados; no exige pathset
  exacto ni rechaza archivos extra (`check_proportional_mapping_feasibility.py:320-324`).
- `candidate_valid` hace checks superficiales, pero no verifica el hash del
  candidato contra `evaluation_evidence`, el core manifest ni el query literal
  (`ibid.:532-546`).
- M1–M5 se derivan de booleanos producidos por el evaluator, no se recomputan
  desde candidato y fuentes (`ibid.:549-555`). En particular, el evaluator
  escribe cuatro incompatibilidades comunes y la autoridad como constantes
  (`evaluate_proportional_mapping_feasibility.py:580-589`).
- La fase final acepta cualquier JSON cuyo campo `status` sea `PASS`; no valida
  schema, hashes de cores, archivos comparados ni mismatches
  (`check_proportional_mapping_feasibility.py:614-624`).
- El `core_manifest.json` tampoco se verifica dentro de `compute`.

Reproducción CPU: sobre una copia de `run_a` se cambió el query del candidato,
se pusieron en `true` los cinco booleanos comunes de la evidencia, se añadió
`prepared/public/UNLISTED_PRIVATE_TRUTH.npy`, se conservó sin cambios el core
manifest viejo y se pasó `{"status":"PASS"}` como replay. El checker salió 0 y
emitió:

```json
{"technical_status":{"artifact_status":"PASS","checker_status":"PASS","replay_status":"PASS","source_status":"PASS"},"mapping_decision":"COMMON_FACTORIAL_FEASIBLE"}
```

El candidato original tenía SHA-256
`7b3532463dd9fc8358477682ad8cd73e53742a9e6f7bf05d009a323e6a2da969` y
el alterado
`f25637f64697505fa3f6757f64d0ef2fd73654a9948ca319adceca0e970ae343`;
el core manifest siguió byte-idéntico con SHA-256
`ee545852fe62d738e6b8e0b396a5d8488e004131dbb3fa541b05b64cc3faccaa`.
Por tanto, hoy puede publicarse cualquiera de las tres hojas a partir de
evidencia no autenticada.

Corrección requerida: validar pathsets exactos y hashes de ambos árboles,
candidato, evidencia, cores y replay; enlazar el hash del candidato; rechazar
extras; recomputar M1–M5 desde fuentes/candidato; y verificar íntegramente el
receipt de replay antes de aplicar el álgebra.

### HIGH-02 — R6 y S6 pasan sin ejecutar los controles shuffled ni los reports matched

R6 exige loss, métricas, convergencia, target-shuffle total y matched report
sobre exactamente las mismas unidades de las cuatro celdas
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:181-193`). S6 exige medir
posterior y acción, aplicar una permutación única y usar la intersección
congelada de filas autorizadas (`ibid.:306-313`).

En set-valued, el evaluator calcula sólo el vector de permutación y cuenta la
intersección de autorizaciones; nunca aplica `target[mapping]`, ni vuelve a
calcular NLL/Brier/compatibilidad/regret/cola bajo shuffle, ni emite un matched
report (`evaluate_proportional_mapping_feasibility.py:403-418`). El checker
sólo reproduce el digest del vector y el conteo de singletons
(`check_proportional_mapping_feasibility.py:327-369`).

En relacional, evaluator y checker construyen el target transportado y su
digest, pero no calculan relation RMSE, quotient RMSE, convergencia/fallo ni un
matched report para las cuatro celdas sobre ese target
(`evaluate_proportional_mapping_feasibility.py:516-557`;
`check_proportional_mapping_feasibility.py:462-483`).

El run nominal informó `R6=PASS` y `S6=PASS`, aunque sus outputs sólo contenían
`mapping_sha256`, singletons/fixed points o `transported_target_sha256`; no
existía ninguna tabla de estimands shuffled/matched. Esto invalida la premisa de
que sobrevivieron dos contrastes nativos completos y, por extensión, la hoja
`BIFURCATE_NATIVE_CONTRASTS`.

Corrección requerida: materializar los targets permutados, ejecutar las cuatro
celdas sin cambiar ningún otro campo, emitir estimands nominal/shuffled/matched
con unidades y soporte, y hacer que el checker los recompute desde cero.

### HIGH-03 — El factorial `GENERIC/TYPED × WLS/IRLS` se certifica con una sola vista por estado

R4 requiere reconstruir las cuatro celdas con la misma IR, gauge, parámetros y
budget; R6 exige métricas y convergencia comunes
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:181-193`). El evaluator fija
`sample = 0` y sólo recompone WLS/IRLS para esa vista de cada uno de los cuatro
estados (`evaluate_proportional_mapping_feasibility.py:470-510`). El checker
repite exactamente la primera vista (`check_proportional_mapping_feasibility.py:440-456`).

Cada estado nominal contiene 631 vistas. En consecuencia, sólo se comparan 4
de 2.524 pares estado-vista, u 8 de 5.048 salidas solver si se cuentan WLS e
IRLS por separado. Las cuatro muestras pasan `atol=1e-8` —WLS diferencia 0;
IRLS entre `4.50e-10` y `6.20e-10`—, pero las otras 2.520 vistas no se
recomponen. Los shapes cacheados `[7507]` no sustituyen esa verificación.

Corrección requerida: ejecutar WLS e IRLS en todas las vistas de cada brazo y
seed, verificar `x_hat`, convergencia e iteraciones a `1e-8`, calcular las
métricas primarias y hacer el replay independiente completo.

### HIGH-04 — Las fases de fuentes no están codificadas y el evaluator abre fuentes P/C-only

La matriz congelada distingue fases P/B/E/C y asigna `W52_POLICY`,
`W53_PRIMITIVE`, `W53_PLATT`, `W54_SELECTION`, `W54_PRIMITIVE`, `GRAPH_SCHEMA`
y `GRAPH_CONFIG` sólo a P/C (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:35-79`).
El config implementado conserva únicamente triples `id/path/hash`, sin rol,
campos ni fases (`proportional_mapping_feasibility_v1.json:8-38`), por lo que
ningún proceso puede hacer cumplir la matriz.

El evaluator importa directamente `GRAPH_SCHEMA`, `W53_PRIMITIVE` y
`W54_PRIMITIVE` (`evaluate_proportional_mapping_feasibility.py:20-34`) y abre
desde `ROOT` `W52_POLICY`, `W53_PLATT`, `W54_SELECTION` y `GRAPH_CONFIG`
(`ibid.:71-80,365-373,442-445`). Esos accesos no pasan por
`prepared/public/` y no están autorizados para E. Aun así, el evaluator fija
`authority_phases_respected:true` y M5 pasa porque el checker confía ese valor
(`ibid.:588`; `check_proportional_mapping_feasibility.py:555`).

El builder sí está libre de `ROOT` y sus lecturas explícitas se restringen al
argumento público (`build_proportional_mapping_candidate.py:40-58,71-105`),
pero ese acierto no compensa la ruptura de E ni la ausencia de enforcement.

Corrección requerida: serializar en P toda receta pública congelada necesaria,
incorporar fases/roles/keysets al config y a receipts verificables, y hacer que
E sólo reciba candidato, `prepared/public` y `prepared/private_dev`. Si se
pretende autorizar esos accesos directos, hace falta revisar y volver a auditar
el plan; no corresponde cambiar la autoridad silenciosamente en código.

### HIGH-05 — Una falla técnica posterior deja persistida una hoja semántica

El plan fija que cualquier falla técnica deja `mapping_decision:null`
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:25-33,142-167,415-420`). El
runner ejecuta los checkers finales y escribe adjudicaciones antes de calcular
el RSS y el estado terminal (`run_proportional_mapping_feasibility.py:178-207`).
Si después detecta timeout/RSS, mismatch final u otro error, escribe runtime
`FAIL/BUDGET_EXCEEDED` y aborta, pero no invalida ni reemplaza los
`adjudication.json` ya emitidos (`ibid.:208-224`). El checker tampoco recibe el
estado runtime: calcula su propio `artifact_status` sólo con manifests
preparados y candidato (`check_proportional_mapping_feasibility.py:532-575`).

Reproducción CPU: se usó el config vigente con sólo
`rss_bytes_exclusive=1`. El runner terminó con exit 1 y runtime:

```json
{"artifact_status":"BUDGET_EXCEEDED","max_ru_maxrss_bytes":223645696,"error":null}
```

Sin embargo, `run_a/adjudication.json` quedó con los cuatro estados técnicos en
`PASS` y `mapping_decision:"BIFURCATE_NATIVE_CONTRASTS"`. Es exactamente la
degradación de una falla técnica a una hoja semántica que el plan prohíbe.
Además, una señal por `RLIMIT_AS` llega como return code no cero, se transforma
en `RuntimeError` (`run_proportional_mapping_feasibility.py:64-84`) y cae en
`artifact_status:FAIL`, no en `BUDGET_EXCEEDED` (`ibid.:198-203`).

Corrección requerida: cerrar presupuesto/replay antes de adjudicar, pasar al
checker el estado terminal autenticado y, ante cualquier falla posterior,
emitir sólo una adjudicación técnica con decisión nula. Clasificar señales del
límite y `MemoryError` del hijo como `BUDGET_EXCEEDED`.

### MEDIUM-01 — Las mutaciones obligatorias son declaraciones, no ejecuciones

El plan exige mutaciones aisladas de los 17 predicados, corrupción real de seis
clases del candidato, tamper byte-a-byte de las 29 fuentes y contrafactuales
privados campo por campo (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:315-359`).
`mutation_suite` construye filas sintéticas ya marcadas PASS/FAIL y devuelve
literalmente `"REJECTED"` para todas las corrupciones, sin alterar ni pasar un
artefacto por el checker (`check_proportional_mapping_feasibility.py:504-529`).
El test sólo afirma esos labels (`tests/test_proportional_mapping_feasibility.py:132-143`).

Los dos fixtures privados sí prueban igualdad byte-exacta de un candidato
TEST_ONLY, pero cambian simultáneamente W49, W54 y grafo y no prueban
eliminaciones uno por uno (`tests/fixtures/proportional_mapping_private_invariance/a/fixture_manifest.json:1`;
`b/fixture_manifest.json:1`; `tests/test_proportional_mapping_feasibility.py:80-103`).
La reproducción de HIGH-01 confirma que una corrupción real contradice los
labels declarados.

Corrección requerida: cada mutación debe crear una copia temporal real, invocar
preparer/builder/checker y comprobar estado, reason code y decisión nula.

### MEDIUM-02 — Los 17 predicados no conservan la evidencia ni los reason codes cerrados del plan

El plan exige por predicado evidencia con `source_id`, locator, observado y
digest y reason code causal dentro del catálogo propio
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:131-140,169-193,306-313`). La
implementación asigna un único reason code fijo a cada predicado
(`check_proportional_mapping_feasibility.py:27-45`) y `pred` acepta una lista
opaca sin ese schema (`ibid.:486-487`). Así, por ejemplo, M1 no puede distinguir
`QUERY_MISMATCH`, `UNIT_BIJECTION_INCOMPLETE` o `SYNTHETIC_ID_EQUIVALENCE`, y
M5 no puede distinguir leakage, fase, monitor o falta de independencia.

Corrección requerida: emitir evidencia estructurada y seleccionar el reason
code exacto desde la condición observada; añadir tests reales para cada rama
del catálogo.

### LOW-01 — La receta logistic no fija literalmente L2 y el checker no valida la versión

El plan congela scikit-learn `1.8.0` y `penalty=L2`
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:251-260`). Evaluator y checker
instancian `LogisticRegression` sin `penalty="l2"`
(`evaluate_proportional_mapping_feasibility.py:200-216`;
`check_proportional_mapping_feasibility.py:209-214`). El evaluator sí exige
versión `1.8.0`, pero el checker importa `sklearn` y no la valida. En el entorno
auditado el default efectivo reprodujo los mismos coeficientes, de modo que no
se observó divergencia numérica; queda, no obstante, una dependencia implícita
contraria al freeze literal.

Corrección requerida: fijar `penalty="l2"` explícitamente y validar versión y
parámetros efectivos también en el checker.

## Verificaciones positivas y alcance observado

- `29/29` hashes de fuentes y el hash del plan coincidieron en el run nominal.
- El builder no importa evaluator/checker ni declara una decisión; sus lecturas
  explícitas son relativas a la raíz pública recibida. El candidato quedó
  read-only antes de E y su hash no cambió durante la evaluación
  (`run_proportional_mapping_feasibility.py:161-169`;
  `evaluate_proportional_mapping_feasibility.py:566-575`).
- La extracción productiva observada separó targets W50/W54 y truth de grafo en
  `private_dev`; pseudonimizó tokens, clusters y vistas; y no expuso los campos
  privados declarados en el run nominal
  (`prepare_proportional_mapping_feasibility.py:141-179,183-209,212-257`).
- MARGINAL y JOINT produjeron masas finitas `[384,15]`, normalizadas; las cuatro
  acciones tuvieron shape `[384,24]`; HARD usa el MAP de cada posterior con
  desempate NumPy por índice menor; no hubo duplicación hard observada
  (`evaluate_proportional_mapping_feasibility.py:83-93,354-402`).
- El design contextual tiene las 17 features y coincide numéricamente entre E y
  C; los fits nominales conservaron ambas clases. MARGINAL tuvo 464 filas/82
  tokens de fit y clases harm `178/286`, incompatibilidad `387/77`; JOINT tuvo
  601/89 y clases `240/361`, `507/94`.
- El orden de selección, desigualdades estrictas, cuantiles lineales y sentinel
  HARD_ONLY coinciden con el plan (`evaluate_proportional_mapping_feasibility.py:237-333`).
- El shuffle relacional construyó 379 masters, 631 vistas, cero self-donors,
  cero singletons y donante consistente por master; transportó 16.950 targets
  de arista. Esa construcción geométrica es correcta, aunque falta ejecutar el
  control como estimand.
- El runner nominal terminó en `12.3067 s`, máximo RSS `223551488` bytes,
  comparó 147 archivos por core sin mismatch y produjo dos adjudicaciones
  byte-exactas. Esto verifica determinismo nominal, no robustez adversarial.
- `RLIMIT_AS=2 GiB`, deadline global y límites de threads están presentes
  (`run_proportional_mapping_feasibility.py:60-78,133-156`). La transición
  terminal requiere la corrección de HIGH-05.
- No hay imports de Torch/CUDA ni llamadas de consulta GPU en los nueve
  archivos. El runner fija `CUDA_VISIBLE_DEVICES=""`; los claims persistentes
  conservan `gpu_used_or_queried:false`.

## Checks CPU reproducibles ejecutados

```text
venv/bin/python -m pytest -q tests/test_proportional_mapping_feasibility.py
=> 8 passed in 0.25s

venv/bin/python experiments/geometria_proporcional/run_proportional_mapping_feasibility.py \
  --development --output <tmp>/output
=> exit 0; 12.3067 s; RSS 223551488; 147/147 core files byte-exactos
=> hoja nominal BIFURCATE_NATIVE_CONTRASTS

# Corrupción real sobre copia de run_a:
# cambiar candidate.query; forzar los cinco flags common; añadir un archivo
# público no manifestado; conservar core_manifest; usar {"status":"PASS"}.
venv/bin/python experiments/geometria_proporcional/check_proportional_mapping_feasibility.py \
  --run <tmp>/tampered_run --phase final --replay-evidence <tmp>/fake_replay.json
=> exit 0; cuatro estados PASS; hoja COMMON_FACTORIAL_FEASIBLE

# Config de prueba idéntico salvo rss_bytes_exclusive=1:
venv/bin/python experiments/geometria_proporcional/run_proportional_mapping_feasibility.py \
  --development --config <tmp>/low_rss_config.json --output <tmp>/low_rss_output
=> exit 1; runtime BUDGET_EXCEEDED; adjudication persistida BIFURCATE_NATIVE_CONTRASTS
```

Los árboles temporales de auditoría se eliminaron después de recoger receipts.

## Matriz de cobertura solicitada

| Área | Resultado |
|---|---|
| source bindings y fases | **FAIL** — HIGH-04 |
| builder sólo prepared/public | PASS en lecturas explícitas; enforcement integral **FAIL** — HIGH-01 |
| leakage público/privado | PASS nominal; rechazo adversarial **FAIL** — HIGH-01 |
| checker independiente | independencia de imports PASS; autoridad/recomputación **FAIL** — HIGH-01 |
| MARGINAL/JOINT + HARD_MAP_SET + contextual | PASS nominal |
| fitting, selección y guards | PASS nominal; freeze literal logistic LOW-01 |
| `GENERIC/TYPED × WLS/IRLS`, `1e-8` | **FAIL** por cobertura 4/2524 — HIGH-03 |
| shuffled/matched | **FAIL** — HIGH-02 |
| álgebra 17 predicados / tres hojas | tabla total PASS; evidencia/mutaciones **FAIL** — HIGH-01, MEDIUM-01/02 |
| TEST_ONLY | smoke contrafactual PASS; aislamiento/cobertura productiva **FAIL** — HIGH-01, MEDIUM-01 |
| replay/manifests sin circularidad | estructura nominal acíclica; validación autoritativa **FAIL** — HIGH-01 |
| RLIMIT/watchdog | límites nominales PASS; semántica terminal **FAIL** — HIGH-05 |
| ausencia GPU/CUDA query | PASS |
| determinismo | PASS nominal, 147 archivos byte-exactos |
| hoja tras fallo técnico | **FAIL reproducido** — HIGH-05 |

```json
{
  "schema_version": "proportional-mapping-feasibility-implementation-audit-v1",
  "audit_id": "R539",
  "runtime_configuration": {
    "model": "gpt-5.6-sol",
    "reasoning_effort": "high"
  },
  "target": {
    "commit": "4205a0407ff5d372e308a8fb307d34a0b0f1feef",
    "parent": "85e8e7165a2257b46a8be4df0bebc2e045791414",
    "tree": "3944babac8c1f4c31b088d3f94ae934cf30fcaa7",
    "path_count": 9,
    "pathset_sha256": "d1f384fbfbba038fcc730d4726756e6ae2ae320ff371af6fcd08c8ebe0b8287d",
    "binary_patch_sha256": "fd0c60a62e243203809f9506234af944c6cce504368a9d4e6cbdc8bfba4cc7c0"
  },
  "approved_plan": {
    "commit": "c1b9b45a8d015e6e147328e2e395402cd4930d5b",
    "sha256": "ef61081fcbd6917368d62da065843e3b397abec0149dfed5c0cc99e77fbdc6d2"
  },
  "prior_final_plan_audit": {
    "id": "R538",
    "sha256": "243ea848f51cd76448aedc178796684bc8715a9162e1a44c5a612692f919fbb6"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 5,
    "medium": 2,
    "low": 1
  },
  "nominal_checks": {
    "source_hashes_matched": 29,
    "source_hashes_declared": 29,
    "pytest_passed": 8,
    "pytest_failed": 0,
    "run_exit_code": 0,
    "wall_seconds": 12.306732449680567,
    "max_ru_maxrss_bytes": 223551488,
    "core_files_compared": 147,
    "core_mismatches": 0,
    "nominal_mapping_decision": "BIFURCATE_NATIVE_CONTRASTS"
  },
  "adversarial_checks": {
    "tampered_candidate_evidence_public_tree_and_replay_accepted": true,
    "forged_mapping_decision": "COMMON_FACTORIAL_FEASIBLE",
    "budget_exceeded_left_nonnull_decision": true,
    "budget_failure_persisted_decision": "BIFURCATE_NATIVE_CONTRASTS"
  },
  "builder_public_only_static_surface": true,
  "phase_bindings_enforced": false,
  "public_private_leakage_rejected_adversarially": false,
  "checker_authoritative_and_independent": false,
  "set_four_cells_nominally_executable": true,
  "hard_map_set_semantics_nominally_correct": true,
  "contextual_design_17_features_nominally_correct": true,
  "relational_full_factorial_recomputed": false,
  "solver_replay_atol": 1e-8,
  "solver_state_view_pairs_checked": 4,
  "solver_state_view_pairs_required": 2524,
  "shuffled_controls_executed_as_estimands": false,
  "matched_reports_executed": false,
  "mutation_suite_executes_real_mutations": false,
  "replay_cryptographically_verified_by_checker": false,
  "technical_failure_forces_null_decision": false,
  "nominal_determinism": true,
  "gpu_used_or_queried": false,
  "lockbox_or_sealed_monitor_opened": false,
  "historical_monitor_opened": false,
  "code_plan_or_canonical_data_modified": false,
  "architecture_promoted": false,
  "scientific_decision": null,
  "decision_authority": "user"
}
```
