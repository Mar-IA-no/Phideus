# R540 — Reauditoría independiente de `MAPPING-FEASIBILITY`

## Dictamen técnico: REVISE

La revisión final `8920f42` resuelve los cinco findings HIGH y el finding LOW
de R539. El run CPU completo es determinista, queda dentro del presupuesto,
recompone las 631 vistas de los cuatro estados con ambos solvers, calcula y
recalcula controles nominal/shuffled/matched, cierra candidato/core/replay y
anula correctamente cualquier hoja tras un fallo terminal.

Persisten los dos findings MEDIUM de R539: la suite declara mutaciones para los
17 predicados sin ejecutar cada condición real, y varias ramas semánticas de
reason codes no están implementadas. Una corrupción controlada aislada mostró
el efecto concreto: M1 acepta como bijección común un mero renaming sintético,
aunque el plan exige `SYNTHETIC_ID_EQUIVALENCE` y `FAIL`. Esto no altera la hoja
nominal —M2, M3 y M4 permanecen falsos—, pero impide un `PASS` de implementación.

Conteo final: **0 HIGH, 2 MEDIUM, 0 LOW**. La hoja nominal observada fue
`BIFURCATE_NATIVE_CONTRASTS`; esta reauditoría valida que la ejecución la
materializa bajo el estado actual, pero no constituye promoción arquitectónica
ni `GO/NO-GO`.

## Identidad y alcance

- Instancia: **GPT-5.6-Sol**, esfuerzo **high**.
- Target exacto: `8920f427a40dc628e75380bc39186bfc387b2713`.
- Parent: `56d361274b7508f555be8a70164d8b0e87cbc462`.
- Tree: `978a62bd4f88da2770baa2b72a20996bebb87aaf`.
- Plan aprobado vigente: SHA-256
  `ef61081fcbd6917368d62da065843e3b397abec0149dfed5c0cc99e77fbdc6d2`,
  blob `29969b938050a815bba5ee5dd7a1db6d0f23adcb`.
- R539 leído completo: SHA-256
  `d689692a3c89d2497fe2481fa04049dd096baa242918300e14905dce26972851`,
  blob `7af5ba2cf1b00a1e9994b4ae833718d016492fc5`.
- Se leyeron completos el plan, R539 y los 14 archivos actuales de
  implementación/tests. El working tree estaba limpio y esos archivos
  coincidían con HEAD antes de ejecutar los checks.
- No se abrió lockbox, monitor sellado ni monitor histórico. No se usó ni
  consultó GPU/CUDA. No se leyó `PENDIENTES.md` ni memoria privada ajena. No se
  modificaron implementación, plan, documentación canónica ni datos.

Digest SHA-256 del pathset ordenado de 14 archivos:
`dcb3ccd9cc70e50861ccf57bbf716a4be31b3eee2c394e684cd00472433fb7aa`.
Digest del patch binario del target:
`8e140e8293f3274e376c4848a3663a783d63c5c79e809707743f8d36be8564dc`.

| Path | SHA-256 |
|---|---|
| `experiments/geometria_proporcional/build_proportional_mapping_candidate.py` | `6be4026c763498fb13eea966d0b6a7aef3641bc6170f6ee3d59a79ede4e742d6` |
| `experiments/geometria_proporcional/check_proportional_mapping_feasibility.py` | `34d2de1cc2f1d0a48b5da68cc4b60f3f1adf745350ef35e7ed12353f41bc176f` |
| `experiments/geometria_proporcional/configs/proportional_mapping_feasibility_v1.json` | `d49ab5a181af00134237153277d8085e93dfa0521862a291de368c480015972e` |
| `experiments/geometria_proporcional/evaluate_proportional_mapping_feasibility.py` | `fe2cd554f39a5a328aae79e09d16ddb2f90cf46eff844984e238f68cb29422c1` |
| `experiments/geometria_proporcional/prepare_proportional_mapping_feasibility.py` | `6610051f13076274bd5b1cd99201ffb3f4e636c720076ddf636a4c90b9f83be7` |
| `experiments/geometria_proporcional/run_proportional_mapping_feasibility.py` | `ccc6864df354f093a39d0cf5cbb00e99b7e594a091e4ecb309e61c236bddef46` |
| `tests/fixtures/proportional_mapping_private_invariance/a/fixture_manifest.json` | `32d9ce95ea083c549eccb9f017cae2bdc38e084796f064496b528e55356a1b01` |
| `tests/fixtures/proportional_mapping_private_invariance/b/fixture_manifest.json` | `dac2f473176ce804b37bc5d66bac6ee903336ec7445bf0ada00549d6d9b46a16` |
| `tests/fixtures/proportional_mapping_private_invariance/graph_clean_removed/fixture_manifest.json` | `7901f06ec66ffa9a04e7aab4d15e544ff8aff1b11c3f93af1ac1a2f1de82f930` |
| `tests/fixtures/proportional_mapping_private_invariance/graph_mechanism/fixture_manifest.json` | `a75e63000746a9afb336cfca5a2e5c7a1765a9ea2b772509841f093844bda4ad` |
| `tests/fixtures/proportional_mapping_private_invariance/graph_x/fixture_manifest.json` | `fdd7afd21fcecdf6e572a8ea10c50d8b8c7233764c55e00a393682f76aab3ddd` |
| `tests/fixtures/proportional_mapping_private_invariance/set_target/fixture_manifest.json` | `038d9b4a86241796fa5f99dc52ba4796574bef2bd820d06c7d73be49cb251dd7` |
| `tests/fixtures/proportional_mapping_private_invariance/w49/fixture_manifest.json` | `1d974d93de8d70205de35ffb608939830850e23c4886ce2159ea8b199fe771c7` |
| `tests/test_proportional_mapping_feasibility.py` | `7b6b3a1633301e2ef1270fa856f093da0d8a1d29eba6853fbb1c615164b32ca3` |

## Resolución uno por uno de R539

### R539 HIGH-01 — RESUELTO

Los manifests público/privado exigen ahora pathsets cerrados y rechazan extras,
incluso si se los agrega al manifest (`check_proportional_mapping_feasibility.py:444-470`).
El checker rederiva el protocolo desde fuentes congeladas, recompone W49/W50,
W54 y los cuatro NPZ de grafo, incluidos pseudónimos
(`ibid.:354-441,505-526,697-721,964-1018`). También valida candidato,
evidencia, core manifest, replay de ambos runs y terminal status
(`ibid.:904-960,964-1018,1111-1134`).

La corrupción controlada de candidato, evidencia y replay fue rechazada con
`artifact_status:FAIL`, `checker_status:FAIL`, `replay_status:FAIL` y
`mapping_decision:null`. No fue posible convertir ese estado inconsistente en
`COMMON_FACTORIAL_FEASIBLE`.

### R539 HIGH-02 — RESUELTO

Set-valued aplica el target permutado y calcula posterior y acción para
nominal, shuffled, matched nominal y matched shuffled sobre las cuatro celdas
(`evaluate_proportional_mapping_feasibility.py:414-503`). El checker recompone
masas, acciones, máscara común, targets shuffled y todos esos estimands
(`check_proportional_mapping_feasibility.py:505-617`).

Relacional transporta target y quotient por master, usa salidas solver recién
recalculadas y publica nominal/shuffled/matched para cada estado
(`evaluate_proportional_mapping_feasibility.py:713-797`). El checker vuelve a
resolver y recalcular esas métricas sin usar los caches privados como resultado
del control (`check_proportional_mapping_feasibility.py:697-837`). El run
observó 15 filas/8 tokens matched set-valued y 629/631 vistas matched
relacionales.

### R539 HIGH-03 — RESUELTO

Evaluator y checker recorren `range(len(n_nodes))` para cada uno de los cuatro
estados y resuelven WLS e IRLS en todas las vistas
(`evaluate_proportional_mapping_feasibility.py:614-707`;
`check_proportional_mapping_feasibility.py:724-770`). Se verificaron **631 × 4
= 2.524** pares estado-vista y **5.048** salidas solver, con tolerancia `1e-8`,
convergencia e iteraciones completas. R4 pasó en ambos checkers.

### R539 HIGH-04 — RESUELTO

P serializa protocolo y recetas; builder y evaluator sólo reciben superficies
preparadas (`prepare_proportional_mapping_feasibility.py:147-216`;
`run_proportional_mapping_feasibility.py:96-117`). El evaluator ya no define
`ROOT`, no importa primitives del repositorio y lee sólo public/private/candidato
(`evaluate_proportional_mapping_feasibility.py:1-47,806-838`). El checker, como
fase C, reabre las fuentes ligadas para recomponer y contrastar los preparados.
El config codifica roles, fases y keysets cerrados
(`proportional_mapping_feasibility_v1.json:39-65`). M5 pasó con la barrera
observada.

### R539 HIGH-05 — RESUELTO

El runner autentica replay y terminal status antes de final, exige los cuatro
estados PASS de ambas adjudicaciones y vuelve a medir presupuesto antes de
publicar (`run_proportional_mapping_feasibility.py:222-263`). Toda excepción o
exceso llama `invalidate_outputs`, elimina salidas finales y sobrescribe ambas
adjudicaciones con decisión nula (`ibid.:139-171,264-292`). Señales y marcadores
de memoria se clasifican como `BudgetExceeded` (`ibid.:68-93`).

Con presupuesto controladamente reducido a 95 s, `run_a` alcanzó una hoja antes
de que `run_b` agotara el deadline. El cierre terminó
`BUDGET_EXCEEDED` a 95.0106 s y sobrescribió **ambas** adjudicaciones con
`mapping_decision:null`; no sobrevivió ninguna hoja no nula.

### R539 MEDIUM-01 — NO RESUELTO

La suite mejoró: hay corrupciones locales reales para pathsets, candidato,
replay, una fuente y presupuesto, y siete fixtures privados campo por campo
(`tests/test_proportional_mapping_feasibility.py:84-128,166-248`). Pero el plan
exige una mutación aislada real de cada M1..M5, R1..R6 y S1..S6 y de cada clase
de candidato. `mutation_suite` todavía fabrica objetos de predicado y etiqueta
las seis clases como `COVERED_BY_EXTERNAL_MUTATION_TEST`, sin ejecutar cada una
(`check_proportional_mapping_feasibility.py:876-900`). El test correspondiente
sólo afirma esos labels (`tests/test_proportional_mapping_feasibility.py:141-152`).

Corrección requerida: parametrizar 17 mutaciones reales, más las seis clases de
candidato, ejecutar cada copia temporal por el checker y comprobar reason code,
estado técnico y decisión nula.

### R539 MEDIUM-02 — NO RESUELTO

La evidencia ya tiene `source_id`, locator, observado y digest, y el catálogo
rechaza códigos ajenos (`check_proportional_mapping_feasibility.py:843-859`).
No obstante, el test recorre el catálogo llamando directamente `pred`; prueba
serialización, no que cada condición produzca su código
(`tests/test_proportional_mapping_feasibility.py:155-163`). R1–R6 y S1–S6
siguen usando por defecto sólo el primer reason code
(`check_proportional_mapping_feasibility.py:1052-1063`).

Una mutación aislada conservó el query literal, igualó los tres namespaces y
declaró `{"kind":"renaming_only"}`. Tras actualizar el hash candidato-evidencia,
el checker pre mantuvo estados source/artifact/checker en PASS y marcó
`M1_QUERY_UNIT:PASS` (`ibid.:1042,1047`), cuando el plan exige FAIL con
`SYNTHETIC_ID_EQUIVALENCE`. M2–M4 siguieron FAIL, por lo que la hoja común no
fue alcanzada; el defecto permanece acotado al predicado y a su cobertura.

Corrección requerida: M1 debe acreditar bijección 1:1:1 desde autoridad previa,
no sólo igualdad textual y bridge no nulo; las demás ramas deben mapear cada
condición observada a su reason code y tener una mutación real dedicada.

### R539 LOW-01 — RESUELTO

Evaluator y checker fijan `penalty="l2"`, `l1_ratio=0`, el resto de parámetros
congelados, validan scikit-learn `1.8.0` y comparan parámetros efectivos
(`evaluate_proportional_mapping_feasibility.py:221-247`;
`check_proportional_mapping_feasibility.py:220-233`). Los cuatro fits se
reprodujeron. Scikit-learn emite un `FutureWarning` de deprecación para
`penalty`, sin alterar la receta ni los resultados bajo la versión fijada.

## Checks CPU reproducibles

```text
MAPPING_FEASIBILITY_RUN_A=<run_a> MAPPING_FEASIBILITY_RUN_B=<run_b> \
venv/bin/python -m pytest -q tests/test_proportional_mapping_feasibility.py
=> 13 passed in 15.98s

venv/bin/python experiments/geometria_proporcional/run_proportional_mapping_feasibility.py \
  --development --output <tmp>/nominal
=> exit 0; 109.5619 s; RSS máximo 382619648 bytes
=> 148 archivos core byte-exactos; closure core/replay/terminal true
=> source/artifact/checker/replay PASS; BIFURCATE_NATIVE_CONTRASTS

# Presupuesto local reducido sólo para validar el estado terminal:
# wall_seconds_exclusive=95, restantes campos idénticos.
venv/bin/python experiments/geometria_proporcional/run_proportional_mapping_feasibility.py \
  --development --config <tmp>/reduced_budget.json --output <tmp>/reduced
=> exit 1; BUDGET_EXCEEDED a 95.0106 s
=> run_a mapping_decision null; run_b mapping_decision null

# Corrupción controlada aislada de M1 sobre copia local:
# query intacto; tres namespaces="renamed"; bridge={"kind":"renaming_only"};
# hash candidato-evidencia actualizado; checker --phase pre.
=> estados source/artifact/checker PASS; M1 PASS observado, FAIL esperado
```

Los árboles temporales se eliminaron al terminar la auditoría.

## Resultado consolidado

| Área | Resultado |
|---|---|
| cierre candidato/pathsets/core/replay | PASS |
| recomputación directa M1–M5 | REVISE acotado a M1 — MEDIUM-02 |
| public/private y barrera P/B/E/C | PASS |
| MARGINAL/JOINT, HARD y contextual | PASS |
| fitting/selección/guards/sklearn | PASS |
| controles nominal/shuffled/matched | PASS |
| 631 vistas × 4 estados × WLS/IRLS | PASS |
| álgebra de hojas | PASS |
| mutaciones reales de 17 predicados | REVISE — MEDIUM-01 |
| evidence estructurada | PASS; ramas causales incompletas — MEDIUM-02 |
| budget/fallo fuerza decisión nula | PASS reproducido |
| determinismo | PASS, 148 archivos core |
| ausencia de uso/consulta GPU | PASS |

```json
{
  "schema_version": "proportional-mapping-feasibility-implementation-reaudit-v1",
  "audit_id": "R540",
  "runtime_configuration": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
  "target": {
    "commit": "8920f427a40dc628e75380bc39186bfc387b2713",
    "parent": "56d361274b7508f555be8a70164d8b0e87cbc462",
    "tree": "978a62bd4f88da2770baa2b72a20996bebb87aaf",
    "path_count": 14,
    "pathset_sha256": "dcb3ccd9cc70e50861ccf57bbf716a4be31b3eee2c394e684cd00472433fb7aa",
    "binary_patch_sha256": "8e140e8293f3274e376c4848a3663a783d63c5c79e809707743f8d36be8564dc"
  },
  "technical_verdict": "REVISE",
  "findings": {"high": 0, "medium": 2, "low": 0},
  "r539_resolution": {
    "HIGH-01": "RESOLVED",
    "HIGH-02": "RESOLVED",
    "HIGH-03": "RESOLVED",
    "HIGH-04": "RESOLVED",
    "HIGH-05": "RESOLVED",
    "MEDIUM-01": "OPEN",
    "MEDIUM-02": "OPEN",
    "LOW-01": "RESOLVED"
  },
  "nominal_run": {
    "exit_code": 0,
    "wall_seconds": 109.56189050525427,
    "max_ru_maxrss_bytes": 382619648,
    "pytest_passed": 13,
    "core_files_compared": 148,
    "core_mismatches": 0,
    "technical_status_all_pass": true,
    "mapping_decision": "BIFURCATE_NATIVE_CONTRASTS"
  },
  "relational_solver_state_view_pairs": 2524,
  "relational_solver_outputs": 5048,
  "solver_replay_atol": 1e-8,
  "set_matched_authorized_rows": 15,
  "set_matched_tokens": 8,
  "relational_matched_views": 629,
  "inconsistent_common_state_rejected_with_null_decision": true,
  "reduced_budget": {
    "wall_seconds_exclusive": 95,
    "observed_wall_seconds": 95.01056867465377,
    "artifact_status": "BUDGET_EXCEEDED",
    "run_a_mapping_decision": null,
    "run_b_mapping_decision": null
  },
  "all_17_predicate_mutations_executed": false,
  "synthetic_renaming_rejected_by_m1": false,
  "gpu_used_or_queried": false,
  "lockbox_or_monitor_opened": false,
  "implementation_plan_docs_or_data_modified": false,
  "architecture_promoted": false,
  "scientific_decision": null,
  "decision_authority": "user"
}
```
