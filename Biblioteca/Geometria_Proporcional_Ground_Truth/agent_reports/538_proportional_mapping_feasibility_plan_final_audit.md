# R538 — Auditoría final independiente del plan `MAPPING-FEASIBILITY`

## Dictamen técnico: PASS

La revisión `c1b9b45` resuelve los tres findings materiales de R537 y no
introduce defectos nuevos en álgebra, bindings de fuentes, cuatro celdas,
checker, replay o presupuesto. El plan queda suficientemente cerrado para
implementación: este `PASS` valida el diseño preexperimental, no ejecuta el gate,
no adelanta su hoja semántica y no constituye promoción ni `GO/NO-GO`.

Conteo final: **0 HIGH, 0 MEDIUM y 0 LOW**.

## Identidad, independencia y alcance

- Configuración de la instancia: **GPT-5.6-Sol**, esfuerzo **high**.
- Target exacto: commit
  `c1b9b45a8d015e6e147328e2e395402cd4930d5b`.
- Parent directo: `9c43659f930dd66a2c834f6749bafedea1e18801`.
- El diff del target modifica un solo path:
  `experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md`.
- SHA-256 del plan: `ef61081fcbd6917368d62da065843e3b397abec0149dfed5c0cc99e77fbdc6d2`.
- Git blob del plan: `29969b938050a815bba5ee5dd7a1db6d0f23adcb`.
- R536 leído completo; SHA-256:
  `656b6e23ce1354eed79fc3887aee63faaf77e343838fc20a4bb8b339eccd75c6`.
- R537 leído completo; SHA-256:
  `0953e717b33dda7e77510dfeb12408ba08c75e5d203451a0aa0e068bc1f319c8`.
- Se contrastaron las 29 fuentes ligadas vigentes y los contratos locales
  relevantes. No se abrió lockbox, `sealed_monitor_bundle.npz` ni monitor
  histórico. No se usó ni consultó GPU. No se modificaron plan, código o datos.

## Resolución explícita de R537

### R537-01 — RESUELTO

El control relacional ya es total y conserva la unidad inferencial. La revisión
permuta una sola vez a nivel de `master_id` dentro de `(split,n_nodes)`, aplica
un donante común a todas las vistas dependientes y trata los singletons como
`NONPERMUTABLE_SINGLETON` sin contarlos como soporte
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:198-206`). El target edge-aligned
ya no se copia entre topologías: centra el potencial donante y lo proyecta sobre
la incidencia receptora, manteniendo observación, topología, salida, executor y
pesos del receptor (`ibid.:208-214`). El fixture obligatorio cubre igual número
de nodos, distinto número de aristas y una mutación de pairing entre vistas
(`ibid.:216-219`).

La operación coincide con el contrato local: la observación tipa
`n_nodes/edge_index/observed_log_ratio/edge_valid/edge_variance` por separado
(`src/geometria_proporcional/proportional_graph_contract.py:119-140`) y los
solvers exigen values y weights estrictamente edge-aligned
(`ibid.:494-523,555-586`). Sobre los cuatro NPZ ligados verifiqué por CPU:

- `631` vistas, `379` masters y `18` estratos `(split,n_nodes)`;
- `252` masters con dos vistas dependientes y `127` con una;
- `0` masters con `split/n_nodes` discordante entre vistas, `0` estratos
  singleton y `0` asignaciones al propio donante;
- `16.950` targets de arista transportados por cada estado
  `GENERIC/TYPED × seed`, todos finitos y con `0` fallas de shape;
- inputs públicos, offsets e IDs exactamente pareados entre `GENERIC` y `TYPED`
  para cada seed, y outputs `corrected_log_ratio/reliability` finitos.

Por tanto, la corrección elimina tanto el fallo edge/node como la ruptura de
pairing que motivaron R537-01.

### R537-02 — RESUELTO

La revisión separa dos autoridades que antes eran incompatibles. El tamper de
una de las 29 fuentes productivas debe fallar por hash antes del build y nunca
desactiva el freeze (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:339-343`). La
invariancia privada se prueba en cambio con fixtures sintéticos autenticados por
un hash embebido, manteniendo idéntica la extracción pública y ejecutando
preparer y builder completos para ambas variantes (`ibid.:344-349`).

El modo de fixtures queda confinado a la raíz temporal creada por el runner,
usa otro schema, no acepta paths canónicos, no escribe en la raíz científica y
sólo puede emitir `test_decision`; su aparición en producción fuerza
`artifact_status:FAIL` y nunca puede producir `mapping_decision`
(`ibid.:351-355`). Esto permite probar dependencia contrafactual sin omitir el
hash freeze ni reautorizar fuentes científicas. La barrera productiva sigue
exigiendo hashes antes de separar `prepared/public` y `prepared/private_dev`
(`ibid.:84-109`).

### R537-03 — RESUELTO

Existe ahora una sola semántica hard. `HARD_MAP_SET` toma el set MAP del
posterior correspondiente, fija desempates por índice binario/familia y declara
cualquier duplicación empírica entre celdas (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:234-241`). El reader contextual usa ese mismo baseline y un
schema adaptado explícito (`ibid.:243-277`), en vez de llamar a la primitive W56
que reconstruye el set histórico por `sigmoid(logits)>=0.5`
(`src/geometria_proporcional/wave56_contextual_gate.py:61-69,117-120`).

El orden exacto de las 17 features está congelado; cardinalidad y masa se
derivan respectivamente de `baseline_map_set` y de su índice binario, y
`hard_risk` usa la acción de ese mismo `HARD_MAP_SET`
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:279-299`). Un fixture donde MAP y
threshold histórico difieren fija el digest float64 del design y exige
`CONTEXTUAL_RECIPE_MISMATCH` ante la receta vieja (`ibid.:301-304`).

La materialización independiente sobre las fuentes ligadas confirmó:

| Posterior | Masa | Hard/candidato | Design adaptado | Filas/tokens activos fit | Harm `0/1` | Incompat. `0/1` |
|---|---|---|---|---:|---:|---:|
| `MARGINAL` | `[384,15]` | `[384,24]` | `[384,24,17]` | `464 / 82` | `178 / 286` | `387 / 77` |
| `JOINT` | `[384,15]` | `[384,24]` | `[384,24,17]` | `601 / 89` | `240 / 361` | `507 / 94` |

Las masas son finitas, no negativas y normalizadas; ambos guards conservan las
dos clases. Las cuatro funciones de decisión son por tanto materializables sin
doble lectura del baseline hard.

## Auditoría transversal

### Álgebra y fuentes

Los estados técnicos permanecen separados de las tres hojas y sólo cuatro
`PASS` habilitan la tabla total (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:131-167`). Una falla operacional conserva `mapping_decision:null`, de modo que
no puede convertirse en una conclusión semántica. Recalculé la matriz congelada:
**29/29 SHA-256 coinciden** con los archivos vigentes; no faltan fuentes ni se
requiere selección heurística en runtime (`ibid.:35-82`).

### Cuatro celdas y controles

Las dos masas operan sobre los mismos cuatro logits y los mismos quince sets;
la utilidad externa no entra al posterior (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:221-232`). Hard y contextual quedan definidos para cada
posterior, con fitting, selección, desigualdades, orden total y freeze comunes
(`ibid.:234-313`). El bundle confirma `384` tokens, roles `192/192`, logits
`[384,4]`, per-seed `[3,384,4]` y targets `[384,4]`. Los `16` estratos del
control set-valued tienen entre `14` y `40` filas y ningún singleton.

### Checker, replay y presupuesto

Preparer, builder, evaluator y checker tienen superficies físicas y permisos
no solapados; el checker recompone hashes, schemas, pseudónimos, joins,
predicados y decisión sin importar helpers de builder/evaluator
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:84-129`). Las tres hojas, los 17
predicados, las mutaciones de leakage y las corrupciones de candidato tienen
tests explícitos (`ibid.:315-359`).

El replay evita self-hash: los cores excluyen manifest propio, runtime y
adjudicación; un receipt canónico de ambos cores precede dos adjudicaciones
independientes byte-exactas, y runtime queda fuera de los manifests
(`ibid.:361-404`). El watchdog global `<600 s`, `RLIMIT_AS=2 GiB`, RSS `<2 GiB`
y la transición explícita a `BUDGET_EXCEEDED` impiden publicar una hoja fuera de
contrato (`ibid.:406-425`). No encontré circularidad ni vía de degradación de una
falla técnica a decisión científica.

## Comprobaciones consolidadas

| Check | Resultado |
|---|---|
| identidad target/parent/pathset/SHA/blob | PASS |
| lectura completa de target, R536 y R537 | PASS |
| hashes de fuentes | PASS, `29/29` |
| R537-01 shuffle edge/node y pairing | RESUELTO |
| R537-02 freeze vs fixtures contrafactuales | RESUELTO |
| R537-03 semántica hard y design exacto | RESUELTO |
| álgebra y tres hojas | PASS |
| cuatro celdas set-valued | PASS |
| checker y mutaciones | PASS como especificación |
| replay | PASS como especificación |
| presupuesto terminal | PASS como contrato |
| findings nuevos materiales | ninguno |
| implementación o modificación de plan/código/datos | no |
| lockbox, sealed monitor o monitor histórico abiertos | no |
| GPU usada o consultada | no |

```json
{
  "schema_version": "proportional-mapping-feasibility-plan-final-audit-v1",
  "audit_id": "R538",
  "runtime_configuration": {
    "model": "gpt-5.6-sol",
    "reasoning_effort": "high"
  },
  "target": {
    "commit": "c1b9b45a8d015e6e147328e2e395402cd4930d5b",
    "parent": "9c43659f930dd66a2c834f6749bafedea1e18801",
    "file": "experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md",
    "sha256": "ef61081fcbd6917368d62da065843e3b397abec0149dfed5c0cc99e77fbdc6d2",
    "git_blob": "29969b938050a815bba5ee5dd7a1db6d0f23adcb",
    "only_path_changed": true
  },
  "prior_audits": {
    "R536_sha256": "656b6e23ce1354eed79fc3887aee63faaf77e343838fc20a4bb8b339eccd75c6",
    "R537_sha256": "0953e717b33dda7e77510dfeb12408ba08c75e5d203451a0aa0e068bc1f319c8"
  },
  "technical_verdict": "PASS",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "r537_resolution": {
    "R537-01": "RESOLVED",
    "R537-02": "RESOLVED",
    "R537-03": "RESOLVED"
  },
  "source_hashes": {
    "matched": 29,
    "declared": 29
  },
  "relational_target_shuffle_total": true,
  "relational_view_pairing_preserved": true,
  "private_counterfactual_suite_executable_without_scientific_reauthorization": true,
  "single_hard_semantics": "HARD_MAP_SET",
  "set_valued_four_cells_numerically_executable": true,
  "checker_specification_sufficient": true,
  "replay_cycle_free": true,
  "budget_contract_terminal": true,
  "new_material_findings": false,
  "plan_modified": false,
  "implementation_performed": false,
  "lockbox_or_sealed_monitor_opened": false,
  "historical_monitor_opened": false,
  "gpu_used_or_queried": false,
  "architecture_promoted": false,
  "scientific_decision": null,
  "decision_authority": "user"
}
```
