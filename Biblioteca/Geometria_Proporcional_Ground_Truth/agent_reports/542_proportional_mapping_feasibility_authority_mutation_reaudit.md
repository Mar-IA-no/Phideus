# R542 — Reauditoría focal de autoridad y mutaciones de `MAPPING-FEASIBILITY`

## Dictamen técnico: REVISE

La corrección `4c62560` resuelve el MEDIUM de autoridad M1 abierto en R541.
El checker ya no autoriza una bijección por declaraciones nominales: exige una
fuente ligada por path y SHA-256, carga triples materializados con keyset exacto
`eiv/set_valued/relational`, comprueba unicidad 1:1:1, recompone conteos y digest
y contrasta todo con el bridge. El estado inconsistente exacto de R541 —un
namespace textual común, `source_id:SYNTHESIS`, conteos ficticios de una unidad
y 64 letras `z` como supuesto digest— quedó rechazado por M1 con
`UNIT_BIJECTION_INCOMPLETE`, mientras `artifact_status` permaneció correctamente
en `PASS` y la decisión pre en `null`.

El segundo MEDIUM mejoró, pero no está resuelto. El recibo canónico ya no usa
`predicate_contract`: enumera exactamente 67 ramas/17 predicados, deriva los
reason codes mediante funciones que también usa `compute`, conserva evidencia
estructurada con digest y registra función y campo mutado. Sin embargo, las 44
ramas R/S se prueban inyectando diccionarios de facts directamente en
`native_reason_codes`, sin alterar los campos o artefactos desde los que
producción calcula esas facts. Varias de ellas están incluso fijadas a
constantes en la ruta productiva, por lo que el test las marca como cubiertas
aunque el reason code sea inalcanzable desde el estado material correspondiente.

Un probe aislado lo confirmó: añadir `mechanism.npy` al árbol público produjo
el fallo técnico esperado y `R1:GRAPH_SCHEMA_INVALID`, pero R2 permaneció
`PASS`; no apareció `GRAPH_PRIVATE_LEAKAGE` porque producción fija
`private_field_exposed:false`. La suite, en cambio, acredita esa rama cambiando
manualmente el booleano a `true`. Por ello queda **un MEDIUM** abierto.

Conteo final: **0 HIGH, 1 MEDIUM, 0 LOW**. El run development completo,
determinismo paired, replay, cierre terminal, seis corrupciones integrales y
claims de no promoción permanecieron correctos.

## Identidad y alcance

- Instancia: **GPT-5.6-Sol**, esfuerzo **high**.
- Target exacto: `4c625602763fb0ba64f271c48e6858f11849c624`.
- Parent: `74d42a0253a5cd3263d42bb07e59de801cec714a`.
- Tree: `a8305d5a454dc5a10716487ffa5bfb1e1ba64dd5`.
- Patch binario `parent..target`: SHA-256
  `76fef6207fce9d863c936b52f73b017ce370aaa6baef06441809e2800cb33b64`.
- Pathset del target: checker y tests; SHA-256 del listado ordenado
  `abd35de936a4987278b4bfd7626378a034e63c400e20f2266732c9410b2d0394`.
- Se leyeron completos el checker vigente, los tests, runner, configuración,
  plan aprobado y R541 antes del dictamen.
- No se modificaron implementación, configuración, tests, datos ni
  documentación canónica. No se abrió monitor/lockbox ni se usó o consultó
  GPU/CUDA.

| Path | SHA-256 |
|---|---|
| `experiments/geometria_proporcional/check_proportional_mapping_feasibility.py` | `7b90b0b9698e74829b0fd97c76989c6903496545471fa5b615f9eb50a76e47bc` |
| `tests/test_proportional_mapping_feasibility.py` | `add035dafe7c4eb1a519b148de814ba2b3d22ee891aedb8ee721058224905ce0` |
| `experiments/geometria_proporcional/run_proportional_mapping_feasibility.py` | `ce9372185a35526f1aced188640402b4726d2224338f70919619e5818b813df9` |
| `experiments/geometria_proporcional/configs/proportional_mapping_feasibility_v1.json` | `d49ab5a181af00134237153277d8085e93dfa0521862a291de368c480015972e` |
| `experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md` | `ef61081fcbd6917368d62da065843e3b397abec0149dfed5c0cc99e77fbdc6d2` |
| R541 | `f33dcdc9ad1e523bb4ba82137879ccfef86de75c4941f90998ffa3f14292e0ef` |

## Resolución individual de R541

### R541 MEDIUM-02 — RESUELTO

`verify_common_unit_authority` exige que el contrato exista en configuración,
que `source_id/path/sha256` coincidan con una fuente ligada y que los bytes
actuales conserven el hash (`check_proportional_mapping_feasibility.py:391-406`).
Luego carga una lista no vacía de triples, exige el keyset cerrado y strings en
los tres lados, calcula cardinalidades únicas y recompone el digest canónico
(`ibid.:407-416`). El bridge sólo pasa con:

- unicidad de cada lado igual al número total de triples;
- `kind:authority_bijection`, `total:true`, `synthetic:false`;
- `source_id` exacto;
- cuatro conteos iguales a los recomputados;
- digest hexadecimal de 64 caracteres igual al digest recomputado
  (`ibid.:417-429`).

`common_unit_contract` agrega después query literal, namespace común y reason
codes causales (`ibid.:432-451`). El probe sobre el estado exacto de R541
obtuvo:

```text
source_status=PASS
artifact_status=PASS
checker_status=PASS
replay_status=NOT_RUN
M1_QUERY_UNIT=FAIL
reason_codes=[UNIT_BIJECTION_INCOMPLETE]
mapping_decision=null
authority.contract_declared=false
authority.materialized=false
```

No hubo degradación indebida del artifact: la inconsistencia corresponde al
predicado semántico M1, no a la integridad del run.

Un segundo probe creó bajo `/tmp` una fuente ligada con dos triples 1:1:1. El
caso material válido pasó `verify_common_unit_authority` y
`common_unit_contract`; su digest recomputado fue
`e6c8e5da9f465f0fab7a24dc522cffbd5b3066aa63fd984c428bef18711a08e2`.
Variantes aisladas con digest no hexadecimal, conteo incorrecto,
`total:false`, keyset extra o clave duplicada fueron rechazadas. Esto satisface
la condición focal de autoridad; no afirma que el run nominal posea una
bijección común —correctamente no la posee—.

### R541 MEDIUM-01 — ABIERTO, ACOTADO A LAS MUTACIONES R/S

#### Avance válido

La corrección elimina por completo el antiguo modo de labels
`predicate_contract`. Para M1–M5, el test altera un campo semántico aislado y
llama a la misma función de contrato usada por `compute`:
`common_unit_contract`, `common_observation_contract`,
`common_target_contract`, `common_decision_contract` o
`authority_phase_contract` (`tests/test_proportional_mapping_feasibility.py:159-236`;
`check_proportional_mapping_feasibility.py:1393-1403`).

El recibo v3 observado contiene:

```text
predicate_mutations=67
predicate_ids=17
pares únicos (id,reason)=67
reason_codes simples y exactos=67/67
mapping_decision null=67/67
digests de evidence válidos=67/67
predicate_contract presente=false
```

Las seis corrupciones integrales `hash/keyset/shape/join/predicate/decision`
se ejecutan sobre copias completas del run mediante el checker y todas quedan
`REJECTED` (`tests/test_proportional_mapping_feasibility.py:294-390`). El caso
`renaming_only` también permanece rechazado con
`SYNTHETIC_ID_EQUIVALENCE` y artifact PASS.

#### Finding MEDIUM — 44 ramas R/S no atraviesan la recomputación productiva

Para R1–R6 y S1–S6, el test fabrica un diccionario de facts base, cambia un
booleano o recibo y llama directamente a `native_reason_codes`; 44 de las 67
filas canónicas siguen esta ruta (`tests/test_proportional_mapping_feasibility.py:238-265`).
Es mejor que etiquetar el reason code directamente porque usa el derivador
productivo, pero no constituye la mutación material aislada exigida por el
plan y R541: no atraviesa `recompute_graph` o `recompute_set`, que son quienes
observan arrays, joins, outputs, recetas y controles antes de construir facts
(`check_proportional_mapping_feasibility.py:1388-1411`).

La diferencia no es sólo formal. Producción fija como constantes:

- `R2.private_field_exposed:false`, `R4.recipe_equal:true`,
  `R4.truth_used:false` y `R5.mechanism_used:false`
  (`check_proportional_mapping_feasibility.py:1083-1089`);
- `S2.cells_present:true`, `S2.utility_used:false`,
  `S5.target_used_as_input:false` y
  `S6.posterior_reader_entangled:false` (`ibid.:825-831`).

El test declara cubiertas las ramas opuestas cambiando esas facts manualmente
(`tests/test_proportional_mapping_feasibility.py:249-265`). Por lo menos estos
ocho reason codes no están acreditados por mutaciones que entren por la lógica
que calcula la condición productiva.

El probe material añadió `mechanism.npy` desde privado a uno de los directorios
públicos de grafo. El checker respondió:

```text
artifact_status=FAIL
mapping_decision=null
R1_SOURCE_COMPLETE=FAIL [GRAPH_SCHEMA_INVALID]
R2_PUBLIC_PARITY=PASS []
R2 observed.private_field_exposed=false
```

Es correcto que el estado total sea rechazado, pero es incorrecta la claim de
cobertura exacta de `GRAPH_PRIVATE_LEAKAGE`: el campo material que representa
esa condición no llega a la fact que gobierna el reason code.

**Corrección requerida.** Las mutaciones R/S deben modificar arrays,
manifests, evidencia, joins o superficies ejecutoras aisladas y atravesar la
misma función que construye las facts productivas. Alternativamente, cada fact
puede extraerse mediante un helper puro compartido por recomputación y test,
pero la entrada del test debe ser el objeto material pertinente, no el booleano
final. El recibo debe registrar la observación recomputada y fallar si una rama
sólo puede alcanzarse alterando el diccionario posterior.

## Run development y cierre

```text
venv/bin/python experiments/geometria_proporcional/run_proportional_mapping_feasibility.py \
  --development --output /tmp/r542.sdc47t/nominal
=> exit 0
=> artifact_status=PASS
=> wall_seconds=204.34837182238698
=> max_ru_maxrss_bytes=377614336
=> pytest interno exit 0; 13 tests; 111.04442333802581 s
=> core files compared=148; mismatches=[]
=> closure core/replay/terminal=true
=> mapping_decision=BIFURCATE_NATIVE_CONTRASTS
```

Los pares `core_manifest`, `mutation_results`, `mapping_matrix`,
`pre_adjudication`, `adjudication`, informe científico y
`scientific_manifest` fueron byte-exactos entre `run_a` y `run_b`.
`mutation_results.json` tuvo SHA-256 idéntico
`35fb48c83e0d500c9328a9e6ea7f1d5e61750e2228ebc22c731a23e41dc037f6`.

Los estados finales fueron source/artifact/checker/replay `PASS`; replay
comparó 148 archivos sin diferencias y los tres checks de cierre fueron
`true`. Todos los recibos relevantes preservaron:

```json
{"gpu_used_or_queried":false,"architecture_promoted":false,
 "scientific_decision":null,"decision_authority":"user"}
```

No se declara promoción arquitectónica ni `GO/NO-GO`.

## Resultado consolidado

| Condición focal | Resultado |
|---|---|
| estado inconsistente exacto R541 rechazado por M1 | PASS |
| artifact permanece PASS en ese probe | PASS |
| autoridad material ligada y recomputada | PASS |
| keyset, unicidad, conteos y digest verificados | PASS |
| 17 predicados / 67 ramas inventariados | PASS |
| M1–M5 por funciones de contrato productivas | PASS |
| R/S por mutación material y recomputación productiva | REVISE |
| seis corrupciones integrales ejecutadas/rechazadas | PASS |
| paired determinism / replay / terminal closure | PASS |
| no promoción / decisión científica reservada | PASS |
| uso o consulta GPU/CUDA | no |

```json
{
  "schema_version": "proportional-mapping-feasibility-authority-mutation-reaudit-v1",
  "audit_id": "R542",
  "runtime_configuration": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
  "target": {
    "commit": "4c625602763fb0ba64f271c48e6858f11849c624",
    "parent": "74d42a0253a5cd3263d42bb07e59de801cec714a",
    "tree": "a8305d5a454dc5a10716487ffa5bfb1e1ba64dd5",
    "path_count": 2,
    "pathset_sha256": "abd35de936a4987278b4bfd7626378a034e63c400e20f2266732c9410b2d0394",
    "binary_patch_sha256": "76fef6207fce9d863c936b52f73b017ce370aaa6baef06441809e2800cb33b64"
  },
  "technical_verdict": "REVISE",
  "findings": {"high": 0, "medium": 1, "low": 0},
  "r541_resolution": {
    "MEDIUM-01": "OPEN_NARROWED_TO_RS_MATERIAL_MUTATIONS",
    "MEDIUM-02": "RESOLVED"
  },
  "r541_inconsistent_authority_probe": {
    "artifact_status": "PASS",
    "m1_status": "FAIL",
    "reason_codes": ["UNIT_BIJECTION_INCOMPLETE"],
    "mapping_decision": null
  },
  "material_authority_probe": {
    "valid_passed": true,
    "invalid_variants_rejected": ["nonhex_digest", "wrong_count", "not_total", "wrong_keyset", "duplicate_key"]
  },
  "mutation_receipt": {
    "schema_version": "proportional-mapping-mutation-execution-v3",
    "predicate_ids": 17,
    "reason_rows": 67,
    "exact_reason_pairs": 67,
    "rs_rows_using_injected_facts": 44,
    "candidate_corruptions_executed_and_rejected": 6,
    "run_a_run_b_sha256": "35fb48c83e0d500c9328a9e6ea7f1d5e61750e2228ebc22c731a23e41dc037f6"
  },
  "nominal_run": {
    "exit_code": 0,
    "wall_seconds": 204.34837182238698,
    "max_ru_maxrss_bytes": 377614336,
    "tests_collected": 13,
    "tests_returncode": 0,
    "core_files_compared": 148,
    "core_mismatches": 0,
    "mapping_decision": "BIFURCATE_NATIVE_CONTRASTS"
  },
  "gpu_used_or_queried": false,
  "monitor_or_lockbox_opened": false,
  "implementation_config_tests_data_docs_modified": false,
  "architecture_promoted": false,
  "scientific_decision": null,
  "decision_authority": "user"
}
```
