# R541 — Reauditoría focal final de `MAPPING-FEASIBILITY`

## Dictamen técnico: REVISE

La corrección `63f8f44` resuelve el caso concreto de renaming sintético señalado
por R540: con query intacta, un namespace textual común y
`{"kind":"renaming_only"}`, el checker mantiene `artifact_status:PASS`, produce
`M1_QUERY_UNIT:FAIL` con el único reason code
`SYNTHETIC_ID_EQUIVALENCE` y conserva `mapping_decision:null` en pre.

No obstante, ninguno de los dos MEDIUM de R540 queda cerrado por completo. M1
todavía acepta una declaración de bijección que no está acreditada por una
correspondencia material: basta nombrar cualquier `source_id` congelado,
declarar `total:true`, repetir un entero positivo en cuatro conteos y aportar
cualquier cadena de 64 caracteres. En una copia temporal, conteos ficticios de
una unidad y un supuesto digest no hexadecimal produjeron
`bridge_authorized:true` y `M1_QUERY_UNIT:PASS`, aunque no existía lista de
correspondencias ni evidencia fuente que soportara esos valores.

La nueva suite serializa 67 filas para los 17 predicados y las seis corrupciones
separadas del candidato sí se ejecutan y se rechazan. Sin embargo, las 67 ramas
de predicado siguen siendo labels fabricados: el test añade un diccionario
booleano `predicate_contract` al candidato y el checker convierte directamente
el único label `false` en el reason code solicitado. Un probe independiente
creó 67 JSON mínimos sin ejecutar preparer ni builder ni alterar una condición
científica; el checker los aceptó todos y emitió `status:PASS`. Por tanto,
`mutation_results.json` canónico cuenta las ramas del catálogo, pero no
demuestra su ejecución real.

Conteo final: **0 HIGH, 2 MEDIUM, 0 LOW**. No se reabren los seis puntos ya
resueltos en R540: el run development completo volvió a cerrar fuente,
artefactos, replay y estado terminal sin regresión focal observada.

## Identidad y alcance

- Instancia: **GPT-5.6-Sol**, esfuerzo **high**.
- Target exacto: `63f8f44078ebf36f7572881a7ae5b77c6089e70e`.
- Parent: `74417ac652c617aa72fc1fb67b0d0bdb2a4670a0`.
- Tree: `c5a30e4dada5624ccba3e7700bfc032e96eb80f4`.
- Patch binario `R540..target`: SHA-256
  `8829ad80e044a8a91b8d0dca3342789225f324e51f2ec0a668b21aa6c89af6b0`.
- El cambio contiene únicamente checker, runner y tests; digest del pathset
  ordenado: `7b21952461abb2075202460a209c8a6efaa1c2132f082d8f43555cdb5a679ddd`.
- No se modificó implementación, configuración, datos ni documentación
  canónica. No se usó ni consultó GPU/CUDA.

| Path | SHA-256 |
|---|---|
| `experiments/geometria_proporcional/check_proportional_mapping_feasibility.py` | `a0197ff980f6171c6a98184edb8c452a91a63be31bfcac1ca90c430ea74aa8fe` |
| `experiments/geometria_proporcional/run_proportional_mapping_feasibility.py` | `ce9372185a35526f1aced188640402b4726d2224338f70919619e5818b813df9` |
| `tests/test_proportional_mapping_feasibility.py` | `51ad3c49b194ca088d02361c5ef36a862e59757f00c8a90068081f8670bf74b9` |

## Resolución focal de los dos MEDIUM

### R540 MEDIUM-01 — ABIERTO

**Lo corregido.** El runner ya no conserva el placeholder producido por
`mutation_suite`. Después del pytest, exige `status:PASS` y que las seis
corrupciones separadas del candidato tengan valor `REJECTED`
(`run_proportional_mapping_feasibility.py:210-221`). El test ejecuta copias
temporales para `hash`, `keyset`, `shape`, `join`, `predicate` y `decision`, y
comprueba decisión nula y fallo técnico (`tests/test_proportional_mapping_feasibility.py:199-295`).
El artefacto canónico observado contiene exactamente esas seis claves, todas
`REJECTED`.

**Lo pendiente.** Las supuestas mutaciones de las 67 ramas no ejercitan las
condiciones que computan M1–M5, R1–R6 y S1–S6. El test construye siempre el
mismo fixture/candidato TEST_ONLY y luego agrega:

```text
predicate_contract = {reason_del_predicado: reason_del_predicado != elegido}
mutation_case = {id, reason}
```

(`tests/test_proportional_mapping_feasibility.py:141-160`). El checker sólo
verifica que haya una clave `false` y llama a `pred(identifier, False, ...,
[reason])` (`check_proportional_mapping_feasibility.py:886-912`). No llama a
`compute`, `recompute_set` ni `recompute_graph`, no inspecciona evidencia
material propia de la condición y fija para todas las filas el mismo estado
técnico `artifact_status:FAIL/checker_status:PASS`.

El probe de control eliminó incluso la apariencia de una ejecución completa:
creó 67 candidatos mínimos con sólo `schema_version`, `fixture_mode` y
`predicate_contract`, sin preparer ni builder. El checker devolvió:

```text
status=PASS
predicate_mutations=67
predicate_ids=17
reason_pairs=67
all rows status=REJECTED
results SHA-256=128503bdc4f88c063e2dad054bf6509ede6acfc8c1e459047ec02dd9b2b571ad
```

Esto demuestra que el recibo canónico acredita enumeración y eco de labels,
no 67 rechazos causados por 67 condiciones reales. La condición de cierre de
R540 MEDIUM-01 permanece incumplida.

**Corrección requerida.** Cada caso TEST_ONLY debe alterar el campo o artefacto
que materializa una sola condición, ejecutar la función de evaluación real
correspondiente y registrar evidencia observada verificable. El recibo debe
fallar si el caso se reduce a un label booleano. Las seis corrupciones de
candidato actuales pueden conservarse.

### R540 MEDIUM-02 — PARCIALMENTE RESUELTO, PERMANECE ABIERTO

**Renaming exacto: resuelto.** `bridge_authorized` requiere ahora
`kind=authority_bijection`, fuente declarada, totalidad, marca no sintética,
conteos y campo de digest (`check_proportional_mapping_feasibility.py:1060-1077`).
El reason selector asigna `SYNTHETIC_ID_EQUIVALENCE` a un bridge
`renaming_only` (`ibid.:1096-1109`). La ejecución focal reprodujo exactamente:

```text
source_status=PASS
artifact_status=PASS
checker_status=PASS
replay_status=NOT_RUN
M1_QUERY_UNIT=FAIL
reason_codes=[SYNTHETIC_ID_EQUIVALENCE]
mapping_decision=null
```

**Autoridad material: pendiente.** Los chequeos de `bridge_authorized` validan
sólo tipos, igualdad interna y presencia nominal. No verifican:

- que `bijection_sha256` sea un SHA-256 hexadecimal;
- que ese digest corresponda a una lista congelada de triples 1:1:1;
- que `unit_count` coincida con los universos evaluables reales;
- que `authority_source_id` contenga o comprometa esa correspondencia.

En una copia del run nominal se igualaron los namespaces a
`fabricated-common` y se declaró `authority_source_id:SYNTHESIS`, cuatro
conteos iguales a `1` y 64 letras `z` como supuesto digest. Tras mantener
consistentes candidato y evidencia, los estados técnicos permanecieron PASS y
el checker produjo `bridge_authorized:true`, `M1_QUERY_UNIT:PASS`. No había
ningún conjunto de triples ni una fuente que justificara esa unidad.

Además, el defecto de las 67 ramas afecta directamente la prueba de reason
codes: R1–R6 y S1–S6 continúan invocando `pred` sin reasons derivados por
condición (`ibid.:1115-1126`), de modo que el helper usa el primer código del
catálogo cuando fallan (`ibid.:851-859`). Las filas TEST_ONLY no prueban esa
ruta productiva.

**Corrección requerida.** M1 debe cargar desde una fuente congelada la
correspondencia total, validar keyset y unicidad de los tres lados, recomputar
conteos y digest, y comparar esos valores con la declaración. Cada rama de
reason code debe derivarse de una condición observada y contar con una
mutación aislada que llegue por la misma lógica usada en producción.

## Run y checks CPU

```text
venv/bin/python experiments/geometria_proporcional/run_proportional_mapping_feasibility.py \
  --development --output /tmp/r541.UCIlZO/nominal
=> exit 0
=> artifact_status=PASS
=> wall_seconds=204.55109903216362
=> max_ru_maxrss_bytes=376561664
=> pytest interno exit 0; 12 tests; 111.06186028942466 s
=> core files compared=148; mismatches=[]
=> closure core/replay/terminal=true
=> mapping_decision=BIFURCATE_NATIVE_CONTRASTS
=> gpu_used_or_queried=false
```

El `mutation_results.json` de `run_a` y `run_b` fue byte-exacto, SHA-256
`778590b5d2e7910498522d016b96ef2af6a0b173b7049d9e8c0e249d9999bf59`.
Contiene 67 filas, 17 IDs, 67 pares únicos `(id,reason)` y las seis
corrupciones de candidato rechazadas. Esos conteos son correctos como
inventario; el finding reside en qué ejecución sustenta cada fila.

Los árboles temporales usados para los checks se eliminaron al cerrar esta
auditoría.

## Resultado consolidado

| Condición focal | Resultado |
|---|---|
| `renaming_only` → M1 FAIL exacto | PASS |
| `renaming_only` conserva artifact PASS | PASS |
| pre conserva decisión nula | PASS |
| bijección respaldada por autoridad congelada y recomputada | REVISE |
| 17 predicados enumerados | PASS |
| 67 ramas enumeradas | PASS |
| 67 condiciones reales ejecutadas | REVISE |
| seis corrupciones separadas del candidato ejecutadas/rechazadas | PASS |
| `mutation_results.json` byte-exacto entre runs | PASS |
| cierre nominal/replay/terminal | PASS |
| uso o consulta GPU/CUDA | no |

```json
{
  "schema_version": "proportional-mapping-feasibility-final-reaudit-v1",
  "audit_id": "R541",
  "runtime_configuration": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
  "target": {
    "commit": "63f8f44078ebf36f7572881a7ae5b77c6089e70e",
    "parent": "74417ac652c617aa72fc1fb67b0d0bdb2a4670a0",
    "tree": "c5a30e4dada5624ccba3e7700bfc032e96eb80f4",
    "path_count": 3,
    "pathset_sha256": "7b21952461abb2075202460a209c8a6efaa1c2132f082d8f43555cdb5a679ddd",
    "binary_patch_sha256": "8829ad80e044a8a91b8d0dca3342789225f324e51f2ec0a668b21aa6c89af6b0"
  },
  "technical_verdict": "REVISE",
  "findings": {"high": 0, "medium": 2, "low": 0},
  "r540_resolution": {
    "MEDIUM-01": "OPEN",
    "MEDIUM-02": "PARTIALLY_RESOLVED_OPEN"
  },
  "renaming_only": {
    "artifact_status": "PASS",
    "m1_status": "FAIL",
    "reason_codes": ["SYNTHETIC_ID_EQUIVALENCE"],
    "mapping_decision": null
  },
  "unsupported_authority_declaration_accepted_by_m1": true,
  "mutation_receipt": {
    "predicate_ids": 17,
    "reason_rows": 67,
    "real_condition_executions": 0,
    "candidate_corruptions_executed_and_rejected": 6,
    "run_a_run_b_sha256": "778590b5d2e7910498522d016b96ef2af6a0b173b7049d9e8c0e249d9999bf59"
  },
  "nominal_run": {
    "exit_code": 0,
    "wall_seconds": 204.55109903216362,
    "max_ru_maxrss_bytes": 376561664,
    "tests_collected": 12,
    "tests_returncode": 0,
    "core_files_compared": 148,
    "core_mismatches": 0,
    "mapping_decision": "BIFURCATE_NATIVE_CONTRASTS"
  },
  "gpu_used_or_queried": false,
  "implementation_plan_docs_or_data_modified": false,
  "architecture_promoted": false,
  "scientific_decision": null,
  "decision_authority": "user"
}
```
