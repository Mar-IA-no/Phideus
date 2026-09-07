# R543 — Reauditoría focal de mutaciones materiales de `MAPPING-FEASIBILITY`

## Dictamen técnico: PASS

La corrección acumulada hasta `07b972d` resuelve el único MEDIUM abierto en
R542. Las 44 ramas nativas R/S ya no se acreditan alterando directamente
booleans del diccionario de facts. Tests y producción comparten
`derive_native_facts`: los tests modifican objetos materiales y el checker
productivo construye esos mismos tipos de entrada desde arrays, nombres de
archivos, recibos, código fuente parseado como AST, recetas, resultados de
replay, joins, estimandos, controles y máscaras de soporte. Recién después se
derivan las facts consumidas por `native_reason_codes`.

Las ocho facts que R542 identificó como constantes quedaron materializadas:
exposición privada R2 desde nombres públicos; igualdad de receta y uso de truth
R4 desde receta y AST; uso de mecanismo R5 desde argumentos del solver; celdas
y uso de utility S2 desde masas y fuente del posterior; uso de target S5 desde
la fuente del posterior; y entrelazamiento del reader S6 desde la fuente de la
métrica. No encontré facts productivas constantes ni ramas acreditadas por
inyección directa del resultado final.

El probe aislado requerido también cambió exactamente como debía: copiar
`mechanism.npy` desde privado al estado público `raw_generic__seed=104729`
produjo `artifact_status:FAIL`, decisión `null`, R1 `FAIL` con el único reason
`GRAPH_SCHEMA_INVALID` y R2 `FAIL` con el único reason
`GRAPH_PRIVATE_LEAKAGE`.

Conteo final: **0 HIGH, 0 MEDIUM, 0 LOW**.

## Identidad y alcance

- Instancia: **GPT-5.6-Sol**, esfuerzo **high**.
- Target exacto: `07b972de755acc08022950d92c65e9224ff72a5e`.
- Parent: `807c93dc8f85b097efa8f6737d444732e7f37831`.
- Tree: `2f06254531b063503c10f4b13b532ad6d5a30fe4`.
- Baseline de la corrección focal: `4c625602763fb0ba64f271c48e6858f11849c624`.
- Patch binario relevante `4c62560..07b972d`, limitado a checker, runner y
  tests: SHA-256
  `967e447e7d817c35e914082c5f9765085150e0f11b6fec7a5bd5851f11c0cd2b`.
- Pathset relevante de tres archivos: SHA-256
  `7b21952461abb2075202460a209c8a6efaa1c2132f082d8f43555cdb5a679ddd`.
- Se leyeron completos checker, runner, tests y R542 vigentes antes del
  dictamen. También se contrastó el patch acumulado relevante.
- No se modificaron implementación, configuración, tests, datos ni
  documentación canónica. No se abrió monitor/lockbox ni se usó o consultó
  GPU/CUDA.

| Path | SHA-256 |
|---|---|
| `experiments/geometria_proporcional/check_proportional_mapping_feasibility.py` | `0ba8ee0b23863b32bc41f0a8d5947b3e2b15f3915381c90fd0e89415ce5fef01` |
| `experiments/geometria_proporcional/run_proportional_mapping_feasibility.py` | `0ea0b80bbb79ae6b9a898a865b498ed5caaf6e8ccdfd78ea6dc9cd20b2ad9d29` |
| `tests/test_proportional_mapping_feasibility.py` | `4150695a87f2be37516726bf22a8248c541b0fce3cd55a0de191b0156ffc0f7f` |
| R542 | `a62871cdb3556fcb9412b6aa7305adec6de59679fa23c64ec61317b4d69f5cc6` |
| config congelada | `d49ab5a181af00134237153277d8085e93dfa0521862a291de368c480015972e` |
| plan aprobado | `ef61081fcbd6917368d62da065843e3b397abec0149dfed5c0cc99e77fbdc6d2` |

## Resolución del MEDIUM de R542

### Una sola derivación para producción y tests

`recompute_set` arma materiales para S1–S6 y los entrega a
`derive_native_facts` (`check_proportional_mapping_feasibility.py:839-856`).
`recompute_graph` hace lo mismo para R1–R6 (`ibid.:1144-1159`). `compute`
consume luego esas facts por la única ruta `native_reason_codes`, añadiendo a
R1/S1 los recibos reales de sources (`ibid.:1594-1617`). Los recibos siguen
siendo registros materiales; no son verdict booleans.

El helper puro deriva las 44 condiciones en
`check_proportional_mapping_feasibility.py:1332-1424`. El test focal crea
materiales base, modifica un campo material por caso, deriva facts y exige el
reason code simple exacto (`tests/test_proportional_mapping_feasibility.py:238-324`).
La única llamada a `native_reason_codes` dentro del test está después de
`derive_native_facts`; no hay asignaciones posteriores a facts ni tablas de
booleans finales inyectados.

| Predicado | Ramas | Material que gobierna las facts |
|---|---:|---|
| R1 | 3 | recibos, estados, nombres públicos/privados, digests fuente |
| R2 | 3 | arrays paired, campos de paridad, nombres públicos prohibidos |
| R3 | 3 | presencia, finitud y shapes de arrays de representación |
| R4 | 4 | AST de llamadas del solver, recetas y pares de replay |
| R5 | 3 | incidencia/targets/gauge y AST de argumentos del solver |
| R6 | 3 | pares de estimandos, controles y máscara de soporte |
| S1 | 4 | recibos, arrays fuente, shapes y roles |
| S2 | 4 | keyset/masas posteriores, joins y fuente del posterior |
| S3 | 4 | cuatro arrays de acciones, shapes, digests y declaración de duplicación |
| S4 | 5 | contextos de fit/selección, clases, roles y estado de fase |
| S5 | 4 | join de keys, target, utility y fuente del posterior |
| S6 | 4 | estimandos, fuente de métricas, controles y soporte |
| **Total R/S** | **44** | **entradas materiales compartidas** |

El test unitario focal ejecutado aisladamente pasó:

```text
pytest -q tests/test_proportional_mapping_feasibility.py::test_real_mutations_execute_production_condition_functions
=> 1 passed in 0.19s
```

### Ocho constantes anteriores

Las condiciones señaladas por R542 ahora se obtienen así:

- R2 busca stems privados dentro de los nombres públicos observados
  (`check_proportional_mapping_feasibility.py:1342-1347`).
- R4 compara receta registrada/esperada y extrae de AST las firmas y
  argumentos efectivos de WLS/IRLS (`ibid.:1353-1363`).
- R5 detecta `mechanism` en argumentos del solver (`ibid.:1364-1372`).
- S2 exige el keyset material `MARGINAL/JOINT` y busca `utility` en la fuente
  efectiva del posterior (`ibid.:1384-1390`).
- S5 deriva join/no-vacío/contrato de utility y busca `target` en la misma
  fuente (`ibid.:1412-1418`).
- S6 compara estimandos/controles, inspecciona la fuente de la métrica y mide
  soporte real (`ibid.:1419-1424`).

Esto satisface la alternativa de corrección explicitada por R542: helper puro
compartido, con objetos materiales como entrada del test y no facts finales.

## Probe material público `mechanism.npy`

Sobre una copia del `run_a` nominal se copió únicamente:

```text
prepared/private_dev/graph/raw_generic__seed=104729/mechanism.npy
→ prepared/public/graph/raw_generic__seed=104729/mechanism.npy
```

El checker pre produjo:

```json
{
  "technical_status": {
    "source_status": "PASS",
    "artifact_status": "FAIL",
    "checker_status": "PASS",
    "replay_status": "NOT_RUN"
  },
  "mapping_decision": null,
  "R1_SOURCE_COMPLETE": {
    "status": "FAIL",
    "reason_codes": ["GRAPH_SCHEMA_INVALID"]
  },
  "R2_PUBLIC_PARITY": {
    "status": "FAIL",
    "reason_codes": ["GRAPH_PRIVATE_LEAKAGE"]
  }
}
```

El mecanismo público llega, por tanto, a la ruta productiva R2 y ya no sólo a
un boolean fabricado por el test. El test integral que fija esta regresión está
en `tests/test_proportional_mapping_feasibility.py:333-350`.

## Recibo v4 y exigencia del runner

El recibo nominal observado conserva:

```text
schema_version=proportional-mapping-mutation-execution-v4
status=PASS
predicate_ids=17
predicate_mutations=67
pares únicos (id,reason_codes)=67
derive_native_facts+native_reason_codes=44
mapping_decision null=67/67
mutated_field no vacío=67/67
```

Las otras 23 ramas pasan por los cinco contratos M productivos. El SHA-256 de
`mutation_results.json` fue byte-exacto en ambos runs:
`1afeaf0761392352c06358bd2793c72ce949d6c341fe2de438e4ac9a8a0b759f`.

El runner exige v4, 67 filas y pares únicos, decisiones nulas, función distinta
de la llamada desnuda a `native_reason_codes`, campo material no vacío y seis
corrupciones rechazadas (`run_proportional_mapping_feasibility.py:217-231`).
El recibo observado rechazó `hash`, `keyset`, `shape`, `join`, `predicate` y
`decision`; el caso M1 de renombrado sintético quedó además rechazado con
`SYNTHETIC_ID_EQUIVALENCE`.

## Run development, determinismo y cierre

```text
venv/bin/python experiments/geometria_proporcional/run_proportional_mapping_feasibility.py \
  --development --output /tmp/r543.n6xMBO/nominal
=> exit 0
=> artifact_status=PASS
=> wall_seconds=219.08160492032766
=> max_ru_maxrss_bytes=392138752
=> pytest interno exit 0; archivo con 14 tests; 125.76205394417048 s
=> replay status=PASS; files_compared=148; mismatches=[]
=> closure core_manifest/replay/terminal_status=true/true/true
=> mapping_decision=BIFURCATE_NATIVE_CONTRASTS
```

`core_manifest` tuvo 148 archivos y
`scientific_manifest` 153. `core_manifest`, `mutation_results`,
`mapping_matrix`, `pre_adjudication`, `adjudication`, informe científico y
`scientific_manifest` fueron byte-exactos entre `run_a` y `run_b`. Los estados
finales source/artifact/checker/replay fueron todos `PASS`; ambos cierres
terminales registraron `artifact_status:PASS`.

Todos los recibos relevantes conservaron:

```json
{"gpu_used_or_queried":false,"architecture_promoted":false,
 "scientific_decision":null,"decision_authority":"user"}
```

La decisión observada es una salida de la algebra técnica ya aprobada; esta
reauditoría no promueve arquitectura ni declara `GO/NO-GO` científico.

## Resultado consolidado

| Condición focal | Resultado |
|---|---|
| R542 MEDIUM: 44 ramas R/S sin facts booleanas inyectadas | RESUELTO |
| helper material compartido por producción y tests | PASS |
| ocho facts antes constantes derivadas desde material | PASS |
| probe público `mechanism.npy`: R1 schema + R2 leakage exactos | PASS |
| v4: 17 predicados / 67 pares únicos y materiales | PASS |
| seis corrupciones integrales ejecutadas/rechazadas | PASS |
| paired determinism / replay / terminal closure | PASS |
| no promoción / decisión científica reservada | PASS |
| uso o consulta GPU/CUDA | no |

```json
{
  "schema_version": "proportional-mapping-feasibility-material-mutation-reaudit-v1",
  "audit_id": "R543",
  "runtime_configuration": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
  "target": {
    "commit": "07b972de755acc08022950d92c65e9224ff72a5e",
    "parent": "807c93dc8f85b097efa8f6737d444732e7f37831",
    "tree": "2f06254531b063503c10f4b13b532ad6d5a30fe4",
    "baseline": "4c625602763fb0ba64f271c48e6858f11849c624",
    "relevant_path_count": 3,
    "relevant_pathset_sha256": "7b21952461abb2075202460a209c8a6efaa1c2132f082d8f43555cdb5a679ddd",
    "relevant_binary_patch_sha256": "967e447e7d817c35e914082c5f9765085150e0f11b6fec7a5bd5851f11c0cd2b"
  },
  "technical_verdict": "PASS",
  "findings": {"high": 0, "medium": 0, "low": 0},
  "r542_resolution": {"MEDIUM-01": "RESOLVED"},
  "material_mutations": {
    "schema_version": "proportional-mapping-mutation-execution-v4",
    "predicate_ids": 17,
    "reason_rows": 67,
    "unique_reason_pairs": 67,
    "rs_material_rows": 44,
    "direct_rs_fact_injections": 0,
    "candidate_corruptions_executed_and_rejected": 6,
    "run_a_run_b_sha256": "1afeaf0761392352c06358bd2793c72ce949d6c341fe2de438e4ac9a8a0b759f"
  },
  "public_mechanism_probe": {
    "artifact_status": "FAIL",
    "mapping_decision": null,
    "r1_status": "FAIL",
    "r1_reason_codes": ["GRAPH_SCHEMA_INVALID"],
    "r2_status": "FAIL",
    "r2_reason_codes": ["GRAPH_PRIVATE_LEAKAGE"]
  },
  "nominal_run": {
    "exit_code": 0,
    "wall_seconds": 219.08160492032766,
    "max_ru_maxrss_bytes": 392138752,
    "tests_in_file": 14,
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
