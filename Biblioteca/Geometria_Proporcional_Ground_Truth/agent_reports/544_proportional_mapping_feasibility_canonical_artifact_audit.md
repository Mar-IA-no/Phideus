# R544 — Auditoría final de artefactos y adjudicación canónicos de `MAPPING-FEASIBILITY`

## Dictamen técnico: PASS

El paquete canónico commiteado en `d763626` satisface el cierre material de las
secciones 9–11 del plan. Los 313 archivos existen como bytes del target, el
working tree coincide con esos blobs y no hay paths ajenos al paquete en el
commit. Los manifests público, privado, core y científico tienen pathsets
cerrados; cada tamaño y SHA-256 declarado coincide; cada NPY conserva dtype y
shape; y los 43 JSON del paquete están serializados en forma canónica.

Los dos runs son científicamente deterministas. Sus 148 archivos core y sus
153 archivos científicos son byte-exactos, los manifests correspondientes son
idénticos y el replay declara 148/148 sin mismatch. La recomputación read-only
del checker sobre ambos runs coincidió exactamente con los
`pre_adjudication.json` commiteados y validó core, replay y cierre terminal.

La adjudicación también es exacta: M1–M4 fallan con los reason codes canónicos;
M5, R1–R6 y S1–S6 pasan. Antes del replay la decisión es `null`; tras los cuatro
estados técnicos `PASS`, la álgebra produce
`BIFURCATE_NATIVE_CONTRASTS`. El recibo v4 contiene 67 pares únicos sobre 17
predicados, 44 ramas R/S materiales y las seis corrupciones integrales
rechazadas.

Conteo final: **0 HIGH, 0 MEDIUM, 0 LOW**.

Esta conclusión es estrictamente una **hoja técnica de compatibilidad de
contratos**. No promueve una arquitectura y no constituye un `GO/NO-GO`
científico.

## Identidad y alcance

- Instancia: **GPT-5.6-Sol**, esfuerzo **high**.
- Target exacto: `d763626d017f0a56a0276bee675c3779ac0f553a`.
- Parent: `32ddf37731be4dc725ef31b1f27e6dab92bec645`.
- Tree: `59e60066030e6eb783fa54a2f7ef2543f28fdf81`.
- Pathset del commit: 313 archivos, todos bajo
  `data/geometria_proporcional/proportional_mapping_feasibility_v1/`.
- SHA-256 del pathset ordenado:
  `65493e0f5e646bfdf9e0af7f19378f73f6b1049d1248cb0dd302746a2d514b06`.
- Tamaño agregado de los blobs materializados: `26186051` bytes.
- SHA-256 de `manifest.json` del paquete:
  `ac9e3a8befa7b9af6e0c0724fe0e872cd1ec6bddd9c980067d48c9664e6603c3`.
- Se leyeron las secciones 9–11 del plan y el código vigente completo; se
  recorrieron los bytes de todos los artefactos según sus manifests.
- No se regeneró ni modificó ningún artefacto. Sólo se ejecutaron validadores
  read-only. No se abrió monitor/lockbox ni se usó o consultó GPU/CUDA.

## Inventario y manifests desde bytes

La raíz tiene exactamente tres archivos (`manifest.json`,
`replay_evidence.json`, `runtime.json`) y dos directorios de 155 archivos cada
uno. Por run:

| Clase | Conteo | Resultado |
|---|---:|---|
| payload público declarado | 64 | 64 presentes, tamaños/hashes/NPY metadata válidos |
| manifest público | 1 | schema, pathset y pathset digest válidos |
| payload privado declarado | 74 | 74 presentes, tamaños/hashes/NPY metadata válidos |
| manifest privado | 1 | schema, pathset y pathset digest válidos |
| archivos core | 148 | 148 presentes y válidos |
| archivos científicos | 153 | 153 presentes y válidos |
| total del run | 155 | exacto |

La allowlist pública es la implementada por preparer/builder: tres JSON de
contrato, cinco arrays W54 y catorce arrays por cada uno de cuatro estados de
grafo (`prepare_proportional_mapping_feasibility.py:13-24,71-88,274-326`;
`build_proportional_mapping_candidate.py:24-32,52-78`). La allowlist privada
es un JSON W49, cinco arrays W54 y diecisiete arrays por estado. No apareció
ningún path adicional ni faltante; en particular, campos de target o mecanismo
no aparecen en público.

Los manifests preparados coinciden en ambos runs:

| Manifest | SHA-256 run_a = run_b |
|---|---|
| público | `30bb8132e66d6e327c3c67480d5e3b4fe99aa1de22c2822f58a93d3baab5e40a` |
| privado | `626f04963a1d99dc0f5f1524a0a19510ca2cd51883b6f5a417b325e430e95cea` |
| core | `52f0ec6ed0c1724fde2e123cc4c6878610f5591187f23f88e9e33ae0f042a303` |
| científico | `23511c37962f20eeb1f3e0dcff08287b76fe576fbb09c4a50813efd099279fd6` |

La fase core precede por diseño a replay, terminal y adjudicación final; por
eso esos artefactos posteriores no están en sus 148 entradas. El manifest
científico final sí cubre todo el run salvo `runtime.json` y a sí mismo, tal
como prescribe el plan (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:361-404`).

Se parsearon y recodificaron los 43 JSON con keys ordenadas, separadores
compactos, ASCII, `allow_nan=false` y newline final: todos fueron byte-exactos
respecto de esa forma canónica. Los 36 JSON contenidos en los dos manifests
científicos tampoco contienen paths absolutos. Los runtime quedan fuera de la
clase científica determinista, conforme al plan.

## Fuentes y congelamiento

`source_inventory.json` es byte-exacto entre runs y registra 30 recibos: 29
fuentes de config más el plan. Todos los archivos existen, sus hashes actuales
coinciden simultáneamente con `expected_sha256` y `actual_sha256`, y no hay
status distinto de `PASS`. El recibo semántico adicional `SOURCE_POLICY`
también pasa.

El manifest raíz liga seis fuentes de ejecución y todos sus hashes se
recalcularon correctamente:

| Fuente | SHA-256 |
|---|---|
| preparer | `6610051f13076274bd5b1cd99201ffb3f4e636c720076ddf636a4c90b9f83be7` |
| builder | `6be4026c763498fb13eea966d0b6a7aef3641bc6170f6ee3d59a79ede4e742d6` |
| evaluator | `fe2cd554f39a5a328aae79e09d16ddb2f90cf46eff844984e238f68cb29422c1` |
| checker | `0ba8ee0b23863b32bc41f0a8d5947b3e2b15f3915381c90fd0e89415ce5fef01` |
| runner | `0ea0b80bbb79ae6b9a898a865b498ed5caaf6e8ccdfd78ea6dc9cd20b2ad9d29` |
| config | `d49ab5a181af00134237153277d8085e93dfa0521862a291de368c480015972e` |

El builder declaró `prepared_public_only`, abrió exactamente 65 paths únicos
—los 64 payloads más su manifest— y no registró ningún path privado. El
evaluator declaró que el candidate quedó congelado antes del acceso privado y
su SHA-256 coincide con el archivo materializado.

## Determinismo, replay y cierre terminal

El `core_manifest.json` es byte-exacto entre runs y todos sus 148 paths tienen
bytes idénticos. La misma prueba sobre `scientific_manifest.json` y sus 153
entradas también pasa. Entre otros, son byte-exactos:

```text
pre_adjudication.json
mapping_matrix.json
mutation_results.json
core_manifest.json
replay_evidence.json
terminal_status.json
adjudication.json
REPORT_MAPPING_FEASIBILITY.md
scientific_manifest.json
```

El replay raíz y sus copias por run son byte-exactos. Su SHA-256 es
`d19329ee6a9d8a5d311409f3b51fdccae8fbeafd78fb89469e99a1c1afa074bf`
y declara:

```text
status=PASS
files_compared=148
mismatches=[]
core_a_sha256=52f0ec6ed0c1724fde2e123cc4c6878610f5591187f23f88e9e33ae0f042a303
core_b_sha256=52f0ec6ed0c1724fde2e123cc4c6878610f5591187f23f88e9e33ae0f042a303
```

El terminal status también es byte-exacto, con SHA-256
`e643bd0bfa4752af134ed82f41c3a541d474937a5772e3adeca27a6d1a999e7a`.
Sus hashes de core y replay apuntan a los bytes correctos y registra:

```text
artifact_status=PASS
wall_seconds_before_final=189.1472895629704
max_ru_maxrss_bytes_before_final=391495680
limits wall<600; address_space<2147483648; rss<2147483648
```

El runtime raíz y sus dos copias son byte-exactos, tienen SHA-256
`20b41f0716b16bc33a64f041006b92bddb9a383f7e6c63cdfcd18de02e5adbf4`
y registran:

```text
artifact_status=PASS
error=null
wall_seconds=220.4572234712541 < 600
max_ru_maxrss_bytes=391495680 < 2147483648
tests returncode=0
prepare/build/evaluate/check_pre/check_final returncode=0 para run_a y run_b
```

La recomputación read-only con las funciones vigentes del checker validó, en
ambos runs, `validate_core_manifest`, `validate_replay` y
`validate_terminal_status` (`check_proportional_mapping_feasibility.py:1452-1510`).
No escribió nuevas adjudicaciones ni tocó los artefactos.

## Adjudicación canónica

La recomputación independiente de `compute` coincidió estructural y
byte-semánticamente con ambos pre-adjudications. Su estado técnico es:

```json
{"source_status":"PASS","artifact_status":"PASS",
 "checker_status":"PASS","replay_status":"NOT_RUN"}
```

Como exige la fase pre, `mapping_decision` es `null`. Los cuatro predicados M
fallidos y sus reasons exactos son:

| Predicado | Status | Reason codes |
|---|---|---|
| M1_QUERY_UNIT | FAIL | `NO_COMMON_UNIT_NAMESPACE`, `UNIT_BIJECTION_INCOMPLETE` |
| M2_OBSERVATION_PARITY | FAIL | `OBSERVATION_SOURCE_MISMATCH` |
| M3_TARGET_CONSERVATION | FAIL | `TARGET_SCHEMA_MISMATCH`, `TARGET_MAP_PARTIAL` |
| M4_DECISION_STACK_PARITY | FAIL | `SCORE_SEMANTICS_MISMATCH`, `EXECUTOR_CLASS_MISMATCH`, `READER_CLASS_MISMATCH` |

M5 y los doce predicados nativos R1–R6/S1–S6 tienen status `PASS` y lista de
reasons vacía: **13 PASS / 4 FAIL** en total.

Después de validar replay y terminal, los cuatro estados técnicos son `PASS`.
La álgebra vigente (`check_proportional_mapping_feasibility.py:1428-1439`)
observa que la hoja común M no es factible mientras ambas hojas nativas sí lo
son, y deriva exactamente `BIFURCATE_NATIVE_CONTRASTS`. Los dos
`adjudication.json` tienen SHA-256
`28d9a1334a345d9310fa155f3d3c61a1505ee219df8317044e1425a9bfffb4fd`,
closure checks core/replay/terminal `true/true/true` y predicates idénticos al
pre. El informe humano reproduce esos cuatro fallos y termina explícitamente:
“Esta hoja decide compatibilidad de contratos; no declara GO/NO-GO.”

## Mutaciones y contrafactuales

Ambos `mutation_results.json` son byte-exactos, con SHA-256
`1afeaf0761392352c06358bd2793c72ce949d6c341fe2de438e4ac9a8a0b759f`.
La validación contra el catálogo cerrado vigente obtuvo:

```text
schema_version=proportional-mapping-mutation-execution-v4
status=PASS
predicate_ids=17
reason rows=67
pares únicos (id,reason)=67
digests de evidencia válidos=67/67
mapping_decision null=67/67
mutated_field no vacío=67/67
ramas R/S por derive_native_facts+native_reason_codes=44
```

Las seis corrupciones integrales `hash`, `keyset`, `shape`, `join`,
`predicate` y `decision` tienen valor `REJECTED`. El contrafactual adicional
de renombrado sintético M1 también está rechazado con
`SYNTHETIC_ID_EQUIVALENCE`.

## Frontera de la conclusión

Cada JSON científico, terminal y de paquete conserva:

```json
{"gpu_used_or_queried":false,"architecture_promoted":false,
 "scientific_decision":null,"decision_authority":"user"}
```

Por ello, `BIFURCATE_NATIVE_CONTRASTS` congela el sucesor ejecutable técnico
previsto por el gate: dos contrastes nativos coordinados. No demuestra que esa
arquitectura deba promoverse, no decide su validez científica general y no
autoriza ni representa un `GO/NO-GO`, que sigue perteneciendo al usuario.

## Resultado consolidado

| Condición | Resultado |
|---|---|
| 313 archivos del target, todos bajo la raíz canónica | PASS |
| manifests/pathsets/tamaños/hashes/NPY metadata | PASS |
| allowlists públicas 64 y privadas 74 por run | PASS |
| JSON canónico y paths científicos relativos | PASS |
| 30 recibos de sources + 6 execution sources | PASS |
| core 148/148 y científico 153/153 byte-exactos | PASS |
| replay 148/148, terminal y closure | PASS |
| runtime `<600 s` y `<2 GiB` | PASS |
| pre decisión nula y álgebra final exacta | PASS |
| M1–M4 FAIL exactos; M5+R/S PASS | PASS |
| mutation v4 67/67; seis corrupciones rechazadas | PASS |
| GPU/CUDA, promoción arquitectónica o decisión científica | no |

```json
{
  "schema_version": "proportional-mapping-feasibility-canonical-artifact-audit-v1",
  "audit_id": "R544",
  "runtime_configuration": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
  "target": {
    "commit": "d763626d017f0a56a0276bee675c3779ac0f553a",
    "parent": "32ddf37731be4dc725ef31b1f27e6dab92bec645",
    "tree": "59e60066030e6eb783fa54a2f7ef2543f28fdf81",
    "path_count": 313,
    "pathset_sha256": "65493e0f5e646bfdf9e0af7f19378f73f6b1049d1248cb0dd302746a2d514b06",
    "total_bytes": 26186051
  },
  "technical_verdict": "PASS",
  "findings": {"high": 0, "medium": 0, "low": 0},
  "artifact_validation": {
    "total_files": 313,
    "canonical_json_files": 43,
    "public_payload_files_per_run": 64,
    "private_payload_files_per_run": 74,
    "core_files_per_run": 148,
    "scientific_files_per_run": 153,
    "paired_core_mismatches": 0,
    "paired_scientific_mismatches": 0
  },
  "replay": {
    "status": "PASS",
    "files_compared": 148,
    "mismatches": 0,
    "sha256": "d19329ee6a9d8a5d311409f3b51fdccae8fbeafd78fb89469e99a1c1afa074bf"
  },
  "runtime": {
    "artifact_status": "PASS",
    "wall_seconds": 220.4572234712541,
    "max_ru_maxrss_bytes": 391495680,
    "wall_limit_exclusive": 600,
    "rss_limit_exclusive": 2147483648
  },
  "adjudication": {
    "pre_mapping_decision": null,
    "final_mapping_decision": "BIFURCATE_NATIVE_CONTRASTS",
    "predicate_pass": 13,
    "predicate_fail": 4,
    "closure_checks": {"core_manifest": true, "replay": true, "terminal_status": true}
  },
  "mutations": {
    "schema_version": "proportional-mapping-mutation-execution-v4",
    "predicate_ids": 17,
    "unique_reason_pairs": 67,
    "material_rs_rows": 44,
    "candidate_corruptions_rejected": 6
  },
  "gpu_used_or_queried": false,
  "monitor_or_lockbox_opened": false,
  "artifacts_regenerated_or_modified": false,
  "architecture_promoted": false,
  "scientific_decision": null,
  "decision_authority": "user"
}
```
