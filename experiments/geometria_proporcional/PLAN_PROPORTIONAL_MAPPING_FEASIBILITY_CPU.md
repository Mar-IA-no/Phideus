# Plan CPU — `MAPPING-FEASIBILITY` para el relevo proporcional

> **Estado:** `PLAN-REVISION-R536 / PRE-IMPLEMENTATION / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Autoridad de promoción:** usuario

## 1. Pregunta finita y alcance

Este gate no busca una geometría nueva, no reentrena redes y no elige una
arquitectura. Decide si tres objetos ya existentes pueden entrar en un mismo
contraste causal sin cambiar de pregunta durante el adapter:

1. la referencia EIV + conformal de Ola 49;
2. el núcleo relacional `GENERIC/TYPED` con executors WLS/IRLS;
3. el posterior `MARGINAL/JOINT` con decisión `HARD/CONTEXTUAL`.

La única query candidata al factorial común es:

> Dada una observación pública de relaciones proporcionales ruidosas, ¿qué
> conjunto de hipótesis relacionales permanece compatible y qué acción externa
> minimiza el regret bajo una utilidad no observada por la representación?

El gate termina en una de tres hojas semánticas:

- `COMMON_FACTORIAL_FEASIBLE`: los tres objetos comparten query, unidad,
  observación, target, interfaz de scores y autoridad sin pérdida;
- `BIFURCATE_NATIVE_CONTRASTS`: el mapeo común falla, pero sobreviven un
  contraste relacional y otro set-valued completos;
- `NO_EXECUTABLE_SUCCESSOR`: falla también al menos uno de los dos contratos
  nativos.

Una falla técnica no alcanza ninguna hoja: deja `mapping_decision:null`.
Ninguna hoja promueve arquitectura ni constituye `GO/NO-GO`.

## 2. Matriz congelada de fuentes, roles y acceso

Todos los paths son relativos al repositorio. Las fases son:

- `P`: preparador; puede abrir fuentes mixtas y debe separarlas por allowlist;
- `B`: builder; sólo puede abrir la extracción pública ya preparada;
- `E`: evaluator; abre verdad de desarrollo después de congelar el candidato;
- `C`: checker externo; abre fuentes ligadas y recomputa todo de manera
  independiente.

`C` no convierte una fuente privada en input del builder. Ningún proceso abre
lockbox, `sealed_monitor_bundle.npz` ni monitor histórico para construir o
adjudicar el mapeo.

| ID | Fuente | Rol y campos autorizados | Fases | SHA-256 |
|---|---|---|---|---|
| `SYNTHESIS` | `Biblioteca/Geometria_Proporcional_Ground_Truth/PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md` | pregunta y portfolio terminal; lectura integral | P,C | `888477866168d1f2e5dbb756835ef0ed6521a636ab8cd752aaaa6181dc490db2` |
| `W49_SCHEMA` | `src/geometria_proporcional/wave49_schema.py` | familias, schema visible y claves prohibidas | P,C | `135c3d4994019cfccf3bc2ec3b7c8c6c9b4772a898eb3a900f58c0135e183f36` |
| `W49_MANIFEST` | `data/geometria_proporcional/wave49/manifest.json` | conteos, archivos y autoridad protocolar | P,C | `46f8d31d7b97d71cea73742251bb4d7e4d3bd3ea09696e8cce54c418f4a8e19a` |
| `W49_CONFIG` | `data/geometria_proporcional/wave49/protocol_config.json` | catálogo y contrato de calibración | P,C | `c45a7fb245950521ceac4c6de75b51e746152f506522c52697d05bdc30673468` |
| `W49_PUBLIC_TRAIN` | `data/geometria_proporcional/wave49/visible/train.jsonl` | `fixture_id,x,y,n,covariance,coordinate_semantics,domain,schema_version,split` | P,C | `2670bee5cb0312d78caf65261f2bf32fc5869038d5a472e056161943d0176010` |
| `W49_PUBLIC_VAL` | `data/geometria_proporcional/wave49/visible/val.jsonl` | mismos campos visibles; desarrollo | P,C | `e8b748e3b35ddda466d7a3e9a257dddf84260625e25eec951b5824a3a51a3860` |
| `W49_PRED_TRAIN` | `data/geometria_proporcional/wave49/predictions/train.jsonl` | scores, sets estructurales, cutoff, status y selector por `fixture_id` | P,C | `0a1c8b58c8cf6271c8702d66144e30cd63251e4464afa7ff4fb6b2d648088f2e` |
| `W49_PRED_VAL` | `data/geometria_proporcional/wave49/predictions/val.jsonl` | misma salida de predictor; desarrollo | P,C | `db815ee2e6d0ea0cdd8a2f25d77a8fd42dd52a876d8caad12b45a106cfaecebe` |
| `W50_TARGET_ATTEST` | `data/geometria_proporcional/wave50_prospective_v1/authorized_labels/target_attestation.json` | compromiso de autoridad del oracle de desarrollo | E,C | `197dad097b05678eefd95164d7bf8c689e208ee071fca63757cb336726f9c282` |
| `W50_TARGET_TRAIN` | `data/geometria_proporcional/wave50_prospective_v1/authorized_labels/train.jsonl` | `fixture_id,pair_token,oracle_compatible_set,oracle_status` y diagnósticos oracle; nunca input | E,C | `9d643bcaea0ba7e0d547aeea295a5fc2d12c188593a8a067392c709f40b3439d` |
| `W50_TARGET_VAL` | `data/geometria_proporcional/wave50_prospective_v1/authorized_labels/val.jsonl` | mismos targets y join de desarrollo; nunca input | E,C | `ea77e914812d4738b2904a7f7b28a71daa097aa230273a15b6f96844c8567b3d` |
| `W52_POLICY` | `data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json` | cuatro niveles y 24 permutaciones de utilidad externa | P,C | `f8a608d396ad48ba3b0336df2dc1955940515be0f51fa08328cd5ddb1e9a21e1` |
| `W52_PRIMITIVE` | `src/geometria_proporcional/wave52_policy.py` | acción autorizada y regret; referencia de contrato | P,C | `14c6b33d972ee1254386b21e08f69554b9880a4cac98c2b85b00b8d3b7015b5a` |
| `W53_PRIMITIVE` | `src/geometria_proporcional/wave53_uncertainty.py` | producto Bernoulli condicionado a set no vacío | P,C | `df08244863c6526d21f9fd5d278a66821814736d103d79fb46275231602be4e2` |
| `W53_PLATT` | `data/geometria_proporcional/wave53_uncertainty_policy_v1/platt_calibrator.json` | calibrador marginal congelado | P,C | `23e2c255bd04ff59ba07390322ff516b94a976bfe345def7e5a5cdfeebbae875` |
| `W54_INPUT_MANIFEST` | `data/geometria_proporcional/wave54_joint_set_inputs_v1/input_bundle_manifest.json` | lineage, conteos y hashes del bundle | P,C | `5acffb885d48e1e78f72ef4a6e5eddf09318ae49248000cafe3d040413cb4208` |
| `W54_FIT_SELECT` | `data/geometria_proporcional/wave54_joint_set_inputs_v1/fit_select_bundle.npz` | mixto: input=`ensemble_logits,per_seed_logits`; fase/join=`split_role,pair_token,cluster_id`; privado=`target,design_stratum,cardinality` | P,E,C | `2c30a844412d0af15ac7df398977e5c626d434e85622272883928b176fbcaae3` |
| `W54_SELECTION` | `data/geometria_proporcional/wave54_joint_set_v1/selection_freeze.json` | `best_independent`, theta `joint_full`, regularización y cronología previas a monitor | P,C | `621c2c7883eb42f530f21eb6e96e4136d8a49d49b49c9f1ddb41de45ed955972` |
| `W54_PRIMITIVE` | `src/geometria_proporcional/wave54_joint_set.py` | masa joint, riesgo esperado y schema de quince sets | P,C | `75b3bb5a65f145a9b7a4787580576723366ac3ca0d49cbc007394220f6d98901` |
| `W56_PRIMITIVE` | `src/geometria_proporcional/wave56_contextual_gate.py` | schema de 17 features, gain y pesos de disagreement | P,C | `474d3274b7d3c8ba3cba1d4f10b8b238119d58957520febbf24bf11de9ef665d` |
| `W57_PRIMITIVE` | `src/geometria_proporcional/wave57_tail_guard.py` | logistic de daño y separación proposer/guard | P,C | `059572890c71de4723b35d44d9c7c91d1698e5d515ba5d2551e971180b479a99` |
| `GRAPH_SCHEMA` | `src/geometria_proporcional/proportional_graph_contract.py` | allowlist pública, autoridad privada, gauge y solvers | P,C | `a2d8c3ebf327aded797b89eddcdcd2cc315fc0a25bf9b49a197878a90c3f5a53` |
| `GRAPH_DATASET` | `data/geometria_proporcional/proportional_graph_neural_smoke_v1/dataset_manifest.json` | universo, IDs, splits y digest público por vista | P,E,C | `f36eaedabad97319554f5ceeb4ab54b498717c21e696f56c75c2f9276cda75e1` |
| `GRAPH_CONFIG` | `data/geometria_proporcional/proportional_graph_neural_smoke_v1/resolved_config.json` | brazos, seeds, WLS/IRLS y métricas | P,C | `ada41589da53b21fe0e3e1703dfe422a248f92d0bf481853ed92761b5de6e62a` |
| `GRAPH_MANIFEST` | `data/geometria_proporcional/proportional_graph_neural_smoke_v1/manifest.json` | inventario y hashes del artefacto | P,C | `7e982a92bd366a4c22fe95c0cc9c8fd5f7a1773bd78c4e8e0781b5c2259624e1` |
| `GRAPH_GENERIC_S1` | `data/geometria_proporcional/proportional_graph_neural_smoke_v1/raw_eval/raw_generic\|seed=104729.npz` | mixto; salida=`corrected_log_ratio,reliability`; observación/verdad según separación §3 | P,E,C | `34f075f27752aa1d1aa40eaf01556a66d0d83602390d374d7880abd3eeaaa598` |
| `GRAPH_GENERIC_S2` | `data/geometria_proporcional/proportional_graph_neural_smoke_v1/raw_eval/raw_generic\|seed=130363.npz` | mismo contrato | P,E,C | `a28a4e9ad0a53897f56a1c9a72a6fdcc1a369fccd519ef9a3749c344c090929b` |
| `GRAPH_TYPED_S1` | `data/geometria_proporcional/proportional_graph_neural_smoke_v1/raw_eval/raw_typed\|seed=104729.npz` | mismo contrato | P,E,C | `7e66148516a1a921a6832178770a392ff5fcf8d74839c6d0e0f0901353ba7a26` |
| `GRAPH_TYPED_S2` | `data/geometria_proporcional/proportional_graph_neural_smoke_v1/raw_eval/raw_typed\|seed=130363.npz` | mismo contrato | P,E,C | `8fe3adc035b9e268ed75535c23207fec142f6405d72a707fcb056f85cd59b9f3` |

No se permite añadir fuentes durante runtime. Una carencia produce
`source_status:FAIL`, no una búsqueda heurística.

## 3. Barrera física de información

La ejecución usa cuatro programas/procesos y directorios no solapados:

### A. Preparer

Verifica hashes y crea dos árboles inmutables:

- `prepared/public/`, único input del builder;
- `prepared/private_dev/`, accesible sólo por evaluator y checker.

Para W49, copia sólo fixtures visibles y predicciones. La verdad W50 queda en
`private_dev`. Para W54, extrae únicamente logits como features; reemplaza
`pair_token/cluster_id` por pseudónimos, conserva `split_role` sólo como control
de fase y mueve `target`, `design_stratum` y `cardinality` a `private_dev`.
Cada pseudónimo es `SHA256(namespace || 0x00 || id)`. Para grafo, extrae como
observación los ocho
`PUBLIC_ARRAY_FIELDS`; offsets son metadata de contenedor, no features. Crea el
mismo `unit_key` pseudónimo desde `master_id/view_id`, conserva en público las
salidas `corrected_log_ratio/reliability` identificadas por brazo y seed, y
mueve a privado `x_true`, `clean_log_ratio`, `causal_corruption_mask`,
`mechanism`, split e IDs originales. Ninguna clave de
`FORBIDDEN_PUBLIC_FIELDS` aparece en la extracción pública.

Cada árbol recibe un manifest canónico con keysets, dtypes, shapes y hashes. El
preparer termina antes de iniciar el builder.

### B. Builder

Su proceso recibe únicamente el path de `prepared/public/`. No recibe repo
root, paths fuente ni `prepared/private_dev/`. Emite `mapping_candidate.json`
y adapters declarativos; no emite una de las tres decisiones.

### C. Evaluator

Después de fijar SHA-256 y permisos read-only del candidato, abre
`prepared/private_dev/`, ejecuta joins/evaluaciones y emite
`evaluation_evidence.json`. Nunca modifica el candidato.

### D. Checker externo

Es un ejecutable separado que no importa builder, evaluator ni sus helpers.
Sólo puede compartir tipos serializados y la especificación congelada. Vuelve a
calcular hashes, keysets, shapes, pseudónimos, joins, bijecciones, predicados y
la tabla de decisión directamente desde fuentes ligadas y outputs congelados.
El único `adjudication.json` autoritativo es suyo.

## 4. Álgebra total de estados

Cada predicado se serializa como:

```json
{"id":"M1","status":"PASS|FAIL","reason_codes":[],"evidence":[]}
```

No existe `PARTIAL`. `evidence` contiene `source_id`, locator, observado y
digest. Los reason codes son cerrados y pertenecen al predicado que los define.

Los estados técnicos son independientes:

```text
source_status  ∈ {PASS, FAIL}
artifact_status ∈ {PASS, FAIL, BUDGET_EXCEEDED}
checker_status ∈ {PASS, FAIL}
replay_status  ∈ {PASS, FAIL, NOT_RUN}
mapping_decision ∈ {COMMON_FACTORIAL_FEASIBLE,
                    BIFURCATE_NATIVE_CONTRASTS,
                    NO_EXECUTABLE_SUCCESSOR, null}
```

Sólo si los cuatro estados técnicos son `PASS` se aplica, en este orden, la
tabla total:

```text
all(M1..M5)                                -> COMMON_FACTORIAL_FEASIBLE
not all(M1..M5) and all(R1..R6) and all(S1..S6)
                                             -> BIFURCATE_NATIVE_CONTRASTS
not all(M1..M5) and (not all(R1..R6) or not all(S1..S6))
                                             -> NO_EXECUTABLE_SUCCESSOR
```

En cualquier otro caso `mapping_decision:null`. La precedencia del factorial
común es intencional: si `all(M)`, los predicados nativos se reportan pero no
cambian la hoja.

## 5. Predicados del factorial común

| ID | `PASS` si y sólo si | Reason codes de `FAIL` |
|---|---|---|
| `M1_QUERY_UNIT` | los tres brazos copian literalmente la query de §1; existe un namespace común y una bijección total 1:1:1 entre todas las unidades evaluables, acreditada por autoridad previa y no por renaming | `QUERY_MISMATCH`, `NO_COMMON_UNIT_NAMESPACE`, `UNIT_BIJECTION_INCOMPLETE`, `SYNTHETIC_ID_EQUIVALENCE` |
| `M2_OBSERVATION_PARITY` | cada unidad común parte de la misma observación pública o de proyecciones deterministas con inversa exacta; los tres brazos reciben igual información y ningún campo privado/prohibido | `OBSERVATION_SOURCE_MISMATCH`, `PROJECTION_NOT_INVERTIBLE`, `INFORMATION_ASYMMETRY`, `PRIVATE_FIELD_EXPOSED` |
| `M3_TARGET_CONSERVATION` | existe un único target tipado y mapas totales de ida/vuelta que reconstruyen exactamente los targets EIV, set-valued y grafo módulo gauge, sin threshold aprendido ni oracle del monitor | `TARGET_SCHEMA_MISMATCH`, `TARGET_MAP_PARTIAL`, `TARGET_ROUNDTRIP_LOSS`, `LEARNED_TARGET_BRIDGE`, `MONITOR_TARGET_USED` |
| `M4_DECISION_STACK_PARITY` | los tres producen el mismo tensor de scores con semántica/shape común y usan el mismo executor, checker, reader y abstención; el efecto de representación queda medido antes del reader | `SCORE_SEMANTICS_MISMATCH`, `EXECUTOR_CLASS_MISMATCH`, `READER_CLASS_MISMATCH`, `CALIBRATION_ENTANGLED`, `EXTERNAL_OPERATION_ASYMMETRY` |
| `M5_AUTHORITY_PHASES` | query/schema/target provienen del freeze; utilidad es `SYNTHETIC_EXTERNAL`; builder sólo vio público; desarrollo precede freeze y ningún monitor/lockbox fue abierto; checker es independiente | `UNBOUND_AUTHORITY`, `UTILITY_LEAKAGE`, `PHASE_VIOLATION`, `MONITOR_OR_LOCKBOX_OPENED`, `CHECKER_NOT_INDEPENDENT` |

## 6. Predicados del contraste relacional nativo

El contraste propuesto es `GENERIC/TYPED × WLS/IRLS`, unidad primaria
`master_id` y vistas pareadas como observaciones dependientes. Targets:
`clean_log_ratio` y potencial `x_true` sólo módulo gauge. Primarias: relation
RMSE, quotient RMSE, convergencia/fallo; IID/grouped son vistas declaradas.

| ID | `PASS` si y sólo si | Reason codes de `FAIL` |
|---|---|---|
| `R1_SOURCE_COMPLETE` | existen y coinciden hashes de manifest/config y los cuatro estados `GENERIC/TYPED × seed`; cada NPZ tiene keyset, dtype y shape requeridos | `GRAPH_SOURCE_MISSING`, `GRAPH_HASH_MISMATCH`, `GRAPH_SCHEMA_INVALID` |
| `R2_PUBLIC_PARITY` | para cada seed, `GENERIC` y `TYPED` poseen exactamente los mismos `unit_key`, observación y offsets; ninguna verdad está en público | `GRAPH_UNIT_MISMATCH`, `GRAPH_INPUT_MISMATCH`, `GRAPH_PRIVATE_LEAKAGE` |
| `R3_REPRESENTATION_OUTPUT` | ambos brazos producen `corrected_log_ratio` y `reliability` finitos, con una salida por arista pública y sin alterar topología | `REPRESENTATION_OUTPUT_MISSING`, `REPRESENTATION_OUTPUT_NONFINITE`, `TOPOLOGY_CHANGED` |
| `R4_EXECUTOR_FACTORIAL` | WLS e IRLS aceptan exactamente la misma IR por cada brazo, comparten gauge, parámetros y budget, y las cuatro celdas se reconstruyen sin usar truth | `EXECUTOR_INPUT_MISMATCH`, `EXECUTOR_RECIPE_MISMATCH`, `EXECUTOR_CELL_MISSING`, `TRUTH_USED_BY_EXECUTOR` |
| `R5_TARGET_AUTHORITY` | evaluator alinea 1:1 cada salida con `clean_log_ratio/x_true`; quotient se compara tras gauge canónico y nunca se usa mecanismo de corrupción como input | `GRAPH_TARGET_JOIN_INVALID`, `GAUGE_NOT_CANONICAL`, `MECHANISM_LEAKAGE` |
| `R6_ESTIMAND_CONTROLS` | unidad, pares, loss, métricas y convergencia se aplican idénticamente a las cuatro celdas; el control target-shuffled usa una única permutación de masters dentro de `(split,mechanism,n_nodes)` con `PCG64(53601)` para todas las celdas, y el matched report usa la intersección congelada de unidades válidas; hay soporte positivo en cada seed y brazo | `RELATIONAL_ESTIMAND_MISMATCH`, `RELATIONAL_CONTROL_MISMATCH`, `RELATIONAL_SUPPORT_EMPTY` |

EIV queda como referencia externa si falla el mapeo común; no se renombra como
una quinta celda relacional.

## 7. Predicados y cuatro celdas del contraste set-valued

La representación cambia sólo la masa sobre los mismos quince sets no vacíos:

- `MARGINAL`: `independent_platt`, probabilidades calibradas por W53 y producto
  Bernoulli condicionado a set no vacío;
- `JOINT`: `joint_full`, con theta y regularización primarias de
  `W54_SELECTION`.

Ambas reciben los mismos cuatro logits ensemble. La utilidad externa son las
24 permutaciones de los cuatro niveles de `W52_POLICY`; penalty incompatible
`1.25`. Nada de ello entra en el posterior.

### 7.1 Reader `HARD_MAP_SET`

Para cada posterior, toma su set MAP entre los quince; los empates usan el
menor índice binario. Para cada utilidad elige el miembro del set MAP de mayor
utilidad; el empate usa el menor índice de familia. Por eso
`MARGINAL×HARD` y `JOINT×HARD` son funciones de masas distintas. Si sus
acciones empíricas coinciden, se registra `observed_cell_duplication:true`, sin
ocultar ni redefinir la celda.

### 7.2 Reader `CONTEXTUAL_PROPOSER_GUARD`

Para cada posterior se ejecuta el mismo algoritmo:

1. candidato: acción de mínimo riesgo posterior, empate por menor familia;
2. baseline: `HARD_MAP_SET` del mismo posterior;
3. features: las 17 de `W56_PRIMITIVE`, con la masa y riesgos de ese posterior;
4. proposer: ridge float64 con estandarización W56, columna de intercept no
   penalizada y `alpha=1`; resuelve
   `(X'WX + alpha*diag(0,1,...,1)) beta = X'Wy` mediante `numpy.linalg.solve`;
   target `regret(hard)-regret(candidate)`;
5. guard de daño: logistic W57/scikit-learn `1.8.0`, `C=1`, `lbfgs`, penalty
   L2 (`l1_ratio=0`), `dual=false`, `class_weight=null`, intercept,
   `max_iter=2000`, `tol=1e-10`, `warm_start=false`; target
   `gain < -1e-12`;
6. guard de incompatibilidad: la misma logistic y fitting, target
   `not target[candidate]`;
7. fitting: sólo filas `calibration_fit`, ponderadas por disagreement W56; cada
   posterior ajusta sus estados por separado con receta idéntica;
8. selección: sólo `decision_select`; grid proposer
   `[0.5,0.6,0.7,0.8,0.9,0.95,0.975]`, y cada guard
   `[0.1,0.2,0.3,0.4,0.5,0.6,0.8]`, cuantiles calculados sobre filas activas;
   usa `numpy.quantile(method="linear")`, proposer autoriza con `score > t`,
   guards con `probability < t`, y se incluye `HARD_ONLY`;
9. clave de selección ascendente:
   `(mean_regret, incompatibility_rate, harm_rate, -authorized_rows,
   proposer_quantile, harm_quantile, incompatibility_quantile)`; no se mira
   monitor; `HARD_ONLY` usa el sentinel numérico `2.0` en los tres cuantiles
   sólo para desempate;
10. freeze: schema, coeficientes, scalers, thresholds, keysets y hashes quedan
    fijos antes de una evaluación futura.

No se heredan silenciosamente estados W56–W60: esas fuentes fijan el tipo de
operación, pero este contrato vuelve a ajustar cada reader sobre su posterior.

| ID | `PASS` si y sólo si | Reason codes de `FAIL` |
|---|---|---|
| `S1_SOURCE_COMPLETE` | manifest, bundle, calibrador, selección y primitives coinciden por hash; bundle contiene 384 tokens únicos y shapes exactas: logits `[384,4]`, per-seed `[3,384,4]`, target `[384,4]`, roles 192/192 | `SET_SOURCE_MISSING`, `SET_HASH_MISMATCH`, `SET_SCHEMA_INVALID`, `SET_ROLE_COUNTS_INVALID` |
| `S2_POSTERIOR_PARITY` | ambas masas son finitas `[384,15]`, no negativas, suman uno y corresponden al mismo orden de sets/tokens/logits; la utilidad no participa | `POSTERIOR_CELL_MISSING`, `POSTERIOR_MASS_INVALID`, `POSTERIOR_ALIGNMENT_MISMATCH`, `UTILITY_IN_POSTERIOR` |
| `S3_FOUR_CELLS_EXECUTABLE` | las cuatro funciones producen acciones `[384,24]`; hard depende de la masa de su posterior, contextual usa la misma receta en ambos, y toda duplicación observada se declara | `SET_DECISION_CELL_MISSING`, `HARD_READER_NOT_POSTERIOR_BOUND`, `CONTEXTUAL_RECIPE_MISMATCH`, `CELL_DUPLICATION_UNDECLARED` |
| `S4_FIT_SUPPORT_FREEZE` | cada posterior tiene disagreement activo, targets finitos y ambas clases para cada guard en `calibration_fit`; todas las candidatas son evaluables en `decision_select`; freeze antecede monitor | `PROPOSER_SUPPORT_EMPTY`, `HARM_CLASS_MISSING`, `INCOMPATIBILITY_CLASS_MISSING`, `SELECTION_SUPPORT_EMPTY`, `SET_PHASE_VIOLATION` |
| `S5_TARGET_UTILITY_AUTHORITY` | evaluator alinea 1:1 target no vacío por token; targets sólo entran en fitting/evaluación; utilidad W52 es externa, común y de rango no nulo | `SET_TARGET_JOIN_INVALID`, `EMPTY_TARGET_SET`, `TARGET_LEAKAGE`, `UTILITY_CONTRACT_MISMATCH` |
| `S6_ESTIMAND_CONTROLS` | posterior se mide con set-NLL/Brier antes de reader; acción con compatibilidad/regret/cola; tupla, input, agregación, target y loss coinciden entre celdas; target-shuffled usa una única permutación dentro de `(split_role,design_stratum,cardinality)` con `PCG64(53602)` para ambos posteriors y matched usa la intersección congelada de filas autorizadas, sin cambiar ningún otro campo | `SET_ESTIMAND_MISMATCH`, `POSTERIOR_READER_ENTANGLED`, `SET_CONTROL_MISMATCH`, `SET_SUPPORT_EMPTY` |

## 8. Fixtures y mutaciones obligatorias

El test suite incluye tres fixtures positivos sintéticos, independientes de los
datos observados:

1. `LEAF_COMMON`: `M1..M5=PASS` → `COMMON_FACTORIAL_FEASIBLE`;
2. `LEAF_BIFURCATE`: un `M` falla y `all(R),all(S)` →
   `BIFURCATE_NATIVE_CONTRASTS`;
3. `LEAF_NONE`: un `M`, un `R` y un `S` fallan →
   `NO_EXECUTABLE_SUCCESSOR`.

Para cada `M1..M5`, `R1..R6` y `S1..S6` existe una mutación aislada que debe
producir `FAIL` y el reason code exacto esperado. Además, el checker rechaza:

- renaming de `fixture_id/pair_token/master_id` como correspondencia;
- truth, oracle, mecanismo o split dentro de público;
- threshold aprendido que convierta relación continua en familia;
- paths sintéticos añadidos sólo a `TYPED`;
- WLS/IRLS renombrados como readers hard/contextual;
- calibración EIV incorporada al encoder;
- utilidad dentro del posterior;
- decisión escrita por el builder;
- `GO`, `NO-GO` o promoción en cualquier artefacto.

Pruebas contrafactuales regeneran el preparador tras cambiar o eliminar, uno por
uno, cada campo privado de W49/W50, W54 y grafo mientras mantienen idéntico el
input público: `prepared/public/`, candidato y adapters deben ser byte-exactos.
Luego se corrompe cada clase del candidato —hash, keyset, shape, join,
predicado y decisión— y el checker debe fallar. Que un output omita truth no
basta: la invariancia contrafactual es obligatoria.

## 9. Artefactos, manifest y replay

Raíz: `data/geometria_proporcional/proportional_mapping_feasibility_v1/`.
Cada run vive en `run_a/` o `run_b/` y produce:

- `prepared/public/` y `prepared/private_dev/` con manifests;
- `source_inventory.json`, `native_contracts.json`;
- `mapping_candidate.json`, `evaluation_evidence.json`;
- `mapping_matrix.json`, `mutation_results.json`;
- `pre_adjudication.json`, con predicados pero `replay_status:NOT_RUN` y
  `mapping_decision:null`;
- `core_manifest.json`;
- `runtime.json`.

Todos salvo `runtime.json` pertenecen a la clase científica determinista y se
comparan byte a byte entre runs independientes. JSON usa UTF-8, keys ordenadas,
separadores canónicos, `allow_nan=false`, newline final, paths relativos al
repo y cero timestamps/UUID/paths absolutos. NPZ, si aparece, se escribe con
orden fijo y también debe ser byte-exacto; de lo contrario se usa NPY por array
con dtype/shape/content hash.

Para evitar circularidad entre replay y decisión, el cierre tiene dos fases:

1. `core_manifest.json` enumera y hashea todos los artefactos científicos del
   run excepto a sí mismo, `runtime.json` y la adjudicación final. Los dos core
   manifests y cada archivo enumerado deben ser byte-exactos.
2. El controlador emite un único `replay_evidence.json` canónico con los hashes
   de ambos cores y la comparación por clase. Con esa evidencia, dos procesos
   nuevos del checker producen sendos `adjudication.json` y
   `REPORT_MAPPING_FEASIBILITY.md`; ambos pares también deben ser byte-exactos.
   Sólo entonces `replay_status:PASS` puede habilitar una decisión no nula.

`scientific_manifest.json` enumera core, `replay_evidence.json`, adjudicación e
informe final, pero no se hashea a sí mismo; también debe coincidir entre las
dos copias normalizadas. `runtime.json` queda fuera de todos los manifests y se
compara sólo por schema, estados y límites, nunca por valores de tiempo/RSS.

Builder, evaluator, checker, tests y replay corren en procesos nuevos y los dos
runs en directorios separados. Todos los artefactos declaran:

```json
{"gpu_used_or_queried":false,"architecture_promoted":false,
 "scientific_decision":null,"decision_authority":"user"}
```

## 10. Presupuesto terminal y detención

- CPU exclusivamente; ningún import, llamada o consulta CUDA;
- watchdog global estricto: `<600` segundos para preparación, dos runs,
  mutaciones, tests y replay;
- límite por proceso `RLIMIT_AS=2 GiB`; se registra `ru_maxrss` y el máximo debe
  ser `<2 GiB`;
- sin training/forward neuronal, bootstrap nuevo ni solves masivos.

Timeout, `MemoryError`, señal por límite, `wall_seconds>=600` o
`max_ru_maxrss_bytes>=2147483648` produce
`artifact_status:BUDGET_EXCEEDED`, `replay_status:NOT_RUN|FAIL` según fase y
`mapping_decision:null`. Otra falla de schema/output produce
`artifact_status:FAIL`. No se publica hoja semántica con ejecución fuera de
presupuesto.

Si completar el contraste posterior exigiera re-forward, training o una
operación de muchas horas que GPU resolviera materialmente mejor, se conserva
la cola y se detiene antes de CUDA. Recién entonces se publica evidencia
durable y se informa por Telegram objetivo, duración y VRAM estimada.

## 11. Secuencia y condición de cierre

1. auditoría independiente de esta revisión y resolución de todo finding
   material;
2. freeze del plan, fuentes y hashes;
3. implementación separada de preparer, builder, evaluator y checker;
4. tests de las tres hojas, cada predicado, contrafactuales y corrupciones;
5. build + replay CPU bajo límites;
6. auditoría independiente de artefactos y adjudicación;
7. documentación de una sola hoja válida o de `mapping_decision:null` si falla
   la ejecución.

El goal termina al congelar un sucesor ejecutable —factorial común o dos
contrastes coordinados— o al demostrar de forma válida que hace falta rediseño.
La promoción arquitectónica y cualquier `GO/NO-GO` permanecen fuera del gate.
