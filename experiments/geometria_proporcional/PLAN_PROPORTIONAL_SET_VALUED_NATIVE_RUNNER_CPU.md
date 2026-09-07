# Plan CPU — runner pre-draw de la rama set-valued nativa

## 1. Propósito y condición de cierre

Este plan materializa el diseño congelado
`MARGINAL/JOINT × HARD/CONTEXTUAL` como un runner CPU ejecutable antes de
crear un nuevo draw. El resultado esperado no es evidencia prospectiva ni una
selección científica de arquitectura. Es un paquete de implementación que debe
demostrar, sobre fixtures sintéticas y poblaciones históricas ya abiertas, que:

1. las dos representaciones pueden ajustarse desde los mismos logits y targets;
2. cada reader queda ligado a su propio posterior;
3. los estados y thresholds se congelan antes de una aplicación target-blind;
4. el target-shuffle y los cinco controles matched conservan las unidades,
   recetas y soportes declarados;
5. una ejecución primaria y su replay producen exactamente los mismos
   artefactos analíticos;
6. un checker independiente detecta leakage, cruces de fase, cambios de receta,
   soporte variable, corrupción de estados y lenguaje de promoción indebido.

El objetivo se cierra sólo después de una auditoría independiente del plan, una
implementación completa, una suite CPU proporcionada, medición de costo y una
auditoría independiente final sin findings altos o medios abiertos.

## 2. Alcance negativo

Este hito no:

- crea ni autoriza el draw fresco previsto por el freeze;
- abre o reutiliza una población de monitor o lockbox;
- entrena sobre datos nuevos;
- usa, consulta o inicializa GPU/CUDA;
- evalúa cobertura prospectiva, generalización ni variabilidad de seeds;
- promueve una representación o reader;
- declara `GO`, `NO-GO` ni una decisión científica;
- reabre una campaña bibliográfica;
- convierte resultados históricos reusados en evidencia nueva.

Los resultados sobre datos abiertos se rotulan
`OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC`. Los mínimos prospectivos
`300/300/600/600` se registran como contrato futuro, pero no se usan para
atribuir elegibilidad al fixture histórico de `192/768/768` filas. El estado
máximo del hito es `RUNNER_PREFLIGHT_VALID`; nunca
`FRESH_DRAW_AUTHORIZED` ni una fórmula equivalente.

## 3. Base vigente y fuentes ligadas

El plan se rebasa sobre `a49fe9b503a5ff366276568c4eded3645e7c3401` y
liga por path y SHA-256:

| Rol | Path | SHA-256 |
|---|---|---|
| freeze set-valued | `experiments/geometria_proporcional/configs/proportional_set_valued_native_freeze_v1.json` | `41b204789d8ba3cbdff057cf6a10f73bb39fe9c1a7b44233f2b0b78181e47409` |
| mapping plan | `experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md` | `ef61081fcbd6917368d62da065843e3b397abec0149dfed5c0cc99e77fbdc6d2` |
| dual-freeze plan | `experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md` | `7f36d2cdfff899fed21d6994014a82f72a5c6f2e8625c44c61c049b9ecd0fb5b` |
| policy primitive | `src/geometria_proporcional/wave52_policy.py` | `14c6b33d972ee1254386b21e08f69554b9880a4cac98c2b85b00b8d3b7015b5a` |
| uncertainty primitive | `src/geometria_proporcional/wave53_uncertainty.py` | `df08244863c6526d21f9fd5d278a66821814736d103d79fb46275231602be4e2` |
| joint primitive | `src/geometria_proporcional/wave54_joint_set.py` | `75b3bb5a65f145a9b7a4787580576723366ac3ca0d49cbc007394220f6d98901` |
| contextual reference | `src/geometria_proporcional/wave56_contextual_gate.py` | `474d3274b7d3c8ba3cba1d4f10b8b238119d58957520febbf24bf11de9ef665d` |
| matched-control reference | `src/geometria_proporcional/wave59_hgb_guard_bracket.py` | `ce38c0b59863cba05a125ee26039b3a18222184bf88de42582a40bef2c205991` |
| utility catalogue | `data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json` | `f8a608d396ad48ba3b0336df2dc1955940515be0f51fa08328cd5ddb1e9a21e1` |
| W54 input manifest | `data/geometria_proporcional/wave54_joint_set_inputs_v1/input_bundle_manifest.json` | `5acffb885d48e1e78f72ef4a6e5eddf09318ae49248000cafe3d040413cb4208` |
| posterior-fit source | `data/geometria_proporcional/wave54_joint_set_inputs_v1/fit_select_bundle.npz` | `2c30a844412d0af15ac7df398977e5c626d434e85622272883928b176fbcaae3` |
| policy-fit source | `data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1/prepared/gate_fit_bundle.npz` | `792394583a1ab7510e991bf30ca40d84856ec0fcc81bc0321cfa976dcad29528` |
| selection public reference | `data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1/prepared/gate_select_inference_bundle.npz` | `cae538b535177dde2141539fd7d75454b9e3a0305896e539d79b85ffb047c3d7` |
| selection truth source | `data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1/prepared/gate_select_truth_bundle.npz` | `6e04b0865ee4bdcf41afe4775347f5384f7850b3d489dd58e09f73c3f2a3e2d4` |
| opened-data receipt | `data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1/preparation_freeze.json` | `d87bff354738ff99ddd0413ca4b1fb73b9b074c283b83f59c50252695b223a4b` |

La implementación no hace discovery de datos. Una allowlist de paths exactos
rechaza cualquier fuente cuyo path normalizado contenga semántica de monitor,
lockbox, sealed truth, oracle secret o generation secret. La referencia pública
de selección de Wave 59 sólo prueba identidad de `pair_token`; su design viejo
no se reutiliza porque corresponde a otro posterior y otro baseline.

## 4. Inventario observado y frontera de inferencia

Las tres poblaciones útiles están ya abiertas y son disjuntas por
`pair_token`:

| Rol de fixture | Fuente | Filas | Uso permitido |
|---|---|---:|---|
| `posterior_fit_opened` | filas `calibration_fit` de W54 | 192 | ajustar MARGINAL/JOINT y target-shuffle |
| `policy_fit_opened` | `gate_fit_bundle.npz` de W59 | 768 | ajustar proposer, guards y controles |
| `decision_select_opened` | `gate_select_truth_bundle.npz` de W59 | 768 | seleccionar thresholds y evaluar sólo el fixture |

No hay solapamientos entre esas poblaciones ni con las 192 filas
`decision_select` restantes de W54. El fixture de posterior tiene ambas clases
en el pool binario (`290` ceros y `478` unos) y todos sus estratos
`design_stratum × cardinality` poseen al menos 14 filas. El diagnóstico previo
halló, para ambos posteriors, disagreement y ambas clases de guard en las dos
poblaciones W59. Estos conteos son observaciones de factibilidad, no resultados
del runner ni claims prospectivos.

La preparación materializa cuatro bundles nuevos:

1. `posterior_fit_truth.npz`: logits, target, identidad y estratos W54;
2. `policy_fit_truth.npz`: logits, per-seed logits, target, identidad y estratos;
3. `decision_select_public.npz`: sólo identidad, logits, per-seed logits y
   metadatos públicos necesarios para construir el posterior y features;
4. `decision_select_truth.npz`: sólo `pair_token` y target.

El preparador histórico puede leer la fuente mixta ya abierta, pero debe
escribir el bundle público antes de cualquier ajuste o evaluación. Por eso este
hito valida separación lógica, schemas y target-blind application; no acredita
aislamiento físico ante un operador malicioso. Una corrida prospectiva futura
deberá materializar equivalentes bajo permisos/procesos separados antes de
abrir los targets.

El applier acepta exclusivamente `decision_select_public`, estados congelados,
utilities y thresholds. El evaluator acepta acciones congeladas,
`decision_select_truth` y utilities. Ninguna API de inferencia recibe `target`,
`gain`, regrets observados ni clases de guard.

## 5. Primitives nuevas

Se implementa un módulo autocontenido
`src/geometria_proporcional/proportional_set_valued_native.py`. El runtime
reutiliza W53–W54 donde la semántica coincide. W52 queda como autoridad
semántica y fuente ligada, pero **no se importa**, porque ese módulo carga
`torch`: las fórmulas NumPy de utilities, acción autorizada y regret se
reimplementan localmente y se validan por fixtures de paridad independientes.
Tampoco se llama
`wave56_contextual_gate.contextual_design`, porque ese builder reconstruye un
set por threshold `0.5` en lugar del set MAP requerido.

### 5.1 Representación MARGINAL

El fit aplana logits y targets como `[4*N,1]` y usa
`LogisticRegression` scikit-learn `1.8.0`: `C=1`, L2 mediante `l1_ratio=0`,
`lbfgs`, intercept, `dual=false`, `class_weight=null`, seed `5301`,
`max_iter=1000`, float64. Se exige pool con ambas clases, convergencia
`n_iter < max_iter` y parámetros finitos.

El estado portable conserva coeficiente, intercept, clases, iteraciones,
versión de sklearn y receta. La predicción canónica se reconstruye con
`expit(coef*logit+intercept)` sin pickle. Las cuatro probabilidades generan una
masa Bernoulli-producto condicionada a set no vacío mediante W53. Una prueba
compara la reconstrucción contra `predict_proba` a tolerancia `2e-15`.

### 5.2 Representación JOINT

El fold id se construye sin labels. Dentro de cada
`(design_stratum,cardinality)`, se ordena por bytes SHA-256 de
`b"set-fold-v1" + pair_token.encode("utf-8")`, con `pair_token` como desempate,
y se asigna `rank % 4`. Cada fold y su complemento deben contener filas.

Para cada lambda `[1e-4,1e-3,1e-2,1e-1,1,10]`, `joint_full` se ajusta en tres
folds y predice el cuarto. La clave ascendente es:

```text
(mean_oof_exact_set_nll, mean_oof_marginal_brier, -regularization)
```

El optimizer es W54 `L-BFGS-B`, `max_iter=2000`, `gtol=1e-9`,
`ftol=1e-12`. Toda no convergencia o no finitud invalida el fit. Seleccionada la
lambda, se reajusta sobre las 192 filas. Se conservan folds, métricas OOF por
fila y lambda, theta final, coeficientes de interacción, gradiente, iteraciones
y mensaje.

### 5.3 Target-shuffle común de posterior

Se implementa literalmente `target_derangement_v1` del freeze con
`PCG64(53602)`, estratos `(fold_id,design_stratum,cardinality)`, consumo
`random_raw` y rotación no nula. El mapa canónico contiene
`receiver,donor,fold_id,design_stratum,cardinality`, orden receiver UTF-8,
JSON compacto con newline. Debe reproducir el digest de fixture
`d7aa2f128b6d42dbe7448415dcd8d4d69ca0ad8311394a5b1209ca9579e03904`.

El mismo target permutado ajusta MARGINAL-SHUFFLED y JOINT-SHUFFLED.
JOINT-SHUFFLED ejecuta de nuevo las seis lambdas sobre ese target, con los
mismos folds target-blind, métricas OOF, clave, optimizer y budget, y reajusta
su propia lambda elegida. No recibe ningún hiperparámetro seleccionado con el
target real. Se conservan por separado el grid OOF, lambda y estado final de
JOINT y JOINT-SHUFFLED. Logits, folds y demás campos permanecen byte-idénticos.
Se registran singletons, fracción permutable y ausencia de cruces de estrato.
Una fixture fuerza lambdas distintas y una mutación que copia la lambda real al
control debe fallar.

### 5.4 HARD ligado al posterior

Para cada fila se toma `argmax(set_mass)` en el orden binario creciente de los
15 sets; `numpy.argmax` resuelve empates por el menor índice. Para cada una de
las 24 utilities, se elige el miembro del set MAP con mayor utilidad y luego la
familia de menor índice. Se persisten `map_set_index`, `map_set`,
`map_set_mass` y acciones `[N,24]` por posterior.

Un fixture fuerza que set MAP y set por marginals `>=0.5` difieran. El checker
debe detectar si HARD o sus features vuelven a la receta histórica.

### 5.5 CONTEXTUAL ligado al posterior

Para cada posterior, el candidato es la acción de mínimo riesgo posterior y el
baseline su HARD-MAP. El design float64 conserva exactamente 17 features:

```text
advantage, hard_risk, minimum_risk, action_risk_margin,
posterior_entropy_norm, posterior_top_mass, posterior_top_margin,
baseline_map_cardinality, posterior_expected_cardinality,
posterior_cardinality_variance, posterior_mass_baseline_map_set,
seed_std_mean, seed_std_max, utility_f0, utility_f1, utility_f2, utility_f3
```

`advantage = hard_risk-minimum_risk` sin clipping adicional. El disagreement
es `hard_action != candidate`. Las filas activas reciben peso `1/k_token`, por
lo que cada token con `k_token>0` aporta masa total uno.

El proposer usa el scaler ponderado W56 y Ridge float64 `alpha=1`. Se resuelve
con intercept no penalizado:

```text
(X_aug.T W X_aug + alpha*diag(0,1,...,1)) beta = X_aug.T W gain
```

mediante `numpy.linalg.solve`; un sistema singular produce un reason code y no
activa pseudoinversa silenciosa. Los guards usan el mismo scaler de receta pero
estados independientes y `LogisticRegression` scikit-learn `1.8.0`, `C=1`,
L2 mediante `l1_ratio=0`, `lbfgs`, `dual=false`, sin class weights, intercept,
`max_iter=2000`, `tol=1e-10`, `warm_start=false`. Sus targets son
`gain < -1e-12` y `not target[candidate]`. Cada guard exige ambas clases.

Los tres modelos se ajustan sólo en `policy_fit_opened`. Se exportan scalers,
coeficientes, intercepts, clases, iteraciones y receta a arrays/JSON portables;
las predicciones reconstruidas deben coincidir con el objeto sklearn antes de
descartarlo.

En `decision_select_opened`, los thresholds usan
`numpy.quantile(method="linear")` sobre todas las filas disagreement:

- proposer: `[.5,.6,.7,.8,.9,.95,.975]`, autoriza `score > threshold`;
- harm guard: `[.1,.2,.3,.4,.5,.6,.8]`, autoriza `p < threshold`;
- incompatibility guard: la misma grilla y comparación.

El producto cartesiano tiene `343` candidatas por posterior, más HARD-ONLY.
Cada candidata se aplica sin target y luego se evalúa, sobre las mismas
`768×24` posiciones, con clave ascendente:

```text
(mean_regret, incompatibility_rate, harm_rate, -authorized_rows,
 proposer_quantile, harm_quantile, incompatibility_quantile)
```

HARD-ONLY usa `2.0` como sentinel de los tres cuantiles sólo en el desempate.
`harm_rate` es la fracción de posiciones cuya acción seleccionada tiene regret
mayor que HARD por más de `1e-12`; `authorized_rows` cuenta overrides efectivos.
Se conservan scores, thresholds, masks, acciones, métricas por candidata y el
estado elegido antes de cualquier aplicación posterior.

## 6. Controles matched

Cada posterior ajusta cinco tríos proposer/harm/incompatibility con seeds
`53611,53617,53623,53629,53633`. El control preserva design, weights y soporte
activo. En cada estrato `(policy_index,disagreement_count)` construye una única
permutación de donantes, común a los tres targets.

Para cerrar la frase operativa “maximiza desplazamiento Hamming con costo hash”,
el algoritmo queda fijado así:

1. cada fila activa tiene firma exacta `(gain_float64_bits,harm,incompatibility)`;
2. la matriz primaria de costo es el Hamming de las tres coordenadas entre
   receiver y donor;
3. `PCG64(seed).bit_generator.random_raw()` genera un valor por arista en orden
   `(policy,disagreement_count,receiver UTF-8,donor UTF-8)`; para cada receiver,
   los donors se ordenan por `(random_raw,donor UTF-8)`;
4. la diagonal se prohíbe para estratos de tamaño mayor que uno;
5. `scipy.optimize.linear_sum_assignment(..., maximize=True)` calcula primero
   el máximo Hamming total; luego, en receiver UTF-8 order, se prueba cada donor
   en su orden seeded y se fija el primero que permite a las filas restantes
   conservar exactamente ese máximo; la factibilidad restante se recalcula
   por assignment Hamming;
6. un singleton conserva donor=receiver, se marca no permutable y permanece en
   el fit para no cambiar soporte;
7. la misma asignación transporta gain, harm e incompatibility.

El vector de ranks seeded de donors elegido en receiver order es un desempate
lexicográfico total a nivel de permutación, no una suma de ranks de aristas. Así
dos matchings con la misma suma secundaria no quedan a criterio de SciPy. La
config y el receipt ligan NumPy `2.3.5` y SciPy `1.17.0`; una fixture de tres
filas reproduce el caso de dos derangements con igual suma y exige un mapa
literal.

Se valida optimalidad primaria contra el costo devuelto, ausencia de identidad
en filas permutables, identidad sólo en singletons declarados, preservación
multiset de cada target dentro del estrato, fracción permutable `>=0.8`, soporte
de fit idéntico al reader verdadero y ambas clases en cada guard. Si los masks
de fit resultantes difieren entre seeds, toda la familia matched es
`NOT_EVALUABLE_CONTROL_SUPPORT`; no se promedian controles faltantes.
Además, los cinco `mapping_sha256` y los cinco digests del triplete transportado
deben ser distintos por posterior. Cualquier colisión produce
`NOT_EVALUABLE_CONTROL_DIVERSITY`; no hay retries ni sustitución de seeds.

Cada control usa las mismas recetas de Ridge/logistics. Tras elegir la tripleta
de cuantiles del reader verdadero, el control calcula sus propios thresholds
en `decision_select_public` con esa misma tripleta y sin target.

Para cada token, `k` es la cantidad de overrides verdaderos. El universo de un
control son posiciones disagreement que sus tres estados autorizan. Se ordena
por `(-proposer_score,harm_probability,incompatibility_probability,policy_index)`
y se toman exactamente `k`. Si no alcanza, `match_valid=false`; si `k=0`, el
match vacío es válido pero el token no pertenece a `U_true`. La máscara común
es la intersección exacta de `U_true` y los cinco `match_valid`. Se registran
todos los mapas y hashes antes de que el evaluator reciba target. El fixture
informa cobertura y métricas matched sólo como diagnóstico de implementación.

## 7. Estimandos, bootstrap y lectura diagnóstica

El runner conserva por `pair_token` y celda, antes de agregar:

- exact-set NLL, Brier marginal, error absoluto de cardinalidad y masa del
  target verdadero;
- accuracy de acción, incompatibility, regret medio sobre las 24 policies y
  worst regret por token;
- overrides, daño, `U_true`, cinco `match_valid`, `U_common` y cobertura.

Los deltas se orientan siempre primer término menos segundo; para losses,
valores negativos favorecen el primer término. Se generan `5000` resamples
pareados con reemplazo sobre índices de `pair_token`, nunca sobre filas
`token×policy`. El soporte global de 768 tokens usa `PCG64(53641)`; los soportes
`U_common` de MARGINAL y JOINT usan, respectivamente, `PCG64(53642)` y
`PCG64(53643)`. Cada matriz `int64 [5000,N_support]`, el orden de tokens y el
SHA-256 se persisten. Los intervalos son percentiles `2.5/97.5` de la media
pareada; no se les atribuye cobertura familiar ni variabilidad de training.

La tabla diagnóstica reconstruye las ocho filas del freeze:

| ID | Delta y soporte |
|---|---|
| `SET_JOINT_NLL` | JOINT-real menos MARGINAL-real en exact-set NLL, 768 tokens |
| `SET_JOINT_BRIER` | JOINT-real menos MARGINAL-real en Brier marginal, mismos tokens |
| `SET_SHUFFLE` | real menos target-shuffled en exact-set NLL, separado por posterior; la condición JOINT usa su propio shuffled |
| `READER_REGRET` | CONTEXTUAL menos HARD en regret, separado por posterior |
| `READER_COMPAT` | CONTEXTUAL menos HARD en incompatibility-rate, separado por posterior |
| `READER_WORST` | CONTEXTUAL menos HARD en worst-regret por token, separado por posterior |
| `READER_CONTROL` | CONTEXTUAL verdadero menos media aritmética por token de los cinco matched, sobre el `U_common` de ese posterior |
| `FACTOR_INTERACTION` | `(JOINT_CONTEXTUAL-JOINT_HARD) - (MARGINAL_CONTEXTUAL-MARGINAL_HARD)` en regret, 768 tokens |

Las primeras siete filas usan las reglas congeladas: condición satisfecha si
`CI_upper < 0` para NLL, shuffle, regret y control; `CI_upper <= 0` para Brier,
compatibility y worst. Si falta soporte requerido, el estado es
`NOT_EVALUABLE`; si no y `CI_lower > 0`, es `ADVERSE`; en los demás casos,
`NOT_RESOLVED`. `NOT_EVALUABLE` tiene precedencia sobre `ADVERSE`, que precede
a `NOT_RESOLVED`. `FACTOR_INTERACTION` es descriptiva. Toda etiqueta se
prefija o anida bajo `OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC`.

`JOINT_PATTERN_PRESENT` requiere las dos filas joint-vs-marginal y
JOINT-real-vs-JOINT-shuffled. `CONTEXTUAL_PATTERN_PRESENT` se calcula por
posterior y requiere sus cuatro filas reader. El runner reconstruye esos
predicados para probar la lógica, pero no los transforma en selección
científica, ranking o promoción.

Las tres entradas `per_seed_logits`, asociadas a checkpoints `17/29/43`, se
aplican una por una a los estados y thresholds congelados. Se persisten
métricas por checkpoint, posterior, reader, policy y cardinalidad sin pooling
entre checkpoints y sin tratarlas como seeds de una población. También se
declaran todas las duplicaciones empíricas exactas o parciales entre celdas;
ninguna celda se elimina por coincidir.

## 8. Arquitectura del runner y artefactos

El runner se implementa en
`experiments/geometria_proporcional/run_proportional_set_valued_native_preflight.py`
y el checker en
`experiments/geometria_proporcional/check_proportional_set_valued_native_preflight.py`.
Una config nueva contiene únicamente recetas, allowlist de fuentes, schemas,
budgets y claims fijos; no contiene un resultado científico.

El artefacto canónico será:

```text
data/geometria_proporcional/proportional_set_valued_native_preflight_v1/
  config.snapshot.json
  source_bindings.json
  input_receipt.json
  prepared/
    fixture_manifest.json
    posterior_fit_truth.npz
    policy_fit_truth.npz
    decision_select_public.npz
    decision_select_truth.npz
  posterior_fit/
    states.json
    state_arrays.npz
    oof_arrays.npz
    target_shuffle_map.json
    target_shuffle_arrays.npz
  policy_fit/
    feature_schema.json
    states.json
    state_arrays.npz
    fit_scores.npz
    control_maps.json
    control_arrays.npz
  decision_select/
    scores.npz
    candidate_metrics.npz
    selection_freeze.json
    action_arrays.npz
  apply_fixture/
    action_freeze.json
    actions_and_matches.npz
  evaluate_fixture/
    diagnostic_metrics.json
    diagnostic_arrays.npz
    bootstrap_indices.npz
    estimand_table.json
    sensitivity_arrays.npz
    cell_duplications.json
  runtime.json
  replay_receipt.json
  artifact_manifest.json
  REPORT.md
```

Todos los NPZ canónicos usan keys ordenadas, NPY con `allow_pickle=false`, ZIP
DEFLATED nivel 9 y timestamp fijo `1980-01-01`, para igualdad byte a byte. JSON
usa UTF-8, `ensure_ascii=false`, keys ordenadas, separadores compactos, rechazo
de NaN/Inf y newline final. No se serializan estimadores con pickle/joblib.

El runner primario y el replay se ejecutan con
`CUDA_VISIBLE_DEVICES=''`, un thread BLAS/OpenMP y un guard que falla si la
variable no es exactamente vacía. Ni el runtime ni el checker pueden importar
`wave52_policy.py` o cargar `torch`; un test falla si `torch` aparece en
`sys.modules` tras importar o ejecutar sus entrypoints dentro de un subprocess
fresco. El output existente no se borra: `--force` lo archiva de forma
recuperable con sufijo explícito.

El manifest clasifica cada archivo como `source_snapshot`, `raw_state`,
`derived_diagnostic`, `receipt` o `regenerable_report`; incluye path relativo,
bytes y SHA-256. El replay exige igualdad byte-exacta de todos los artefactos
salvo `runtime.json`, `replay_receipt.json`, paths de output y timestamps. Los
campos excluidos quedan enumerados, no implícitos.

## 9. Checker y mutaciones

El checker no confía en `REPORT.md`. No puede importar
`proportional_set_valued_native.py`, el runner ni helpers definidos por ellos.
Puede usar NumPy/SciPy y fuentes históricas congeladas, pero recompone de forma
independiente folds, masas desde estados portables, MAP, features, modelos
lineales, assignment, matching, métricas, bootstrap y tabla de estimandos.
Produce una tabla PASS/FAIL con reason codes y sale distinto de cero ante
cualquier FAIL. Tests de independencia corrompen un helper del runner y raw
coherente con ese bug para exigir que la recomputación externa lo detecte.

| ID | Condición exacta | Reason code principal |
|---|---|---|
| `P1_SOURCE_AND_SCOPE` | hashes, allowlist, git source, CPU y status preflight válidos | `SOURCE_OR_SCOPE_INVALID` |
| `P2_PREPARED_PHASES` | cuatro bundles, schemas, roles y tokens disjuntos; public sin truth | `PHASE_BUNDLE_INVALID` |
| `P3_MARGINAL_NATIVE` | receta, convergencia, estado portable y masa normalizada | `MARGINAL_RECIPE_INVALID` |
| `P4_JOINT_NATIVE` | folds, seis lambdas, OOF, selección y refit reproducibles | `JOINT_RECIPE_INVALID` |
| `P5_TARGET_SHUFFLE` | digest fixture, mapa común, estratos y soporte exactos | `TARGET_SHUFFLE_INVALID` |
| `P6_HARD_MAP_BINDING` | set MAP, tie-break y acciones ligados a cada masa | `HARD_POSTERIOR_BINDING_INVALID` |
| `P7_CONTEXTUAL_DESIGN` | 17 features, MAP adapter, paridad de receta y weights | `CONTEXTUAL_DESIGN_INVALID` |
| `P8_MODEL_STATES` | Ridge/logistics exactos, clases, escalers y predicción portable | `CONTEXTUAL_STATE_INVALID` |
| `P9_SELECTION` | 344 celdas, thresholds, aplicación target-blind y clave exacta | `SELECTION_PROTOCOL_INVALID` |
| `P10_MATCHED_CONTROLS` | 5 seeds, mapa común por trío, soporte igual y matching exacto | `MATCHED_CONTROL_INVALID` |
| `P11_CELL_AND_ESTIMAND_PARITY` | raw por token, 5000 bootstraps, ocho filas, orientación, soportes, CIs, precedencia, patterns, sensitivities y duplicaciones exactos | `CELL_ESTIMAND_MISMATCH` |
| `P12_RAW_AND_REPLAY` | inventario completo, hashes y replay byte-exacto | `RAW_OR_REPLAY_INVALID` |
| `P13_CLAIM_BOUNDARY` | sólo diagnóstico abierto; sin promoción ni decisión científica | `CLAIM_BOUNDARY_INVALID` |
| `P14_COST_CONTRACT` | runtime primario+replay y RSS dentro de límites duros; suites auxiliares dentro de su budget separado | `COST_CONTRACT_INVALID` |

La suite de mutaciones debe cambiar un solo elemento por caso y verificar el
reason code esperado. Como mínimo cubre:

- hash de fuente, path prohibido y `CUDA_VISIBLE_DEVICES` no vacío;
- target agregado al bundle público o lectura de truth por el applier;
- overlap o duplicado de `pair_token` entre fases;
- utilidad inyectada al ajuste del posterior;
- Platt por familia, class weights o receta divergente;
- fold construido con target, lambda omitida, tie-break invertido o lambda real
  copiada a JOINT-SHUFFLED;
- shuffle distinto entre representaciones, identidad o cruce de estrato;
- HARD por marginals threshold en vez de set MAP;
- feature order alterado, cardinalidad threshold o masa del set incorrecto;
- clipping de advantage, peso por fila o scaler divergente;
- Ridge con pseudoinversa, alpha distinto o intercept penalizado;
- guard de signo invertido, clase ausente o Logistic divergente;
- quantile `higher`, comparación no estricta, grilla incompleta o HARD_ONLY
  ausente;
- control con mapas distintos para los tres targets, seed omitido, identidad,
  mapa/target-triplet duplicado, soporte variable o promedio de menos de cinco;
- matching que usa target, toma otro `k`, cambia el sort o usa unión de masks;
- pérdida/penalty/utility distintos entre celdas;
- bootstrap por policy-row, seed/soporte distinto, orientación invertida, fila
  faltante, precedencia o pattern alterado;
- raw faltante, NPZ mutable, replay divergente o manifest incompleto;
- import de W52/torch, checker que importa el código bajo prueba, o frase/campo
  que promueva una arquitectura o declare decisión científica.

Fixtures unitarias adicionales verifican: set MAP distinto del threshold 0.5,
empates de set/utilidad, exactitud de gradiente JOINT, portabilidad de los tres
modelos, optimalidad del assignment frente a enumeración exhaustiva para
estratos pequeños, estabilidad ante reordenar filas y common support exacto.

## 10. Ejecución y presupuesto

La secuencia es:

1. tests unitarios de primitives y checker;
2. preflight de fuentes, scope y entorno CPU;
3. corrida primaria sobre los tres fixtures abiertos;
4. checker independiente sobre el artefacto primario;
5. replay desde cero en un output separado;
6. comparación byte-exacta y checker del replay;
7. suite completa de mutaciones;
8. auditoría independiente final del código, artefactos y claims.

El diagnóstico exploratorio previo completó 24 fits JOINT OOF, refit MARGINAL y
refit JOINT en menos de dos segundos de pared con un thread. El presupuesto del
runner completo conserva el freeze: `120/420/1800 s` para
lower/central/upper, primario más replay, y `1.5 GiB` de RSS máximo. Se miden
wall time, CPU time y peak RSS por fase. Esos límites cubren exactamente corrida
primaria más replay; el checker doble y las mutaciones tienen un budget auxiliar
separado de `900 s` total y el mismo límite RSS por proceso. Exceder el central
sólo clasifica `ABOVE_CENTRAL_ESTIMATE`; exceder `1800 s`, `900 s` auxiliar o
`1.5 GiB` produce siempre FAIL con reason code, aunque la explicación del exceso
se preserve. No se sustituye ninguna futura etapa GPU por cómputo CPU largo:
este pipeline es nativamente tabular y CPU.

## 11. Auditoría, documentación y siguiente objetivo

La auditoría de plan se archiva verbatim en
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/`; todo finding
válido alto o medio se corrige y, si cambia el contrato, se reaudita. La
auditoría final debe leer plan, código, config, tests, artefacto primario,
replay, checker y reporte completo, y repetir una muestra independiente de
cálculos desde raw.

Al cierre se escribe una síntesis durable en `Biblioteca/` y se propaga sólo el
estado canónico necesario a la wiki, Estado Actual, bitácora y documentos
transversales conforme a la política documental. Los resultados permanecen
separados como implementación, diagnóstico histórico y deuda prospectiva.

Si el preflight resulta válido, el siguiente objetivo automático será diseñar
y auditar el paquete de ejecución prospectiva físicamente separada —sin crear
todavía el draw—: schemas finales, workers por fase, permisos, source freeze,
escrow, receipts, recovery y costo. Si el preflight no es válido, el siguiente
objetivo automático será una corrección finita del runner delimitada por los
reason codes observados. Ninguna rama habilita GPU ni promoción por sí misma.
