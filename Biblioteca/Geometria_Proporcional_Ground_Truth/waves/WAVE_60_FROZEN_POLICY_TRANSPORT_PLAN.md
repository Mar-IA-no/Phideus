# Ola 60 — transporte prospectivo de políticas congeladas entre draws

> **Estado:** `REVISED-AFTER-R457 / PRE-IMPLEMENTATION / PRE-DRAW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-05
> **Antecedente:** `WAVE_59_FRESH_HGB_GUARD_BRACKET_CLOSED.md`
> **Auditoría del draft:** `R457 / REVISE / 3 HIGH + 2 MEDIUM`
> **Informe R457:** `../agent_reports/457_wave60_frozen_transport_plan_audit.md`
> **Pregunta:** ¿la señal de las políticas HGB/HGB de Ola 59 transporta a una
> realización independiente sin refit, recalibración ni selección?

## 1. Motivo del experimento

La Ola 59 produjo una observación doble. Las políticas HGB/HGB de
incompatibilidad y harm mejoraron varias métricas frente al hard-set, pero
ninguna separó sus cinco controles de desplazamiento condicional máximo. El
primer patrón cerró `7/8`; el segundo, `6/8`. Ese resultado debilita el bracket
vigente, aunque no permite distinguir todavía entre tres explicaciones:

1. la ley aprendida contiene señal real, pero el draw no tuvo potencia para
   separarla de los controles;
2. la señal depende de fit y calibración dentro de la misma realización;
3. el efecto proviene principalmente de la magnitud y localización de la acción,
   no del target aprendido por el guard.

Repetir el mismo roster con refit y recalibración volvería a mezclar esas
explicaciones. La Ola 60 cambia el estimando: transporta la ley completa de Ola
59 a un draw nuevo. Los estados, umbrales, roster, controles, prioridad y
criterios quedan congelados antes de generar la nueva realización.

El resultado seguirá limitado a un generador sintético. Puede informar
transporte entre realizaciones de esa ley; no confiere autoridad física, no
valida una geometría proporcional general y no decide promoción ni
`GO/NO-GO`.

## 2. Hipótesis y estimandos

### 2.1 Hipótesis principal

Una regla que usa información contextual pertinente debe conservar, fuera del
draw donde fue ajustada, una ventaja respecto de hard y respecto de controles
entrenados con targets de desplazamiento matched. El transporte se evalúa con
los mismos dos objetivos, sin combinarlos:

- **media:** `P-HGB-HGB-INCOMPATIBILITY-Q90`;
- **cola:** `P-HGB-HGB-HARM-Q70`.

### 2.2 Estimando de transporte y contraste comparativo

Para cada política, el estimando primario es el vector de diferencias sobre el
lockbox fresco:

```text
(accuracy, compatibility, regret, worst_regret)
  política congelada de Ola 59 − hard-set fresco
```

El contraste comparativo de transporte de pipelines congeladas es:

```text
métrica de la política principal
  − promedio de la misma métrica en sus cinco controles congelados
```

Todos los términos se calculan sobre los mismos pair tokens, acciones hard y
posterior, utility matrix y bootstrap. No se iguala ex post el soporte, no se
seleccionan controles por resultado y no se recalibran thresholds para recuperar
cobertura.

Este contraste no identifica por separado el efecto causal del target aprendido,
la localización contextual, el estado ajustado, el threshold o la cobertura que
transporta. Compara pipelines completas congeladas. Una atribución específica a
target o localización exigiría otro factorial predeclarado o una intervención de
cobertura; no se incorporará post hoc a esta ola.

### 2.3 Preguntas secundarias predeclaradas

- transporte del proposer HGB sin guard frente a hard;
- cambio de soporte de propuesta y autorización respecto de Ola 59;
- estabilidad de scores por modelo: media, desviación, cuantiles y fracción
  fuera del rango observado en validation de Ola 59;
- solapamiento de acciones entre política principal y controles;
- descomposición descriptiva por policy index y número de desacuerdos.

Estas salidas son diagnósticas. No pueden sustituir el patrón principal ni
seleccionar un nuevo threshold.

## 3. Ley congelada de origen

La única fuente de modelos y calibración es la ejecución primaria válida de Ola
59:

```text
data/geometria_proporcional/
  wave59_fresh_hgb_guard_bracket_replay_normalized_v1/
```

El protocolo liga por SHA-256, como mínimo:

| Artefacto | SHA-256 |
|---|---|
| `fit/fit_freeze.json` | `1cd2807970a8c8f6b2d0e88abf212c4a0ba797ed67e9f36cb115a90fe941db8a` |
| `fit/model_states/manifest.json` | `942651e39c69a65378185de60fdbf770b2df221e62eaf6c6a614055cb719b5c7` |
| `fit/model_state_arrays.npz` | `6539ac1fed2f2d030ba8c8dcb0b76c64aed2a76795f63cff245e1f16e371b3a4` |
| `calibration/calibration_freeze.json` | `6ad2c2c9512c206e0982bede3b0dfaee694187292fb1badeaecb69cd3451367a` |
| `adjudication/monitor_scores.npz` | `61e691babb482360e5e4ea4dd112e3c3c16c9a8e25c0670e2f618db46438d787` |
| `adjudication/monitor_policy_arrays.npz` | `e44fe6d3a0ecc5510c814703ad9eced822c76e7c3bef7a0f827a3a9055fcfb74` |
| `artifact_manifest.json` | `909361b45e9fb51977229063afcdf9587144bfb80280fd3c70dda86de19a6371` |
| config Ola 59 | `f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6` |
| auditoría R454 | `e320d48c1c1b198fd2e324a0df7d534c088f417502d1bb20167c4050a29fe926` |

Se congelan trece modelos portables:

- `proposer-hgb`;
- `guard-hgb-incompatibility`;
- `guard-hgb-harm`;
- `control-hgb-incompatibility-59041..59045`;
- `control-hgb-harm-59031..59035`.

La config Wave 60 copiará literalmente del calibration freeze:

- proposer threshold `0.3352357782028221`, comparación estricta `>`;
- incompatibility threshold `0.05147286376407591`, comparación estricta `<`;
- harm threshold `0.4714312385695055`, comparación estricta `<`;
- los diez thresholds individuales de control, sin promediarlos ni
  recuantilizarlos.

El checker verificará esos números contra el artefacto físico. La config no es
autoridad para inventar valores distintos.

## 4. Verificación previa del transporte

Antes de admitir un draw nuevo, la implementación debe demostrar sobre Ola 59:

1. que el manifest tiene forma cerrada y contiene exactamente dieciséis estados:
   los trece requeridos y, como únicos no usados, `proposer-ridge`,
   `guard-logistic-harm` y `guard-logistic-incompatibility`;
2. que cada tree key del estado portable existe exactamente una vez en
   `model_state_arrays.npz` y no hay keys usadas por dos modelos;
3. que el scorer portable reproduce bit a bit los scores raw preservados de
   monitor para los trece modelos: finitos exactamente sobre `score_mask =
   disagreement` y NaN fuera de ella;
4. que `apply_calibrated_policies()` completo reproduce retrospectivamente las
   proposals, authorized masks y actions Wave 59 con los dieciséis estados, y
   que el aplicador transport-only de Ola 60 reproduce exactamente las 26 arrays
   seleccionadas usando sólo los trece modelos autorizados;
5. que feature count, orden, dtype y semántica son los 17 de
   `wave56_contextual_gate.FEATURE_NAMES`;
6. que fit freeze, calibration freeze y manifest están incluidos con sus hashes
   exactos en el manifest cerrado de Ola 59.

Esta comprobación acredita portabilidad mecánica. No mira el futuro draw ni
produce una inferencia nueva.

## 5. Draw fresco y fronteras de acceso

La preparación reutiliza la ley generativa y los contratos de Ola 59, pero crea
una config, tres claves, escrow, attestation y raíces nuevas. La semántica
esperada por split permanece:

- `4.992` filas visibles;
- `1.152` pair tokens totales;
- `768` pair tokens elegibles;
- `192` no canónicos;
- `384` out-of-catalog.

Estos conteos son expectativas de contrato; deben comprobarse sobre la nueva
realización y no copiarse como resultado.

### 5.1 Guardia de realización nueva

La independencia no se presume a partir de una seed o de un nombre de output.
Antes de declarar `PREPARED`, el coordinador compara opacamente la realización
Ola 60 contra estas cinco raíces físicas de Ola 59, congeladas en la config:

```text
wave59_fresh_hgb_guard_bracket_replay_normalized_v1/
wave59_fresh_hgb_guard_bracket_replay_normalized_v1_replay/
wave59_fresh_hgb_guard_bracket_v1/
wave59_fresh_hgb_guard_bracket_v1.failed_20260905T071003529517Z/
wave59_fresh_hgb_guard_bracket_v1_replay.failed_20260905T102929791142Z/
```

Para cada raíz compara todo elemento existente de la siguiente lista cerrada,
sin descifrar claves ni abrir semántica sellada:

- SHA-256 de `generation_escrow.json` y sus tres `key_commitments`;
- `generation_key_commitment`, `identity_key_commitment` y
  `semantic_commitment_key_commitment` del manifest;
- hash de `attestations/semantic_root.json`, mapa completo de
  `commitments/*` y commitments de los archivos del benchmark;
- hashes de todos los bundles preparados inference-safe y truth;
- paths resueltos y pares `(st_dev, st_ino)` de los archivos protegidos.

Las roots Ola 60 deben resolver a directorios distintos, sin aliases, symlinks
ni hardlinks hacia ninguna raíz anterior. Escrow, tres key commitments, semantic
root, mapa de commitments y bundles Ola 60 deben ser no idénticos a cada
realización Ola 59 que contenga el elemento comparable. La primaria y el replay
Ola 60, en cambio, deben compartir por bytes y hashes el mismo escrow nuevo y
sus commitments, aunque cada paquete y attestation sea independiente; no se
comparten archivos mediante hardlinks.

Cualquier identidad o colisión aborta antes de acceso semántico como
`INVALID_NEW_DRAW_IDENTITY`. No se convierte en `NOT_EVALUABLE` ni habilita un
redraw silencioso.

### 5.2 Topología física del lockbox

Se agrega un worker dedicado:

```text
experiments/geometria_proporcional/_wave60_phase_worker.py
```

El runner privilegiado coordina, valida firmas, materializa directorios de fase
cerrados y promueve outputs; no calcula scores, acciones ni métricas. Lanza dos
invocaciones independientes del worker como UID/GID `65534`, capabilities
vacías y `NoNewPrivs=true`:

1. `score_apply`, con allowlist física exacta:
   `config.snapshot.json`, `source_bindings.json`,
   `transport_law_manifest.json`, `transport_law_arrays.npz`,
   `frozen_policy_spec.json`, `feature_schema.json` y
   `sealed_monitor_inference_bundle.npz`. No monta ni vuelve legibles truth,
   train o validation del draw nuevo.
2. `evaluate`, con allowlist física exacta:
   `config.snapshot.json`, `source_bindings.json`, `evaluation_index.npz`,
   `monitor_policy_arrays.npz`, `monitor_action_freeze.json`,
   `sealed_monitor_truth_bundle.npz` y `utilities.npy`. No recibe design,
   scores, modelos, thresholds, train ni validation, y por tanto no puede
   recalcular o alterar acciones.

`transport_law_manifest.json` y `transport_law_arrays.npz` son una proyección
lossless, hasheada y closed-world de los trece estados y sus 1.300 tree keys;
los tres estados no transportados no aparecen. Cada worker rechaza entradas
extra o faltantes, paths no allowlisted y outputs preexistentes. Su receipt
registra schema/phase, UID/GID, capabilities, `NoNewPrivs`, inventario exacto de
inputs/outputs, bytes y SHA-256, paths abiertos y el resultado de probes físicos
de denegación. El coordinador valida el receipt antes de promover la fase.

El generador puede materializar train y validation por compatibilidad con la ley
existente, pero Wave 60 los clasifica como **no usados**. El runner científico
recibe únicamente:

- inference-safe lockbox: design, disagreement, primary, hard actions,
  posterior actions y pair tokens;
- truth lockbox sellado: targets, weights y estados necesarios para métricas.

Orden obligatorio:

```text
PREPARED
  -> SOURCE_LAW_VERIFIED
  -> LOCKBOX_SCORES_FROZEN
  -> LOCKBOX_ACTIONS_FROZEN
  -> LOCKBOX_TRUTH_AUTHORIZED
  -> COMPLETE
```

Antes de `LOCKBOX_ACTIONS_FROZEN` no se puede abrir truth de lockbox. Train y
validation del draw nuevo no se abren en ninguna fase científica. Un acceso,
refit, cuantile calculation o selección sobre ellos invalida el intento.

## 6. Política de acciones

El scorer conserva dos dominios separados:

```text
score_mask = disagreement
decision_mask = primary AND disagreement
```

Los scores raw son finitos en todo `score_mask` y NaN exactamente fuera de él.
Sobre `decision_mask`:

```text
proposal = proposer_score > frozen_proposer_threshold
authorized = proposal AND guard_score < frozen_guard_threshold
action = posterior_action si authorized; hard_action en otro caso
```

Fuera de `decision_mask`, proposal y authorized son false y la acción debe ser
hard, aunque un score raw pueda ser finito en un disagreement no primario. Cada
control usa el mismo proposal HGB y su propio guard/threshold congelado.

El módulo Ola 60 define un aplicador transport-only closed-world. Acepta
exactamente un proposer HGB, dos políticas principales, diez controles y las
referencias `HARD-SET` y `HGB-PROPOSER-ONLY`; rechaza todo modelo, policy key,
threshold o salida adicional. El aplicador completo de Ola 59 sólo participa en
la prueba retrospectiva de equivalencia y nunca procesa el draw nuevo. No se
permite:

- refit de ningún estimator;
- cálculo de quantiles sobre el draw nuevo;
- normalización aprendida en el draw nuevo;
- reemplazo de thresholds por valores de igual soporte;
- selección entre Q70/Q90;
- combinación harm+incompatibility;
- tuning por policy index, estrato o soporte observado.

Las referencias informadas son `HARD-SET` y `HGB-PROPOSER-ONLY`. Pure posterior
y oracle pueden aparecer sólo como descriptivos separados si el plan auditado y
la config final los enumeran; nunca entran al patrón.

## 7. Unidad estadística y bootstrap

La unidad primaria continúa siendo `pair_token` elegible y primario. Los tokens
se ordenan lexicográficamente antes de cualquier resampling. Un único tensor de
índices `PCG64`, seed nueva congelada y `5.000` réplicas se comparte entre todos
los brazos, controles y métricas del lockbox.

Los intervalos son condicionales a:

- el draw nuevo;
- los modelos, thresholds y controles congelados de Ola 59;
- la utility matrix y penalty;
- la población de pair tokens observada.

No estiman variación universal entre generadores ni corrigen multiplicidad. El
reader debe etiquetarlos `CONDITIONAL_ON_NEW_DRAW_AND_FROZEN_W59_LAW`.

## 8. Patrones y salida ternaria

Para comparabilidad, las condiciones numéricas de Ola 59 se preservan.

### 8.1 Incompatibility / media

1. soporte autorizado `>=25` pair tokens;
2. accuracy vs hard: IC95 inferior `>=-0,01`;
3. compatibility vs hard: IC95 inferior `>=0`;
4. regret vs hard: media `<=-0,005`;
5. regret vs hard: IC95 superior `<0`;
6. worst regret vs hard: IC95 superior `<=+0,01`;
7. regret vs promedio de cinco controles: IC95 superior `<0`;
8. replay exacto.

### 8.2 Harm / cola

1. soporte autorizado `>=25` pair tokens;
2. accuracy vs hard: IC95 inferior `>=-0,01`;
3. compatibility vs hard: IC95 inferior `>=0`;
4. regret vs hard: IC95 superior `<=0`;
5. worst regret vs hard: media `<=-0,01`;
6. worst regret vs hard: IC95 superior `<0`;
7. worst regret vs promedio de cinco controles: IC95 superior `<0`;
8. replay exacto.

El soporte es una puerta de evaluabilidad anterior a la agregación. El registro
conserva el predicado `authorized_pair_tokens_at_least_25`, pero si es falso no
interpreta los intervalos de una política con cobertura insuficiente: el patrón
global queda `null / NOT_EVALUABLE`. Si el soporte pasa, la agregación es
ternaria:

- `true`: todas las condiciones evaluables son verdaderas;
- `false`: todas son evaluables y al menos una es falsa;
- `null / NOT_EVALUABLE`: la puerta de soporte falla o un control pierde
  integridad/evaluabilidad.

Una caída de soporte debajo de `25` no se repara y no se llama automáticamente
fracaso estadístico. Se informa como falta de cobertura transportada.

`scientific_decision` permanece siempre `null`; `decision_authority` es
`user`.

## 9. Controles y alcance inferencial

Los diez controles son estados aprendidos en Ola 59 contra targets de máximo
desplazamiento dentro de estratos `(policy_index, disagreement_count)`. Wave 60
no vuelve a permutar targets. Su identidad de diseño es la del entrenamiento
original, y su pregunta aquí es de transporte comparado: ¿la ley principal
generaliza mejor que reglas de igual capacidad entrenadas sobre targets
adversariales?

La respuesta se limita al transporte diferencial de esas pipelines completas.
Un resultado favorable no atribuye por sí mismo la diferencia al target ni a la
localización contextual, porque estados, thresholds y cobertura también pueden
mediar el contraste.

El análisis debe conservar por control:

- ID, seed, target, threshold y hashes del estado;
- proposal, authorization y action masks;
- soporte por filas y pair tokens;
- cuatro métricas y diferencias pareadas;
- promedio predeclarado de la familia completa de cinco.

No se descartan controles por soporte o signo. Si un estado, threshold o score
no valida, toda la familia correspondiente queda `NOT_EVALUABLE`.

## 10. Replay, manifest y fallo

Primaria y replay nacen del mismo escrow nuevo, pero sus paquetes se firman y
verifican individualmente. La comparación reutiliza la normalización tipada ya
auditada en Ola 59:

- artefactos científicos JSON/MD: byte exactos;
- NPZ científicos: keys, dtype, shape, valores y NaN exactos;
- secretos: sólo hashes opacos;
- estados portables: igualdad funcional y bindings físicos;
- artefactos operacionales: igualdad semántica allowlisted después de validar
  enlaces locales brutos.

Cada raíz tiene manifest closed-world. Archivos condicionales de failure y
replay se enumeran por estado. Un fallo posterior al acceso a truth no permite
parchear y reanudar con código distinto; sólo preservación post hoc o protocolo
y draw nuevos. Un fallo previo puede recuperarse únicamente bajo amendment
auditado y paquete firmado, sin redibujo ni acceso semántico adelantado.

### 10.1 Outputs y schemas congelados

Los JSON usan forma cerrada: ninguna clave extra queda tolerada.

| Fase | Outputs obligatorios | Schema / contenido cerrado |
|---|---|---|
| source law | `source_law_freeze.json`, `transport_law_manifest.json`, `transport_law_arrays.npz`, `frozen_policy_spec.json`, `feature_schema.json` | `wave60-source-law-v1`; commit y nueve hashes fuente; roster 13+3; 13 estados usados; 1.300 keys únicas; 13 thresholds; operadores; 17 features; hashes de todos los outputs |
| score/apply | `monitor_scores.npz`, `monitor_policy_arrays.npz`, `evaluation_index.npz`, `monitor_action_freeze.json`, `score_apply_receipt.json` | `wave60-score-apply-v1`; 13 score arrays raw; 1 proposal; 12 authorized; 14 actions; `primary`, `pair_token` y orden de evaluación; `score_mask`; `decision_mask`; hashes de inputs/outputs |
| evaluate | `bootstrap_indices.npz`, `analysis_arrays.npz`, `analysis.json`, `evaluation_freeze.json`, `evaluate_receipt.json` | `wave60-evaluate-v1`; 5.000 índices; métricas y deltas pareados; IC95; 16 condiciones; dos patrones ternarios; diagnósticos predeclarados; hashes de inputs/outputs |
| final | `REPORT.md`, `runtime.json`, `artifact_manifest.json` | `wave60-final-v1`; estado terminal, presupuesto observado, inventario closed-world, decisión null y autoridad user |

Los conteos de arrays son contractuales: las 14 actions corresponden a hard,
proposer-only, dos principales y diez controles. Pure posterior no forma parte
del aplicador transport-only ni del patrón; si se informa descriptivamente se
deriva sólo durante evaluación, bajo una key separada enumerada en la config y
sin alterar el conteo de acciones congeladas.

Las keysets JSON mínimas y exactas son:

```text
source_law_freeze.json = {
  schema_version, phase, source_commit, source_hashes, roster,
  feature_schema_sha256, transport_law_manifest_sha256,
  transport_law_arrays_sha256, frozen_policy_spec_sha256
}
transport_law_manifest.json = {
  schema_version, used_models, unused_models, model_states,
  array_keys, feature_names, source_bindings
}
frozen_policy_spec.json = {
  schema_version, proposer, main_policies, controls, references,
  thresholds, comparison_operators, score_mask, decision_mask
}
monitor_action_freeze.json = {
  schema_version, phase, source_law_freeze_sha256,
  inference_bundle_sha256, scores_sha256, policy_arrays_sha256,
  evaluation_index_sha256, score_apply_receipt_sha256
}
evaluation_freeze.json = {
  schema_version, phase, truth_bundle_sha256, action_freeze_sha256,
  policy_arrays_sha256, evaluation_index_sha256, utilities_sha256,
  bootstrap_sha256, analysis_arrays_sha256, analysis_sha256,
  evaluate_receipt_sha256
}
```

Los dos receipts comparten la keyset `{schema_version, phase, status, uid, gid,
capabilities, no_new_privs, inputs, outputs, opened_paths,
denied_path_probes, started_at, completed_at}`. `analysis.json` conserva
`{schema_version, status, estimand, population, policies, controls,
references, metrics, deltas, intervals, conditions, patterns, diagnostics,
scientific_decision, decision_authority, limitations}`. Los manifests de
estados y de artefactos contienen mapas ordenados; toda subestructura define en
el módulo un validador de claves exactas antes de que pueda firmarse o
promoverse.

El manifest clasifica todo path en exactamente una de estas clases:

```text
PRE_GENERATION_PUBLIC
BENCHMARK_PUBLIC
BENCHMARK_SEALED_SECRET
PREPARED_INFERENCE_SAFE
PREPARED_TRUTH_SECRET
SOURCE_LAW_FROZEN
SCORE_APPLY_SCIENTIFIC
EVALUATION_SCIENTIFIC
OPERATIONAL_JOURNAL
FINAL_PUBLIC
FAILURE_CONDITIONAL
REPLAY_COMPARISON_CONDITIONAL
```

Cada clase congela owner, group, modo, bytes y hash. Missing, extra, duplicado,
symlink, hardlink no autorizado o clase incorrecta invalidan el intento.

### 10.2 Estados terminales

Los únicos terminales operacionales son:

```text
INVALID_NEW_DRAW_IDENTITY
INVALID_PREPARATION
SOURCE_LAW_INVALID
SCORE_APPLY_FAILED_PRE_TRUTH
EVALUATION_FAILED_POST_TRUTH
COMPLETE
```

`NOT_EVALUABLE` es un valor científico dentro de una ejecución `COMPLETE`, no
un terminal operativo. Todo fallo conserva inventario y attestation conforme a
su fase. Sólo un fallo pre-truth puede entrar en un recovery auditado; uno
post-truth exige protocolo y draw nuevos. Primaria y replay usan la misma
máquina de estados y schemas, con diferencias operacionales allowlisted.

## 11. Implementación propuesta

La implementación significativa queda limitada a:

```text
src/geometria_proporcional/wave60_frozen_policy_transport.py
experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
experiments/geometria_proporcional/_wave60_phase_worker.py
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

El módulo nuevo importa y reutiliza el scorer portable, métricas y bootstrap de
Ola 59 cuando su semántica coincide. Define el aplicador transport-only estricto;
no copia árboles ni reimplementa métricas. El worker nuevo contiene sólo el
dispatch de las dos fases allowlisted. El preparador compartido sólo agrega una
rama tipada para schema/config/output de Wave 60 y mantiene byte-invariantes las
rutas Wave 56–59.

La config final se crea después de aceptar la implementación:

```text
experiments/geometria_proporcional/configs/
  wave60_frozen_policy_transport.json
```

Raíces canónicas:

```text
data/geometria_proporcional/
  wave60_frozen_policy_transport_v1/
  wave60_frozen_policy_transport_v1_replay/
```

## 12. Pruebas obligatorias antes del draw

### 12.1 Ley congelada

- hashes exactos de los nueve artefactos fuente enumerados;
- manifest con los trece modelos requeridos;
- tree keys completas, no compartidas y sin categorical splits;
- reproducción exacta de scores, proposals, authorizations y actions Wave 59;
- distinción exacta `score_mask=disagreement` y
  `decision_mask=primary AND disagreement`, incluidos disagreements no
  primarios con score finito pero acción hard;
- rechazo por un bit alterado en estado, threshold, calibration freeze, source
  manifest o arrays preservados;
- rechazo de modelo adicional usado por una política no enumerada;
- aplicador transport-only: 13 modelos exactos, 26 arrays Wave 59 equivalentes,
  y rechazo cerrado ante Ridge, Logistic, legacy o policy extra.

### 12.1.1 Independencia de la realización

- no identidad opaca contra las cinco raíces Ola 59 para escrow, tres key
  commitments, semantic root, commitments y bundles;
- rechazo antes de `PREPARED` si se copia cualquiera de esos elementos;
- rechazo de alias, symlink o hardlink por path resuelto y `(st_dev, st_ino)`;
- primaria/replay comparten exactamente el escrow Ola 60, pero no inodos;
- el checker no abre secrets ni truth para demostrar no identidad.

### 12.2 Prohibición de aprendizaje

- monkeypatch de `fit()` de Ridge, Logistic y HGB que falle si se invoca;
- monkeypatch de `quantile`, `quantile_higher` y selectores que falle si se
  invocan sobre el draw nuevo;
- ausencia de artefactos `fit` o `calibration` nuevos salvo receipts que declaren
  explícitamente `REUSED_FROZEN_W59` y sólo contengan hashes;
- thresholds de salida idénticos bit a bit a calibration freeze de Ola 59.

### 12.3 Frontera de acceso

- inference-safe lockbox separado de truth;
- acciones congeladas y hasheadas antes de autorizar truth;
- train/validation nuevos nunca abiertos por el runner;
- prueba negativa que haga fallar si el worker solicita cualquier path no
  allowlisted;
- UID/GID y modos de paquetes conforme a la atestación;
- dos sandboxes físicos distintos: `score_apply` no puede leer truth,
  train/validation; `evaluate` no puede leer design, scores, estados ni
  thresholds;
- receipts prueban UID/GID `65534`, capabilities vacías, `NoNewPrivs`, paths
  abiertos e inventarios exactos;
- alteración de action freeze, actions o evaluation index aborta antes de
  evaluar truth.

### 12.4 Estadística y controles

- soporte contado por pair token primario, no por filas;
- exactamente cinco controles por target y promedio sin selección;
- bootstrap compartido `5.000 × n_tokens` y orden lexicográfico;
- recomputación independiente de metrics, deltas, IC95 y patrones;
- casos sintéticos `true`, `false` y `NOT_EVALUABLE`;
- control de bordes estrictos `>`/`<` cuando score==threshold.

### 12.5 Replay y regresión

- primary/replay exactos bajo diferencias operacionales legítimas;
- schemas, keysets, conteos NPZ, clases de manifest y seis terminales exactos;
- rechazo de hash derivado desligado, firma inválida, array alterado, state no
  portable, extra o missing en manifest;
- suite completa Wave 56–60 con CUDA invisible y cuatro threads;
- preservación de hashes de las dos raíces Wave 59 y del intento anterior no
  adjudicable;
- ningún test crea o modifica las raíces canónicas Wave 60.

## 13. Cadena de autoridad

1. commit exclusivo del draft `a51b7fa`;
2. auditoría R457 `REVISE`, archivada con dictamen parseable;
3. commit exclusivo de este plan revisado;
4. reauditoría independiente del plan revisado;
5. implementación en los cinco paths autorizados;
6. auditoría independiente de implementación, con tests y hashes;
7. config final que liga plan, auditorías, implementación, fuentes y outputs;
8. auditoría independiente de config y autoridad como HEAD exacto;
9. preparación primaria y replay;
10. ejecución científica primaria y replay;
11. auditoría independiente de artefactos y resultados;
12. integración documental sin promoción ni decisión automática.

Cada paso verifica parent directo, paths exclusivos, hashes físicos, HEAD y
worktree limpio. Un informe `REVISE`, decisiones contradictorias o `PASS`
incidental en prosa no constituyen autoridad. La config no se ejecuta hasta que
la auditoría final sea HEAD y el parser canónico la acepte.

## 14. Presupuesto y recursos

El experimento usa CPU porque no entrena y la ejecución Wave 59 consumió menos
de cuatro minutos combinando primaria y replay. Presupuesto predeclarado:

- cuatro threads por proceso;
- tiempo combinado máximo `900 s`;
- RSS máximo `1,5 GiB` por proceso;
- `CUDA_VISIBLE_DEVICES=""`;
- sin Colab ni Mendieta.

Si la implementación real contradice esta estimación y una etapa GPU pasa a ser
necesaria o materialmente más eficiente, el trabajo se detiene antes de cargar
CUDA. Se preserva el estado y se informa al usuario por Telegram con objetivo,
duración y VRAM estimados. No se sustituye una espera de GPU por una corrida CPU
de horas.

## 15. Resultados posibles

- **Ambos patrones true:** evidencia sintética de transporte de la ley completa
  bajo este generador; no promoción automática.
- **Main vs hard favorable, main vs control no separado:** replica el problema
  de atribución fuera del draw y favorece rediseñar target/control.
- **Soporte transportado insuficiente:** la calibración no transporta cobertura;
  resultado `NOT_EVALUABLE`, útil para descartar thresholds como interfaz fija.
- **Main pierde frente a hard:** evidencia contra el transporte de esta ley y
  thresholds, no contra toda arquitectura proposer/guard.
- **Incompatibility y harm divergen:** preserva una bifurcación media/cola que
  requiere decisión del usuario; no autoriza selección post hoc.

En todos los casos, la salida registra observación, inferencia acotada y límites
por separado. No se declara techo, arquitectura promovida ni `GO/NO-GO`.
