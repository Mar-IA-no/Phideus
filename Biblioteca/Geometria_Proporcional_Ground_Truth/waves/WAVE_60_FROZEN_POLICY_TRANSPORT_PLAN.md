# Ola 60 — transporte prospectivo de políticas congeladas entre draws

> **Estado:** `REVISED-AFTER-R462 / PRE-IMPLEMENTATION / PRE-DRAW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-05
> **Antecedente:** `WAVE_59_FRESH_HGB_GUARD_BRACKET_CLOSED.md`
> **Auditoría del draft:** `R457 / REVISE / 3 HIGH + 2 MEDIUM`
> **Informe R457:** `../agent_reports/457_wave60_frozen_transport_plan_audit.md`
> **Reauditoría:** `R458 / REVISE / 2 HIGH + 2 MEDIUM`
> **Informe R458:** `../agent_reports/458_wave60_revised_plan_reaudit.md`
> **Tercera auditoría:** `R459 / REVISE / 2 HIGH + 1 MEDIUM`
> **Informe R459:** `../agent_reports/459_wave60_second_revision_audit.md`
> **Cuarta auditoría:** `R460 / REVISE / 1 HIGH + 1 MEDIUM`
> **Informe R460:** `../agent_reports/460_wave60_atomic_pair_plan_audit.md`
> **Quinta auditoría:** `R461 / REVISE / 2 HIGH + 1 MEDIUM`
> **Informe R461:** `../agent_reports/461_wave60_peer_abort_plan_reaudit.md`
> **Reauditoría focal:** `R462 / REVISE / 0 HIGH + 1 MEDIUM`
> **Informe R462:** `../agent_reports/462_wave60_terminal_lifecycle_focal_reaudit.md`
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
cerrados y promueve outputs; no calcula scores, acciones ni métricas. Existen
tres tipos de invocación del worker como UID/GID `65534`, capabilities vacías y
`NoNewPrivs=true`: `verify_source_law` se ejecuta una sola vez antes del draw;
`score_apply` y `evaluate` se ejecutan una vez por root.

1. `verify_source_law`, con allowlist física exacta:
   `source_law_request.json`, `wave59_fit_freeze.json`,
   `wave59_model_states_manifest.json`, `wave59_model_state_arrays.npz`,
   `wave59_calibration_freeze.json`, `wave59_monitor_inference_bundle.npz`,
   `wave59_monitor_scores.npz`, `wave59_monitor_policy_arrays.npz`,
   `wave59_monitor_action_freeze.json`, `wave59_artifact_manifest.json`,
   `wave59_config_snapshot.json` y `r454_audit.md`. Los aliases distinguen sin
   ambigüedad la config fuente de la futura config Ola 60. El worker verifica
   físicamente los nueve hashes de §3, reproduce 13 scores y las 26 arrays
   seleccionadas más hard, y emite la proyección transportable de trece estados.
   No recibe truth de Ola 59 ni ningún archivo del draw nuevo.
2. `score_apply`, con allowlist física exacta:
   `config.snapshot.json`, `source_bindings.json`,
   `transport_law_manifest.json`, `transport_law_arrays.npz`,
   `frozen_policy_spec.json`, `feature_schema.json` y
   `sealed_monitor_inference_bundle.npz`. No monta ni vuelve legibles truth,
   train o validation del draw nuevo.
3. `evaluate`, con allowlist física exacta:
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

Cada phase freeze hashea sólo inputs previos y outputs científicos que ya
existen; nunca hashea el receipt de su propia fase. El worker escribe primero
los outputs y su freeze, luego construye el receipt sobre ese inventario. Por
último, el coordinador firma una attestation externa que liga freeze, receipt,
journal, request/config y commit. La clave privada permanece fuera del sandbox; el
checker usa la clave pública ya congelada. Este orden es acíclico y se aplica a
las tres fases.

La autoridad `verify_source_law` vive una sola vez en:

```text
data/geometria_proporcional/
  wave60_frozen_policy_transport_source_law_v1/
```

`source_law_request.json` liga plan, commit de implementación y auditoría de
implementación. La autoridad pre-draw se audita por separado; recién después la
config final congela sus hashes. Primaria y replay copian los cinco outputs
científicos, receipt y attestation por bytes —nunca por hardlink— y validan el
binding contra la config. No vuelven a ejecutar ni reinterpretar la ley fuente.

El generador puede materializar train y validation por compatibilidad con la ley
existente, pero Wave 60 los clasifica como **no usados**. El runner científico
recibe únicamente:

- inference-safe lockbox: design, disagreement, primary, hard actions,
  posterior actions y pair tokens;
- truth lockbox sellado: targets, weights y estados necesarios para métricas.

Orden obligatorio, distinguiendo la autoridad pair-level de las roots:

```text
SOURCE_LAW_VERIFIED (única, pre-draw)
  -> CONFIG_AUTHORITY_FROZEN
  -> ROOTS_INITIALIZED
  -> PREPARED_PRIMARY_AND_REPLAY
  -> SOURCE_LAW_BOUND
  -> LOCKBOX_SCORES_FROZEN
  -> LOCKBOX_ACTIONS_FROZEN
  -> LOCKBOX_TRUTH_AUTHORIZED
  -> EVALUATED_IMMUTABLE
  -> AWAITING_REPLAY_FINALIZE
  -> PAIR_COMMIT_COMPLETE
```

Antes de `LOCKBOX_ACTIONS_FROZEN` no se puede abrir truth de lockbox. Train y
validation del draw nuevo no se abren en ninguna fase científica. Un acceso,
refit, cuantile calculation o selección sobre ellos invalida el intento.

`analysis.json` y los outputs de `evaluate` quedan inmutables en
`EVALUATED_IMMUTABLE`: contienen sólo las siete condiciones intradraw de cada
política y sus patrones core. `replay_exact` se conoce recién cuando ambas
raíces alcanzaron `AWAITING_REPLAY_FINALIZE`; nunca se inserta reescribiendo un
output previo.

Una vez superada la barrera `SCORE` en ambas roots, el coordinador autoriza truth
para ambas y lleva **cada evaluación a un terminal propio**, aunque la otra
falle primero. Un fallo no cancela al peer. Un worker interrumpido se reanuda con
el mismo commit, action freeze y truth bundle hasta producir
`EVALUATED_IMMUTABLE` o `EVALUATION_FAILED_POST_TRUTH`; no queda ninguna root en
`SCORE` cuando se publica un terminal pair-level post-truth.

Las dos roots no se mutan durante el cierre cruzado. El coordinador construye en
un staging nuevo el paquete pair-level completo y lo publica con un único
`rename` atómico a:

```text
data/geometria_proporcional/
  wave60_frozen_policy_transport_attempt_v1/pair/
```

Ese paquete consume por hash las dos evaluaciones inmutables y contiene replay,
análisis final, receipt, freeze, journal, attestation, report, runtime y manifest.
Un crash antes del `rename` deja ambas roots intactas y la root pair-level
ausente; puede reintentarse idempotentemente bajo el mismo commit y los mismos
inputs sin reabrir truth. Nunca existe una promoción parcial entre dos roots.

El contenedor `attempt_v1` también elimina la doble inicialización. Antes de
generar, el coordinador crea
`wave60_frozen_policy_transport_attempt_v1.initializing/` con subdirectorios
`primary/` y `replay/`, cada uno con su config y bindings iniciales; valida y
publica el contenedor entero mediante un solo `rename`. Un crash pre-rename deja
el namespace canónico ausente y un staging reanudable; post-rename ambas roots
existen necesariamente. `pair/` se crea después por otro rename atómico dentro
del contenedor ya publicado.

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

Cada root de ejecución tiene manifest o inventario de fallo closed-world; el
replay y la decisión final viven sólo en la root pair-level. Un fallo posterior
al acceso a truth no permite parchear ni reanudar el análisis con código
distinto. La única recuperación post-truth admitida es terminar el staging
pair-level idempotente con el mismo commit sobre evaluaciones ya inmutables, sin
reabrir truth. Todo otro caso exige protocolo y draw nuevos. Un fallo previo
puede recuperarse únicamente bajo amendment auditado y paquete firmado, sin
redibujo ni acceso semántico adelantado.

### 10.1 Outputs y schemas congelados

Los JSON usan forma cerrada: ninguna clave extra queda tolerada.

| Fase / scope | Outputs obligatorios | Schema / contenido cerrado |
|---|---|---|
| source law / única pre-draw | `source_law_freeze.json`, `transport_law_manifest.json`, `transport_law_arrays.npz`, `frozen_policy_spec.json`, `feature_schema.json`, `verify_source_law_receipt.json`, `source_law_attestation.json`, `journals/verify_source_law.json` | `wave60-source-law-v1`; plan/implementación/auditoría; nueve hashes físicos; reproducción 13 scores/26 arrays+hard; roster 13+3; 13 estados usados; 1.300 keys únicas; 13 thresholds; operadores; 17 features |
| source binding / por root | copia byte-exacta de los siete outputs no-journal de source law, `source_law_binding.json`, `journals/source_bind.json` | `wave60-source-binding-v1`; hashes de autoridad y config; rol de root; no hardlinks |
| score/apply / por root | `monitor_scores.npz`, `monitor_policy_arrays.npz`, `evaluation_index.npz`, `monitor_action_freeze.json`, `score_apply_receipt.json`, `score_apply_attestation.json`, `journals/score_apply.json` | `wave60-score-apply-v1`; 13 score arrays raw; 1 proposal; 12 authorized; 14 actions; `primary`, `pair_token` y orden de evaluación; `score_mask`; `decision_mask` |
| evaluate / por root | `bootstrap_indices.npz`, `analysis_arrays.npz`, `analysis.json`, `evaluation_freeze.json`, `evaluate_receipt.json`, `evaluation_attestation.json`, `journals/evaluate.json` | `wave60-evaluate-v1`; 5.000 índices; métricas y deltas pareados; IC95; 14 condiciones intradraw; dos patrones core ternarios; diagnósticos predeclarados |
| root seal / por root | `runtime.json`, `artifact_manifest.json` | `wave60-evaluated-root-v1`; estado `EVALUATED_IMMUTABLE`; inventario closed-world local; self-reference tipada |
| replay finalize / pair | `replay_comparison.json`, `final_analysis.json`, `replay_finalize_freeze.json`, `replay_finalize_receipt.json`, `journals/replay_finalize.json`, `replay_finalize_attestation.json` | `wave60-replay-finalize-v1`; comparación de dos roots inmutables; 16 condiciones finales; dos patrones finales ternarios; exactitud científica y normalización operacional |
| final / pair | `REPORT.md`, `runtime.json`, `artifact_manifest.json` | `wave60-pair-final-v1`; estado terminal, presupuesto observado, inventario closed-world pair-level, decisión null y autoridad user |

Los conteos de arrays son contractuales: las 14 actions corresponden a hard,
proposer-only, dos principales y diez controles. Pure posterior no forma parte
del aplicador transport-only ni del patrón; si se informa descriptivamente se
deriva sólo durante evaluación, bajo una key separada enumerada en la config y
sin alterar el conteo de acciones congeladas.

Cada receipt inventaría los outputs científicos y el freeze que lo preceden,
pero excluye su propio archivo y la attestation posterior. `replay_finalize` lo
ejecuta el coordinador sólo sobre hashes, manifests, freezes, attestations y
outputs científicos ya inmutables de ambas roots; no reabre truth ni estados de
modelo. Construye sus seis outputs, report y manifest dentro de un único staging
pair-level, los valida y promueve el directorio completo por un solo `rename`.

Las keysets JSON mínimas y exactas son:

```text
source_law_request.json = {
  schema_version, plan_commit, plan_sha256, implementation_commit,
  implementation_audit_commit, implementation_audit_sha256,
  source_paths, source_sha256, output_path, runtime_budget
}
source_law_freeze.json = {
  schema_version, phase, source_law_request_sha256, source_commit,
  implementation_commit, implementation_audit_sha256, source_hashes, roster,
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
source_law_binding.json = {
  schema_version, run_role, config_sha256, source_authority_path_sha256,
  source_law_freeze_sha256, source_law_attestation_sha256,
  copied_output_hashes, hardlink_checks
}
monitor_action_freeze.json = {
  schema_version, phase, source_law_freeze_sha256,
  inference_bundle_sha256, scores_sha256, policy_arrays_sha256,
  evaluation_index_sha256
}
evaluation_freeze.json = {
  schema_version, phase, truth_bundle_sha256, action_freeze_sha256,
  policy_arrays_sha256, evaluation_index_sha256, utilities_sha256,
  bootstrap_sha256, analysis_arrays_sha256, analysis_sha256
}
replay_comparison.json = {
  schema_version, status, primary_scientific_hashes, replay_scientific_hashes,
  exact_json_md, exact_npz, functional_states, secret_hashes,
  operational_semantic, mismatches
}
final_analysis.json = {
  schema_version, primary_analysis_sha256, replay_analysis_sha256,
  replay_comparison_sha256,
  conditions, patterns, scientific_decision, decision_authority, limitations
}
replay_finalize_freeze.json = {
  schema_version, phase, primary_evaluation_attestation_sha256,
  replay_evaluation_attestation_sha256, primary_root_manifest_sha256,
  replay_root_manifest_sha256, replay_comparison_sha256,
  final_analysis_sha256, pair_status_sha256
}
replay_finalize_receipt.json = {
  schema_version, phase, status, coordinator_uid, coordinator_gid,
  inputs, outputs_before_receipt, staging_path_sha256,
  publish_target_path_sha256, started_at, completed_at
}
pair_status.json = {
  schema_version, terminal, primary_terminal, replay_terminal,
  primary_terminal_binding_sha256, replay_terminal_binding_sha256,
  any_truth_accessed,
  recovery_allowed, created_at
}
```

Los tres receipts comparten la keyset `{schema_version, phase, status, uid, gid,
capabilities, no_new_privs, inputs, outputs, opened_paths,
denied_path_probes, started_at, completed_at}`. `analysis.json` conserva
`{schema_version, status, estimand, population, policies, controls,
references, metrics, deltas, intervals, core_conditions, core_patterns, diagnostics,
scientific_decision, decision_authority, limitations}`. Los manifests de
estados y de artefactos contienen mapas ordenados; toda subestructura define en
el módulo un validador de claves exactas antes de que pueda firmarse o
promoverse. El receipt del coordinador pair-level usa la keyset separada recién
declarada y tampoco se incluye en su propio inventario.

Cada attestation de las tres fases worker tiene la keyset exacta `{schema_version, phase,
payload, public_key_fingerprint, signature_base64}`. Su payload liga hashes de
request/config, commit, freeze, receipt y journal, más el scope pre-draw o el rol
primary/replay. `replay_finalize_attestation.json` usa la misma envoltura y su
payload liga freeze, receipt y journal pair-level, además de ambas attestations
de evaluación. Ningún archivo incluido contiene el hash de la attestation que
lo envuelve.

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
SELF_REFERENCE
```

Cada clase congela owner, group, modo, bytes y hash, excepto
`SELF_REFERENCE`, que contiene únicamente `artifact_manifest.json` en una root
de ejecución/pair-level o `source_authority_manifest.json` en la autoridad
pre-draw, con `hashes_omitted=true`; el hash físico lo liga respectivamente la
auditoría final o la auditoría source law y luego la config. Missing, extra,
duplicado, symlink, hardlink no autorizado o clase incorrecta invalidan el
intento.

### 10.2 Estados terminales y presencia condicional

Los archivos de fallo root-level usan formas cerradas:

```text
FAILURE.json = {
  schema_version="wave60-root-failure-v1", status, terminal, phase, run_role, truth_accessed,
  recovery_allowed, error_type, error_message_sha256, authority_binding_sha256,
  git_commit, peer_terminal, peer_terminal_binding_sha256, created_at
}
failure_inventory.json = {
  schema_version="wave60-root-failure-inventory-v1", terminal,
  last_complete_phase, files, classes, missing_expected,
  forbidden_present, created_at
}
failure_attestation.json = {
  schema_version="wave60-root-failure-attestation-v1", phase, payload,
  public_key_fingerprint, signature_base64
}
```

El paquete pair-level conserva los mismos tres nombres, pero schemas separados
sin semántica de peer:

```text
FAILURE.json = {
  schema_version="wave60-pair-failure-v1", status, terminal, phase,
  run_role="pair", truth_accessed, recovery_allowed, error_type,
  error_message_sha256, authority_binding_sha256, git_commit, created_at
}
failure_inventory.json = {
  schema_version="wave60-pair-failure-inventory-v1", terminal,
  root_terminal_bindings, files, classes, missing_expected,
  forbidden_present, created_at
}
failure_attestation.json = {
  schema_version="wave60-pair-failure-attestation-v1", phase, payload,
  public_key_fingerprint, signature_base64
}
```

Para ambos terminales pair abortados,
`authority_binding_sha256=sha256(pair_status.json)`. El payload de la
attestation liga `pair_status.json`, `FAILURE.json` y
`failure_inventory.json` después de validar los dos root terminal bindings.
El orden es único:

```text
root terminals
  -> pair_status.json
  -> FAILURE.json
  -> failure_inventory.json
  -> failure_attestation.json
  -> artifact_manifest.json
  -> rename atómico de pair/
```

Los bindings son direccionales y anulables sólo como sigue:

- en todo fallo propio (`INVALID_*`, `SOURCE_BINDING_FAILED_*`,
  `SCORE_APPLY_FAILED_*`, `EVALUATION_FAILED_*`), `peer_terminal=null` y
  `peer_terminal_binding_sha256=null`;
- únicamente `PEER_ABORTED_PRE_TRUTH` contiene el terminal de la root que falló
  primero y el SHA-256 físico de su `failure_attestation.json` ya publicada;
- el terminal binding de una root fallida es el SHA-256 de su
  `failure_attestation.json`; el de `EVALUATED_IMMUTABLE` es el SHA-256 de su
  `artifact_manifest.json`;
- `pair_status.json` se escribe al final y liga esos dos terminal bindings.

La attestation de la root fallida sólo liga su failure e inventory locales. La
peer abortada depende unidireccionalmente de ella; la fallida nunca se reescribe
para apuntar hacia atrás. No existe hash cruzado ni ciclo.

Un output parcial permanece sólo en staging y nunca ocupa su path canónico. El
inventario de fallo usa marcadores self-reference para sí mismo y para la
attestation futura; ésta firma después los hashes físicos de `FAILURE.json` y
`failure_inventory.json`, sin ciclo.

`COMMON` completo contiene exactamente `config.snapshot.json`,
`source_bindings.json`, `pre_generation_freeze.json`,
`generation_escrow.json`, `generation_receipt.json`,
`preparation_freeze.json`, `preparation_receipt.json`,
`preparation_attestation.json`, `journals/prepare.json`, los dos archivos
públicos `benchmark/{manifest.json,protocol_config.json}`, el inventario cerrado
del benchmark enumerado por ese manifest y los bundles preparados enumerados
por `preparation_freeze.json`. `SOURCE` es la fila source binding de §10.1;
`SCORE` y `EVAL` son sus filas por root completas. En
`INVALID_PREPARATION`, sólo preceden al triple de failure
`config.snapshot.json`, `source_bindings.json` y `failed_preparation/`, cuyo
inventario exacto está en `failure_inventory.json`.

El rename único del contenedor inicializa ambas roots con
`config.snapshot.json` y `source_bindings.json` antes de preparar cualquiera.
Así, incluso si la primera preparación falla, existe una peer identificable y
sellable; ninguna root queda implícitamente “no intentada”.

La autoridad pre-draw tiene sólo dos terminales:

| Terminal source authority | Obligatorio | Prohibido |
|---|---|---|
| `SOURCE_LAW_INVALID` | `source_law_request.json`, `journals/verify_source_law.json`, triple de failure | outputs científicos source law, config y cualquier draw Ola 60 |
| `SOURCE_LAW_VERIFIED` | fila source law completa de §10.1, `source_law_request.json`, `source_authority_manifest.json` | triple de failure y cualquier draw Ola 60 |

La coordinación impone una barrera: ambas roots deben completar `SOURCE` y
`SCORE` antes de autorizar truth en cualquiera. Por eso un fallo pre-truth no
puede coexistir con una evaluación iniciada en el par.

| Terminal por root | Obligatorio | Prohibido / futuro no alcanzado |
|---|---|---|
| `INVALID_NEW_DRAW_IDENTITY` | `COMMON`, triple de failure | `SOURCE`, `SCORE`, `EVAL`, root seal |
| `INVALID_PREPARATION` | subset exacto recién definido, triple de failure | `SOURCE`, `SCORE`, `EVAL`, root seal |
| `SOURCE_BINDING_FAILED_PRE_TRUTH` | `COMMON`, `journals/source_bind.json`, triple de failure | outputs `SOURCE`, `SCORE`, `EVAL`, root seal |
| `SCORE_APPLY_FAILED_PRE_TRUTH` | `COMMON`, `SOURCE`, `journals/score_apply.json`, triple de failure | outputs `SCORE`, `EVAL`, root seal |
| `PEER_ABORTED_PRE_TRUTH` | presencia exacta según `last_complete_phase`, triple de failure | toda fase posterior a `last_complete_phase`, root seal |
| `EVALUATION_FAILED_POST_TRUTH` | `COMMON`, `SOURCE`, `SCORE`, `journals/evaluate.json`, triple de failure | outputs `EVAL`, root seal |
| `EVALUATED_IMMUTABLE` | `COMMON`, `SOURCE`, `SCORE`, `EVAL`, root seal | triple de failure y cualquier output `REPLAY` o final pair-level |

Para `PEER_ABORTED_PRE_TRUTH`, `FAILURE.json.phase=peer_abort` y
`failure_inventory.json.last_complete_phase` fija la presencia permitida:

```text
INITIALIZED             -> config.snapshot + source_bindings + triple de failure
PREPARED                -> COMMON + triple de failure
SOURCE_LAW_BOUND        -> COMMON + SOURCE + triple de failure
LOCKBOX_ACTIONS_FROZEN  -> COMMON + SOURCE + SCORE + triple de failure
```

El coordinador sella así la peer sana —en cualquiera de los dos roles— después
de verificar el terminal y binding de la root fallida y antes de publicar
`PAIR_ABORTED_PRE_TRUTH`. Sólo agrega el triple de failure; no modifica outputs
ya promovidos. Ambas roots quedan entonces ligables por
`pair_status.json`. Después de este sellado local, el paquete pair-level no
reescribe ninguna root.

El resultado conjunto vive en un único paquete pair-level, publicado
atómicamente tanto para éxito como para aborto:

| Terminal pair-level | Obligatorio | Prohibido |
|---|---|---|
| `PAIR_ABORTED_PRE_TRUTH` | `pair_status.json`, triple pair-failure, `artifact_manifest.json` | outputs replay/final analysis; `truth_accessed=true` |
| `PAIR_ABORTED_POST_TRUTH` | `pair_status.json`, triple pair-failure, `artifact_manifest.json` | outputs replay/final analysis; `recovery_allowed=true` |
| `COMPLETE` | filas replay finalize y final pair de §10.1, `pair_status.json` | triple de failure |

`pair_status.json` tiene `{schema_version, terminal, primary_terminal,
replay_terminal, primary_terminal_binding_sha256,
replay_terminal_binding_sha256,
any_truth_accessed, recovery_allowed, created_at}`. Si cualquier root falla
antes de la barrera, el par termina `PAIR_ABORTED_PRE_TRUTH`; si una evaluación
falla después de abrir truth, termina `PAIR_ABORTED_POST_TRUTH`. Ninguna root se
reescribe para reflejar el estado del par.

El journal de una fase fallida usa `{schema_version, phase, status,
input_sha256, error_type, error_message_sha256, truth_accessed,
duration_seconds, max_rss_bytes}` y no finge outputs promovidos. Los seis outputs
de replay finalize son obligatorios juntos o ausentes juntos.

`NOT_EVALUABLE` es un valor científico dentro del terminal pair-level
`COMPLETE`, no un terminal operativo. Un staging pair-level interrumpido puede
reconstruirse idempotentemente sólo con el mismo commit y los mismos hashes de
roots `EVALUATED_IMMUTABLE`; si esos bindings cambian, termina
`PAIR_ABORTED_POST_TRUTH` y exige protocolo y draw nuevos.

### 10.3 Namespace de intentos y recuperación

Toda root que alcanzó un terminal —exitoso o fallido— es inmutable. Un
`PAIR_ABORTED_PRE_TRUTH` cierra definitivamente el contenedor de ese intento;
ningún recovery reemplaza `primary/`, `replay/` ni `pair/`. Una recuperación
autorizada requiere amendment, config y contenedor versionados:

```text
wave60_frozen_policy_transport_attempt_v{N}/
  primary/
  replay/
  pair/
```

con `N>=2`. La config nueva liga el pair failure previo, el amendment, los tres
paths nuevos y, si el fallo fue pre-truth, los hashes del mismo escrow/draw que
se preserva y copia sin hardlinks. Si conservar ese draw no puede probarse, el
recovery no se ejecuta; un draw distinto constituye otro protocolo, no la misma
recuperación. Los abortos post-truth nunca reutilizan draw.

`created_at`, `started_at`, `completed_at` y duraciones son campos operacionales
normalizables, no igualdad científica. Dentro de un staging pair-level, un
archivo ya escrito se valida y conserva; sólo se completan outputs faltantes.
Si el staging aún no produjo receipt, el intento reanudado puede registrar sus
timestamps operacionales reales. Una vez publicado el único `rename`, ningún
timestamp ni receipt se reescribe.

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
dispatch de las tres fases allowlisted. El preparador compartido sólo agrega una
rama tipada para schema/config/output de Wave 60 y mantiene byte-invariantes las
rutas Wave 56–59.

Después de aceptar la implementación se crea y ejecuta la autoridad source law
pre-draw; una auditoría independiente la acepta antes de crear la config final.
La config liga luego sus hashes exactos:

```text
experiments/geometria_proporcional/configs/
  wave60_frozen_policy_transport.json
```

Raíces canónicas:

```text
data/geometria_proporcional/
  wave60_frozen_policy_transport_source_law_v1/
  wave60_frozen_policy_transport_attempt_v1/
    primary/
    replay/
    pair/
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
  y rechazo cerrado ante Ridge, Logistic, legacy o policy extra;
- `verify_source_law` produce freeze, receipt y attestation durables; cada uno
  liga request, commit de implementación, auditoría, nueve fuentes, reproducción
  exacta y proyección de 13 estados sin acceder a truth;
- aliases físicos inequívocos para `wave59_artifact_manifest.json`,
  `wave59_config_snapshot.json` y `r454_audit.md`, con rechazo si cualquiera de
  los nueve hashes no coincide;
- ningún path, key, escrow o bundle Ola 60 existe al completar la autoridad
  source law.

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
- tres sandboxes físicos distintos: `verify_source_law` no puede leer truth
  fuente ni el draw nuevo; `score_apply` no puede leer truth o train/validation;
  `evaluate` no puede leer design, scores, estados ni thresholds;
- receipts prueban UID/GID `65534`, capabilities vacías, `NoNewPrivs`, paths
  abiertos e inventarios exactos;
- alteración de action freeze, actions o evaluation index aborta antes de
  evaluar truth.
- inicialización: crash antes/después del rename del contenedor nunca deja una
  sola root canónica;
- source binding alterado o hardlinked produce
  `SOURCE_BINDING_FAILED_PRE_TRUTH` y sella correctamente al peer.

### 12.4 Estadística y controles

- soporte contado por pair token primario, no por filas;
- exactamente cinco controles por target y promedio sin selección;
- bootstrap compartido `5.000 × n_tokens` y orden lexicográfico;
- recomputación independiente de metrics, deltas, IC95 y patrones;
- casos sintéticos `true`, `false` y `NOT_EVALUABLE`;
- control de bordes estrictos `>`/`<` cuando score==threshold.

### 12.5 Replay y regresión

- primary/replay exactos bajo diferencias operacionales legítimas;
- `analysis.json` permanece byte-inmutable antes y después del replay;
- `replay_finalize` es la única fase que agrega `replay_exact`, produce seis
  outputs acíclicos y liga ambas attestations de evaluación;
- rechazo si intenta finalizar con una sola raíz, reescribir un output previo o
  ligar una evaluación distinta;
- receipts no se hashean a sí mismos ni son hasheados por su freeze; la
  attestation externa liga ambos y falla ante cualquier sustitución;
- staging pair-level: crash antes del rename deja target ausente; reintento bajo
  mismo commit/inputs produce el mismo paquete y un único rename;
- barrera pre-truth: fallo de cualquiera de las roots impide que ambas evalúen;
- fallos asimétricos primary/replay en preparación y score sellan la peer como
  `PEER_ABORTED_PRE_TRUTH` con la presencia exacta de su última fase;
- terminales `PAIR_ABORTED_PRE_TRUTH` y `PAIR_ABORTED_POST_TRUTH` con manifests
  closed-world; ambas roots quedan selladas antes de publicar el paquete pair;
- un aborto v1 rechaza toda reutilización de su contenedor; recovery pre-truth
  sólo bajo amendment/config `v{N>=2}`, contenedor nuevo y mismo escrow/draw ligado;
- timestamps y duraciones se comparan sólo bajo normalización operacional; los
  outputs científicos permanecen exactos;
- post-truth: fallo antes de iniciar el peer, con peer en curso o después de que
  el peer complete siempre lleva ambas roots a
  `EVALUATED_IMMUTABLE|EVALUATION_FAILED_POST_TRUTH` antes del pair abort;
- bindings unidireccionales: fallo propio con peer fields null, peer abort con
  hash de la failure attestation previa y pair status con ambos bindings;
- schemas, keysets, conteos NPZ, clases y matrices terminales exactas por scope;
- rechazo de hash derivado desligado, firma inválida, array alterado, state no
  portable, extra o missing en manifest;
- suite completa Wave 56–60 con CUDA invisible y cuatro threads;
- preservación de hashes de las dos raíces Wave 59 y del intento anterior no
  adjudicable;
- ningún test crea o modifica el source authority ni el contenedor canónico
  Wave 60.

## 13. Cadena de autoridad

1. commit exclusivo del draft `a51b7fa`;
2. auditoría R457 `REVISE`, archivada en `4189c89`;
3. primera revisión `702776d`;
4. reauditoría R458 `REVISE`, archivada en `1b24425`;
5. segunda revisión `eb41df2`;
6. auditoría R459 `REVISE`, archivada en `aecc049`;
7. tercera revisión `8a428d1`;
8. auditoría R460 `REVISE`, archivada en `fbd83fc`;
9. cuarta revisión `221f483`;
10. auditoría R461 `REVISE`, archivada en `9682e44`;
11. quinta revisión `efbd785`;
12. reauditoría focal R462 `REVISE`, archivada con limitación declarada;
13. commit exclusivo de esta sexta revisión;
14. reauditoría independiente/focal del plan vigente;
15. implementación en los cinco paths autorizados;
16. auditoría independiente de implementación, con tests y hashes;
17. request y ejecución de la autoridad source law pre-draw;
18. auditoría independiente de la autoridad source law;
19. config final que liga plan, auditorías, implementación, source law y outputs;
20. auditoría independiente de config y autoridad como HEAD exacto;
21. preparación primary/replay con barrera pre-truth;
22. ejecución científica de ambas roots a terminal;
23. publicación atómica del paquete pair-level;
24. auditoría independiente de artefactos y resultados;
25. integración documental sin promoción ni decisión automática.

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
