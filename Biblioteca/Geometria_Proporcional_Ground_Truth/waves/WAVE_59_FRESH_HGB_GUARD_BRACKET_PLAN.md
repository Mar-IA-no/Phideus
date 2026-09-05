# Ola 59 — plan prospectivo para incompatibilidad y daño de cola

> **Estado:** `FINAL-AUDITED-R420 / FRESH-DRAW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-04
> **Origen de diseño:** cierre y auditoría de resultados de Ola 58

## Pregunta

La Ola 58 localizó, sobre un draw ya abierto, dos políticas HGB/HGB que no deben
fundirse en una única noción de «riesgo». El guard de incompatibilidad posterior
produjo el menor regret medio bajo el orden total congelado. El guard de harm
produjo una mejora de cola mayor. La Ola 59 preguntará si esas dos políticas,
con cuantiles ya fijados y la misma máscara de propuestas, transportan su señal
a una realización fresca de la misma ley.

La pregunta no es qué ID del roster abierto vuelve a ganar. Tampoco se volverá a
buscar entre JOINT y SEQUENTIAL: esos nombres realizaron la misma política en
Ola 58. El experimento separará una comparación de políticas de una atribución
de componentes y conservará `scientific_decision=null`. La promoción de una
arquitectura y cualquier `GO/NO-GO` pertenecen a Mariano.

## Frontera prospectiva

Antes de crear el triplete de secretos se congelarán plan, auditoría aceptada,
config, implementación, tests, hashes y commits. Después de ese freeze:

- no se leerán otra vez los outcomes abiertos de Ola 58 para escoger modelos,
  targets, cuantiles, thresholds, márgenes o brazos;
- no se redibujará una realización por un resultado desfavorable;
- FIT sólo verá train; CALIBRATE-SCORES verá una vista inference-safe de
  validation sin truth ni utilidad; VALIDATE abrirá truth de validation recién
  después de congelar modelos, scores, cuantiles, thresholds, máscaras y
  acciones; ADJUDICATE abrirá primero sólo la vista inference-safe del monitor
  y abrirá su truth una vez, después de congelar las acciones monitor;
- ninguna métrica de validation elegirá una celda: sólo se materializarán los
  cuantiles fijos sobre las distribuciones de scores;
- un mínimo fallido produce `NOT_EVALUABLE` o cierre premonitor según la fase,
  nunca tuning o redraw.

Los artefactos Wave 58 son autoridad de diseño abierta. No son baseline
prospectivo ni se mezclarán con el nuevo bootstrap.

## Objeto, población y acción

Se reutilizan sin modificación semántica:

- benchmark relacional `wave49-relational-benchmark-v2`;
- encoder, normalizador, ensemble de seeds `{17,29,43}` y posterior conjunto
  congelados en Olas 50–54;
- catálogo de 24 políticas y utilidad ordinal de Ola 52;
- hard-set con `tau=0.5`, acción bayesiana posterior y penalidad incompatible
  `1.25`;
- población primaria `NEAR_RIVAL and cardinality >= 2`;
- diecisiete features inference-safe de Olas 56–58;
- ajuste fila-wise sólo sobre `primary ∩ disagreement`, con peso `1/d_t`;
- métricas token-wise de accuracy, compatibilidad, regret y worst regret.

Los targets privados quedan definidos exactamente como en Ola 58:

```text
gain(t,p) = regret_hard(t,p) - regret_posterior(t,p)
harm(t,p) = 1[gain(t,p) < -1e-12]
posterior_incompatibility(t,p) = 1[target_t[action_posterior(t,p)] = 0]
```

Truth, gain, harm, incompatibilidad, oracle, regrets y métricas nunca son
inputs de inferencia. FIT puede leerlos exclusivamente en train para construir
los targets supervisados; VALIDATE y ADJUDICATE pueden leerlos exclusivamente
después de que las acciones de su split ya sean inmutables.

## Modelos congelables

Se conservarán los contratos exactos de scikit-learn `1.8.0` auditados en Ola
58:

- proposer Ridge, como control lineal;
- proposer HistGradientBoostingRegressor, seed `5801`;
- guard LogisticRegression, como control lineal;
- guard HistGradientBoostingClassifier de harm, seed `5802`;
- guard HistGradientBoostingClassifier de incompatibilidad, seed `5803`.

Todos reciben las mismas features y pesos. Los estados completos y scores
`float64` por split se preservarán. Durante FIT→CALIBRATE-SCORES, coeficientes,
scalers y nodes HGB son estado ejecutable autenticado, no mera metadata. Una
vez materializadas, las scores preservadas son la autoridad de reanálisis. Una
sola clase, no finitud, error de fit o split categórico inesperado deja el
modelo `NOT_EVALUABLE`; no habilita cambiar hiperparámetros.

## Cuantiles fijados por el diagnóstico abierto

La Ola 59 no hará búsqueda de threshold. Usará `method="higher"`, desigualdades
estrictas y estos cuantiles:

| Componente | Cuantil fijo | Procedencia abierta |
|---|---:|---|
| proposer HGB o Ridge | `0.8` | propuesta común de las dos políticas a contrastar |
| guard harm HGB o Logistic | `0.7` | política HGB/HGB-harm secuencial de Ola 58 |
| guard incompatibility HGB o Logistic | `0.9` | política HGB/HGB-incompatibility nominada |
| proposer legacy Ridge | `0.8` | Ola 57 |
| guard legacy Logistic-harm | `0.4` | Ola 57 |

CALIBRATE-SCORES calcula cada threshold como el cuantil fijo de todas las filas
primarias en desacuerdo de la vista inference-safe de validation para el
proposer correspondiente. Cada guard calcula su threshold sobre todas las
filas propuestas por ese proposer, antes de aplicar cualquier guard. Dentro de
una familia de proposer, harm e incompatibility reciben exactamente la misma
máscara de propuestas. La fase no recibe `target`, `gain`, `harm`,
`posterior_incompatibility`, oracle, regrets, métricas ni una tabla de utilidad
desde la que esos campos puedan reconstruirse.

```text
proposal = disagreement AND proposer_score > tau_proposer
authorized_target = proposal AND guard_target_score < tau_target
action = posterior_action if authorized_target else hard_action
```

Un score igual al threshold no propone o no autoriza. `hard_only` es identidad
byte-exacta, pero no participa en una búsqueda porque no hay grilla.

## Roster reducido y atribución

### Políticas principales

1. `HGB-PROPOSER-HGB-INCOMPATIBILITY`: candidata de regret medio;
2. `HGB-PROPOSER-HGB-HARM`: candidata de daño/cola.

Comparten modelo proposer, cuantil `0.8` y máscara de propuestas. Difieren sólo
el target, el modelo guard correspondiente y su cuantil precongelado.

### Factorial de atribución

Se ejecutarán las dieciséis combinaciones
`proposer {Ridge,HGB} × guard {Logistic,HGB} × target {harm,incompatibility}
× q_guard {.7,.9}`. Este cruce permite comparar los dos targets a un cuantil
común, en vez de confundir target con presupuesto de autorización. Las dos
políticas principales ya son celdas de esta matriz: HGB/HGB-harm en `q=.7` y
HGB/HGB-incompatibility en `q=.9`; no se materializarán otra vez bajo otro ID.

La matriz no identifica «no linealidad» en abstracto. Ridge/Logistic con scaler
y HGB con features crudas son contratos completos diferentes. Tampoco prueba
selector alguno: sólo se ejecuta la regla secuencial congelada. Las catorce
celdas que no son principales son diagnósticas y ninguna puede convertirse en
candidata post-hoc.

Los contrastes factoriales quedan preenumerados y siempre se calculan
token-wise antes de agregar:

1. ocho efectos condicionales de target,
   `policy(incompatibility,q)-policy(harm,q)`, uno por
   `(proposer, guard, q)`;
2. ocho efectos condicionales de cuantil,
   `policy(target,.9)-policy(target,.7)`, uno por
   `(proposer, guard, target)`;
3. ocho efectos del contrato de proposer, `HGB-Ridge`, uno por
   `(guard, target, q)`;
4. ocho efectos del contrato de guard, `HGB-Logistic`, uno por
   `(proposer, target, q)`;
5. cuatro interacciones target×cuantil,
   `[(incompatibility,.9)-(harm,.9)]-
   [(incompatibility,.7)-(harm,.7)]`, una por `(proposer, guard)`.

No habrá promedios marginales no predeclarados ni selección entre estos
contrastes.

### Referencias y controles adicionales

- `HARD-SET`;
- `PURE-POSTERIOR`, sólo como referencia diagnóstica;
- `RIDGE-PROPOSER-ONLY` y `HGB-PROPOSER-ONLY`;
- `LEGACY-W57`, Ridge `q=.8` más Logistic-harm `q=.4`;
- `ORACLE-POSITIVE-GAIN`, no seleccionable y materializado sólo por VALIDATE o
  MONITOR-EVALUATE después del freeze de acciones no-oracle;
- cinco controles HGB de desplazamiento máximo para harm y cinco para
  incompatibilidad, cada familia aplicada sobre la misma propuesta HGB y el
  mismo cuantil guard de su política principal.

No se añadirá un guard combinado retrospectivamente. Si harm e incompatibility
transportan señales complementarias, su conjunción será otra hipótesis y otro
freeze.

## Controles de desplazamiento máximo

La etiqueta de incompatibilidad es escasa: en el FIT abierto de Ola 57 su
prevalencia ponderada fue `0.02374` y el máximo Hamming ponderado factible bajo
la estratificación histórica fue sólo `0.03424`. Exigirle el `0.25` usado para
harm sería matemáticamente imposible y confundiría rareza con invalidez.

Cada control conservará exactamente la prevalencia dentro de
`(policy_index, disagreement_count)` y alcanzará el máximo Hamming ponderado
factible en cada estrato. Para una etiqueta binaria con `n0` ceros y `n1` unos,
se cambian `2*min(n0,n1)` posiciones: todas las posiciones de la clase
minoritaria y el subconjunto de igual cardinalidad y mayor peso de la clase
mayoritaria. Empates de peso se resuelven con una permutación PCG64 sembrada;
la asignación final es una biyección explícita entre índices fuente y destino.

Como los pesos `1/d_t` son constantes dentro de cada estrato, las cinco seeds
pueden generar receptores distintos sin sacrificar el máximo. Se usarán seeds
PCG64 `59031–59035` para el mapping de harm y `59041–59045` para el de
incompatibilidad. Los diez guards conservarán el mismo `random_state=5802` o
`5803` que el guard verdadero de su target; la seed del mapping no altera el
fit HGB. Cada réplica debe preservar mapping, target, máximos teóricos
global/ponderado, valores alcanzados, cociente alcanzado/máximo, fracción de
positivos desplazados, diagnóstico por estrato y hashes.

Estos controles no son permutaciones nulas exchangeable: fuerzan la máxima
destrucción de correspondencia compatible con la estratificación y la
prevalencia. Sus contrastes son stress-tests adversariales condicionados en
los cinco mappings, no estimaciones de un null, pruebas causales ni votos sobre
la validez del target.

Condiciones de evaluabilidad del control:

- cociente Hamming ponderado alcanzado/máximo `=1` a tolerancia de máquina;
- `mapping_permutable_fraction >=0.80`;
- máximo ponderado factible `>=0.25` para harm y `>=0.02` para
  incompatibilidad;
- cinco `mapping_sha256` distintos, cinco `target_sha256` distintos y cinco
  fits completos y finitos por familia.

Con `A = primary AND disagreement`, una entrada activa es el par fila-política
que pertenece a `A`. Las fórmulas son:

```text
mapping_permutable_fraction =
  count(active entries in strata with n >= 2) / count(active entries)
label_swappable_fraction =
  count(active entries in strata with n0 > 0 and n1 > 0) / count(active entries)
hamming_global = count(active entries whose label changes) / count(active entries)
hamming_weighted = sum_active(weight * label_changed) / sum_active(weight)
attained_over_max = hamming_weighted_attained / hamming_weighted_max
positive_displacement_fraction = changed_positive_entries / positive_active_entries
```

Todos los denominadores, conteos y sumas usan sólo FIT, `weight=1/d_t` y
`float64`; un denominador cero produce `NOT_EVALUABLE`, nunca `NaN` silencioso.
La tolerancia de `attained_over_max=1` es `rtol=0`, `atol=1e-12`.

El emparejamiento fuente→destino se construye en orden lexicográfico de
`(policy_index, disagreement_count, pair_token, row_index)`: PCG64 permuta sólo
los empates de receptoras de igual peso y luego empareja fuentes minoritarias y
receptoras por ese orden permutado. El mapping y el target resultante deben ser
byte-reproducibles a partir de la seed.

Si el máximo teórico del draw no alcanza el mínimo, no hay cinco mappings y
cinco targets efectivos distintos o un fit falla, la familia correspondiente
queda `NOT_EVALUABLE`; el draw no se reemplaza. El promedio de control se
calcula token-wise sobre las cinco réplicas evaluables antes del bootstrap:

```text
control_metric_t = (1/5) * sum_s metric_t(action_control_s)
```

No se promedian scores, probabilidades ni acciones antes de calcular la
métrica de cada réplica.

## Separación física

El pipeline tendrá cinco fronteras materiales:

```text
PREPARE -> FIT -> CALIBRATE-SCORES -> VALIDATE -> ADJUDICATE
```

- PREPARE genera y sella un draw nuevo, con tres splits disjuntos por
  `pair_token`. Dibuja una sola vez tres secretos de 32 bytes —identidad,
  compromiso semántico y generación—, publica sus commitments y conserva el
  triplete en escrow durable;
- FIT recibe sólo el bundle train read-only con su truth privada, ajusta los
  modelos verdaderos y los diez controles, y produce estados autenticados,
  scores train, mappings y `fit_freeze.json`;
- CALIBRATE-SCORES valida el freeze y recibe una vista validation read-only
  inference-safe. Produce scores, thresholds, propuestas, autorizaciones,
  acciones, soportes y `calibration_freeze.json`. Su parser usa una allowlist
  positiva y rechaza por nombre y contenido toda truth, target, gain, utilidad,
  regret, oracle o métrica;
- VALIDATE verifica el freeze de calibración y sus hashes antes de materializar
  truth validation. Calcula métricas diagnósticas sin capacidad de cambiar
  estados, scores, thresholds, máscaras o acciones, y sella
  `validation_freeze.json`;
- ADJUDICATE valida los tres freezes y se divide en dos subworkers físicos. El
  primero, `MONITOR-APPLY`, recibe sólo la vista inference-safe del monitor,
  estados FIT y thresholds de calibración; calcula scores, propuestas,
  autorizaciones, soportes y acciones, publica `monitor_action_freeze.json` y
  termina. El coordinador valida y promueve atómicamente ese freeze. Sólo
  entonces crea un sandbox nuevo para `MONITOR-EVALUATE`, que recibe truth,
  utilidad y acciones read-only, pero no estados de modelo ni una interfaz para
  recalibrar o reescribir acciones. Ese segundo subworker produce métricas y
  patrones sin reselección.

`monitor_action_freeze.json` vincula por hash config, source bindings,
preparation, FIT, calibration, validation, estado portable, scores monitor,
propuestas, autorizaciones, soportes y acciones. Ninguno de esos campos puede
regenerarse después de materializar truth monitor.

Workers analíticos corren como UID/GID `65534`, sin paths de splits futuros,
con archivos `0444`, directorios `0555`, capabilities nulas y configs por fase
con allowlists positivas y negativas. El coordinador verifica hashes antes y
después, publica cada fase atómicamente y conserva un intento fallido sin usarlo
como fuente para un redraw.

Los tests negativos deben demostrar que CALIBRATE-SCORES y MONITOR-APPLY no
pueden resolver ni enumerar archivos truth, y que VALIDATE y MONITOR-EVALUATE
no reciben estados ejecutables ni pueden modificar thresholds, máscaras o
acciones congeladas. La finalización del primer subworker y la promoción del
freeze se verifican antes de crear el segundo sandbox.

### Contrato de implementación y preparación

La Ola 59 no presupone que el preparador Wave 56/57 ya admita este protocolo.
Se extenderá de manera tipada el preparador compartido
`experiments/geometria_proporcional/prepare_wave56_fresh.py` con dispatch
explícito para `wave59-fresh-hgb-guard-bracket-v1`, preservando byte-exactos los
contratos Wave 56/57. Por compatibilidad con el generador, los roles físicos
siguen siendo:

| Split lógico | Split físico | Rol fuente |
|---|---|---|
| train | `train` | `gate_fit` |
| validation | `val` | `gate_select` |
| monitor | `lockbox` | `sealed_monitor` |

Esos nombres no autorizan reutilizar el SELECT outcome-aware previo. Los nuevos
wrappers producen vistas con schema Wave 59 y el coordinador registra estados
`PREPARED`, `FIT_COMPLETE`, `CALIBRATION_FROZEN`, `VALIDATION_OPENED`,
`VALIDATION_COMPLETE`, `MONITOR_ACTIONS_FROZEN`, `MONITOR_OPENED` y
`COMPLETE`. Cada transición escribe journal, hashes de inputs/outputs y el
máximo nivel de truth materializado. `NOT_EVALUABLE` es terminal científico sin
redraw. `IMPLEMENTATION_ERROR` no habilita por sí mismo una transición.

La recuperación se rige por esta matriz cerrada:

- un crash operativo con código/config/hashes idénticos puede reanudarse desde
  el último journal y freeze publicados;
- un delta de código/config sólo puede autorizarse antes de todo acceso
  semántico, mediante amendment pre-oráculo y auditoría independiente que
  demuestren que corrige una frontera sin cambiar el contrato científico y que
  reutiliza el mismo escrow;
- después de abrir truth train, ningún delta puede cambiar modelos, targets,
  features, cuantiles, métricas, mínimos, estimandos ni código científico para
  ese draw;
- después de abrir truth validation o monitor, cualquier delta vuelve el
  intento inválido para adjudicación prospectiva: se preserva íntegro y sólo un
  nuevo protocolo con nuevo draw puede continuar.

No existe transición desde un error a una fase posterior salvo reanudación
hash-idéntica. `NOT_EVALUABLE`, `INVALID-PROSPECTIVE-ATTEMPT` y `COMPLETE` son
terminales distintos y no se intercambian.

Inventario canónico de código y configuración:

- `experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket.json`;
- `src/geometria_proporcional/wave59_hgb_guard_bracket.py`;
- `experiments/geometria_proporcional/_wave59_phase_worker.py`;
- `experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py`;
- extensión tipada del preparador compartido;
- `tests/test_wave59_prospective.py` y regresiones físicas Wave 56/57.

La config se valida completa antes del escrow. Fija schema, roles, fases,
allowlists, archivos runtime, modelos, seeds, cuantiles, roster, mínimos,
bootstrap, shards, artefactos y hashes. Los receipts y namespaces nuevos usan
`wave59`; no se renombran receipts históricos.

## Mínimos predeclarados

Para cada split se define primero
`T_primary = sorted({t : primary[t] is true})`. La población complementaria se
excluye por diseño, no por missingness. Cada token de `T_primary` debe aportar
exactamente las 24 filas de política, con índices únicos y schema válido; una
fila faltante, duplicada o extra es un error de integridad terminal.

La config copiará estos mínimos exactos, sin una referencia mutable a Wave 57:

| Split/fase | Mínimos |
|---|---|
| FIT/train | 100 pair tokens de `T_primary`; 400 filas y 120 tokens en desacuerdo |
| FIT/harm | 80 tokens con alguna positiva; 50 con alguna no positiva |
| FIT/incompatibility | 20 filas positivas; 8 tokens con alguna positiva |
| CALIBRATE-SCORES/validation | 80 pair tokens de `T_primary`; 300 filas y 120 tokens en desacuerdo; 40 tokens propuestos; 25 autorizados por cada política principal |
| cada shard validation | 40 pair tokens de `T_primary`; 120 filas y 50 tokens en desacuerdo; 20 tokens propuestos; 12 autorizados por cada política principal |
| monitor | 100 pair tokens de `T_primary`; 300 filas y 120 tokens en desacuerdo |

Si FIT o CALIBRATE-SCORES fallan un mínimo global, el protocolo se cierra antes
de abrir el split siguiente. El monitor no tiene un mínimo de autorizaciones.
Para cada policy se preservan ambos soportes:

```text
authorized_rows = count_{t,p}(primary[t] AND authorized[t,p])
authorized_pair_tokens = count_t(primary[t] AND any_p authorized[t,p])
```

Sus rangos son `0..24*len(T_primary)` y `0..len(T_primary)`. Cualquier valor,
incluido cero, permite computar la policy; sólo la condición descriptiva
`authorized_pair_tokens >=25` pasa o falla. Con cero, la policy es idéntica a
hard y sus deltas contra hard son exactamente cero; nunca se recalibra.

## Estimandos y bootstrap

La unidad de inferencia es `pair_token`; todas sus policies y filas permanecen
juntas. Métricas, mínimos y bootstrap usan exclusivamente `T_primary`. Primero
se calcula cada métrica por token sobre sus 24 filas —no sólo sobre filas
autorizadas— y después se promedian tokens. `worst_regret_t` es el máximo regret
entre esas filas después de aplicar la acción del brazo. La integridad de las
24 filas se comprueba antes del análisis.

Se usarán `5000` réplicas PCG64, seed `5907`, orden lexicográfico y percentiles
exactos `[2.5,97.5]`. Los mismos índices se aplican a hard, todos los brazos,
proposer-only, head-to-head y cada promedio de controles dentro del split. Los
arrays se comparan con `equal_nan=True`, aunque cualquier `NaN` en una métrica
definida convierte el contraste en `NOT_EVALUABLE`. Los IC son descriptivos,
sin corrección de multiplicidad, y condicionan en este draw, los fits, la
calibración y los cinco mappings observados; no incorporan variación entre
draws, refits o recalibraciones. Validation y monitor se reportan por separado;
sólo monitor alimenta los patrones prospectivos.

Contrastes primarios:

1. HGB/HGB-incompatibility menos hard en regret, compatibilidad, accuracy y
   worst regret;
2. HGB/HGB-harm menos hard en worst regret, regret, compatibilidad y accuracy;
3. incompatibility menos harm en las cuatro métricas, sin convertirlas en un
   escalar ni declarar un ganador total;
4. cada política principal menos el promedio de sus cinco controles de
   desplazamiento máximo, usando regret
   para incompatibility y worst regret para harm;
5. cada principal menos `HGB-PROPOSER-ONLY`, para atribuir autorización;
6. los 36 contrastes factoriales ya enumerados y dos shards como diagnósticos de mecanismo y
   estabilidad, no como búsquedas alternativas.

Toda diferencia sigue el orden `policy-reference`; un valor negativo favorece
a policy en regret y worst regret, mientras que un valor positivo la favorece
en accuracy y compatibilidad. No se invierte el signo por conveniencia de
presentación.

### Shards de estabilidad

Los dos shards se asignan por el bit menos significativo de
`SHA256(pair_token || "wave59-bracket-shard")`. Sólo particionan validation. En
cada shard se recalculan los cuantiles fijos `.8`, `.7` y `.9` sobre las scores
del shard, por lo que diagnostican estabilidad del procedimiento de calibración,
no de una policy con thresholds globales. Incluyen las dieciséis celdas, ambos
proposer-only y las diez réplicas de control; hard es la referencia común.

Se reportan por shard soportes, thresholds, no identidad y estos siete deltas
direccionales: incompatibility−hard en regret, harm−hard en worst regret,
incompatibility−harm en regret, incompatibility−su control medio en regret,
harm−su control medio en worst regret, incompatibility−HGB-proposer-only en
regret y harm−HGB-proposer-only en worst regret. Se conserva valor y signo, sin
convertir coincidencia de signos en gate. Ningún resultado de shard cambia el
freeze global, los patrones, la apertura del monitor ni los thresholds
aplicados al monitor. Un shard bajo sus mínimos queda `NOT_EVALUABLE` y no
invalida por sí solo el resultado global.

## Patrón prospectivo informativo

El reporte calculará condiciones predeclaradas sin traducirlas a GO/NO-GO:

Todas las condiciones siguientes se calculan sólo sobre monitor. Validation se
publica como diagnóstico post-freeze y no se combina con monitor.

### Política de incompatibilidad

- `mean(delta_regret_vs_hard) <= -0.005` y
  `ci95_high(delta_regret_vs_hard) < 0`;
- `ci95_low(delta_accuracy_vs_hard) >= -0.01` y
  `ci95_low(delta_compatibility_vs_hard) >= 0`;
- `ci95_high(delta_worst_regret_vs_hard) <= +0.01`;
- `ci95_high(delta_regret_vs_mean_max_displacement) < 0`;
- `authorized_pair_tokens >=25` y replay exacto.

### Política de harm

- `mean(delta_worst_regret_vs_hard) <= -0.01` y
  `ci95_high(delta_worst_regret_vs_hard) < 0`;
- `ci95_high(delta_regret_vs_hard) <= 0`;
- `ci95_low(delta_accuracy_vs_hard) >= -0.01` y
  `ci95_low(delta_compatibility_vs_hard) >= 0`;
- `ci95_high(delta_worst_regret_vs_mean_max_displacement) < 0`;
- `authorized_pair_tokens >=25` y replay exacto.

Cada condición conserva `true | false | NOT_EVALUABLE`. Una comparación contra
controles es `NOT_EVALUABLE` si su familia no tiene cinco mappings distintos y
válidos; eso no altera las condiciones ya calculables. El patrón agregado es
`NOT_EVALUABLE` si cualquiera de sus condiciones lo es, incluso si otra ya es
`false`; es `true` sólo si todas son `true`, y `false` en los demás casos
completamente evaluables. Cero overrides monitor produce deltas cero y soporte
`false`, no `NOT_EVALUABLE`.

Los dos patrones se reportan separados. No se exige que uno domine al otro y no
se promedian sus métricas. La comparación head-to-head informa el costo de
priorizar media o cola; la elección de esa prioridad no se delega al runner.

## Artefactos y replay

Los outputs canónicos, distintos y hermanos serán:

```text
data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1/
data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1_replay/
```

La config contendrá una matriz cerrada `artifact_classes`; todo archivo debe
pertenecer a exactamente una clase y el inventario rechazará faltantes, extras
o solapamientos antes y después de cada fase:

| Clase | Paths cerrados | Comparación primaria↔replay |
|---|---|---|
| `scientific_exact` | `config.snapshot.json`, `source_bindings.json`, `pre_generation_freeze.json`, `preparation_freeze.json`; `recovery_amendment.json` condicional; `benchmark/{manifest.json,protocol_config.json,attestations/semantic_root.json,commitments/semantic.jsonl,visible/{calibration_null,train,val,lockbox}.jsonl}`; `fit/{feature_schema,fit_freeze,max_displacement_diagnostics}.json`; `calibration/calibration_freeze.json`; `validation/{validation_freeze,validation_summary}.json`; `adjudication/monitor_action_freeze.json`; `analysis.json`, `REPORT.md` | bytes exactos |
| `scientific_array_exact` | `inference/logits/seed{17,29,43}__{train,val,lockbox}.npz`; `prepared/{gate_select_inference,sealed_monitor_inference}_bundle.npz`; `fit/{model_state_arrays,train_scores,max_displacement_mappings}.npz`; `calibration/{validation_scores,validation_policy_arrays}.npz`; `validation/validation_metrics.npz`; `adjudication/{monitor_scores,monitor_policy_arrays,bootstrap_indices,analysis_arrays}.npz` | mismas claves, dtype, shape y valores con `equal_nan=True` |
| `functional_state` | los dieciséis `fit/model_states/<model_id>.joblib` y `fit/model_states/manifest.json` | equivalencia funcional y binding intra-run |
| `operational_semantic` | `generation_receipt.json`, `preparation_receipt.json`, `preparation_replay.json` sólo en replay, `inference/access_receipt.json`, `journals/{prepare,fit,calibrate_scores,validate,monitor_apply,monitor_evaluate}.json`, `runtime.json` | igualdad tras retirar sólo campos allowlisted de modo, timestamp, path raíz, duración y RSS |
| `secret_excluded_from_public_manifest` | `generation_escrow.json`; `benchmark/sealed/{calibration_null.jsonl,train.jsonl,val.jsonl,lockbox.jsonl,generation_secret.json,identity_secret.json,semantic_commitment_secret.json}`; `prepared/{gate_fit,gate_select_truth,sealed_monitor_truth}_bundle.npz` | SHA-256 exacto por el coordinador; nunca inventario público ni input de worker no autorizado |
| `self_reference` | `artifact_manifest.json`; `replay_comparison.json` sólo en replay | schema y cobertura completas; no igualdad consigo mismo |

Las llaves con `{...}` expresan expansión cartesiana exhaustiva, no globs
abiertos. Ningún directorio acepta un `**` residual.

Las cardinalidades condicionales se gobiernan con dos ejes ortogonales,
`run_role = primary|replay` y `recovery_context = false|true`:

| Estado | `preparation_replay.json` | `recovery_amendment.json` | `FAILURE.json` |
|---|---:|---:|---:|
| primary normal | 0 | 0 | 0 |
| replay normal | 1 | 0 | 0 |
| primary recuperado | 0 | 1 | 0 |
| replay del recuperado | 1 | 1 | 0 |
| failed | 0 o 1 según `run_role` y último journal | 0 o 1 según `recovery_context` | exactamente 1 |

En modo recovery, `recovery_amendment.json` debe ser byte-exacto entre primary
y replay y coincidir con el SHA autorizado antes de todo acceso semántico. Su
ausencia en modo recovery o su presencia en modo normal son errores.

Un intento `.failed_<timestamp>` usa el schema separado
`wave59-failed-attempt-v1`. Debe contener `FAILURE.json` y
`failure_inventory.json`; el primero registra tipo de error, hash del mensaje,
último estado, máximo nivel de truth materializado, rol, contexto de recovery y
paths original/archivado sin incluir secretos. El inventario clasifica cada
archivo parcial con las mismas seis clases base y añade `failure_record` para
esos dos JSON. Sólo son obligatorios los artefactos publicados hasta el último
journal; los posteriores deben estar ausentes. No se compara el árbol fallido
contra el primary ni se permite ningún path extra.

Los tests cubrirán las cuatro combinaciones primary/replay × normal/recovery y
un fallo inyectado antes y después de cada apertura de truth. Cada inventario
debe terminar con cero paths faltantes, extras, solapados o sin clase; el replay
recuperado debe verificar el mismo amendment.

Los IDs de modelo son independientes de los IDs de policy:

- proposers: `proposer-ridge`, `proposer-hgb`;
- guards verdaderos: `guard-{logistic,hgb}-{harm,incompatibility}`;
- controles harm: `control-hgb-harm-{59031,59032,59033,59034,59035}`;
- controles incompatibility:
  `control-hgb-incompatibility-{59041,59042,59043,59044,59045}`.

Las dieciséis policies referencian esos IDs compartidos; no generan dieciséis
modelos por coincidencia cardinal. `fit/model_states/manifest.json` enumera los
dieciséis filenames permitidos, sus hashes intra-run y rechaza faltantes o
extras. Los joblib son copias funcionales para recuperación y diagnóstico, no
inputs de workers posteriores. `fit/model_state_arrays.npz` es el estado
ejecutable, queda hash-bound en `fit_freeze.json` y su scorer se prueba contra
el objeto ajustado antes de publicar FIT con `rtol=0`, `atol=2e-15` y
`equal_nan=True`; esa tolerancia sólo absorbe redondeo de acumulación. La
equivalencia funcional de cada joblib se prueba sobre las matrices
inference-safe exactas de train, validation y monitor:
salida `float64`, misma shape y valores `rtol=0`, `atol=0`,
`equal_nan=True`. La representación portable de coeficientes, scaler y nodes
HGB queda en `fit/model_state_arrays.npz`; una vez materializadas, esas states y
las scores exactas son la autoridad de reanálisis.

Los bundles truth permanecen sellados y sólo el coordinador los materializa en
el sandbox temporal de VALIDATE o MONITOR-EVALUATE; su preservación canónica no
amplía las allowlists de los workers.

Replay parte de un directorio vacío, reutiliza exactamente el triplete de
secretos del escrow primario, refitea y recalibra sin acceso del worker al
output primario. Sólo después de publicar el replay el coordinador compara:

- byte-exactos: todos los miembros de `scientific_exact`, incluidos los cinco
  freezes previos y `monitor_action_freeze.json`;
- array-exactos por clave, dtype y shape con `equal_nan=True`: todos los NPZ de
  preparación, states portables, scores, mappings, máscaras, acciones,
  métricas, índices bootstrap y arrays de análisis;
- por hashes y validadores existentes: benchmark visible, attestations,
  commitments, logits upstream y bundles preparados;
- equivalencia funcional más hash de origen: objetos joblib ejecutables.

Las únicas exclusiones de igualdad byte-exacta son las declaradas por las
clases `operational_semantic`, `functional_state` y `self_reference`; secretos
se comparan por hash sin publicarlos. El comparador rechaza cualquier path no
clasificado y los inventarios pre/post cubren todo el árbol real.

La config ligará plan, auditoría aceptada, implementación, tests, dependencias,
modelos upstream, política, utilidad y claves públicas. El manifest repetirá
como campos top-level los hashes de plan y auditoría, además del hash de config.

## Recursos y condición de parada

El régimen previsto es CPU con cuatro hilos. HGB sobre estos bundles tomó `53.6
s` y aproximadamente `1.02 GiB` RSS en Ola 58. Wave 59 comparte scores entre
las dieciséis celdas y requiere trece fits HGB por corrida —tres verdaderos y
diez controles—, por lo que se presupuesta un máximo de `30 min` y `8 GiB` RSS
por primaria o replay, `60 min` combinados. `CUDA_VISIBLE_DEVICES=''` se fija en
coordinador y workers y el runtime verifica threadpools efectivos.

Antes del freeze se ejecutará un preflight CPU representativo que incluya cada
tipo de modelo y un control de cada target. Si su proyección conservadora supera
`30 min` por corrida, o si una corrida real cruza ese wall time o `8 GiB` RSS,
se la detiene antes de CUDA y se preserva el estado durable. Como Mariano pidió
aviso por Telegram cuando la GPU sea necesaria, se publica primero una nota
inmutable en inbox con `request_id`, evidencia, próximo responsable y condición
de reanudación; luego se usa `/usr/local/bin/m2-alert` con objetivo, duración y
VRAM estimada. No se usa GPU ni se inicia una sustitución CPU prolongada sin su
indicación.

## Secuencia de aceptación

1. auditoría independiente de este plan;
2. corrección y reauditoría de todo finding que afecte validez, trazabilidad o
   capacidad de avanzar;
3. implementación y tests sin crear el triplete del draw;
4. auditoría independiente de implementación;
5. freeze de commit y hashes;
6. creación única del triplete de secretos, primaria y replay;
7. auditoría independiente de resultados;
8. actualización de cierre, wiki y memoria sin promoción ni decisión
   científica automática.
