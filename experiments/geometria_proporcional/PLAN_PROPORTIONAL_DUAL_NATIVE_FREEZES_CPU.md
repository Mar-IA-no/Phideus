# Plan CPU — dos design freezes nativos coordinados

> **Estado:** `AMENDED-AUDITED-PASS-R551 / DESIGN-AND-PREFLIGHT-ONLY / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-07
> **Autoridad de promoción y GO/NO-GO:** usuario

## 1. Pregunta finita

`MAPPING-FEASIBILITY` descartó un factorial común entre EIV, el núcleo
relacional y el posterior set-valued, pero validó las dos hojas nativas. Este
goal no ejecuta todavía esos experimentos. Congela dos protocolos ejecutables,
implementa sus checkers y preflights y estima el costo de la corrida siguiente:

1. **freeze relacional:** `GENERIC/TYPED × WLS/IRLS` dentro del banco de grafos
   de log-razones;
2. **freeze set-valued:** `MARGINAL/JOINT × HARD/CONTEXTUAL` dentro del banco de
   compatibilidad de cuatro familias y quince conjuntos no vacíos.

Las dos ramas comparten reglas de procedencia, fases, replay, preservación y
auditoría. No comparten unidad, observación, target, métrica ni una función de
mérito. Una rama puede quedar lista aunque la otra falle su preflight. Ningún
resultado selecciona una arquitectura ni constituye GO/NO-GO.

## 2. Envolvente común sin IR común

La coordinación se limita a estos campos:

```text
experiment_id, branch_id, schema_version, source_commit,
plan_sha256, config_sha256, phase, role, lineage_id,
artifact_class, content_sha256, replay_status,
gpu_used_or_queried=false, architecture_promoted=false,
scientific_decision=null, decision_authority=user
```

No existe `combined_score`, ranking cruzado ni conversión entre RMSE relacional
y regret set-valued. El coordinador sólo devuelve uno de cinco estados técnicos
de diseño:

```text
BOTH_DESIGN_FREEZES_VALID
RELATIONAL_FREEZE_ONLY_VALID
SET_VALUED_FREEZE_ONLY_VALID
NEITHER_DESIGN_FREEZE_VALID
TECHNICAL_FAILURE
```

`DESIGN_FREEZE_VALID` significa que el contrato declarativo, los schemas, las
fases, las primitives numéricas ya existentes o añadidas en este goal y el
presupuesto proyectado superaron preflight. Significa únicamente
`READY_FOR_RUNNER_IMPLEMENTATION`: este goal no puede acreditar que un runner
aún inexistente respete el contrato ni que la primitive vaya a superar sus
controles.

## 3. Grafo de fases y autoridad

Cada rama usa procesos y directorios no solapados:

```text
source verifier
  -> public/private preparer
  -> train-or-fit
  -> calibration/selection
  -> freeze hash + read-only
  -> monitor apply without target
  -> monitor evaluate with target
  -> independent checker
  -> replay in a second root
```

En este goal sólo se ejecutan `source verifier`, construcción declarativa del
freeze, checker sobre fixtures y preflight. No se genera un draw nuevo, no se
entrena, no se hace forward neuronal y no se abre ningún monitor o lockbox.

La corrida sucesora tendrá los siguientes roles:

| Rol | Puede leer | No puede leer |
|---|---|---|
| preparer | fuentes ligadas, manifests y sidecars privados | monitor antes de su fase |
| trainer/fitter | público y targets de su split de ajuste | selección, monitor, IDs originales no permitidos |
| selector | outputs congelados y target de selección | monitor |
| monitor applier | inputs públicos, estados y thresholds congelados | target del monitor |
| evaluator | acciones/scores congelados y target autorizado | parámetros mutables |
| checker | todos los artefactos ligados para recomputar | helpers del runner como raíz de autoridad |

Toda apertura se registra por path, rol, fase, modo y SHA-256. Una lectura fuera
de fase produce `INVALID-PROSPECTIVE-ATTEMPT`; no se repara sobre el mismo draw.

## 4. Freeze relacional

### 4.1 Objeto y unidad

El objeto sigue siendo una clase de potenciales positivos módulo reescala
global, representada en log-espacio por `x ~ x+c`. La unidad inferencial es el
`master_id`; las vistas IID y grouped de test son observaciones dependientes del
mismo master. La incidencia canónica conserva `-1` en origen y `+1` en destino.

El generador será `proportional-graph-contract-v1`, con un draw nuevo y público:

| Campo | Freeze |
|---|---:|
| masters | `1024` |
| train / calibration / validation / test | `512 / 128 / 128 / 256` masters |
| nodos | `8..16` |
| ruido gaussiano | `0.04` |
| tasa de corrupción | `0.15` |
| amplitud | `0.6..1.4` |
| mecanismo test | vistas pareadas `iid/grouped` |
| seed del draw | `2026090717` |

Todos los derivados de un master permanecen en el mismo split. `x_true`,
relación limpia, máscara causal, mecanismo, seed e IDs persistentes viven en
el sidecar privado. El modelo recibe sólo topología reindexada, relación
observada, máscaras, paths orientados y varianza pública.

### 4.2 Brazos y control causal

Los brazos principales son `RAW-GENERIC` y `RAW-TYPED`. Ambos usan el mismo
encoder, heads, ancho `64`, dos bloques, tensor público, número de parámetros
(`50.435` bajo la implementación vigente), shapes, inicialización por seed,
batches y cantidad de llamadas al path MLP. `RAW` evita reabrir el factor de
evidencia analítica `CLOSURE`, ya adjudicado por el smoke previo.

Tres seeds nuevas y predeclaradas se aplican a ambos brazos:

```text
15485863, 32452843, 49979687
```

`RAW-TYPED-PATH-SHUFFLE` es un control entrenado con el mismo presupuesto sobre
el roster elegible común. Reutiliza exactamente `shuffled_path_tensors` y la
seed vigente del smoke
`_stable_seed("path-shuffle", training_seed, _path_structure_digest(view))`.
El digest serializa sólo arrays públicos de estructura y excluye
`observed_log_ratio`; no recibe `master_id`, `mechanism`, targets ni sidecar
privado. Dentro de cada vista, reasigna conjuntamente pares de operandos y
signos mediante asignación lineal. Conserva target edge, cantidad de paths,
marginales de operandos/signos y shapes; exige que toda fila válida cambie y que
ningún operando reasignado sea el target.

La elegibilidad se calcula una sola vez, sin targets, antes del entrenamiento.
Mutar cualquier campo privado conservando el bundle público debe dejar
byte-idénticos seed, tensor shuffled y roster; esa invariancia es fixture y
mutación obligatoria.
Un master entra al roster común de un split si todas sus vistas requeridas son
elegibles; el mismo roster se aplica a `GENERIC`, `TYPED` y `PATH-SHUFFLE`, a
todos los seeds y executors. No se redibuja ni se prometen los `1024` masters
como soporte efectivo: se preservan conteos iniciales, exclusiones y reason
codes. Los mínimos son `480/120/120/240` masters efectivos en
train/calibration/validation/test. Menor soporte invalida sólo este freeze. El
control no agrega una tercera celda al factorial: adjudica si una diferencia
tipada depende de la composición correcta.

### 4.3 IR y objetivo común de entrenamiento

Cada modelo produce exactamente dos arrays por arista; el executor materializa
y preserva dos objetos adicionales, que no deben confundirse:

```text
corrected_log_ratio:         float64 [n_edges]
raw_reliability:             float64 [n_edges], weight_floor <= w <= 1
normalized_wls_weight:       float64 [n_edges], valid w > 0, mean(valid w)=1
normalized_irls_base_weight: float64 [n_edges], valid w > 0, mean(valid w)=1
```

La cota superior sólo aplica a la salida cruda de la cabeza; después de dividir
por la media, un peso relativo puede ser mayor que uno. Estos arrays son pesos
relativos, no una reliability calibrada. Los dos executors reciben la misma
pareja `corrected_log_ratio/raw_reliability` y el mismo gauge. La pérdida común
es el siguiente objetivo compuesto fijo; coeficientes iguales no se interpretan
como neutralidad de escala o gradiente entre solvers:

```text
L = 1.0 * relation_mse
  + 0.05 * local_closure_l1
  + 0.5 * quotient_mse(differentiable_WLS)
  + 0.5 * quotient_mse(fixed_K192_IRLS_surrogate)
```

R354 es evidencia antecedente sólo para el caso unit-base y no certifica pesos
base aprendidos. R550 ejecutó el primer preflight base-weighted sobre `32`
grafos/`96` estados: Torch↔NumPy y ambos gradientes pasaron, pero `K=64` falló
la aproximación al executor convergido por un estado grouped que necesitó `136`
iteraciones. Un scan de valor congelado probó `K={64,96,128,160,192,256}`;
`K=160` fue el primero ensayado que pasó y se elige `K=192` como margen
conservador, sin afirmar minimalidad sobre enteros no probados.

La extensión `base_weights` conserva el default unitario y la referencia NumPy
fixed-K independiente no importa el helper Torch. La selección de profundidad
queda cerrada con R550: no se vuelve a ajustar K. El preflight confirmatorio
`BASE_WEIGHTED_K192_CONFORMANCE` usa un draw numérico nuevo con seed
`2026090731`, `64` grafos deterministas, mecanismos IID/grouped, `n=8..16`, tres
patrones no unitarios por grafo y `float64`, el dtype real de entrenamiento. No
reutiliza los estados de calibración `2026090723`. Compara:

1. potenciales, pesos finales y objetivo Torch frente a NumPy fixed-K;
2. potenciales frente al executor canónico convergido;
3. autograd frente a diferencias finitas centrales para coordenadas estables de
   `corrected_log_ratio` y `raw_reliability`, excluyendo sólo pesos dentro de
   `10h` del floor o residuos dentro de `10h` de un kink Huber, con conteo y
   motivo.

Los umbrales heredados de R354 permanecen sin cambio: máximo Torch↔NumPy `<=1e-9`; p99/máximo
RMSE frente al canónico `<=1e-4/1e-3`; coseno mediano `>=0.999`, p95 de error
relativo `<=1e-2` y cero inversiones de signo para magnitud `>=1e-6`, por cada
familia de gradiente. Ninguna exclusión cuenta como acierto. Si falla una
condición, `R11_BASE_WEIGHTED_K192_CONFORMANCE=FAIL` y el freeze relacional
queda inválido; no se prueba otro K sobre el draw confirmatorio ni se entrena
con ese surrogate.

WLS y el surrogate reciben el mismo `raw_reliability`; el target privado sólo
entra en el loss, nunca como feature. Se preservan por batch los cuatro términos
y sus normas de gradiente separadas. No hay normalización dinámica, calibración
de coeficientes ni ajuste retrospectivo.

El schedule es fijo: `20` épocas, batch `64`, AdamW `lr=1e-3`,
`weight_decay=1e-4`, clipping `5`, ocho threads Torch y determinismo estricto.
No hay early stopping. `last_epoch` es el checkpoint canónico; se preservan
además épocas `5,10,15` como diagnósticos no seleccionables.

### 4.4 Executors y fallos

La evaluación usa implementaciones externas NumPy:

- WLS con `weight_floor=1e-3`;
- Huber-IRLS con `delta=1.5`, damping `1.0`, tolerancia `1e-6` y máximo `7500`.

La convergencia es un outcome, no un filtro. Por brazo, seed y slice se reportan
failure rate e iteraciones sobre todos los masters del roster. Los denominadores
son distintos y explícitos:

- WLS `TYPED-GENERIC`: todos los masters con grafo válido; un fallo WLS invalida
  la celda porque el contrato exige solución cerrada;
- IRLS `TYPED-GENERIC`: soporte donde convergen ambos brazos, más failure rate
  sobre el roster completo;
- interacción WLS/IRLS: soporte donde están definidas las cuatro celdas.

Cada estimando persiste `n_total`, `n_complete`, máscara y patrón de faltantes.
IRLS o interacción quedan `NOT_EVALUABLE_SOLVER_SUPPORT` si su
`n_complete/n_total < 0.99`; WLS no pierde masters por un fallo IRLS. La
diferencia de failure rate `TYPED-GENERIC` se estima por separado sobre el
roster completo y no se convierte una selección de supervivientes en evidencia
positiva. Relación pre-solver siempre conserva su propio soporte.

### 4.5 Controles de interfaz

Se ejecutan sin re-forward:

1. `UNIT-WEIGHT`: sustituye reliability por uno y conserva la relación;
2. `WEIGHT-LOCATION-SHUFFLE`: derangea pesos dentro de cada vista, conserva su
   distribución y no mueve relaciones;
3. `TOTAL-TARGET-SHUFFLE`: aplica exactamente el transporte por master definido
   en el plan de `MAPPING-FEASIBILITY`, con un donante común para IID/grouped;
4. permutación de nodos, inversión de orientación y gauge shift como sanities
   metamórficos, no como muestras independientes.

Los dos primeros separan contribución de relación y localización de peso. El
tercero prueba asociación con el target sin cambiar input, output ni executor.

### 4.6 Estimandos y lectura

El slice confirmatorio es `test/grouped`; `test/iid` es control de transporte.
El estimando primario es el efecto medio condicional a las tres inicializaciones
fijadas: se calcula el delta pareado dentro de cada seed, se promedia por master
y se usan `5000` bootstraps pareados sólo por master con índices persistidos. El
intervalo no generaliza a la población de seeds. Se informan las tres
direcciones individuales; un ensemble que promedia outputs antes del executor
es secundario y nunca sustituye al estimando primario.

Orden obligatorio:

1. integridad, igualdad de inicialización y compute;
2. RMSE de relación `TYPED-GENERIC` y `TYPED-PATH-SHUFFLE`;
3. failure rate e iteraciones por executor;
4. quotient RMSE `TYPED-GENERIC` por WLS e IRLS sobre soporte común;
5. interacción `(TYPED-GENERIC)_IRLS-(TYPED-GENERIC)_WLS`;
6. controles de peso y target;
7. IID, sanities, seeds individuales y ensemble.

El checker usa percentiles `2.5/97.5` del bootstrap y orienta todos los deltas
como primer término menos segundo, con menor valor favorable. No hay selección
post hoc de métricas: el patrón es conjuntivo y sólo se rotula si todas sus
filas obligatorias pasan. La tabla total es:

| ID | Estimando / soporte | Condición requerida | Si falla |
|---|---|---|---|
| `REL_RELATION` | relation RMSE `TYPED-GENERIC`, grouped, roster completo | `CI_upper < 0` | `ADVERSE` si `CI_lower > 0`; si no `NOT_RESOLVED` |
| `REL_PATH` | relation RMSE `TYPED-PATH_SHUFFLE`, mismo roster | `CI_upper < 0` | misma regla |
| `REL_WLS` | quotient RMSE `TYPED-GENERIC`, todos los WLS válidos | `CI_upper < 0` | misma regla; fallo WLS → `NOT_EVALUABLE` |
| `REL_IRLS` | quotient RMSE `TYPED-GENERIC`, soporte convergido común | cobertura `>=0.99` y `CI_upper < 0` | soporte bajo → `NOT_EVALUABLE`; resto misma regla |
| `REL_FAILURE` | failure-rate `TYPED-GENERIC`, roster completo | `CI_upper <= 0` | `ADVERSE` si `CI_lower > 0`; si no `NOT_RESOLVED` |
| `REL_TARGET` | quotient RMSE real menos `TOTAL_TARGET_SHUFFLE`, por executor | `CI_upper < 0` en ambos | soporte bajo → `NOT_EVALUABLE`; resto misma regla |
| `REL_INTERACTION` | `(TYPED-GENERIC)_IRLS-(TYPED-GENERIC)_WLS` | descriptivo, sin condición de patrón | se reporta o `NOT_EVALUABLE` |

`PATTERN_PRESENT` exige las primeras seis filas. `NOT_EVALUABLE` tiene
precedencia; `ADVERSE` tiene precedencia sobre `NOT_RESOLVED`. Los controles de
peso son atribución secundaria y no se fuerzan a mostrar ventaja. Son intervalos
descriptivos condicionados a este protocolo, sin claim de cobertura familiar.
Estos rótulos no promueven arquitectura ni son GO/NO-GO.

## 5. Freeze set-valued

### 5.1 Objeto, unidad y draw

La unidad es `pair_token`. El target es un conjunto booleano no vacío sobre
cuatro familias; el posterior vive sobre los quince conjuntos no vacíos. La
observación común son cuatro logits ensemble y tres juegos de logits por seed,
producidos por los checkpoints W51 congelados `17/29/43`. Las 24 utilidades
ordinales de W52 y la penalización incompatible `1.25` son externas al
posterior.

La corrida sucesora usa el generador y la frontera ciega de Wave 59, pero con
una clave fresca creada después del commit del freeze. Se esperan `768`
pair_tokens elegibles por split físico. `train` se divide por hash antes de
abrir targets en dos roles disjuntos:

```text
posterior_fit: 384 esperados
policy_fit:    384 esperados
decision_select: 768 esperados desde val
monitor:         768 esperados desde lockbox
```

El mínimo técnico es `300/300/600/600`; quedar debajo produce
`INSUFFICIENT_PHASE_SUPPORT` y no habilita redibujar.

Además del conteo global, `posterior_fit` debe contener target no vacío, ambas
clases en el pool binario de MARGINAL y al menos dos filas por estrato de fold;
cada `policy_fit` debe tener disagreement activo y ambas clases para sus guards
de daño e incompatibilidad; `decision_select` debe evaluar todas las candidatas
de thresholds. La ausencia de cualquiera produce un reason code de soporte y
no se toma prestada una fila de otra fase.

### 5.2 Factor representación

Las dos representaciones reciben los mismos logits y el mismo orden de sets:

- `MARGINAL`: un calibrador Platt pooled sobre las filas `token × family`, como
  W53, y producto Bernoulli condicionado a set no vacío;
- `JOINT`: `joint_full`, cuatro términos unary, tres de cardinalidad y cinco
  coordenadas libres de interacción con suma cero.

MARGINAL conserva la receta nativa W53: `LogisticRegression` con `C=1`, `l2`,
`lbfgs`, intercept, sin class weights, seed `5301`, `max_iter=1000` y datos
float64 aplanados como `[4*n_token,1]`. Exige ambas clases en el pool y
`n_iter < max_iter`; no hace selección de hiperparámetros.

JOINT selecciona su lambda nativa
`[1e-4,1e-3,1e-2,1e-1,1,10]` sólo dentro de `posterior_fit`. Cuatro folds se
construyen sin usar `cluster_id` —es único—: dentro de cada
`(design_stratum,cardinality)`, se ordena por
`SHA256("set-fold-v1" || pair_token)` y se asigna `rank mod 4`. Cada candidato
se ajusta en tres folds y se evalúa en el cuarto; la clave es NLL exacta
out-of-fold media, luego Brier marginal y luego mayor lambda. El objetivo es el
de `wave54_joint_set.py`, con penalización
`0.5*lambda*||theta-reference||²`, `L-BFGS-B`, `max_iter=2000`, `gtol=1e-9` y
`ftol=1e-12`. Cualquier no convergencia, no finito o fold sin soporte produce
`POSTERIOR_FIT_FAIL`. Tras seleccionar, ambas recetas se reajustan sobre todo
`posterior_fit` y se congelan. La igualdad causal es de inputs, fase y target;
no se finge igualdad de parametrización o regularización entre familias.

Un control target-shuffled se ajusta para MARGINAL y JOINT con una única
permutación común. `target_derangement_v1` es total y byte-determinista:

```text
rng = numpy.random.Generator(numpy.random.PCG64(53602))
for stratum in sorted(unique(fold_id, design_stratum, cardinality)):
    tokens = sorted(pair_token UTF-8 del estrato)
    if len(tokens) == 1: registrar singleton y donor=receiver
    else:
        key[i] = rng.bit_generator.random_raw()  # uint64
        order = sorted(tokens, key=(key[i], token UTF-8))
        shift = 1 + rng.integers(0, len(order)-1, endpoint=False)
        donor[order[i]] = order[(i+shift) mod len(order)]
```

Los estratos se ordenan por `(int fold_id, design_stratum UTF-8,
int cardinality)`. `random_raw` se consume en ese orden y `shift` se convierte a
entero Python; no existe otro RNG. El mapa canónico es una lista JSON ordenada
por receiver con keys `receiver,donor,fold_id,design_stratum,cardinality`, UTF-8,
`ensure_ascii=false`, separadores `(',',':')` y newline final; su SHA-256 es
obligatorio. Una fixture contiene al menos un estrato con más de un derangement
posible y fija el digest esperado.

El derangement no tiene identidad cuando el estrato posee al menos dos filas;
un singleton se registra y queda fijo. Se persisten mapa, folds, singletons y
fracción permutable. Menos de `0.8` de filas permutables produce
`TARGET_SHUFFLE_NOT_MATERIALIZABLE`; no se cambia seed ni se redibuja. El mismo
mapa se usa durante validación interna y refit, conserva logits, folds,
cardinalidad y budget y nunca cruza un fold. Los controles shuffled no son
celdas del factorial.

### 5.3 Factor reader

`HARD` es `HARD_MAP_SET`: toma el set MAP del posterior con empate por menor
índice binario y luego la familia de mayor utilidad dentro de ese set con
empate por menor índice. Se calcula separadamente desde MARGINAL y JOINT.

`CONTEXTUAL` usa la receta cerrada por `MAPPING-FEASIBILITY` para cada posterior:

1. candidato de mínimo riesgo posterior;
2. baseline HARD del mismo posterior;
3. schema adaptado de 17 features cuya cardinalidad y masa corresponden al set
   MAP, no al threshold histórico `0.5`;
4. proposer Ridge float64 `alpha=1`;
5. guard Logistic de daño, target `gain < -1e-12`;
6. guard Logistic de incompatibilidad, target `not target[candidate]`;
7. ajuste sólo en `policy_fit`, ponderado por disagreement por token;
8. selección sólo en `decision_select` sobre cuantiles proposer
   `[0.5,0.6,0.7,0.8,0.9,0.95,0.975]` y guards
   `[0.1,0.2,0.3,0.4,0.5,0.6,0.8]`, más `HARD_ONLY`;
9. clave ascendente `(mean_regret, incompatibility_rate, harm_rate,
   -authorized_rows, proposer_q, harm_q, incompatibility_q)`;
10. estados, scalers, thresholds, feature order y hashes congelados antes de
    monitor.

Cada posterior ajusta su propia instancia con receta y presupuesto idénticos.
Esto estima el efecto del sistema representación+reader; el efecto de
representación pre-reader se informa antes y nunca se deduce del action score.
Una igualdad empírica entre celdas se registra como duplicación, no se elimina.

### 5.4 Controles matched

Para cada posterior se ajustan cinco controles proposer/guard con seeds
`53611,53617,53623,53629,53633`. Los tres targets de cada control se derangean
con `PCG64` dentro de `(policy_index,disagreement_count)` mediante asignación de
costo hash que maximiza desplazamiento Hamming; la fracción permutable debe ser
`>=0.8`. Cada seed usa un mapa único común a proposer y guards, persistido antes
del fit. Los controles conservan diseño, weights, clases, modelos, cuantiles y
filas; identidad, clase perdida o cambio de soporte invalida ese control.

El `monitor applier`, todavía target-blind, construye el matching por posterior,
seed y token sobre el vector congelado de `24` políticas:

1. `k` es el número de posiciones donde `CONTEXTUAL != HARD` para ese token;
2. el universo control son posiciones donde candidato control `!= HARD` y sus
   dos guards congelados autorizan;
3. se ordena sin target por
   `(-proposer_score,harm_probability,incompatibility_probability,policy_index)`
   y se toman exactamente las primeras `k`;
4. si hay menos de `k`, el token recibe `MATCH_IMPOSSIBLE`; si `k=0`, el match
   vacío es válido pero no entra al soporte de atribución;
5. la acción matched usa el candidato control en esas `k` posiciones y HARD en
   las restantes, por lo que el Hamming respecto de HARD es exactamente el del
   reader verdadero.

El universo, la clave, los desempates, el mapa y su SHA-256 se persisten antes
de que el evaluator abra targets. Para cada posterior se persisten
`true_override_mask` y las cinco `match_valid_mask`. El soporte confirmatorio es
su intersección exacta:

```text
U_true   = tokens con k > 0
U_common = U_true & match_valid_seed_1 & ... & match_valid_seed_5
coverage = |U_common| / |U_true|
```

`U_true` vacío o `coverage < 0.8` produce
`NOT_EVALUABLE_MATCHED_CONTROL`. Sobre `U_common`, y sólo allí, se calcula por
token la media aritmética de la métrica de los cinco controles y se la compara
con el reader verdadero sobre el mismo token; no existen unión ni promedios con
cantidad variable de controles. El bootstrap remuestrea `U_common`. La mejora
frente a HARD puede reportarse fuera de esta atribución como eficacia local, no
como evidencia del target aprendido.

### 5.5 Estimandos y lectura

Se persisten `5000` bootstraps pareados por `pair_token`. La representación se
lee antes del reader:

- exact set NLL;
- Brier marginal;
- error absoluto de cardinalidad;
- masa del target verdadero.

La decisión se lee después:

- accuracy de acción;
- compatibilidad;
- regret medio;
- worst regret por token;
- overrides, daño y soporte.

Orden obligatorio:

1. integridad, phase support y posterior normalization;
2. `JOINT-MARGINAL` pre-reader y `JOINT-JOINT_SHUFFLED`;
3. `CONTEXTUAL-HARD` dentro de cada posterior;
4. efecto de representación bajo HARD y bajo CONTEXTUAL;
5. interacción de los dos factores;
6. controles matched y duplicaciones;
7. resultados por utilidad, seed y cardinalidad como secundarios.

El estimando primario usa logits ensemble congelados; los tres checkpoints se
reportan por separado como sensibilidad, sin interpretar el bootstrap por
pair_token como variabilidad de entrenamiento. Todos los deltas se orientan
primer término menos segundo y las pérdidas son menores-mejor. La tabla total,
con percentiles bootstrap `2.5/97.5`, es:

| ID | Estimando / soporte | Condición requerida | Si falla |
|---|---|---|---|
| `SET_JOINT_NLL` | exact-set NLL `JOINT-MARGINAL`, todos los tokens | `CI_upper < 0` | `ADVERSE` si `CI_lower > 0`; si no `NOT_RESOLVED` |
| `SET_JOINT_BRIER` | Brier marginal `JOINT-MARGINAL`, mismo soporte | `CI_upper <= 0` | misma regla |
| `SET_SHUFFLE` | exact-set NLL real menos su target-shuffled, por posterior | permutable `>=0.8` y `CI_upper < 0` | soporte bajo → `NOT_EVALUABLE`; resto misma regla |
| `READER_REGRET` | regret `CONTEXTUAL-HARD`, por posterior | `CI_upper < 0` | `ADVERSE` si `CI_lower > 0`; si no `NOT_RESOLVED` |
| `READER_COMPAT` | incompatibility-rate `CONTEXTUAL-HARD` | `CI_upper <= 0` | misma regla |
| `READER_WORST` | worst-regret `CONTEXTUAL-HARD` | `CI_upper <= 0` | misma regla |
| `READER_CONTROL` | regret contextual verdadero menos media por token de los cinco matched controls sobre `U_common` | cobertura común `>=0.8` y `CI_upper < 0` | soporte bajo → `NOT_EVALUABLE`; resto misma regla |
| `FACTOR_INTERACTION` | interacción representación×reader en regret | descriptivo, sin condición de patrón | se reporta o `NOT_EVALUABLE` |

`JOINT_PATTERN_PRESENT` exige las tres primeras filas, incluyendo que JOINT se
separe de su propio shuffled; `CONTEXTUAL_PATTERN_PRESENT` se adjudica por
posterior y exige las cuatro filas `READER_*`. `NOT_EVALUABLE` tiene precedencia
y `ADVERSE` precede a `NOT_RESOLVED`. Son intervalos descriptivos condicionados
a este protocolo, sin claim de cobertura familiar. No hay rótulo global, ranking
contra la rama relacional, promoción ni GO/NO-GO.

## 6. Fuentes que el freeze debe ligar

El preflight calcula SHA-256 actuales y exige paths explícitos, sin discovery
durante runtime. Como mínimo:

- plan y cierre de `MAPPING-FEASIBILITY`, adjudicación y R544;
- contrato, generador, modelos y runner relacionales vigentes;
- config, compute contract y runtime del smoke;
- implementación y evidencia unit-base `K=64` de R354, ligada sólo como
  antecedente; diagnóstico/calibración base-weighted R550 y confirmación
  independiente `K=192` de este goal;
- schemas W49, checkpoints/split W51, utilidad W52, posterior W53/W54 y
  proposer/guards W56/W57;
- implementación de control matched W59 y manifests de sus fuentes;
- este plan, R547, su reauditoría independiente y las tres configs del freeze.

Una discrepancia de hash, source dirty o dependencia ausente produce
`SOURCE_BINDING_FAIL`. El preflight no actualiza hashes por sí solo.

## 7. Checkers y mutaciones

El checker independiente no importa el futuro runner de entrenamiento. Valida
JSON/schema, fuentes, fases, factoriales, budgets, controles, estimandos y
límites inferenciales. Predicados mínimos:

| Rama | Predicados |
|---|---|
| coordinación | `C1_SOURCE_BINDING`, `C2_BRANCH_SEPARATION`, `C3_PHASE_DAG`, `C4_REPLAY_CONTRACT`, `C5_NO_GPU_NO_PROMOTION`, `C6_READINESS_SEMANTICS` |
| relacional | `R1_PUBLIC_PRIVATE_SCHEMA`, `R2_MASTER_SPLIT`, `R3_ARM_PARITY`, `R4_DUAL_SOLVER_LOSS`, `R5_EXECUTOR_PARITY`, `R6_SOLVER_DENOMINATORS`, `R7_PATH_CONTROL_ROSTER`, `R8_WEIGHT_CONTROLS`, `R9_TOTAL_TARGET_SHUFFLE`, `R10_SEED_ESTIMAND`, `R11_BASE_WEIGHTED_K192_CONFORMANCE`, `R12_DECISION_TABLE` |
| set-valued | `S1_PHASE_SUPPORT`, `S2_LOGIT_TARGET_SEPARATION`, `S3_POSTERIOR_PARITY`, `S4_NATIVE_FIT_RECIPES`, `S5_HARD_POSTERIOR_BINDING`, `S6_CONTEXTUAL_RECIPE`, `S7_PROPOSER_GUARD_CREDIT`, `S8_TARGET_SHUFFLE_FOLDS`, `S9_MATCHED_CONTROL_TARGET_BLIND`, `S10_CELL_DUPLICATION`, `S11_ESTIMAND_ORDER`, `S12_DECISION_TABLE` |

Cada predicado tiene una fixture positiva y una mutación negativa con reason
code cerrado. Además se prueban: target o utilidad como input, split cruzado,
monitor abierto temprano, posterior con masa inválida, hard calculado desde el
threshold histórico, receta contextual distinta entre posteriors, peso crudo
confundido con peso normalizado, K64 unit-base presentado como base-weighted,
reutilización del draw R550 como confirmación, K distinto de `192`, retuning
después de una confirmación fallida,
IRLS con parámetros desiguales, path-control sin roster común, target-shuffle
por `cluster_id` o cruzando folds, control identidad, matching que lee target,
soporte unión en vez de intersección, mapa no byte-canónico, mutaciones aisladas
de `master_id`, `mechanism` y cualquier campo privado que alteren seed/tensor/
roster público, tabla de decisión incompleta, CI por masters presentado como
población de seeds, ranking cruzado de ramas, `READY_FOR_EXECUTION`,
`gpu_allowed:true`, `architecture_promoted:true` y cualquier campo GO/NO-GO.

El preflight produce dos copias deterministas en roots separados. Todos los
JSON científicos usan UTF-8, keys ordenadas, floats finitos, paths relativos y
sin timestamps. Runtime y RSS quedan fuera del byte replay.

## 8. Preservación obligatoria de la corrida sucesora

### Relacional

- config resuelta, fuentes, generator manifest, lineages y splits;
- checkpoints `last_epoch` e intermedios diagnósticos;
- optimizer state, historia, términos y gradientes por batch;
- inputs públicos, targets privados, relación, `raw_reliability`, pesos
  normalizados y controles por arista;
- potenciales, pesos base/finales, convergencia, iteraciones y condiciones por
  executor;
- métricas por vista/master/seed, bootstrap e índices de sanities.

### Set-valued

- attestations, manifests, split roles y logits ensemble/per-seed;
- targets privados, folds, mapa target-shuffled, singletons, calibrador pooled,
  theta y grid completo;
- masas `[token,15]`, riesgos `[token,policy,action]`, features y scores;
- estados proposer/guards verdaderos y controles, thresholds y support;
- acciones de las cuatro celdas, mapas matched target-blind, imposibilidades,
  métricas y bootstrap.

Los artefactos raw son canónicos. Informes agregados y vistas narrativas son
regenerables. No se purga ningún checkpoint o estado antes de la auditoría de
cierre del experimento.

## 9. Presupuesto y frontera GPU

Este goal de diseño tiene techo `<900 s` y `<2 GiB` para tests, dos preflights y
replay. El checker estructural no importa Torch. El probe numérico K192 sí
importa Torch con `CUDA_VISIBLE_DEVICES=''`, device fijado a `cpu` y sin llamar
APIs de discovery CUDA; no consulta dispositivos.

La proyección de la corrida relacional parte de evidencia histórica:

- smoke completo: `1320,29 s`, ocho brazos × dos seeds × diez épocas, pico
  `1,047 GiB`;
- contraste head-only con surrogate K64: `501,08 s`, cuatro brazos × dos seeds
  × dos objetivos × cinco épocas, pico `0,811 GiB`.

La corrida set-valued histórica más pesada del bracket necesitó alrededor de
`109,14 s` incluyendo preparación y `0,70 GiB` de RSS; Wave 54 tardó `0,319 s`
después de preparar logits. El preflight debe emitir rango bajo/central/alto,
supuestos y costo de replay, no un único número espurio.

Como todavía no existe el runner, la clasificación es una proyección trazable,
no un runtime medido de la corrida:

```text
PROJECTED_CPU_PROPORTIONATE       upper_bound < 12 h por primary+replay
PROJECTED_GPU_MATERIALLY_BETTER   upper_bound >= 12 h o memoria CPU >= 8 GiB
UNKNOWN_REQUIRES_RUNNER_PROBE     evidencia insuficiente para acotar
```

No se ejecuta el training probe en este goal. Si una rama queda
`PROJECTED_GPU_MATERIALLY_BETTER` o `UNKNOWN_REQUIRES_RUNNER_PROBE` por una
operación que CUDA resolvería materialmente mejor, se preserva en cola.
Mientras rija la
suspensión actual no se consulta ni carga GPU. Antes de cualquier uso futuro se
publicará evidencia durable y se avisará por Telegram objetivo, duración y
VRAM estimadas para esperar habilitación explícita.

## 10. Artefactos de este goal y cierre

Se materializarán:

```text
experiments/geometria_proporcional/configs/proportional_relational_native_freeze_v1.json
experiments/geometria_proporcional/configs/proportional_set_valued_native_freeze_v1.json
experiments/geometria_proporcional/configs/proportional_dual_native_freeze_v1.json
src/geometria_proporcional/proportional_graph_neural.py  # extensión base-weighted compatible
experiments/geometria_proporcional/check_proportional_dual_native_freeze.py
experiments/geometria_proporcional/run_proportional_dual_native_preflight.py
tests/test_proportional_dual_native_freeze.py
data/geometria_proporcional/proportional_dual_native_freeze_v1/run_a/
data/geometria_proporcional/proportional_dual_native_freeze_v1/run_b/
```

El goal cierra sólo si:

1. el plan recibe auditoría independiente y se resuelven findings materiales;
2. configs y sources quedan ligadas a un commit limpio;
3. todas las fixtures/mutaciones pasan;
4. dos preflights son byte-exactos en su clase científica;
5. un checker independiente adjudica validez del design freeze y costo
   proyectado de cada rama, nunca `READY_FOR_EXECUTION`;
6. una auditoría final recompone hashes, predicados, replay y límites;
7. la documentación registra qué corrida CPU puede abrirse después y qué queda
   en cola, sin promoción ni GO/NO-GO.

Al cerrar se abre automáticamente el siguiente goal finito autorizado por el
usuario: implementar y auditar el runner de la rama CPU-proporcionada con mayor
poder discriminante, antes de ejecutar entrenamiento; o detenerse y escalar si
la única continuación proporcionada requiere GPU.
