# Ola 58 — plan del diagnóstico abierto de clase de modelo

> **Estado:** `REVISED / AWAITING-REAUDIT / OPEN-DATA / HYPOTHESIS-GENERATING / CPU-ONLY`
> **Fecha:** 2026-09-04
> **Autoridad:** este diagnóstico puede orientar un protocolo futuro; no puede validar una arquitectura ni producir una decisión `GO/NO-GO`.

## Motivo

La Ola 57 dejó dos problemas distintos. El control shuffled fue parcialmente no
evaluable, pero aun sin ese control la política compuesta falló tres condiciones
contra el hard-set. Reparar sólo los shams produciría un control más completo
para una arquitectura que ya mostró beneficio medio pequeño, cesión de
compatibilidad e intervalos de cola no concluyentes.

Antes de consumir otro draw, este diagnóstico pregunta si el límite observado
proviene también de la clase lineal de los dos estimadores. Usa deliberadamente
los tres splits ya abiertos de la Ola 57, incluido el antiguo monitor, para
comparar modelos de baja capacidad y formulaciones de guard. Por esa razón
todo resultado será retrospectivo y adaptativo: sólo podrá justificar qué
candidata congelar en una realización futura independiente.

## Registro de adaptación ya ocurrida

Antes de escribir este plan se realizaron cuatro probes interactivos sobre los
artefactos abiertos. No se ocultarán ni se presentará la candidata resultante
como preregistrada:

1. se compararon labels de daño, pérdida de compatibilidad, pérdida de accuracy
   y breach de cola con regresiones lineales/logísticas;
2. se comparó selección secuencial contra búsqueda conjunta de umbrales;
3. se probaron variantes de compatibilidad con y sin `class_weight="balanced"`;
4. se comparó el baseline Ridge/Logistic con
   `HistGradientBoostingRegressor/Classifier` de capacidad acotada.

El mejor punto inspeccionado fue `HistGradientBoosting` para gain y daño. En el
monitor ya abierto pasó de hard-set
`accuracy=0.841231, compatible=0.940359, regret=0.122617,
worst_regret=0.385349` a
`accuracy=0.848720, compatible=0.947032, regret=0.109284,
worst_regret=0.360022`. Es una observación de diseño contaminada por selección
adaptativa, no una estimación generalizable.

## Pregunta diagnóstica

¿Una no linealidad tabular acotada recupera, dentro del draw abierto, una región
Pareto que los estimadores lineales no alcanzan, y la señal se atribuye al
proposer de gain, al guard de daño o a su interacción bajo selección conjunta?

## Fuentes congeladas

El coordinador verificará SHA-256 antes y después de la ejecución; el worker
recibirá copias read-only y no paths a los originales:

| Fuente | SHA-256 |
|---|---|
| `wave57.../fit.complete/analytics.complete/gate_fit_bundle.npz` | `eca613133e39a92d9c110c3d078856d7ec21e6a456bc7aa07769f64eaf04971c` |
| `wave57.../select.complete/analytics.complete/gate_select_bundle.npz` | `144f8cb2a479e39040653cff7cd66127c660e244533a2fcc5e46bf1dd0fda88d` |
| `wave57.../select.complete/analytics.complete/selection_arrays.npz` | `79b935fc32153e9c43022126d8d3f2670cae0654fe799d0fbb43b0de2e42647c` |
| `wave57.../adjudicate.complete/analytics.complete/sealed_monitor_bundle.npz` | `5c6043187bbf8f5cb7f94bc2fc2edf9aca5a0e18a1fcf1617f5670d0ade69bd3` |
| `wave57.../adjudicate.complete/analytics.complete/result_arrays.npz` | `8529c7787e74ad64de275d32cc8c52362283dcb41f7d81bae5ce027ba04e5064` |
| `wave52_policy_transport_v1/policy_manifest.json` | `f8a608d396ad48ba3b0336df2dc1955940515be0f51fa08328cd5ddb1e9a21e1` |
| `wave57_contextual_tail_guard_fresh.json` | `fb21a43cb4e356a7f10c293e5cb0c0fc037a8ecc206883e99122115b18812a83` |
| `src/geometria_proporcional/wave55_policy_bridge.py` | `5d0941ab57e969d12bd4a18624f4f76eb0542299a3b0f0ea2ffa258a6af951ad` |
| `src/geometria_proporcional/wave52_policy.py` | `14c6b33d972ee1254386b21e08f69554b9880a4cac98c2b85b00b8d3b7015b5a` |
| `src/geometria_proporcional/wave56_contextual_gate.py` | `474d3274b7d3c8ba3cba1d4f10b8b238119d58957520febbf24bf11de9ef665d` |
| `src/geometria_proporcional/wave57_tail_guard.py` | `059572890c71de4723b35d44d9c7c91d1698e5d515ba5d2551e971180b479a99` |
| `experiments/geometria_proporcional/run_wave56_retrospective.py` | `51703287087a7e2bfdf3f253416164a81aebe2835080d2e9edb831ed710d02da` |
| `experiments/geometria_proporcional/_wave57_phase_worker.py` | `35698118492bf412c5cf68e86c3e72e02dc02c597cc2dc95bfb0e701a5f47465` |

`wave57...` abrevia exclusivamente
`data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1/phases/`.
El manifest de salida conservará los paths completos, hashes y tamaños.

## Población, targets y métricas

Se mantiene la población primaria de la Ola 57: `NEAR_RIVAL` y cardinalidad
mayor o igual que dos. El fit usa sólo `gate_fit`; la búsqueda de umbrales usa
sólo `gate_select`; el antiguo `sealed_monitor` se lee al final, pero ya no es
lockbox y no recupera autoridad prospectiva.

Sea `o_t,p` la acción autorizada por el target, `h_t,p` la acción hard,
`a_t,p` la acción posterior, `r_h(t,p)` y `r_a(t,p)` sus regrets bajo la
utilidad congelada, y `epsilon=1e-12`. Sobre las mismas 17 features
inference-safe se reconstruyen seis targets por fila de desacuerdo:

- `gain(t,p) = r_h(t,p) - r_a(t,p)`;
- `harm(t,p) = 1[gain(t,p) < -epsilon]`;
- `compatibility_loss(t,p) = 1[target_t[h_t,p] = 1 and
  target_t[a_t,p] = 0]`;
- `posterior_incompatibility(t,p) = 1[target_t[a_t,p] = 0]`;
- `accuracy_loss(t,p) = 1[h_t,p = o_t,p and a_t,p != o_t,p]`;
- `tail_breach(t,p) = 1[r_a(t,p) > max_j r_h(t,j) + epsilon]`.

Todos se ajustan sólo en `primary ∩ disagreement`, con peso `1/d_t` por fila,
donde `d_t` es el número de políticas en desacuerdo del token. Los targets
binarios usan como positiva la clase de riesgo indicada y todos los guards
autorizan únicamente por score estrictamente menor que su threshold. La
mediación completa queda fijada como
`fila -> score -> máscara AND -> acción -> métrica token-wise`; ningún target
fila-wise se interpreta directamente como worst regret del sistema.

Accuracy, compatibilidad, regret y worst regret se recomputan mediante
`action_metric_arrays`, con las mismas 24 políticas, utilidad fija y penalidad
`1.25` de la Ola 57. El reporte separará siempre validation del monitor abierto.

## Ledger de probes ya inspeccionados

Los stdout interactivos no se preservaron como artefactos canónicos separados;
permanecen en el transcript de sesión. Por eso este ledger se declara
`HISTORICAL-PROBE / RAW-NOT-CANONICAL` y el ejecutor reproducirá todos sus IDs:

En esta tabla `Qp=[0.5,0.6,0.7,0.8,0.9,0.95,0.975]`,
`Qg8=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]` y
`Qg9=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]`. `L` significa Logistic
con contrato Wave 57, `HGBr` regressor HGB seed `5801`, `HGBh` classifier
seed `5802` y `HGBi` classifier seed `5803`. Un guion en seed significa que
el estimador es determinista sin `random_state`.

| ID único | Proposer | Guard 1 | Guard 2 / composición | Selector | Grilla | `class_weight` alternativo | Seeds |
|---|---|---|---|---|---|---|---|
| `P1-H` | Ridge score y threshold Wave 57 `0.37195414707872076` | L harm | — | `FIXED-W57-PROPOSER` | `Qg8` | null | — |
| `P1-C` | igual | L compatibility-loss | — | `FIXED-W57-PROPOSER` | `Qg8` | null | — |
| `P1-T` | igual | L tail-breach | — | `FIXED-W57-PROPOSER` | `Qg8` | null | — |
| `P1-HC` | igual | L harm | L compatibility-loss, AND | `FIXED-W57-PROPOSER` | `Qg8 × Qg8` | null | — |
| `P1-HT` | igual | L harm | L tail-breach, AND | `FIXED-W57-PROPOSER` | `Qg8 × Qg8` | null | — |
| `P1-CT` | igual | L compatibility-loss | L tail-breach, AND | `FIXED-W57-PROPOSER` | `Qg8 × Qg8` | null | — |
| `P1-HCT` | igual | L harm | L compatibility-loss AND L tail-breach | `FIXED-W57-PROPOSER` | `Qg8³` | null | — |
| `P2-H-J` | Ridge | L harm | — | `JOINT` | `Qp × Qg8` | null | — |
| `P2-H-R` | Ridge | L harm | — | `JOINT_SHARD_ROBUST` | `Qp × Qg8` | null | — |
| `P2-T-J` | Ridge | L tail-breach | — | `JOINT` | `Qp × Qg8` | null | — |
| `P2-T-R` | Ridge | L tail-breach | — | `JOINT_SHARD_ROBUST` | `Qp × Qg8` | null | — |
| `P2-MAXHT-J` | Ridge | `max(L harm,L tail)` | un threshold común | `JOINT` | `Qp × Qg8` | null | — |
| `P2-MAXHT-R` | Ridge | `max(L harm,L tail)` | un threshold común | `JOINT_SHARD_ROBUST` | `Qp × Qg8` | null | — |
| `P3-C-N` | Ridge | L harm | L compatibility-loss, AND | `JOINT` | `Qp × Qg8 × Qg9` | null | — |
| `P3-C-B` | Ridge | L harm | L compatibility-loss, AND | `JOINT` | `Qp × Qg8 × Qg9` | balanced sólo guard 2 | — |
| `P3-I-N` | Ridge | L harm | L posterior-incompatibility, AND | `JOINT` | `Qp × Qg8 × Qg9` | null | — |
| `P3-I-B` | Ridge | L harm | L posterior-incompatibility, AND | `JOINT` | `Qp × Qg8 × Qg9` | balanced sólo guard 2 | — |
| `P3-A-N` | Ridge | L harm | L accuracy-loss, AND | `JOINT` | `Qp × Qg8 × Qg9` | null | — |
| `P3-A-B` | Ridge | L harm | L accuracy-loss, AND | `JOINT` | `Qp × Qg8 × Qg9` | balanced sólo guard 2 | — |
| `P4-RL` | Ridge | L harm | — | `JOINT` | `Qp × Qg9` | null | — |
| `P4-RLI` | Ridge | L harm | L posterior-incompatibility, AND | `JOINT` | `Qp × Qg9²` | null | — |
| `P4-HH` | HGBr | HGBh harm | — | `JOINT` | `Qp × Qg9` | null | `5801,5802` |
| `P4-HHI` | HGBr | HGBh harm | HGBi posterior-incompatibility, AND | `JOINT` | `Qp × Qg9²` | null | `5801,5802,5803` |
| `P4-HLI` | HGBr | L harm | L posterior-incompatibility, AND | `JOINT` | `Qp × Qg9²` | null | `5801` |

El reporte conservará resultados de todos esos IDs aunque sean redundantes,
desfavorables o no evaluables. `MAXHT` autoriza cuando
`max(p_harm,p_tail) < threshold`; las demás letras múltiples usan AND de
thresholds independientes.

## Roster canónico de atribución

Separado del ledger histórico, el roster canónico será un factorial completo:

| Factor | Niveles |
|---|---|
| proposer de gain | `RIDGE`, `HGB` |
| clase del estimador de riesgo | `LOGISTIC`, `HGB` |
| guard set | `HARM`, `INCOMPATIBILITY`, `HARM_AND_INCOMPATIBILITY` |
| selector | `SEQUENTIAL`, `JOINT`, `JOINT_SHARD_ROBUST` |

Esto produce `2 × 2 × 3 × 3 = 36` candidatos activos, con ID estable
`C-{proposer}-{risk_model}-{guard_set}-{selector}`. Las ablaciones
`HARM`, `INCOMPATIBILITY` y su conjunción aíslan el segundo guard; el 2×2
proposer×risk-model atribuye clase de modelo. Los targets
`compatibility_loss`, `accuracy_loss` y `tail_breach`, incluidos sus brazos
balanced ya inspeccionados, permanecen en el ledger exploratorio pero no
arbitran la nominación canónica. Ningún candidato intentado desaparece.

## Contratos de modelo

### Lineales

- proposer Ridge: `alpha=1.0`, `solver="svd"`, `fit_intercept=true`,
  `copy_X=true`, `tol=1e-4`, `max_iter=null`, `positive=false`,
  `random_state=null`;
- guards Logistic: `C=1.0`, `solver="lbfgs"`, `max_iter=2000`,
  `tol=1e-10`, `penalty="deprecated"`, `dual=false`,
  `fit_intercept=true`, `intercept_scaling=1`, `l1_ratio=0.0`,
  `n_jobs=null`, `random_state=null`, `warm_start=false`, `verbose=0`,
  `class_weight=null` salvo los brazos declarados, y escalado ponderado de
  Wave 57;
- para `compatibility_loss`, `posterior_incompatibility` y `accuracy_loss`,
  brazos separados con `class_weight=null` y `class_weight="balanced"`.

### No lineales acotados

- proposer `HistGradientBoostingRegressor`;
- guards `HistGradientBoostingClassifier` para `harm` y
  `posterior_incompatibility`;
- contrato común: `max_iter=100`, `learning_rate=0.05`,
  `max_leaf_nodes=7`, `max_depth=null`, `min_samples_leaf=20`,
  `l2_regularization=1.0`, `max_bins=255`, `max_features=1.0`,
  `categorical_features="from_dtype"`, `monotonic_cst=null`,
  `interaction_cst=null`, `early_stopping=false`, `n_iter_no_change=10`,
  `validation_fraction=0.1`, `scoring="loss"`, `tol=1e-7`, `verbose=0`,
  `warm_start=false`;
- regressor: `loss="squared_error"`, `quantile=null`;
- classifier: `loss="log_loss"`, `class_weight=null`;
- `random_state=5801` proposer, `5802` daño y `5803` incompatibilidad;
- scikit-learn debe ser exactamente `1.8.0`.

Ridge, Logistic y HGB reciben el mismo `sample_weight=1/d_t`. Los brazos
`class_weight="balanced"` del ledger conservan además esa multiplicación y se
marcan `OBJECTIVE-CHANGED`, no como comparación pura de clase de modelo. Cada
fit preserva kwargs efectivos, clases, número de iteraciones y scores de los
tres splits; una sola clase, no convergencia o estado no finito produce
`NOT_EVALUABLE` para ese candidato, nunca su eliminación.

Ridge y Logistic reciben el `WeightedScaler` de Wave 57 ajustado sólo en train;
HGB recibe las 17 features `float64` crudas, sin scaler. Para Ridge y Logistic,
media, escala, coeficientes e intercepto constituyen estado reconstruible. Para
HGB no se serializa un pickle ni se promete inferencia fuera de estos datos:
la autoridad de reanálisis son los scores `float64` preservados para train,
validation y monitor junto con kwargs, `n_iter_`, clases y hashes de fit. El
replay independiente vuelve a ajustar HGB y exige igualdad exacta de esos
scores. Los tests hablan de autoridad de score HGB, no de reconstrucción del
árbol desde un formato privado de sklearn.

Este inventario no incluye búsqueda de hiperparámetros. La comparación de clase
de modelo queda confundida con este único contrato HGB y se declarará así.

## Replay legacy, máscaras y selección

### Replay exacto Wave 57

Un brazo aislado `LEGACY-W57` reajusta Ridge y Logistic exclusivamente desde
`gate_fit_bundle.npz`, puntúa y selecciona exclusivamente desde
`gate_select_bundle.npz`, congela esa selección y recién entonces evalúa
`sealed_monitor_bundle.npz`. Usa exactamente los cuantiles, restricciones,
orden secuencial y tie-breaks de Wave 57, sin pasar por la grilla ampliada.
`selection_arrays.npz` y `result_arrays.npz` permanecen inaccesibles durante
esa recomputación y se abren sólo después como referencias. Deben igualarse
estados lineales reconstruibles, scores, thresholds, máscaras, acciones,
resúmenes y deltas. Sólo ese brazo se denomina reproducción exacta.

### Selectores diagnósticos

Los cuantiles de proposer serán
`[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]`; los cuantiles de cada guard serán
`[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Los thresholds se calculan
con `method="higher"` sobre filas elegibles de `gate_select`.

Para cada threshold proposer, la máscara es
`proposal = disagreement AND proposer_score > proposer_threshold`. El soporte
no terminal mínimo es `40` tokens propuestos. Cada threshold de guard se
calcula independientemente sobre todas las filas propuestas de la población de
selección, nunca después de aplicar otro guard. Para cualquier lista de
`k >= 1` guards, la autorización es
`proposal AND AND_i(score_i < threshold_i)`. La grilla es el producto completo
`q_proposer × product_i(q_guard_i)`, sin dependencia del orden de aplicación;
esto incluye explícitamente `Qg8³` para `P1-HCT`. `MAXHT` cuenta como un único
score compuesto y usa un único threshold.

Se reportarán tres selectors:

1. secuencial, que fija primero el proposer y luego el guard como en Wave 57;
2. conjunto, que enumera el producto completo y minimiza regret entre celdas
   evaluables y factibles;
3. conjunto robusto, que añade las mismas restricciones en ambos shards.

La factibilidad conserva los márgenes de Wave 57 contra hard en validation:

- accuracy no menor que hard menos `0.01`;
- compatibilidad no menor que hard;
- worst regret no mayor que hard más `0.01`;
- al menos `25` tokens autorizados.

En el selector robusto cada shard exige además `40` tokens totales, `120`
filas y `50` tokens con desacuerdo, `20` tokens propuestos y `12` autorizados;
la factibilidad local se compara contra el hard del mismo shard.

Cada selector incluye un único terminal `HARD_ONLY`, evaluable y factible con
soporte de acción cero. Si no hay celda activa factible, selecciona ese
terminal. Los thresholds duplicados permanecen como celdas separadas.

Para celdas activas, el orden total es:

1. menor regret, con `atol=1e-12` para formar el empate;
2. menos filas autorizadas;
3. menos tokens autorizados;
4. menos filas propuestas;
5. menos tokens propuestos;
6. mayor threshold y luego mayor cuantil proposer;
7. thresholds de guard de menor a mayor, seguidos por sus cuantiles de menor a
   mayor, en el orden canónico de targets presentes:
   `harm, compatibility_loss, posterior_incompatibility, accuracy_loss,
   tail_breach, max_harm_tail`;
8. índice lexicográfico de la celda.

El terminal se ordena antes que cualquier celda activa factible dentro del
mismo empate de regret; fuera de ese empate gana quien tenga menor regret. El
selector secuencial usa primero las restricciones proposer de Wave 57
(accuracy y compatibilidad), congela esa máscara y aplica luego el orden del
guard. `JOINT` y `JOINT_SHARD_ROBUST` aplican las tres restricciones al sistema
compuesto.

Dentro del paso proposer secuencial, el orden después del empate de regret es:
menos filas propuestas, menos tokens propuestos, mayor threshold, mayor
cuantil e índice de celda. Si gana `HARD_ONLY`, no se ajusta ningún threshold
de guard para esa celda y el sistema permanece hard.

Los shards usan exactamente
`least_significant_bit_of_SHA256(pair_token || "wave57-shard")`. Esto no
reemplaza un intervalo ni convierte el monitor abierto en test.

## Comparaciones mínimas

El reporte debe incluir, como mínimo:

1. `LEGACY-W57` exacto;
2. todos los IDs `P1`–`P4` del ledger histórico;
3. los `36` IDs del factorial canónico, sin podar resultados;
4. hard-set y oracle-positive-gain sólo como referencias de piso y headroom.

Para cada brazo se conservarán thresholds, soporte, deltas contra hard,
validation, shards, monitor abierto y matriz de overrides. No se calculará un
promedio selectivo que excluya brazos desfavorables.

## Artefactos

Salida canónica:
`data/geometria_proporcional/wave58_open_model_class_diagnostic_v1/`.

Debe contener:

- `config.json` con el inventario completo y hashes de entrada;
- `fit/` con estados lineales, metadata HGB, scores de train y freeze previo a
  validation;
- `select/` con las grillas completas, elecciones y freeze previo al monitor;
- `analysis.json` y `REPORT.md`;
- `scores_and_masks.npz` con scores, targets, thresholds y máscaras por brazo;
- `manifest.json` con hashes y tamaños;
- salida hermana canónica
  `data/geometria_proporcional/wave58_open_model_class_diagnostic_v1_replay/`.

Los JSON científicos no contienen timestamps, host ni path de salida. El
replay parte de un directorio vacío, repite fit y selección y debe igualar
byte a byte `config.json`, `fit/model_states.json`, `fit/fit_freeze.json`,
`select/selection_grids.json`, `select/selection_freeze.json`, `analysis.json`
y `REPORT.md`, y array por array `fit/model_scores.npz`,
`select/selection_arrays.npz` y `scores_and_masks.npz`. Sólo `runtime.json`,
que conserva tiempos y RSS, y el propio `manifest.json`, cuyos inventarios
identifican primary/replay, quedan excluidos de igualdad byte pero no de
inventario. El manifest enumera exactamente esas exclusiones. Los artefactos
se preservan para reanálisis sin re-fit.

El coordinador copia inputs a un staging temporal, fija archivos `0444` y
directorios `0555`, ejecuta el worker como UID/GID `65534` con un output
separado escribible, y vuelve a verificar los hashes originales antes de
publicar atómicamente. El worker no recibe los paths originales. El manifest
conserva hashes del nuevo coordinador, worker, tests, config y commit Git.

Después de que el plan quede aceptado y ya no se edite, la config de ejecución
ligará además el SHA-256 de este plan y del informe de reauditoría que lo acepte.
Esos dos hashes, los del código Wave 58 y el commit de implementación serán
precondiciones de ejecución y quedarán repetidos en el manifest.

La ejecución tiene tres invocaciones físicas. `FIT` recibe únicamente train y
publica su freeze; `SELECT` recibe validation más el freeze FIT y publica su
freeze; `MONITOR` recibe el antiguo monitor sólo después de verificar ambos
freezes. Cada staging enumera un allowlist y rechaza inputs de fases futuras.
El replay repite las tres invocaciones desde cero.

## Tests y auditoría

Antes de ejecutar sobre datos reales deben pasar tests que prueben:

1. rechazo de cualquier hash de entrada divergente;
2. inventario exacto de candidatos, grids y seeds;
3. targets de compatibilidad, accuracy y cola con fixtures manuales;
4. fórmula, dominio, epsilon y polaridad de los seis targets;
5. `sample_weight` idéntico en todos los modelos y multiplicación explícita de
   `class_weight` sólo en brazos `OBJECTIVE-CHANGED`;
6. contrato completo HGB, versión sklearn y autoridad exacta de scores;
7. ponderación y escalado idénticos al contrato;
8. identidad exacta y prioridad definida del hard terminal;
9. selector secuencial, conjunto y shard-robust, incluida la conjunción AND
   n-aria, producto cartesiano, mínimos, thresholds duplicados y tie-break
   total; un fixture específico debe cubrir `P1-HCT`, `Qg8³` y su desempate;
10. una sola clase, no convergencia y ninguna celda activa factible;
11. reproducción exacta Wave 57 fuera de la grilla ampliada;
12. monitor ausente del staging hasta que FIT/SELECT estén congelados;
13. conservación separada de validation y monitor abierto;
14. hashes pre/post, inputs read-only y ausencia de paths originales en worker;
15. replay independiente desde vacío, no copia de outputs;
16. ausencia de imports o uso CUDA, `CUDA_VISIBLE_DEVICES=''` y límite efectivo
    de cuatro threads mediante variables BLAS/OpenMP y `threadpoolctl`.

Se preservarán deltas pareados e IC95 bootstrap por `pair_token` con `5000`
réplicas, seed `5807`, generador `numpy.random.PCG64`, orden lexicográfico de
`pair_token` e intervalo percentil `[2.5,97.5]`. Cada split construye una única
matriz de índices sobre toda su población primaria, con rango `[0,n_tokens)`,
y la reutiliza para todos los brazos y métricas, garantizando pairing. Los
tests verifican determinismo, forma, rango, orden y reutilización. Los
intervalos quedan separados por split y etiquetados
`CONDITIONAL / ADAPTIVE / POST-SELECTION`. No estiman variación entre draws ni
arbitran la nominación.

Una instancia independiente auditará este plan antes de implementación y el
resultado completo después de la ejecución. Todo finding que afecte validez,
trazabilidad o capacidad de avanzar se corregirá y reauditará.

## Criterio para diseñar el prospectivo

El diagnóstico no tiene umbral de éxito científico. Para hacer auditable la
nominación de diseño, se define dominancia sobre las ocho coordenadas
`validation × monitor` de accuracy, compatibilidad, regret y worst regret: una
celda domina a otra si no es peor en ninguna coordenada por más de `1e-12` y es
mejor en al menos una por más de `1e-12`, con direcciones mayor-mejor para las
dos primeras y menor-mejor para las dos últimas.

Un candidato activo es elegible para nominación si tiene al menos `25` tokens
autorizados en ambos splits, cumple en ambos los márgenes de factibilidad de
Wave 57 y pertenece al frente no dominado. Se nomina cero candidatos si el
conjunto es vacío. Si no lo es, se nomina exactamente uno por este orden total:

1. menor regret en monitor abierto;
2. menor regret en validation;
3. menor worst regret en monitor;
4. menor worst regret en validation;
5. mayor compatibilidad en monitor y luego validation;
6. mayor accuracy en monitor y luego validation;
7. menos guards, Ridge antes que HGB y Logistic antes que HGB;
8. ID canónico.

La nominación debe llevar literalmente
`OPEN-DATA / ADAPTIVE / SELECTED-AFTER-MONITOR-INSPECTION`. Es una
descomposición dentro de este draw, no atribución causal generalizable. Los
shams de la futura realización deberán construirse
por asignación condicional que alcance por diseño el máximo Hamming ponderado
factible, con contrato y mínimos congelados antes del draw. Ese mecanismo no se
añade retrospectivamente a este diagnóstico.

## Recursos

Ejecución local CPU. El runner fija `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` y usa
`threadpoolctl(limits=4)`. Los cinco NPZ suman aproximadamente `54.17 MiB`
expandidos; el roster canónico materializa sólo scores por modelo y máscaras
seleccionadas, no cada máscara de la grilla. La estimación es menor a dos
minutos y menor a `2 GiB` de RSS, registrada en `runtime.json`. No hay
justificación para GPU: si la implementación real
contradijera esta estimación, se detendrá antes de usar CUDA y se solicitará
habilitación explícita por Telegram.
