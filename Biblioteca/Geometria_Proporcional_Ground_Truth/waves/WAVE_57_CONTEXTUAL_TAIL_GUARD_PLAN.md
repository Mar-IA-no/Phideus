# Ola 57 — plan prospectivo para propuesta de valor y guard de daño

> **Estado:** `REVISED-AFTER-R395 / FOR-FOCAL-REAUDIT / PRE-IMPLEMENTATION / PRE-KEY-DRAW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-04
> **Antecedente inmediato:** `WAVE_56_STAGE1_PROSPECTIVE_CLOSED.md`

## Pregunta experimental

La Ola 56 mostró una señal contextual real pero insuficiente: el predictor de
gain redujo regret medio frente a hard, advantage-only y shuffled, mientras
autorizaba más reemplazos perjudiciales que beneficiosos y empeoraba el worst
regret. El resultado distingue dos operaciones que el gate anterior había
reunido en un único score. Proponer una acción por su valor esperado no equivale
a autorizarla cuando el costo de equivocarse se concentra en una cola.

La Ola 57 pregunta si una interfaz de dos cabezas, ambas de baja capacidad y
alimentadas sólo por observables disponibles en inferencia, puede conservar la
mejora media del proposer y filtrar una fracción suficiente de sus daños para
recuperar simultáneamente accuracy y worst regret. La primera cabeza estima
gain esperado. La segunda estima la probabilidad de que reemplazar la acción
dura sea perjudicial. La acción bayesiana se ejecuta sólo si ambas condiciones
la habilitan.

Es un contraste de política sobre la misma ley sintética, el mismo encoder y el
mismo posterior conjunto. No adjudica geometría proporcional, autoridad física,
utilidad natural, transporte externo ni PPU. Tampoco promueve una arquitectura
ni decide `GO/NO-GO`.

## Hipótesis y falsación

**Hipótesis.** La señal que reduce regret medio y la señal que localiza el daño
no son idénticas. Una cabeza explícita de riesgo puede retirar overrides con
probabilidad alta de gain negativo sin destruir la ventaja media obtenida por
el proposer.

**Falsación informativa.** La hipótesis pierde apoyo si la interfaz conjunta no
supera al proposer solo en accuracy y worst regret, si para hacerlo retorna a
`hard_only`, si pierde la reducción de regret frente a hard, o si no supera a
un guard de igual capacidad entrenado con labels de daño barajados. Un resultado
negativo se limita a esta familia lineal, esta ley y este catálogo de políticas;
no constituye un techo para toda forma de estimación de cola.

## Arquitectura congelable

Para cada pair token `t` y política `p`, sea `g_tp` el gain realizado de
reemplazar la acción dura por la acción bayesiana. Las diecisiete features de
Ola 56 se conservan sin adiciones ni reordenamiento:

```text
x_tp
  ├─ Ridge(alpha=1.0)                  -> mu_hat_tp
  └─ LogisticRegression(C=1.0, L2)    -> p_harm_tp

propose_tp  := disagreement_tp and mu_hat_tp > tau_mu
authorize_tp := propose_tp and p_harm_tp < tau_harm
action_tp    := posterior_tp if authorize_tp else hard_tp
```

La cabeza de valor usa target `g_tp`, objetivo MSE, escalado ponderado y solver
SVD. La cabeza de riesgo usa target binario `y_tp=1[g_tp < -1e-12]`, log-loss,
escalado ponderado y `LogisticRegression(C=1.0, l1_ratio=0.0, dual=false,
solver="lbfgs", class_weight=None, fit_intercept=true, max_iter=2000,
tol=1e-10, warm_start=false)`. En scikit-learn `1.8.0`, `l1_ratio=0.0`
especifica penalización L2 sin depender del parámetro `penalty` deprecado. La
clase positiva debe ser exactamente daño: `classes_ == [0,1]` y la columna uno
de `predict_proba` produce `p_harm`.

No se seleccionan hiperparámetros de modelos. Ambas cabezas se ajustan sólo
sobre filas de desacuerdo de la población primaria de `train`, con peso
`1/d_t`, donde `d_t` es la cantidad de políticas en desacuerdo del token. La
implementación verifica kwargs efectivos, uso de `sample_weight`, ausencia de
`ConvergenceWarning`, finitud de scaler/coeficientes/intercept/probabilidades y
reconstrucción de `p_harm` desde el estado persistido. Una sola clase, no
convergencia o estado no finito produce `FIT_NOT_EVALUABLE` terminal; no habilita
reintento con otro solver, tolerancia, regularización o número de iteraciones.

Los signos son estrictos. Un score igual a `tau_mu` no propone; una probabilidad
igual a `tau_harm` no autoriza. `hard_only` reproduce byte por byte la acción
dura. El target de daño, el gain y toda métrica dependen de truth y nunca forman
parte del diseño de inferencia.

## Control de capacidad igualada

Cinco guards sham conservan exactamente features, escalador, familia,
regularización, solver, presupuesto de ajuste y selector del guard principal.
Sólo cambia su target: la etiqueta binaria de daño se permuta dentro de cada
estrato `(policy_index, disagreement_count)` con seeds
`{57031,57032,57033,57034,57035}`. El null es una permutación condicional exacta,
no una destrucción total de labels: conserva la prevalencia dentro de cada
estrato y sólo rompe correspondencia donde el estrato contiene ambas clases.
Cada réplica preserva mapping, target, fracción de índices permutables, fracción
de mapping no identidad, Hamming real global y ponderado, Hamming por estrato,
número y peso de estratos mixtos/homogéneos y hashes de mapping/target. Para ser
evaluable debe alcanzar Hamming global y ponderado `>=0.25`; el umbral se fija a
partir del rango histórico ya abierto `0.327–0.366` global y
`0.318–0.359` ponderado, no del draw futuro.

El proposer se selecciona una sola vez y su `tau_mu` y su máscara de propuestas
quedan congelados antes de seleccionar cualquier guard. El guard verdadero y
los cinco shams reciben exactamente esa misma máscara y seleccionan únicamente
`tau_harm`. Así, el contraste contra `mean_proposer_only` mantiene fijas las
propuestas, mientras el contraste contra los shams iguala segunda cabeza,
presupuesto y búsqueda. El contraste primario usa el promedio predeclarado de
las cinco réplicas; cada una permanece visible.

## Brazos

1. `hard_set_policy`: ancla que nunca reemplaza;
2. `pure_joint_full`: acción bayesiana sin gate;
3. `mean_proposer_only`: Ridge de gain con selector unidimensional;
4. `mean_plus_harm_guard`: interfaz principal de dos cabezas;
5. `mean_plus_shuffled_harm_guard`: cinco controles de capacidad igualada;
6. `advantage_only_value_gate`: control escalar aprendido heredado;
7. `oracle_positive_gain`: techo diagnóstico, nunca seleccionable.

El gate escalar de Ola 55 se conserva sólo como diagnóstico secundario porque
Ola 56 ya estableció que eligió identidad. No participa en el patrón conjunto
principal. Ningún brazo recibe features nuevas ni reentrena encoder, posterior
o políticas.

## Población y realización fresca

La población primaria permanece `NEAR_RIVAL` con cardinalidad verdadera mayor
o igual que dos. La cardinalidad sólo define el estrato de análisis; no se
incorpora como feature. Las veinticuatro políticas de un pair token permanecen
juntas en ajuste, selección, shards y bootstrap.

La corrida genera una realización nueva del protocolo
`wave49-relational-benchmark-v2` mediante tres claves nuevas extraídas después
del freeze del paquete. Los roles físicos son:

| Split físico | Rol | Autoridad |
|---|---|---|
| `train` | `gate_fit` | ajusta proposer, guard y cinco shams |
| `val` | `gate_select` | selecciona proposer, luego guards, y prueba estabilidad |
| `lockbox` | `sealed_monitor` | adjudica una sola vez sin reselección |

Los tres roles deben ser disjuntos por `pair_token` entre sí y respecto de los
artefactos analíticos de Olas 54–56. No hay redraw. El monitor abierto de Ola 56
sirve como antecedente para formular la hipótesis, pero no aporta filas, labels,
thresholds ni decisiones al nuevo ajuste.

## Selección congelada

La selección ocurre en dos pasos y no vuelve atrás. Primero, el proposer evalúa
los cuantiles de `mu_hat` `{0.50,0.60,0.70,0.80,0.90,0.95,0.975}` y
`hard_only`. Usa sólo las restricciones heredadas de accuracy y compatibilidad,
minimiza regret y congela una única `tau_mu`. Si selecciona `hard_only` o su
máscara no alcanza los mínimos de tokens propuestos, Stage SELECT termina
`SELECT_NOT_EVALUABLE` y el monitor permanece sellado: no existe un proposer
sobre el cual atribuir el efecto incremental del guard.

Segundo, con `tau_mu` y la máscara ya inmutables, el guard verdadero y cada sham
evalúan sólo cuantiles de aceptación por bajo riesgo
`{0.10,0.20,0.30,0.40,0.50,0.60,0.80}` y `hard_only`. Cada threshold se calcula
sobre `p_harm` únicamente entre las filas propuestas fijas, con
`method="higher"`; se autoriza estrictamente por debajo. Un candidato que no
alcance los mínimos de tokens autorizados queda `NOT_EVALUABLE`, sin relajar el
umbral.

Para ser factible, un operating point debe satisfacer en `guard_select`:

- `accuracy >= accuracy_hard - 0.01`;
- `compatible >= compatible_hard`;
- `worst_regret <= worst_regret_hard + 0.01`.

Entre candidatos factibles se minimiza regret medio. El proposer enumera
cuantiles en el orden escrito y `hard_only` al final; dentro de `1e-12`, su clave
total es menor cantidad de filas propuestas, menor cantidad de tokens
propuestos, mayor `tau_mu`, mayor `q_mu` y por último índice canónico de celda.
Cada guard enumera cuantiles en el orden escrito y `hard_only` al final; dentro
de `1e-12`, su clave total es menor cantidad de filas autorizadas, menor cantidad
de tokens autorizados, menor `tau_harm`, menor `q_harm` e índice canónico de
celda. `hard_only` tiene threshold nulo tipado, cero acciones e índice terminal;
es siempre factible. El proposer usa sólo las dos primeras restricciones,
reproduciendo su interfaz previa; el guard añade la tercera.

Cada sham recorre la misma grilla unidimensional y las mismas tres restricciones
que el guard verdadero sobre la máscara común. En cada uno de dos shards
deterministas por `pair_token`, salt `wave57-shard`, se repite la secuencia
completa: seleccionar una vez el proposer dentro del shard, congelar su máscara
y seleccionar luego cada guard. Modelos, grids y orden total no cambian.

## Mínimos y cierres no evaluables

Antes de ajustar se exigen en población primaria:

- `gate_fit`: al menos `100` tokens primarios, `400` filas de desacuerdo, `120`
  tokens con algún desacuerdo, `80` tokens que aporten alguna label perjudicial
  y `50` que aporten alguna label no perjudicial;
- `gate_select`: al menos `80` tokens primarios, `300` filas de desacuerdo y
  `120` tokens con algún desacuerdo;
- proposer seleccionado en SELECT completo: al menos `40` tokens con alguna
  propuesta; cada candidato de guard: al menos `25` tokens con alguna
  autorización;
- cada shard de selección: al menos `40` tokens primarios, `120` filas de
  desacuerdo, `50` tokens con desacuerdo, `20` con propuesta y `12` con
  autorización por candidato;
- `sealed_monitor`: al menos `100` tokens primarios, `300` filas de desacuerdo y
  `120` tokens con algún desacuerdo;
- cada shuffle: fracción permutable mínima `0.80` y Hamming global y ponderado
  mínimos `0.25`;
- soporte ausente heredado: `30` tokens por cada set individual.

Los mínimos token-wise se fijan a partir de la realización ya abierta de Ola
56, que produjo respectivamente `165/164/165` tokens primarios con desacuerdo
en FIT/SELECT/monitor; FIT tuvo `119` tokens con alguna label perjudicial y `70`
con alguna no perjudicial. R395 consignó `189` tokens FIT, pero la recomputación
exacta de `primary & disagreement.any(axis=1)` sobre el NPZ y hash que el propio
informe cita da `165`; la revisión usa el conteo reproducible.

Si falla un mínimo global de fit o select, el proceso termina antes de abrir el
split siguiente. Si un sham no alcanza sus mínimos de permutación, sólo ese
control y las condiciones que lo requieren quedan `NOT_EVALUABLE`. Si el
monitor no alcanza sus mínimos, se preserva el paquete sin emitir el patrón
agregado. Una política congelada que transporta cero overrides al monitor sí es
un resultado evaluable de transporte, siempre que el monitor alcance los
mínimos de tokens y desacuerdos; no se reabre selección.

Para los cinco support sets heredados se materializa `set_index` y se calculan
conteos, estado, summaries, contrastes e índices bootstrap por set. Cada uno
queda `NOT_EVALUABLE` por separado si tiene menos de `30` tokens. La unión puede
reportarse como diagnóstico, pero nunca autoriza un set individual ni entra en
el patrón conjunto.

## Patrón prospectivo predeclarado

Para cada token `t` y las 24 políticas fijas se definen:

```text
accuracy_t       = (1/24) sum_p 1[action_tp = oracle_tp]
compatible_t     = (1/24) sum_p compatible_tp
regret_t         = (1/24) sum_p regret_tp
worst_regret_t   = max_p regret_tp
```

`worst_regret` es, por tanto, la media entre tokens del peor regret dentro del
catálogo fijo de 24 utilidades; no es un máximo global, un cuantil entre tokens
ni CVaR. Cada contraste poblacional es la media de los deltas token-wise sobre
la población primaria del monitor.

Para el control sham se evalúa primero cada una de las cinco políticas sham por
separado. Después, para cada token y cada métrica ya calculada, se promedian las
cinco réplicas; nunca se promedian acciones antes de evaluar ni se toma un nuevo
máximo después de mezclar shams.

Se generan `5000` bootstraps pareados de tokens con `PCG64`, seed `5707`, orden
lexicográfico e intervalo percentil `[2.5,97.5]`. Los IC95 cuantifican muestreo
del monitor condicionado al FIT, SELECT, operating points y cinco permutaciones
observados. No incluyen la variación de regenerar y reajustar train/val; el
replay verifica determinismo y no constituye una réplica estadística. El patrón
conjunto queda observado sólo si se cumplen las seis condiciones:

1. regret principal menos hard `<= -0.01` y límite superior IC95 `< 0`;
2. accuracy principal menos hard con límite inferior IC95 `>= -0.01`, y
   compatibilidad con límite inferior IC95 `>= 0`;
3. worst regret principal menos hard con media `<= 0` y límite superior IC95
   `<= 0`;
4. frente al proposer de máscara exactamente fija, accuracy mejora con límite
   inferior IC95 `> 0` y worst regret mejora con límite superior IC95 `< 0`,
   mientras regret no empeora por más de `0.005` (`IC95 superior de
   principal-proposer <= 0.005`);
5. frente al promedio de los cinco shams, regret y worst regret mejoran con
   límites superiores IC95 `< 0`;
6. ambos shards seleccionan proposer y guard no `hard_only`, conservan los
   signos de regret/accuracy/worst-regret frente a hard, y el replay exacto
   satisface su matriz completa.

Las seis condiciones son diagnósticas. Su conjunción no es una decisión de
promoción; su falla no refuta arquitecturas no lineales, cuantiles explícitos ni
otras leyes. También se reportan sin autoridad decisoria: curvas completas,
frecuencia y magnitud de overrides beneficiosos/perjudiciales, calibración por
deciles de `p_harm`, Brier/log-loss, slices observacionales, políticas
individuales, support sets ausentes y techo oracle.

## Frontera pre-oracle y máquina de estados

La preparación y ejecución heredan la frontera física ya auditada en Ola 56:
escrow atómico root-only, freeze público previo al generador, truth sellada,
inferencia con UID/GID `65534`, runtime staged no escribible, materializador
split-scoped y worker analítico sin privilegios. El coordinador no carga truth
ni calcula métricas.

La máquina de estados es:

```text
PREPARED
  -> FIT_PENDING -> FIT_COMPLETE | FIT_NOT_EVALUABLE
  -> SELECT_PENDING -> SELECT_COMPLETE | SELECT_NOT_EVALUABLE
  -> ADJUDICATE_PENDING -> COMPLETE | MONITOR_NOT_EVALUABLE
```

Cada fase escribe bajo `.pending`, registra journal e inventarios, hace `fsync`
y promueve con `os.replace`. Antes de reanudar valida hashes de inputs y outputs
existentes. Después de abrir labels, ningún cambio de código, config, plan,
estimando, estimador, selector, seed o criterio puede adjudicar ese draw. Un bug
que exija un delta invalida la realización para inferencia y obliga a preservar
el fallo.

Un fallo de ajuste predeclarado —una clase, no convergencia o estado no finito—
promueve FIT a `FIT_NOT_EVALUABLE` con diagnóstico y preserva el material ya
abierto. Un error de implementación no se recodifica como no evaluabilidad: deja
el intento fallido y exige código, plan y draw nuevos.

## Preservación y replay

Se preservan, como mínimo:

- visibles, logits raw por seed y ensemble, labels autorizados y receipts de
  cada fase;
- tokens, targets, estratos, posterior, riesgos, acciones hard/posterior,
  features, disagreement, gain y pesos;
- estados completos de Ridge, Logistic principal y cinco Logistic sham;
- labels binarios, mappings, targets barajados, Hamming y composición de
  estratos de cada sham;
- scores `mu_hat`, probabilidades `p_harm`, thresholds, propuestas,
  autorizaciones, acciones y métricas por brazo;
- grillas completas, orden canónico, estados de factibilidad, shards,
  calibración y `set_index` con estados por support set;
- índices bootstrap y arrays token×política suficientes para recomputar toda
  métrica sin re-forward ni re-training.

Primary y replay usan las mismas claves, el mismo commit y la misma cronología.
La comparación exige byte-exactitud de cores/freezes/configs, array-exactitud
con dtype/shape/`equal_nan` de NPZ analíticos y equivalencia por hash de
visibles, logits y bundles. Paths absolutos, timestamps, duración, runtime y
receipts operativos quedan fuera del núcleo determinista, con exclusión
enumerada.

## Implementación prevista

- `experiments/geometria_proporcional/configs/wave57_contextual_tail_guard_fresh.json`
- `src/geometria_proporcional/wave57_tail_guard.py`
- `experiments/geometria_proporcional/prepare_wave57_fresh.py`
- `experiments/geometria_proporcional/_wave57_phase_worker.py`
- `experiments/geometria_proporcional/run_wave57_tail_guard.py`
- `tests/test_wave57_tail_guard.py`
- `tests/test_wave57_prospective.py`
- `data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1/`
- `data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1_replay/`

La implementación reutiliza, sin duplicarlos, el inference worker y el
materializador split-scoped auditados de Ola 56: su input y su autoridad no
cambian. El preparador y el coordinador existentes reciben hooks tipados de
config para escoger el worker analítico de Ola 57, mientras wrappers nuevos fijan
la interfaz canónica de comandos. Esta distribución reduce superficie de
seguridad y duplicación sin prestar semántica analítica: nuevos nombres, source
bindings, schemas, estados, runtime closure y tests quedan explícitos y
versionados. No se ejecuta el draw fresco hasta que plan, config, código y tests
estén versionados, el worktree esté limpio y una auditoría independiente emita
`PASS` sin findings materiales.

## Resolución de R395

R395 emitió `REVISE` con dos findings altos, uno medio-alto y cuatro medios.
Esta revisión: (1) selecciona y congela `tau_mu` antes de cualquier guard, por lo
que main, proposer y shams comparten propuestas; (2) formula las cuatro métricas
token-wise, el orden del promedio sham y el alcance condicional del bootstrap;
(3) agrega mínimos por token, clase, propuesta, autorización y desacuerdo,
incluida la decisión previa sobre cero overrides transportados; (4) tipa el null
condicional y exige Hamming real global y ponderado; (5) fija órdenes totales de
selección; (6) especifica la Logistic para scikit-learn `1.8.0` y sus fallos
terminales; y (7) reemplaza la unión de support sets por estados individuales.
La discrepancia factual `189` vs `165` tokens FIT queda resuelta mediante
recomputación exacta sobre el artefacto citado, sin alterar el informe crudo.

## Presupuesto operativo

Todo el ciclo es CPU nativo: tres forwards del encoder congelado por split y
ajustes lineales/logísticos pequeños con `cpu_threads=4`. El runtime esperado es
del mismo orden que Ola 56 y no justifica reservar GPU. Si la implementación
real revelara que una etapa requiere CUDA o que CPU se vuelve un sustituto
materialmente ineficiente, el ciclo se detiene antes de ejecutarla, se registra
el bloqueo y se avisa a Mariano con objetivo, duración y VRAM estimada.
