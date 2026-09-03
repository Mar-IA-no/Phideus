# Ola 56 — plan prospectivo de la compuerta residual contextual de valor

> **Estado:** `STAGE-0-CLOSED / STAGE-1-PREGEN-AUDIT-PENDING / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Antecedentes:** `WAVE_55_CONSERVATIVE_POLICY_BRIDGE_CLOSED.md` y
> `WAVE_56_CONTEXTUAL_RESIDUAL_GATE_RESEARCH_NOTE.md`

## Pregunta y estimando

La Ola 56 pregunta si una regla de baja capacidad, entrenada para estimar el
valor contextual de reemplazar la acción dura por la acción bayesiana del
posterior conjunto, puede reducir regret sin perder la exactitud y la
compatibilidad que hicieron inviable al umbral escalar de la Ola 55.

Se congelan el generador, el encoder `sigmoid_only`, los checkpoints seeds
`17/29/43`, el normalizador, el posterior `joint_full` primary de la Ola 54, las
veinticuatro políticas ordinales, la penalización incompatible, el conjunto
duro y ambas acciones candidatas. La única pieza nueva es el selector residual.
No se reentrena ni agranda la representación.

## Stage 0 — diagnóstico retrospectivo abierto

La fuente de desarrollo es explícita: `decision_select` de Ola 55 funciona como
`dev_fit`; su `sealed_monitor`, ya abierto por aquel experimento, funciona como
`dev_eval`. No se mezclan para entrenar ni para producir predicciones OOF. Stage
0 elige una familia para un futuro ensayo fresco, pero nunca adjudica la
arquitectura.

El análisis primario de Stage 0 usa exclusivamente los tokens `NEAR_RIVAL` con
cardinalidad verdadera mayor o igual que dos, tanto para fit, scoring OOF,
restricciones y selección final en `dev_eval`. La cardinalidad verdadera define
la población de evaluación y nunca entra como feature. Una sensibilidad repite
el pipeline completo sobre todos los tokens in-catalog y sólo reporta si cambia
la familia o el signo de sus contrastes; no reemplaza la selección primaria.

En `dev_fit` primario se construyen cinco folds deterministas de tokens mediante
`StratifiedKFold(n_splits=5, shuffle=True, random_state=5601)`, estratificando
por cardinalidad verdadera `{2,3,4}`. Cada fold expande después sus tokens a las
veinticuatro políticas. Si `d_t` es el número de políticas en desacuerdo del
token, cada fila usada en fit recibe peso `1/d_t`; así cada token con al menos un
desacuerdo aporta peso total uno. Todos los fits se restringen a esas filas.

El diseño tiene diecisiete columnas, en este orden: `advantage`, `hard_risk`,
`minimum_risk`, `action_risk_margin`, `posterior_entropy_norm`,
`posterior_top_mass`, `posterior_top_margin`, `hard_cardinality`,
`posterior_expected_cardinality`, `posterior_cardinality_variance`,
`posterior_mass_hard_set`, `seed_std_mean`, `seed_std_max`, `utility_f0`,
`utility_f1`, `utility_f2`, `utility_f3`. `disagreement` no entra porque vale
uno en toda fila de fit. `StandardScaler(with_mean=True, with_std=True)` se
ajusta fold-local sobre las diecisiete columnas usando los mismos sample
weights cuando la versión instalada lo soporte; si no soporta `sample_weight`,
media y varianza ponderadas se calculan con NumPy y se preservan. Una columna de
varianza exactamente cero recibe escala `1.0` y queda centrada, sin eliminarse.

Las familias y grillas cerradas son:

- ridge contextual y advantage-only: `Ridge(alpha ∈
  {0.01,0.1,1,10,100}, solver="svd")`; alpha minimiza MSE OOF;
- Huber contextual: `HuberRegressor(epsilon ∈ {1.1,1.35,1.75}, alpha ∈
  {1e-4,1e-3,1e-2}, max_iter=2000, tol=1e-8)`; minimiza MAE OOF;
- logística contextual: `LogisticRegression(C ∈ {0.01,0.1,1,10},
  class_weight="balanced", solver="lbfgs", max_iter=2000,
  random_state=5602)` sobre `gain_realizado>1e-12`; minimiza log-loss OOF.

Empates numéricos `<=1e-12` eligen mayor regularización; en Huber, además menor
`epsilon`. Un candidato sin convergencia queda `FIT_FAILED` y no habilita ampliar
la grilla. Se comparan el advantage escalar de Ola 55, ridge advantage-only,
ridge contextual, Huber contextual, logística contextual y controles
contextuales con gain barajado.

Cada score OOF se convierte en sistema evaluando thresholds en los cuantiles
`{0.50,0.60,0.70,0.80,0.90,0.95,0.975}` de sus scores entre desacuerdos, con
`numpy.quantile(method="higher")`, más `hard_only`. Sólo se reemplaza cuando el
score es estrictamente mayor al threshold. Por familia se elige el threshold de
menor regret entre los que mantienen accuracy dentro de `0.01` y compatibilidad
no inferior al hard; empate `1e-12`, menor coverage.

Los OOF seleccionan hiperparámetro y operating point dentro de `dev_fit`. Luego
se ajusta cada familia sobre todo `dev_fit`; el cuantil elegido se convierte en
threshold numérico usando sus scores in-sample sin labels adicionales, y se
aplica una vez a `dev_eval`. La familia prospectiva se elige sólo entre
`ridge_contextual`, `huber_contextual` y `logistic_contextual`: menor regret en
`dev_eval` entre las que mantienen allí las mismas restricciones; empate,
ridge, luego Huber, luego logística y por último menor coverage. El score
escalar, advantage-only y shuffled son controles, nunca candidatos primarios.
Si ninguna familia contextual es elegible, no se abre evidencia fresca.

Se reportan error de predicción, correlación rank-based con el gain realizado,
precision/recall de reemplazos beneficiosos, curvas regret-cobertura y métricas
del sistema. La sensibilidad leave-policy-group-out usa la partición balanceada
3×8 de Ola 52: ajusta sobre dieciséis políticas y evalúa ocho, rotando los tres
grupos. No modifica la selección primaria ni autoriza transporte fuera del
catálogo.

Stage 0 es `NOT_EVALUABLE` si `dev_fit` o `dev_eval` primarios tienen menos de
`100` tokens o `400` filas en desacuerdo, o si algún fold tiene menos de `20`
tokens. La logística queda `NOT_EVALUABLE` si carece de ambas clases en un fold
de fit o en `dev_eval`; la regresión puede continuar cuando sólo ella falla. Se
preservan coeficientes, predicciones OOF,
fold assignment, pesos, schema/orden/dtypes de la matriz de diseño, escaladores,
grillas completas, convergencia, thresholds, curvas, targets y mappings de
shuffle. Todo queda rotulado `RETROSPECTIVE-OPENED-DEVELOPMENT`.

## Stage 1 — protocolo fresco y cronología de acceso

Si Stage 0 identifica una familia elegible, se versionan plan resuelto, config,
código, grillas, hashes upstream y política de fallos. Una **segunda auditoría
independiente** verifica la familia elegida, la config y la implementación; sólo
después de resolver sus findings se extraen claves.
Se genera una única realización fresca de la misma ley con tres particiones
físicamente separadas:

- `gate_fit`: ajusta escalador y coeficientes;
- `gate_select`: selecciona operating point y aplica restricciones;
- `sealed_monitor`: adjudica una sola vez después del freeze de selección.

Los nombres físicos del generador pueden corresponder a `train`, `val` y
`lockbox`, pero sus roles Wave 56 se registran explícitamente. No queda un
lockbox adicional: la tercera partición es el monitor sellado de este
experimento. La inferencia de los tres seeds se ejecuta sobre observaciones
visibles antes de materializar cualquier oracle. La cronología se implementa
como tres comandos idempotentes y estados físicos distintos, no como una demora
de carga dentro de un único proceso:

1. `fit` asserta ausencia de labels/bundle de `gate_select` y
   `sealed_monitor`, materializa sólo `gate_fit`, ajusta y escribe
   `fit_freeze.json`;
2. `select` verifica el hash de `fit_freeze`, vuelve a assertar ausencia del
   monitor, materializa sólo `gate_select` y escribe `selection_freeze.json`;
3. `adjudicate` verifica ambos freezes y recién entonces materializa
   `sealed_monitor`.

Tests de integración inspeccionan ausencia de cada path de oracle, labels y
bundle posterior antes y después de cada fase. El preparador de Ola 55 no se
copia sin cambios: materializa juntos train y val y no satisface este contrato.

Los tres roles deben ser disjuntos por `pair_token` y no solaparse con los
bundles de fit/selección/monitor de Olas 54 y 55. Las claves criptográficas son
nuevas y diferentes. No hay redraw: un fallo técnico se cuarentena y sólo puede
recuperarse con las mismas claves mediante un delta de código que no altere el
estimando.

## Features y target

Para token `t` y política `p`, el target de fit es:

```text
y_tp = regret_hard_tp - regret_posterior_tp
```

Las features se calculan sin usar el target del caso:

1. `advantage_p`, `hard_risk_p`, `minimum_risk_p` y margen entre las dos
   mejores acciones;
2. entropía normalizada, masa máxima y margen top-1/top-2 del posterior;
3. cardinalidad dura, cardinalidad posterior esperada y varianza;
4. masa posterior del conjunto duro;
5. desacuerdo `a_hard != a_posterior`;
6. media y máximo de la desviación estándar entre seeds para las cuatro
   familias;
7. vector de cuatro utilidades centrado y escalado.

No se usa `policy_id`, oracle del target, cardinalidad verdadera, set verdadero,
cluster ID semántico ni ninguna estadística calculada sobre monitor etiquetado.
El fit se restringe a desacuerdos, porque en acuerdos el reemplazo es idéntico y
la masa de ceros dominaría el objetivo. En evaluación se conserva la población
completa: los acuerdos son no-overrides triviales. El estimando primario se
limita al catálogo fijo de veinticuatro políticas; el vector de utilidades es
biyectivo con su identidad y no demuestra transporte externo.

El escalador se ajusta sólo en `gate_fit`. La regularización o robustez elegida
por Stage 0 queda fija antes de generar. No se vuelve a comparar familias en
`gate_select` ni monitor.

## Operating point y selector

El modelo produce un score de gain esperado. En `gate_select` se construye una
grilla cerrada de thresholds a partir de cuantiles no etiquetados del score
entre desacuerdos:

```text
q in {0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, "hard_only"}
```

Cada `q` se convierte allí en un valor numérico, y ese valor queda congelado
para monitor; no se recalcula el cuantil en monitor. Sólo se permite override
si el score es estrictamente mayor que el threshold y las acciones difieren.
`hard_only` reproduce byte-exact el baseline.

Son elegibles los thresholds cuya accuracy no cae más de `0.01` frente al hard
y cuya compatibilidad no cae frente a hard. Entre elegibles se minimiza regret;
empate absoluto `1e-12`: menor tasa de override. La selección opera en la
población primaria `NEAR_RIVAL` con cardinalidad observacional mayor o igual que
dos, igual que Ola 55. Una sensibilidad repite la selección sobre todos los
tokens in-catalog, sin reemplazar el estimando primario.

La estabilidad se audita dividiendo `gate_select` en dos shards deterministas
por el bit menos significativo de `SHA256(pair_token || "wave56-shard")`. Cada
shard reelige sólo el threshold contextual, con modelo y controles congelados;
se compara cada shard con la selección completa y ambos thresholds se aplican
al mismo monitor sólo como sensibilidad. Los contrastes exactos son regret,
accuracy y compatibilidad contextual−hard; regret, accuracy y compatibilidad
contextual−gate-escalar; regret contextual−advantage-only; regret
contextual−shuffled-promedio; y regret/accuracy contextual−pure-joint. El signo
es `-1/0/+1` con tolerancia `1e-12`. Si cualquier shard difiere del selector
completo en cualquier estado algebraico, `selector_sensitive=true`. Los
controles conservan sus thresholds de la selección completa.

## Brazos

1. `hard_set_policy`: baseline y fallback.
2. `pure_joint_full`: acción bayesiana sin gate.
3. `scalar_advantage_gate`: la receta de Ola 55, re-seleccionada sólo en
   `gate_select` fresco con su grilla congelada original.
4. `contextual_value_gate`: candidata primaria elegida en Stage 0.
5. `advantage_only_value_gate`: misma familia y protocolo, sólo con
   `advantage_p`; controla el valor del contexto.
6. `contextual_shuffled_gain`: misma familia, features y protocolo, con `y_tp`
   barajado dentro de la población de desacuerdos y por índice de política.
7. `oracle_positive_gain`: referencia privilegiada que reemplaza exactamente
   cuando `y_tp>0`; nunca deployable ni techo universal.

El control barajado conserva exactamente la matriz de diseño, la población y la
medida ponderada del learner. Para cada `(policy_index, d_t)`, permuta
`gain_realizado` sólo entre tokens que discrepan bajo esa política y tienen el
mismo número total de desacuerdos; así conserva el multiconjunto de targets y su
peso efectivo `1/d_t`, incluida la proporción positiva/negativa/cero, sin
importar ceros estructurales desde acuerdos. Grupos de tamaño uno permanecen
fijos y se reporta la fracción realmente movida; si menos del `80%` de las filas
de desacuerdo son movibles, el control queda `NOT_EVALUABLE` y sale del criterio
vinculante. Se usan cinco permutaciones PCG64 predeclaradas con seeds
`{56031,56032,56033,56034,56035}`. El control reportado promedia por token las
métricas de los cinco modelos antes del bootstrap; el mapping íntegro queda
preservado. No es candidato a selección.

## Métricas e inferencia

La unidad inferencial es `pair_token`. Primero se promedian las veinticuatro
políticas dentro del token y luego se aplica bootstrap pareado, con los mismos
índices para todos los brazos y métricas. Se preservan y reportan:

- accuracy, compatibilidad, regret y peor regret por token;
- coverage/tasa de override;
- precisión, recall y magnitud de overrides beneficiosos y perjudiciales;
- curvas regret-cobertura del score contextual y del advantage escalar;
- correlaciones Pearson y Spearman score-gain sólo como diagnóstico;
- coeficientes estandarizados y estabilidad entre shards;
- desempeño por política y por estrato como sensibilidad, sin cambiar la unidad
  primaria.

Los IC95 usan `5.000` remuestras pareadas de token con un índice PCG64 generado
una vez y preservado. JSON estricto: métricas sin denominador se serializan
`null`, nunca `NaN`.

## Criterio diagnóstico prospectivo

El brazo contextual queda marcado como candidato a réplica independiente, sin
promoción ni GO, sólo si en `sealed_monitor` primario:

1. reduce regret frente a hard al menos `0.01`, con IC95 superior debajo de
   cero;
2. el IC95 inferior de accuracy contextual−hard es al menos `-0.01` y el de
   compatibilidad contextual−hard al menos `0`;
3. reduce regret frente al gate escalar y al advantage-only al menos `0.005`,
   con IC95 superior debajo de cero;
4. reduce regret frente al control shuffled al menos `0.01`, con IC95 superior
   debajo de cero;
5. frente a pure joint, el IC95 inferior de accuracy es mayor que cero y el
   IC95 superior del delta de regret es como máximo cero;
6. no es selector-sensitive y el replay es exacto.

Estos criterios adjudican un patrón experimental; no delegan al script la
decisión científica del usuario.

## Soporte ausente y ley enriquecida

Los mínimos operativos prospectivos son `100` tokens primarios y `400` filas en
desacuerdo en `gate_fit`, `80/300` en `gate_select`, `40/120` por shard y `100`
tokens primarios en monitor. Si no se cumplen, la fase correspondiente queda
`NOT_EVALUABLE`; no se redibuja. La logística seleccionada retrospectivamente
exige ambas clases de gain en `gate_fit` y `gate_select`, o el experimento queda
`NOT_EVALUABLE` en vez de cambiar de familia post hoc.

Los cinco conjuntos ausentes del fit Wave 54 se mantienen fuera del claim
primario. Si el monitor contiene menos de `30` tokens de ese soporte, quedan
`NOT_EVALUABLE`. Un futuro banco enriquecido debe declararse como otra ley,
mantenerse separado y evaluar transporte; no puede completar silenciosamente la
población primaria.

## Replay, artefactos y fallos

La preparación y adjudicación se repiten con las mismas claves. Deben coincidir
arrays, folds, features, escaladores, coeficientes, thresholds, acciones,
métricas, bootstrap y archivos analíticos. Se guardan además inferencia raw por
seed, targets autorizados por fase, manifests de acceso, hashes upstream,
predicciones y gains por token×política, curvas, selección por shards, runtime y
manifest integral de artefactos. El schema de diseño fija nombre, orden, dtype,
normalización y dominio de cada columna; también se guardan pesos efectivos por
token, índices de filas de desacuerdo, convergencia de cada candidato, grillas
completas, método exacto de cuantiles, mappings de las cinco permutaciones y
receipts de apertura de oracle por fase.

Un directorio primario preexistente bloquea la corrida. `--force` archiva, no
borra. Un fallo posterior a la apertura de oracle no habilita otra generación.

## Archivos previstos

- `src/geometria_proporcional/wave56_contextual_gate.py`
- `experiments/geometria_proporcional/run_wave56_retrospective.py`
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`
- `experiments/geometria_proporcional/run_wave56_contextual_gate.py`
- `experiments/geometria_proporcional/configs/wave56_contextual_gate.json`
- `tests/test_wave56_contextual_gate.py`

## Límites operativos

CPU-only en esta formulación. La implementación comienza después de una
auditoría independiente del plan y de resolver findings materiales. Stage 0 no
puede adjudicar; Stage 1 usa una realización fresca de la misma ley y no prueba
transferencia de aparato, utilidad natural, geometría física ni PPU. Cualquier
uso futuro de GPU se informa antes con propósito, duración y VRAM estimada.
Toda promoción arquitectónica y todo GO/NO-GO permanecen en el usuario.

## Resolución de la auditoría R314

R314 emitió `REVISE` con tres findings altos y tres medios. Esta revisión: (1)
redefine el shuffled dentro de filas de desacuerdo y por política, preserva el
marginal exacto y usa cinco seeds; (2) congela datos dev, folds, escalado,
solvers, grillas, objetivos, operating point y desempates de Stage 0; (3) exige
una segunda auditoría tras elegir familia y una máquina de estados física
`fit→select→adjudicate`; (4) enumera contrastes y tolerancia de sensibilidad;
(5) restringe el claim al catálogo de políticas y agrega leave-policy-group-out;
y (6) fija mínimos, estados `NOT_EVALUABLE` y artefactos ejecutables. Requiere
reauditoría focal antes de implementar Stage 0.

R315 mantuvo dos findings altos y uno medio. La segunda resolución: (1) estratifica
el shuffle por `(policy_index,d_t)`, preservando el peso efectivo por token, y
exige al menos `80%` de filas movibles; (2) fija la población primaria de Stage
0, los pesos `1/d_t`, el schema exacto de diecisiete columnas y el escalado
ponderado con tratamiento de constantes; y (3) extiende mínimos y estados de
no-evaluabilidad a `dev_eval`. Requiere reauditoría final focal.

R316 verificó esas tres resoluciones, no reabrió findings y dio `PASS` para
implementar exclusivamente Stage 0. La familia prospectiva, la config y el
código Stage 1 deberán pasar una segunda auditoría independiente antes de
extraer claves frescas.

## Resolución de las auditorías de implementación R317-R318

R317 emitió `REVISE` por ausencia de bootstrap, preservación incompleta,
fuentes todavía no congeladas, una sensibilidad leave-policy contaminada,
ausencia de la repetición all-in-catalog y reglas de borde incompletas. La
resolución incorporó: una matriz única de 5.000 remuestras pareadas por token;
diagnósticos de overrides; serialización de inputs, features, candidatos,
modelos fold-locales y full, curvas, acciones, métricas, shuffles y schema;
preflight antes de crear output; selección leave-policy completa sin acceso a
las ocho políticas retenidas; pipeline all-in-catalog; desempates tolerantes y
estados logísticos no evaluables. Un smoke sobre los bundles reales volvió a
seleccionar `ridge_contextual`.

R318 confirmó bootstrap, ausencia de leakage y la mayor parte de esas
resoluciones, y mantuvo tres observaciones. La resolución final: (1) congela en
Git runner, primitivas, plan, nota y config para que el entrypoint satisfaga su
propio preflight; (2) repite también los cinco null barajados y todos los signos
de contraste en la sensibilidad all-in-catalog; y (3) preserva por split
leave-policy los scores OOF de toda la grilla, modelos fold-locales, curvas,
scores y acciones de evaluación, y modelo full. Antes de la corrida oficial
Stage 0 se exige una reauditoría focal final sobre este estado versionado.

R319 auditó el estado ya versionado, ejecutó independientemente tests y un
primary+replay temporal, verificó replay exacto de los cuatro artefactos
analíticos y dio `PASS`. Confirmó que el entrypoint satisface su freeze, que la
sensibilidad all-in-catalog incluye cinco null y los diez signos predeclarados,
y que leave-policy preserva evidencia suficiente sin acceso selectivo a las
ocho políticas retenidas. Stage 0 queda habilitado para ejecución oficial en
CPU; cualquier Stage 1 continúa sujeto a la segunda auditoría independiente
prospectiva fijada arriba.

## Cierre de Stage 0 y resolución R320

La corrida oficial y su replay seleccionaron `ridge_contextual` con
`alpha=1.0` y cuantil OOF `q=0.70`. En `dev_eval` primario obtuvo accuracy
`0.826987`, compatibilidad `0.949917` y regret `0.117883`. Frente a la política
dura redujo regret en `-0.011739`, IC95 `[-0.022260,-0.002414]`, pero su delta
de accuracy `-0.004139` tuvo IC95 `[-0.015315,+0.006485]`: el punto satisface
la restricción retrospectiva, aunque el extremo inferior no alcanzaría el
criterio prospectivo de Stage 1.

La evidencia no identifica todavía valor contextual específico. Frente a
`advantage_only` el delta de regret fue `-0.006921`, IC95
`[-0.016303,+0.001782]`; frente al promedio de cinco targets barajados fue
`+0.000676`, IC95 `[-0.004857,+0.006117]`. La sensibilidad all-in-catalog
mantuvo la familia ridge, cambió `alpha` de `1` a `100` e invirtió `4/10`
signos predeclarados. Por eso Stage 0 selecciona una receta para falsación
fresca, no demuestra que el contexto aprendido supere controles simples o
nulos ni habilita promoción arquitectónica.

R320 reprodujo integridad, hashes, selección, constraints, los cinco mil
bootstraps, los nulls, leave-policy-group-out y el replay exacto, y emitió
`PASS` sin findings materiales. El cierre durable está en
`WAVE_56_STAGE0_RETROSPECTIVE_CLOSED.md`; Stage 1 queda habilitado sólo para
implementar y auditar su protocolo prospectivo antes de extraer claves nuevas.
