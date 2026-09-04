# Ola 57 — plan prospectivo para propuesta de valor y guard de daño

> **Estado:** `DRAFT-FOR-INDEPENDENT-AUDIT / PRE-IMPLEMENTATION / PRE-KEY-DRAW / CPU-ONLY / NO-GO-NOGO`
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
SVD. La cabeza de riesgo usa target `1[g_tp < -1e-12]`, log-loss, escalado
ponderado, `solver=lbfgs`, `penalty=l2`, `C=1.0`, `class_weight=None`,
`fit_intercept=true`, `max_iter=2000` y `tol=1e-10`. No se seleccionan
hiperparámetros de modelos. Ambas cabezas se ajustan sólo sobre filas de
desacuerdo de la población primaria de `train`, con peso `1/d_t`, donde `d_t`
es la cantidad de políticas en desacuerdo del token.

Los signos son estrictos. Un score igual a `tau_mu` no propone; una probabilidad
igual a `tau_harm` no autoriza. `hard_only` reproduce byte por byte la acción
dura. El target de daño, el gain y toda métrica dependen de truth y nunca forman
parte del diseño de inferencia.

## Control de capacidad igualada

Cinco guards sham conservan exactamente features, escalador, familia,
regularización, solver, presupuesto de ajuste y selector del guard principal.
Sólo cambia su target: la etiqueta binaria de daño se permuta dentro de cada
estrato `(policy_index, disagreement_count)` con seeds
`{57031,57032,57033,57034,57035}`. Cada réplica preserva prevalencia por estrato,
mapping, fracción movible y fracción efectivamente movida.

El proposer es común al guard verdadero y a los cinco shams. Así, una diferencia
no puede atribuirse a sumar una segunda regresión o a multiplicar el número de
operating points. Cada sham selecciona sus umbrales por el mismo procedimiento
que el guard verdadero. El contraste primario usa el promedio predeclarado de
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
| `train` | `guard_fit` | ajusta proposer, guard y cinco shams |
| `val` | `guard_select` | selecciona operating points y prueba estabilidad |
| `lockbox` | `sealed_monitor` | adjudica una sola vez sin reselección |

Los tres roles deben ser disjuntos por `pair_token` entre sí y respecto de los
artefactos analíticos de Olas 54–56. No hay redraw. El monitor abierto de Ola 56
sirve como antecedente para formular la hipótesis, pero no aporta filas, labels,
thresholds ni decisiones al nuevo ajuste.

## Selección congelada

El proposer solo evalúa cuantiles de su score
`{0.50,0.60,0.70,0.80,0.90,0.95,0.975}` y `hard_only`.

La interfaz conjunta evalúa el producto cartesiano de:

- los mismos siete cuantiles de `mu_hat`;
- cuantiles de aceptación por bajo riesgo
  `{0.10,0.20,0.30,0.40,0.50,0.60,0.80}` calculados sobre `p_harm` únicamente
  entre filas propuestas del candidato correspondiente;
- `hard_only` como terminal común.

Cada cuantil de riesgo se materializa con `method="higher"`; se autoriza
estrictamente por debajo del threshold resultante. Un candidato sin al menos
`60` filas propuestas o `20` filas autorizadas en la población primaria de
selección queda `NOT_EVALUABLE`, no se relaja.

Para ser factible, un operating point debe satisfacer en `guard_select`:

- `accuracy >= accuracy_hard - 0.01`;
- `compatible >= compatible_hard`;
- `worst_regret <= worst_regret_hard + 0.01`.

Entre candidatos factibles se minimiza regret medio. Empates dentro de
`1e-12` se resuelven por menor tasa de autorización y luego por mayor umbral de
`mu_hat`. `hard_only` es siempre factible. El proposer solo usa las dos primeras
restricciones, reproduciendo su interfaz previa; esta asimetría es constitutiva
del nuevo guard y queda visible en el contraste.

Cada sham recorre la misma grilla y las mismas tres restricciones que el guard
verdadero. Además se repite la selección principal en dos shards deterministas
por `pair_token`, con salt `wave57-shard`, sin cambiar modelos ni grids.

## Mínimos y cierres no evaluables

Antes de ajustar se exigen en población primaria:

- `guard_fit`: al menos `100` tokens, `400` filas de desacuerdo, `150` labels
  perjudiciales y `100` no perjudiciales;
- `guard_select`: al menos `80` tokens y `300` filas de desacuerdo;
- cada shard de selección: al menos `40` tokens y `120` filas de desacuerdo;
- `sealed_monitor`: al menos `100` tokens;
- cada shuffle: fracción movible mínima `0.80`;
- soporte ausente heredado: `30` tokens por set.

Si falla un mínimo global de fit o select, el proceso termina antes de abrir el
split siguiente. Si un sham no alcanza su fracción movible, sólo ese control y
las condiciones que lo requieren quedan `NOT_EVALUABLE`. Si el monitor no
alcanza su mínimo, se preserva el paquete sin emitir el patrón agregado. Los
cinco support sets ausentes de Ola 54 se reportan por separado y no se imputan.

## Patrón prospectivo predeclarado

La unidad inferencial es el pair token después de promediar las veinticuatro
políticas. Se generan `5000` bootstraps pareados con `PCG64`, seed `5707`, orden
lexicográfico de token e intervalo percentil `[2.5,97.5]`. El patrón conjunto
queda observado sólo si se cumplen las seis condiciones:

1. regret principal menos hard `<= -0.01` y límite superior IC95 `< 0`;
2. accuracy principal menos hard con límite inferior IC95 `>= -0.01`, y
   compatibilidad con límite inferior IC95 `>= 0`;
3. worst regret principal menos hard con media `<= 0` y límite superior IC95
   `<= 0`;
4. frente al proposer solo, accuracy mejora con límite inferior IC95 `> 0` y
   worst regret mejora con límite superior IC95 `< 0`, mientras regret no es
   inferior por más de `0.005` (`IC95 superior de principal-proposer <= 0.005`);
5. frente al promedio de los cinco shams, regret y worst regret mejoran con
   límites superiores IC95 `< 0`;
6. ambos shards eligen operating points no `hard_only`, conservan los signos de
   regret/accuracy/worst-regret frente a hard, y el replay exacto satisface su
   matriz completa.

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

## Preservación y replay

Se preservan, como mínimo:

- visibles, logits raw por seed y ensemble, labels autorizados y receipts de
  cada fase;
- tokens, targets, estratos, posterior, riesgos, acciones hard/posterior,
  features, disagreement, gain y pesos;
- estados completos de Ridge, Logistic principal y cinco Logistic sham;
- labels binarios, mappings y targets barajados;
- scores `mu_hat`, probabilidades `p_harm`, thresholds, propuestas,
  autorizaciones, acciones y métricas por brazo;
- grillas completas, estados de factibilidad, shards, calibración y support
  sets;
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
- `experiments/geometria_proporcional/_wave57_infer_worker.py`
- `experiments/geometria_proporcional/_wave57_oracle_materializer.py`
- `experiments/geometria_proporcional/_wave57_phase_worker.py`
- `experiments/geometria_proporcional/run_wave57_tail_guard.py`
- `tests/test_wave57_tail_guard.py`
- `tests/test_wave57_prospective.py`
- `data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1/`
- `data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1_replay/`

La implementación puede reutilizar funciones puras y copiar el armazón de
aislamiento ya auditado, pero los nuevos nombres, source bindings, inventarios,
transiciones y schemas deben quedar explícitos. No se ejecuta el draw fresco
hasta que plan, config, código y tests estén versionados, el worktree esté limpio
y una auditoría independiente emita `PASS` sin findings materiales.

## Presupuesto operativo

Todo el ciclo es CPU nativo: tres forwards del encoder congelado por split y
ajustes lineales/logísticos pequeños con `cpu_threads=4`. El runtime esperado es
del mismo orden que Ola 56 y no justifica reservar GPU. Si la implementación
real revelara que una etapa requiere CUDA o que CPU se vuelve un sustituto
materialmente ineficiente, el ciclo se detiene antes de ejecutarla, se registra
el bloqueo y se avisa a Mariano con objetivo, duración y VRAM estimada.
