# R397 — Auditoría independiente de implementación de Ola 57

## Dictamen

**REVISE**

El estado combinado terminado en
`d29d1aa52256f112bc6fa2e1569dc9a7e20e77f7` implementa correctamente el
nucleo estadistico principal de Ola 57: no encontre leakage de truth hacia las
diecisiete features; el proposer se selecciona una sola vez y su mascara queda
fija para el guard verdadero y los cinco shams; el target logistico orienta la
clase positiva hacia daño; el shuffle es condicional a
`(policy_index, disagreement_count)` y conserva diagnosticos de Hamming real;
los estimandos y el bootstrap operan token-wise; y los cinco support sets se
adjudican individualmente. La correccion `d29d1aa` tambien cierra los tres
imports transitivos que faltaban en `04852d1`, y el worker Wave 57 arranca desde
el runtime staged.

El paquete no esta listo para extraer claves ni ejecutar el draw. Quedan dos
defectos altos reproducidos: el preflight invoca una validacion Wave 57 que
acepta drift de constantes decisorias y hasta del worker analitico, y el checker
de replay declara exactitud aunque difieran los freezes FIT/SELECT que el plan
exige comparar byte por byte. Hay ademas tres incumplimientos medios de
cobertura o reporte que conviene cerrar antes de abrir labels. Conforme al
criterio pedido, cualquier finding material impide `PASS`.

No ejecute preparacion oficial, generacion de claves, draw, oracle fresco,
materializacion de labels oficial ni GPU/CUDA.

## Identidad y alcance

- Commit base implementado: `04852d110d9cd20fb01d7ad1bfcaedbe67f402e8`.
- Commit correctivo incluido: `d29d1aa52256f112bc6fa2e1569dc9a7e20e77f7`.
- HEAD auditado: `d29d1aa52256f112bc6fa2e1569dc9a7e20e77f7`.
- Plan vigente: commit `b271b8885cd5bcf08421fecf9e81a9df884bea2a`.
- Estado Git antes de crear este informe: limpio.
- Material leido completo: plan Ola 57, R395, R396 y los diez archivos de
  implementacion modificados por `04852d1`, incorporando las tres correcciones
  de `d29d1aa`.
- Reglas cargadas: `/root/.codex/AGENTS.md`, `AGENTS.md`, `CODEX.md` y
  `/mnt/m2-1TB/MENSAJES_RECURSIVOS.md`. No se leyo `PENDIENTES.md` ni
  `.claude/memory.md`.

## Findings priorizados

### F1 — ALTO — La validacion llamada pre-key no congela el contrato Wave 57 completo

La correccion `d29d1aa` hace que `validate_prospective_config()` invoque
`validate_wave57_frozen_config()` antes de crear output o extraer claves
(`prepare_wave56_fresh.py:474-491,665-685,1959-1969`). Esa ubicacion temporal es
correcta. El defecto es que la funcion invocada valida solamente schema,
device/seeds/features, kwargs Logistic, grids, shuffle seeds, splits, brazos,
seed/repeticiones de bootstrap y minimos (`wave57_tail_guard.py:34-102`). El
preparador agrega el conjunto de nombres —no los valores— de criterios
diagnosticos (`prepare_wave56_fresh.py:492-525`).

Quedan fuera del freeze efectivo valores que cambian targets, acciones,
seleccion, adjudicacion o frontera: `hard_set_tau`, penalty de regret, margenes
de seleccion, valores de criterios, `quantile_method`, `harm_epsilon`, intervalo
y semantica de bootstrap, shard salt/count, declaracion de compatibilidad y
`phase_worker_relative`. Esto ultimo es especialmente grave: el runtime incluye
ambos workers y `build_phase_runtime()` acepta como entrypoint cualquier
`phase_worker_relative` contenido en la lista configurada
(`run_wave56_contextual_gate.py:660-692`).

Reproduccion en memoria, ejecutada contra el HEAD auditado: diez mutaciones
independientes pasaron simultaneamente
`validate_wave57_frozen_config()` y `validate_prospective_config()`:

```text
hard_set_tau ACCEPTED
incompatible_regret_penalty ACCEPTED
selection_accuracy_margin ACCEPTED
diagnostic_regret_min ACCEPTED
quantile_method ACCEPTED
harm_epsilon ACCEPTED
bootstrap_interval ACCEPTED
shard_salt ACCEPTED
boundary_semantics ACCEPTED
phase_worker_relative ACCEPTED
accepted_count 10 of 10
```

El ultimo caso cambio el entrypoint a
`experiments/geometria_proporcional/_wave56_phase_worker.py`. Por tanto, la
nueva llamada no demuestra el checkpoint de R396 que exige seleccion exclusiva
del worker analitico ligado a la config. `source-at-HEAD` autentica bytes
versionados, pero no demuestra que esos bytes satisfagan el contrato cientifico;
un config erroneo commiteado sigue siendo “at HEAD”.

**Correccion requerida.** Validar por igualdad exacta todos los campos
result-affecting y de frontera del config, en especial modelos, thresholds,
strictness, penalty, seleccion y desempates, criterios con valores, bootstrap,
shards, `shared_boundary_interface`, lista cerrada de
`phase_runtime_sources` y `phase_worker_relative`. Agregar tests de mutacion que
rechacen al menos una alteracion por cada familia y uno que pruebe explicitamente
que el worker Wave 56 no puede seleccionarse bajo schema Wave 57.

### F2 — ALTO — El replay puede producir un falso `all_exact` con freezes distintos

El plan exige byte-exactitud de cores, freezes y configs
(`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:323-328`). Sin embargo,
`compare_reference()` compara `fit_core.json`, `feature_schema.json`,
`selection_core.json`, `analysis_core.json`, tres NPZ analiticos y tres bundles;
no compara `fit_freeze.json` ni `selection_freeze.json`
(`run_wave56_contextual_gate.py:933-978`). La prueba sintetica reproduce la misma
omision: sus seis paths exactos no contienen freezes
(`tests/test_wave57_prospective.py:166-178`).

Reproduccion CPU aislada: genere una cadena sintetica FIT/SELECT/ADJUDICATE,
copie el paquete, agregue un campo distinto a `fit_freeze.json` sin tocar los
hashes de archivos que ese freeze autentica, y llame `compare_reference()`:

```text
fit_freeze_bytes_equal False
compare_reference_all_exact True
checks_count 10
freeze_checks []
```

La mutacion sigue pasando `_load_freeze()` porque los hashes internos de cores y
arrays permanecen validos. El resultado contradice directamente el contrato de
replay y debilita el checkpoint source/replay de R396.

**Correccion requerida.** Comparar byte-exacto `fit_freeze.json` y
`selection_freeze.json` y hacer visible ambos checks en el receipt. Agregar
regresiones que muten por separado cada freeze y exijan rechazo. La igualdad de
config ya esta cubierta indirectamente por los preparation freezes, pero
conviene exponerla tambien como check nombrado para que la matriz de replay sea
literalmente la del plan.

### F3 — MEDIO — Los tests no prueban la granularidad `NOT_EVALUABLE` exigida por R396

Los trece tests Wave 57 incluyen un unico caso de no evaluabilidad global del
monitor (`tests/test_wave57_prospective.py:181-189`) y un test de la primitiva
que impide prestar soporte desde la union de cinco sets
(`tests/test_wave57_tail_guard.py:154-161`). No hay tests que fuercen:

- proposer `hard_only` y cierre global de SELECT;
- una celda guard de bajo soporte mientras las otras y SELECT siguen vivas;
- un unico sham con Hamming/permutabilidad insuficiente mientras FIT, main y
  otros shams permanecen evaluables;
- un shard no evaluable sin terminar SELECT completo;
- minimos de propuesta/autorizacion diferenciados entre full y shard.

La implementacion inspeccionada sigue la granularidad prevista: celdas guard se
marcan individualmente (`_wave57_phase_worker.py:293-346`), un sham invalido no
termina FIT ni SELECT (`:140-180,371-383`), y los shards se registran por separado
(`:432-453`). Pero R396 pidio expresamente que esos limites fueran demostrados
por tests, no solo inferidos del codigo. Este hueco permitio que la suite focal
pasara sin detectar F1 ni F2.

**Correccion requerida.** Agregar tests focales para las cinco transiciones
anteriores y aserciones sobre estados/artefactos ausentes o preservados en cada
caso.

### F4 — MEDIO — El reporte por support set omite la summary del brazo sham promedio

La dimension individual esta correctamente materializada: `set_index` se
preserva (`_wave57_phase_worker.py:90-94`), `per_set_support()` decide cada set
sin union (`wave57_tail_guard.py:325-350`) y el adjudicador crea bootstrap y
contrastes por set evaluable (`_wave57_phase_worker.py:752-776`).

Pero las summaries por set se construyen recorriendo `arms` (`:767`), cuyo
inventario no incluye `mean_plus_shuffled_harm_guard` (`:643-650`). Al mismo
tiempo, los contrastes por set si incluyen `main_minus_shuffled` mediante
`references` (`:700-707,768-774`). El resultado es una superficie asimetrica:
existe un contraste contra el sham promedio sin la summary correspondiente del
brazo, pese a que el plan exige “summaries, contrastes e indices bootstrap por
set” (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:220-224`). El unico test de support
fuerza cinco estados `NOT_EVALUABLE`; no cubre la rama evaluable ni detecta la
omision.

**Correccion requerida.** Incluir summary y estado del promedio de cinco shams
—y conservar estados de replicas— en cada set evaluable; agregar un fixture con
al menos un set sobre el minimo y verificar inventario, bootstrap y contrasts.

### F5 — MEDIO — Dos diagnosticos secundarios prometidos no se materializan

El plan dice que el gate escalar de Ola 55 se conserva como diagnostico
secundario (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:116-119`) y que se reportan
slices observacionales (`:270-274`). No existe ninguna referencia a `scalar`,
`slice`, `policy_summary` o `monitor_diagnostic_curve` en el worker Wave 57, sus
primitivas o sus tests. El adjudicador produce hard, pure joint, proposer, main,
advantage-only Ridge y oracle (`_wave57_phase_worker.py:629-650`), calibracion y
support sets (`:778-815`), pero no el gate gamma de Ola 55 ni summaries por
slices observacionales.

Las metricas crudas por politica si quedan preservadas en NPZ mediante
`_store_arm()` (`_wave56_phase_worker.py:1045-1049`), por lo que no cuento
“politicas individuales” como ausentes. El faltante se limita al gate escalar y
a slices explícitos. Son report-only y no cambian las seis condiciones, pero
deben resolverse antes del draw porque el propio plan prohibe agregar codigo
post-label para adjudicar esa realizacion (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:293-303`).

**Correccion requerida.** Implementar y preservar ambos diagnosticos o revisar
formalmente el plan antes del draw para retirar/precisar esas promesas.

## Matriz plan → codigo → tests

| Contrato | Codigo auditado | Cobertura | Estado |
|---|---|---|---|
| Features inference-safe; truth solo en gain/labels/metricas | `run_wave56_retrospective.py:152-194`; `_wave57_phase_worker.py:108-115,620-642` | end-to-end sintetico | **PASS** |
| Proposer seleccionado una vez; mascara comun main/shams | `_wave57_phase_worker.py:349-383,417-421,463-545,629-666` | primitiva de umbral + smoke end-to-end | **PASS** |
| Ridge gain ponderado, SVD, alpha 1; advantage alpha 100 | `_wave57_phase_worker.py:108-115`; `_wave56_phase_worker.py:654-679` | end-to-end | **PASS** |
| Logistic positiva = daño, kwargs congelados, pesos, warning/finitud/reconstruccion | `wave57_tail_guard.py:19-31,105-110,123-206` | `test_wave57_tail_guard.py:33-65` | **PASS** |
| Shuffle condicional, mapping, Hamming global/ponderado/estrato, hashes | `wave57_tail_guard.py:209-275`; `_wave57_phase_worker.py:136-182` | `test_wave57_tail_guard.py:68-121` | **PASS** |
| Signos estrictos y hard-only identidad | `wave57_tail_guard.py:278-317` | `test_wave57_tail_guard.py:124-151` | **PASS** |
| Orden total y minimos de propuesta/autorizacion | `_wave57_phase_worker.py:227-383` | sin adversariales de granularidad | **PASS-CODE / COVERAGE GAP F3** |
| Shards por hash y seleccion local completa | `_wave57_phase_worker.py:386-453,718-738` | solo smoke nominal | **PASS-CODE / COVERAGE GAP F3** |
| Estimandos token-wise, sham promedio despues de evaluar replicas, bootstrap PCG64 | `_wave56_phase_worker.py:1020-1037`; `_wave57_phase_worker.py:667-717,789-815` | aserciones parciales end-to-end | **PASS** |
| Support individual por cinco sets | `wave57_tail_guard.py:325-350`; `_wave57_phase_worker.py:752-776` | solo rama no evaluable | **PARTIAL — F4** |
| FIT/SELECT/ADJUDICATE y no evaluabilidad terminal | `run_wave56_contextual_gate.py:324-366,1073-1237` | regresion heredada amplia + smoke | **PASS** |
| Runtime staged transitivamente cerrado | config `:169-183`; coordinator `:660-692` | arranque real `--help` desde staging | **PASS** |
| Compatibilidad honesta de nombres Wave 56 en boundary | config `:161-168`; receipts compartidos conservan namespace Wave 56 | smoke de runtime; declaracion no esta congelada | **PASS-CON-RIESGO F1** |
| Sources at HEAD y hashes de modulos staged | preparer `:400-419,665-745`; base worker `:132-202`; Wave57 worker `:840-854` | regresion heredada | **PASS**, condicionado a cerrar F1 |
| Replay de cores/freezes/configs y NPZ/bundles | coordinator `:933-978` | smoke omite freezes | **FAIL — F2** |
| Diagnosticos secundarios declarados | plan `:116-119,270-274` | ausentes | **PARTIAL — F5** |

## Comprobaciones favorables y conteos

1. **Runtime closure corregido.** La lista staged ahora incluye
   `wave49_schema.py`, `wave50_model.py` y `wave53_uncertainty.py`; el test crea
   el runtime cerrado y ejecuta el worker real con `--help` desde ese directorio
   (`tests/test_wave57_prospective.py:86-106`). Paso dentro de la suite focal.
2. **Preservacion SELECT.** Se preservan score, propuestas, acciones, threshold,
   flag hard-only y metricas del proposer; autorizaciones, acciones, threshold y
   metricas del guard; y score/autorizacion/accion/threshold/hard-only/metrica
   para las cinco replicas sham (`_wave57_phase_worker.py:455-545`). Las grillas
   y estados quedan en `selection_core.json` y `selection_freeze.json`
   (`:560-589`).
3. **Separacion de splits.** El worker prueba disjointness contra historia y
   fases anteriores por `pair_token` (`_wave56_phase_worker.py:488-549`); el
   worker restringido recibe solo el split materializado y los probes de splits
   presentes/futuros deben fallar por permisos.
4. **Support sets.** El test de union suma `50` tokens pero mantiene los cinco
   sets en `10 < 30`, y los cinco salen `NOT_EVALUABLE`; no hay prestamo de
   soporte.
5. **Tests focales y frontera compartida.** `126 passed in 211.62s` para los dos
   archivos Wave 57 y los dos archivos Wave 56 prospectivo/pre-oracle.
6. **Suite completa.** Una primera ejecucion con solo
   `CUDA_VISIBLE_DEVICES=''` termino `468 passed, 1 failed`; el fallo fue
   exclusivamente el guard ambiental de un test de solver que exige
   `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. Ese test aislado
   paso `1/1` al declarar el entorno requerido. Repeti despues toda la suite
   bajo esas variables y termino `469 passed in 260.23s`. No hubo fallos de
   Ola 57 ni de la frontera compartida bajo el entorno CPU prescrito.

## Comandos ejecutados

```text
git status --short
git rev-parse HEAD
git show --stat --oneline d29d1aa52256f112bc6fa2e1569dc9a7e20e77f7
git diff 04852d110d9cd20fb01d7ad1bfcaedbe67f402e8 d29d1aa52256f112bc6fa2e1569dc9a7e20e77f7 --

venv/bin/python -m pytest -q \
  tests/test_wave57_tail_guard.py \
  tests/test_wave57_prospective.py \
  tests/test_wave56_prospective.py \
  tests/test_wave56_preoracle_recovery.py
# 126 passed in 211.62s

CUDA_VISIBLE_DEVICES='' venv/bin/python -m pytest -q
# 468 passed, 1 failed in 265.49s; fallo por OPENBLAS_NUM_THREADS != 1

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 venv/bin/python -m pytest -q \
  tests/test_proportional_graph_solver_disentanglement.py::\
test_tiny_main_writes_complete_byte_exact_artifact_lifecycle
# 1 passed in 0.07s

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 venv/bin/python -m pytest -q
# 469 passed in 260.23s

# Probe en memoria de 10 mutaciones de config mediante ambas validaciones:
PYTHONPATH=src:experiments/geometria_proporcional venv/bin/python - <<'PY'
# ... carga config, aplica una mutacion por vez y llama
# validate_wave57_frozen_config + validate_prospective_config
PY
# accepted_count 10 of 10

# Probe sintetico de replay con fit_freeze divergente:
PYTHONPATH=src:experiments/geometria_proporcional:tests venv/bin/python - <<'PY'
# ... run_fit/run_select/run_adjudicate en TemporaryDirectory,
# copia, muta fit_freeze y llama compare_reference
PY
# fit_freeze_bytes_equal False
# compare_reference_all_exact True
# freeze_checks []
```

## SHA-256 de inputs auditados en HEAD `d29d1aa`

```text
db0d69ba197bfa0dcf853c9cecc1f848487ccc25910424036ebcde745a1ccedd  Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md
45dfc162dcae815226dfc76a1c569b861ca794777f8974970a67a338d4329188  Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/395_wave57_contextual_tail_guard_plan_audit.md
14bbfaff89a2d07ea7d1dfed82e347e3061c170090889b27d1235801dcc2d783  Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/396_wave57_contextual_tail_guard_plan_reaudit.md
78a5c87fc36a87f4c1aadb8c81308802e8bcfc81b4f401ea7a895e539b5e363d  experiments/geometria_proporcional/_wave56_phase_worker.py
3e8734f6d8b1c0c1c06451647c4b78edec330500777b9f655dad3aace20c1073  experiments/geometria_proporcional/_wave57_phase_worker.py
da775fc6e09119cfe8b9cbea43f207b7a3e6c03b775ae3ae02e9f36f5f64d0a6  experiments/geometria_proporcional/configs/wave57_contextual_tail_guard_fresh.json
796b5e8f580c1e98f0f1061ed16c7116cd2db752cc38a44c3651250d56c3bea7  experiments/geometria_proporcional/prepare_wave56_fresh.py
70827dca518ddec9a38c4f2da1ee66290cead0cd6417e5e6e67b860b68065693  experiments/geometria_proporcional/prepare_wave57_fresh.py
fc03f0a36d349108bc84ab1e31115a8fd784a888e5b0d1a393beb3e295dfb56f  experiments/geometria_proporcional/run_wave56_contextual_gate.py
821089654ff6d92a33af384848cf9ba43312a975cdd8bf9f3104be05fec52efc  experiments/geometria_proporcional/run_wave57_tail_guard.py
32906ce452fd153937d5ad68d3f3ea7e03175aab243cf1bc114d1001a1c7a37d  src/geometria_proporcional/wave57_tail_guard.py
a6a03e07b838375d32424b2b32dc355c7b3ded770f5690b5f4d5f2b6b45ecf55  tests/test_wave57_prospective.py
a2759c76caf1323e33a085a2a0cc6ae00271db03ef4baed5cb9edb2621d3cde4  tests/test_wave57_tail_guard.py
```

## Condicion para reauditoria

Una reauditoria puede aspirar a `PASS` cuando: (1) el validator rechace drift
de todo campo que afecta analisis o frontera, incluido el worker exacto; (2) el
replay compare los dos freezes y tenga pruebas adversariales; (3) la granularidad
`NOT_EVALUABLE` este cubierta por tests; (4) cada support set evaluable incluya
la summary sham declarada; y (5) los diagnosticos secundarios prometidos se
materialicen o el plan se revise antes de cualquier key draw.

Este dictamen no adjudica el patron prospectivo, no promueve arquitectura y no
declara `GO/NO-GO`; determina unicamente que el paquete auditado requiere
correccion antes del draw.
