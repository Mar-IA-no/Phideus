# R326 - Reauditoria focal independiente de implementacion de Ola 56 Stage 1

> **Commit auditado:** `64c874514f2273f454e2b9be6692e731166ee742`
> **Delta revisado:** `0cca33404a242298ec0b3e21f8ea52989929e462..64c874514f2273f454e2b9be6692e731166ee742`
> **Fecha:** 2026-09-03
> **Alcance:** resolucion de findings 1-7 de R325, regresiones materiales y evidencia CPU pre-key.
> **Resultado:** `FAIL / REVISE / PRE-KEY`

## Dictamen

El delta cierra materialmente la clausura dinamica de imports, la copia
transaccional de resultados, el binding del replay canonico, la disjuncion de
splits y el contrato de arrays/manifest. Sin embargo, no habilita aun la
extraccion de claves: queda un fallo alto que permite materializar una truth
sellada modificada despues de `PREPARED`, y quedan dos faltantes medios en la
revalidacion integral de upstream y en la integracion fisica prescrita del
preparador.

## Findings materiales

### 1. ALTA - La materializacion se autentica contra la truth actual, no contra la truth congelada

**Evidencia.** La preparacion valida el manifest y la attestation antes de
publicar `preparation_freeze.json` (`prepare_wave56_fresh.py:787-803`) y congela
el hash del manifest (`prepare_wave56_fresh.py:852-869`). Ese manifest contiene
hash y bytes de cada truth sellada (`wave49_generator.py:520-531` y
`wave49_generator.py:601-604`). No obstante, el materializador lee directamente
la truth presente y escribe su hash actual en el receipt
(`_wave56_oracle_materializer.py:92-124`). La revalidacion posterior vuelve a
calcular el hash de ese mismo archivo actual y lo compara con el receipt recien
generado (`run_wave56_contextual_gate.py:441-480`); nunca lo compara con el
registro sellado dentro del manifest congelado.

Un probe temporal altero semanticamente `benchmark/sealed/train.jsonl` despues
de `PREPARED`, sin cambiar `benchmark/manifest.json`. `run_phase(...,
enforce_sources=True)` creo `fit.pending/authorized_labels/train.jsonl`, avanzo
el journal a `ORACLE_MATERIALIZED` y emitio un receipt cuyo
`sealed_truth_sha256` coincidia con la truth alterada. El manifest permanecio
byte-identico y con su hash congelado.

**Impacto.** Una corrupcion o sustitucion de truth entre preparacion y apertura
de fase puede cambiar labels, targets y adjudicacion bajo el mismo commit,
config, manifest y compromisos declarados. El receipt es autoconsistente, pero
no autentica que el material consumido sea el comprometido antes de labels. Se
rompen la validez prospectiva, el contrato de recovery con el mismo material y
la semantica del replay.

**Fix minimo.** Antes de computar el oracle de cada split, exigir que el hash del
manifest coincida con `preparation_freeze`, verificar la attestation/manifest
congelados y comparar hash mas bytes de la truth split-scoped contra
`manifest["files"]["sealed/<split>.jsonl"]`. El receipt debe preservar ese
registro esperado y el observado. Agregar un test que altere un campo semantico
de la truth con `enforce_sources=True` y demuestre ausencia de `.pending` y de
labels.

### 2. MEDIA - La revalidacion pre-label no recorre todo el upstream congelado

**Evidencia.** `preparation_freeze.json` conserva `upstream` y los inputs del
re-forward historico (`prepare_wave56_fresh.py:852-869`). La funcion que declara
revalidar todos los inputs de preparacion comprueba config, fuentes de ejecucion,
manifest, protocolo, visibles, inferencia y receipts
(`run_wave56_contextual_gate.py:158-208`), pero no itera ni `upstream` ni
`historical_preflight.inputs`. Mas tarde se verifican los dos upstream
consumidos directamente y solo tres fuentes de tokens historicos
(`run_wave56_contextual_gate.py:343-393` y
`run_wave56_contextual_gate.py:939-964`). Quedan sin revalidar, entre otros, los
artefactos congelados de Stage 0 y los inputs que respaldan el re-forward.

El probe temporal preparo un paquete autoconsistente con un registro upstream
tipo `stage0/analysis_core.json`, altero luego ese archivo y comprobo que
`run_phase(..., enforce_sources=True)` alcanzo `begin_or_resume`; el drift no fue
rechazado antes de la frontera `.pending`.

**Impacto.** Los archivos omitidos ya no alimentan directamente las formulas de
fase, por lo que el probe no demuestra cambio inmediato del resultado. Si
demuestra que pueden abrirse labels cuando la base upstream que el freeze
declara recuperable e integra ya no coincide con la preparacion auditada. Esto
degrada trazabilidad y recovery, y deja una frontera insegura ante futuras
dependencias de esos registros.

**Fix minimo.** Revalidar antes de `begin_or_resume` cada fila de `upstream` y
`historical_preflight.inputs` por path regular, ausencia de symlink, hash y
bytes. Si algun artefacto se considera solo evidencia historica y no parte del
contrato de fase, excluirlo explicitamente del freeze operativo en lugar de
mantener un inventario que no se verifica.

### 3. MEDIA - La integracion fisica sigue sin ejecutar el preparador completo

**Evidencia.** El nuevo helper sintetico escribe a mano benchmark, manifest,
truth, `preparation_freeze.json`, `generation_receipt.json` y
`preparation_receipt.json` (`tests/test_wave56_prospective.py:133-269`). Invoca
`stage_and_infer`, y luego la fixture atraviesa el coordinador real, el
materializador, `setpriv`, journals y promociones para primary/replay
(`tests/test_wave56_prospective.py:278-330`). Pero ningun test invoca
`execute_preparation` (`prepare_wave56_fresh.py:764`) ni `main`
(`prepare_wave56_fresh.py:911`). Por eso el pipeline probado no es literalmente
el `prepare -> fit -> select -> adjudicate` requerido en el plan
(`WAVE_56_STAGE1_PROSPECTIVE_IMPLEMENTATION_PLAN.md:243-250`).

La inyeccion de crash agregada cubre la copia parcial de analytics y su recovery
(`tests/test_wave56_prospective.py:864-913`), y el caso de materializacion
preexistente cubre el hueco entre promocion de labels y journal. No hay una
integracion del preparador con escrow/recovery/replay ni una matriz alrededor de
sus publicaciones y del cierre de fase.

**Impacto.** La suite puede pasar aunque falle la composicion real entre
preflight, escrow/freeze, generador, inferencia ciega y primera fase. Es evidencia
pre-key incompleta, aunque las fronteras de fase incorporadas si quedaron
ejercitadas fisicamente.

**Fix minimo.** Extraer una entrada de preparacion inyectable que permita un
generador/protocolo sintetico pequeno y secretos de fixture fijos, y usarla en
`/tmp` para primary, recovery y replay antes de correr las tres fases reales.
Agregar crash-injection al menos antes y despues de las promociones de escrow,
freeze, labels, analytics y fase, verificando recovery idempotente y no-redraw.

## Verificacion uno por uno de R325

| R325 | Estado en `64c8745` | Evidencia focal |
|---|---|---|
| 1. Imports fuera del runtime | **RESUELTO** | El worker fija `RUNTIME_ROOT`, elimina rutas temporales/checkout, rechaza runtime escribible por `nobody` y verifica `module.__file__` mas hash (`_wave56_phase_worker.py:22-100`, `132-202`). El probe con `PYTHONPATH` envenenado resolvio el modulo retrospectivo dentro del runtime staged. |
| 2. Preparacion/upstream sin revalidar | **PARCIAL** | Manifest, protocolo, visibles, inferencia y ambos upstream decisivos ya fallan antes de `.pending`; queda abierto el finding medio 2 para el resto del inventario upstream. |
| 3. Recovery/materializacion/copia parcial | **PARCIAL** | Label y receipt preexistentes se validan (`run_wave56_contextual_gate.py:441-500`) y analytics se publica por staging, validacion, `fsync` y `os.replace` (`727-774`). Queda abierto el finding alto 1: la truth fuente no esta ligada al compromiso pre-label. |
| 4. Integracion fisica | **PARCIAL** | La suite cruza inferencia, materializador, `setpriv`, fases y replay, pero omite el entrypoint completo de preparacion y su recovery; finding medio 3. |
| 5. Referencia replay canonica | **RESUELTO** | Primary rechaza referencia; replay exige nombre, modo, sibling primary, preparacion exacta y mismos compromisos antes de abrir monitor (`run_wave56_contextual_gate.py:825-863`, `905-938`). |
| 6. Disjuncion fresh/historica | **RESUELTO** | Se comparan tokens historicos, fit, select y split actual, con hashes/conteos preservados (`_wave56_phase_worker.py:481-542`, `545-580`). El test negativo de overlap falla. |
| 7. Posterior/risk/manifest | **RESUELTO** | `posterior_mass` y `action_risk` se calculan y archivan para los tres roles (`_wave56_phase_worker.py:564-572`, `675-682`); el manifest final cubre todo el arbol publico y excluye secretos de forma explicita (`run_wave56_contextual_gate.py:866-879`, `1053-1055`). |

No se detectaron regresiones materiales en features, modelos, seleccion,
thresholds, siete brazos, bootstrap, condiciones diagnosticas o estados
`NOT_EVALUABLE` dentro del delta revisado.

## Riesgos residuales

- Existe una ventana TOCTOU estrecha entre hash y copia de visibles/upstream/logits
  (`run_wave56_contextual_gate.py:343-359`, `434-438`, `504-519`). Los artefactos
  oficiales son root-owned y el runtime staged se vuelve no escribible por el
  worker, por lo que no se eleva como finding separado; conviene verificar el
  hash del destino staged contra el freeze y no solo contra `staged_input_hashes`
  calculado despues de copiar.
- Persiste el riesgo menor ya documentado en R325 de un `SIGKILL` entre crear el
  output primario y publicar escrow: queda un directorio vacio sin recovery
  automatica. No habilita redraw silencioso ni expone labels.
- Los tests fisicos dependen de root y `setpriv`; en este host se ejecutaron sin
  skip. En CI sin esos privilegios la evidencia de frontera debe tratarse como
  ausente, no como PASS implicito.

## Comandos y resultados

```text
git rev-parse HEAD
# 64c874514f2273f454e2b9be6692e731166ee742

git status --short
# limpio antes de redactar este informe

git diff --check 0cca334..64c8745
# sin errores

venv/bin/python -m py_compile \
  experiments/geometria_proporcional/prepare_wave56_fresh.py \
  experiments/geometria_proporcional/_wave56_infer_worker.py \
  experiments/geometria_proporcional/_wave56_oracle_materializer.py \
  experiments/geometria_proporcional/_wave56_phase_worker.py \
  experiments/geometria_proporcional/run_wave56_contextual_gate.py
# PASS

venv/bin/python -m pytest -q tests/test_wave56_prospective.py
# 30 passed in 29.05s

venv/bin/python -m pytest -q tests/test_wave56_contextual_gate.py
# 12 passed in 1.05s

venv/bin/python -m pytest -q \
  tests/test_wave49_relational_benchmark.py tests/test_wave50_neural.py \
  tests/test_wave50_protocol.py tests/test_wave50_runner.py \
  tests/test_wave51_factored.py tests/test_wave52_policy.py \
  tests/test_wave53_uncertainty.py tests/test_wave54_joint_set.py \
  tests/test_wave55_policy_bridge.py tests/test_wave56_contextual_gate.py \
  tests/test_wave56_prospective.py
# 146 passed in 34.39s
```

Se ejecutaron ademas cuatro probes Python temporales, todos eliminados al
terminar:

1. `build_phase_runtime` mas subprocess `setpriv` con un
   `run_wave56_retrospective.py` falso antepuesto en `PYTHONPATH`: retorno `0` y
   resolucion exclusiva al runtime staged.
2. `historical_preflight` sobre Wave 50/51/52 reales: `384` tokens, seeds
   `17/29/43`, ambas cabezas y targets array-exact.
3. Paquete sintetico con upstream congelado y luego alterado: alcanzo
   `begin_or_resume`, confirmando el finding medio 2.
4. Paquete sintetico con `sealed/train.jsonl` alterado tras `PREPARED`: labels
   materializados, journal `ORACLE_MATERIALIZED` y receipt ligado al hash
   alterado, confirmando el finding alto 1.

No se generaron claves ni outputs oficiales. La suite/probes ejercitaron solo
materializacion sintetica temporal bajo `/tmp`; no se materializo el oracle de
ningun draw real. No se uso GPU.

## Cierre pre-key

**FAIL.** No extraer claves ni iniciar primary. Resolver los tres findings y
repetir una reauditoria focal independiente sobre un HEAD limpio. El bloqueo
principal es ligar la truth materializada al compromiso congelado, no solo al
receipt producido durante esa misma materializacion.
