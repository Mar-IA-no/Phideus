# R329 — Auditoría final independiente pre-key de Ola 56 Stage 1

> **Commit auditado:** `498932b2ac763fa18a2ccf0d495fd9341040e3dc`
> **Base focal:** `8bd0626b63335d68820a752bc736d4179337af8f`
> **Fecha:** 2026-09-03
> **Alcance:** cierre del finding de R328, no regresión de R325–R327 y evidencia CPU pre-key.
> **Resultado:** `PASS / TECHNICALLY DEFENSIBLE FOR OFFICIAL KEY EXTRACTION / PRE-KEY`

## Findings materiales

**Ninguno.** No encontré un defecto vigente que permita confundir los dos
outcomes terminales de `adjudicate`, finalizar un paquete promovido sin validar
su estado durable, reabrir labels o analytics durante la finalización, redibujar
después del escrow, o aceptar una recuperación divergente en cualquiera de los
seis límites durables expuestos por el preparador.

## Dictamen

La extracción de claves oficiales es técnicamente defendible en el commit
auditado. `COMPLETE` y `MONITOR_NOT_EVALUABLE` tienen estados físicos
pre-manifest distintos, se recuperan mediante una rama de finalización anterior
a toda materialización o ejecución analítica, y publican o revalidan el manifest
raíz idempotentemente. La matriz física del preparador continúa atravesando el
handler real de archivo y recuperación para sus seis cortes, conserva escrow y
claves después de su publicación, recupera exactamente los artefactos durables
y completa luego las tres fases sin redraw.

Este `PASS` habilita únicamente el paso técnico pre-key previsto por el plan. No
adjudica el patrón prospectivo, no promueve una arquitectura y no constituye una
decisión científica `GO/NO-GO`.

## 1. Cierre focal de R328: ambos outcomes terminales son recuperables

### 1.1 Estados pre-manifest distintos

La topología física distingue `adjudicate.complete` de
`adjudicate.not_evaluable` (`run_wave56_contextual_gate.py:320-335`). Sobre esa
distinción, `current_state` devuelve:

- `FINALIZATION_PENDING` sólo para el outcome físico `COMPLETE` sin manifest
  (`run_wave56_contextual_gate.py:336-338`);
- `MONITOR_FINALIZATION_PENDING` sólo para el outcome físico
  `MONITOR_NOT_EVALUABLE` sin manifest
  (`run_wave56_contextual_gate.py:339-340`).

La presencia prematura de un manifest sigue siendo inválida para cualquier otro
estado (`run_wave56_contextual_gate.py:341-343`). Por lo tanto, la corrección no
colapsa outcomes ni ensancha estados intermedios permitidos.

### 1.2 Semántica del estado promovido

La validación terminal recibe el outcome esperado explícitamente. Para éxito
exige el directorio `.complete` y los core files de `adjudicate`; para no
evaluabilidad exige `.not_evaluable`, `monitor_not_evaluable.json` y ausencia de
`analysis_core.json` y `result_arrays.npz`
(`run_wave56_contextual_gate.py:398-416`). En ambos casos exige además:

1. journal de la fase correcta en `READY_TO_PROMOTE`
   (`run_wave56_contextual_gate.py:417-419`);
2. igualdad exacta entre inventario actual, `inventory_after.json` e inventario
   durable del journal (`run_wave56_contextual_gate.py:420-431`);
3. árbol analítico sin symlinks y exactamente igual al inventario autenticado
   por `access_receipt.json` (`run_wave56_contextual_gate.py:432` y `:824-837`).

Esto valida identidad, completitud y no mutación del outcome promovido antes de
cerrar el paquete. En particular, un marker no evaluable aislado no basta, y un
resultado exitoso no puede hacerse pasar por no evaluable.

### 1.3 Finalización sin reapertura de labels ni analytics

La tabla de finalización mapea por separado los cuatro estados físicos —pendiente
y final de cada outcome— a `COMPLETE` o `MONITOR_NOT_EVALUABLE`
(`run_wave56_contextual_gate.py:1068-1074`). La rama valida el promovido,
publica/revalida el manifest y retorna en `run_wave56_contextual_gate.py:1075-1079`.
Ese retorno ocurre antes de `validate_transition`, `begin_or_resume`,
`materialize_labels` y `run_restricted_worker`, que comienzan recién en
`run_wave56_contextual_gate.py:1080-1120`. Por construcción, reintentar después
de cualquiera de las dos promociones no vuelve a materializar oracle, labels ni
analytics.

La suite reproduce crash después de promoción y después de manifest para
`COMPLETE`, exige bytes estables del manifest y envenena el worker para impedir
su relanzamiento (`tests/test_wave56_prospective.py:1331-1432`). Repite esas dos
fronteras para `MONITOR_NOT_EVALUABLE`, exige el marker sin `analysis_core`,
envenena el worker y agrega tamper del marker promovido antes de finalización
(`tests/test_wave56_prospective.py:1435-1581`).

### 1.4 Manifest raíz integral e idempotente

El inventario público cubre todo archivo regular del run excepto escrow, truth
sellada y el propio manifest (`run_wave56_contextual_gate.py:144-155`). El
payload raíz declara esas exclusiones; si el manifest ya existe, exige archivo
regular y equivalencia semántica completa en vez de reescribirlo; si falta, lo
publica atómicamente y lo relee (`run_wave56_contextual_gate.py:963-979`).

La prueba integral recalcula hash y tamaño de cada miembro, exige preparación,
inferencia y las tres fases, y comprueba que el estado preservado permite
reanálisis (`tests/test_wave56_prospective.py:1670-1699`). Las pruebas de ambas
finalizaciones conservan además los bytes exactos del manifest a través de
reintentos (`tests/test_wave56_prospective.py:1389-1400`, `:1419-1432`,
`:1510-1521` y `:1540-1552`).

## 2. Matriz de crashes del preparador

### 2.1 Seis límites durables y handler real

`execute_preparation` expone exactamente seis hooks:
`before_escrow`, `after_escrow`, `after_pre_generation_freeze`,
`after_generation`, `after_inference` y `after_preparation_freeze`
(`prepare_wave56_fresh.py:810-818`, `:862-864`, `:879-887` y `:916-921`). La
prueba parametrizada enumera los mismos seis, sin omisiones
(`tests/test_wave56_prospective.py:571-578`).

Cada inyección atraviesa `run_preparation_transaction`, no un simulador de
archivo (`tests/test_wave56_prospective.py:581-602`). Ese handler crea el output,
ejecuta la preparación, escribe `FAILURE.json` y archiva mediante `os.replace`
todo estado físico fallido (`prepare_wave56_fresh.py:950-1003`); `main` delega
directamente en ese mismo handler (`prepare_wave56_fresh.py:1006-1031`). La
prueba verifica un único archivo fallido y su receipt para cada corte
(`tests/test_wave56_prospective.py:604-608`).

### 2.2 Escrow, no-redraw y recuperación exacta

Antes del escrow no existe secreto durable y la repetición primaria es válida.
Después del escrow, cada caso exige `escrow_present=true`, obtiene las claves
exclusivamente mediante `validate_reused_escrow` y entra en modo `recovery`
(`tests/test_wave56_prospective.py:617-626`). La ejecución rechaza combinar
escrow reutilizado con `keys_override`, deriva las tres claves del escrow y
exige igualdad semántica total del escrow reconstruido
(`prepare_wave56_fresh.py:794-809`). La regla ordinaria de invocación también
rechaza un primary existente aun con `--force` y obliga a reutilizar todo escrow
primario previo (`prepare_wave56_fresh.py:284-300`; test en
`tests/test_wave56_prospective.py:807-819`).

La recuperación verifica el mismo contrato y las mismas tres claves, un
`preparation_receipt` terminal, y hash exacto para cada artefacto durable ya
visible antes del crash (`tests/test_wave56_prospective.py:628-650`). Entre los
seis casos exige además preparación exacta mediante `compare_preparation`, que
compara compromisos, protocolo, visibles, logits array-exact y
`preparation_freeze` (`tests/test_wave56_prospective.py:652-655`;
`prepare_wave56_fresh.py:753-776`). Finalmente, cada recovery atraviesa físicamente
`fit -> select -> adjudicate` y termina `COMPLETE`
(`tests/test_wave56_prospective.py:657-664`).

`FAILURE.json` se excluye correctamente de la igualdad porque documenta el
intento fallido; `generation_receipt.json` también puede cambiar sólo en su rol
operativo `primary/recovery`. El material científico durable —escrow/contrato,
benchmark visible y sellado, inferencia y freezes— queda ligado y exacto.

## 3. No regresión de la cadena R325–R327

| Finding previo | Estado en `498932b` | Evidencia vigente |
|---|---|---|
| R325-1, clausura dinámica de imports | **RESUELTO, sin regresión** | El runtime staged elimina rutas temporales o checkout-locales y no puede ser escrito por `nobody` (`_wave56_phase_worker.py:22-100`); cada módulo local debe resolver dentro del runtime y coincidir con el hash congelado (`:132-202`). |
| R325-2 / R326-2, inputs y upstream | **RESUELTO, sin regresión** | Se revalidan config embebida, fuentes, manifest, protocolo, visibles, inferencia, receipts, `upstream` e `historical_preflight.inputs` antes de labels (`run_wave56_contextual_gate.py:158-239`); la truth split-scoped se liga al manifest (`:242-255`). |
| R325-3 / R326-1, materialización y recovery | **RESUELTO, sin regresión** | Labels y receipt preexistentes se autentican contra split, rol, manifest, truth, configs y hash del label (`run_wave56_contextual_gate.py:534-598`); analytics se copia, valida, sincroniza y promueve atómicamente (`:840-871`). |
| R325-4 / R326-3, integración física | **RESUELTO** | El fixture cruza inferencia sin privilegios, materializador, tres journals/promociones (`tests/test_wave56_prospective.py:1003-1043`); la integración completa y la matriz transaccional cubren preparación, recovery y replay (`:463-665`). |
| R325-5, referencia replay | **RESUELTO, sin regresión** | Primary rechaza referencia; replay exige nombre, modo, sibling canónico, preparación exacta y compromisos iguales (`run_wave56_contextual_gate.py:922-960`), con negativo primary/replay (`tests/test_wave56_prospective.py:1584-1615`). |
| R325-6, disjunción | **RESUELTO, sin regresión** | La fase contrasta tokens históricos, fit, select y split actual por pares y preserva conteos/hashes (`_wave56_phase_worker.py:481-580`); el test físico prueba disjunción y rechazo de overlap (`tests/test_wave56_prospective.py:1618-1667`). |
| R325-7, posterior/risk/manifest | **RESUELTO, sin regresión** | `posterior_mass` y `action_risk` se derivan y preservan (`_wave56_phase_worker.py:545-579`); la prueba final exige esos arrays para monitor, fit y select además del manifest integral (`tests/test_wave56_prospective.py:1670-1699`). |
| R327-1, finalización post-promoción | **RESUELTO para ambos outcomes** | Estado, autenticación y rama temprana: `run_wave56_contextual_gate.py:311-343`, `:398-433` y `:1068-1079`; crashes/tamper: `tests/test_wave56_prospective.py:1331-1581`. |
| R327-2, matriz física de preparación | **RESUELTO, sin regresión** | Seis hooks en `prepare_wave56_fresh.py:810-921`; handler/archivo real en `:950-1003`; verificación completa en `tests/test_wave56_prospective.py:561-665`. |

El delta focal `8bd0626..498932b` sólo modifica la máquina terminal y agrega su
prueba, además de incorporar R328. No altera features, estimadores, thresholds,
brazos, bootstrap, criterios diagnósticos, preparación, materializador ni worker
analítico.

## 4. Verificación CPU

```text
git rev-parse HEAD
# 498932b2ac763fa18a2ccf0d495fd9341040e3dc

git status --short
# limpio antes de crear R329

git diff --check 8bd0626..498932b
# PASS

venv/bin/python -m py_compile \
  experiments/geometria_proporcional/prepare_wave56_fresh.py \
  experiments/geometria_proporcional/_wave56_infer_worker.py \
  experiments/geometria_proporcional/_wave56_oracle_materializer.py \
  experiments/geometria_proporcional/_wave56_phase_worker.py \
  experiments/geometria_proporcional/run_wave56_contextual_gate.py
# PASS

CUDA_VISIBLE_DEVICES='' venv/bin/python -m pytest -q \
  tests/test_wave56_prospective.py tests/test_wave56_contextual_gate.py
# 48 passed in 180.29s

CUDA_VISIBLE_DEVICES='' venv/bin/python -m pytest -q \
  tests/test_wave49_relational_benchmark.py tests/test_wave50_neural.py \
  tests/test_wave50_protocol.py tests/test_wave50_runner.py \
  tests/test_wave51_factored.py tests/test_wave52_policy.py \
  tests/test_wave53_uncertainty.py tests/test_wave54_joint_set.py \
  tests/test_wave55_policy_bridge.py tests/test_wave56_contextual_gate.py \
  tests/test_wave56_prospective.py
# 152 passed in 185.44s
```

Los tests físicos no fueron skipped en este host: corrieron como root con
`setpriv`. No se usó GPU, Mendieta, claves oficiales, oracle oficial ni outputs
oficiales. Al cierre de la auditoría no existía ningún directorio
`data/geometria_proporcional/wave56_contextual_gate_fresh_v1*`.

## Riesgos residuales no bloqueantes

- La evidencia física depende de root y `setpriv`; la propia suite declara skip
  en un host que carezca de esa frontera (`tests/test_wave56_prospective.py:557-560`).
  En el host auditado la frontera sí fue ejercitada y no hubo skips.
- La matriz llama directamente al mismo handler que usa `main`, en vez de probar
  `argparse` mediante un subprocess. Esto no reduce la evidencia sobre archivo,
  escrow, no-redraw o recuperación; una futura modificación del wiring CLI debe
  conservar el delegado visible en `prepare_wave56_fresh.py:1006-1031`.
- Persiste el riesgo estrecho ya documentado por R326 de caída antes de publicar
  escrow: puede quedar un intento archivado sin secretos, pero no habilita redraw
  después de compromiso ni expone labels. No afecta la defensa técnica del draw.

## Cierre pre-key

**PASS.** El commit `498932b` resuelve el último finding material de R328 y no
reabre ninguno de R325–R327. Puede procederse a la extracción de claves y a la
preparación oficial siguiendo literalmente el plan congelado y el HEAD auditado.
La interpretación del resultado y cualquier `GO/NO-GO` permanecen bajo autoridad
del usuario.
