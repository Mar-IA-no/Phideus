# R328 — Reauditoría independiente de recuperación terminal de Ola 56 Stage 1

> **Commit auditado:** `8bd0626b63335d68820a752bc736d4179337af8f`
> **Base de comparación:** `d61dba6447b4030e5d3369cc8ee42730d2dcb888`
> **Fecha:** 2026-09-03
> **Alcance:** cierre de los dos findings de R327, regresiones materiales y evidencia CPU pre-key.
> **Resultado:** `FAIL / REVISE / PRE-KEY`

## Finding material

### 1. ALTA — `MONITOR_NOT_EVALUABLE` perdió su semántica terminal y conserva la ventana irrecuperable de R327

**Observación.** El plan define dos finales legítimos de `adjudicate`: `COMPLETE`
con `analysis_core.json + manifest` y `MONITOR_NOT_EVALUABLE` con
`monitor_not_evaluable.json + manifest` (`WAVE_56_STAGE1_PROSPECTIVE_IMPLEMENTATION_PLAN.md:101-112`).
También exige promover atómicamente el segundo cuando no se cumplen los mínimos
y cerrar el paquete sin adjudicación (`WAVE_56_STAGE1_PROSPECTIVE_IMPLEMENTATION_PLAN.md:191-206`).

El código nuevo reconoce la ausencia del manifest como `FINALIZATION_PENDING`
sólo cuando la topología contiene `adjudicate.complete`; para cualquier otro
estado, la presencia de `artifact_manifest.json` es un error
(`run_wave56_contextual_gate.py:311-341`). La ruta idempotente de finalización
también acepta sólo `FINALIZATION_PENDING` o `COMPLETE`
(`run_wave56_contextual_gate.py:987-1055`). Sin embargo, `run_phase` publica el
manifest después de promover **cualquier** resultado de `adjudicate`, incluido
`adjudicate.not_evaluable` (`run_wave56_contextual_gate.py:1103-1144`).

Dos probes físicos temporales con el materializador, worker `setpriv` y
promociones reales reprodujeron ambos fallos:

```text
# camino normal con sealed_monitor_tokens por encima de la población sintética
fit FIT_COMPLETE
select SELECT_COMPLETE
adjudicate MONITOR_NOT_EVALUABLE
manifest True
current_state_error RuntimeError artifact manifest exists before analytical completion

# crash después de promover adjudicate.not_evaluable y antes del manifest
first_error RuntimeError crash before monitor-not-evaluable manifest
state_after_crash MONITOR_NOT_EVALUABLE
manifest_after_crash False
resume_error RuntimeError cannot run adjudicate from terminal/current state MONITOR_NOT_EVALUABLE
```

La suite no observa esta regresión. Su test de monitor no evaluable invoca
directamente `worker.run_adjudicate` y sólo comprueba la ausencia de
`analysis_core.json` (`tests/test_wave56_prospective.py:882-891`); el único test
de `current_state` para un final no evaluable usa `FIT_NOT_EVALUABLE`
(`tests/test_wave56_prospective.py:916-923`).

**Impacto.** Si el monitor fresco no alcanza el mínimo prospectivo, el camino
sin crash devuelve el outcome correcto pero deja un paquete que la máquina de
estados ya no puede reconocer. Si hay un crash en la frontera promoción →
manifest, el mismo comando no puede completar el paquete. En ambos casos el
monitor ya fue abierto y el draw no puede redibujarse; por tanto, la
recuperabilidad terminal exigida por R327 no está resuelta para todo outcome
permitido y la extracción de claves oficiales no es técnicamente defendible.

**Corrección mínima.** Modelar la finalización pendiente y completa de ambos
outcomes de `adjudicate`. El reintento de `MONITOR_NOT_EVALUABLE` debe validar
el journal e inventario promovidos, exigir `monitor_not_evaluable.json` y la
ausencia de `analysis_core.json`, publicar o revalidar idempotentemente el
manifest sin rematerializar labels ni relanzar analytics, y devolver
`MONITOR_NOT_EVALUABLE`. Agregar crashes antes y después del manifest para este
outcome, más tamper tests del inventario promovido.

## Revisión de los findings de R327

| R327 | Estado en `8bd0626` | Evidencia |
|---|---|---|
| 1. Promoción final sin recovery ni manifest | **PARCIAL / NO RESUELTO** | Para `adjudicate.complete`, `current_state` distingue `FINALIZATION_PENDING`, la ruta temprana valida el inventario promovido y publica/revalida el manifest antes de cualquier materialización (`run_wave56_contextual_gate.py:311-415`, `:945-961`, `:987-1063`). El test de crashes antes/después del manifest prueba idempotencia y bloquea el relanzamiento del worker (`tests/test_wave56_prospective.py:1331-1432`). Un probe adicional alteró `analysis_core.json` tras promoción y el reintento abortó con `promoted adjudicate differs from its terminal inventory`. El finding sigue abierto por el outcome permitido `MONITOR_NOT_EVALUABLE`, descrito arriba. |
| 2. Matriz física de recovery del preparador | **RESUELTO** | `run_preparation_transaction` contiene el mismo handler que usa `main`: escribe `FAILURE.json`, archiva el estado físico y vuelve a propagar el error (`prepare_wave56_fresh.py:950-1003`); `main` delega en él (`:1006-1031`). La matriz recorre los seis cortes expuestos —antes/después de escrow, después de pre-generation freeze, generación, inferencia y preparation freeze—, verifica archivo fallido, escrow/contrato/claves, artefactos durables regenerados, igualdad exacta de preparación y luego ejecuta `fit → select → adjudicate` en cada caso (`tests/test_wave56_prospective.py:557-664`). El test integral separado conserva recovery + replay exacto (`tests/test_wave56_prospective.py:459-554`). |

La autoexclusión del manifest es estable: `public_run_inventory` omite
`artifact_manifest.json`, escrow y truth sellada (`run_wave56_contextual_gate.py:144-155`),
y `write_public_artifact_manifest` compara el payload completo cuando el archivo
ya existe en lugar de reescribirlo (`run_wave56_contextual_gate.py:945-961`). No
encontré regresiones materiales adicionales en esa ruta, en la validación del
inventario de `adjudicate.complete` ni en el refactor de
`run_preparation_transaction`.

## Verificación CPU

```text
git rev-parse HEAD
# 8bd0626b63335d68820a752bc736d4179337af8f

git diff --check d61dba6..8bd0626
# PASS

python -m py_compile <cinco ejecutables Wave 56>
# PASS

pytest -q tests/test_wave56_prospective.py tests/test_wave56_contextual_gate.py
# 47 passed in 170.26s

pytest -q tests/test_wave49_relational_benchmark.py ... tests/test_wave56_prospective.py
# 151 passed in 174.41s
```

Los probes usaron únicamente paquetes sintéticos temporales bajo `/tmp` y se
eliminaron al terminar. No se usó GPU, Mendieta, claves oficiales, oracle
oficial ni outputs oficiales.

## Cierre pre-key

**FAIL.** No extraer claves ni iniciar el primary. La matriz física de
preparación de R327 queda cerrada y la finalización `COMPLETE` quedó endurecida,
pero el outcome terminal `MONITOR_NOT_EVALUABLE` todavía rompe tanto la lectura
normal del estado como el recovery entre promoción y manifest. Este dictamen es
exclusivamente técnico pre-key; no constituye GO/NO-GO científico.
