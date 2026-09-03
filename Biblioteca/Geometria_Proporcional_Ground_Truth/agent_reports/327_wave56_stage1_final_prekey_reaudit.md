# R327 — Reauditoría final independiente pre-key de Ola 56 Stage 1

> **Commit auditado:** `d61dba6447b4030e5d3369cc8ee42730d2dcb888`
> **Base de comparación:** `64c874514f2273f454e2b9be6692e731166ee742`
> **Fecha:** 2026-09-03
> **Alcance:** cierre de los tres findings de R326, regresiones materiales y evidencia CPU pre-key.
> **Resultado:** `FAIL / REVISE / PRE-KEY`

## Dictamen

El delta resuelve la autenticación de truth señalada por R326, recorre los
inventarios `upstream` e `historical_preflight.inputs` antes de abrir una
transacción y agrega una integración física que cruza `execute_preparation`,
escrow recuperado, generación firmada, inferencia ciega sin privilegios, las
tres fases y replay exacto. La conversión inference-only de checkpoints a NPZ
es byte-determinista en el entorno auditado y el worker rechaza inventarios y
claves de estado ajenos al modelo.

No obstante, la extracción de claves oficiales todavía no es defendible. Un
crash después de promover `adjudicate.pending` a `adjudicate.complete`, pero
antes de publicar `artifact_manifest.json`, deja un estado que la máquina llama
`COMPLETE` aunque carece de su artefacto terminal integral. El mismo comando no
puede reanudarlo. Además, la matriz de crash de preparación pedida por R326
sigue reducida a un solo corte del escrow.

## Findings materiales

### 1. ALTA — La promoción final puede dejar un `COMPLETE` sin manifest y sin ruta de recuperación

**Observación.** `run_phase` fija el journal en `READY_TO_PROMOTE` y renombra
atómicamente la fase en `run_wave56_contextual_gate.py:1092-1106`. Sólo después
del rename escribe el manifest integral en
`run_wave56_contextual_gate.py:1107-1108`. `current_state` declara `COMPLETE`
por la mera presencia de los tres directorios `.complete`, sin exigir el
manifest raíz (`run_wave56_contextual_gate.py:311-335`), y
`validate_transition` rechaza volver a ejecutar `adjudicate` desde ese estado
(`run_wave56_contextual_gate.py:338-345`).

El test nuevo de promoción inyecta el crash únicamente en `fit`; después del
rename incluso espera que el reintento sea rechazado
(`tests/test_wave56_prospective.py:1138-1217`). El test del manifest sólo cubre
el camino sin crash (`tests/test_wave56_prospective.py:1306-1328`). No existe un
caso equivalente para la frontera adicional y exclusiva de `adjudicate`.

Un probe temporal sobre el pipeline físico sintético reprodujo exactamente la
ventana:

```text
first_error RuntimeError probe crash after adjudicate rename
state_after_crash COMPLETE
adjudicate_complete True
artifact_manifest False
resume_error RuntimeError cannot run adjudicate from terminal/current state COMPLETE
artifact_manifest_after_resume False
```

**Impacto.** En ese punto el monitor ya fue abierto y el resultado analítico ya
fue promovido. El draw queda sin el manifest terminal exigido por el plan y no
puede cerrarse mediante el ejecutable congelado; repararlo exige una acción
manual no protocolizada o un delta post-label, mientras que redibujar está
prohibido. Es el mismo tipo de pérdida de recuperabilidad que el plan pretendía
evitar con la máquina transaccional.

**Corrección mínima.** Hacer idempotente la finalización desde el estado físico
`COMPLETE`: validar íntegramente `adjudicate.complete`, publicar o revalidar
atómicamente `artifact_manifest.json` si falta y devolver `COMPLETE` sin volver
a materializar labels ni ejecutar el worker. Agregar un test de crash justo
después del rename de `adjudicate` y otro después de la publicación del
manifest; ambos deben reanudar con el mismo HEAD, producir el mismo manifest y
no reabrir cómputo analítico.

### 2. MEDIA — La evidencia de recovery de `execute_preparation` no cubre la matriz pedida por R326

**Observación.** El preparador ofrece cortes `before_escrow`, `after_escrow`,
`after_pre_generation_freeze`, `after_generation`, `after_inference` y
`after_preparation_freeze` (`prepare_wave56_fresh.py:779-921`). La integración
nueva sólo dispara `after_escrow`
(`tests/test_wave56_prospective.py:463-508`). Luego completa generación,
inferencia, fases y replay por el camino normal
(`tests/test_wave56_prospective.py:509-554`).

Los tests previos sí aportan recuperación separada tras materialización, copia
parcial de analytics y promoción de `fit`, pero no ejercitan crashes alrededor
de la publicación del pre-generation freeze, la generación ya sellada, la
inferencia ya promovida o el preparation freeze. Tampoco atraviesan el handler
real de `main` que escribe `FAILURE.json` y archiva el intento
(`prepare_wave56_fresh.py:950-987`). R326 había pedido explícitamente una matriz
antes y después de las publicaciones de escrow, freezes, labels, analytics y
fase, no sólo disponibilidad de hooks.

**Impacto.** La integración demuestra el happy path completo y una recuperación
del escrow, pero no constituye evidencia ejecutable suficiente de que todos los
cortes durables del preparador preserven no-redraw e idempotencia. Dada la
irrepetibilidad del draw oficial, esta deuda de recovery debe cerrarse antes de
extraer claves.

**Corrección mínima.** Parametrizar la integración física sobre todos los
cortes ya expuestos, incluyendo el flujo de archival/recovery de `main`, y
atestiguar en cada caso mismas claves, mismo contrato, ausencia de redraw,
preparación exacta y continuidad posterior de `fit -> select -> adjudicate`.

## Revisión focal de los findings de R326

| R326 | Estado en `d61dba6` | Evidencia |
|---|---|---|
| 1. Truth actual no ligada al compromiso congelado | **RESUELTO** | El coordinador compara hash+bytes del split con `manifest["files"]` antes de `current_state`, `begin_or_resume` o labels (`run_wave56_contextual_gate.py:242-255`, `956-1022`). El materializador repite la comparación antes de crear su staging y preserva esperado/observado más hash del manifest (`_wave56_oracle_materializer.py:91-134`). El tamper semántico falla sin crear `phases/` (`tests/test_wave56_prospective.py:955-1041`). |
| 2. Upstream e inputs históricos incompletamente revalidados | **RESUELTO EN CÓDIGO** | `validate_prepared_package` recorre ambas listas antes de la frontera transaccional (`run_wave56_contextual_gate.py:158-239`, `956-1022`). Un probe temporal alteró un `posterior_state.npz` congelado: obtuvo `RuntimeError: frozen upstream source changed` y `phases_exists False`. Falta convertir esta cobertura directa en test parametrizado por cada fila/clase de inventario. |
| 3. Integración física sin preparador completo | **PARCIAL** | El test nuevo usa el generador real con claves Ed25519 temporales, valida la attestation, ejecuta inferencia `setpriv`, las tres fases y replay exacto (`tests/test_wave56_prospective.py:303-407`, `463-554`). Cierra el recorrido funcional, pero no la matriz de recovery ni la frontera final descritas en los findings 1 y 2. |

## Checkpoints NPZ y schema inference-only

`strip_checkpoints` extrae únicamente `seed`, `output` y tensores de
`model_state`, elimina optimizer/history y escribe un NPZ comprimido con orden
estable (`prepare_wave56_fresh.py:558-590`). El staging exige un inventario
exacto y prohíbe nombres sensibles (`_wave56_infer_worker.py:36-57`). Al cargar,
el worker exige identidad, conjunto exacto de claves `state::*` frente a
`DualHeadDeepSet.state_dict()` y deja a `load_state_dict` validar dimensiones
(`_wave56_infer_worker.py:136-159`). `allow_pickle=False` conserva la frontera
de datos.

Dos conversiones independientes del mismo checkpoint produjeron, con NumPy
`2.3.5`, el mismo tamaño (`51,796` bytes) y SHA-256
`4dfaa7cf082d346d7949af5f98114c56c036c82b52b958c7474c57150a2123f4`.
Además, la integración primary/replay obliga a igualdad de
`preparation_freeze`, que incluye hashes del staging y receipts de checkpoints
(`prepare_wave56_fresh.py:889-916`, `753-775`). No encontré un finding material
en esta conversión.

## Verificaciones CPU

```text
git diff --check 64c8745..d61dba6
# PASS

python -m py_compile <cinco ejecutables Wave 56>
# PASS

pytest -q tests/test_wave56_prospective.py tests/test_wave56_contextual_gate.py
# 45 passed in 58.19s

pytest -q tests/test_wave49_relational_benchmark.py ... tests/test_wave56_prospective.py
# 149 passed in 62.23s
```

Los probes adicionales usaron únicamente fixtures temporales bajo `/tmp` y se
eliminaron al terminar. No se usó GPU, no se generaron claves oficiales, no se
materializó oracle oficial y no se crearon outputs oficiales.

## Cierre pre-key

**FAIL.** No extraer claves ni iniciar el primary. Resolver la finalización
recuperable de `adjudicate`, completar la matriz física de crashes y repetir
una reauditoría independiente sobre un HEAD limpio.
