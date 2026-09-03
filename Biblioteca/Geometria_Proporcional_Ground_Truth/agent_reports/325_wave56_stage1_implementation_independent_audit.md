# R325 - Auditoria independiente de implementacion de Ola 56 Stage 1

> **Commit auditado:** `0cca33404a242298ec0b3e21f8ea52989929e462`
> **Fecha:** 2026-09-03
> **Alcance:** config, preparador, materializador, workers, coordinador y tests pre-key.
> **Resultado:** `FAIL / REVISE / PRE-KEY`

## Findings

### 1. ALTA - El worker copiado puede importar codigo ajeno al manifest desde `/tmp`

**Evidencia.** `_wave56_phase_worker.py:20-27` deriva `REPO_ROOT` con
`Path(__file__).parents[2]` y ante la copia a
`/tmp/<workspace>/source/_wave56_phase_worker.py` obtiene `/tmp`. Luego antepone
`/tmp/experiments/geometria_proporcional` y `/tmp/src` a `sys.path`. El runtime
se construye en `run_wave56_contextual_gate.py:399-406`, pero esas dos rutas
globales, escribibles por otros usuarios, quedan por delante del runtime
auditado. El probe verificado produjo:

```text
repo_root=/tmp
src_root=/tmp/src
experiment_root=/tmp/experiments/geometria_proporcional
sys.path[:4]=[/tmp/experiments/geometria_proporcional, /tmp/src, "", <runtime auditado>]
```

**Impacto.** Un modulo residual o creado concurrentemente en `/tmp/src` o
`/tmp/experiments/geometria_proporcional` puede reemplazar
`run_wave56_retrospective` o un modulo `geometria_proporcional` y alterar fit,
seleccion o adjudicacion sin aparecer en `required_execution_sources`. La caida
de privilegios protege el oracle futuro, pero no la clausura del codigo que
calcula el resultado.

**Fix minimo.** Eliminar esos `sys.path.insert`; ejecutar desde el runtime
copiado con una ruta de import unica y no escribible por `nobody` (idealmente
modo aislado), y verificar en el receipt `module.__file__` mas hash para cada
modulo local importado. Agregar un test que falle si cualquier import local se
resuelve fuera del runtime temporal.

### 2. ALTA - Los inputs preparados y dos upstream decisivos no se revalidan contra el freeze

**Evidencia.** `run_wave56_contextual_gate.py:670-687` solo compara commit y hash
de config contra `preparation_freeze.json`. No valida
`benchmark_manifest_sha256`, `protocol_config_sha256`, `visible_sha256` ni
`inference_hashes`, aunque el preparador los congela en
`prepare_wave56_fresh.py:852-870`. `_copy_logits` acepta todos los NPZ que
coincidan nominalmente (`run_wave56_contextual_gate.py:383-389`). Ademas,
`_source_candidate` prioriza un path explicito sin exigir su hash congelado
(`run_wave56_contextual_gate.py:273-281`), y los argumentos CLI de policy
manifest y freeze Wave 54 evitan `_bound_upstream_path` en
`run_wave56_contextual_gate.py:677-686`.

**Impacto.** Visibles, logits, protocolo, utilidades de las 24 politicas o
`theta` del posterior pueden cambiar despues de `PREPARED` y aun ser consumidos.
Esto puede cambiar features, acciones, targets de decision y las seis
condiciones bajo el mismo freeze nominal, rompiendo la validez prospectiva y el
replay.

**Fix minimo.** Antes de crear cualquier `.pending` u abrir labels, validar un
inventario exacto (hash, size y ausencia de extras) de manifest, protocolo,
visibles e inferencia contra `preparation_freeze`. Eliminar los overrides de
upstream o exigir siempre los hashes de `source_binding`, incluso cuando el path
sea explicito. Ligar tambien el `phase_request` a esos hashes y comprobarlos en
el worker.

### 3. ALTA - Recovery no autentica una materializacion preexistente y no tolera copia parcial

**Evidencia.** Si hay crash despues de promover `authorized_labels` pero antes de
`update_journal`, el journal aun tiene inventario vacio. En reanudacion,
`materialize_labels` solo verifica dos nombres de archivo y devuelve el
directorio (`run_wave56_contextual_gate.py:360-367`); no comprueba el hash del
label contra el receipt ni split, rol, config, protocolo o truth sellada. Un
probe con `train.jsonl` alterado y un receipt deliberadamente falso fue
aceptado. Por otra parte, `_copy_worker_results` copia entradas directamente al
pending (`run_wave56_contextual_gate.py:576-584`). Como `access_receipt.json`
ordena primero, un crash durante la copia deja receipt mas outputs incompletos;
la reanudacion detecta el faltante en `run_wave56_contextual_gate.py:702-728`,
pero solo aborta y repetira el mismo aborto: no existe promocion atomica ni ruta
idempotente de recuperacion.

**Impacto.** La primera ventana puede consumir labels no autenticados; la
segunda convierte un fallo recuperable con el mismo HEAD y oracle en un pending
atascado. Ambas contradicen la garantia de validar todos los inputs/outputs
presentes antes de reanudar.

**Fix minimo.** Validar integramente toda materializacion ya existente contra su
receipt antes de actualizar el journal. Copiar los resultados del worker a un
subdirectorio transaccional, verificar su inventario completo, hacer `fsync` y
promoverlo atomicamente; agregar crash-injection inmediatamente antes y despues
de cada publicacion y update del journal.

### 4. ALTA - Falta la integracion sintetica prescrita del pipeline fisico

**Evidencia.** `tests/test_wave56_prospective.py:93-123` llama directamente
`run_fit`, `run_select` y `run_adjudicate`. Omite `prepare_wave56_fresh.py`, el
worker de inferencia, el materializador, `run_phase`, `setpriv`, los journals y
las promociones. El test de replay en `tests/test_wave56_prospective.py:326-353`
repite ese atajo. No hay inyeccion de crashes ni tests negativos de alteracion de
los hashes congelados. Esto no satisface la verificacion pre-draw exigida por el
plan en `WAVE_56_STAGE1_PROSPECTIVE_IMPLEMENTATION_PLAN.md:243-250`.

**Impacto.** Los tres findings anteriores quedan fuera de la suite aunque los
`131` tests Wave 49-56 pasen. No existe evidencia ejecutable de la cronologia
`prepare -> fit -> select -> adjudicate`, recovery o replay fisico.

**Fix minimo.** Agregar una integracion temporal que invoque los entrypoints y
la maquina real con un generador/oracle sintetico pequeno, mas una matriz de
crash/reanudacion, ausencia de material futuro, tamper de cada input congelado y
replay completo.

### 5. MEDIA - La prueba de replay que completa la condicion 6 no esta ligada al primary canonico

**Evidencia.** `--reference-dir` acepta cualquier directorio en adjudicacion
(`run_wave56_contextual_gate.py:74-76`). `compare_reference` compara los
artefactos analiticos, pero no la preparacion, compromisos de claves, modo ni
cronologia (`run_wave56_contextual_gate.py:587-631`). Luego la mera presencia de
ese argumento se convierte en `replay_exact=True` en
`run_wave56_contextual_gate.py:737-745`. No se exige que el run actual sea el
replay canonico ni que la referencia sea el primary usado por
`preparation_replay.json`.

**Impacto.** La condicion diagnostica 6 puede quedar satisfecha por una copia o
referencia analiticamente igual pero no por el replay canonico de las mismas
claves y preparacion.

**Fix minimo.** Rechazar `--reference-dir` en primary. En replay, exigir nombres
canonicos, `execution_mode=replay`, referencia coincidente con el source del
escrow, compromisos iguales y `preparation_replay.all_exact=true` antes de
comparar fases y emitir la condicion 6.

### 6. MEDIA - No se verifica la disjuncion entre los tres splits frescos

**Evidencia.** `load_inputs` comprueba solo el solapamiento del split actual con
tokens historicos (`_wave56_phase_worker.py:349-356`). Aunque select y
adjudicate reciben arrays de fases previas, nunca comparan sus `pair_token` con
los del split actual. El generador incorpora `split` al HMAC, por lo que no hay
evidencia de solapamiento actual, pero la propiedad requerida queda asumida y no
atestiguada.

**Impacto.** Una regresion o sustitucion de inputs podria introducir leakage
fit/select/monitor sin un fallo explicito.

**Fix minimo.** En select y adjudicate, comparar los tokens actuales contra los
tokens archivados de todas las fases previas, exigir interseccion vacia y
preservar los conteos/hashes en el freeze y tests.

### 7. MEDIA - El contrato final de artefactos no es integral

**Evidencia.** `data_arrays` preserva `posterior_actions`, pero no la masa
posterior ni `action_risk` (`_wave56_phase_worker.py:451-458`); el probe del NPZ
sintetico encontro `0` claves `posterior_mass`, `set_mass` o `action_risk`. El
plan exige preservar el posterior en
`WAVE_56_STAGE1_PROSPECTIVE_IMPLEMENTATION_PLAN.md:168-175`. Ademas,
`artifact_manifest.json` inventaria solo `adjudicate.pending`
(`run_wave56_contextual_gate.py:754-757`), no preparacion, inferencia, fit,
select ni el paquete completo.

**Impacto.** Parte del estado necesario para reanalisis exige recomputar desde
upstream, y el supuesto manifest integral no permite verificar el paquete
completo como una unidad.

**Fix minimo.** Persistir `posterior_mass` y `action_risk` por token/politica y
emitir un manifest final del arbol completo con inclusiones y exclusiones
explicitas, sin leer escrow ni truth sellada.

## Verificaciones sin finding material

- `HEAD` coincide con `0cca334`; el worktree estaba limpio y los siete archivos
  del commit estan trackeados.
- El manifest declarado cubre la clausura estatica de imports Python locales.
  El finding 1 concierne la resolucion efectiva del runtime, no una omision
  nominal de esa lista.
- El preparador realiza preflight antes de crear output, genera tres claves de
  32 bytes distintas, publica escrow `0600` y freeze atomicos con `fsync`, sella
  truth root `0700`, ejecuta inferencia como UID/GID `65534` sin capabilities y
  exige `PermissionError` sobre truth.
- El bloqueo de redraw tras escrow y el archival no destructivo estan
  implementados para las rutas ordinarias. Riesgo residual menor: un `SIGKILL`
  entre crear el output y publicar el escrow deja un primary vacio sin ruta
  automatica de recuperacion.
- Features, weighting `1/d_t`, Ridge `alpha=1/100`, shuffle por
  `(policy_index,d_t)`, thresholds `higher` con override estricto, siete brazos,
  bootstrap PCG64 pareado por token y signos de shards coinciden con el plan.
- Las seis condiciones usan los sentidos y bordes correctos. El probe de borde
  dio las seis `true`; con shuffled no evaluable, condicion 4 fue
  `NOT_EVALUABLE` y el agregado fue `null`.
- Los minimos producen los tres estados terminales y fit/select no evaluables
  impiden abrir la fase siguiente. Absent-support usa los cinco set indices
  congelados y minimo `30` sin incorporarlo al claim primario.
- El re-forward historico CPU fue array-exact para seeds `17/29/43`, incluidas
  ambas cabezas y targets, sobre `384` tokens.

## Comandos verificados

```text
git rev-parse HEAD
git status --short --branch
git diff --check HEAD^ HEAD
venv/bin/python -m py_compile <cinco scripts Wave 56>
venv/bin/python -m pytest -q tests/test_wave56_prospective.py
# 15 passed
venv/bin/python -m pytest -q tests/test_wave56_contextual_gate.py
# 12 passed
venv/bin/python -m pytest -q tests/test_wave49_relational_benchmark.py ... tests/test_wave56_prospective.py
# 131 passed
```

Tambien se ejecutaron probes temporales, eliminados al terminar, para clausura
dinamica de imports, recovery de labels, bordes de las seis condiciones,
contenido de `result_arrays.npz` y re-forward historico. No se generaron claves,
oracle, outputs oficiales ni trabajo GPU.

## Dictamen pre-key

**FAIL.** El commit no habilita aun la extraccion de claves. Las formulas
diagnosticas son correctas, pero la implementacion no garantiza que datos y
codigo congelados sean los efectivamente consumidos, y recovery no cumple el
contrato transaccional. Deben resolverse los findings 1-7, completar la
integracion fisica y realizar una reauditoria independiente sobre un nuevo HEAD
limpio antes de cualquier draw fresco.
