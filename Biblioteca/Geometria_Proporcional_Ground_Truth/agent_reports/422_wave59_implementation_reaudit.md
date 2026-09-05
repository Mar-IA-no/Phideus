# Ola 59 — reauditoría independiente de implementación posterior a R421

## Dictamen: REVISE

El commit estable `3755bd4cad0c0edf56f4d05cc7a1f196a894dfba`
resuelve dos de los cinco P1 de R421, resuelve el contraste monitor P2 y aplica
límites efectivos al runner analítico y al worker de inferencia. No alcanza
`PASS`: persisten tres defectos P1 reproducibles. La reanudación que se presenta
como hash-idéntica puede reautenticarse después de alterar el resultado
científico; el binding auto-normalizado de la config no ancla el runner al blob
Git auditado; y todo cierre canónico `NOT_EVALUABLE` anterior a
MONITOR-EVALUATE falla al construir su manifest terminal. El P2 de recursos
queda además incompleto en el coordinador del preparador.

Este dictamen es técnico y prospectivo. No interpreta resultados científicos,
no promueve una arquitectura y no toma ninguna decisión `GO/NO-GO`.

## Identidad y alcance

- Commit auditado: `3755bd4cad0c0edf56f4d05cc7a1f196a894dfba`.
- Padre directo: `c7442ee822e57c925bdc8de9e4fc9061cb2dac42`.
- Subject: `Harden Wave 59 prospective authority boundaries`.
- Plan completo, 626 líneas, SHA-256
  `7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb`.
- R421 completo, SHA-256
  `2b6807e6d724ebc3033751e06395920a57a9a8b528deb9e974e625577b777104`.
- El diff estable contiene exactamente los cinco paths autorizados y suma
  1110 inserciones y 58 eliminaciones. `git diff --check` terminó con exit 0.
- La auditoría releyó completos el plan, R421 y los cinco archivos modificados.
  Todos los probes usaron copias efímeras bajo `/tmp`, CPU, como máximo cuatro
  hilos y `CUDA_VISIBLE_DEVICES=''`. No se consultaron GPU, web ni Mendieta y no
  se tocaron outputs canónicos.

## Resultado de las siete condiciones de R421

| Condición | R422 | Evidencia |
|---|---|---|
| P1: arrays de validation/monitor autenticados por freeze | Resuelta | `validate_freeze` contrasta hashes y cobertura; los dos tests negativos pasan (`_wave59_phase_worker.py:169-203,458-468,853-860`; `tests/test_wave59_prospective.py:273-342`). |
| P1: binding pre-draw de implementación y sources | Parcial, P1 persiste | El status solo ya se rechaza, pero el self-hash recalculable no ancla el runner a Git; reproducción debajo. |
| P1: resume hash-idéntico rechaza estado alterado | Parcial, P1 persiste | Una mutación aislada se rechaza, pero mutar resultado+journal+inventario se acepta; reproducción debajo. |
| P1: failure schema y terminales closed-world | Parcial, P1 persiste | El preparador usa schema redacted e inventario, pero el manifest `NOT_EVALUABLE` real es imposible antes de fase 5; reproducción debajo. |
| P1: recovery amendment exacto y contexto autenticado | Resuelta | El contexto proviene de `preparation_freeze`, exige presencia simétrica y SHA autorizado, y el amendment entra a `scientific_exact` (`runner:784-794,864-889`). |
| P2: contrastes contra `HGB-PROPOSER-ONLY` | Resuelta | Se publican `mean_vs_hgb_proposer_only` y `tail_vs_hgb_proposer_only` sobre las cuatro métricas (`_wave59_phase_worker.py:871-888`). |
| P2: 1800 s/8 GiB/3600 s y CUDA invisible | Parcial, P2 persiste | El runner y el hijo de inferencia están limitados; el coordinador del preparador no lo está y no oculta CUDA para sí mismo. |

## Findings prospectivamente invalidantes

### P1 — el failure inventory no autentica la reanudación

`_validate_failure_inventory` prueba que el árbol coincide con los hashes que
declara el propio `failure_inventory.json`, y `_validate_resumed_journals`
prueba outputs contra journals guardados en ese mismo árbol
(`run_wave59_hgb_guard_bracket.py:604-652`). Ningún hash de confianza externo al
árbol fallido ancla el inventario ni los journals. Por lo tanto, actualizar los
tres niveles después de una mutación crea otra historia autoconsistente que el
resume acepta. Esto contradice la única transición autorizada, la reanudación
hash-idéntica (`plan:330-347`).

**Reproducción ejecutada.** Sobre una copia temporal del output físico COMPLETE
creado por la suite:

1. `archive_failed_attempt` produjo el intento fallido;
2. se cambió `analysis.json["scientific_decision"]` de `null` a
   `"TAMPERED_AFTER_FAILURE"`;
3. se actualizó `journals/monitor_evaluate.json.output_sha256.analysis.json`;
4. se actualizaron sólo los records de esos dos paths en
   `failure_inventory.json`;
5. se ejecutaron `restore_identical_hash_attempt` y `execute`.

Resultado observado:

```json
{"accepted": true, "journal_matches": true, "scientific_decision_after_resume": "TAMPERED_AFTER_FAILURE"}
```

Una variante mínima alteró `source_bindings.json`, recalculó sólo su record de
inventario y también fue restaurada: `{"accepted":true,
"restored_source_bindings":{"tampered":true}}`.

**Corrección mínima.** La autoridad de resume debe vivir fuera del árbol que
autentica. Persistir antes de archivar un root digest inmutable o autenticado en
un ledger separado, y contrastarlo antes de copiar cualquier byte; alternativamente,
no permitir resume desde fases cuya autoridad no tenga ese ancla. El validador
debe exigir schema cerrado de inventory/journals, archivos físicos regulares y
la identidad exacta de todos los records. Añadir un test que cambie
`analysis.json`, journal e inventario juntos: hoy el test sólo cambia el primer
nivel (`tests/test_wave59_prospective.py:225-252`).

### P1 — el self-binding de config es recalculable y el runner no exige el blob Git auditado

El digest de config normaliza a ceros su propia entrada y luego hashea el resto
(`wave59_hgb_guard_bracket.py:247-254`). Eso resuelve la recursión, pero no crea
una raíz de confianza: quien cambia config o sources puede recalcular el mapa.
El preparador sí exige sources limpios e idénticos a HEAD
(`prepare_wave56_fresh.py:429-448,733-782`), pero el runner sólo verifica los
hashes declarados por esa misma config, que el commit mencionado exista y que
su texto aparezca en un Markdown con la cadena PASS
(`run_wave59_hgb_guard_bracket.py:236-268`). No exige que el config ni cada
source sean blobs del commit auditado, ni worktree limpio. Después de PREPARE,
`_ensure_preparation_authority` tampoco contrasta el `config_sha256` ni la
config prospectiva contenidos en `preparation_freeze.json`; sólo compara los
bundles (`runner:1373-1403`).

**Reproducción ejecutada.** En un repo Git efímero se construyó una config
frozen válida, se la comprometió, se cambió después
`models.random_states.hgb_proposer` a `999999`, se recalculó sólo el digest
auto-normalizado y se llamó `validate_execution_bindings`. El archivo estaba
dirty respecto de Git y fue aceptado:

```json
{"accepted_post_commit_drift": true, "changed_hgb_proposer_seed": 999999, "dirty": "M experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket.json"}
```

El mismo mecanismo permite reemplazar un execution source y actualizar su hash
en la config: `build_runtime` copiaría la fuente actual, no el blob auditado
(`runner:309-325`).

**Corrección mínima.** En cada entrada al runner, resolver el commit aceptado y
exigir para todo `required_execution_source`, incluida la config normalizada,
igualdad contra el blob Git del commit/commit de freeze autorizado; exigir
además que la config usada coincida con la config autenticada por
`preparation_freeze.json`. No usar un mapa contenido exclusivamente en el mismo
archivo mutable como raíz de autoridad. El test debe modificar config y source,
recalcular todos los hashes internos y aun así obtener rechazo.

### P1 — los manifests `NOT_EVALUABLE` canónicos fallan por construcción

`_finalize_terminal` escribe primero `runtime.json` y luego intenta el manifest
canónico (`runner:1417-1423`). Para `NOT_EVALUABLE`, el manifest llama
`_failure_coverage`; allí `runtime.json` está asignado rígidamente al índice 5,
MONITOR-EVALUATE (`runner:1318-1334`). Si el último journal es FIT,
CALIBRATE-SCORES o MONITOR-APPLY, el runtime terminal recién escrito se clasifica
como artefacto futuro y el manifest aborta. En consecuencia, un mínimo fallido
no queda como el terminal científico cerrado exigido por el plan
(`plan:323-328,501-503`). El camino de tests usa outputs no canónicos y omite el
manifest (`runner:1491-1495,1513-1517,1584-1588`), por lo que la suite verde no
lo ejerce.

**Reproducción ejecutada.** Se creó un árbol temporal con journal FIT
`NOT_EVALUABLE`, `fit/fit_not_evaluable.json`, `runtime.json` terminal y los
metadatos base; `write_artifact_manifest(..., run_role="primary")` devolvió:

```json
{"accepted": false, "error": "RuntimeError: Wave 59 closed artifact inventory mismatch: missing=[], extra=['runtime.json']"}
```

Además, las clases terminales se filtran por lo que ya existe
(`runner:1054-1081`), de modo que no son un manifest esperado verdaderamente
phase-specific: un artefacto de éxito adicional de la misma fase puede pasar a
ser esperado por mera presencia.

**Corrección mínima.** Definir conjuntos terminales explícitos por último
journal/status, incluir siempre `runtime.json` como output terminal permitido y
derivar la clase esperada sin filtrar por existencia. Rechazar tanto un path
posterior como un output de éxito no declarado de la misma fase. Ejecutar tests
canónicos para cada salida `NOT_EVALUABLE`, no sólo el helper sobre árboles
sintéticos.

## Finding importante no prospectivamente invalidante

### P2 — los límites y CUDA invisible no cubren al coordinador de preparación

El runner analítico comparte un deadline real de 1800 s, detiene el process
group por wall time/RSS, calcula el remanente combinado de 3600 s y fija CUDA y
cuatro hilos en cada worker (`runner:364-462,1426-1448,1650-1664`). El worker
ciego del preparador ahora recibe esas variables y tiene polling de 1800 s/8
GiB (`prepare_wave56_fresh.py:2296-2356`). Esa parte de R421 está resuelta.

Sin embargo, el límite del preparador empieza recién al lanzar inferencia. No
cubre `preparation_preflight`, generación, validación, materialización root de
los cinco bundles ni el proceso coordinador completo
(`prepare_wave56_fresh.py:733-844,2505-2781`). Tampoco se fija
`CUDA_VISIBLE_DEVICES=''` en el entorno del coordinador que importa y usa Torch;
la única asignación está dentro del `env` del hijo (`prepare_wave56_fresh.py:2302`).
Esto no cumple todavía la regla explícita de CUDA invisible en coordinador y
workers ni el presupuesto por corrida (`plan:596-608`).

**Corrección mínima.** Fijar/verificar CUDA invisible al comienzo del entrypoint
Wave 59, establecer un deadline y medición RSS para la transacción completa de
preparación y registrar ese receipt. Añadir tests del coordinador, además de los
dos tests de budget del worker analítico (`tests/test_wave59_prospective.py:345-371`).

## Reparaciones verificadas sin finding material

- Los hashes de `validation_policy_arrays.npz` y
  `monitor_policy_arrays.npz` ahora se verifican contra sus freezes antes de
  abrir truth. Las dos mutaciones negativas de R421 fallan como corresponde.
- `monitor_action_freeze.json` vincula config, source bindings, preparation,
  FIT, calibración, validación, states, scores, bundle inference-safe y acciones
  (`_wave59_phase_worker.py:644-671`); el coordinador vuelve a verificar la
  cadena antes de MONITOR-EVALUATE (`runner:1589-1608`).
- El amendment se deriva de metadata de preparación y su hash se exige antes de
  comparar (`runner:784-794`). Primary y replay deben compartir contexto y el
  archivo entra al conjunto byte-exacto (`runner:872-889`).
- `_finalize_replay_condition` actualiza el hash de `analysis.json` en el journal
  después de reemplazar `PENDING` (`runner:961-983`). En el output temporal
  finalizado por el test de replay, el hash real y el journal coincidieron en
  `4763b6294e4cbe069eed301206c7551bd90ddb3b22d030483c21020f7c73999e`;
  el resume ordinario posterior también pasó.
- Los dos contrastes principales contra proposer-only están presentes y la
  suite conserva los 36 contrastes factoriales.
- El preparador Wave 59 usa el mismo writer redacted del runner; el mensaje crudo
  ya no aparece y se crea `failure_inventory.json`
  (`prepare_wave56_fresh.py:2819-2844`).

## Comandos y resultados

```text
git rev-parse 3755bd4^ && git rev-parse 3755bd4
# c7442ee822e57c925bdc8de9e4fc9061cb2dac42
# 3755bd4cad0c0edf56f4d05cc7a1f196a894dfba

git diff --check 3755bd4^ 3755bd4
# exit 0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/python -m pytest -q \
  tests/test_wave59_hgb_guard_bracket.py tests/test_wave59_prospective.py \
  tests/test_wave56_preoracle_recovery.py tests/test_wave56_prospective.py \
  tests/test_wave57_prospective.py
# 196 passed in 254.98s (0:04:14)
```

Los tres probes adversariales se ejecutaron con `venv/bin/python` y las mismas
variables CPU sobre repos/árboles copiados bajo `/tmp`: self-binding post-commit;
archive → mutación de analysis+journal+inventory → restore → execute; y manifest
FIT terminal. Todos sus directorios útiles se eliminaron después de capturar el
resultado. No escribieron en el repositorio.

## Hashes del corpus auditado

| Archivo | SHA-256 |
|---|---|
| `experiments/geometria_proporcional/_wave59_phase_worker.py` | `271c3fb3d886035778a59b43de99fa7d42778cd0f93f8c83d8571f0fa85b1c57` |
| `experiments/geometria_proporcional/prepare_wave56_fresh.py` | `04ed370fb256c24b5272d65fbecefc1fa6f8067df9cbd94a52c8b639dfde0f2f` |
| `experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py` | `1636e3533c1debc6dcf4c1ba3387b28d553e5785c234b3e041d5fef1f48e49f0` |
| `src/geometria_proporcional/wave59_hgb_guard_bracket.py` | `33ff257b9c54f69ebf26d1884416f8bf77f26dbf425a96653d784ab08feff40d` |
| `tests/test_wave59_prospective.py` | `4d34bbd9f70af7e9da6bb9b6a8123b000500708396a15a211dedd739ffa88cd7` |
| Config no modificada | `dc6d10141307b5d4dd1456b9082a558e0aa7f133438dd5eb70995b37b2113e18` |

## Condición para PASS

Cerrar los tres P1 con tests adversariales que no confíen en la misma autoridad
que atacan, completar el P2 del coordinador de preparación y reauditar otro
commit estable. La config debe permanecer sin freeze y no debe crearse el
triplete mientras estos defectos sigan reproducibles.
