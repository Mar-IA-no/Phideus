# Ola 59 — cuarta reauditoría independiente de implementación

## Dictamen: REVISE

El commit estable `5c83aefdb1f3a2c865173edc8948747d06061f43`
resuelve los tres casos concretos de R424 y el límite CPU del coordinador. Los
campos probatorios vacíos, la ausencia coordinada de
`fit/feature_schema.json` y la metadata reservada anidada son rechazados antes
de copiar. El request queda ligado a su contenido, los probes a cantidad,
identidad y orden, y los receipts de threadpools ya no pueden estar vacíos. El
runner normaliza variables heredadas de 17 a 4, mantiene cuatro hilos efectivos
dentro de `execute` y publica esa evidencia en `runtime.json`. La regresión
relevante completa pasa: `215 passed`.

El paquete aún no alcanza `PASS`. La cobertura de outputs fija ahora todos los
miembros **obligatorios** de cada `(phase,status)`, pero no impone exclusión
mutua frente a otros outputs clasificados de la misma fase. Un archive firmado
con historia COMPLETE y los tres sentinels `NOT_EVALUABLE` adicionales fue
restaurado; también se restauraron los tres terminales `NOT_EVALUABLE` cuando
cada uno contenía un output de éxito de su propia fase. Es un P1: el failure
inventory declara `extra=[]` y el restore acepta estados que la matriz canónica
normal rechaza.

Además queda un P2 acotado: si se alteran de forma autoconsistente un hash de
input, su `stage_hash` y el request, y luego se vuelve a firmar el archive, el
restore copia el árbol. `execute` lo rechaza inmediatamente antes de reutilizar
la fase, por lo que no se observó aceptación científica incorrecta, pero la
promesa de validación previa al copy sigue incompleta para valores de inputs.

Este dictamen evalúa conformidad prospectiva, trazabilidad y capacidad de
avanzar. No interpreta resultados científicos ni toma una decisión `GO/NO-GO`.

## Identidad, alcance y método

- Commit auditado: `5c83aefdb1f3a2c865173edc8948747d06061f43`.
- Padre directo: `b4e5600822f869a7512392156387a5495b2a7331`.
- Subject: `Close Wave 59 phase coverage and CPU bounds`.
- Timestamp: `2026-09-05T02:37:38-03:00`.
- Diff estable: cuatro archivos, `460` inserciones y `205` eliminaciones.
  `git diff --check 5c83aef^ 5c83aef` terminó con exit 0.
- Se leyeron completos R424, el diff y los cuatro archivos actuales implicados;
  también se revalidaron los contratos relacionados del plan, config y
  preparador.
- Todos los casos negativos y tests usaron CPU,
  `CUDA_VISIBLE_DEVICES=''` y como máximo cuatro hilos. No se usaron GPU, web ni
  Mendieta; no se editaron fuentes, config ni outputs canónicos.
- `/tmp` estaba al 100% por artefactos preexistentes. La primera selección
  focal produjo ocho `ENOSPC` y dos passes; se descartó como medición de código.
  La repetición usó un `--basetemp` propio en `/mnt/m2-1TB`, pasó completa y el
  árbol temporal se retiró al terminar. No se eliminaron artefactos ajenos.

## Revalidación directa de R424

| Caso | Resultado R425 | Evidencia |
|---|---|---|
| `stage_hashes={}` y `forbidden_probes=[]`, journal/inventory actualizados y nueva firma | Resuelto | Rechazo: `Wave 59 fit stage coverage drifted`; destino ausente. El validador exige igualdad con `PHASE_FILES[phase]` (`run_wave59_hgb_guard_bracket.py:971-981`). |
| Eliminar `fit/feature_schema.json` y retirarlo de ambos output maps e inventory antes de firmar | Resuelto | Rechazo: `derived coverage drifted: missing=['fit/feature_schema.json']`; destino ausente. `_failure_coverage` deriva outputs del contrato, no del journal (`runner:1991-2007`). |
| Añadir `nested/failure_attestation.json` sin firmarlo | Resuelto | Rechazo: `failed-attempt inventory coverage drifted`; destino ausente. Sólo se excluyen los dos paths raíz del set físico (`runner:858-865`) y el copy ignora metadata únicamente en raíz (`runner:725-735`). |
| Hash de `phase_request.json` distinto | Resuelto | Rechazo: `Wave 59 fit request hash drifted`; el hash se reconstruye desde phase, allowlist e inputs (`runner:982-988`). |
| Cantidad, identidad u orden de probes distintos | Resuelto | Las tres variantes rechazaron `forbidden-probe coverage drifted`; el esperado se deriva de fase y path original (`runner:992-1006`). |
| `threadpools=[]` | Resuelto | Rechazo: `Wave 59 fit threadpool receipt drifted`; los pools presentes también deben informar `1..4` hilos (`runner:1017-1031`). |
| Output obligatorio ausente en los cinco estados de éxito | Resuelto | FIT, CALIBRATE-SCORES, VALIDATE, MONITOR-APPLY y MONITOR-EVALUATE rechazaron respectivamente la ausencia de `feature_schema`, `calibration_freeze`, `validation_freeze`, `monitor_action_freeze` y `analysis`; destino siempre ausente. |
| Output obligatorio ausente en los tres estados `NOT_EVALUABLE` | Resuelto | Los baselines FIT, CALIBRATE-SCORES y MONITOR-APPLY restauraron; al retirar su sentinel exacto, los tres rechazaron con `derived coverage drifted`. |

## Contratos congelados y flujo de cobertura

`PHASE_FILES` vive ahora en el módulo core como cinco `frozenset` cerrados
(`src/geometria_proporcional/wave59_hgb_guard_bracket.py:36-102`). Worker y
runner importan el mismo objeto (`_wave59_phase_worker.py:23-48`;
`run_wave59_hgb_guard_bracket.py:53-65`). El worker valida y hashea exactamente
ese stage (`_wave59_phase_worker.py:934-938`), mientras recovery exige las
mismas keys, igualdad stage/input y hash canónico del request
(`runner:971-988,1113-1134`). Esto elimina la duplicación detectada por R424.

Los outputs ya no se derivan del mapa presentado por el journal.
`_expected_phase_output_names` fija los tres singletons `NOT_EVALUABLE`, los
tres outputs de MONITOR-EVALUATE y obtiene los outputs exitosos restantes desde
la matriz canónica de clases (`runner:1653-1689`). `_failure_coverage` usa ese
resultado durante el archivado (`runner:1888-1904,1991-2007`) y el restore lo
recalcula de nuevo desde el árbol autenticado (`runner:898-915`).
`_validate_resumed_journals` exige además igualdad entre el set declarado y el
set esperado antes de validar hashes físicos (`runner:1145-1203`).

Por tanto, las coberturas positivas solicitadas están conectadas a fuentes
independientes del journal. El finding siguiente no revierte ese avance: señala
que todavía falta aplicar esos mismos sets como límite superior sobre todo el
árbol real de cada estado.

## Finding prospectivamente invalidante

### P1 — estados mutuamente excluyentes de una misma fase conviven y restauran

El inventario añade incondicionalmente los tres sentinels terminales a las
clases conocidas, cualquiera sea el status de los journals
(`run_wave59_hgb_guard_bracket.py:866-884`). Luego
`_validate_resumed_journals` construye `observed_outputs` filtrando sólo los
nombres esperados; un archivo real de la misma carpeta que no pertenezca a ese
set simplemente queda fuera de la comparación (`runner:1175-1191`).

`_failure_coverage` calcula un conjunto `allowed`, pero nunca compara el árbol
real contra él. Su única detección de extras es `future`, y sólo marca paths
cuya fase sea **posterior** al último journal (`runner:1941-1954,1991-2039`). Un
output incompatible de la misma fase o de una fase anterior no es `future`,
tiene clase válida y pasa tanto archive como restore.

**Casos negativos reproducidos.** No requirieron manipulación posterior de
firma: los archivos contradictorios ya estaban presentes cuando
`archive_failed_attempt` construyó inventory y attestation.

```json
{
  "success_with_terminal_extras": "accepted",
  "fit_not_evaluable_with_success_extra": "accepted",
  "calibrate_scores_not_evaluable_with_success_extra": "accepted",
  "monitor_apply_not_evaluable_with_success_extra": "accepted"
}
```

La primera variante conservó la historia COMPLETE y añadió simultáneamente
`fit/fit_not_evaluable.json`,
`calibration/calibration_not_evaluable.json` y
`adjudication/monitor_not_evaluable.json`. Las otras tres construyeron un
terminal `NOT_EVALUABLE` válido y añadieron respectivamente
`feature_schema.json`, `calibration_freeze.json` o
`monitor_action_freeze.json`. En los cuatro casos el inventory firmado declaró
listas de cobertura vacías y la restauración aceptó.

**Riesgo.** Un fallo puede preservar y reanudar dos resultados incompatibles
para la misma transición. Aunque el manifest canónico terminal rechazaría ese
árbol más adelante, recovery lo trata como hash-idéntico y lo copia al path
operativo. Esto contradice la matriz closed-world por `(phase,status)` y reabre
el defecto de terminales incompatibles en una forma distinta de la ya resuelta
por R423.

**Corrección mínima.** Para cada journal, comparar el conjunto **completo** de
archivos físicos propiedad de su fase con
`_expected_phase_output_names(run_dir, phase, status)`. En
`_failure_coverage`, hacer efectiva una allowlist status-aware y devolver como
extra todo `actual - allowed`, no sólo fases posteriores. Los sentinels
`NOT_EVALUABLE` deben clasificarse condicionalmente por status, no agregarse a
toda historia fallida. Añadir los cuatro casos anteriores como regresiones
negativas.

## Finding importante no prospectivamente invalidante

### P2 — valores de input autoconsistentes se validan recién después del copy

Recovery exige las keys exactas de inputs y hace coincidir los hashes del stage
con el propio `journal_inputs`; después reconstruye el request a partir de esos
mismos valores (`runner:971-988,1113-1134`). No contrasta en esa etapa cada hash
contra su fuente física canónica.

En una copia del pipeline se reemplazó `input_sha256['bundle.npz']` por un hash
sintácticamente válido, se aplicó el mismo valor a `stage_hashes`, se recalculó
correctamente el hash del request, se actualizaron journal e inventory y se
firmó el anchor. Resultado:

```json
{
  "restore": "accepted",
  "execute": "RuntimeError: Wave 59 fit resume inputs differ"
}
```

El rechazo posterior es correcto: `_reuse_or_run_phase` calcula los hashes de
las fuentes actuales y falla antes de reutilizar outputs (`runner:669-693`). No
hubo continuación ni aceptación científica incorrecta. El riesgo se limita a
copiar un archive inválido al path canónico y abrir otro ciclo de fallo, en vez
de rechazarlo antes del copy como promete la validación de recovery.

**Corrección mínima.** Donde exista una fuente durable en el archive, ligar el
hash de input del journal a ese path antes de restaurar. Para inputs efímeros
derivados —como `utilities.npy`—, preservar un binding verificable o pasar la
autoridad externa necesaria al validador. Mantener la comprobación de
`_reuse_or_run_phase` como segunda barrera.

## CPU, presupuesto y evidencia runtime

El runner ahora asigna, no hereda, CUDA vacío y las cuatro variables de hilos
antes de importar runtimes numéricos (`runner:28-41`). Un subproceso iniciado
con CUDA `0` y los cuatro caps en `17` observó tras el import:

```json
{
  "CUDA_VISIBLE_DEVICES": "",
  "OMP_NUM_THREADS": "4",
  "OPENBLAS_NUM_THREADS": "4",
  "MKL_NUM_THREADS": "4",
  "NUMEXPR_NUM_THREADS": "4",
  "effective_pool_threads": [4, 4, 4, 4]
}
```

El envelope crea además `threadpool_limits(4)`, verifica pools no vacíos y
efectivos y restaura sus límites sólo al salir (`runner:2054-2132`). Un
`_execute_once` instrumentado observó `[4,4,4,4]` dentro de `execute`. Su
`runtime.json` publicó `budget_enforced=true`, CUDA vacío, `max_seconds=1800`,
RSS permitido `8589934592` y los cuatro pools a cuatro hilos
(`runner:2588-2617,2621-2660`). La corrida física completa mostró la misma
evidencia y `total_run_seconds=13.851296681910753`.

Los tests de demora y exceso RSS en manifest/`fsync` siguen rechazando bajo el
watchdog (`tests/test_wave59_prospective.py:676-727`). Reporte, comparación,
actualización replay, manifests y contabilidad final permanecen dentro del
envelope; no hay regresión del cierre de R424.

## Restauración canónica, regresiones y comandos

La restauración canónica autosuficiente volvió a pasar: no retuvo metadata de
fallo y reconstruyó el manifest con `missing=[]` y `extra=[]`
(`tests/test_wave59_prospective.py:885-1049`). También pasaron terminales
`NOT_EVALUABLE`, cuatro contextos primary/replay × normal/recovery, comparación
replay, source/preparation bindings, HGB families y checks de truth boundary.

Comandos principales:

```text
git diff --check 5c83aef^ 5c83aef
# exit 0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/pytest -q \
  --basetemp=/mnt/m2-1TB/r425-pytest \
  [10 tests focales de recovery, recursos y restore canónico]
# 10 passed in 17.87s

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/python -m pytest -q \
  --basetemp=/mnt/m2-1TB/r425-pytest \
  tests/test_wave59_hgb_guard_bracket.py tests/test_wave59_prospective.py \
  tests/test_wave56_preoracle_recovery.py tests/test_wave56_prospective.py \
  tests/test_wave57_prospective.py
# 215 passed in 266.66s (0:04:26)

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/python -m py_compile \
  experiments/geometria_proporcional/prepare_wave56_fresh.py \
  experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py \
  experiments/geometria_proporcional/_wave59_phase_worker.py \
  src/geometria_proporcional/wave59_hgb_guard_bracket.py \
  tests/test_wave59_prospective.py
# exit 0
```

## Hashes del corpus auditado

| Archivo | SHA-256 |
|---|---|
| Plan final Wave 59 | `7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb` |
| R424 | `297c61ea8943243e449451938a3f2f43703ba2b2908e0755c1ac09942ecc2892` |
| `prepare_wave56_fresh.py` | `fb5345dd978a3bd2d658a8709f874e6ff3944f6f88e62f0330c9b7d9946c9778` |
| `run_wave59_hgb_guard_bracket.py` | `2266f558927c5fbc5c477eba674d3078c82faad8bb1cf59e1c854e46e983edfd` |
| `_wave59_phase_worker.py` | `4552dbf1ef80ee3f3b07361a8c0c3d442ce28a52ccf4ee11ceee69b69dfa5642` |
| `wave59_hgb_guard_bracket.py` | `8f4daf8c545407b5b0e1c7a50af7350e00c16cf0817f7f81b66f9a6a046cd8a8` |
| `test_wave59_hgb_guard_bracket.py` | `d837a8ae4948eec6a59c3a62c751935d2d90521a7d7178f7411006d002898f51` |
| `test_wave59_prospective.py` | `0cfbc79e759547a222dc5ea87903ac576d74b70c963ea488057c7383b87444d3` |
| Config prospectiva | `dc6d10141307b5d4dd1456b9082a558e0aa7f133438dd5eb70995b37b2113e18` |

## Estado prospectivo y condición para PASS

Antes de escribir este informe, el worktree estaba limpio. La config permanece
`IMPLEMENTATION_PRE_DRAW` y su binding sigue
`PENDING_IMPLEMENTATION_AUDIT` (`wave59_fresh_hgb_guard_bracket.json:3,24-25`).
Los outputs canónicos
`data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1/` y
`data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1_replay/` están
ausentes. No se creó escrow ni se realizó draw.

Para `PASS`: imponer igualdad exacta entre archivos físicos de cada fase y el
set status-aware, rechazar los outputs mutuamente excluyentes en recovery,
añadir esas variantes a la suite y repetir la regresión. Conviene cerrar también
el P2 de inputs antes de promover el audit binding. Hasta entonces no
corresponde congelar la config ni iniciar el fresh draw.
