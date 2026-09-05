# Ola 59 — quinta reauditoría independiente de implementación

## Dictamen: PASS

El commit inmutable `ff9c6ed4d42119520169ba3922b036093afe4428`
cierra los dos findings pendientes de R425 sin introducir una regresión
observada. La restauración exige ahora exclusividad exacta de outputs por
`(phase,status)` y reconstruye los hashes de inputs desde la config, los
artefactos preservados y el policy manifest ligado antes de copiar el archive.
Los casos COMPLETE con sentinels y los tres terminales NOT_EVALUABLE con un
output de éxito de su misma fase fueron rechazados con el destino todavía
ausente. También fueron rechazados antes del copy un hash de input falso pero
internamente coherente, `source_bindings.json` alterado con journals e inventory
recalculados y nueva firma, y un policy manifest no ligado.

Una restauración física válida y la reconstrucción canónica sintética completa
siguen aceptadas. La selección focal terminó `19 passed`; ocho probes
independientes produjeron siete rechazos esperados y una aceptación válida; la
regresión de cinco archivos terminó `222 passed in 267.48s`. El runtime físico
fue CPU-only, publicó CUDA vacío, cuatro hilos efectivos en cada fase y un pico
RSS de `721,920,000` bytes frente al límite de `8,589,934,592` bytes.

Este PASS establece conformidad de implementación con el plan aceptado y los
findings R421–R425. No congela la config, no autoriza ni realiza el fresh draw,
no interpreta evidencia científica y no toma una decisión GO/NO-GO.

## Identidad, alcance y método

- Commit auditado: `ff9c6ed4d42119520169ba3922b036093afe4428`.
- Padre directo: `0310e6fc70e82d3ea6e8ff1a0258e375cdece2f4`.
- Subject: `Close Wave 59 recovery authority gaps`.
- Timestamp: `2026-09-05T03:18:12-03:00`.
- Diff: dos archivos, `452` inserciones y `21` eliminaciones;
  `git diff --check HEAD^ HEAD` terminó con exit 0.
- Se leyeron el plan final vigente, R421–R425, el diff completo y los archivos
  actuales pertinentes del runner, core compartido, worker, preparador, config
  y pruebas antes de emitir el dictamen.
- Todas las ejecuciones usaron `CUDA_VISIBLE_DEVICES=''` y caps explícitos de
  cuatro hilos. No se usaron GPU, web ni Mendieta; no se editaron fuentes,
  config ni outputs canónicos.
- Los temporales propios vivieron bajo `/mnt/m2-1TB/r426-pytest` y fueron
  retirados al terminar. No se tocaron artefactos ajenos.

## Findings priorizados

### P0 / P1 / P2 — ninguno

No encontré un defecto de validez prospectiva, aislamiento, recovery,
replay, closed-world o límites CPU en el alcance solicitado.

### P3 — cobertura de prueba mejorable, no bloqueante

`test_resume_rejects_unbound_policy_manifest_before_copy` llama directamente a
`_expected_resumed_input_hashes` (`tests/test_wave59_prospective.py:597-608`),
pero no atraviesa la API pública de restore ni afirma allí que el destino quede
ausente. El probe independiente de esta auditoría sí ejecutó
`restore_identical_hash_attempt` y obtuvo el rechazo pre-copy esperado. Conviene
convertir ese recorrido en regresión permanente para proteger el cableado,
aunque la implementación actual está correctamente conectada tanto en restore
como en CLI (`run_wave59_hgb_guard_bracket.py:706-748,2843-2849`).

## Cierre de R425: exclusividad exacta de outputs

`_expected_phase_output_names` fija el singleton de cada estado
NOT_EVALUABLE, los tres outputs de MONITOR-EVALUATE y el conjunto completo de
cada estado exitoso (`runner:1789-1825`). `_failure_coverage` ya no usa esos
sets sólo como piso: construye un `allowed` por historia durable y devuelve
también `actual - allowed` (`runner:2066-2208`). Por eso un output clasificado
pero mutuamente excluyente queda en `failure_inventory.extra`; el restore exige
que `extra=[]` antes del copy (`runner:903-923`).

La comprobación independiente observó:

| Historia inválida | `failure_inventory.extra` | Resultado de restore | Destino tras rechazo |
|---|---|---|---|
| COMPLETE + tres sentinels | los tres sentinels | `extra is not empty` | ausente |
| FIT NOT_EVALUABLE + `fit/feature_schema.json` | ese output de éxito | `extra is not empty` | ausente |
| CALIBRATE-SCORES NOT_EVALUABLE + `calibration/calibration_freeze.json` | ese output de éxito | `extra is not empty` | ausente |
| MONITOR-APPLY NOT_EVALUABLE + `adjudication/monitor_action_freeze.json` | ese output de éxito | `extra is not empty` | ausente |

La validación posterior del journal añade otra frontera exacta: exige igualdad
entre el output map y el set esperado, y compara contra todos los archivos
físicos pertenecientes a la fase, no sólo contra nombres declarados
(`runner:1295-1317,1333-1397`). En una restauración válida los cinco pares
status/output coincidieron exactamente: FIT `23`, CALIBRATE-SCORES `3`,
VALIDATE `3`, MONITOR-APPLY `3` y MONITOR-EVALUATE `3` outputs.

## Cierre de R425: autoridad de inputs antes del copy

El orden de restore es adecuado: valida failure record, attestation e
inventory, compara el snapshot de config y llama a
`_validate_resumed_journals` antes de `shutil.copytree`
(`runner:706-743`). Para cualquier journal analítico, esa función exige que
`source_bindings.json` sea exactamente el binding de la config y revalida la
autoridad de preparación (`runner:1234-1260`). En protocolo congelado,
`_ensure_preparation_authority` liga config, fuentes de ejecución,
source-bindings y hashes físicos de los cinco bundles
(`runner:2355-2398`).

Después `_expected_resumed_input_hashes` reconstruye por fase los hashes de
config, binding, preparation freeze, bundles y outputs upstream; cuando la fase
usa `utilities.npy`, lo vuelve a materializar determinísticamente desde el
policy manifest. El hash de ese manifest debe coincidir con el upstream ligado
en config (`runner:1153-1231`). El journal sólo se acepta si su mapa completo
es idéntico a esta reconstrucción (`runner:1287-1294,1323-1332`).

Los probes independientes confirmaron:

| Caso | Resultado de restore | Destino tras rechazo |
|---|---|---|
| `bundle.npz` con hash falso, stage hash y request recalculados, inventory actualizado y archive re-firmado | `resumed fit input values drifted` | ausente |
| source-bindings alterado, cinco journals/requests e inventory recalculados y archive re-firmado | `recovery source bindings drifted` | ausente |
| policy manifest no ligado pasado a la API pública | `policy manifest differs from the bound upstream` | ausente |

## Restauraciones válidas y regresiones

Una copia física completa fue archivada y restaurada con config y policy
manifest válidos: restore devolvió `ACCEPTED`, recreó el path original y no
retuvo ninguno de los tres miembros de `FAILURE_METADATA`. La prueba canónica
sintética también pasó y reconstruyó un manifest cerrado con `missing=[]` y
`extra=[]` (`tests/test_wave59_prospective.py:1120-1310`).

La corrida física verificó cinco receipts con identidad sin privilegios,
probes de truth denegados, hashes de stage exactos y `num_threads=4` en todos
los pools observados. `runtime.json` informó:

```text
status=COMPLETE
device=cpu
cuda_visible_devices=""
budget_enforced=true
max_rss_bytes=721920000
budget.max_rss_bytes=8589934592
budget.max_seconds_per_run=1800
```

La regresión completa cubrió aislamiento por fase, reproducción histórica,
replay exacto y rechazo de amendments no ligados, recuperación durable,
matrices primary/replay × normal/recovery, manifests cerrados, terminales
NOT_EVALUABLE, wall/RSS y normalización de caps heredados.

Comandos principales:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python -m pytest -q \
  --basetemp=/mnt/m2-1TB/r426-pytest/focused \
  [selección focal de recovery, exclusividad, restore, closed-world y recursos]
# 19 passed in 17.91s

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python -m pytest -q \
  --basetemp=/mnt/m2-1TB/r426-pytest/regression \
  tests/test_wave59_hgb_guard_bracket.py tests/test_wave59_prospective.py \
  tests/test_wave56_preoracle_recovery.py tests/test_wave56_prospective.py \
  tests/test_wave57_prospective.py
# 222 passed in 267.48s (0:04:27)

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python -m py_compile \
  experiments/geometria_proporcional/prepare_wave56_fresh.py \
  experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py \
  experiments/geometria_proporcional/_wave59_phase_worker.py \
  src/geometria_proporcional/wave59_hgb_guard_bracket.py \
  tests/test_wave59_prospective.py
# exit 0
```

Una primera invocación focal, antes de crear el padre del `--basetemp`, abortó
18 setups con `FileNotFoundError` y ejecutó una prueba; fue un error del arnés y
se descartó. La repetición idéntica tras crear el contenedor produjo los 19
passes anteriores. Un chequeo auxiliar intentó reconstruir inputs desde el
output de fixture sin su directorio `prepared` externo y se descartó por no ser
un árbol de recovery; repetido sobre el árbol restaurado completo dio igualdad
exacta de inputs y outputs en las cinco fases.

## Hashes del corpus auditado

| Archivo | SHA-256 |
|---|---|
| Plan final Wave 59 | `7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb` |
| R420, auditoría de plan aceptada | `c7d1ed28554bb3b1174bfb787ae9469ee20392062f0c36deda7d0d10dcab0134` |
| R421 | `2b6807e6d724ebc3033751e06395920a57a9a8b528deb9e974e625577b777104` |
| R422 | `df053fb3fdcc2b9dfe3dfc06a2dae43e2b653943bc847457b4129be051a61977` |
| R423 | `0ebf6e320af223f8abaf341a73304cb65a84f94d485080b58cab9ff993a55aed` |
| R424 | `297c61ea8943243e449451938a3f2f43703ba2b2908e0755c1ac09942ecc2892` |
| R425 | `d48cf09e23118bc00389b1210bc6a80af08e6645691e37f2fd6c0930dbd3cd35` |
| `prepare_wave56_fresh.py` | `fb5345dd978a3bd2d658a8709f874e6ff3944f6f88e62f0330c9b7d9946c9778` |
| `run_wave59_hgb_guard_bracket.py` | `d78a414de79eea4ae51b98913de37b0c83a579b509574386f71d5060e5b9b816` |
| `_wave59_phase_worker.py` | `4552dbf1ef80ee3f3b07361a8c0c3d442ce28a52ccf4ee11ceee69b69dfa5642` |
| `wave59_hgb_guard_bracket.py` | `8f4daf8c545407b5b0e1c7a50af7350e00c16cf0817f7f81b66f9a6a046cd8a8` |
| `test_wave59_hgb_guard_bracket.py` | `d837a8ae4948eec6a59c3a62c751935d2d90521a7d7178f7411006d002898f51` |
| `test_wave59_prospective.py` | `b34ac5648f49e99b728825bb5deb471cf24c1e66813ad1561c16379fabf7dfd0` |
| Config prospectiva | `dc6d10141307b5d4dd1456b9082a558e0aa7f133438dd5eb70995b37b2113e18` |

## Estado prospectivo al cierre

Antes de escribir este informe el worktree estaba limpio. La config permanece
`IMPLEMENTATION_PRE_DRAW` y su binding sigue
`PENDING_IMPLEMENTATION_AUDIT` (`wave59_fresh_hgb_guard_bracket.json:3,24-28`).
Los outputs canónicos primary y replay declarados en config están ausentes. No
se creó escrow, no se realizó draw y no se modificó el protocolo.

El siguiente paso pertenece al coordinador: integrar este PASS y, si corresponde,
realizar el binding documental de implementación mediante un cambio posterior
trazable. Ese acto no forma parte de esta auditoría.
