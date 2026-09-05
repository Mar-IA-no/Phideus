# Ola 59 — segunda reauditoría independiente de implementación

## Dictamen: REVISE

El commit estable `bdeb305c9664f73863dab7dbe67682520b1ae84b`
resuelve los tres P1 principales descritos por R422 en cuanto a autoridad
criptográfica, binding Git/preparación y terminales `NOT_EVALUABLE`. La firma
Ed25519 externa rechaza alteraciones autoconsistentes del árbol fallido antes de
copiar bytes; el runner exige sources físicos iguales a blobs limpios de HEAD,
ancestry del commit implementado, auditoría `PASS` y `preparation_freeze`
exactamente concordante; y los tres terminales tempranos producen inventarios
explícitos que rechazan outputs de éxito o fases futuras.

El paquete todavía no alcanza `PASS`. Hay dos defectos P1 en el mismo camino de
reanudación: el schema autenticado no es cerrado y una restauración válida copia
el propio `failure_attestation.json` al output operativo, donde después viola el
manifest closed-world canónico. También persiste un P2 de recursos: los límites
de 1800/3600 segundos y RSS terminan antes del postprocesado del coordinador
analítico. La suite agregada pasa, pero su test de tamper depende del orden de
ejecución y no detecta la imposibilidad del cierre canónico restaurado.

Este dictamen sólo evalúa validez prospectiva, trazabilidad y capacidad de
avanzar. No interpreta resultados científicos ni toma una decisión `GO/NO-GO`.

## Identidad y alcance

- Commit auditado: `bdeb305c9664f73863dab7dbe67682520b1ae84b`.
- Padre directo: `3be4417a756b6f960a81bd99662a9b2995073ee2`.
- Subject: `Authenticate Wave 59 recovery and terminal states`.
- Timestamp: `2026-09-05T01:21:27-03:00`.
- Plan completo: 626 líneas, SHA-256
  `7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb`.
- R421 completo: 366 líneas, SHA-256
  `2b6807e6d724ebc3033751e06395920a57a9a8b528deb9e974e625577b777104`.
- R422 completo: 246 líneas, SHA-256
  `df053fb3fdcc2b9dfe3dfc06a2dae43e2b653943bc847457b4129be051a61977`.
- El diff estable modifica exactamente preparador, runner y test prospectivo:
  468 inserciones y 61 eliminaciones. `git diff --check bdeb305^ bdeb305`
  terminó con exit 0.
- Se releyeron completos el plan, R421, R422 y los tres archivos modificados.
  Los probes usaron copias efímeras bajo `/tmp`, CPU,
  `CUDA_VISIBLE_DEVICES=''` y como máximo cuatro hilos. No se usaron GPU, web ni
  Mendieta; no se modificaron config ni outputs canónicos.

## Revalidación de R422 y R421

| Condición | Resultado R423 | Evidencia |
|---|---|---|
| Tamper de `analysis` + journal + inventory | Resuelta frente a un atacante sin clave | La prueba física emparejada rechaza con `failure attestation payload drifted`; la firma se verifica antes del copy (`runner:616-621,635-652`). |
| Tamper de `source_bindings` + record | Resuelta frente a un atacante sin clave | Probe efímero: rechazo por `failure attestation payload drifted`, con destino todavía ausente. |
| Firma ausente o alterada | Resuelta | Ausencia: `lacks its external-trust attestation`; firma Ed25519 alterada: `AttestationError`; en ambos casos `target_exists=false`. |
| Config/source dirty con hashes internos recalculados | Resuelta | Ambos probes rechazados como `execution source is dirty`; cada source se contrasta además contra `HEAD:<path>` (`runner:245-299`). |
| Ancestry y auditoría `PASS` | Resuelta | El validador exige commit existente, `merge-base --is-ancestor`, hash del audit, commit citado y heading `## Dictamen: PASS` (`runner:266-284`). |
| Autoridad exacta de PREPARE | Resuelta | Freeze exacto aceptado; `prospective_config` alterada rechazada. Se comparan config raw, objeto completo, sources raw y source bindings (`runner:1508-1551`). |
| Terminales FIT/CALIBRATE-SCORES/MONITOR-APPLY | Resuelta | Manifests válidos de 42/67/76 archivos; cada uno rechazó un output de éxito de su propia fase y uno de fase futura (`runner:1093-1229`). |
| Hashes de arrays antes de abrir truth | Resuelta | Regresiones negativas de VALIDATE y MONITOR-EVALUATE pasan; freezes exigen cobertura exacta (`runner:302-324,1689-1701,1726-1778`). |
| Amendment recovery y comparación replay | Resuelta | Continúan la autenticación de contexto y la inclusión del amendment en igualdad científica verificadas por R422; las regresiones correspondientes pasan. |
| Contrastes contra `HGB-PROPOSER-ONLY` | Resuelta | La suite física vuelve a exigir familias `mean_vs_hgb_proposer_only` y `tail_vs_hgb_proposer_only`, además de 36 contrastes factoriales (`tests/test_wave59_prospective.py:138-176`). |
| CUDA y watchdog de preparación | Resuelta en preparación | CUDA y cuatro variables de hilos se fijan antes de importar NumPy/Torch (`prepare_wave56_fresh.py:27-38`); el contexto cubre preflight y toda la transacción (`prepare_wave56_fresh.py:2867-2980`) y los probes de wall/RSS rechazan. |
| Deducción de tiempo de preparación | Resuelta parcialmente | Se resta de 1800 y de 3600 antes del análisis (`runner:1581-1618`); probes con 1801 y 3601 segundos acumulados rechazaron antes de abrir fases. Persiste el P2 de postprocesado descrito debajo. |

## Findings prospectivamente invalidantes

### P1 — el schema autenticado del intento fallido permanece abierto

La firma externa corrige la autoridad circular de R422: primero se verifica la
firma con la clave pública confiable y luego se exige que su payload sea el
anchor exacto de inventory, `FAILURE.json` y path archivado
(`run_wave59_hgb_guard_bracket.py:635-652`). Esa raíz es suficiente para impedir
el tamper postarchivo ordinario sin acceso a la clave privada.

Pero el contenido autenticado se valida mediante `.get` y subconjuntos, no por
sets exactos. El inventory sólo exige versión, `records`, unicidad/cobertura,
bytes/hash y cuatro listas vacías (`runner:653-676`). No valida las keys exactas
del top-level, de cada record ni el valor de `failure_records`. El restore sólo
consulta dos campos de `FAILURE.json` antes de continuar (`runner:609-615`) y
los journals reanudados tampoco tienen schema cerrado (`runner:680-698`). Esto
no implementa la corrección explícita de R422 (`R422:80-87`) ni el contrato de
schema/inventario cerrado del plan (`plan:501-515,532-545`).

**Reproducción.** En un archive mínimo válido se añadieron simultáneamente una
key top-level no declarada y una key no declarada al primer record. Luego se
recalculó el hash del inventory y se produjo una firma Ed25519 válida con la
clave externa. `restore_identical_hash_attempt` devolvió:

```json
{"accepted":true,"copied":["config.snapshot.json","failure_attestation.json","source_bindings.json"],"target_exists":true}
```

Una segunda variante agregó una key no declarada a `FAILURE.json`, actualizó su
record, recalculó el inventory y firmó el anchor exacto; también fue aceptada.
El control con firma realmente alterada fue rechazado. Por tanto, el finding no
cuestiona la criptografía: demuestra que la autoridad externa autentica un
lenguaje más amplio que el schema cerrado aprobado.

**Corrección mínima.** Exigir igualdad exacta de keys para attestation/payload,
inventory, cada record, `FAILURE.json` y cada variante de journal; exigir el
valor exacto de `failure_records`; rechazar tipos/campos inesperados antes de
usar cualquier contenido. Añadir variantes re-firmadas a los tests para separar
autenticidad criptográfica de clausura del schema.

### P1 — una reanudación válida restaura el anchor de fallo como path extra

Después de autenticar el archive, `restore_identical_hash_attempt` copia el
árbol con `shutil.copytree`. La allowlist de exclusión contiene
`FAILURE.json`, `failure_inventory.json` y `artifact_manifest.json`, pero omite
`failure_attestation.json` (`runner:621-629`). El anchor archivístico reaparece
así dentro del output operativo. No pertenece a ninguna de las seis clases de
artefactos (`runner:1034-1090`), y el manifest canónico exige igualdad exacta
entre árbol real y matriz (`runner:1182-1196`).

**Reproducción.** Un restore válido copió exactamente:

```json
{"copied":["config.snapshot.json","failure_attestation.json","source_bindings.json"]}
```

En un árbol COMPLETE sintético con los 81 miembros canónicos, el manifest base
cerró con cero faltantes/extras. Al representar el resultado del restore
añadiendo sólo ese anchor, falló:

```text
RuntimeError: Wave 59 closed artifact inventory mismatch: missing=[], extra=['failure_attestation.json']
```

Esto vuelve imposible completar o cerrar como `NOT_EVALUABLE` una reanudación
canónica auténtica, aunque todos sus bytes científicos sean exactos. El test
nuevo no lo detecta: usa fixture module-scoped y depende de que el test anterior
haya creado `output/prepared` (`tests/test_wave59_prospective.py:225-293`). Al
ejecutarlo aisladamente, el rechazo de tamper ocurre, pero la restauración de
control termina luego en `FileExistsError`; ejecutado detrás del test anterior,
ambos pasan y el output no contiene `benchmark`, por lo que no se escribe
manifest canónico.

**Corrección mínima.** Excluir también `failure_attestation.json` de la copia de
restauración y verificar explícitamente que ninguno de los tres metadatos de
fallo quede en el output reanudado. Crear una prueba canónica autosuficiente con
`benchmark/` y `prepared/` que archive, restaure, complete y vuelva a construir
el manifest closed-world sin depender del orden de tests.

## Finding importante no prospectivamente invalidante

### P2 — wall time, RSS y límite combinado no incluyen el postprocesado final

La reparación del preparador sí cubre preflight, generación, inferencia y
materialización con un único `SIGALRM` y watcher RSS; el worker de inferencia
mantiene además su control de proceso (`prepare_wave56_fresh.py:2282-2368,
2867-2980`). El runner resta correctamente la preparación de los presupuestos
de 1800/3600 segundos (`run_wave59_hgb_guard_bracket.py:1581-1618`) y entrega el
mismo deadline a las cinco fases.

Sin embargo, el último check temporal ocurre inmediatamente después de las
fases (`runner:1820-1821`). A continuación, ya sin deadline ni watcher del
coordinador, se escriben runtime y reporte, se carga/compara el replay, se
actualizan ambos terminales y se construyen los manifests (`runner:1822-1859`).
En cierres tempranos, `_finalize_terminal` tampoco recibe deadline
(`runner:1565-1571`). `primary_plus_replay_seconds` se calcula antes de
`compare_runs` y de los manifests (`runner:1822-1846`), de modo que esos costos
no cuentan contra 3600 segundos. El RSS publicado toma sólo máximos de journals
de workers (`runner:1480-1505`), no el coordinador durante comparación y
materialización final. Esto deja incompleto el máximo por corrida explícito del
plan (`plan:596-608`).

**Corrección mínima.** Mantener wall/RSS hasta que todo output terminal esté
fsyncado y validado; comprobar el deadline después de reporte, comparación y
manifests; contabilizar ese tramo en `duration_seconds` y
`primary_plus_replay_seconds`. El test debe inyectar demora y exceso RSS en
`compare_runs`/manifest, no sólo dentro de un worker.

## Probes y regresiones

Resultados adversariales capturados:

```json
{
  "source_binding_plus_record": {"accepted": false, "target_exists": false, "error": "failure attestation payload drifted"},
  "missing_signature": {"accepted": false, "target_exists": false, "error": "lacks its external-trust attestation"},
  "altered_signature": {"accepted": false, "target_exists": false, "error": "AttestationError"},
  "resigned_extra_inventory_record_schema": {"accepted": true},
  "resigned_extra_failure_schema": {"accepted": true},
  "dirty_config_recomputed_internal_hash": {"accepted": false, "error": "execution source is dirty"},
  "dirty_source_recomputed_internal_hash": {"accepted": false, "error": "execution source is dirty"},
  "exact_preparation_freeze": {"accepted": true},
  "drifted_preparation_freeze": {"accepted": false},
  "primary_1801_preparation": {"accepted": false},
  "combined_3601_before_analysis": {"accepted": false},
  "preparer_rss_limit_one_byte": {"accepted": false, "error": "exceeded RSS budget"}
}
```

Los tres terminales `NOT_EVALUABLE` aceptaron su conjunto explícito y rechazaron
dos clases de extra:

| Terminal | Archivos canónicos | Extra de éxito misma fase | Extra futuro |
|---|---:|---|---|
| FIT | 42 | rechazado: `fit/feature_schema.json` | rechazado: `calibration/calibration_not_evaluable.json` |
| CALIBRATE-SCORES | 67 | rechazado: `calibration/validation_scores.npz` | rechazado: `validation/validation_summary.json` |
| MONITOR-APPLY | 76 | rechazado: `adjudication/monitor_scores.npz` | rechazado: `analysis.json` |

Comandos principales:

```text
git diff --check bdeb305^ bdeb305
# exit 0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/pytest -q \
  tests/test_wave59_prospective.py::test_identical_hash_resume_reuses_completed_journals \
  tests/test_wave59_prospective.py::test_identical_hash_resume_rejects_post_failure_tamper
# 2 passed in 14.86s

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/python -m pytest -q \
  tests/test_wave59_hgb_guard_bracket.py tests/test_wave59_prospective.py \
  tests/test_wave56_preoracle_recovery.py tests/test_wave56_prospective.py \
  tests/test_wave57_prospective.py
# 201 passed in 255.63s (0:04:15)
```

La selección focal sin el test precedente produjo `1 failed, 6 passed`: el
único fallo fue la dependencia de orden descrita arriba, posterior al rechazo
correcto del tamper.

## Hashes del corpus auditado

| Archivo | SHA-256 |
|---|---|
| `experiments/geometria_proporcional/prepare_wave56_fresh.py` | `fb5345dd978a3bd2d658a8709f874e6ff3944f6f88e62f0330c9b7d9946c9778` |
| `experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py` | `b58ebceada45e8f1ab3619e91520738ce358e97d5d2478dda288afae3069b297` |
| `tests/test_wave59_prospective.py` | `2e56091a12fb57d170e9ef98c4a0da886ea9a67c9cc1e51a319a2c30fdc46ed3` |
| `experiments/geometria_proporcional/_wave59_phase_worker.py` | `271c3fb3d886035778a59b43de99fa7d42778cd0f93f8c83d8571f0fa85b1c57` |
| `src/geometria_proporcional/wave59_hgb_guard_bracket.py` | `33ff257b9c54f69ebf26d1884416f8bf77f26dbf425a96653d784ab08feff40d` |
| Config no modificada | `dc6d10141307b5d4dd1456b9082a558e0aa7f133438dd5eb70995b37b2113e18` |

## Estado prospectivo y condición para PASS

Al cierre, config sigue `IMPLEMENTATION_PRE_DRAW`, su
`implementation_binding.status` sigue `PENDING_IMPLEMENTATION_AUDIT`, el
worktree previo al informe estaba limpio y ambos outputs canónicos Wave 59
estaban ausentes. No se creó escrow ni draw.

Para `PASS`: cerrar ambos P1 de recovery con una prueba canónica independiente,
cerrar schema en todos los objetos autenticados, extender wall/RSS al
postprocesado terminal y repetir la regresión completa. Hasta entonces no debe
congelarse la config ni iniciarse el fresh draw.
