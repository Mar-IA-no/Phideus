# Ola 59 — tercera reauditoría independiente de implementación

## Dictamen: REVISE

El commit estable `e6bdd0491d1cbc1771e226e68338d01fa58c6abf`
resuelve materialmente los dos P1 y el P2 informados por R423: cierra las keys y
varios valores de los objetos autenticados, excluye toda la metadata raíz de
fallo al restaurar, incorpora una prueba canónica autosuficiente y mantiene el
watchdog wall/RSS activo durante reporte, comparación, replay, manifests y
`fsync`. La regresión completa pasa: `211 passed`.

El paquete todavía no alcanza `PASS`. Un probe físico encontró que el lenguaje
autenticado continúa abierto en **cobertura y semántica**, aunque las keys sean
exactas: una firma válida acepta receipts vaciados, acepta eliminar un output
obligatorio de FIT si se lo retira también de journal e inventory, y ni siquiera
cubre un archivo anidado cuyo basename sea `failure_attestation.json`. Esto es
un P1 porque la restauración idéntica deja de probar que el estado durable es el
estado exacto producido por la fase. Además, el coordinador conserva un P2
CPU-only: usa `setdefault` para los límites de hilos, por lo que variables
heredadas mayores que cuatro prevalecen; el probe observó pools de 10, 16 y 17
hilos.

Este dictamen evalúa sólo validez prospectiva, trazabilidad y capacidad de
avanzar. No interpreta evidencia científica ni toma una decisión `GO/NO-GO`.

## Identidad, alcance y método

- Commit auditado: `e6bdd0491d1cbc1771e226e68338d01fa58c6abf`.
- Padre directo: `e0a26422616041ded1473f8d1f7cb86d494e2e5a`.
- Subject: `Close Wave 59 recovery schemas and runtime envelope`.
- Timestamp: `2026-09-05T02:02:17-03:00`.
- Diff estable: exactamente runner y test prospectivo, `1092` inserciones y
  `25` eliminaciones. `git diff --check e6bdd04^ e6bdd04` terminó con exit 0.
- Se releyeron completos el plan final de 626 líneas, R421, R422, R423, los dos
  archivos modificados y los contratos relevantes del preparador, worker,
  core, config y tests HGB.
- Todos los probes y tests fueron CPU-only. No se usaron GPU, web ni Mendieta y
  no se modificaron config, implementación ni outputs canónicos.
- Los artefactos efímeros de esta auditoría se limitaron a `/tmp`. La primera
  regresión agotó ese filesystem después de `206 passed` y produjo cinco
  `OSError: [Errno 28]` durante escrituras NPZ de Wave 57; se eliminaron sólo
  los directorios temporales propios de esta auditoría y la repetición completa
  pasó. No se atribuyen esos cinco errores al commit.

## Revalidación de R421, R422 y R423

| Condición | Resultado R424 | Evidencia |
|---|---|---|
| Firma externa y tamper sin clave | Resuelta | Attestation y payload exigen keys/valores exactos, verifican Ed25519 y anclan hashes reales de inventory y `FAILURE.json` antes del copy (`run_wave59_hgb_guard_bracket.py:738-782`). El test físico de tamper sigue rechazando (`tests/test_wave59_prospective.py:255-307`). |
| Schema exacto de attestation, payload, inventory, records y `FAILURE.json` | Resuelto para keys y valores declarados | `_require_exact_keys` cubre los cinco objetos, `failure_records` exige la lista exacta y cada record valida path, clase, bytes y hash (`runner:714-818`). Las cinco variantes re-firmadas de extensión son negativas (`tests/test_wave59_prospective.py:334-376`). Persiste el P1 de cobertura/semántica descrito debajo. |
| Schema exacto de journals | Resuelto para keys, status, truth e inputs | Prepare exige bundles exactos; las otras cinco variantes exigen keys, estados, truth, inputs, duración/RSS y receipt (`runner:952-1098`). La extensión re-firmada se rechaza (`tests/test_wave59_prospective.py:379-408`). Persiste ausencia de outputs y contenido de receipt exactos. |
| Binding entre `FAILURE.json` y último journal | Resuelto | Estado y máximo truth se derivan del último journal durable y se comparan con igualdad (`runner:1152-1174`); la mutación re-firmada de `last_state` se rechaza (`tests/test_wave59_prospective.py:411-440`). |
| Restore sin metadata raíz de fallo | Resuelto | El copy excluye `FAILURE.json`, inventory, attestation y manifest; luego verifica residuo antes de retornar (`runner:677-711`). |
| Restore canónico y manifest reconstruible | Resuelto para el árbol canónico válido ejercitado | La prueba crea el universo COMPLETE, archiva, restaura sin metadata y reconstruye el manifest con `missing=[]`, `extra=[]` (`tests/test_wave59_prospective.py:744-924`). |
| Wall/RSS por corrida y combinado hasta cierre | Resuelto | Un solo envelope externo engloba `_execute_once` y contabilidad final (`runner:2505-2544`); reporte, `compare_runs`, condiciones replay y manifests ocurren dentro de `_execute_once` (`runner:2410-2422`); `_finalize_accounted_runtime` actualiza duración, RSS, suma combinada y manifest bajo checkpoints activos (`runner:2435-2502`). |
| Inyecciones durante manifest/`fsync` | Resueltas | Los tests de demora y RSS rechazan bajo el watchdog (`tests/test_wave59_prospective.py:577-627`). Probes adicionales rechazaron demora en `_write_report` y en `compare_runs`. |
| Terminales `NOT_EVALUABLE`, matriz canónica y comparación replay | Sin regresión observada | Pasan las pruebas de los tres terminales, cuatro contextos, extras, familias HGB y replay exacto. La regresión completa relevante terminó `211 passed`. |
| CPU-only/CUDA | Parcial | CUDA se fuerza a vacío antes de imports (`runner:28-40`) y workers reciben vacío + cuatro hilos (`runner:470-568`). El máximo efectivo de cuatro hilos falla en el coordinador; P2 debajo. |

## Finding prospectivamente invalidante

### P1 — la firma autentica mapas autoconsistentes, no la cobertura exacta de fase ni todo el árbol

La reparación cierra la forma superficial de los objetos, pero no su universo
obligatorio. `_validate_access_receipt` sólo comprueba que `stage_hashes` sea un
mapa de hashes, que `forbidden_probes` sea una lista de probes bien formados y
que `threadpools` sea una lista de pools válidos. No exige cobertura igual a
`PHASE_FILES[phase]`, número e identidad de probes, ni presencia de los pools
realmente cargados (`run_wave59_hgb_guard_bracket.py:882-949`). El productor,
en cambio, registra todos los miembros de `PHASE_FILES`, cada probe recibido y
`threadpool_info()` (`_wave59_phase_worker.py:950-997`). El propio test canónico
construye y acepta `threadpools=[]`, `stage_hashes={}` y
`forbidden_probes=[]` (`tests/test_wave59_prospective.py:890-907`), de modo que
normaliza el hueco en vez de detectarlo.

Los outputs sufren la misma apertura. El validador define sets exactos para
**inputs**, pero no para outputs por `(phase, status)` (`runner:1029-1083`). Al
reanudar, deriva `observed_outputs` de los archivos que todavía existen y sólo
exige igualdad con el mapa del journal (`runner:1101-1143`). A su vez, el
inventario fallido deriva `required` de las keys de ese mismo mapa
(`runner:1898-1912`). Por tanto, quitar simultáneamente un output físico y sus
dos declaraciones conserva una historia autoconsistente pero incompleta.

Finalmente, tanto archivado como validación excluyen metadata por
`path.name`, no por path relativo raíz. Cualquier archivo anidado llamado
`failure_inventory.json` o `failure_attestation.json` queda fuera de los
records y del set físico comparado (`runner:829-836,1771-1776`). El anchor no
lo autentica. `copytree(ignore_patterns(...))` también lo omite al restaurar,
por lo que no ensucia el output canónico, pero el supuesto inventory
closed-world del archive es falso.

**Reproducción física.** Sobre tres archives válidos derivados del pipeline
Wave 59 físico, cada variante se volvió a firmar con un anchor Ed25519 válido
cuando correspondía:

```json
{
  "resigned_empty_stage_hashes_and_forbidden_probes": {
    "accepted": true,
    "restored_exists": true
  },
  "resigned_missing_fit_feature_schema": {
    "accepted": true,
    "restored_exists": true
  },
  "unsigned_nested_failure_attestation": {
    "accepted": true,
    "restored_exists": true
  }
}
```

La primera variante reemplazó por `{}` y `[]` los campos de receipt, actualizó
journal e inventory y re-firmó. La segunda eliminó
`fit/feature_schema.json`, lo retiró de `output_sha256`,
`output_inventory_before_receipt` e inventory y re-firmó. La tercera añadió
`nested/failure_attestation.json` **después** de firmar, sin modificar firma ni
inventory. Las tres restauraron. Esto separa claramente el finding de la
criptografía: la firma es válida, pero el schema firmado permite omitir
evidencia y el filtro por basename deja bytes fuera de la firma.

**Riesgo.** Un archive re-firmado por una ruta de recuperación defectuosa, o
por cualquier actor con autoridad de firma, puede declarar una fase durable
sin todos sus outputs ni receipts probatorios. Además, un archivo anidado queda
fuera de toda autenticación incluso sin re-firmar. La reanudación ya no prueba
la identidad exacta del checkpoint parcial que pretende reutilizar.

**Corrección mínima.** Definir sets exactos de output por `(phase, status)` y
usarlos tanto al archivar como al restaurar; exigir `stage_hashes` con cobertura
exacta de `PHASE_FILES[phase]` y hashes concordantes, y el conjunto exacto de
probes requerido por cada fase. Exigir evidencia no vacía y coherente de
threadpools cuando el runtime correspondiente está cargado. Excluir únicamente
los dos paths raíz de metadata al construir/verificar el inventory; todo path
anidado debe clasificarse y autenticarse o rechazarse. Añadir las tres
mutaciones anteriores como tests negativos re-firmados.

## Finding importante no prospectivamente invalidante

### P2 — el coordinador no fuerza el máximo efectivo de cuatro hilos

El runner fija correctamente `CUDA_VISIBLE_DEVICES=''` antes de importar
NumPy/joblib, pero usa `os.environ.setdefault(..., "4")` para OMP, OpenBLAS,
MKL y NumExpr (`runner:28-40`). Una variable heredada con valor mayor no se
reemplaza. Esto contradice el contrato explícito de cuatro hilos y verificación
de pools del plan (`plan:596-604`). Tampoco hay una verificación de
`threadpool_info()` para el proceso coordinador; sólo se verifican receipts de
workers.

**Reproducción.** Importando el runner con CUDA inicialmente `0` y las cuatro
variables de hilos en `17`, después del import se observó:

```text
CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS="17"
openblas num_threads: 16, 16
openmp num_threads: 17, 10
```

El proceso sigue CPU-only —CUDA sí quedó vacío—, pero no respeta el máximo de
cuatro hilos. El riesgo es sobreconsumo y pérdida de comparabilidad operativa,
no contaminación científica ni uso de GPU.

**Corrección mínima.** Asignar las cuatro variables con `os.environ[name] =
"4"` antes de imports, como hace el preparador, y validar los pools efectivos
del coordinador tras cargar sus runtimes. Añadir una prueba de import en
subproceso con variables heredadas mayores que cuatro.

## Presupuesto, probes y regresiones

Los probes adicionales de postprocesado redujeron el envelope a `0.30 s` e
inyectaron `0.40 s` en dos puntos distintos:

```text
_write_report: RuntimeError: Wave 59 analytical coordinator exceeded wall-time budget
compare_runs replay: RuntimeError: Wave 59 analytical coordinator exceeded wall-time budget
```

Comandos principales:

```text
git diff --check e6bdd04^ e6bdd04
# exit 0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/pytest -q \
  tests/test_wave59_prospective.py::test_identical_hash_resume_reuses_completed_journals \
  tests/test_wave59_prospective.py::test_identical_hash_resume_rejects_post_failure_tamper \
  tests/test_wave59_prospective.py::test_resume_rejects_resigned_schema_extensions \
  tests/test_wave59_prospective.py::test_resume_rejects_resigned_journal_extension \
  tests/test_wave59_prospective.py::test_resume_rejects_resigned_semantic_value_drift \
  tests/test_wave59_prospective.py::test_canonical_complete_restore_rebuilds_closed_manifest \
  tests/test_wave59_prospective.py::test_analytical_coordinator_enforces_budget_during_manifest_fsync \
  tests/test_wave59_prospective.py::test_analytical_coordinator_enforces_rss_during_manifest_fsync
# 12 passed in 16.16s

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/python -m pytest -q \
  tests/test_wave59_hgb_guard_bracket.py tests/test_wave59_prospective.py \
  tests/test_wave56_preoracle_recovery.py tests/test_wave56_prospective.py \
  tests/test_wave57_prospective.py
# 211 passed in 257.22s (0:04:17)

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
| R421 | `2b6807e6d724ebc3033751e06395920a57a9a8b528deb9e974e625577b777104` |
| R422 | `df053fb3fdcc2b9dfe3dfc06a2dae43e2b653943bc847457b4129be051a61977` |
| R423 | `0ebf6e320af223f8abaf341a73304cb65a84f94d485080b58cab9ff993a55aed` |
| `prepare_wave56_fresh.py` | `fb5345dd978a3bd2d658a8709f874e6ff3944f6f88e62f0330c9b7d9946c9778` |
| `run_wave59_hgb_guard_bracket.py` | `8d328a41da5749daf2154e1f12020dd4e583cd6a746a4e15a1de3244e5b614b5` |
| `_wave59_phase_worker.py` | `271c3fb3d886035778a59b43de99fa7d42778cd0f93f8c83d8571f0fa85b1c57` |
| `wave59_hgb_guard_bracket.py` | `33ff257b9c54f69ebf26d1884416f8bf77f26dbf425a96653d784ab08feff40d` |
| `test_wave59_hgb_guard_bracket.py` | `d837a8ae4948eec6a59c3a62c751935d2d90521a7d7178f7411006d002898f51` |
| `test_wave59_prospective.py` | `06a719fbbe6485bdbf04c2ccb08381d2f5897d8682cf9143ae3839606fa4437e` |
| Config prospectiva | `dc6d10141307b5d4dd1456b9082a558e0aa7f133438dd5eb70995b37b2113e18` |

## Estado prospectivo y condición para PASS

Antes de escribir este informe, el worktree estaba limpio. La config permanece
`IMPLEMENTATION_PRE_DRAW` y su binding sigue
`PENDING_IMPLEMENTATION_AUDIT` (`wave59_fresh_hgb_guard_bracket.json:3,24-25`).
Los outputs canónicos
`data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1/` y
`data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1_replay/` están
ausentes. No se creó escrow ni se realizó draw.

Para `PASS`: cerrar el P1 con contratos de cobertura independientes de los
mapas que se autentican, rechazar metadata anidada no inventariada, forzar y
verificar el máximo de cuatro hilos del coordinador, incorporar los probes
negativos y repetir la regresión completa. Hasta entonces no corresponde
congelar la config ni iniciar el fresh draw.
