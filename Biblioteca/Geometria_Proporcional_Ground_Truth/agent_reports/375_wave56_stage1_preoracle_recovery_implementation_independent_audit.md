# R375 — Auditoría independiente de implementación del recovery pre-oráculo de Wave 56 Stage 1

**Implementation commit:** `f62f0d057fae608a3dfa7bc844f2a53b7532898f`
**Preparer SHA-256:** `47aabf2eabad1203b52696242a23a5cb1a95dcec44466b98478b702e3a4f35e5`
**Test SHA-256:** `7c24a7503a6e1162ba4236544d4ebc9b40de24d8e4303a6f7bc0a05ea9f5e35f`
**Result:** `REVISE`

## Dictamen ejecutivo

El commit auditado corrige los dos defectos concretos que declara su diff: fija
el path del plan canónico y reemplaza las relaciones de mera ancestralidad entre
`I`, `A`, `J` y `F` por parentesco directo. La suite focal pasa `13/13`, incluido
el negativo que inserta dos commits cuyo diff acumulado vuelve a cero. También
se mantienen el binding commit→blob, el `HEAD == F`, el contrato de único delta
del preparador, la reutilización exclusiva del escrow y el orden fail-closed que
impide llegar a inferencia fresca antes de validar `F`, el origen y el benchmark.

La implementación no está lista para producir `J`. Encontré un bypass alto y
reproducible en la propia frontera de procedencia: los paths de `A` y `F` son
libres dentro del repositorio. El validador acepta como supuesto informe de
auditoría un archivo Python ejecutable —por ejemplo `tests/conftest.py`— siempre
que sea el único path agregado por el commit correspondiente y contenga las
cuatro o tres cadenas esperadas, incluso sólo como comentarios. Esto contradice
la obligación de que `A` y `F` sean informes bajo `Biblioteca/` y de que `F` no
modifique código, config ni amendment. Además permite que el mismo artefacto
presentado como evidencia independiente altere la suite que debe sostener esa
evidencia.

## Alcance y frontera de inspección

Audité el árbol exacto de
`f62f0d057fae608a3dfa7bc844f2a53b7532898f`, cuyo único parent es
`1b72823063da71fe0985a7bff00dcc000033e40c`, y el diff exacto `parent..I`.
Todas las pruebas se ejecutaron con `CUDA_VISIBLE_DEVICES=''` y
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`. No usé GPU ni web, no hice commits y no modifiqué código,
tests, configs, planes, datos ni artefactos existentes.

Del intento fallido
`data/geometria_proporcional/wave56_contextual_gate_fresh_v1.failed_20260903T171827485015Z`
inspeccioné únicamente inventario físico, tipos, permisos, ownership, tamaños y
hashes. No abrí ni mostré valores de `generation_escrow.json`, ningún miembro de
`benchmark/sealed/**`, truth, labels u oracle. El inventario observado tiene 6
directorios y 18 archivos regulares, todos `root:root`, sin symlinks ni tipos
especiales; no aparecen `inference/`, `authorized_labels/`, `bundles/`,
`phases/`, `benchmark/sealed/oracle`, `generation_receipt.json`,
`preparation_freeze.json` ni `preparation_receipt.json`. Los hashes del escrow,
freeze público, `FAILURE.json` y manifest coinciden con los congelados en el
plan.

## Archivos leídos completos

- `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/330_wave56_stage1_preoracle_recovery_plan_audit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/331_wave56_stage1_preoracle_recovery_plan_reaudit.md`
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`
- `tests/test_wave56_preoracle_recovery.py`

También inspeccioné la historia Git estrictamente necesaria para identificar
`P`, el parent de `I`, los blobs auditados y los cambios previos del recovery.

## Finding por severidad

### ALTA — `A` y `F` pueden ser código ejecutable presentado como informe

**Observación.** El plan define `A` y `F` como informes y exige que el commit de
`F` no modifique código, config ni amendment
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:115-128`); la lista de
archivos permitidos ubica informes de auditoría y cierre en `Biblioteca/`
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:219-230`). El código sí fija
el plan mediante `RECOVERY_PLAN_RELATIVE`
(`prepare_wave56_fresh.py:69-74`, `:902-906`), pero no define un path o una raíz
canónica equivalente para ninguna auditoría.

`require_repo_artifact` sólo exige path relativo canónico, archivo regular,
tracking, limpieza e identidad con `HEAD` (`prepare_wave56_fresh.py:225-251`).
Después, `implementation_audit.path` y `final_audit_path` entran directamente
desde el JSON a ese validador genérico (`:939-958`). Los commits quedan
restringidos a cambiar exactamente esos paths, pero los paths mismos siguen
siendo libres (`:959-969`, `:983-996`). Finalmente, `_require_report_fields`
busca subcadenas sin validar que el artefacto sea un informe Markdown ni que las
líneas funcionen como afirmaciones del informe (`:732-736`, `:944-953`,
`:974-982`).

**Reproducción.** En repos Git efímeros bajo `/tmp` construí dos cadenas limpias
y de parentesco directo `P→I→A→J→F`, con blobs, hashes, contratos y campos de
attestation consistentes. `validate_recovery_amendment` aceptó:

1. `implementation_audit.path = tests/conftest.py`;
2. `final_audit_path = tests/conftest.py`.

En ambos casos las líneas exigidas estaban en comentarios Python y el mismo
archivo contenía un hook `pytest_collection_modifyitems`. Una prueba aislada
adicional demostró el impacto ejecutable: un `conftest.py` con hooks de colección
y cierre convirtió una suite que contenía un test incondicionalmente fallido en
`PYTEST_EXIT=0` con `no tests ran`. Los repos temporales fueron destruidos al
terminar y no tocaron el worktree.

La suite permanente no contiene negativos para el rol de los paths de `A` o
`F`; sus casos de procedencia cubren hash del preparador, path extra en `I`,
campos del informe, plan alternativo y commits intermedios
(`tests/test_wave56_preoracle_recovery.py:352-380`).

**Impacto.** El contrato permite que un cambio de código posterior a `I` sea
clasificado como `A` o `F`, justo donde el plan prohíbe cambios ejecutables. En
el caso de `conftest.py`, el artefacto puede intervenir las pruebas requeridas y
a la vez satisfacer textualmente el marcador `PASS`. Esto invalida la garantía
estructural de que la cadena posterior a `I` contiene sólo evidencia documental
y amendment, aunque el parentesco Git sea correcto.

**Corrección necesaria.** Fijar paths canónicos exactos para `A` y `F`, o como
mínimo exigir que ambos sean archivos `.md` ubicados directamente bajo
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/`, con rechazo
explícito de cualquier path de código, tests o config. Agregar negativos
separados para `implementation_audit.path` y `final_audit_path` fuera de esa
frontera. Como la corrección toca preparador y tests, requiere un nuevo commit
`I` y reiniciar desde allí la cadena independiente; no corresponde construir
`J` sobre el commit auditado aquí.

### MEDIA

Ninguno adicional.

### BAJA

Ninguno que justifique ampliar el cambio.

## Controles que sí quedaron confirmados

1. **Plan canónico fijado.** El path constante apunta al amendment plan exacto y
   se compara antes de aceptar su hash (`prepare_wave56_fresh.py:69-74`,
   `:902-906`). El negativo de plan alternativo pasa
   (`tests/test_wave56_preoracle_recovery.py:352-380`). El blob del plan tiene
   SHA-256 `d4f306ffb8d34a9d3499b99e18434d57085438fd4cd0202f4f8658d3bdbe923d`
   tanto en el parent de `I` como en `I` y en el worktree.
2. **Parentesco directo y rechazo de commits intermedios.**
   `require_direct_parent` exige exactamente un parent y su identidad exacta
   (`prepare_wave56_fresh.py:219-223`); se aplica a `A←I`, `J←A` y `F←J`
   (`:965-969`). El negativo agrega un commit y su revert antes de `A`, dejando
   net diff cero, y es rechazado (`tests/test_wave56_preoracle_recovery.py:206-210`,
   `:359-362`). La misma primitiva protege los otros dos bordes.
3. **Binding de implementación.** `I` debe cambiar exclusivamente preparador y
   test, y ambos blobs Git deben coincidir con los hashes del amendment
   (`prepare_wave56_fresh.py:908-937`). El contrato de ejecución admite como
   único delta respecto del escrow el preparador y exige `git_commit == HEAD`
   (`:739-773`). Los hashes de los dos blobs en `I` son exactamente los
   declarados al inicio de este informe.
4. **`F` como `HEAD` exacto.** El commit de introducción de `F` debe ser el
   `HEAD` actual (`prepare_wave56_fresh.py:956-971`); por lo tanto un commit
   posterior, incluso vacío o con net diff cero, no puede ejecutar el recovery.
5. **Fail-closed pre-oráculo.** La CLI valida amendment y reused escrow antes de
   `run_preparation_transaction` y, por tanto, antes de crear o archivar output
   (`prepare_wave56_fresh.py:1711-1748`). Dentro del amendment, la existencia,
   cadena y marcadores de `F` se comprueban antes de abrir el origen preservado
   (`:939-1025`). La inferencia fresca ocurre sólo después de la segunda
   validación física del origen, identidad del manifest y conteos
   (`:1486-1577`); el script no materializa oracle ni labels y vuelve a comprobar
   la frontera antes del freeze (`:1301-1311`, `:1583-1616`).
6. **No-redraw y no sustitución exitosa de escrow.** Una recovery/replay no
   acepta `keys_override`; extrae exclusivamente las claves del escrow durable y
   sólo el modo primary llama a `secrets.token_bytes`
   (`prepare_wave56_fresh.py:1427-1450`). El origen se ata mediante whitelist y
   hashes (`:775-822`), se revalida antes de extraer claves y después de generar
   (`:1431-1438`, `:1504-1513`), y escrow/freeze republicados deben conservar los
   hashes originales (`:1453-1468`). El replay compara además escrow, freeze y
   copia del amendment (`:1348-1408`).

Estos controles son reales, pero no compensan que `A` y `F` carezcan de una
frontera de tipo/path y puedan introducir código después de `I`.

## Comandos y pruebas

- `git show -s --format=... I` y `git show -s --format=... parent`: identidad,
  parent único y metadatos del commit.
- `git diff --full-index parent I`, `git diff-tree -r I`, `git diff --check
  parent I`: diff exacto de 2 archivos, 38 inserciones, 5 borrados, sin errores
  de whitespace.
- `sha256sum` del worktree y `git show I:path | sha256sum`: igualdad exacta de
  preparador y test con los hashes declarados.
- `venv/bin/python -m pytest -q tests/test_wave56_preoracle_recovery.py`:
  **13 passed in 12.91s**.
- `find -xdev` con metadata y `sha256sum` sobre el intento preservado: inventario
  físico seguro, sin parsear ni imprimir contenido sellado o valores secretos.
- Dos pruebas adversariales efímeras de `validate_recovery_amendment`: aceptación
  de `tests/conftest.py` como `A` y como `F`.
- Prueba efímera de impacto pytest: **exit 0 / no tests ran** pese a contener un
  test fallido, mediante hooks permitidos por el path aceptado.

No repetí la suite amplia de 165 tests: la focal reprodujo el contrato auditado
y el finding no depende de otra superficie; la suite amplia ya constaba como
verde CPU-only.

## Límites

- Por la frontera pre-oráculo no recomputé desde `benchmark/sealed/**` los
  conteos 1152/768/384/192 ni ejecuté validadores que abrieran truth sellada.
  Esta auditoría confirma el inventario y los hashes preservados, no vuelve a
  adjudicar el contenido secreto.
- No ejecuté recovery físico ni replay oficial, porque `A` es precisamente la
  etapa que debe preceder a `J` y `F`; hacerlo habría violado la DAG auditada.
- Las reproducciones adversariales aislaron la lógica Git/path con un origen
  físico stub en repos temporales. No sustituyen la suite física, pero sí son
  suficientes para demostrar que los paths ejecutables atraviesan el validador.

## Decisión

`REVISE`. No avanzar a `J`, no abrir inferencia, oracle ni labels y no ejecutar
el recovery oficial con `f62f0d057fae608a3dfa7bc844f2a53b7532898f` como `I`.
La cadena de parentesco, plan, blobs, no-redraw y exact-HEAD está correctamente
endurecida, pero la clasificación de `A` y `F` como paths libres deja abierto un
cambio ejecutable posterior a `I` y contradice una condición expresa del plan.
