# R379 — Quinta auditoría independiente de implementación del recovery pre-oráculo de Wave 56 Stage 1

**Implementation commit:** `6e4b7925797fe7046bb7c6894919020672a15ec5`
**Preparer SHA-256:** `aa68ca636b1536df27de739927fd4afdea18e6417838b1e7e6be0dee8a8769b8`
**Test SHA-256:** `4fdbe9d4043254f799beb884e77f6c05c5353c6ba645830ee3f7094c8f9c5b7d`
**Result:** `PASS`

## Dictamen ejecutivo

El commit auditado cierra el diferencial de lexer hallado por R378. La
atestación se decodifica como UTF-8 estricto, admite únicamente `LF` como fin
de línea, exige `LF` terminal y rechaza `CR`, `CRLF`, controles C0 distintos de
tab y `LF`, `DEL`, `NEL`, `U+2028` y `U+2029`. Sobre esos bytes impone la
secuencia posicional H1, blanco, campos exactos y contiguos, blanco y H2, y
mantiene la comprobación global de unicidad de cada prefijo autoritativo.

No encontré un bypass material nuevo. El control positivo renderizó H1, un
párrafo top-level con las cuatro afirmaciones visibles y H2 separados. Las
variantes con finales no canónicos, UTF-8 inválido, BOM inicial, ausencia de
newline terminal y controles embebidos fueron rechazadas. Los negativos
permanentes para fence, comentario HTML, contradicción, duplicación, orden,
separación y ubicación posterior a una sección también permanecen verdes.

El resto del recovery no cambia en este commit. Se conservan los paths
documentales de las dos auditorías, su separación, los commits directos de la
cadena, los blobs ligados, la exigencia futura de `HEAD == F`, el único delta
ejecutable del preparador, la validación física doble del origen, no-redraw, el
orden fail-closed previo a inferencia y el replay exacto. En esta etapa
pre-`A`, `HEAD` es correctamente `I`; la cadena real posterior todavía no
existe y no fue simulada como si ya autorizara una ejecución oficial.

## Alcance y estado Git

Audité el objeto exacto indicado, con parent único
`4667dd31f83ff9604da318911de66434b21b739f`. El diff `parent..I` contiene sólo:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`: 17 inserciones
  y 3 eliminaciones;
- `tests/test_wave56_preoracle_recovery.py`: 47 inserciones.

Los archivos del worktree son byte-idénticos a sus blobs en `I` y producen los
dos hashes de la cabecera. `git diff --check` no reportó errores. El plan
corregido se congeló en `P =
8062c83ccea1fe6ce8c087f6e1c9bb1ff4ea30c2`; su SHA-256
`d4f306ffb8d34a9d3499b99e18434d57085438fd4cd0202f4f8658d3bdbe923d`
permanece idéntico en el parent de `I`, en `I` y en el worktree.

La historia mínima confirma la sucesión de correcciones R375–R378 y que el
parent inmediato de este `I` es el commit documental que incorporó R378. No se
hicieron commits durante esta auditoría.

## Archivos leídos completos

- `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/330_wave56_stage1_preoracle_recovery_plan_audit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/331_wave56_stage1_preoracle_recovery_plan_reaudit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/375_wave56_stage1_preoracle_recovery_implementation_independent_audit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/376_wave56_stage1_preoracle_recovery_implementation_reaudit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/377_wave56_stage1_preoracle_recovery_implementation_final_reaudit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/378_wave56_stage1_preoracle_recovery_implementation_visible_header_reaudit.md`
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`
- `tests/test_wave56_preoracle_recovery.py`

También inspeccioné el diff exacto, los blobs, el plan congelado y la historia
Git mínima necesaria para reconstruir `P→I→A→J→F`.

## Verificación de texto canónico y visibilidad Markdown

La implementación lee bytes y usa `decode("utf-8")`; ya no delega el concepto
de línea a `splitlines()`. El barrido de caracteres invalida todos los C0 salvo
tab y `LF`, y agrega explícitamente `DEL`, `NEL`, los separadores Unicode de
línea y párrafo. Después separa sólo por `"\n"` y exige newline terminal
(`prepare_wave56_fresh.py:748-788`).

La matriz directa independiente obtuvo:

- aceptación exclusiva del documento unido con `LF` y terminado en `LF`;
- rechazo de `CR`, `CRLF`, `VT`, `FF`, `FS`, `GS`, `RS`, `DEL`, `NEL`,
  `U+2028` y `U+2029`;
- rechazo de UTF-8 inválido, BOM al inicio, NUL embebido y ausencia de newline
  terminal.

El control `LF` se renderizó con Python-Markdown 3.10.1 como tres bloques
separados: H1, párrafo con las afirmaciones y H2. Por construcción, las líneas
de campo son ASCII exacto, no tienen indentación, quote, list marker, fence,
HTML ni trailing spaces. El parser exige además que cada prefijo aparezca una
sola vez como comienzo de línea; por eso una segunda afirmación top-level, aun
con valor distinto, aborta.

Los adversariales end-to-end permanentes construyen cadenas completas para
ambos roles documentales. La variante `U+2028` es rechazada por separado en
`A` y en `F`; los casos anteriores de reporte ejecutable, contradicción,
fence y comentario oculto siguen cubiertos en ambos roles
(`tests/test_wave56_preoracle_recovery.py:204-512`).

## Cadena de procedencia y transacción

La revisión estática y los repos sintéticos confirman estas propiedades:

1. `I` tiene exactamente un parent, cambia sólo preparador y test, y los dos
   blobs se resuelven desde el objeto Git declarado.
2. `A` y `F` deben ser `.md` distintos, directamente bajo el directorio
   canónico de informes; no pueden ser tests, config ni código ejecutable.
3. Los bordes `A←I`, `J←A` y `F←J` exigen parent directo. Cada commit posterior
   sólo puede introducir su artefacto previsto, y el validador exige que `F`
   sea el `HEAD` exacto (`prepare_wave56_fresh.py:991-1054`).
4. El contrato de ejecución sólo admite el cambio de hash del preparador
   respecto del contrato físico de origen. El plan, los hashes y el informe
   ligado no pueden autodescribirse desde archivos sucios o distintos de
   `HEAD`.
5. Amendment, DAG, reportes, limpieza global, origen y escrow reutilizado se
   validan antes de crear o archivar output. Antes de inferencia se repiten el
   inventario físico, los hashes, la identidad del manifest y los conteos.
6. Recovery y replay no admiten `keys_override`; sólo el primary fresco alcanza
   `secrets.token_bytes`. Escrow y freeze republicados deben conservar sus
   bytes, y replay compara además amendment, manifest, visibles, logits,
   freeze de preparación y procedencia.

La prueba física sintética incluida en la focal recorrió el fallo inicial, el
rechazo de una mutación tardía del origen, recovery hasta `PREPARED` y replay
exacto. También sustituyó `secrets.token_bytes` por una excepción, de modo que
una tentativa de redraw habría hecho fallar el test
(`tests/test_wave56_preoracle_recovery.py:546-684`).

## Origen preservado

La inspección se limitó a inventario, tipos, modos, ownership, tamaños y
SHA-256. No se abrieron valores del escrow ni contenidos de
`benchmark/sealed/**`, truth, labels u oracle.

El árbol preservado contiene 6 directorios y 18 archivos regulares, todos
`root:root`, sin symlinks ni tipos especiales. Los directorios son `0700`; los
archivos son `0600`, salvo el freeze público `0644`. Los hashes públicos
congelados permanecen:

- escrow: `f86fb936651552a757b46acd56e1c17674635eb37cfc6d0cd2a8a02e2f06e978`;
- freeze pre-generación: `c65d581a755d611f9f86264402bfea89503a8599d7967d94882d2db91f5083e8`;
- fallo: `710b7d29de8c0436304ffb7abdfb2adcd958ed443ab72a9190ce74495e8602af`;
- manifest: `7582efe3fdcd40125929cbe2c6783a37b1ba3f8ffb2fb6cce6b5578979d29ef8`.

No existen `inference/`, `authorized_labels/`, `bundles/`, `phases/`,
`benchmark/sealed/oracle`, `generation_receipt.json`,
`preparation_freeze.json` ni `preparation_receipt.json`.

## Pruebas y comandos

Todo se ejecutó con `CUDA_VISIBLE_DEVICES=''` y los límites
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` y
`NUMEXPR_NUM_THREADS=1`.

- Suite focal completa: `venv/bin/python -m pytest -q
  tests/test_wave56_preoracle_recovery.py` — **33 passed in 15.25s**.
- Matriz directa de bytes y render Markdown: un control aceptado, quince
  adversariales rechazados; estructura renderizada H1/párrafo/H2 confirmada.
- Git: `rev-parse`, `show`, `diff-tree`, `diff`, `diff --check`, historia del
  plan y SHA-256 de blobs/worktree.
- Origen: `find -xdev` con metadata y SHA-256, sin parsear material sellado.

No corrí una suite amplia: la focal incluye el camino físico sintético y todos
los contratos modificados; no apareció un finding que justificara ampliar el
alcance.

## Findings y límites

No hay findings altos, medios o bajos que requieran revisar este `I`.

La auditoría no ejecutó recovery ni replay oficiales, no creó `A`, `J` o `F`
reales y no abrió inferencia, oracle ni labels. La aprobación acredita la
implementación pre-oráculo y permite continuar la DAG; no reemplaza las dos
auditorías futuras ni anticipa su contenido. Los recovery/replay recorridos por
pytest usan fixtures sintéticos. Tampoco se recomputaron desde truth sellada
los conteos oficiales: se verificó su contrato, su flujo y los hashes físicos,
no sus valores secretos.

## Decisión

La implementación queda técnicamente aprobada para avanzar a `A` y continuar,
sin saltos, la secuencia `P→I→A→J→F`. Esto no autoriza por sí solo la corrida
oficial, no abre material posterior y no constituye un `GO/NO-GO` científico.
