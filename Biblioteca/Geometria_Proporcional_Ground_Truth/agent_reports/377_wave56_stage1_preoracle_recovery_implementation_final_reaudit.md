# R377 — Reauditoría final de implementación del recovery pre-oráculo de Wave 56 Stage 1

**Implementation commit:** `a08aeaed403b0f315643c42b4c3c91ec10933782`
**Preparer SHA-256:** `ca7f96fc45dc1c13be56f3f00c5ed102f9196c7b2a02702462307a30d35c9400`
**Test SHA-256:** `48b523674806ee91453abbf6b11b8d395e56487e3e69c5389a763c59a78a0510`
**Result:** `REVISE`

## Dictamen ejecutivo

El commit auditado cierra el bypass exacto de R376: una línea de resultado
contradictoria seguida por una cita literal positiva ya produce dos matches del
mismo campo y es rechazada por separado tanto para `A` como para `F`. También
conserva los controles previamente confirmados sobre paths documentales,
parentesco directo, blobs Git, `HEAD == F`, origen físico, no-redraw, orden
pre-oráculo y replay. La suite focal completa pasa `17/17` en CPU.

La implementación todavía no está lista para producir `J`. El nuevo parser
opera sobre líneas crudas, pero no comprueba que el bloque sea prosa visible del
informe. Una cadena completa y limpia `P→I→A→J→F` es aceptada cuando las líneas
canónicas están únicamente dentro de un comentario HTML; fuera de ese bloque
oculto el documento declara `REVISE`. El bypass se reproduce de extremo a
extremo por separado para `A` y para `F`. Un fenced code block produce la misma
aceptación. Por tanto, el cambio evita la cita duplicada ensayada por el test,
pero aún permite presentar una cita o ejemplo como si fuera la atestación que
habilita el recovery.

## Alcance y frontera de inspección

Audité el árbol exacto de
`a08aeaed403b0f315643c42b4c3c91ec10933782`, cuyo único parent es
`e4651d2efa0157c7e01bb1993d7158d48afefd42`, y el diff exacto `parent..I`.
El delta contiene sólo:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`: endurecimiento
  del parser de atestaciones;
- `tests/test_wave56_preoracle_recovery.py`: negativos independientes para
  informe contradictorio en `A` y en `F`.

El diff suma 42 inserciones y 4 eliminaciones, no contiene otros paths ni
errores de whitespace. El plan canónico conserva SHA-256
`d4f306ffb8d34a9d3499b99e18434d57085438fd4cd0202f4f8658d3bdbe923d`
en el parent, en `I` y en el worktree. Los blobs del preparador y del test en
`I` coinciden con los hashes de la cabecera y con los archivos del worktree.

Todas las pruebas se ejecutaron con `CUDA_VISIBLE_DEVICES=''` y
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`. No usé GPU ni web, no hice commits y no modifiqué
código, tests, configs, planes, datos ni artefactos existentes.

Del intento fallido
`data/geometria_proporcional/wave56_contextual_gate_fresh_v1.failed_20260903T171827485015Z`
inspeccioné solamente nombres, tipos, permisos, ownership, tamaños y hashes.
No abrí valores del escrow, miembros sellados, truth, labels u oracle. El árbol
observado conserva 6 directorios y 18 archivos regulares, todos `root:root`,
sin symlinks ni tipos especiales. Los hashes públicos de escrow, freeze
pre-generación, `FAILURE.json` y manifest coinciden con el plan. El inventario
no muestra `inference/`, `authorized_labels/`, `bundles/`, `phases/`, oracle
materializado ni freezes o receipts posteriores.

## Archivos leídos completos

- `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/330_wave56_stage1_preoracle_recovery_plan_audit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/331_wave56_stage1_preoracle_recovery_plan_reaudit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/375_wave56_stage1_preoracle_recovery_implementation_independent_audit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/376_wave56_stage1_preoracle_recovery_implementation_reaudit.md`
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`
- `tests/test_wave56_preoracle_recovery.py`

También inspeccioné la historia Git mínima para fijar el plan, el parent de
`I`, los blobs auditados y la secuencia exigida.

## Finding por severidad

### ALTA — Un bloque citado u oculto todavía puede constituir la aprobación de `A` o `F`

**Observación.** `_require_report_fields` divide el Markdown en líneas y busca
cada prefijo globalmente (`prepare_wave56_fresh.py:748-766`). Después sólo
comprueba contigüidad y posición anterior a la primera línea que comienza con
`## ` (`prepare_wave56_fresh.py:767-771`). No exige que el bloque esté en la
cabecera visible ni rechaza contextos Markdown que cambian su significado,
como fences o comentarios HTML. Las dos llamadas que gobiernan `A` y `F` usan
esta misma primitiva (`prepare_wave56_fresh.py:980-989`, `:1013-1021`).

**Reproducción.** Primero probé directamente el parser con los cuatro campos de
`A`: aceptó el caso canónico y rechazó resultado contradictorio más cita,
duplicación, separación, reordenamiento y posición posterior a `##`; sin
embargo, aceptó el mismo bloque dentro de un fence seguido por una decisión
`REVISE`, y también dentro de un comentario HTML seguido por esa decisión.
La matriz equivalente para los campos de `A` y `F` dio cuatro aceptaciones:
fence y comentario en cada rol.

Luego construí repos Git efímeros con cadenas completas, limpias y de
parentesco directo. En una cadena oculté la supuesta aprobación de `A` dentro
de `<!-- ... -->`; en otra hice lo mismo con `F`. Fuera del comentario cada
informe declaraba `REVISE`. `validate_recovery_amendment` aceptó ambas cadenas
de extremo a extremo. Los repos temporales fueron eliminados al terminar y no
tocaron el worktree.

**Impacto.** El amendment puede ligar como aprobación una cadena que el
renderer no presenta como dictamen emitido. Esto conserva el bypass semántico
de R376 bajo otra representación y rompe la condición del plan que asigna a
`A` la aprobación de la implementación y a `F` la autorización procedimental
final. Los bindings Git y el path `.md` no compensan que el propio contenido
aprobatorio pueda ser sólo una cita u ocultarse.

**Corrección necesaria.** Validar una cabecera canónica completa, no líneas
globales. Como forma mínima y determinista, exigir título visible en la primera
línea, línea en blanco, bloque de campos en posiciones fijas, otra línea en
blanco y recién después el cuerpo; rechazar fences y comentarios HTML antes o
dentro de esa cabecera. Agregar adversariales separados para `A` y `F` con el
bloque positivo dentro de fence y comentario, seguido por un dictamen negativo
visible. La corrección vuelve a tocar preparador y tests, por lo que requiere un
nuevo `I` y reiniciar desde allí la cadena.

### MEDIA

Ninguno adicional.

### BAJA

Ninguno que justifique ampliar el cambio.

## Controles confirmados

1. El path del plan es canónico, su hash permanece congelado y el único delta
   de `I` son preparador y test.
2. Los hashes de blobs Git, del worktree y de la cabecera coinciden para ambos
   archivos auditados.
3. Las relaciones `A←I`, `J←A` y `F←J` requieren parent directo; un commit
   intermedio, aunque luego se revierta a diff neto cero, no puede atravesarlas.
4. `A` y `F` deben ser `.md` distintos directamente bajo el directorio
   canónico de informes. Los negativos ejecutables separados siguen pasando.
5. Un resultado exacto `REVISE` seguido por la línea positiva exacta se rechaza
   separadamente en ambos roles; el finding actual necesita envolver el único
   bloque positivo en semántica Markdown no afirmativa.
6. `F` debe ser `HEAD` exacto y el worktree debe estar globalmente limpio antes
   del recovery.
7. El origen fallido se liga mediante whitelist físico y hashes, se revalida
   antes de extraer claves y después de regenerar, y el manifest debe coincidir
   antes de inferencia.
8. Recovery y replay reutilizan exclusivamente las claves del escrow; el modo
   amended no puede llegar a `secrets.token_bytes` ni autorizar un primary
   fresco.
9. Los conteos total/elegible permanecen separados y el filtro se aplica por
   fila antes de deduplicar tokens.
10. Replay compara escrow, freeze pre-generación, amendment, manifest,
    visibles, logits, freeze de preparación y procedencia/conteos relevantes.

## Comandos y pruebas

- Identidad e historia: `git rev-parse`, `git show -s`, `git log`,
  `git diff-tree` y `git diff parent I`.
- Integridad: `sha256sum` del worktree y `git show I:path | sha256sum`; hash del
  plan en parent, `I` y worktree; `git diff --check`.
- Suite focal CPU-only:
  `venv/bin/python -m pytest -q tests/test_wave56_preoracle_recovery.py`:
  **17 passed in 13.54s**.
- Inventario físico con `find -xdev` y hashes limitados a los cuatro artefactos
  públicos autorizados del intento preservado.
- Adversariales temporales del parser: canónico aceptado; duplicado,
  reordenado, no contiguo, posterior a sección y contradicción exacta
  rechazados; fence y comentario HTML aceptados.
- Dos adversariales Git temporales de extremo a extremo: aceptación indebida
  del bloque oculto en `A` y aceptación indebida del bloque oculto en `F`.

No ejecuté suite amplia: el finding se reproduce dentro de la superficie focal
y no depende de otros módulos.

## Límites

- No leí ni recomputé desde `benchmark/sealed/**` los conteos oficiales; esta
  auditoría confirma inventario, permisos y hashes, no contenido secreto.
- No ejecuté recovery ni replay oficiales porque `J` y `F` todavía no existen
  para este `I`, y el finding impide autorizar esa secuencia.
- Los recovery/replay físicos de la suite son sintéticos. Sí cubren el camino
  transaccional y el replay exacto sin abrir el origen oficial.

## Decisión

`REVISE`. No avanzar a `J`, no ejecutar el recovery oficial y no abrir
inferencia, oracle ni labels. El commit corrige la duplicación contradictoria
ensayada por R376, pero la aprobación sigue siendo fail-open ante un único
bloque positivo que Markdown representa como cita de código o contenido
oculto.
