# R378 — Cuarta auditoría independiente de implementación del recovery pre-oráculo de Wave 56 Stage 1

**Implementation commit:** `9ea8d984b8924088d472fd2b48780a0f0f1e98e9`
**Preparer SHA-256:** `b7435251a16ff6f4ec90490fd845e0e24233611ca558df87a291f20ab847351c`
**Test SHA-256:** `f6e7f92b99a0877612e8886742fdcd830bf8ab412a37d4a00df3aea157d49b96`
**Result:** `PASS`

## Dictamen ejecutivo

El commit auditado cierra los bypasses concretos de R377 para fences y
comentarios HTML: obliga a que el archivo empiece con un H1, una línea vacía,
el bloque exacto y contiguo, otra línea vacía y un H2; además rechaza los
delimitadores HTML ensayados. La suite focal completa pasa `21/21` en CPU y se
conservan los controles previos sobre paths documentales, plan canónico,
parentesco directo, blobs Git, `HEAD == F`, origen físico, no-redraw, orden
pre-oráculo y replay.

La implementación todavía no está lista para producir `J`. El parser usa
`str.splitlines()` como lexer de líneas Markdown. Esa función reconoce varios
separadores Unicode y de control que Markdown no reconoce como fin de línea.
Un archivo unido exclusivamente con `U+2028`, por ejemplo, es aceptado como si
tuviera el layout canónico, mientras un renderer Markdown lo interpreta como
un único H1: el bloque ligado no existe como bloque top-level, las líneas
vacías no existen y el supuesto primer H2 queda como texto dentro del H1. El
mismo defecto se reprodujo con `VT`, `FF`, `U+0085` y `U+2029`. Como la misma
primitiva gobierna `A` y `F`, el bypass afecta ambas compuertas.

El valor fijo de la cabecera precedente conserva el layout solicitado para
este artefacto. El dictamen operativo de esta auditoría es `REVISE`, reiterado
en la última línea.

## Alcance y estado Git

Audité el objeto exacto `9ea8d984b8924088d472fd2b48780a0f0f1e98e9`.
Tiene un único parent,
`bc59f01c35e3166f9e1a4c84c7eb88d1cb7af527`, y `HEAD` coincidía con el
objeto auditado antes de crear este informe. El diff `parent..I` contiene sólo:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `tests/test_wave56_preoracle_recovery.py`.

El diff suma 65 inserciones y 17 eliminaciones. `git diff --check` pasó. Los
hashes del worktree y de los blobs resueltos desde `I` coinciden exactamente
con los dos valores de la cabecera. El plan canónico conserva SHA-256
`d4f306ffb8d34a9d3499b99e18434d57085438fd4cd0202f4f8658d3bdbe923d`.

La historia mínima confirma la secuencia de correcciones: R375 cerró plan y
parentesco pero halló paths ejecutables; R376 confirmó paths `.md` directos y
distintos pero halló aprobación por subcadenas; R377 confirmó unicidad y orden
de líneas pero halló fences y comentarios. El nuevo `I` desciende directamente
del commit que agregó R377, como corresponde al reinicio de la cadena.

## Archivos leídos completos

- `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/330_wave56_stage1_preoracle_recovery_plan_audit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/331_wave56_stage1_preoracle_recovery_plan_reaudit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/375_wave56_stage1_preoracle_recovery_implementation_independent_audit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/376_wave56_stage1_preoracle_recovery_implementation_reaudit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/377_wave56_stage1_preoracle_recovery_implementation_final_reaudit.md`
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`
- `tests/test_wave56_preoracle_recovery.py`

También inspeccioné el diff exacto, los blobs y la historia Git estrictamente
necesaria para reconstruir `P→I→A→J→F`.

## Finding por severidad

### ALTA — Diferencial de lexer permite aprobar una cabecera que Markdown no emite

La función `_require_report_fields` lee UTF-8 y llama `splitlines()` antes de
aplicar las posiciones fijas
(`prepare_wave56_fresh.py:748-759`). En Python, ese método corta no sólo por
`LF`, `CRLF` y `CR`, sino también por separadores como tabulación vertical,
form feed, next-line y los separadores Unicode de línea y párrafo. Markdown
define sus líneas por los finales ordinarios; los demás caracteres permanecen
dentro del mismo bloque.

La reproducción construyó el H1, el vacío, los cuatro campos, el segundo vacío,
el H2 y el cuerpo, pero los unió con cada separador adversarial en lugar de
`LF`. El validador aceptó cinco variantes: `VT`, `FF`, `U+0085`, `U+2028` y
`U+2029`. Un render local con Python-Markdown 3.10.1 mostró para todas ellas un
solo H1 que absorbía campos, marcador de H2 y cuerpo. En particular, no había
un párrafo top-level de atestación ni un primer H2 real. El caso canónico con
`LF`, usado como control positivo, produjo H1, párrafo de campos y H2 separados.

Esto no es una contradicción semántica posterior al bloque autoritativo. Es una
diferencia entre el lenguaje que valida el preparador y el lenguaje que
renderiza el informe: el bloque autoritativo nunca se materializa con la
estructura exigida. Los hashes, commits y paths pueden ligar perfectamente los
bytes adversariales, de modo que las comprobaciones posteriores no cierran el
defecto. Las dos llamadas a la misma función para `A` y `F` están en
`prepare_wave56_fresh.py:986-995` y `:1019-1027`.

Corrección necesaria: tokenizar únicamente finales de línea Markdown. El
cambio mínimo es reemplazar `splitlines()` por una separación estricta por
`"\n"` después de la normalización universal de `CRLF/CR`, o leer con política
de newlines explícita y rechazar cualquier separador de control/Unicode no
admitido. Deben agregarse negativos directos y de extremo a extremo para `A` y
`F`; al menos un caso debe usar `U+2028`, y conviene cubrir toda la familia que
Python separa adicionalmente. Esta corrección vuelve a cambiar preparador y
tests, por lo que requiere un nuevo `I` y reiniciar desde él la DAG.

### MEDIA

Ninguno adicional.

### BAJA

Ninguno que justifique ampliar el delta.

## Controles confirmados

1. Los paths de `A` y `F` deben ser `.md`, vivir directamente bajo el
   directorio canónico de informes y ser distintos
   (`prepare_wave56_fresh.py:257-267`, `:980-1002`).
2. El plan se fija por path constante y hash; el parent de `I` ya contiene ese
   blob (`:943-961`).
3. `I` sólo puede contener preparador y test; ambos blobs se resuelven desde el
   objeto Git declarado y deben coincidir con amendment y worktree
   (`:968-978`).
4. `A←I`, `J←A` y `F←J` exigen parent directo; cada commit sólo puede introducir
   su artefacto, y `F` debe ser exactamente `HEAD` (`:1003-1018`).
5. Fences y comentarios HTML en la cabecera son rechazados por los negativos
   actuales; duplicación, contradicción, reordenamiento y separación de los
   campos top-level también se rechazan. Una cita posterior con prefijo de
   blockquote o lista no invalida por sí sola la atestación canónica: puede
   documentar un dictamen anterior y su coherencia semántica corresponde a la
   auditoría humana, no a este parser.
6. El origen preservado conserva 6 directorios y 18 archivos regulares, todos
   `root:root`, sin symlinks ni tipos especiales. Escrow, freeze público,
   `FAILURE.json` y manifest mantienen respectivamente los hashes congelados
   `f86fb936…e978`, `c65d581a…083e`, `710b7d29…2af` y `7582efe3…9ef8`.
7. No-redraw permanece efectivo: recovery y replay extraen las claves sólo del
   escrow durable; una ejecución amended no admite `keys_override`, y
   `secrets.token_bytes` queda en la rama primary (`:1472-1495`).
8. El origen físico se revalida antes de extraer claves y después de regenerar;
   el manifest debe conservar identidad antes de inferencia
   (`:1476-1483`, `:1549-1558`). Los conteos total y elegible siguen separados
   y el guard amended usa el segundo (`:1565-1598`).
9. Replay compara visibles, logits, manifest, escrow, freeze pre-generación,
   copia del amendment, freeze de preparación y procedencia/conteos de recibo
   (`:1415-1453`).
10. La CLI valida amendment, reports, Git, limpieza global y escrow reutilizado
    antes de crear o archivar el output (`:1756-1793`). El hallazgo actual hace
    fail-open a la atestación documental, no altera ese orden transaccional.

## Pruebas ejecutadas

- Suite focal CPU-only:
  `venv/bin/python -m pytest -q tests/test_wave56_preoracle_recovery.py`:
  `21 passed in 14.34s`.
- Matriz directa del parser: control `LF` aceptado y renderizado con bloques
  separados; cinco separadores no Markdown aceptados indebidamente y
  renderizados como un único H1.
- Inspección Git: identidad de `HEAD`, parent único, diff de dos paths,
  `diff-tree`, hashes worktree/blob y `diff --check`.
- Inventario del intento fallido con `find -xdev`, metadata y hashes; sin leer
  valores del escrow ni contenido sellado.

No corrí la suite amplia: el finding se reproduce en la superficie focal y
obliga a revisar `I`; ampliar la regresión no cambiaría el dictamen.

## Límites y riesgos residuales

No abrí valores de `generation_escrow.json`, miembros de `benchmark/sealed/**`,
truth, labels u oracle. Sólo observé inventario, permisos, ownership, tamaños y
hashes. No ejecuté recovery o replay oficiales, no materialicé `J` o `F`, no
usé web ni GPU y no modifiqué código, tests, config, plan o datos.

El test físico sintético de recovery/replay que pasa en la suite acredita el
camino transaccional bajo fixtures, no una corrida oficial. `HEAD == F` y la
cadena completa sólo pueden verificarse sobre los futuros commits reales. Tras
cerrar el diferencial de finales de línea, la superficie residual principal
será la revisión semántica humana del cuerpo: el parser debe autenticar el
bloque top-level, no intentar decidir por grep si toda mención histórica a
`REVISE` contradice ese bloque.

## Decisión

No avanzar a `J`, no ejecutar recovery oficial y no abrir inferencia, oracle o
labels. El layout ordinario y los casos conocidos de R377 quedaron cerrados,
pero una familia completa de separadores aceptados por Python y no por Markdown
permite que `A` o `F` pase sin emitir la estructura autoritativa exigida.

REVISE
