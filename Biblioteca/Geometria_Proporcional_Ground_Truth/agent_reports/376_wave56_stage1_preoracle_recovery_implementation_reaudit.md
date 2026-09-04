# R376 — Reauditoría independiente de implementación del recovery pre-oráculo de Wave 56 Stage 1

**Implementation commit:** `08ca02d41d8b20add6efdaf7657016e2d7a0074d`
**Preparer SHA-256:** `37f705d4b8d9be447f8a7b078436dc3dc94f6f5bf31ddb9b520d3cf6c1c0baca`
**Test SHA-256:** `05c31548eb30cad5f92baec436170bd0c4d72269a4c3e21c381795ecc35e90a4`
**Result:** `REVISE`

## Dictamen ejecutivo

El commit auditado cierra el finding R375 sobre paths ejecutables: `A` y `F`
deben ser archivos `.md` distintos ubicados directamente bajo
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/`. Los dos
negativos independientes construyen un `tests/conftest.py` Python válido y
ejecutable como `A` y como `F`; ambos son rechazados. La suite focal completa
pasa `15/15` en CPU.

La implementación todavía no está lista para producir `J`. El validador
determina la aprobación de `A` y `F` buscando subcadenas en el texto. Dos
cadenas Git sintéticas completas, limpias y con parentesco directo demostraron
que acepta un informe cuyo dictamen explícito es `REVISE` si otro párrafo cita
la cadena literal esperada para un resultado positivo. El bypass funciona por
separado para `A` y para `F`. Por tanto, la nueva frontera de path impide que un
`conftest.py` modifique pytest, pero no garantiza todavía que las auditorías que
habilitan el recovery realmente hayan emitido `PASS`.

## Alcance y frontera de inspección

Audité el árbol exacto de
`08ca02d41d8b20add6efdaf7657016e2d7a0074d`, cuyo único parent es
`84a0f327773bd3d8650a0da443f75d15305e02ae`, y el diff exacto `parent..I`.
El delta contiene únicamente:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`: 20 inserciones;
- `tests/test_wave56_preoracle_recovery.py`: 31 inserciones y 7 eliminaciones.

Todas las pruebas se ejecutaron con `CUDA_VISIBLE_DEVICES=''` y
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`. No usé GPU ni web, no hice commits y no modifiqué
código, tests, configs, planes, datos ni artefactos existentes.

Del intento fallido
`data/geometria_proporcional/wave56_contextual_gate_fresh_v1.failed_20260903T171827485015Z`
inspeccioné sólo nombres, tipos, permisos, ownership, tamaños y hashes. No abrí
ni expuse valores del escrow, miembros sellados, truth, labels u oracle. El
árbol observado conserva 6 directorios y 18 archivos regulares, todos
`root:root`, sin symlinks ni tipos especiales. No aparecen `inference/`,
`authorized_labels/`, `bundles/`, `phases/`, `benchmark/sealed/oracle`,
`generation_receipt.json`, `preparation_freeze.json` ni
`preparation_receipt.json`. Los cuatro hashes públicos revalidados coinciden
con el plan: escrow `f86fb936…e978`, freeze pre-generación
`c65d581a…083e`, `FAILURE.json` `710b7d29…2af` y manifest
`7582efe3…9ef8`.

## Archivos leídos completos

- `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/330_wave56_stage1_preoracle_recovery_plan_audit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/331_wave56_stage1_preoracle_recovery_plan_reaudit.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/375_wave56_stage1_preoracle_recovery_implementation_independent_audit.md`
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`
- `tests/test_wave56_preoracle_recovery.py`

También inspeccioné la historia Git mínima para fijar el plan canónico, el
parent de `I`, los blobs auditados y la secuencia del recovery.

## Finding por severidad

### ALTA — Un informe con dictamen `REVISE` puede satisfacer la atestación `PASS`

**Observación.** `_require_report_fields` lee todo el Markdown y considera
presente cada campo cuando la cadena aparece en cualquier posición
(`prepare_wave56_fresh.py:748-752`). La función no interpreta un bloque de
atestación, no exige líneas autónomas o únicas y no rechaza un `Result`
contradictorio. Esa misma función valida el supuesto `PASS` de `A`
(`prepare_wave56_fresh.py:961-970`) y de `F`
(`prepare_wave56_fresh.py:994-1002`).

**Reproducción.** En repos Git temporales construí dos veces la secuencia
directa `P→I→A→J→F`, con worktree limpio, hashes correctos, blobs ligados y
todos los commits exclusivos. En el primer caso `A` contenía como dictamen:

```text
**Result:** `REVISE`
```

y luego citaba en prosa el ejemplo literal positivo. En el segundo caso hice lo
mismo con `F`. `validate_recovery_amendment` aceptó ambas cadenas de extremo a
extremo y devolvió contexto de recovery. La salida fue:

```text
END_TO_END_CONTRADICTORY_A_ACCEPTED final.md
END_TO_END_CONTRADICTORY_F_ACCEPTED final.md
```

Los repos temporales fueron eliminados al salir y no tocaron el worktree.

**Impacto.** El plan condiciona la habilitación técnica a una auditoría
independiente `PASS` y asigna a `F` la autorización procedimental final
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:124-128`, `:263-269`).
Aceptar como positiva una auditoría que explícitamente ordena revisar rompe esa
compuerta fail-closed. Es un bypass de procedencia, aunque ya no pueda usarse un
archivo Python fuera de `Biblioteca/`.

**Corrección necesaria.** Parsear una atestación inequívoca en vez de buscar
subcadenas. Como cambio mínimo, exigir que cada campo ligado sea una línea
autónoma exacta y única, y que exista exactamente una línea `**Result:** ...`,
cuyo único valor admisible sea `PASS`. Agregar negativos independientes para
`A` y `F` con dictamen `REVISE` más una cita literal del marcador positivo. Una
cabecera canónica estructurada sería aún más robusta, pero no es necesaria para
cerrar este bypass.

### MEDIA

Ninguno adicional.

### BAJA

Ninguno que justifique ampliar el cambio.

## Controles confirmados

1. **Plan canónico y binding de blobs.** El path del plan permanece fijado y el
   blob del plan debe estar congelado en el parent de `I`. `I` tiene un solo
   parent, cambia exclusivamente preparador y test, y sus blobs Git coinciden
   byte a byte con los hashes declarados arriba. `git diff --check` no reporta
   errores.
2. **Cadena directa sin commits intermedios.** El validador exige parentesco
   directo `A←I`, `J←A` y `F←J`; también exige `F == HEAD`. El commit `I`
   auditado desciende directamente del predecessor explícito
   `84a0f327…`. En esta etapa `J` y `F` aún no existen, por lo que esas
   propiedades se verificaron sobre el código y los repos sintéticos, no sobre
   una corrida oficial materializada.
3. **Frontera documental de `A/F`.** `require_audit_report_path` restringe ambos
   paths a `.md` directamente bajo el directorio canónico; el validador general
   rechaza además paths no canónicos, symlinks, archivos no versionados, sucios
   o distintos de `HEAD`. `A` y `F` deben usar paths distintos
   (`prepare_wave56_fresh.py:955-978`). Casos nested, traversal y extensión
   alternativa fueron rechazados; una forma con `//` sólo atraviesa el helper
   específico y luego es rechazada por `require_repo_artifact` como path no
   canónico.
4. **Negativos ejecutables separados.** El harness crea un
   `tests/conftest.py` con `pytest_collection_modifyitems` para `A` y, en otro
   caso, para `F` (`tests/test_wave56_preoracle_recovery.py:211-231`,
   `:287-305`). Los dos casos parametrizados son distintos y pasan al esperar
   el rechazo (`:376-404`).
5. **Fail-closed operativo pre-oráculo.** Amendment, `A`, `J`, `F`, `HEAD`
   limpio, origen preservado y escrow reutilizado se validan antes de crear o
   archivar output. La segunda validación del árbol origen, la identidad del
   manifest y los conteos ocurren antes de inferencia. El script no materializa
   oracle ni labels y vuelve a comprobar la frontera preparada antes de
   publicar freezes/receipts.
6. **No-redraw y escrow inmutable.** Recovery/replay con escrow reutilizado no
   admiten `keys_override`; sólo primary fresco llega a `secrets.token_bytes`.
   Escrow y freeze republicados deben conservar los hashes del origen, y el
   whitelist se revalida antes de extraer claves y después de regenerar.
7. **Replay exacto.** Se comparan manifest, protocolo, visibles, logits,
   escrow, freeze pre-generación, copia del amendment, freeze de preparación y
   campos de procedencia/conteos de `generation_receipt`. El test físico
   sintético de recovery y replay pasó dentro de la suite focal.

Estos controles son efectivos, pero no compensan la aceptación semánticamente
contradictoria de `A` o `F`.

## Comandos y pruebas

- `git show -s --format=... I` y parent; `git diff --stat`, `--name-status`,
  `--full-index`, `git diff-tree`, `git log` focal y `git diff --check`.
- `sha256sum` del worktree y `git show I:path | sha256sum`: preparador
  `37f705d4…baca` y test `05c31548…a4` idénticos a sus blobs en `I`.
- `venv/bin/python -m pytest -q tests/test_wave56_preoracle_recovery.py` bajo
  entorno CPU-only: **15 passed in 12.90s**.
- Ejecución focal explícita de los dos negativos `tests/conftest.py`: **2 passed
  in 1.29s**.
- Inventario físico con `find -xdev`, metadatos y `sha256sum` limitado a los
  cuatro artefactos públicos autorizados del intento preservado.
- Dos adversariales Git temporales de extremo a extremo: aceptación indebida
  de `A=REVISE` y de `F=REVISE` cuando el mismo Markdown cita la subcadena
  positiva.

No repetí la suite amplia: no apareció un finding que dependiera de otra
superficie y la focal cubre el recovery físico/replay sintético.

## Límites

- No leí ni recomputé el contenido de `benchmark/sealed/**`, escrow, truth,
  labels u oracle. Esta auditoría confirma inventario, permisos y hashes, no
  vuelve a adjudicar datos secretos ni los conteos del benchmark oficial.
- No ejecuté recovery ni replay oficiales: `A`, `J` y `F` todavía deben seguir
  la secuencia y el blocker actual impide autorizarla.
- `HEAD == F` sólo puede verificarse materialmente cuando existan `J` y `F`; en
  el commit `I` se verificó la lógica que lo exige y su reproducción sintética.

## Decisión

`REVISE`. No avanzar a `J`, no ejecutar el recovery oficial y no abrir
inferencia, oracle ni labels. La corrección de paths resuelve el bypass
ejecutable de R375, pero la compuerta de aprobación sigue siendo fail-open ante
un informe contradictorio por su validación mediante subcadenas.
