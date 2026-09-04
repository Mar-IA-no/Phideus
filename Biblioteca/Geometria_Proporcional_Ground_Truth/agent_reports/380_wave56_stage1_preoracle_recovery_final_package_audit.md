# R380 — Auditoría final independiente del paquete de recovery pre-oráculo de Wave 56 Stage 1

**Audited package commit:** `5f357026c6958691753cdd105aee109c26e038f6`
**Amendment SHA-256:** `67bfecbff47e934b514c031e93f2cd54dfe403eefc6e9a244c2b4778ba7c05e7`
**Result:** `PASS`

## Dictamen ejecutivo

El paquete real `I+A+J` conserva correctamente la DAG, los commits exclusivos,
los blobs ligados, el amendment canónico, el contrato dual con un único delta
de fuente, las cabeceras documentales canónicas, el inventario físico del
intento fallido, no-redraw, el orden fail-closed y la exigencia futura de que
`F` sea el `HEAD` exacto. La focal termina `33 passed` y la suite amplia exacta
Wave 49–56 termina `185 passed`, ambas en CPU y sin fallos. No existe actualmente
primary, replay, inferencia, oracle materializado, labels ni bundles de Wave 56
Stage 1.

La autorización final debe revisarse, no obstante, por un blocker de replay
reproducido de extremo a extremo sobre datos sintéticos. El replay valida el
manifest del paquete recién generado y vuelve a validar el del intento fallido,
pero no valida el manifest del primary recuperado contra sus miembros actuales.
`compare_preparation` acepta como exacta una referencia cuyo
`benchmark/sealed/train.jsonl` fue alterado después de la preparación, aunque el
validador de manifest rechaza ese mismo primary. Esto incumple la obligación
explícita de validar el manifest contra todos sus miembros y permite emitir
`all_exact=true` contra una referencia físicamente inconsistente.

El bloque fijo de cabecera conserva literalmente el layout solicitado para este
artefacto. El dictamen operativo de esta auditoría es `REVISE`, reiterado en la
última línea. Este informe no debe convertirse en el commit `F` habilitante.

## Alcance y estado auditado

La auditoría comenzó con `HEAD` exactamente en
`5f357026c6958691753cdd105aee109c26e038f6` y worktree limpio. Se inspeccionaron
los objetos Git, los blobs, el árbol de trabajo y el origen preservado dentro de
la frontera pre-oráculo. No se ejecutó el recovery oficial ni se materializó
ningún oracle o label.

La DAG real queda así:

- `I = 6e4b7925797fe7046bb7c6894919020672a15ec5`, parent único
  `4667dd31f83ff9604da318911de66434b21b739f`, modifica exclusivamente
  `prepare_wave56_fresh.py` y `test_wave56_preoracle_recovery.py`;
- `A = 1978af93b466845d07c53c5d881aae3109894ceb`, parent directo exacto de
  `I`, agrega exclusivamente R379;
- `J = 5f357026c6958691753cdd105aee109c26e038f6`, parent directo exacto de
  `A`, agrega exclusivamente el amendment JSON;
- el path congelado de `F` no tenía commit de introducción ni archivo al iniciar
  la auditoría.

Los tres diffs pasan `git diff --check`. El preparador y el test en el worktree
son byte-idénticos a sus blobs en `I`; R379 es byte-idéntico a su blob en `A`; el
amendment es byte-idéntico a su blob en `J`.

## Lecturas completas y complementarias

Se leyeron completos:

- `WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md`;
- R330 y R331;
- R375, R376, R377, R378 y R379;
- `wave56_stage1_preoracle_recovery_amendment.json`;
- `prepare_wave56_fresh.py`;
- `test_wave56_preoracle_recovery.py`;
- `wave56_contextual_gate_fresh.json`.

Se inspeccionaron además las secciones estrictamente necesarias de
`WAVE_56_STAGE1_PROSPECTIVE_IMPLEMENTATION_PLAN.md`,
`run_wave56_contextual_gate.py`, `wave49_checker.py`, `wave49_schema.py` y
`wave50_neural.py` para contrastar replay, validación de manifest, consumidor y
predicado. No se abrió contenido oficial sellado para esa inspección.

## Finding bloqueante

### ALTA — Replay no valida el manifest del primary contra sus miembros

El plan exige un replay físico sintético que incluya la validación del manifest
contra todos sus miembros
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:235-258`). R330 hace la
obligación aún más inequívoca: debe validarse contra sus miembros en ambos
paquetes (`330_wave56_stage1_preoracle_recovery_plan_audit.md:203-207`).

La implementación sí ejecuta `validate_manifest` sobre el intento fallido
(`prepare_wave56_fresh.py:835-866`) y sobre el benchmark recién generado en
recovery o replay (`prepare_wave56_fresh.py:1542-1557`). Sin embargo,
`compare_preparation` sólo carga el JSON del manifest del replay y del primary,
compara compromisos, el hash del propio manifest, protocolo, visibles, logits,
escrow, freeze, amendment y algunos campos del recibo
(`prepare_wave56_fresh.py:1404-1464`). No llama a `validate_manifest` sobre
`primary / "benchmark"` ni compara los miembros sellados del primary. La llamada
que publica `all_exact` ocurre después, sin otro control del árbol de referencia
(`prepare_wave56_fresh.py:1679-1689`).

La reproducción aislada creó dos paquetes preparados sintéticos inicialmente
idénticos, agregó un byte LF a
`primary/benchmark/sealed/train.jsonl` sin cambiar su manifest y ejecutó:

```text
compare_preparation(replay, primary, config)
```

El resultado fue `all(checks.values()) == True`. Inmediatamente después,
`validate_manifest(primary / "benchmark")` levantó `ProtocolViolation` por el
mismo cambio. La reproducción usó sólo fixtures sintéticos bajo `/tmp`, se
eliminó al terminar y no leyó ni modificó datos oficiales.

El test físico vigente prueba recovery y replay exitosos con un primary intacto
(`tests/test_wave56_preoracle_recovery.py:546-684`), pero no contiene el negativo
que altera un miembro gobernado por el manifest del primary antes del replay.

#### Impacto

El replay puede declarar exactitud contra una referencia cuyo árbol real ya no
corresponde al manifest que presenta. La regeneración sigue produciendo un
benchmark internamente válido y el no-redraw permanece intacto; el defecto está
acotado a la certificación de igualdad e integridad del primary usado como
referencia. Aun así, contradice un requisito explícito del plan y de R330, por lo
que `F` no puede autorizar procedimentalmente la ejecución.

#### Corrección mínima

Validar el manifest contra sus miembros en ambos argumentos de
`compare_preparation` —como mínimo en el primary, además del replay ya validado
antes de la llamada— y agregar un negativo que altere un miembro del manifest
del primary y exija aborto antes de escribir un recibo de replay exacto. Como la
corrección toca preparador y test, requiere un nuevo `I` y repetir sin saltos
`I→A→J→F`.

## Controles confirmados

### Git, blobs y JSON canónico

- El plan corregido conserva SHA-256
  `d4f306ffb8d34a9d3499b99e18434d57085438fd4cd0202f4f8658d3bdbe923d`.
- Los blobs de `I` coinciden con los hashes declarados: preparador
  `aa68ca636b1536df27de739927fd4afdea18e6417838b1e7e6be0dee8a8769b8`
  y test
  `4fdbe9d4043254f799beb884e77f6c05c5353c6ba645830ee3f7094c8f9c5b7d`.
- R379 coincide con el hash ligado
  `45cab2d8eddf4c1fff12d37053aa2fa14b08ddfab8cb810a723057acb2865451`.
- El amendment es JSON pretty canónico, con UTF-8/LF terminal, schema
  `wave56-stage1-preoracle-recovery-amendment-v1` y estado
  `APPROVED_PREORACLE_RECOVERY`.

### Contrato dual, predicado y conteos declarados

El contrato público de origen tiene el hash compacto y commit fijados por el
amendment. Su conjunto de fuentes coincide exactamente con
`required_execution_sources`; al compararlo con los hashes del árbol actual,
hay un solo delta: `prepare_wave56_fresh.py`. Los hashes old/new de ese delta
coinciden con el amendment y `_validate_contract_delta` acepta la reconstrucción
en memoria del contrato de ejecución actual.

El predicado declarado exige `is_out_of_catalog == false`, población
`canonical_preserving` y filtrado por fila antes de deduplicar `pair_token`. Los
tres splits declaran de forma idéntica 4992 filas, 1152 tokens totales, 768
elegibles, 384 out-of-catalog, 192 no canónicos y una intersección
elegible/no-canónica de 192. La focal confirma la semántica del filtro y los
negativos de drift. Por la frontera impuesta, no se recomputaron esas
cardinalidades desde truth sellada oficial; su soporte material se preserva por
el inventario y hashes exactos del origen.

### Cabeceras, no-redraw, fail-closed y `F` futuro

R379 usa UTF-8 canónico, sólo LF, newline terminal, H1 visible, blanco, cuatro
campos exactos/contiguos/únicos, blanco y primer H2. Los negativos de paths
ejecutables, resultados contradictorios, fences, HTML oculto y separadores no
canónicos permanecen verdes.

La recuperación y el replay sólo extraen claves del escrow reutilizado; no
admiten `keys_override`, y `secrets.token_bytes` queda en primary fresco. El
amendment, DAG, limpieza, origen y escrow se validan antes de crear o archivar
output; el origen se vuelve a inventariar antes de extraer claves y después de
regenerar, aún antes de inferencia.

La lógica exige que el futuro `F` sea un Markdown distinto bajo el directorio
canónico, que su commit agregue sólo ese path, descienda directamente de `J`,
sea exactamente `HEAD` y deje el worktree globalmente limpio. Esas condiciones
están cubiertas por código y tests sintéticos, pero no pueden materializarse en
el paquete actual porque el informe no estaba versionado y el finding impide
usarlo como autorización.

### Origen preservado y ausencia de material futuro

El inventario observado coincide exactamente con las 24 entradas del amendment:
6 directorios y 18 archivos regulares, todos `root:root`, sin symlinks ni tipos
especiales. Se comprobaron modos, tamaños y los 18 SHA-256 sin mostrar
contenido. `validate_manifest` y `validate_visible_package` pasan; los visibles
contienen 4992 filas en cada split.

No existen los paths canónicos de primary o replay. En el intento fallido no
existen `inference/`, `authorized_labels/`, `bundles/`, `phases/`, oracle
materializado, `generation_receipt.json`, `preparation_freeze.json` ni
`preparation_receipt.json`.

## Pruebas ejecutadas

Todo pytest y toda reproducción Python se ejecutaron con
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
`PYTHONDONTWRITEBYTECODE=1` y cache de pytest deshabilitada.

- Focal: `tests/test_wave56_preoracle_recovery.py` — `33 passed in 18.41s`.
- Suite amplia exacta Wave 49–56 solicitada — `185 passed in 253.16s`.
- Validadores seguros del origen: manifest `PASS`; visibles `PASS`, 4992 por
  split.
- Reproducción sintética del finding: comparación de replay aceptada con todos
  los checks verdaderos; validación del manifest del primary rechazada con
  `ProtocolViolation`.

No se ejecutó `validate_semantic_attestation` sobre el origen oficial porque su
implementación abre secretos bajo `benchmark/sealed/**`; la restricción de esta
auditoría permite inventario, permisos y hashes, pero prohíbe abrir esos valores.

## Límites y recursos

No se usó GPU, web, servicios externos ni subagentes. No se ejecutó recovery o
replay oficial, no se abrió `generation_escrow.json`, no se mostraron claves y
no se leyó contenido de truth, labels, oracle ni miembros sellados oficiales.
Las lecturas físicas del árbol oficial quedaron limitadas a metadata y hashing.
No se modificaron código, tests, configs, plan o datos y no se hizo commit.

El finding no cuestiona los conteos declarados, el único delta del contrato, la
reutilización del draw ni la seguridad pre-oráculo del recovery. Impide
únicamente declarar cerrado el contrato de replay hasta que el primary sea
validado contra todos los miembros de su manifest y el negativo correspondiente
quede verde.

REVISE
