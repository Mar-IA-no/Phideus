# R331 — Reauditoría focal del plan de recuperación pre-oráculo de Wave 56 Stage 1

> **HEAD auditado:** `8062c83ccea1fe6ce8c087f6e1c9bb1ff4ea30c2`
> **Commit de origen del escrow:** `51aae0715dfe8318f5333c568429c8e9af59f866`
> **Fecha:** 2026-09-03
> **Alcance:** cierre material exclusivo de los cuatro findings R330 y búsqueda de nuevos blockers de validez, no-redraw o replay.
> **Resultado:** `PASS / PRE-ORACLE / NO INFERENCE / NO ORACLE / NO LABELS`

## Dictamen ejecutivo

Los cuatro findings materiales de R330 quedaron cerrados en el plan vigente. El
amendment ahora liga el commit de implementación con el blob Git auditado,
separa sin circularidad las auditorías `A` y `F`, convierte el intento fuente en
un árbol físico cerrado y explicita que los 192 tokens no canónicos se solapan
con los 768 elegibles. Esas correcciones aparecen tanto en las obligaciones del
contrato como en los negativos exigidos a la implementación
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:93-178`, `:235-261`).

No encontré un blocker nuevo de validez, no-redraw o replay. El plan conserva
el único draw ya efectuado, obliga a regenerar con las mismas tres claves y
requiere que el manifest resultante sea idéntico al del intento preservado antes
de inferencia (`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:184-199`).
Replay agrega igualdad de escrow, freeze pre-generación y amendment a la matriz
exacta ya congelada (`:201-217`; plan prospectivo original `:214-229`).

Este `PASS` habilita sólo la implementación de la recuperación según la DAG
`P→I→A→J→F`. No habilita todavía recovery físico, inferencia, oracle o labels:
esas acciones siguen condicionadas a implementar el contrato, completar `A`,
publicar `J`, completar `F`, mantener `HEAD` limpio y pasar las suites.

## Evidencia material revalidada

- `HEAD` es exactamente `8062c83ccea1fe6ce8c087f6e1c9bb1ff4ea30c2` y
  estaba limpio al iniciar esta reauditoría.
- El preparador vigente conserva el blob pre-corrección
  `7ff5919d2b0bdd607ca179180c4f94de3ff5be6e23e6024b21e748d22c61fb44`;
  el guard defectuoso sigue contando todos los `pair_token` sellados y aborta
  antes del recibo de generación y de la inferencia
  (`prepare_wave56_fresh.py:820-877`). Por lo tanto, el plan precede realmente
  al cambio de código que regula.
- Ninguna fuente incluida en `required_execution_sources` cambió entre el commit
  de origen del escrow y `HEAD`. El único archivo de protocolo modificado por la
  resolución R330 es el amendment de recuperación, que no pertenece al contrato
  original del draw.
- El intento preservado mantiene SHA-256 de `FAILURE.json`
  `710b7d29de8c0436304ffb7abdfb2adcd958ed443ab72a9190ce74495e8602af`,
  manifest
  `7582efe3fdcd40125929cbe2c6783a37b1ba3f8ffb2fb6cce6b5578979d29ef8`,
  escrow
  `f86fb936651552a757b46acd56e1c17674635eb37cfc6d0cd2a8a02e2f06e978`
  y freeze público
  `c65d581a755d611f9f86264402bfea89503a8599d7967d94882d2db91f5083e8`.
- El inventario físico observado contiene sólo directorios y archivos regulares,
  todos `root:root`; no contiene `inference/`, `authorized_labels/`, `bundles/`,
  `phases/`, `generation_receipt.json`, `preparation_freeze.json`,
  `preparation_receipt.json` ni `benchmark/sealed/oracle`.
- El manifest público conserva 4992 filas para cada split de decisión, y la
  config pre-draw fija `expected_eligible_pair_tokens_per_split=768` junto con
  `no_redraw_after_escrow=true`
  (`wave56_contextual_gate_fresh.json:124-134`).
- El consumidor preexistente filtra primero
  `is_out_of_catalog == false` y
  `calibration_population == "canonical_preserving"`
  (`wave50_neural.py:123-154`). El generador asigna el `pair_token` antes de
  crear la vista `origin_translation_break` y reutiliza ese mismo token en ella
  (`wave49_generator.py:223-229`, `:317-386`), lo que sustenta el solapamiento
  que ahora explicita el plan.

La inspección del intento se limitó a nombres, tipos, modos, ownership, tamaños,
hashes, compromisos y metadatos públicos. No se leyeron valores de claves ni
contenido de truth sellada.

## Findings por severidad

### ALTA

Ninguno.

### MEDIA

Ninguno.

### BAJA

Ninguno que requiera revisar el plan. Queda como punto obligatorio para las
auditorías `A` y `F` comprobar que la implementación materializa literalmente
los enlaces ya exigidos —incluida la identidad y el hash del informe final en
los receipts y en replay—; no es una deuda del diseño vigente ni una licencia
para resolverlos de forma más laxa.

## Cierre de los cuatro findings R330

### 1. Commit de implementación ligado al blob auditado — CERRADO

El JSON debe declarar el commit de implementación y el hash corregido del
preparador; el validador debe leer directamente con Git el blob
`<implementation_commit>:experiments/geometria_proporcional/prepare_wave56_fresh.py`
y exigir ese hash. También debe comprobar el archivo del worktree contra el
mismo hash y `HEAD`, y el informe `A` debe nombrar el mismo par commit/hash
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:99-105`, `:144-159`).

La DAG prohíbe además cambios ejecutables posteriores a `I`: `J` agrega sólo el
JSON y `F` no toca código, config ni amendment (`:111-128`). Los negativos
incluyen explícitamente commit ancestro con blob distinto, informe desalineado y
cualquier delta ejecutable posterior (`:239-244`). Esto cierra tanto el bypass
criptográfico como su ruta procedimental.

### 2. Dos auditorías y secuencia no circular — CERRADO

El plan congela cinco hitos distintos: plan `P`, implementación `I`, auditoría
de implementación `A`, JSON `J` que liga `P+I+A`, y auditoría final `F` de
`I+A+J` y las suites (`:111-122`). Declara de manera inequívoca que `A` es el
informe incluido en el amendment y que `F`, necesariamente posterior y externo
al JSON, autoriza procedimentalmente la corrida; su identidad y contenido deben
propagarse a los receipts, al freeze de preparación y al replay (`:124-128`).
No queda autorreferencia.

### 3. Árbol físico cerrado del intento fuente — CERRADO

El amendment debe contener un whitelist recursivo exacto con directorios y
archivos, tipo, modo, uid/gid, bytes y SHA-256. La validación usa `lstat` y
rechaza symlinks, tipos especiales, extras, ausencias, escapes o divergencias;
así, la ausencia de material posterior deriva del conjunto cerrado y no de una
blacklist (`:160-169`). El plan liga además el hash exacto de `FAILURE.json` y
ordena dos validaciones completas, antes de extraer las claves y después de
regenerar el benchmark pero todavía antes de inferencia (`:160-178`). Los tests
negativos cubren extra, symlink, owner/modo y mutación entre lecturas
(`:245-249`).

### 4. Solapamiento de los 192 tokens no canónicos — CERRADO

El diagnóstico ya no presenta tres particiones disjuntas. Explica que los 192
tokens con filas no canónicas reutilizan tokens de realizaciones base elegibles
(`:37-49`) y congela por split las cinco cardinalidades: 4992 filas, 1152 tokens
totales, 768 elegibles, 384 out-of-catalog, 192 no canónicos e intersección
elegible/no-canónica de 192 (`:174-178`). El predicado debe aplicarse por fila
antes de deduplicar, y existe un test obligatorio específico de esa álgebra
(`:237-251`).

## Búsqueda de regresiones de validez, no-redraw y replay

### Validez

El delta permitido sigue limitado al predicado del guard y a la procedencia
necesaria para recuperarlo. No cambian config prospectiva, generador, claves,
benchmark, modelos, features, seeds, criterios, estimando, selector, bordes ni
workers (`:75-91`, `:221-233`). El contrato dual separa expresamente el origen
inmutable del escrow del `HEAD` de ejecución recuperada (`:130-137`), sin
atribuir retrospectivamente el código corregido al sorteo original.

### No-redraw

El plan restringe el amendment al basename y manifest del único intento, exige
su whitelist físico exacto y prohíbe usarlo para un primary nuevo o para otro
archivo fallido (`:160-182`). Recovery extrae exclusivamente las claves del
escrow durable y vuelve a publicar escrow y freeze byte-idénticos; cualquier
fallo archiva el nuevo estado antes de inferencia y nunca modifica el intento
fuente (`:184-199`). Esto conserva y endurece la semántica vigente del
preparador, que ya bloquea un nuevo primary cuando existe un archive con escrow
(`prepare_wave56_fresh.py:251-300`).

### Replay

Además de la matriz original de contenidos y arrays, replay debe usar el primary
recuperado como única fuente/referencia y el mismo amendment. Compara bytes/hash
de escrow y freeze pre-generación, amendment, ambos contratos, manifest contra
intento y primary, y conteos totales/elegibles (`:201-217`). La suite exige
replay físico sintético y validación del manifest contra todos sus miembros
(`:252-258`). No queda una exclusión capaz de ocultar un delta científico como
receipt meramente operativo.

## Veredicto

**PASS.** Las condiciones materiales de R330 quedaron incorporadas al plan y no
apareció un blocker nuevo de validez, no-redraw o replay. Puede comenzar `I`
según la DAG congelada; la corrida oficial permanece cerrada hasta que `A`, `J`
y `F` existan, la implementación satisfaga todos los negativos, las suites estén
verdes y `HEAD` esté limpio. Este dictamen no declara `GO/NO-GO` científico.
