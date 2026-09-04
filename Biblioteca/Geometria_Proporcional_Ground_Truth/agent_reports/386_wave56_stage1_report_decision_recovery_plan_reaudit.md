# R386 — Reauditoría independiente del plan v2 de autoridad del recovery de Wave 56 Stage 1

**Plan commit:** `2832bb0b83c7343aa8370ccdd109daa659b59a23`
**Plan SHA-256:** `6deafe7660b18f92a54b435239166d2c8956ac62cf22721bfa852d574a95d1a6`
**Result:** `REVISE`

## Dictamen ejecutivo

P5 corrige la mayor parte de los findings de R385. La DAG ahora declara cada
flecha como parent directo, incorpora R386 como eslabón aprobatorio antes de I4,
liga su path y hash, exige commit documental exclusivo y agrega negativos para
omisión, mutación, resultado no aprobatorio y parent incorrecto. También separa
correctamente `runner_commit=I3` de `implementation.commit=I4`, fija de qué
commit sale cada blob y exige los negativos inter-commit y de source deltas que
faltaban. La prohibición total de fences en los informes de autoridad, junto
con negativos de backticks y tildes, cierra conservadoramente el caso de fence
abierto señalado por R385.

El plan todavía no es ejecutable tal como está escrito. Su schema obliga a J5 a
registrar el hash de R388, pero R388 debe ligarse al commit J5 y al hash del
propio amendment. Esto crea una dependencia circular hacia el futuro que no
puede materializarse en la DAG lineal declarada. Además, la identidad
`plan_audit` sigue omitiendo el campo `commit` que R385 exigió expresamente,
aunque el preflight proponga derivarlo del commit de introducción. Ambos puntos
se corrigen de forma documental y acotada antes de implementar.

El resultado es `REVISE`. No autoriza I4, amendment, recovery, fases ni apertura
de oracle.

## Alcance y verificaciones

Leí completos P5, R384, R385, el preparador, el runner y la focal vigentes.
Inspeccioné el commit, SHA-256, parent y diff de P5; la lineage y los blobs de
I3; y el fixture Git actual que todavía modela una implementación única. P5
resuelve al commit y SHA-256 declarados, es `HEAD`, introduce sólo su path y el
worktree estaba limpio antes de escribir este informe.

I3 es ancestro de P5, tiene un solo parent y cambia exactamente preparador,
runner y test. El runner de I3 y el de P5 son byte-idénticos y ambos tienen
SHA-256 `a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`.
No abrí escrow, secretos, sealed, truth, labels, oracle o logits; no usé web o
GPU y no ejecuté recovery, replay ni fases.

## Cierres confirmados de R385

### DAG y autoridad del plan

P5:44-60 declara la cadena directa
`P5 → R386 → I4 → R387 → J5 → R388`, diffs exclusivos, commits de
introducción únicos, validación de cabeceras y decisiones terminales,
`HEAD == R388` y limpieza global. P5:51-52 liga plan y plan audit por path y
SHA-256, y P5:79 exige que I4 sea hijo directo del commit de R386. La matriz
P5:93-103 exige rechazo de auditoría omitida, mutada, `REVISE`, no exclusiva o
con parent incorrecto, además del rechazo de I4 con parent incorrecto. Esto
cierra funcionalmente el bypass principal de R385 F1.

### Lineage I3 e I4

P5:74-89 fija I3 por constante y amendment, exige ancestry, parent único, diff
exacto y hash del runner. I4 sólo puede cambiar preparador y test, debe descender
directamente de R386 y conservar byte-idéntico el runner de I3. La cabecera de
R387 liga ambos commits y los tres blobs finales. P5:96-100 añade negativos
para commit de runner incorrecto o no ancestral, runner alterado después de I3,
hash falso, diff incorrecto de I4, pérdida de cualquiera de los dos deltas y
tercer delta. Esto cubre la separación que el fixture vigente aún no modela en
`tests/test_wave56_preoracle_recovery.py:247-330`.

### Decisión terminal y fences

P5:21-38 exige resultado de cabecera único, decisión terminal única y
concordante, LF final, ausencia de contenido posterior, comentarios HTML y
separadores no canónicos. Prohíbe cualquier fence de backticks o tildes en los
informes de la DAG, una regla más conservadora que rastrear el estado Markdown.
P5:101-103 exige negativos para R386, R387 y R388, incluidos bloques terminales
dentro de ambos tipos de fence. Esto cierra R385 F2 y el bypass de R384 sin
atribuir al parser juicio semántico sobre el cuerpo.

### Fuentes, origen y no-redraw

P5 conserva exactamente dos deltas de ejecución, preparador y runner, y rechaza
pérdida, adición o hashes sin blob. También conserva inventario y hashes del
intento fallido original, population contract, doble revalidación del origen,
manifest primary/replay, exactitud, ausencia de claves externas y archivo
recuperable del primario v2. Son consistentes con los guards vigentes de
`prepare_wave56_fresh.py:792-835`, `prepare_wave56_fresh.py:838-885` y
`prepare_wave56_fresh.py:1450-1555`.

## Findings materiales

### F1 — Alto — El hash de R388 dentro de J5 hace circular la DAG

P5:49-56 dice que el amendment registra los paths y hashes de R387 y R388. J5
se crea antes de R388, pero el informe final debe ligar J5 y el SHA-256 del
amendment. Por tanto, los bytes y el hash de R388 no existen hasta conocer J5 y
el hash del amendment; incorporar luego ese hash en J5 cambiaría el commit y el
hash que R388 acaba de declarar. No hay orden lineal
`R387 → J5 → R388` que satisfaga simultáneamente esas identidades.

El validador vigente evita correctamente esta circularidad: el amendment guarda
sólo `final_audit_path` y, después de resolver J y F, deriva en runtime el hash y
el commit del informe final, valida su parent directo, exige `F == HEAD` y liga
la cabecera de F a J y al hash del amendment
(`prepare_wave56_fresh.py:1014-1062`).

Corrección necesaria: mantener path y hash de R387 dentro de J5, pero registrar
para R388 sólo el path canónico. El preflight debe derivar el SHA-256 y el commit
de introducción de R388 desde HEAD, comprobar su diff exclusivo y parent J5, y
validar que su cabecera liga J5 y el hash ya congelado del amendment. Si se
quiere preservar el hash final en receipts posteriores, debe emitirse después
de validar R388, nunca dentro de J5.

### F2 — Medio — `plan_audit` no conserva la identidad completa exigida por R385

R385:104-110 pidió que el schema registrara para la auditoría del plan path,
SHA-256 y commit. P5:51 sí da identidad completa a `plan`, pero P5:52 enumera
para `plan_audit` sólo path y SHA-256. P5:58 permite derivar el commit de
introducción y los checks de parent/diff hacen la cadena funcionalmente fuerte,
pero el contrato almacenado queda menos explícito que la condición de cierre
auditada y que la identidad de `plan`.

Corrección necesaria: definir `plan_audit` con `commit`, `path` y `sha256`, y
exigir igualdad entre su `commit` declarado y el único commit de introducción
derivado. Mantener además diff exclusivo, parent directo P5, decisión terminal
`PASS` y parent directo de I4. El costo es mínimo porque el commit de R386 ya
existe antes de materializar J5 y no introduce circularidad.

## Reauditabilidad

Una revisión focal puede aprobar el plan cuando se elimine el hash futuro de
R388 del amendment y se complete `plan_audit.commit`. No hace falta reabrir el
diseño de lineage, los negativos, fences, source deltas, origen o no-redraw:
esas partes están suficientemente especificadas. La nueva versión debe mantener
la DAG y matriz adversarial de P5 sin reinterpretar los JSON históricos.

## Machine-verifiable decision

**Final decision:** `REVISE`
