# R387 — Reauditoría focal final del plan v3 de autoridad del recovery de Wave 56 Stage 1

**Plan commit:** `32cd3f0b3f9c32c27c85956750c6db9992398b44`
**Plan SHA-256:** `ab67928fd595aa7e0c2ab670351f12370fd0b3770ea838aeb4ecd9bbd978f09b`
**Result:** `PASS`

## Dictamen ejecutivo

P6 cierra literalmente los dos findings remanentes de R386. El amendment v6
registra la auditoría de plan como `plan_audit = {commit, path, sha256}` y exige
que el commit declarado coincida con el commit de introducción único derivado.
Para la auditoría final registra solamente `final_audit_path`; su commit y hash
se derivan después de J6 desde el único commit que introduce R389. La cadena
queda lineal y verificable, sin que J6 dependa de bytes futuros de R389.

No encontré findings materiales nuevos dentro del alcance focal. El resultado
es `PASS`. Esto aprueba el plan de implementación documental; no autoriza por sí
mismo recovery, fases, acceso a oracle o labels, ni una decisión científica.

## Alcance y verificaciones

Leí completos P6, R385 y R386 y contrasté el plan con el preparador, runner y
test focal vigentes. Verifiqué que P6 resuelve al commit y SHA-256 declarados,
que es introducido por un commit exclusivo y que su parent directo es el commit
exclusivo de R386. I3 es ancestro de P6, tiene parent único y cambió exactamente
preparador, runner y test. El runner de I3 y el vigente son byte-idénticos, con
SHA-256 `a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`.

No abrí escrow, secretos, sealed, truth, labels, oracle ni logits; no usé web o
GPU y no ejecuté recovery, replay, fases ni tests.

## Cierres focales confirmados

### Auditoría final sin identidad circular

P6:42-54 conserva path y hash de R388, pero para R389 almacena únicamente el
path canónico. El preflight debe derivar commit y hash desde R389 ya existente,
exigir diff exclusivo, parent directo J6, `HEAD == R389`, worktree limpio y una
cabecera ligada al commit J6 y al hash congelado del amendment. Este orden
coincide con el patrón no circular ya presente en el código vigente para la
auditoría final y satisface la corrección requerida por R386 F1.

### Identidad completa y borde directo de `plan_audit`

P6:39-49 y P6:80-87 exigen para R387 commit, path y SHA-256, introducción única,
diff exclusivo, decisión aprobatoria coherente, parent directo P6→R387 y parent
directo R387→I4. También exigen igualdad entre el commit declarado y el derivado
y negativos para omisión, mutación, `REVISE`, commit falso, path no exclusivo e
interposición. La identidad ya no queda implícita ni lateral; esto satisface
R386 F2 y conserva el cierre funcional de R385 F1.

### Invariantes preservados

P6 mantiene la separación entre `runner_commit=I3` e
`implementation.commit=I4`: I4 sólo modifica preparador y test, mientras el
runner final permanece byte-idéntico al blob auditado de I3. La matriz conserva
negativos de lineage inter-commit, pérdida o adición de source deltas y cambio
posterior del runner. Frente al escrow de origen siguen permitidos exactamente
dos deltas de ejecución, preparador y runner.

La gramática exige una cabecera única concordante con una decisión terminal
única y prohíbe fences de backticks y tildes, comentarios HTML, separadores
alternativos y contenido posterior para R387, R388 y R389. También conserva los
34 tests de calibración, inventario, origen físico, no-redraw, manifest
primary/replay y exactitud. La ejecución posterior sigue atada al intento
fallido original, a preservación recuperable, a `HEAD == R389` y a CPU-only.

## Conclusión

Los dos únicos cierres solicitados por R386 están especificados de manera
completa, no circular y directamente testeable, sin reabrir ni debilitar
lineage, fences, negativos, procedencia de fuentes o no-redraw. P6 queda apto
para implementación bajo su DAG declarada.

## Machine-verifiable decision

**Final decision:** `PASS`
