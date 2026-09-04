# R385 — Auditoría independiente del plan de cierre documental del recovery de Wave 56 Stage 1

**Plan commit:** `fb000de9104e7ad2d472072b14d4e8220084b37e`
**Plan SHA-256:** `5e286272cb5544b938fc4a741940388e9ea11d01fad6c005d5e0eb9c9d0e86a0`
**Result:** `REVISE`

## Dictamen ejecutivo

P4 identifica correctamente el bypass de R384 y propone una gramática terminal
acotada: liga una línea única de resultado de cabecera con una decisión final
estructural, sin atribuir al parser capacidad para juzgar la semántica completa
del cuerpo. También conserva correctamente el runner de I3, separa la nueva
implementación `I4` en preparador y test, mantiene dos y sólo dos deltas de
fuente frente al origen y no altera no-redraw, el orden fail-closed ni la
preservación del recovery.

El plan todavía no es ejecutable de forma cerrada. Hay un finding alto en la
cadena de autoridad: R385 aparece en el diagrama, pero no queda ligado por los
requisitos verificables de la DAG ni por el schema v4. Además, la matriz mínima
no exige negativos para las nuevas fronteras entre `runner_commit=I3` e
`implementation.commit=I4`, y omite el caso adversarial más delicado de la
propia gramática: una sección terminal lexicográficamente exacta que queda dentro
de un fence abierto. Ambos puntos deben quedar explícitos antes de implementar.

El resultado es `REVISE`. No cuestiona la corrección ya auditada del runner ni
autoriza recovery, fases, labels, oracle o una decisión científica.

## Alcance y restricciones

Leí completos P4, P3, R383, R384 y el diff exacto de I3. Inspeccioné el
preparador, el test focal y los objetos Git necesarios para contrastar parents,
paths y blobs. P4 resuelve al commit y SHA-256 declarados; su parent directo es
el commit exclusivo de R384. El runner, el preparador y el test en P4 son
byte-exactos respecto de I3.

No abrí escrow, secretos, datos sellados, truth, labels, oracle ni logits; no
usé web o GPU; no ejecuté tests, recovery, replay o fases oficiales; no modifiqué
código, configs o datos.

## Aspectos correctos del plan

### Gramática terminal y límite semántico

La forma terminal propuesta es inequívoca para un informe aprobatorio: exige una
sección final exacta, unicidad de `**Result:**` y `**Final decision:**`, igualdad
entre ambos valores y ausencia de contenido posterior. Esta estrategia evita
inferir el dictamen mediante menciones incidentales de `PASS` o `REVISE` en el
cuerpo. P4 también dice expresamente que el parser no valida el juicio semántico
completo y conserva lectura integral y auditoría independiente. Ese alcance es
correcto: el control mecánico cierra contradicciones estructurales, no reemplaza
el juicio cualitativo.

### Lineage acumulativa, blobs y fuentes

La separación propuesta refleja el historial real:

- I3 cambió exactamente preparador, runner y test bajo P3;
- el blob actual del runner es
  `a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`
  tanto en I3 como en P4;
- I4 debe cambiar sólo preparador y test;
- el runner final debe resolverse desde I3 y permanecer idéntico en I4;
- frente al contrato del origen continúan existiendo exactamente dos deltas de
  ejecución: preparador y runner. El test está ligado por blob, pero no es una
  fuente del contrato prospectivo.

Esto evita fabricar un segundo cambio del runner y conserva procedencia dual. La
revalidación de blobs finales, ancestry y source deltas que P4 enumera es
compatible con el diseño fail-closed vigente.

### No-redraw, preservación y reejecución

P4 vuelve a partir del intento fallido original, no usa el primario v2 como
fuente de claves, conserva el archivado recuperable mediante `--force`, prohíbe
borrado y exige retener estados crudos y receipts. No introduce acceso a verdad
ni cambia el protocolo científico. La secuencia recovery → fases → replay sólo
después de `HEAD == F4` y worktree limpio conserva el orden operativo esperado.

## Findings materiales

### F1 — Alto — R385 puede quedar fuera de la autoridad verificable de la DAG

P4 dibuja `P4 → R385(plan audit) → I4 → A4 → J4 → F4`, pero sus requisitos
verificables sólo declaran como hijos directos sucesivos a `A4`, `J4` y `F4`
(P4:76-81). Para el tramo anterior exige únicamente que I3 sea ancestro de I4 y
que el plan esté congelado antes de I4 (P4:53-59). No exige que:

1. R385 use el path canónico esperado y tenga hash ligado;
2. R385 sea introducido por un commit exclusivo, hijo directo de P4;
3. R385 concluya `PASS` de forma estructuralmente coherente;
4. I4 sea hijo directo del commit que introduce R385;
5. el schema v4 registre la identidad exacta de P4 y R385.

El validador vigente refuerza el riesgo de una implementación literal: sólo
comprueba que el parent del commit de implementación contenga el blob del plan
(`prepare_wave56_fresh.py:979-987`) y recién impone parents directos desde la
auditoría de implementación hacia adelante
(`prepare_wave56_fresh.py:1039-1051`). P4 no ordena cerrar esa asimetría. Por
tanto, una implementación puede satisfacer los checks descritos aunque R385 se
omita, sea `REVISE`, no descienda directamente de P4 o no sea el parent de I4.
Eso convierte la auditoría del plan en evidencia lateral, no en el eslabón de
autorización que declara la DAG.

**Corrección necesaria.** P4 debe exigir que v4 ligue `plan.commit=P4`, el path y
SHA-256 del plan, y una identidad completa de `plan_audit` con path, SHA-256 y
commit. El preflight debe comprobar introducciones únicas, diffs exclusivos y
parents directos para `P4 → R385 → I4`, además de los ya exigidos para
`I4 → A4 → J4 → F4`. R385 debe tener resultado aprobatorio coherente antes de
que I4 pueda integrar la cadena. El test sintético debe demostrar rechazo al
omitir, sustituir, mutar o interponer commits en ese tramo.

### F2 — Medio — Los negativos mínimos no cubren el nuevo split de lineage ni el fence terminal abierto

P4 agrega negativos end-to-end para `PASS`/`REVISE`, ausencia, duplicación y
contenido posterior, pero confía en conservar las 34 pruebas para DAG, blobs y
fuentes (P4:95-108). Esas pruebas modelan el contrato anterior: el fixture
vigente crea preparador, runner y test en un único `implementation_commit`
(`tests/test_wave56_preoracle_recovery.py:247-255`) y el amendment liga el
runner dentro de esa misma implementación
(`tests/test_wave56_preoracle_recovery.py:316-329`). No existe todavía
`runner_commit` separado ni un segundo commit de implementación. Mantener esos
negativos, aun adaptados sólo para que vuelvan a pasar, no demuestra los nuevos
invariantes inter-commit.

La matriz debe exigir al menos rechazo de:

- `runner_commit` distinto de I3, no ancestro de I4 o con blob de runner
  incorrecto;
- I4 que modifica el runner, que no modifica exactamente preparador y test, o
  cuyos blobs declarados no se resuelven desde I4;
- runner final distinto del blob auditado de I3, incluso si el mapa contractual
  termina mostrando dos deltas;
- bypass, resultado no aprobatorio o parent incorrecto de R385, en coordinación
  con F1;
- source delta tercero o pérdida de uno de los dos deltas requeridos.

Hay además una omisión específica de la gramática. P4 exige rechazar una
decisión terminal cercada (P4:30-36), pero no incluye ese caso en los tests
nuevos (P4:97-100). Un check ingenuo de sufijo exacto y unicidad aceptaría este
patrón si un fence se abre antes de la sección y nunca se cierra: el texto acaba
con bytes canónicos, pero Markdown interpreta la supuesta sección dentro del
fence. Debe existir un negativo explícito para fences de backticks y tildes,
incluido un fence abierto antes del bloque terminal. No es cosmética: protege el
mismo límite estructural cuya omisión produjo R384.

**Corrección necesaria.** Rehacer el fixture de procedencia con commits
separados `I3` e `I4`, sumar los negativos anteriores y comprobar ambos reportes
aprobatorios. La implementación del parser debe reconocer que el bloque terminal
está fuera de cualquier fence, no sólo que coincide con el sufijo textual.

## Condiciones de reauditabilidad

Una revisión acotada del plan puede aprobarse cuando:

1. la DAG verificable incluya explícitamente P4 y R385 hasta I4, con identities,
   paths, hashes, diffs exclusivos y parents directos;
2. el schema v4 distinga sin ambigüedad `runner_commit=I3` de
   `implementation.commit=I4` y fije de qué commit sale cada blob;
3. la matriz adversarial cubra los cruces incorrectos entre esos commits y los
   bypasses de R385;
4. la gramática terminal tenga un negativo para fence abierto/cercado, además de
   mismatch, ausencia, duplicación y contenido posterior;
5. se mantengan las restricciones ya correctas de dos source deltas, no-redraw,
   origen exacto, preservación, CPU-only y lectura humana completa.

## Machine-verifiable decision

**Final decision:** `REVISE`
