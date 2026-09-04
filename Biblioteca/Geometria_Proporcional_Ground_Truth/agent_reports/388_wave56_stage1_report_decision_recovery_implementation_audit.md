# R388 — Auditoría independiente de la implementación de autoridad documental del recovery de Wave 56 Stage 1

**Implementation commit:** `3f404103111a67721fa7a3d15cbf4ec392025e5f`
**Runner commit:** `7b37b5381b0c7540e86de2d53001903475d321ab`
**Preparer SHA-256:** `0e0f076b4b789b9a520fdb6b6319b40d07a75cc9208c8394c764b07023a51b44`
**Runner SHA-256:** `a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`
**Test SHA-256:** `e8d25934c3da6bed2eeb9db63035e2371eb78ad418adebd6110775a2ec696017`
**Result:** `REVISE`

## Dictamen ejecutivo

I4 implementa correctamente la gramática terminal que faltaba en R384 y la
cadena de autoridad aprobada por P6. El parser exige cabecera única, resultado
válido, decisión terminal única y concordante, ausencia de contenido posterior,
UTF-8/LF canónico y ausencia total de fences, comentarios HTML y separadores no
canónicos. La procedencia también queda bien separada: I4 modifica sólo el
preparador y el test, mientras el runner final conserva exactamente el blob de
I3. El schema liga P6 y R387 mediante commit, path y SHA-256, y verifica ambos
commits de introducción, diffs exclusivos y parents directos.

La implementación requiere revisión porque la focal no materializa toda la
matriz adversarial que P6 declaró obligatoria. Los 46 tests pasan, pero faltan
negativos explícitos para un `runner_commit` falso o no ancestral, I4 con uno de
sus dos paths requerido ausente, y varios bypasses del primer borde: auditoría
de plan omitida, mutada, introducida junto con otro path o con parent incorrecto.
También falta el caso de R387 con cabecera `PASS` y decisión final `REVISE`.
Los guards correspondientes existen y no encontré un bypass directo por
inspección, pero P6 exige que la focal demuestre estos rechazos. La ausencia de
esas pruebas deja sin verificar precisamente el nuevo borde de autoridad.

El resultado es `REVISE`. La corrección es acotada al fixture y a negativos del
test focal; no requiere cambiar el runner, no-redraw ni el diseño del parser.

## Alcance

Leí completos P6 y R384–R387, el preparador y test vigentes y el runner completo
resuelto tanto desde I3 como desde I4. Inspeccioné objetos Git, parents, diffs,
commits de introducción, blobs y hashes. No abrí escrow, secretos, datos
sellados, truth, labels, oracle o resultados oficiales; no usé web o GPU y no
ejecuté recovery, replay ni fases oficiales.

## Identidad, diff y lineage

I4 tiene un solo parent,
`793c4d3655ebdd3e45cb9dba91dd34bf082c8519`, que introduce exclusivamente
R387 y desciende directamente de P6. I4 cambia exactamente:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `tests/test_wave56_preoracle_recovery.py`.

Los blobs de I4 y del worktree coinciden con los hashes de cabecera. El runner
en I3, I4 y el worktree es byte-idéntico y tiene SHA-256
`a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`.
I3 es ancestro de P6 e I4. `git diff --check` no reportó errores.

P6 fue introducido exclusivamente por
`32cd3f0b3f9c32c27c85956750c6db9992398b44` y conserva SHA-256
`ab67928fd595aa7e0c2ab670351f12370fd0b3770ea838aeb4ecd9bbd978f09b`.
R387 fue introducido exclusivamente por el parent directo de I4, conserva
SHA-256 `999524e7dc9d9098438b94a540436076bf935187c49d4594ebedaac5c2cebe59`
y termina con decisión `PASS` concordante.

## Parser y cadena de autoridad

`_require_report_fields` construye la decisión esperada desde el único campo de
resultado requerido y exige que las cuatro líneas finales sean exactamente la
sección machine-verifiable, su línea vacía, la decisión concordante y el LF
terminal. Además rechaza cualquier otra línea que empiece con el campo de
decisión final. La búsqueda global de marcadores de fence, comentarios HTML y
separadores no canónicos cierra el bypass estructural de R384 sin intentar
inferir la semántica libre del cuerpo.

`validate_recovery_amendment` exige `plan_audit = {commit, path, sha256}`,
resuelve el commit de introducción único de P6 y R387, compara los commits
declarados, limita ambos diffs a su único Markdown, valida la cabecera y decisión
de R387, y exige los bordes directos P6→R387→I4. Luego fija
`runner_commit=I3`, exige el diff exacto de I3 y el de I4, liga los tres blobs
finales y prohíbe que el runner cambie después de I3. El patrón no circular para
R389 queda preservado: el amendment sólo declara su path y el preflight deriva
commit y hash después de J6.

## Finding material

### F1 — Medio — La focal no implementa toda la matriz adversarial congelada

P6 exige que el fixture separado I3/P6/R387/I4/R388/J6/R389 rechace cada cruce
incorrecto de lineage y autoridad. El fixture actual sí separa los siete
commits y cubre hash falso de preparador/runner, path adicional en I4,
interposición entre R387 e I4, cambio del runner después de I3, resultado
`REVISE` de R387, contradicción cabecera/decisión en R388 y R389, fences y los
dos deltas contractuales.

Sin embargo, los parámetros del fixture y la matriz de
`test_recovery_amendment_rejects_broken_provenance` no generan estos casos
requeridos:

- `runner_commit` falso y `runner_commit` válido pero no ancestro de P6/I4;
- I4 con preparador o test ausente, en contraste con el único caso de path
  adicional;
- R387 omitido o con hash mutado;
- commit de R387 no exclusivo o cuyo parent no sea P6;
- R387 con cabecera `PASS` y decisión terminal `REVISE`.

Los guards de `prepare_wave56_fresh.py:989-1074` parecen rechazar esos estados,
pero los negativos presentes en `tests/test_wave56_preoracle_recovery.py:246-504`
y `tests/test_wave56_preoracle_recovery.py:552-650` no lo demuestran. Esto
incumple la sección 6 de P6 y deja el nuevo primer borde menos probado que los
informes posteriores. Deben añadirse casos end-to-end al mismo fixture y volver
a ejecutar la focal antes de crear J6.

## Dos deltas de fuente y no-redraw

Sobre las 26 fuentes congeladas por la configuración prospectiva, la comparación
entre el commit de origen `51aae0715dfe8318f5333c568429c8e9af59f866` e I4
arroja exactamente dos cambios: preparador y runner. Sus hashes finales son los
de la cabecera. `_validate_contract_delta` exige igualdad del conjunto de
fuentes y el conjunto exacto de esos dos deltas; la focal rechaza tanto un tercer
delta como la ausencia del delta del runner.

No-redraw permanece intacto: un amendment no autoriza un primary fresco,
recovery/replay no aceptan claves externas junto con escrow reutilizado, las
claves se extraen del escrow durable y el origen físico se revalida antes de la
extracción y después de regenerar. La prueba física mantiene el fallo explícito
de cualquier llamada a `secrets.token_bytes`, el replay exacto y los rechazos de
mutación del origen y del inventario visible. No encontré un bypass adicional
en estas fronteras.

## Pruebas

La focal CPU-only terminó `46 passed in 16.68s`, sin fallos ni skips, con GPU
oculta y threads numéricos limitados a uno. No se realizó ninguna ejecución
oficial.

Este dictamen es técnico y pre-oráculo. No autoriza recovery, fases, labels u
oracle y no constituye `GO/NO-GO` científico.

## Machine-verifiable decision

**Final decision:** `REVISE`
