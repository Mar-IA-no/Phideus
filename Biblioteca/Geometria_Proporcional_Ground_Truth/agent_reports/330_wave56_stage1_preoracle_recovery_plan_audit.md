# R330 — Auditoría independiente del plan de recuperación pre-oráculo de Wave 56 Stage 1

> **HEAD auditado:** `06e39215c9b72fb663893a0fc838105762796267`
> **Commit de origen del escrow:** `51aae0715dfe8318f5333c568429c8e9af59f866`
> **Fecha:** 2026-09-03
> **Alcance:** validez epistémica, no-redraw, amendment de contrato dual, replay, secuencia de commits y alternativa mínima.
> **Resultado:** `REVISE / PRE-ORACLE / NO INFERENCE / NO ORACLE / NO LABELS`

## Dictamen ejecutivo

La recuperación del draw existente es epistemológicamente defendible. El valor
esperado `768` estaba congelado antes de las claves en la config canónica
(`wave56_contextual_gate_fresh.json:124-134`), el consumidor ya definía antes
del draw exactamente la población elegible
(`wave50_neural.py:91-155`) y el fallo provino de contar todos los
`pair_token` sellados en vez de aplicar ese mismo filtro
(`prepare_wave56_fresh.py:843-859`). El árbol fallido no contiene inferencia,
oracle materializado, labels, bundles ni freezes posteriores a generación; su
`FAILURE.json` registra precisamente el guard abortado
(`FAILURE.json:1-6`). Corregir sólo ese guard, conservar las mismas claves y
prohibir cualquier otro delta no adapta el estimador, el selector ni los
criterios a resultados observados.

El plan, sin embargo, todavía no debe implementarse literalmente. Su vínculo
entre `implementation_commit`, hash corregido y auditoría es insuficiente: sólo
exige que el commit declarado sea ancestro de `HEAD`, no que el blob del
preparador en ese commit sea el blob cuyo hash fue auditado. Esto permite que
un commit posterior cambie preparador y amendment de manera coordinada mientras
conserva como ancestro y como informe ligado una implementación anterior. Hay
además tres ambigüedades que no invalidan la idea del contrato dual, pero sí
deben quedar resueltas antes de convertirla en código.

## Findings por severidad

### ALTA — El commit de implementación no está criptográficamente ligado al blob auditado

**Observación.** El amendment debe contener por separado el hash corregido del
preparador y un commit de implementación (`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:95-101`),
pero la validación exigida para ese commit es únicamente
`merge-base --is-ancestor` o su equivalente
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:126-136`). El contrato de
fuentes sí exige que el preparador actual tenga el `new_sha256`, pero no exige
que `git show <implementation_commit>:experiments/geometria_proporcional/prepare_wave56_fresh.py`
tenga ese mismo hash. El informe ligado puede, por tanto, haber auditado el
commit declarado mientras `HEAD` ejecuta otro blob que el JSON se limita a
autodescribir.

**Impacto.** Se rompe precisamente la garantía que justifica incluir commit e
informe en el amendment: la ejecución puede separarse de la implementación que
recibió la auditoría independiente sin violar la mera ancestralidad. La
auditoría final prevista reduce el riesgo procedimental, pero el contrato
ejecutable seguiría aceptando una procedencia falsa o accidentalmente
desalineada.

**Corrección obligatoria.** El validador debe resolver el blob desde el objeto
Git declarado, hashearlo y exigir simultáneamente:

1. `sha256(git_blob(implementation_commit, prepare_wave56_fresh.py)) == new_prepare_sha256`;
2. `sha256(worktree_prepare) == new_prepare_sha256` y archivo idéntico a
   `HEAD`;
3. el informe de implementación ligado nombra ese mismo commit y ese mismo
   hash;
4. después del commit de implementación no cambia ninguna fuente ejecutable:
   los commits posteriores sólo pueden agregar el amendment canónico y los
   informes previstos, mientras el mapa `required_execution_sources` conserva
   como único delta el blob del preparador.

Costo de corrección: bajo. Efecto esperado: cerrar el bypass de procedencia sin
ampliar el delta científico.

### MEDIA — La secuencia de commits no distingue las dos auditorías y queda expuesta a autorreferencia

**Observación.** El JSON se publica después de una auditoría independiente y
liga el hash de ese informe (`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:95-110`),
pero la verificación final debe auditar también el propio JSON
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:198-215`). Un único
informe no puede a la vez preceder al JSON para ser incluido en él y auditar el
JSON ya materializado. El plan es ejecutable sin ciclo, pero sólo si se
distinguen explícitamente una auditoría de implementación y una auditoría final
del paquete.

**Corrección obligatoria.** Congelar esta DAG de commits, o una equivalente que
mantenga las mismas dependencias:

1. `P`: plan R330-corregido y congelado;
2. `I`: preparador corregido más tests, sin JSON final;
3. `A`: informe independiente de `I`, que fija commit y hashes;
4. `J`: únicamente el JSON canónico, que liga `P`, `I` y el hash de `A`;
5. `F`: informe independiente final que audita `I + A + J` y los tests;
6. ejecución de recovery y replay en un `HEAD` limpio cuyo único delta posterior
   a `I` sean `A`, `J` y `F`.

El informe `A` es el ligado por el JSON; `F` queda necesariamente fuera de él y
funciona como autorización procedimental final. El commit que agrega `F` no
puede tocar código, config ni amendment. Costo: sólo documentación del orden;
efecto: elimina interpretaciones circulares y hace auditable cada transición.

### MEDIA — El inventario fuente está fijado nominalmente, pero no como árbol físico cerrado

**Observación.** El plan exige basename, `FAILURE.json`, manifest, validadores e
inventario sin artefactos posteriores
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:121-147`). El checker
vigente, no obstante, valida miembros del manifest mediante `exists`, `stat` y
hash siguiendo el path, sin rechazar por sí mismo symlinks ni exigir archivos
regulares (`wave49_checker.py:99-109`). El escrow sí tiene control explícito de
owner y modo (`prepare_wave56_fresh.py:441-455`), mientras los demás miembros
del intento fuente no reciben en el plan un contrato físico equivalente.

**Impacto.** El intento actual es sano —la inspección recursiva encontró sólo
directorios y archivos regulares, todos `root:root`, con escrow/FAILURE en
`0600`, freeze público en `0644` y benchmark sellado en `0700/0600`—, pero una
sustitución posterior por symlink o un archivo agregado podría ser seguida por
los validadores genéricos o escapar de un blacklist incompleto. Es un riesgo de
integridad del recovery, no evidencia de contaminación presente.

**Corrección obligatoria.** Definir un whitelist recursivo exacto del intento,
validarlo con `lstat`, rechazar symlinks y tipos no regulares, comprobar owner y
modos esperados, fijar además el SHA-256 de `FAILURE.json`
(`710b7d29de8c0436304ffb7abdfb2adcd958ed443ab72a9190ce74495e8602af`)
y repetir inventario/hashes inmediatamente antes y después de consumir el
escrow y el benchmark. Los paths de ausencia deben ser una consecuencia del
whitelist, no sólo una lista negra. Costo: bajo; efecto: cierra sustitución y
TOCTOU accidental del archivo preservado.

### MEDIA — Los 192 tokens no canónicos se solapan con los 768 elegibles

**Observación.** El diagnóstico enumera `768` tokens elegibles, `384`
out-of-catalog y `192` de calibración no canónica
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:35-50`), pero no declara
que la tercera categoría no es disjunta. El generador reutiliza el mismo
`pair_token` de la realización base (`wave49_generator.py:223-229`) al agregar
la vista `origin_translation_break` de `PROP`
(`wave49_generator.py:317-386`). La recomputación agregada dio, en cada split,
`1152` tokens totales, `768` elegibles, `384` OOD, `192` no canónicos y una
intersección elegible/no-canónica de `192`.

**Impacto.** No altera el predicado correcto —el loader filtra filas antes de
agrupar (`wave50_neural.py:123-155`)—, pero una implementación que interprete
las tres cifras como partición puede construir un guard o un test inconsistente.

**Corrección obligatoria.** Declarar explícitamente el solapamiento y probar la
álgebra de categorías por split: `total=1152`, `eligible=768`, `OOD=384`,
`noncanonical=192`, `eligible ∩ noncanonical=192`, con elegibilidad calculada
sobre filas mediante el predicado congelado antes de deduplicar tokens. Costo:
mínimo; efecto: evita una segunda corrección del guard por ambigüedad de conteo.

## Controles que sí pasan

### Validez epistémica y no-redraw

- El contrato previo al draw fija `expected_eligible_pair_tokens_per_split=768`
  y `no_redraw_after_escrow=true`
  (`wave56_contextual_gate_fresh.json:124-134`).
- La población consumida downstream ya excluía OOD y no canónica antes del draw
  (`wave50_neural.py:123-155`). El amendment no inventa el predicado después de
  ver resultados.
- El intento abortó en el guard exacto documentado (`FAILURE.json:1-6`) y el
  código ubica ese guard antes de `generation_receipt`, inferencia y
  `preparation_freeze` (`prepare_wave56_fresh.py:833-889`).
- La invocación vigente ya impide un primary nuevo cuando existe cualquier
  archivo previo con escrow y obliga a `--recovery-secrets-from`
  (`prepare_wave56_fresh.py:251-300`). La restricción adicional al basename y
  hashes exactos del único intento mantiene esta propiedad.

La ausencia de archivos no es, por sí sola, una prueba forense de que ningún
operador root leyó truth sellada. No obstante, la legitimidad de esta corrección
no depende de esa afirmación fuerte: el delta queda determinado por un expected
count precongelado y por el predicado del consumidor preexistente, y no por
targets, logits, métricas o decisiones del draw.

### Integridad material observada

Los hashes publicados en el plan coinciden con el estado vigente: preparador
`7ff5919d...`, manifest `7582efe3...`, escrow `f86fb936...` y freeze público
`c65d581a...` (`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:53-69`). El
hash canónico compacto del contrato es `6fb76aef...`, coherente con la función
canónica del proyecto —claves ordenadas, separadores compactos y ASCII—
(`wave49_schema.py:133-134`). El commit de origen y los hashes de config y
preparador aparecen también en el freeze público
(`pre_generation_freeze.json:3-6`, `:349-375`).

El manifest fija `4992` filas visibles por split y los hashes/tamaños de todo el
benchmark (`manifest.json:19-81`). `validate_manifest`,
`validate_visible_package` y `validate_semantic_attestation` pasaron sobre el
intento preservado. La recomputación mecánica de conteos, sin exponer tokens ni
valores secretos, reprodujo las cinco cardinalidades indicadas arriba para
`train`, `val` y `lockbox`.

### Contrato dual y replay

La separación entre contrato de origen inmutable y contrato de ejecución
recuperada es necesaria y suficiente como arquitectura
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:89-119`). La comparación
exacta de todos los campos del contrato salvo `git_commit` y el único hash de
fuente autorizado, junto con igualdad byte-exact del manifest y preservación
del escrow/freeze originales, evita que la excepción se convierta en una
relajación general. El replay propuesto vuelve a ligar amendment, ambos
contratos, manifest y conteos
(`WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md:149-180`), y conserva la
matriz array-exact ya congelada por el plan original
(`WAVE_56_STAGE1_PROSPECTIVE_IMPLEMENTATION_PLAN.md:214-229`).

La reauditoría de implementación debe exigir de forma explícita que replay
compare también bytes/hash de `generation_escrow.json`,
`pre_generation_freeze.json` y la copia del amendment; que valide el manifest
contra sus miembros en ambos paquetes; y que no excluya diferencias científicas
bajo la categoría de receipts operativos.

## ¿Existe una solución materialmente más simple y segura?

No bajo la máquina vigente. Copiar el benchmark del intento fallido y reanudar
desde allí evitaría ejecutar nuevamente el generador, pero introduciría una ruta
de copia/promoción no probada, duplicaría un árbol sensible y requeriría los
mismos controles de contrato dual, inventario, fsync y replay. Regenerar con las
mismas claves reutiliza el camino transaccional existente, y la exigencia de
manifest byte-idéntico hace que cualquier deriva de generador, entorno,
attestation o datos falle antes de inferencia
(`prepare_wave56_fresh.py:796-877`). Tampoco conviene hardcodear una excepción
opaca en el script: sería más corta en líneas, pero perdería la procedencia
explícita y revisable del amendment.

La solución mínima segura es, por tanto, conservar el JSON canónico y el
contrato dual, pero cerrar el enlace commit→blob→auditoría, explicitar la DAG de
dos auditorías, endurecer el árbol fuente y aclarar el solapamiento de
poblaciones. No hace falta modificar config, generador, workers, modelos,
features, estimando, selector, thresholds ni claves.

## Condiciones concretas para PASS

1. Resolver el finding alto con verificación del blob Git del preparador en el
   commit de implementación y prohibición verificable de deltas ejecutables
   posteriores.
2. Documentar y seguir la DAG `P → I → A → J → F`, distinguiendo el informe
   ligado por el amendment de la auditoría final del JSON.
3. Convertir el inventario del intento en whitelist físico exacto, sin symlinks,
   con owner/modos/hashes y doble revalidación; incluir el hash de
   `FAILURE.json`.
4. Congelar en plan, schema y tests el solapamiento de los `192` tokens no
   canónicos con los tokens elegibles.
5. Probar negativos para: commit ancestro con blob distinto, amendment/report
   posterior al blob auditado, archivo extra o symlink en el intento, cambio de
   cualquiera de los hashes, amendment aplicado a primary ordinario u otro
   archivo, y replay con escrow/freeze/amendment divergentes.
6. Reauditar de manera independiente código, tests y JSON final; ejecutar sólo
   con `HEAD` limpio y suites verdes.

## Veredicto

**REVISE.** La corrección pre-oráculo y la reutilización de las tres claves son
epistémicamente válidas; no hay motivo para redibujar ni abandonar el draw. El
contrato dual es la arquitectura adecuada y el replay exacto es factible, pero
el plan todavía permite separar el blob ejecutado del commit/informe que dice
haberlo auditado. Cumplidas las seis condiciones anteriores, una reauditoría
focal podría emitir `PASS` técnico. Este dictamen no abre inferencia, oracle o
labels, no interpreta resultados y no constituye una decisión científica
`GO/NO-GO`.
