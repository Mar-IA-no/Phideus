# R382 — Auditoría final independiente del paquete v2 de recovery pre-oráculo de Wave 56 Stage 1

**Audited package commit:** `abf7a3c7ccf320682b57b2c7fa3f8409a93246c0`
**Amendment SHA-256:** `0827c38559a840a33d61e2cdb86250ef57f8fb22c2131e3826cc1cb56c45356a`
**Result:** `PASS`

## Dictamen ejecutivo

El paquete v2 cierra el blocker de replay hallado en R380 y conserva sin
regresiones materiales el contrato completo de recuperación pre-oráculo. La
cadena real liga el implementation commit `I`, la auditoría aprobatoria `A` y
el amendment `J` mediante parents directos; cada hito modifica exclusivamente
los paths permitidos y todos los blobs coinciden con los hashes declarados. El
plan `P` permanece congelado byte a byte desde su commit documental y es
ancestro de `I`.

El amendment v2 es JSON pretty canónico, tiene el SHA-256 de la cabecera y fue
introducido una sola vez por `J`. Su inventario físico coincide con las 24
entradas del intento preservado, el contrato público de origen conserva su hash
y commit, y el mapa de 26 fuentes presenta un único delta: el preparador, entre
los hashes old/new autorizados. No existe primary ni replay oficial y el árbol
fallido no contiene inferencia, oracle materializado, labels, bundles, fases ni
freezes o receipts posteriores.

La comparación de replay valida ahora los manifests del replay y del primary
contra sus miembros antes de cargar los manifests o comparar exactitud. El
negativo físico sintético altera un miembro sellado del primary sin modificar
su tamaño y es rechazado por hash. La suite focal terminó `33 passed` y la
suite amplia exacta Wave 49–56 terminó `185 passed`, ambas en CPU-only, sin
fallos ni skips.

No encontré un bypass material abierto en procedencia, no-redraw, fail-closed,
inventario o replay. Este informe puede constituir `F` cuando sea agregado por
un commit documental exclusivo, directamente hijo de `J`. Hasta entonces el
recovery oficial permanece deliberadamente inejecutable: el validador exige
que el commit de introducción de este informe sea el `HEAD` exacto y que el
worktree esté globalmente limpio.

## Alcance y restricciones

La auditoría se ejecutó con:

```text
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1
```

No usé GPU, web, servicios externos ni subagentes. No ejecuté recovery o replay
oficial, no materialicé oracle o labels y no abrí ni expuse valores del escrow,
claves, truth, miembros sellados o resultados oficiales. Sobre el origen
preservado limité la inspección a nombres, tipos, owner, modos, tamaños y hashes;
los validadores de contenido ejecutados allí fueron únicamente manifest y
paquete visible.

Leí completos el plan de amendment, R330, R331 y R375–R381, además del
preparador vigente, el test focal y el amendment v2. La revisión estática se
hizo sobre los archivos actuales y sobre los blobs Git de los commits
declarados, no por confianza en los informes anteriores.

## Cadena `P → I → A → J`

El ancla documental del plan es:

- `P = 8062c83ccea1fe6ce8c087f6e1c9bb1ff4ea30c2`;
- SHA-256 del plan:
  `d4f306ffb8d34a9d3499b99e18434d57085438fd4cd0202f4f8658d3bdbe923d`.

`P` es ancestro de la implementación vigente. Entre `P` e `I` quedan
preservados los ciclos fallidos de implementación y auditoría anteriores; el
plan no exige parent directo en ese borde. El blob del plan mantiene el mismo
hash en el parent inmediato de `I`, en `I`, en `A`, en `J` y en el worktree.

La cadena ejecutable posterior queda fijada así:

- `I = c2781d5fe1980b11a4c8a96f8ed03d99df0c6672`, con parent único
  `05a01d44067ebdf4faaa500f9a3173579ad9e805`;
- `A = cd899d6ab22df6cffe578f2ea333bd345ba4c547`, con parent directo exacto
  `I`;
- `J = abf7a3c7ccf320682b57b2c7fa3f8409a93246c0`, con parent directo exacto
  `A`.

Los diffs por commit son exclusivos:

- `I` modifica solamente
  `experiments/geometria_proporcional/prepare_wave56_fresh.py` y
  `tests/test_wave56_preoracle_recovery.py`;
- `A` agrega solamente
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/381_wave56_stage1_preoracle_recovery_replay_manifest_reaudit.md`;
- `J` agrega solamente
  `experiments/geometria_proporcional/configs/wave56_stage1_preoracle_recovery_amendment_v2.json`.

Los tres diffs pasan `git diff --check`. El conjunto acumulado `I..J` contiene
exactamente R381 y el amendment v2; no hay cambios post-implementación en
código, test, plan, config prospectiva u otra fuente ejecutable.

Las introducciones Git son únicas: R381 fue introducido por `A`, el amendment
v2 por `J`, y el path de R382 no tenía ninguna introducción previa al crear
este informe. Esto deja disponible el único borde restante `F ← J`.

## Blobs, amendment y cabeceras canónicas

Los hashes del worktree y de los blobs resueltos directamente desde sus commits
coinciden:

- preparador en `I`:
  `adaacfe13740b477048adb24e9625b9fa19496672b8f007da429466016d40b83`;
- test focal en `I`:
  `25c323c99b3ef91dd8b7344856ea81e2608d5283cc859d5c35d037e25f42e37b`;
- R381 en `A`:
  `0ac13789ddcf56b11ae27cf23f3fe6aba604bf50836460b9d8cbbdcdb750e2cc`;
- amendment v2 en `J`:
  `0827c38559a840a33d61e2cdb86250ef57f8fb22c2131e3826cc1cb56c45356a`.

El preparador en el commit de origen del escrow conserva el hash old
`7ff5919d2b0bdd607ca179180c4f94de3ff5be6e23e6024b21e748d22c61fb44`.
El amendment declara exactamente ese old hash, el new hash de `I` y el hash del
test de `I`.

El JSON v2 usa UTF-8, LF terminal, orden lexicográfico, indentación de dos
espacios y serialización ASCII; reserializarlo con la función canónica pretty
produce bytes idénticos. Su schema, estado, top-level keys, assertions, paths de
plan, implementación, R381 y R382, contrato de población e inventario coinciden
con lo exigido por el preparador.

R381 contiene la cabecera visible canónica en posiciones fijas: H1, blanco,
commit `I`, hashes de preparador y test, resultado positivo, blanco y H2. Usa
UTF-8/LF canónico, no tiene separadores alternativos, fence o comentario HTML
en la cabecera, y su cuerpo y decisión concuerdan con la aprobación. La misma
gramática queda satisfecha por la cabecera de R382.

## Contrato dual y origen preservado

La verificación del contrato se efectuó desde el freeze público, sin abrir el
escrow. El hash compacto del contrato coincide con el amendment, su commit de
origen es `51aae0715dfe8318f5333c568429c8e9af59f866` y la config prospectiva
actual conserva el hash congelado. El conjunto de
`required_execution_sources` tiene 26 paths y coincide exactamente con el mapa
de fuentes del contrato.

Al recalcular los hashes de esas 26 fuentes en `J`, el único cambio respecto
del contrato de origen es
`experiments/geometria_proporcional/prepare_wave56_fresh.py`. Sus hashes old y
new coinciden con el amendment; no hay un segundo delta oculto bajo receipts,
auditorías o archivos auxiliares.

El inventario observado del intento fallido coincide entrada por entrada con
el whitelist del amendment:

- 24 entradas totales: 6 directorios y 18 archivos regulares;
- owner `root:root` en todas las entradas;
- directorios `0700`, archivos `0600` y freeze público `0644`;
- cero symlinks o tipos especiales;
- tamaños y los 18 SHA-256 coincidentes.

`validate_manifest` pasó contra sus 14 miembros declarados y
`validate_visible_package` pasó con 4992 filas en cada uno de `train`, `val` y
`lockbox`. No ejecuté `validate_semantic_attestation` sobre el origen oficial
porque esa rutina abre material sellado y excede la frontera de esta auditoría.

El whitelist demuestra la ausencia de `inference/`, `authorized_labels/`,
`bundles/`, `phases/`, oracle materializado, `generation_receipt.json`,
`preparation_freeze.json` y `preparation_receipt.json`. Los paths canónicos de
primary y replay tampoco existen.

## No-redraw y orden fail-closed

La corrección científica continúa limitada al predicado del guard. En modo
amended, el conteo usa tokens elegibles por fila antes de deduplicar; en modo
primary ordinario se conserva el comportamiento contractual original. El
amendment mantiene por split 4992 filas, 1152 tokens totales, 768 elegibles,
384 out-of-catalog, 192 no canónicos y una intersección elegible/no-canónica de
192. La suite cubre el solapamiento y los drifts de ambas poblaciones.

Recovery y replay con escrow reutilizado rechazan `keys_override` y obtienen
las tres claves exclusivamente del escrow durable. Sólo la rama primary fresca
puede alcanzar `secrets.token_bytes`; el test físico sustituye esa función por
una excepción y completa recovery/replay sin invocarla. El amendment no puede
autorizar un primary fresco ni otro basename de intento.

La CLI valida primero config, fuentes y preflight; luego amendment, DAG,
reportes, blobs, `HEAD`, limpieza global, origen físico y escrow reutilizado;
sólo después entra en la transacción que crea o archiva output. El origen se
revalida antes de extraer claves y nuevamente después de regenerar el
benchmark, todavía antes de inferencia. Escrow y freeze republicados deben ser
byte-idénticos al origen, y el manifest regenerado debe conservar el hash
fijado. Cualquier excepción dentro de la transacción produce un estado fallido
archivado y no un receipt exitoso.

## Manifest del primary y del replay

`compare_preparation` rechaza autorreferencia y, como primeras operaciones
sustantivas, llama a `validate_manifest` sobre:

1. `replay / "benchmark"`;
2. `primary / "benchmark"`.

Sólo después carga los dos manifests y compara commitments, manifest y
protocolo, visibles, escrow, freeze pre-generación, amendment, logits,
`preparation_freeze` y los campos relevantes del recibo de generación. Una
divergencia aborta antes de publicar `preparation_replay.json` o un receipt de
exactitud exitoso; el handler archiva el output fallido.

El adversarial permanente altera un bit del primer byte de un miembro del
primary y conserva exactamente el tamaño. La comprobación de tamaño no basta,
pero la validación del SHA-256 levanta `ProtocolViolation`. Así queda cubierto
el bypass exacto reproducido por R380 sin depender de una mutación meramente
nominal del manifest.

## Pruebas ejecutadas

La suite focal se ejecutó como:

```text
venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_wave56_preoracle_recovery.py
```

Resultado: `33 passed in 15.59s`.

La suite amplia exacta de R380 se reconstruyó como los once archivos Wave
49–56 que ya sumaban 152 pruebas más la focal de recovery que agrega 33:

```text
tests/test_wave49_relational_benchmark.py
tests/test_wave50_neural.py
tests/test_wave50_protocol.py
tests/test_wave50_runner.py
tests/test_wave51_factored.py
tests/test_wave52_policy.py
tests/test_wave53_uncertainty.py
tests/test_wave54_joint_set.py
tests/test_wave55_policy_bridge.py
tests/test_wave56_contextual_gate.py
tests/test_wave56_prospective.py
tests/test_wave56_preoracle_recovery.py
```

Resultado: `185 passed in 231.45s (0:03:51)`. No hubo fallos ni skips; los
tests físicos no quedaron omitidos en este host.

## Estado de `HEAD` y condición de ejecución

Antes de crear R382, `HEAD` era exactamente
`abf7a3c7ccf320682b57b2c7fa3f8409a93246c0` y el worktree estaba globalmente
limpio. Las suites y todas las verificaciones del paquete se ejecutaron en ese
estado. La creación solicitada de este informe deja, por diseño, un único path
documental nuevo sin commit; no se modificó ningún archivo preexistente.

`/Biblioteca/` está cubierto por `.gitignore`, de modo que el nuevo R382 existe
pero todavía no aparece en `git status` ni en `git ls-files`. Esa apariencia de
limpieza no satisface `F`: `require_repo_artifact` exige tracking e identidad
con `HEAD`. El commit final deberá agregar explícitamente —con force-add por el
ignore— sólo este path y luego comprobar su diff y la limpieza global.

La ejecución oficial exige todavía que R382 sea introducido por un commit que:

1. tenga como parent directo exacto `J`;
2. agregue exclusivamente el path de este informe;
3. sea el `HEAD` exacto;
4. deje el worktree globalmente limpio;
5. conserve byte-idénticos preparador, test, R381 y amendment.

El validador comprueba esas condiciones antes de crear o archivar output. Por
eso este `PASS` no vuelve ejecutable el árbol sin su commit `F`; deja preparada
la evidencia que ese commit debe fijar.

## Findings y límites

No hay findings altos, medios o bajos que requieran revisar el paquete v2. La
búsqueda adversarial no encontró una vía real para sustituir el blob auditado,
insertar commits intermedios, usar reports fuera de la frontera documental,
ocultar un resultado no aprobatorio en la cabecera, reutilizar el amendment en
un primary nuevo, redibujar claves, aceptar un origen físico divergente o
certificar replay contra un primary cuyo miembro no corresponde al manifest.

La auditoría no valida el contenido semántico secreto ni los resultados
oficiales, porque todavía no deben abrirse. Los recorridos completos de
recovery y replay usados como evidencia son fixtures físicos sintéticos; el
origen oficial se verificó sólo mediante metadata, hashes y validadores
públicos seguros. Este dictamen es técnico y pre-oráculo: no constituye una
decisión científica `GO/NO-GO`.

## Decisión

`PASS`. El paquete `I+A+J` queda técnicamente aprobado para completar el hito
documental `F` mediante un commit exclusivo de R382 y, sólo después de cumplir
`HEAD == F` con worktree limpio, habilitar procedimentalmente el recovery
pre-oráculo oficial. No se autoriza un redraw, no se ejecutó recovery o replay
oficial y no se abrieron inferencia, oracle, labels ni resultados.
