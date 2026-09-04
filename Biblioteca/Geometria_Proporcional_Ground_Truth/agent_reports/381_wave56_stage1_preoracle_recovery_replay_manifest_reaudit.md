# R381 — Reauditoría independiente del recovery pre-oráculo de Wave 56 Stage 1

**Implementation commit:** `c2781d5fe1980b11a4c8a96f8ed03d99df0c6672`
**Preparer SHA-256:** `adaacfe13740b477048adb24e9625b9fa19496672b8f007da429466016d40b83`
**Test SHA-256:** `25c323c99b3ef91dd8b7344856ea81e2608d5283cc859d5c35d037e25f42e37b`
**Result:** `PASS`

## Dictamen ejecutivo

El commit auditado cierra el blocker de replay identificado en R380. La
comparación de preparación valida ahora los manifests del replay y del primary
contra sus miembros físicos antes de cargar los manifests y antes de efectuar
cualquier comparación de exactitud. El negativo agregado altera un byte de un
miembro sellado del primary sin cambiar su tamaño y confirma que la divergencia
es rechazada por hash.

El cambio conserva el resto del contrato aprobado: el diff de `I` contiene
exclusivamente el preparador y su test focal; no cambia plan, config, datos,
generador, claves, modelos, criterios ni fuentes auxiliares. El nuevo path
canónico con sufijo `_v2` todavía no existe ni tiene un commit de introducción,
por lo que puede ser agregado una sola vez por el futuro hito `J`. El amendment
anterior sin ese sufijo y R380 permanecen como evidencia de la cadena fallida;
no autorizan recovery, replay, inferencia, oracle ni labels.

No encontré findings altos, medios o bajos que requieran revisar este `I`. La
suite focal completa terminó con 33 pruebas verdes en CPU y el adversarial
temporal de orden confirmó que los dos validadores de manifest se invocan antes
de que una inconsistencia del primary pueda alcanzar los checks de exactitud.

## Alcance y lecturas completas

Leí completos:

- `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_PREORACLE_RECOVERY_AMENDMENT_PLAN.md`;
- R330 y R331;
- R375, R376, R377, R378, R379 y R380;
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `tests/test_wave56_preoracle_recovery.py`.

También inspeccioné el diff exacto `parent..I`, los blobs Git de ambos archivos,
el parent único, la historia mínima de la cadena y la implementación de
`validate_manifest` necesaria para determinar qué significa validar contra los
miembros. No usé informes previos como sustituto del examen del código vigente.

La inspección no abrió valores de escrow, miembros sellados, truth, labels ni
oracle. No se usó web ni GPU, no se ejecutó recovery o replay oficial y no se
modificaron código, tests, configs, plan o datos.

## Identidad, diff y blobs

El objeto auditado tiene como parent único
`05a01d44067ebdf4faaa500f9a3173579ad9e805`. `git diff-tree` enumera exactamente
dos paths:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `tests/test_wave56_preoracle_recovery.py`.

El diff suma 11 inserciones y una eliminación: cambia el basename canónico del
amendment, agrega dos validaciones de manifest y añade el adversarial de
integridad. `git diff --check` no reportó errores. Los SHA-256 del worktree y de
los blobs resueltos directamente desde `I` coinciden byte a byte con los dos
valores congelados en la cabecera.

El parent agrega R380 como evidencia documental fallida. No se reinterpretó su
cabecera literal como autorización: su cuerpo reproduce un finding alto y
termina en `REVISE`. Del mismo modo, el JSON anterior permanece fuera de la
nueva cadena admisible.

## Cierre del finding de replay

`compare_preparation` comienza rechazando autorreferencia y luego ejecuta, en
este orden:

1. `validate_manifest(replay / "benchmark")`;
2. `validate_manifest(primary / "benchmark")`;
3. carga de ambos `manifest.json`;
4. comparación de compromisos, contenidos, arrays, freezes y receipts.

Las dos llamadas están en
`experiments/geometria_proporcional/prepare_wave56_fresh.py:1407-1408`. Por
tanto, ninguna igualdad entre los JSON de manifest ni ningún `all_exact` puede
ocultar que uno de sus árboles físicos dejó de corresponder a sus miembros.

El validador subyacente recorre cada entrada de `manifest["files"]` y exige
existencia, tamaño y SHA-256 del archivo. El nuevo negativo, en
`tests/test_wave56_preoracle_recovery.py:687-692`, toma
`primary/benchmark/sealed/train.jsonl`, invierte un bit del primer byte y
reescribe exactamente la misma cantidad de bytes. La comprobación de tamaño
sigue pasando por construcción; la de hash levanta `ProtocolViolation`. Esto
cubre de manera material el caso exacto que R380 había demostrado fail-open.

La focal también conserva el replay físico sintético exitoso antes del tamper:
recovery reutiliza las mismas claves, el monkeypatch de `secrets.token_bytes`
hace fallar cualquier redraw, el replay llega a exactitud completa y recién
después se prueba la corrupción del primary. Así, el control negativo no
reemplaza al positivo ni debilita la matriz previa.

## Path canónico y DAG

La constante vigente apunta a
`experiments/geometria_proporcional/configs/wave56_stage1_preoracle_recovery_amendment_v2.json`.
Ese path:

- no existe en el parent;
- no existe en `I`;
- no existe en el worktree auditado;
- registra cero commits de introducción en la historia alcanzable desde `I`.

El path viejo sí existe, pero no coincide con la constante y no puede atravesar
la igualdad canónica de `validate_recovery_amendment`. Cuando el archivo v2 sea
creado, `git_introduction_commit` exigirá exactamente un commit de introducción
y el validador requerirá que ese commit cambie exclusivamente ese path. Esto
permite reiniciar sin ambigüedad la secuencia:

`P → I actual → A nuevo → J v2 → F nuevo`.

El informe presente corresponde a `A` una vez que sea introducido por un commit
documental exclusivo. Todavía no existen `J` ni `F` de la nueva cadena. El
estado actual, por sí solo, no autoriza ejecución.

## Contrato preservado

La revisión estática y la focal confirman que el parche no altera estas
propiedades:

1. El plan canónico queda congelado antes de `I` y el binding de implementación
   resuelve preparador y test desde el objeto Git declarado.
2. `I` debe tener un solo parent y cambiar exactamente esos dos archivos.
3. `A` y `F` deben ser Markdown distintos bajo el directorio canónico de
   informes, con cabeceras visibles, únicas, contiguas y de texto canónico.
4. Los bordes posteriores requieren parentesco directo; `F` debe coincidir
   exactamente con `HEAD`, y el worktree debe estar globalmente limpio.
5. El contrato dual admite un único delta ejecutable respecto del origen: el
   preparador aprobado. No cambian config, bindings ni las demás fuentes.
6. El amendment no puede autorizar un primary fresco. Recovery y replay sólo
   obtienen las tres claves desde el escrow durable; el camino amended no puede
   recurrir a `secrets.token_bytes`.
7. El origen preservado se ata mediante whitelist físico cerrado y hashes y se
   revalida antes de extraer claves y después de regenerar, todavía antes de
   inferencia.
8. Los conteos total y elegible permanecen separados; la elegibilidad se
   filtra por fila antes de deduplicar tokens.
9. Escrow, freeze pre-generación, amendment, manifest, visibles, logits,
   freeze de preparación y procedencia permanecen dentro de la matriz de
   replay.
10. La CLI valida amendment, procedencia Git, informes, limpieza y escrow antes
    de crear o archivar el output. Una falla posterior dentro de la transacción
    archiva el estado y no publica un receipt de exactitud exitoso.

El nuevo cambio se limita a fortalecer el punto 9: ya no se confía en el JSON
del primary sin confrontarlo con cada miembro que gobierna.

## Pruebas ejecutadas

Todas las pruebas se corrieron con GPU invisible y un hilo para OpenMP,
OpenBLAS, MKL y NumExpr; se deshabilitaron bytecode y cache de pytest.

- Focal completa:
  `venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wave56_preoracle_recovery.py`
  — `33 passed in 15.27s`.
- Adversarial permanente same-size del primary — incluido en la focal y
  rechazado con `ProtocolViolation: hash mismatch`.
- Probe temporal de orden — sustituyó el validador por una sonda que falla en
  el primary y confirmó la secuencia `replay`, `primary`; el fallo ocurrió antes
  de cualquier lectura/comparación posterior.
- Verificación de path — ausencia del JSON v2 en parent, `I` y worktree, y cero
  introducciones previas.
- Git — parent exacto, dos paths en el diff, hashes worktree/blob y
  `git diff --check`.

No corrí la suite amplia: R380 aisló el defecto en esta superficie, el parche y
el negativo son focales, las 33 pruebas recorren recovery/replay físico
sintético y no apareció un finding que justificara ampliar el alcance.

## Findings y límites

No hay findings materiales abiertos en el alcance auditado.

La aprobación es exclusivamente técnica y pre-oráculo. No valida valores
oficiales sellados, no ejecuta la recuperación real, no interpreta resultados y
no constituye una decisión científica `GO/NO-GO`. La cadena deberá continuar
sin saltos: commit documental exclusivo de este informe, introducción única del
JSON v2 y auditoría final nueva antes de cualquier ejecución oficial.

## Decisión

`PASS`. El commit corrige la certificación de replay contra un primary
físicamente inconsistente, conserva el no-redraw y el fail-closed, y deja libre
el path canónico v2 para un único hito `J`. Puede avanzarse al siguiente paso de
la DAG; este dictamen no autoriza todavía recovery, inferencia, oracle ni
labels.
