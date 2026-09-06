# R529 — Auditoría de implementación de la resolución atómica de publicación de Ola 60

## Dictamen técnico: PASS

R528 implementa el contrato autorizado por R527 sin reabrir la ciencia ni el
attempt sellado. La única API pública con capacidad de escritura vuelve a
derivar el payload desde las autoridades R528/R529; recorre el árbol físico con
descriptores `O_NOFOLLOW`; prepara y valida los bytes en un inode anónimo
`O_TMPFILE` con `nlink=0`; y crea el nombre canónico mediante una única llamada
`linkat(AT_EMPTY_PATH)`, cuya colisión preserva al competidor. Después de ese
punto no existe cleanup nominal: cualquier deriva o fallo de durabilidad se
expone como `PublicationCommittedUnverified` y conserva el residuo.

La focal de publicación pasó `13/13`, la suite completa R528 más Wave 60 pasó
`208/208`, y tres probes independientes reprodujeron la transición `nlink
0→1`, la ausencia de fallback cuando falla `AT_EMPTY_PATH` y el estado
committed-unverified ante desplazamiento del parent en la ventana de la syscall.
Encontré **0 HIGH, 0 MEDIUM y 0 LOW**. R528 puede autorizar la publicación R530
como candidato pendiente de R531; este dictamen no publica el artefacto, no
activa la vista corregida y no emite `GO/NO-GO`.

## Identidad y alcance auditados

El target es R528
`3cbe8f6f2fa90691b76caf5931cdfec6b1698ccc`, hijo directo único de R527
`b2852c4e90678a4c3f04a48b25df043ceb80a587`. Su diff modifica exactamente, con
status `M`:

- `experiments/geometria_proporcional/adjudicate_wave60_v4_result.py`, SHA-256
  físico/blob
  `7a0e53e9215adbeb00aabb1fb0505fcaec472971bda86d1141ff8f8de16bba4c`;
- `tests/test_wave60_v4_result_adjudication.py`, SHA-256 físico/blob
  `419aa0aa7eb5d29521e4e173c11eaa6fcd3ad7442ae09c192f26301948d7a0b9`.

Leí completos ambos archivos y R521–R527, incluidos los tres planes de
resolución y sus auditorías. Contrasté contra Git parents, pathsets, status,
blobs, hashes físicos y las autoridades JSON. La auditoría fue CPU-only con
`CUDA_VISIBLE_DEVICES=''`; no usó ni consultó GPU, no corrigió implementación,
no publicó el candidato y no modificó config, sources, datos ni attempt. El
runtime no expone una introspección independiente del identificador de modelo y
del esfuerzo, por lo que no puedo verificar desde dentro `gpt-5.6-sol/high`; sí
consta como requisito explícito del dispatch recibido.

## Autoridad y genealogía

- R509–R527 recomponen la cadena declarada. Los commits R521–R527 son hijos
  directos, cada uno con el único path y status prescrito.
- Los veredictos históricos quedan representados sin reetiquetado: R521
  `REVISE 0/1/0`, R523 `REVISE 0/2/0`, R525 `REVISE 0/1/0` y R527
  `PASS 0/0/0`. También permanecen R509/R511/R513/R517 como `REVISE` y las
  autoridades sucesoras reales.
- La validación dinámica exige que R528 sea hijo directo de R527 y que modifique
  sólo adjudicador y test; deriva sus dos hashes físicos y los iguala a los blobs
  de R528. R529 debe ser hijo directo de R528, añadir sólo este informe y
  contener una única autoridad `PASS 0/0/0` ligada al commit y a esos hashes.
- `publish_correction()` no acepta un payload ni un `Mapping`: recibe los tres
  bindings dinámicos y llama internamente a `build_correction_payload()`. El
  modo CLI `build` permanece no-write y `publish` no ofrece un bypass alternativo.

## Frontera física y punto de publicación

La implementación exige que repo, attempt y parent resuelvan estrictamente como
directorios físicos, que el parent permanezca dentro del repo y fuera del
attempt, y que sus paths resueltos coincidan con los canónicos. Después vuelve a
recorrer cada componente desde un descriptor del repo mediante
`O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`; no crea parents y rechaza symlinks,
componentes no directorio y output no canónico.

Sobre el parent anclado abre `.` con `O_TMPFILE|O_RDWR|O_CLOEXEC`, sin temporal
nombrado ni fallback. Un loop completa la escritura, `fchmod` fija `0444`,
`fsync` persiste el inode y la validación por el mismo descriptor exige archivo
regular, identidad estable, `nlink=0`, modo, tamaño, bytes y JSON exactos. Una
segunda travesía confirma identidad del parent y ausencia del leaf antes del
commit point.

La llamada libc `linkat(tmp_fd, "", parent_fd, leaf, AT_EMPTY_PATH)` es la única
creación de nombre. La semántica de hardlink es `NOREPLACE`: `EEXIST` se expone
como `FileExistsError` y deja intacto el competidor. No hay `rename`, overwrite,
`unlink`, `remove`, `replace`, búsqueda destructiva por inode ni fallback en el
publicador.

Tras un enlace exitoso se revalida por el descriptor anónimo `nlink=1`,
regularidad, inode, modo, tamaño, bytes y JSON; se hace `fsync` del parent; y una
tercera travesía desde un descriptor nuevo del repo exige la misma identidad de
parent y abre el leaf con `O_NOFOLLOW` para repetir todas las invariantes. Sólo
entonces retorna. Cualquier excepción posterior a `linkat` se encadena en
`PublicationCommittedUnverified`, incluyendo path, `created_identity` y la
declaración explícita de que no se intentó cleanup.

Un probe desplazó el parent exactamente dentro del wrapper de `linkat`, después
de la segunda travesía. La operación creó bytes canónicos en el parent ya
desplazado, detectó que el path canónico no conservaba parent/leaf, elevó
`PublicationCommittedUnverified` y dejó el inode regular `nlink=1` intacto. Es
el comportamiento previsto para interferencia posterior al punto linealizable,
no una publicación silenciosamente exitosa.

## Payload y ciencia preservada

El payload queda en `CANDIDATE_PENDING_R531_AUDIT`. Su condición de activación
exige R531, el scope `WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION`, `PASS 0/0/0` y
los sentinels exactos `DIRECT_PARENT_OF_R531`, `SELF_OUTPUT_RELATIVE` y
`SHA256_OF_THIS_ARTIFACT`. R530 por sí solo no activa la vista.

La recomposición conserva la observación histórica `MISMATCH 35/36` con el único
mismatch operacional de `preparation_receipt.json`, y la normalización
condicional `36/36`. Los patrones permanecen `incompatibility=false` y
`harm=false`; `scientific_decision=null`, `decision_authority=user` y
`architecture_promoted=false`. Las limitaciones científicas y los bindings
numéricos R509 permanecen intactos.

## Pruebas, ataques e inmutabilidad

Todas las invocaciones fijaron CUDA invisible, cuatro threads,
`PYTHONDONTWRITEBYTECODE=1`, plugins externos y cache de pytest deshabilitados y
basetemp propio bajo `/mnt/m2-1TB` con parent `0755`.

| Check | Resultado | wall | max RSS | swaps |
|---|---:|---:|---:|---:|
| focal de publicación R528 | 13 PASS, 21 deselected | 5,09 s | 841.028 KiB | 0 |
| suite completa R528 + Wave 60 | 208 PASS | 344,93 s pytest / 346,69 s proceso | 853.628 KiB | 0 |
| `check-attempt` real | PASS | 2,51 s | 769.016 KiB | 0 |
| probes independientes `O_TMPFILE/linkat/race` | 3 PASS | 2,56 s | 756.092 KiB | 0 |

Los probes confirmaron en el filesystem vigente: inode anónimo regular
`nlink=0`, transición a `nlink=1` tras `linkat`, colisión/no-soporte sin leaf ni
fallback, y residuo committed-unverified preservado. La suite cubrió además
autoridad sintética, inyección de payload, symlinks hacia attempt y fuera del
repo, parent ausente, leaf symlink/preexistente, competidor antes de `linkat`,
sustitución de parent, deriva post-link de modo/hardlink/bytes y fallo de
`fsync(parent_fd)`, con instrumentación que falla ante cleanup destructivo.

Antes y después, el attempt conservó `141` archivos regulares, `0` symlinks, la
huella agregada de metadata/inodos
`7dd459d97a1ae52709f522752e50c1b783cb7db4d5670681e7fef0f8cce9ce60` y
la huella agregada de contenido
`78fe6a059868ccda3b11719011369ff471b3b9305637eb4292a481fc63145937`.
Había aproximadamente 22 GiB de RAM disponible al cierre; el swap global tenía
uso histórico, mientras todos los procesos medidos registraron cero swaps.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R529",
  "scope": "R521_PUBLICATION_BOUNDARY_RESOLUTION_IMPLEMENTATION",
  "target": {
    "implementation_commit": "3cbe8f6f2fa90691b76caf5931cdfec6b1698ccc",
    "files": {
      "experiments/geometria_proporcional/adjudicate_wave60_v4_result.py": "7a0e53e9215adbeb00aabb1fb0505fcaec472971bda86d1141ff8f8de16bba4c",
      "tests/test_wave60_v4_result_adjudication.py": "419aa0aa7eb5d29521e4e173c11eaa6fcd3ad7442ae09c192f26301948d7a0b9"
    }
  },
  "technical_verdict": "PASS",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
