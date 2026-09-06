# R527 — Auditoría del plan de resolución atómica de publicación de Ola 60

## Dictamen técnico: PASS

R526 cierra el finding R525-01 sin introducir otra ventana destructiva. La
preparación ocurre sobre un inode anónimo creado con `O_TMPFILE`; antes del
enlace, todo fallo se resuelve cerrando el descriptor y no existe un nombre que
retirar. La única creación del nombre canónico es
`linkat(AT_EMPTY_PATH)`, cuyo éxito constituye el punto de linealización y cuya
semántica `NOREPLACE` deja intacto un competidor preexistente. Después de ese
punto, el contrato prohíbe toda retirada, rename o búsqueda por inode: una falla
de durabilidad o revalidación devuelve `PublicationCommittedUnverified` y
preserva el residuo para inspección explícita.

La autoridad vuelve a derivarse dentro de la única API con capacidad de
escritura, la topología R526–R531 y los contratos R527/R529/R531 quedan fijados
sin ambigüedad, y los sentinels de R530 son relaciones externas resolubles sin
autorreferencia. Encontré **0 HIGH, 0 MEDIUM y 0 LOW**. El plan puede pasar a
R528 bajo la precondición ya declarada de esta auditoría `PASS 0/0/0`.

## Identidad y alcance

Audité el plan R526 en commit
`cda7f23cec9f7d99b8eae90b5f10c2f6733a3c73`, hijo directo único de R525
`4b8c14e4f35a1664feed9c5614dbb003fb269eab`. El commit añade exclusivamente
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R525_ATOMIC_PUBLICATION_RESOLUTION_PLAN.md`,
con SHA-256 físico y de blob
`bc633e888530df74d7ad5bc467ea38fe79c7478c452f634d491ac6410c28bcea`.

Leí completos R521–R526, el adjudicador R520 y su test, y contrasté la cadena
contra Git. Esta auditoría es CPU-only: mantuvo `CUDA_VISIBLE_DEVICES=''`, no
usó ni consultó GPU, no publicó el candidato, no modificó checker, tests,
config, sources, datos ni attempt y no emitió `GO/NO-GO`. El runtime no ofrece
una introspección verificable del identificador de modelo ni del nivel de
esfuerzo; por ello no puedo confirmar de forma independiente
`gpt-5.6-sol/high`, aunque el dispatch recibido lo exige.

## Cierre del finding R525-01

### Ausencia de cleanup nominal

- Antes de `linkat`, el inode tiene `nlink=0`. Escritura, `fchmod`, `fsync`,
  validación de tipo/modo/tamaño/bytes/JSON y cualquier fallo de soporte ocurren
  sin leaf; cerrar el descriptor es suficiente y no ejecuta cleanup por nombre.
- No hay fallback a `mkstemp`, temporal nombrado, `rename` ni otra ruta de
  publicación. La ausencia de `O_TMPFILE` o `AT_EMPTY_PATH` falla antes de
  materializar el leaf.
- Después de un `linkat` exitoso, el plan prohíbe expresamente `unlink`,
  `remove`, `rename`, `replace` y búsquedas destructivas por inode, incluso si
  fallan `fsync(parent_fd)` o las comprobaciones posteriores. Esto elimina tanto
  la carrera `stat→unlink` señalada por R525 como cualquier cleanup nominal
  post-commit.

### Punto linealizable `NOREPLACE`

La llamada única `linkat(tmp_fd, "", parent_fd, leaf, AT_EMPTY_PATH)` crea el
enlace dentro del directorio ya anclado. Si el nombre existe, devuelve `EEXIST`
sin reemplazarlo; no necesita confiar en el lookup previo de ausencia. Por
tanto, el resultado concurrente queda determinado en esa syscall: o el
competidor ya ganó y permanece intacto, o se instala el inode anónimo validado.
No existe un estado visible con bytes parciales.

Una reproducción propia y aislada bajo un `TemporaryDirectory` en
`/mnt/m2-1TB` confirmó en el filesystem vigente:

- primer `linkat`: retorno `0`;
- transición `st_nlink: 0→1` e identidad `(st_dev,st_ino)` conservada;
- bytes exactos y modo final `0444`;
- segundo `linkat` al mismo leaf: retorno `-1`, `errno=EEXIST`;
- el temporal se retiró automáticamente al finalizar el probe.

El probe sólo confirma soporte y semántica observada en este host; el plan no
los extrapola a otros filesystems y conserva el fallo cerrado cuando falten.

### Contención y revalidación física

La resolución conserva la frontera construida por R522–R524 y la vuelve a
aplicar alrededor del nuevo commit point: repo, attempt y parent deben ser
directorios físicos resueltos estrictamente; el parent queda físicamente dentro
del repo y fuera del attempt; cada componente se abre con
`O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`; no se aceptan `..`, symlinks ni creación de
parents. La identidad del parent abierto se conserva y una segunda travesía la
confirma inmediatamente antes de enlazar.

Tras el enlace, `fstat(tmp_fd)`, `fsync(parent_fd)` y una tercera travesía desde
un descriptor nuevo vuelven a exigir identidad de parent y leaf, regularidad,
`nlink=1`, modo `0444`, tamaño, bytes canónicos y JSON exacto. Una sustitución o
desplazamiento detectado después del punto linealizable no autoriza borrar nada:
el error conserva path e identidad creada y deja visible que existe un estado
committed pero no verificado.

## Autoridad y contratos sucesores

- R526 es hijo directo de R525 y sólo añade el plan; R527 sólo añade este
  informe; R528 modifica exactamente adjudicador y test; R529 sólo añade su
  auditoría; R530 sólo añade el candidato; R531 sólo añade su auditoría. Cada
  target queda en un commit anterior y separado de su auditor.
- La API pública de escritura recibe únicamente los tres bindings de
  implementación/auditoría y llama internamente a
  `build_correction_payload(...)`. Un `Mapping` o JSON schema-valid no concede
  capacidad de publicación; `build` sólo escribe bytes a stdout.
- El builder debe reautenticar la cadena R509–R527 y los bindings R528/R529
  mediante parents directos, pathsets/status exactos, blobs, SHA físicos y la
  autoridad JSON exacta. R521/R523/R525 permanecen con sus veredictos históricos
  `REVISE`; ninguna auditoría sucesora los reetiqueta.
- R527, R529 y R531 fijan path, scope, keyset top-level cerrado, target exacto y
  condiciones autorizantes `PASS 0/0/0`, `files_modified=false` y
  `gpu_used_or_queried=false`. R528 y R530 sólo pueden existir después de las
  auditorías precedentes autorizantes.
- Los tres sentinels de R530 no contienen el SHA ni el commit futuros como
  literales: R531 los resuelve a su parent directo R530, al path canónico del
  candidato y al SHA físico/blob ya determinado de R530. Así la condición liga
  exactamente el artefacto auditado sin un fixed point circular; R530 solo no
  activa la vista.

## Cobertura y semántica preservada

La cobertura obligatoria es proporcional a los riesgos: autoridad sintética e
inyección de payload; soporte ausente sin fallback; éxito físico; symlinks hacia
attempt o fuera del repo; parent/leaf/path inválidos; competidor en el instante
previo a `linkat`; sustitución de parent antes del enlace; y deriva post-link de
modo, hardlink, bytes, JSON o `fsync`. Los casos post-link exigen el error
específico, preservación del residuo e instrumentación que falla ante cualquier
primitiva destructiva. Las huellas pre/post del attempt cubren la no invasión de
la evidencia canónica. Esta matriz prueba los dos lados del punto linealizable y
elimina el falso negativo que tenía la cobertura anterior alrededor de
`stat→unlink`.

El plan no altera `MISMATCH 35/36`, el único mismatch operacional, la vista
condicional normalizada `36/36`, `replay_exact=true` sólo condicional, los
patrones `false/false`, métricas, intervalos, soportes ni limitaciones. Conserva
`scientific_decision=null`, `decision_authority=user`,
`architecture_promoted=false` y ausencia de GPU. `check-attempt` terminó con
exit `0`; el intento y la ciencia permanecen fuera del diff.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R527",
  "scope": "R525_ATOMIC_PUBLICATION_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "cda7f23cec9f7d99b8eae90b5f10c2f6733a3c73",
    "plan_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R525_ATOMIC_PUBLICATION_RESOLUTION_PLAN.md",
    "plan_sha256": "bc633e888530df74d7ad5bc467ea38fe79c7478c452f634d491ac6410c28bcea"
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
