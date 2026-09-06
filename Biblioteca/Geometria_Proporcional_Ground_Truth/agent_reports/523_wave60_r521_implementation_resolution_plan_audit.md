# R523 — Auditoría del plan de resolución de la frontera de publicación de Ola 60

## Dictamen técnico: REVISE

R522 identifica correctamente los dos componentes de R521-01 y propone la
dirección adecuada: la única operación con capacidad de escritura volvería a
derivar el payload desde autoridades Git físicas, mientras que el destino se
abriría por descriptores de directorio con `O_NOFOLLOW` y el leaf con
`O_EXCL`. También conserva el attempt y la lectura científica, mantiene R521
como `REVISE 0/1/0` y posterga toda activación hasta una auditoría posterior.

El plan no cierra todavía la frontera fail-closed. El cleanup normativo puede
desvincular un archivo sustituto que la operación no creó, y la revalidación
final no vuelve a exigir todas las invariantes físicas del leaf. Además, R524
debe codificar autoridades R525/R527 futuras cuyo contrato exacto —paths,
targets JSON y topología Git completa— no quedó congelado. Encontré **0 HIGH,
2 MEDIUM y 0 LOW**. No corresponde implementar R524 hasta resolverlos.

## Identidad y alcance

Audité el plan R522 en commit
`cd223f697fb79a43455e3567d5316687ea7f22a3`, hijo directo único de R521
`c90df0391ef4e8a3eb2cba7a3d36ccf8a10916cd`. El commit añade exclusivamente
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R521_IMPLEMENTATION_RESOLUTION_PLAN.md`.
Su SHA-256 físico y de blob es
`837e8836a457cdc0dae71347a1bceb02bfaee0a338d02a41267b65812fb432f5`.

Leí completos el plan, el adjudicador R520, sus tests y el informe R521; además
contrasté parents, pathsets y blobs contra Git. La auditoría es CPU-only, no
consulta GPU, no publica el candidato y no modifica checker, tests, attempt,
config, sources ni datos.

## Findings

### R523-01 — MEDIUM — El commit físico de escritura y su cleanup no son identity-safe hasta el retorno

R522 conserva `(st_dev, st_ino)` tras crear el leaf y exige una reapertura final
con igual identidad, bytes y JSON (`WAVE_60_R521_IMPLEMENTATION_RESOLUTION_PLAN.md:89-96`).
Sin embargo, ante cualquier excepción posterior prescribe
`os.unlink(leaf, dir_fd=parent_fd)` sin comprobar que la entrada actual siga
apuntando al inode creado (`:97-98`). Si otro proceso renombra ese inode y pone
otro archivo bajo el mismo nombre entre la detección y el cleanup, el `unlink`
borra el sustituto, no el artefacto propio. La cobertura formula la condición
correcta —retirar el leaf sólo “si sigue ligado” al descriptor anclado
(`:140-141`)—, pero el algoritmo normativo no la implementa.

Una reproducción aislada bajo un temporal propio en `/mnt/m2-1TB` abrió el
leaf con `O_EXCL|O_NOFOLLOW`, guardó su inode, lo desplazó, instaló un sustituto
y ejecutó exactamente `os.unlink(leaf, dir_fd=parent_fd)`: desapareció el
sustituto y sobrevivió el inode originalmente creado bajo el nombre desplazado.
El temporal se retiró automáticamente y no tocó el repositorio ni el attempt.

La frontera final también queda incompleta: después de la escritura sólo se
enumeran identidad, bytes, JSON y cadena de paths (`:94-96`), pero no se vuelve
a exigir sobre el descriptor reabierto `S_ISREG`, `st_nlink == 1` y modo
`0444`. Esas propiedades se comprueban antes de escribir (`:91-92`) y pueden
cambiar antes del retorno. El test de estado final (`:134-135`) no sustituye un
check dentro de la operación ni cubre la ventana concurrente.

Resolución mínima requerida: antes de cualquier cleanup, hacer un lookup
no-follow relativo al mismo `parent_fd` y desvincular sólo si `(dev, ino)` aún
coincide con el leaf creado; si no coincide, preservar la entrada ajena y
reportar el fallo. En el commit point final, el `fstat` del descriptor reabierto
debe revalidar regularidad, inode, `nlink=1` y `0444`, además de bytes/JSON y la
cadena física. Debe haber negativos deterministas para sustitución antes del
cleanup y para deriva final de link-count/modo.

### R523-02 — MEDIUM — La autoridad sucesora que R524 debe materializar no tiene schema ni bindings completos congelados

R522 fija los números y scopes de R523–R527 (`:23-40`) y exige que el builder
autentique R522–R525 (`:60-69`). No fija, sin embargo, los paths normativos de
los informes R525 y R527, los keysets y valores exactos de sus `target`, ni la
forma exacta de la `activation_condition` que ligará R527 al commit, path y SHA
del artefacto R526. Para R525 sólo dice que la autoridad queda “ligada al commit
y a ambos hashes” (`:68-69`); para R527 sólo exige `PASS 0/0/0` sobre el
artefacto (`:38-40`, `:151-153`). Tampoco congela de forma inequívoca los
parents y name-status/pathsets de R526 y R527.

Esto es material porque R524 se compromete antes de que existan R525–R527 y
debe incorporar sus schemas esperados. Si el checker valida sólo
`audit_id/scope/verdict/findings`, un `PASS` no ligado al blob R526 podría
activar la vista. Si cada actor inventa luego el target o el path, la cadena no
es reproducible ni puede distinguir drift de una convención posterior. La
autoridad actual R523 sí queda ligada externamente por este encargo, pero eso no
resuelve los dos contratos futuros que el código R524 debe congelar.

Resolución mínima requerida: declarar para R523, R525 y R527 el path de informe,
schema completo, keyset exacto de `target` y bindings de commit/path/SHA/files;
declarar para cada commit R522–R527 parent directo, pathset exclusivo y
name-status esperado (`A` o `M`). La `activation_condition` del candidato debe
exigir una R527 cuyo target sea exactamente el commit R526 y el path+SHA del
candidato, no sólo una autoridad nominal con verdict correcto.

## Superficies que sí quedan resueltas

- `publish_correction` deja de aceptar un `Mapping` como capacidad y recibe
  únicamente bindings dinámicos; el payload se deriva dentro de la operación.
- R521 permanece como antecedente `REVISE 0/1/0`; R525 no lo reetiqueta.
- R524 queda acotado al adjudicador y su test, y R525 debe autenticar ambos
  hashes físicos contra blobs.
- La apertura relativa con directorios `O_DIRECTORY|O_NOFOLLOW`, leaf
  `O_EXCL|O_NOFOLLOW`, `fsync` de archivo y parent y ausencia de creación de
  parents cierran el bypass simple por ancestro symlink identificado en R521.
- Attempt, config, sources, métricas, `false/false`, limitaciones científicas,
  `scientific_decision=null`, `decision_authority=user` y
  `architecture_promoted=false` no se alteran. No hay decisión `GO/NO-GO`.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R523",
  "scope": "R521_PUBLICATION_BOUNDARY_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "cd223f697fb79a43455e3567d5316687ea7f22a3",
    "plan_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R521_IMPLEMENTATION_RESOLUTION_PLAN.md",
    "plan_sha256": "837e8836a457cdc0dae71347a1bceb02bfaee0a338d02a41267b65812fb432f5"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 2,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
