# Ola 60 — resolución final de los findings R523 sobre publicación

> **Estado:** `PROPOSED / PRE-R525-AUDIT / CPU-ONLY / NO-PUBLICATION`
>
> **Cadena precedente:** R522 `cd223f697fb79a43455e3567d5316687ea7f22a3`;
> R523 `5f371c8e1d9369a03a1c168a3626018529276348`, `REVISE 0/2/0`.

## 1. Decisión de diseño

R522 sigue siendo la base del arreglo: el publicador deriva el payload dentro
de la operación con capacidad de escritura y recorre el destino mediante
descriptores `O_NOFOLLOW`. R523 detectó dos huecos. Esta resolución los cierra
sin reabrir ciencia, config, sources ni el attempt:

1. cleanup y commit point pasan a estar ligados a identidad física hasta el
   retorno;
2. toda la cadena sucesora queda congelada con paths, parents, name-status,
   schemas y targets exactos.

## 2. Topología Git normativa

Cada commit es hijo directo único del anterior y tiene el pathset completo que
se indica; cualquier path extra, status distinto o parent distinto invalida la
autoridad.

| ID | Parent | Pathset y status |
|---|---|---|
| R524 | R523 `5f371c8e1d9369a03a1c168a3626018529276348` | `A Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R523_PLAN_FINDINGS_RESOLUTION.md` |
| R525 | R524 | `A Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/525_wave60_r523_plan_findings_resolution_audit.md` |
| R526 | R525 | `M experiments/geometria_proporcional/adjudicate_wave60_v4_result.py`; `M tests/test_wave60_v4_result_adjudication.py` |
| R527 | R526 | `A Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/527_wave60_r523_resolution_implementation_audit.md` |
| R528 | R527 | `A Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json` |
| R529 | R528 | `A Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/529_wave60_v4_replay_normalization_correction_audit.md` |

R526 no puede existir antes de R525 `PASS 0/0/0`; R528 no puede existir antes
de R527 `PASS 0/0/0`; R529 no puede compartir commit con el artefacto que
audita.

## 3. Autoridades JSON exactas

Todas las auditorías contienen exactamente un bloque JSON y el keyset
top-level común:

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "...",
  "scope": "...",
  "target": {},
  "technical_verdict": "PASS",
  "findings": {"high": 0, "medium": 0, "low": 0},
  "files_modified": false,
  "gpu_used_or_queried": false
}
```

No se admiten claves adicionales. Los targets normativos son:

### R525 — auditoría de este plan

- path:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/525_wave60_r523_plan_findings_resolution_audit.md`;
- scope: `R523_PUBLICATION_BOUNDARY_PLAN_FINDINGS_RESOLUTION`;
- target exacto:

```json
{
  "plan_commit": "<R524>",
  "plan_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R523_PLAN_FINDINGS_RESOLUTION.md",
  "plan_sha256": "<SHA-256 físico/blob de ese path en R524>"
}
```

### R527 — auditoría de implementación

- path:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/527_wave60_r523_resolution_implementation_audit.md`;
- scope: `R521_PUBLICATION_BOUNDARY_RESOLUTION_IMPLEMENTATION`;
- target exacto:

```json
{
  "implementation_commit": "<R526>",
  "files": {
    "experiments/geometria_proporcional/adjudicate_wave60_v4_result.py": "<SHA-256 físico/blob>",
    "tests/test_wave60_v4_result_adjudication.py": "<SHA-256 físico/blob>"
  }
}
```

### R529 — auditoría del candidato

- path:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/529_wave60_v4_replay_normalization_correction_audit.md`;
- scope: `WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION`;
- target exacto:

```json
{
  "artifact_commit": "<R528>",
  "artifact_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json",
  "artifact_sha256": "<SHA-256 físico/blob del candidato en R528>"
}
```

R525/R527/R529 sólo autorizan con `technical_verdict=PASS`, findings
`0/0/0`, `files_modified=false` y `gpu_used_or_queried=false`.

## 4. Condición de activación sin autorreferencia imposible

El candidato R528 no puede contener literalmente su propio SHA o el hash del
commit que lo contendrá sin crear una dependencia circular. Congela en cambio
el predicado externo exacto que R529 debe satisfacer:

```json
{
  "required_audit_id": "R529",
  "required_audit_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/529_wave60_v4_replay_normalization_correction_audit.md",
  "required_scope": "WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION",
  "required_verdict": "PASS",
  "required_findings": {"high": 0, "medium": 0, "low": 0},
  "required_target": {
    "artifact_commit": "DIRECT_PARENT_OF_R529",
    "artifact_path": "SELF_OUTPUT_RELATIVE",
    "artifact_sha256": "SHA256_OF_THIS_ARTIFACT"
  },
  "authority_effect": "ACTIVATES_CONDITIONAL_CORRECTED_VIEW"
}
```

Los tres sentinels son relaciones normativas, no valores libres. R529 debe
resolverlas a: su parent directo R528; el path canónico del propio candidato;
y el SHA físico/blob de ese candidato. El checker R526 valida el schema del
candidato; el auditor R529 valida además la instancia resuelta contra Git y
filesystem. Un cierre posterior puede declarar la condición satisfecha; R528
por sí solo sigue siendo candidato.

## 5. Commit point y cleanup identity-safe

R522 se refina con este algoritmo obligatorio:

1. la creación conserva `created_identity=(st_dev, st_ino)` desde el descriptor
   del leaf;
2. toda reapertura usa el mismo `parent_fd`, `O_NOFOLLOW` y compara esa
   identidad;
3. el `fstat` final exige simultáneamente `S_ISREG`, identidad original,
   `st_nlink==1` y modo `0444`;
4. los bytes leídos desde ese descriptor son exactamente los bytes canónicos y
   parsean al payload exacto;
5. una segunda travesía desde un nuevo descriptor de `REPO_ROOT`, siempre con
   `O_DIRECTORY|O_NOFOLLOW`, debe llegar al mismo parent `(dev,ino)`; el leaf
   reabierto desde allí debe conservar regularidad, identidad, `nlink=1`, modo
   `0444`, bytes y JSON;
6. si cualquier validación falla, el cleanup hace primero
   `stat(..., dir_fd=parent_fd, follow_symlinks=False)` y sólo llama
   `unlink(..., dir_fd=parent_fd)` cuando `(dev,ino)==created_identity`;
7. si el nombre falta o apunta a otra identidad, no se borra nada bajo ese
   nombre; se eleva un error que declara que el leaf propio no pudo retirarse
   por pérdida de identidad;
8. después de un unlink permitido se hace `fsync(parent_fd)` y se confirma que
   el nombre ya no existe. No se buscan ni retiran nombres alternativos.

Esto impide borrar un sustituto. Si el inode propio fue renombrado por un actor
concurrente, queda fuera del nombre canónico y se reporta como residuo no
localizable por esta operación; no se amplía la limpieza a un barrido del
directorio.

La operación exitosa devuelve sólo después de la segunda travesía. La garantía
cubre el commit point; no pretende volver el archivo inmutable frente a actores
que lo alteren después del retorno.

## 6. Cobertura determinista adicional

R526 conserva toda la cobertura R520 y añade:

- rechazo de autoridad sintética por la API pública, sin leaf;
- éxito sólo cuando `publish_correction` deriva internamente el payload;
- ancestro symlink hacia attempt y hacia fuera del repo;
- parent ausente, leaf symlink, leaf preexistente y output no canónico;
- sustitución del nombre antes del cleanup: el sustituto sobrevive y la
  operación informa pérdida de identidad;
- deriva de modo antes del commit point final: fallo, cleanup sólo del inode
  propio y ausencia del leaf;
- hardlink antes del commit point final: fallo por `nlink!=1`, cleanup sólo del
  nombre propio; el alias de test se conserva hasta que el fixture temporal lo
  retira;
- reemplazo del parent entre travesías: fallo por identidad de parent/leaf y
  cleanup identity-safe mediante el descriptor original;
- bytes/JSON alterados antes de la segunda travesía: fallo cerrado;
- huellas físicas y de contenido del attempt canónico idénticas pre/post.

Los hooks de carrera sólo existen como helpers internos monkeypatchables o
wrappers de validación; la API pública no incorpora un modo de omitir
autoridad, contención o commit point. Temporales bajo `/mnt/m2-1TB`, parent
`0755`, GPU invisible, wall/RSS/swaps registrados.

## 7. Semántica científica inalterada

El payload conserva `35/36` histórico, mismatch operacional único, `36/36`
normalizado, `replay_exact=true` sólo en la vista condicional y patrones
`false/false`. También conserva métricas R509, limitaciones, decisión `null`,
autoridad del usuario, no promoción y GPU no usada/consultada.

R521 y R523 permanecen visibles como `REVISE`; R525 y R527 acreditan planes e
implementación sucesores, no reetiquetan el pasado. R529 activa sólo la vista
corregida condicional. No hay techo, arquitectura promovida ni `GO/NO-GO`.

