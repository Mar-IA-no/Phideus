# Ola 60 — resolución atómica del finding R525

> **Estado:** `PROPOSED / PRE-R527-AUDIT / CPU-ONLY / NO-PUBLICATION`
>
> **Cadena precedente:** R524 `1fd667979b70a0ab82f626e731f369a58220b774`;
> R525 `4b8c14e4f35a1664feed9c5614dbb003fb269eab`, `REVISE 0/1/0`.

## 1. Cambio decisivo

R525 mostró que ningún `stat` seguido de `unlink` puede borrar
condicionalmente por inode de forma atómica. Esta resolución elimina por
completo el cleanup post-publicación por nombre. El candidato se construye en
un inode sin nombre con `O_TMPFILE`, se valida por descriptor y se publica con
una única operación `linkat(AT_EMPTY_PATH)` que falla si el leaf ya existe.

Antes de `linkat`, cerrar el descriptor destruye automáticamente el inode
anónimo: no existe un nombre parcial que limpiar. Después de `linkat`, el
enlace exitoso es el punto de linealización irreversible de esta operación. Si
una validación posterior detecta interferencia, se informa el residuo y no se
ejecuta `unlink`, `rename`, búsqueda por inode ni otra acción destructiva.

El host fue probado de forma aislada bajo `/mnt/m2-1TB`: el filesystem acepta
`O_TMPFILE`, `linkat(AT_EMPTY_PATH)`, `nlink 0→1` y modo final `0444`. La
implementación no incorpora fallback a temporales con nombre; ausencia de
soporte falla cerrado antes de publicar.

## 2. Topología Git y contratos sucesores

La cadena normativa reemplaza la numeración futura de R524. Cada commit es
hijo directo único, con pathset completo y status exacto:

| ID | Parent | Pathset y status |
|---|---|---|
| R526 | R525 `4b8c14e4f35a1664feed9c5614dbb003fb269eab` | `A Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R525_ATOMIC_PUBLICATION_RESOLUTION_PLAN.md` |
| R527 | R526 | `A Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/527_wave60_r525_atomic_publication_resolution_plan_audit.md` |
| R528 | R527 | `M experiments/geometria_proporcional/adjudicate_wave60_v4_result.py`; `M tests/test_wave60_v4_result_adjudication.py` |
| R529 | R528 | `A Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/529_wave60_r525_atomic_publication_resolution_implementation_audit.md` |
| R530 | R529 | `A Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json` |
| R531 | R530 | `A Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/531_wave60_v4_replay_normalization_correction_audit.md` |

R528 requiere R527 `PASS 0/0/0`; R530 requiere R529 `PASS 0/0/0`; R531 no
comparte commit con el artefacto auditado.

Todas las auditorías usan exclusivamente el keyset top-level
`{schema_version,audit_id,scope,target,technical_verdict,findings,files_modified,gpu_used_or_queried}`,
con schema `wave60-audit-authority-v1`. Para autorizar deben declarar
`technical_verdict=PASS`, findings `0/0/0`, `files_modified=false` y
`gpu_used_or_queried=false`.

### R527

- path:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/527_wave60_r525_atomic_publication_resolution_plan_audit.md`;
- scope: `R525_ATOMIC_PUBLICATION_RESOLUTION_PLAN`;
- target exacto:
  `{plan_commit:<R526>, plan_path:"Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R525_ATOMIC_PUBLICATION_RESOLUTION_PLAN.md", plan_sha256:<SHA físico/blob>}`.

### R529

- path:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/529_wave60_r525_atomic_publication_resolution_implementation_audit.md`;
- scope: `R521_PUBLICATION_BOUNDARY_RESOLUTION_IMPLEMENTATION`;
- target exacto:
  `{implementation_commit:<R528>, files:{"experiments/geometria_proporcional/adjudicate_wave60_v4_result.py":<SHA físico/blob>, "tests/test_wave60_v4_result_adjudication.py":<SHA físico/blob>}}`.

### R531

- path:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/531_wave60_v4_replay_normalization_correction_audit.md`;
- scope: `WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION`;
- target exacto:
  `{artifact_commit:<R530>, artifact_path:"Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json", artifact_sha256:<SHA físico/blob>}`.

## 3. Autoridad de la operación

La única API que escribe conserva la interfaz fijada por R522:

```text
publish_correction(
  resolution_implementation_commit,
  resolution_implementation_audit_commit,
  resolution_implementation_audit_sha256,
  *, output=None,
)
```

La función llama internamente a `build_correction_payload(...)`. El builder
reautentica R509–R527, R528 y R529 mediante parents, pathsets/status, blobs,
hashes físicos y JSON exactos. R520/R521, R522/R523 y R524/R525 permanecen en
la cadena con sus veredictos históricos reales; R529 no los reetiqueta.

Un `Mapping` o JSON schema-valid no es argumento de publicación. El CLI
`publish` pasa los tres bindings; `build` sólo emite bytes por stdout.

## 4. Preparación anónima y publicación linealizable

### 4.1 Árbol físico

Antes de crear el inode anónimo:

1. `output` se normaliza a `OUTPUT` cuando es `None` y debe ser exactamente el
   path normativo;
2. `REPO_ROOT`, attempt y parent del output existen y resuelven estrictamente
   a directorios físicos;
3. el parent físico queda dentro del repo físico y fuera del attempt físico;
4. desde un descriptor `REPO_ROOT` se recorre cada componente mediante
   `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, sin `..`, symlinks ni creación de
   parents;
5. el leaf debe estar ausente según lookup no-follow relativo al parent;
6. se conserva la identidad `(dev,ino)` del parent abierto.

### 4.2 Inode sin nombre

Se abre `"."` relativo al parent con
`O_TMPFILE|O_RDWR|O_CLOEXEC`, modo de creación `0444`. No hay fallback. Sobre
ese descriptor se:

- escriben todos los bytes canónicos con loop que rechaza short/zero writes;
- fija modo `0444` con `fchmod`;
- ejecuta `fsync`;
- valida `S_ISREG`, `nlink=0`, modo `0444`, tamaño exacto y bytes exactos
  mediante lectura/pread del mismo descriptor;
- valida que esos bytes parsean exactamente al payload ya autenticado.

Toda excepción aquí sólo cierra el descriptor: el kernel retira el inode
anónimo, sin lookup ni cleanup por nombre.

### 4.3 Punto de linealización

Inmediatamente antes de publicar, una segunda travesía desde `REPO_ROOT`
confirma el mismo parent `(dev,ino)` y ausencia del leaf. Luego una única
llamada libc `linkat(tmp_fd, "", parent_fd, leaf, AT_EMPTY_PATH)` crea el
nombre. `EEXIST` se expone como `FileExistsError`; cualquier otro errno falla
cerrado. No se permiten `rename`, overwrite ni fallback.

El éxito de `linkat` es el commit point. A partir de allí jamás se llama
`unlink` o `rename` sobre el leaf, aun si `fsync(parent_fd)` o una comprobación
posterior falla.

### 4.4 Revalidación post-commit no destructiva

Tras el enlace:

- `fstat(tmp_fd)` exige identidad original, regularidad, `nlink=1`, modo
  `0444`, tamaño y bytes exactos;
- se hace `fsync(parent_fd)`;
- una tercera travesía desde un descriptor nuevo de `REPO_ROOT` exige parent
  idéntico y abre el leaf con `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`;
- `fstat` y lectura exigen la misma identidad, regularidad, `nlink=1`, modo
  `0444`, tamaño, bytes canónicos y JSON exacto.

Sólo entonces retorna éxito. Si falla, eleva
`PublicationCommittedUnverified` —subclase de `AdjudicationError`— con path e
identidad creada, y no modifica ninguna entrada. La existencia de un residuo
se trata como estado que requiere inspección y recuperación explícita; no se
oculta ni se borra automáticamente.

La garantía termina en el retorno. No afirma inmunidad frente a mutaciones de
terceros posteriores.

## 5. Cobertura obligatoria

R528 conserva toda la suite R520 y añade pruebas deterministas de:

- autoridad sintética rechazada por `publish_correction`, sin leaf;
- derivación interna del payload y éxito canónico en repo temporal físico;
- falta de soporte `O_TMPFILE` o `AT_EMPTY_PATH`: fallo antes de leaf y sin
  fallback;
- ancestro symlink hacia attempt y hacia fuera del repo;
- parent ausente, output no canónico, leaf symlink y archivo preexistente;
- competidor que crea el leaf justo antes de `linkat`: `EEXIST`, competidor
  intacto y ningún unlink;
- parent sustituido entre travesías: fallo antes de link;
- deriva de modo, hardlink, bytes o JSON después de link: excepción
  `PublicationCommittedUnverified`, residuo preservado y cero llamadas a
  `unlink/rename`;
- excepción de `fsync(parent_fd)` después de link: mismo estado committed pero
  no verificado, residuo preservado;
- instrumentación explícita que hace fallar el test si cualquier branch
  post-link invoca `os.unlink`, `os.remove`, `Path.unlink`, `os.rename` o
  `os.replace`;
- attempt canónico con huellas físicas y de contenido idénticas pre/post.

Los tests de residuos operan sólo bajo temporales propios de pytest; su fixture
retira el árbol completo al terminar. Basetemp bajo `/mnt/m2-1TB`, parent
`0755`, CUDA invisible, wall/RSS/swaps medidos. No se reejecuta el experimento.

## 6. Candidato y activación

R530 conserva el path de salida y cambia su estado a
`CANDIDATE_PENDING_R531_AUDIT`. La condición de activación exige:

```json
{
  "required_audit_id": "R531",
  "required_audit_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/531_wave60_v4_replay_normalization_correction_audit.md",
  "required_scope": "WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION",
  "required_verdict": "PASS",
  "required_findings": {"high": 0, "medium": 0, "low": 0},
  "required_target": {
    "artifact_commit": "DIRECT_PARENT_OF_R531",
    "artifact_path": "SELF_OUTPUT_RELATIVE",
    "artifact_sha256": "SHA256_OF_THIS_ARTIFACT"
  },
  "authority_effect": "ACTIVATES_CONDITIONAL_CORRECTED_VIEW"
}
```

R531 resuelve los sentinels a su parent R530, el path normativo y el SHA
físico/blob del candidato. R530 por sí solo no activa nada.

## 7. Ciencia inalterada

El arreglo no modifica la observación histórica `MISMATCH 35/36`, el mismatch
operacional único, la normalización condicional `36/36`, los patrones
`false/false`, métricas, intervalos, soportes ni limitaciones. Conserva
`scientific_decision=null`, autoridad del usuario, no promoción y GPU no
usada/consultada.

No se declara techo, arquitectura promovida ni `GO/NO-GO`.
