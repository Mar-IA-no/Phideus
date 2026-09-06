# Ola 60 — plan de resolución del finding R521 sobre la frontera de publicación

> **Estado:** `PROPOSED / PRE-R523-AUDIT / CPU-ONLY / NO-PUBLICATION`
>
> **Autoridad precedente:** R520 `3fa1c5ecfa3cf8107174d2bf6b9a7f58cd96f9d3`; R521
> `c90df0391ef4e8a3eb2cba7a3d36ccf8a10916cd`, `REVISE 0/1/0`.

## 1. Alcance

R521 verificó la genealogía R509–R519, la config y ocho sources, el inventario
físico `141/141`, los receipts locales, el mismatch histórico único y la
recomposición científica `14 acciones / 56 arrays / 301 pair-tokens / 5.000
bootstraps`. Ambos patrones siguen `false/false` con `replay_exact` histórico o
normalizado. Ese resultado no se reabre.

La única deuda es la capacidad que materializa el JSON candidato. R520 permite
llamar `publish_correction(payload)` con autoridad Git sintética que sólo pasa
el schema, y su control léxico de externalidad no impide que un ancestro
symlink resuelva dentro del attempt o fuera del repositorio. R522 corrige sólo
esa frontera; no modifica config, sources, datos, métricas ni el attempt
inmutable.

## 2. Cadena sucesora y nombres normativos

La numeración queda congelada así:

1. **R522:** este plan, hijo directo de R521;
2. **R523:** auditoría independiente del plan, scope
   `R521_PUBLICATION_BOUNDARY_RESOLUTION_PLAN`; debe ser `PASS 0/0/0` antes de
   implementar;
3. **R524:** implementación exclusiva sobre el adjudicador y su test;
4. **R525:** auditoría independiente de R524, scope
   `R521_PUBLICATION_BOUNDARY_RESOLUTION_IMPLEMENTATION`; debe ser
   `PASS 0/0/0`;
5. **R526:** publicación exclusiva de
   `WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json`, todavía con estado
   `CANDIDATE_PENDING_R527_AUDIT`;
6. **R527:** auditoría independiente del artefacto, scope
   `WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION`. Sólo un `PASS 0/0/0` activa la
   vista corregida condicional.

R521 se preserva como autoridad `REVISE`; no puede reetiquetarse ni ser
reemplazada silenciosamente por R525.

## 3. Autoridad de publicación

La función pública con capacidad de escritura deja de aceptar un payload. Su
interfaz recibe exclusivamente:

```text
publish_correction(
  resolution_implementation_commit,
  resolution_implementation_audit_commit,
  resolution_implementation_audit_sha256,
  *,
  output=None,
)
```

Dentro de la misma operación debe llamar a `build_correction_payload(...)`.
Ese builder revalida de forma fail-closed:

- R509–R521, incluidos R516 histórico y R521 `REVISE 0/1/0`;
- R522 físico/blob y parent R521;
- R523 físico/blob, parent R522 y autoridad JSON `PASS 0/0/0`;
- R524 como hijo directo exclusivo de R523, con exactamente adjudicador y test;
- hashes físicos = blobs de esos dos archivos;
- R525 como hijo directo exclusivo de R524, con informe único, SHA físico/blob
  y autoridad JSON `PASS 0/0/0` ligada al commit y a ambos hashes.

Un payload schema-valid no constituye capacidad de publicación. La función de
escritura no expone un argumento alternativo para inyectarlo. El modo CLI
`publish` pasa los tres bindings a esa función; `build` conserva la salida a
stdout sin escribir.

## 4. Contención física del destino

El destino debe ser exactamente el `OUTPUT` normativo y estar léxicamente bajo
`REPO_ROOT`, pero esas condiciones no bastan. La operación debe:

1. exigir que `REPO_ROOT`, el attempt y el parent del output existan como
   directorios físicos;
2. resolver de forma estricta repo, attempt y parent y comprobar que el parent
   físico sigue dentro del repo físico y fuera del attempt físico;
3. abrir `REPO_ROOT` como directorio y recorrer cada componente existente del
   parent mediante `os.open(..., dir_fd=..., O_DIRECTORY | O_NOFOLLOW)`;
4. rechazar cualquier componente symlink, no-directorio, `..`, path absoluto o
   cambio entre el parent abierto y el parent físico esperado;
5. crear sólo el leaf con `os.open(..., dir_fd=parent_fd,
   O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW, 0o444)`;
6. verificar con `fstat` que es regular, `nlink=1`, modo `0444`, y conservar
   `(st_dev, st_ino)`;
7. escribir bytes canónicos, `flush`, `fsync` del archivo y del parent;
8. reabrir el leaf desde el mismo `parent_fd` con `O_NOFOLLOW`, exigir el mismo
   `(dev, ino)`, bytes exactos y JSON exacto, y revalidar la cadena física desde
   el repo hasta el leaf antes de devolver;
9. ante cualquier excepción posterior a la creación, retirar únicamente el
   leaf creado mediante `os.unlink(leaf, dir_fd=parent_fd)` y propagar el fallo.

No se crean parents. Un parent ausente falla cerrado. El descriptor anclado
evita que un cambio del pathname redirija la escritura después del precheck;
la revalidación final detecta sustitución o desplazamiento de la cadena
canónica durante la operación.

## 5. Payload y semántica

El artefacto sucesor incorpora en `authority_chain` las autoridades separadas:

- `r520_initial_implementation`;
- `r521_initial_implementation_audit` con `REVISE 0/1/0`;
- `r522_publication_boundary_resolution_plan`;
- `r523_publication_boundary_resolution_plan_audit` con `PASS 0/0/0`;
- `r524_resolution_implementation`;
- `r525_resolution_implementation_audit` con `PASS 0/0/0`.

La activación cambia sólo a R527. Se conservan sin alteración:

- observación histórica `MISMATCH`, `35/36` y único mismatch operacional;
- vista normalizada `36/36` y `replay_exact=true` condicional;
- patrones `incompatibility=false` y `harm=false`;
- métricas, soportes e intervalos de R509;
- `scientific_decision=null`, `decision_authority=user`,
  `architecture_promoted=false`, `gpu_used_or_queried=false`;
- las limitaciones científicas y la sustitución explícita de la limitación
  operacional obsoleta.

## 6. Cobertura obligatoria

Además de la suite R520 y la regresión Wave 60, R524 debe añadir pruebas que:

- rechacen bindings sintéticos y comprueben que no aparece ningún leaf;
- demuestren que `publish_correction` deriva el payload dentro de la operación
  y no acepta un `Mapping` como sustituto;
- publiquen correctamente en un árbol físico temporal íntegramente bajo un
  repo temporal, con bytes canónicos, modo `0444`, `nlink=1` e inode estable;
- rechacen un ancestro symlink hacia el attempt;
- rechacen un ancestro symlink hacia fuera del repo;
- rechacen symlink en el leaf, parent ausente, path no canónico y archivo ya
  existente;
- simulen sustitución entre apertura y revalidación final y exijan fallo más
  retiro del único leaf propio si sigue ligado al descriptor anclado;
- conserven intactas identidad física y contenido del attempt canónico antes y
  después de toda prueba.

Las pruebas usan temporales bajo `/mnt/m2-1TB`, parent `0755`,
`CUDA_VISIBLE_DEVICES=''`, y registran wall, RSS y swaps. No consultan GPU ni
re-ejecutan generación, score o evaluación.

## 7. Criterio de cierre

R522 sólo autoriza R524 después de una R523 independiente `PASS 0/0/0`. R524
no autoriza R526: requiere R525 independiente `PASS 0/0/0`. R526 no se presenta
como evidencia activa hasta R527 `PASS 0/0/0` sobre su commit, path y SHA.

Ningún resultado de esta cadena promueve arquitectura, declara techo o decide
`GO/NO-GO`.

