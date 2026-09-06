# R525 — Auditoría de la resolución R524 de los findings del plan de publicación de Ola 60

## Dictamen técnico: REVISE

R524 resuelve el segundo finding de R523 y la mitad correspondiente al commit
point del primero. La topología Git R524–R529, los paths y pathsets/status, los
targets exactos de R525/R527/R529 y la condición de activación de R528 quedan
congelados. La autorreferencia imposible del candidato se reemplaza por tres
sentinels normativos que R529 debe resolver contra su parent, el path canónico y
el SHA físico/blob de R528. El commit point final vuelve a exigir regularidad,
identidad, `nlink=1`, modo `0444`, bytes, JSON y cadena física mediante una
segunda travesía.

El cleanup, en cambio, sigue sin ser identity-safe de manera atómica. La
secuencia normativa `stat(..., follow_symlinks=False)` seguida por
`unlink(..., dir_fd=parent_fd)` deja una ventana en la que el nombre puede ser
sustituido después del lookup y antes del unlink. Encontré **0 HIGH, 1 MEDIUM y
0 LOW**. La ciencia, el attempt y el candidato inexistente no fueron alterados.

## Identidad y alcance

Audité el plan R524 en commit
`1fd667979b70a0ab82f626e731f369a58220b774`, hijo directo único de R523
`5f371c8e1d9369a03a1c168a3626018529276348`. El commit añade exclusivamente
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R523_PLAN_FINDINGS_RESOLUTION.md`,
con SHA-256 físico/blob
`c32f043342e8dc3d027ee523a4fc86440427e5fafd1c71c0c1b66d726c9ce895`.

Leí completos R521, R522, R523 y R524, además del adjudicador R520 y su test.
Contrasté parent, pathset, status y hashes contra Git. La auditoría fue
CPU-only; no consultó ni usó GPU, no ejecutó publicación, no modificó checker,
tests, config, sources, datos ni attempt y no emitió `GO/NO-GO`. El runtime no
expone introspección independiente del identificador de modelo; no observé una
señal de configuración distinta del dispatch requerido `gpt-5.6-sol/high`.

## Finding

### R525-01 — MEDIUM — El lookup de identidad y el unlink siguen separados por una carrera que puede borrar un sustituto

R524 ordena consultar el nombre con `stat(..., dir_fd=parent_fd,
follow_symlinks=False)` y, si su `(dev,ino)` coincide con `created_identity`,
desvincular después el mismo nombre mediante `unlink` (plan R524:154-156). El
descriptor de directorio estabiliza qué directorio se usa, pero no estabiliza
qué inode ocupa una entrada. Otro actor puede renombrar el leaf propio e
instalar un sustituto exactamente entre esas dos syscalls. El `stat` habrá
observado el inode propio; el `unlink` posterior resolverá de nuevo el nombre y
borrará el sustituto. POSIX/Linux no hace que `unlinkat` sea condicional al
`st_dev/st_ino` observado previamente.

La cobertura propuesta sólo exige sustitución "antes del cleanup" y que el
sustituto sobreviva (plan R524:180-181). Un test que sustituye antes del `stat`
pasa porque detecta otra identidad, pero no prueba la ventana decisiva entre el
lookup y el unlink. La afirmación "Esto impide borrar un sustituto" (plan
R524:163) es por tanto más fuerte que el algoritmo especificado.

Impacto: una excepción posterior a crear el candidato todavía puede borrar un
archivo ajeno colocado concurrentemente bajo el nombre canónico. Esto conserva
el defecto material central de R523-01 aunque la revalidación del commit point
sí haya sido completada. R526 no debe implementarse bajo este contrato.

Resolución mínima requerida:

1. eliminar todo cleanup post-publicación que haga `unlink` del nombre canónico
   a partir de un lookup previo no atómico, o introducir un mecanismo real que
   excluya mutaciones del directorio durante lookup+unlink; un lock meramente
   cooperativo no autoriza la garantía frente a actores no participantes;
2. preferentemente construir y validar los bytes en una ubicación privada y
   publicar mediante una única operación atómica `NOREPLACE`, tratándola como
   punto de linealización; después de ese punto, un fallo no debe intentar
   borrar por nombre una entrada cuya identidad ya no puede condicionarse
   atómicamente;
3. añadir un negativo determinista con hook exactamente después del lookup de
   cleanup y antes del unlink, que sustituya el nombre y exija que el sustituto
   no sea borrado ni desplazado.

No corresponde ampliar cleanup a búsquedas por inode ni borrar alias o nombres
alternativos. Si no puede garantizarse retirada segura, el resultado debe
fallar cerrado y declarar el residuo sin tocar una entrada de identidad
incierta.

## Superficies resueltas sin findings

- El commit point final revalida sobre descriptores regularidad, inode,
  `nlink=1`, modo `0444`, bytes canónicos, JSON exacto y la cadena física desde
  un nuevo descriptor de `REPO_ROOT` (plan R524:142-153,168-170).
- La topología declara parents directos y pathsets/status exclusivos para
  R524–R529; R525 y R529 son commits separados de sus targets y R526/R528 están
  bloqueados por las auditorías precedentes (plan R524:20-37).
- R525, R527 y R529 tienen path, scope, top-level keyset cerrado y target exacto
  ligado respectivamente a plan, dos archivos de implementación y candidato
  (plan R524:39-107).
- R528 evita la autorreferencia literal mediante sentinels relacionales; R529
  debe resolverlos al parent R528, path canónico y SHA físico/blob y contrastar
  Git más filesystem. R528 solo permanece candidato (plan R524:109-136).
- Los hooks internos hacen testables las carreras sin abrir bypass público de
  autoridad o contención. Los negativos de modo, hardlink, parent y bytes/JSON
  cubren las demás derivas materiales (plan R524:172-195).
- Permanecen visibles R521/R523 `REVISE`; no cambia `35/36` histórico, mismatch
  único, `36/36` normalizado condicional, patrones `false/false`, métricas,
  limitaciones, decisión `null`, autoridad del usuario ni no-promoción (plan
  R524:197-206).

El único diagnóstico de `git diff --check` fue una línea vacía final en el plan;
es cosmético y no se eleva como finding. El artefacto R528 no existe.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R525",
  "scope": "R523_PUBLICATION_BOUNDARY_PLAN_FINDINGS_RESOLUTION",
  "target": {
    "plan_commit": "1fd667979b70a0ab82f626e731f369a58220b774",
    "plan_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R523_PLAN_FINDINGS_RESOLUTION.md",
    "plan_sha256": "c32f043342e8dc3d027ee523a4fc86440427e5fafd1c71c0c1b66d726c9ce895"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 1,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
