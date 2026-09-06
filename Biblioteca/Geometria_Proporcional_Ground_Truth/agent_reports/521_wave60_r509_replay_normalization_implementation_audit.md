# R521 — Auditoría de implementación de la resolución de normalización de replay de Ola 60

## Dictamen técnico: REVISE

La implementación R520 autentica correctamente la evidencia histórica y
recompone sin deriva la vista científica condicional. La genealogía R509–R519,
la config y sus ocho sources, el inventario físico de 141 archivos, los tres
self-manifests, el terminal, el único mismatch histórico y la normalización
local de receipts pasan de forma fail-closed. La recomputación independiente
desde los estados analíticos preservados reproduce 14 acciones, 56 arrays
métricos, 301 pair-tokens, 5.000 réplicas bootstrap y ambos patrones
`false/false`.

Sin embargo, la frontera que materializa el JSON candidato no conserva esas
garantías. `publish_correction()` acepta un payload meramente coherente con el
schema sin reautenticar la autoridad Git dinámica R520/R521 y comprueba la
externalidad del destino sólo de forma léxica, por lo que sigue symlinks en sus
ancestros. Una reproducción aislada hizo que un output léxicamente externo se
resolviera dentro de un attempt temporal y fuera creado allí. Encontré **0
HIGH, 1 MEDIUM y 0 LOW**. El finding no modifica ni cuestiona la ciencia, pero
bloquea la publicación R522 hasta cerrar la frontera de escritura.

## Identidad y alcance auditados

El target es el commit
`3fa1c5ecfa3cf8107174d2bf6b9a7f58cd96f9d3`, hijo directo único de R519
`5e2b2a6fcacc55bc3abc25af603dee0250e26561`. Su diff añade exclusivamente:

- `experiments/geometria_proporcional/adjudicate_wave60_v4_result.py`, SHA-256
  físico/blob
  `f0641c6646c8b8ed413e0d6b509a60934e50cfd448b0bb4bcc1b138cfb18d9f3`;
- `tests/test_wave60_v4_result_adjudication.py`, SHA-256 físico/blob
  `99bd64cc188a46dcb409e079edc5f0dc576e9796d87c6efd0604f2d92c8954a7`.

Leí completos ambos archivos y la cadena documental R509–R519, además de la
config R508, las autoridades R475/R476 y las funciones congeladas necesarias
de configuración, manifiestos, attestations, comparación de roots,
finalización y recomputación estadística. `git diff --check` pasa. Esta
auditoría no corrige implementación, no publica R522, no reejecuta el
experimento y no emite `GO/NO-GO`.

## Finding

### R521-01 — MEDIUM — La frontera de publicación no reautentica autoridad ni contiene físicamente el destino

La derivación CLI normal autentica R520/R521 antes de construir el payload:
`validate_implementation_authority()` exige parent, pathset, blobs, SHA y un
informe R521 exacto. Pero `publish_correction()` recibe un `Mapping`, sólo
llama a `validate_correction_payload()` y luego escribe
(`adjudicate_wave60_v4_result.py:1443-1467`). Esa validación comprueba que el
commit y hashes dinámicos tengan formato hexadecimal y que el JSON R521 sea
internamente coherente con ellos; no vuelve a ejecutar `_require_commit()`,
`git_blob_sha256()` ni `_validate_audit()` para esos valores
(`adjudicate_wave60_v4_result.py:1261-1295`).

La cobertura demuestra involuntariamente el bypass: `_candidate_payload()`
construye `implementation_commit="a"*40`, audit commit `"b"*40` y audit SHA
`"c"*64` (`test_wave60_v4_result_adjudication.py:44-70`), y la prueba de
publicación confirma que ese payload se materializa
(`test_wave60_v4_result_adjudication.py:430-442`). Por tanto, la API que tiene
autoridad de escritura permite omitir la precondición según la cual R522 sólo
puede existir después de una R521 física/Git `PASS 0/0/0`. Que el dispatch CLI
vigente construya primero el payload limita la vía ordinaria, pero no vuelve
fail-closed al publicador ni impide su invocación programática.

La contención física tiene un segundo componente del mismo boundary. La guarda

`output != OUTPUT or OUTPUT.is_relative_to(ATTEMPT)`

en la línea 1445 compara paths léxicos. Después, `mkdir`, `os.open` y la
reapertura siguen ancestros symlink. En un temporal propio se fijó:

- attempt físico: `/mnt/m2-1TB/r521-publish-alias.UIdaVl/fake_attempt`;
- output léxico:
  `/mnt/m2-1TB/r521-publish-alias.UIdaVl/alias/candidate.json`;
- `alias -> fake_attempt`.

El publicador aceptó el payload, creó un archivo regular de 19.040 bytes en
modo `0444` y su destino resuelto fue
`.../fake_attempt/candidate.json`; la prueba explícita
`resolved_output.is_relative_to(resolved_attempt)` dio `true`. Esto contradice
la prohibición de publicar dentro del attempt y la exigencia de output físico
canónico. También admite redirección fuera del repositorio mediante un
ancestro equivalente. El temporal fue retirado por su path exacto después de
la reproducción.

Impacto acotado: el output R522 todavía no existe, el attempt canónico no fue
modificado y toda la ruta read-only es íntegra. El defecto sí permite crear un
candidato sin la autoridad dinámica obligatoria, bloquear luego el path por
`O_EXCL`, escribir en una localización físicamente distinta o —ante una
sustitución concurrente del ancestro— invadir el paquete que se promete
inmutable. Por ello no autoriza pasar a R522.

Corrección mínima requerida:

1. hacer que la operación con autoridad de escritura derive nuevamente el
   payload desde los bindings R520/R521 o reautentique esos bindings contra
   Git, filesystem e informe antes de abrir el destino; un payload
   schema-valid no debe ser por sí solo una capacidad de publicación;
2. comprobar canonicalidad y externalidad físicas de todos los ancestros y
   abrir el parent mediante un descriptor de directorio no-symlink, creando el
   leaf con `dir_fd`, `O_EXCL` y `O_NOFOLLOW` o mecanismo equivalente para no
   reabrir la carrera después del precheck;
3. añadir negativos que intenten publicar con commits/auditoría sintéticos y
   mediante un ancestro symlink que resuelva dentro del attempt o fuera del
   repositorio.

No hace falta modificar R509–R519, la config, los sources congelados ni el
attempt. La resolución puede limitarse al publicador y sus tests, seguida por
una nueva auditoría independiente antes de R522.

## Superficies verificadas sin findings

### Autoridad Git, config y sources

- R509–R519 forman la genealogía exacta declarada; cada documento/informe
  coincide con su blob, parent y pathset, y cada auditoría contiene una única
  autoridad JSON con scope, target, verdict y conteos normativos.
- La config física/blob coincide en
  `191483d2909c3a95a1e82488e1834b55c849763f55c72549f07f2c4cf81d6416`
  y su self-binding en
  `eab40e2d34cfcd532437c5e7567ac94b7988a46afb865bab84728fa90735e810`.
- El mapa de ocho sources coincide `8/8`. El módulo principal usa el valor
  correcto
  `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`
  (`...fc4e...`), no la variante falsa `...fc4f...` preservada sólo como
  historia rechazada y negativo de regresión.
- R516 permanece histórico no autorizante, R517 conserva `REVISE 1/0/0`, y
  R514/R515 siguen siendo autoridades sustantivas como dispusieron R518/R519.

### Identidad física, terminal y replay

- Attempt, primary, replay y pair son directorios físicos canónicos; el único
  roster es `65/66/10`, total `141` archivos regulares.
- Los 141 tienen `st_nlink=1` y 141 pares `(st_dev,st_ino)` únicos; no hay
  symlinks, nodos especiales, extras ni faltantes.
- Los 138 paths manifestados coinciden en bytes, SHA, UID, GID y mode. Los
  self-manifests coinciden `3/3`: 17.792/18.047/2.448 bytes, UID/GID `0:0`,
  modo `0444` y hashes normativos.
- Los seis hashes target R509, manifests, firmas, receipts y pair attestation
  pasan. El terminal es `COMPLETE`, ambos roots son
  `EVALUATED_IMMUTABLE`, `any_truth_accessed=true` y
  `recovery_allowed=false`.
- La comparación histórica recompone exactamente 36 checks, 35 verdaderos y
  el único falso `operational:preparation_receipt.json`; el listado publicado
  de mismatch es exactamente `operational:preparation_receipt.json`.
- Cada preparation receipt liga su generation receipt local. Los freezes son
  exactos y los generation receipts sólo difieren en
  `execution_mode=recovery/replay`; sus SHA locales permanecen distintos y
  válidos.

### Recomposición científica y estado de autoridad

Una recomputación independiente, sin forward, training ni reapertura semántica
manual de secretos, leyó `analysis_arrays.npz`, `bootstrap_indices.npz`,
`evaluation_index.npz`, `monitor_policy_arrays.npz` y `analysis.json`:

- 14 acciones y 56 arrays métricos reproducen exactamente los 56 summaries;
- los 16 deltas principales, dos deltas de controles y 12 soportes coinciden;
- el bootstrap conserva 5.000×301 índices PCG64 y orden de tokens exacto;
- incompatibility conserva `mean=-0.000622923588039867`, IC95
  `[-0.0031284606866002216, 0.0015481381506090807]`;
- harm conserva `mean=-0.0016611295681063123`, IC95
  `[-0.014064230343300111, 0.009468438538205979]`;
- con replay histórico y normalizado, los patrones continúan
  `incompatibility=false`, `harm=false`.

El payload previsto permanece honestamente
`CANDIDATE_PENDING_R523_AUDIT`; `scientific_decision=null`,
`decision_authority=user`, `architecture_promoted=false` y las tres
limitaciones científicas no se borran. No hay promoción ni `GO/NO-GO`.

## Pruebas, inmutabilidad y recursos

Todas las invocaciones fijaron `CUDA_VISIBLE_DEVICES=''`, cuatro threads,
`PYTHONDONTWRITEBYTECODE=1`; pytest deshabilitó plugins externos y cache y usó
un basetemp propio bajo `/mnt/m2-1TB` con parent `0755`.

| Check | Resultado | wall | max RSS | swaps |
|---|---:|---:|---:|---:|
| suite completa R520 + regresión Wave 60 | 195 PASS | 423,72 s pytest / 424,86 s proceso | 902.928 KiB | 0 |
| `check-attempt` real | PASS | 2,28 s | 830.148 KiB | 0 |
| autoridades estáticas + config/sources | 11 autoridades / 8 sources PASS | 1,83 s | 825.648 KiB | 0 |
| recomputación analítica independiente | 14/56/301/5.000 exactos | 1,61 s | 839.468 KiB | 0 |
| ataque de ancestro symlink | publicación indebida reproducida | 2,6 s | no medido separadamente | no observado |

Antes y después de todos los checks, el attempt tuvo 141 archivos y cero
symlinks. La huella agregada de metadata/inodos permaneció
`25fde5298e5745b54d388a134ff055c000435bb5e8dc1dfef66d9a9010dd2846` y
la huella agregada de contenido
`78fe6a059868ccda3b11719011369ff471b3b9305637eb4292a481fc63145937`.
Los temporales propios, 4,6 GiB de basetemp y el probe de publicación, fueron
retirados por sus paths exactos. Al cierre operativo había aproximadamente 22
GiB de RAM disponible; el swap global conserva uso histórico alto, mientras
la segunda muestra registró `so=0` y los procesos medidos registraron cero
swaps.

No se usó ni consultó GPU, no se ejecutó replay/recovery del experimento y no
se modificaron config, sources ni data.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R521",
  "scope": "R509_REPLAY_NORMALIZATION_RESOLUTION_IMPLEMENTATION",
  "target": {
    "implementation_commit": "3fa1c5ecfa3cf8107174d2bf6b9a7f58cd96f9d3",
    "files": {
      "experiments/geometria_proporcional/adjudicate_wave60_v4_result.py": "f0641c6646c8b8ed413e0d6b509a60934e50cfd448b0bb4bcc1b138cfb18d9f3",
      "tests/test_wave60_v4_result_adjudication.py": "99bd64cc188a46dcb409e079edc5f0dc576e9796d87c6efd0604f2d92c8954a7"
    }
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
