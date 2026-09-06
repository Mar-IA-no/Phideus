# R506 — Auditoría de implementación de la resolución de autoridad R504

## Dictamen: PASS

La implementación objetivo corrige el defecto bloqueante de R504-01 sin
debilitar las garantías preexistentes. En un recovery hard-set tipado, la
config v3 autenticada pasa a gobernar los cinco sources invariantes de la
transición; el commit correctivo gobierna únicamente preparer y test. La
partición queda cerrada sobre los siete sources no autorreferentes del roster
de ocho, y el runner pre-v3 es rechazado aunque sea un blob histórico válido.

No encontré findings altos, medios ni bajos. La nueva capa suplementaria
preserva R502/R503 como autoridad histórica independiente, liga R504 REVISE,
el plan y R505, y reserva R506/R507 para la corrección y su amendment. No se
abrió truth, no se creó `attempt_v4`, no se ejecutó el experimento y no se usó
ni consultó GPU.

## Identidad, alcance y parentage

El target es exactamente
`464ceb59b8b634e34d625fe9458201075cdc8e3f`, hijo directo de
`e84bfe701bbc9cf0443c92f02372f74520a135ea` (R505 PASS). El commit modifica
exclusivamente:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`
- `tests/test_wave60_frozen_policy_transport.py`

`git diff --check` no reportó errores. El namespace
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4`
permanece ausente.

## Resolución de la autoridad de sources

`validate_wave60_final_config_authority()` recupera el blob de la config
inmediatamente anterior desde el padre de su auditoría autenticada, deriva el
universo no autorreferente y exige que changed e invariant sean disjuntos y
cubran ese universo. Para hard-set exige además igualdad exacta entre los
invariantes derivados y `unchanged_source_law_sources`, y compara cada uno
contra config final y filesystem. La ruta de validación integral vuelve a
comprobar también el blob de cada invariante en el commit correctivo.

La partición observada es exactamente:

- changed, 2/7: preparer y test;
- invariant, 5/7: módulo científico, runner, worker, R475 y R476;
- self-bound, 1/8: config.

Los cinco blobs invariantes del target coinciden con los hashes fijados por la
config v3. En particular, el runner conserva
`b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261`,
no el runner pre-v3 `1c778c3e...`. La prueba sintética construye ambos blobs,
acepta la composición con el runner v3 y rechaza el rollback físico y de
config al runner original. También itera ataques contra cada uno de los cinco
invariantes.

El caso phase-bound
`test_hard_set_v4_source_baseline_is_v3_not_v1_escrow` ya no carga la config
canónica de HEAD como supuesto baseline: obtiene explícitamente la autoridad
de origen v3 y la copia antes de construir el escenario sintético.

## Esquema suplementario y cadena de auditoría

El schema suplementario
`wave60-hard-set-authority-r504-recovery-amendment-v1` tiene un keyset exacto:
reutiliza el bloque hard-set original y agrega únicamente
`base_recovery_authority`, `rejected_config_authority`, `correction_plan` y
`correction_plan_audit`. Los keysets internos también se validan por igualdad,
no por subconjunto.

La validación recompone y autentica dos capas separadas:

1. amendment inicial/R503 y su implementación R502, incluida la cadena R502
   REVISE y sus correcciones;
2. config rechazada/R504 REVISE, plan de resolución/R505 PASS, implementación
   correctiva/R506 y futuro amendment/R507.

La implementación correctiva debe ser hija directa de R505; su auditoría R506
debe ser hija directa y exclusiva de esa implementación. El amendment futuro
debe ser hijo directo de R506, y R507 debe ser su auditoría exclusiva. La
config final queda reservada para R508. Los parsers siguen exigiendo un único
bloque JSON normativo, verdict PASS y conteos cero para otorgar autoridad.

No se relajaron self-binding, exclusividad de commits, HEAD físico, parentage,
auditorías ni hashes: la ampliación de schema se enruta por las mismas guardas
de canonicalidad, Git blob, estado limpio, parent directo, ancestry y hash
físico que el recovery hard-set original.

## Pruebas CPU-only

Se usaron `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, plugins
externos de pytest deshabilitados y un basetemp propio con modo `0755` bajo
`/mnt/m2-1TB`. No se usó `/tmp` para copias ni artefactos grandes.

| Check | Resultado | wall | max RSS | swaps |
|---|---:|---:|---:|---:|
| `py_compile` de los dos archivos objetivo | PASS | 0.15 s | 35.208 KiB | 0 |
| `pytest -q tests/test_wave60_frozen_policy_transport.py -k 'hard_set_v4 or r504_resolution'` | 20 PASS / 154 deselected | 90.58 s | 891.184 KiB | 0 |

El primer probe buscó el alias global `python`, ausente en este entorno, y
terminó antes de ejecutar el compilador (exit 127, 0 swaps). La comprobación
normativa se repitió inmediatamente con `venv/bin/python` y es la registrada
como PASS en la tabla.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R506",
  "scope": "HARD_SET_AUTHORITY_R504_RESOLUTION_IMPLEMENTATION",
  "target": {
    "implementation_commit": "464ceb59b8b634e34d625fe9458201075cdc8e3f"
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
