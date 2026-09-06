# R484 — Auditoría independiente del plan de resolución de R483

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

El plan cubre exhaustivamente el único finding MEDIUM de R483. Cada caso físico, de ledger y de lineage omitido por la implementación `5aee5fb` reaparece como prueba independiente; la secuencia positiva pendiente puede construirse con las APIs productivas vigentes; y la autoridad futura preserva como historia rechazada tanto `e617e15`/R481 como `5aee5fb`/R483. No introduce cambios científicos ni circularidad documental.

## Identidad y alcance

El target comprobado es:

- commit `a034a9aa36e673f0376c2af0a26a54dbf463eb43`;
- parent directo R483 `b944cac9fbcbb3f28e2d7753161c58c978868226`;
- único path del commit `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md`;
- SHA-256 físico y del blob Git `0c22469829176132efb9f039ab261f8cc93821f440117f54669f20d3623bd352`.

Leí completos el plan auditado, R483, la resolución R481 y R482, y contrasté sus obligaciones con el preparador, el runner y la suite actuales. La cadena física también coincide con lo declarado: `f844526` introduce exclusivamente R482; `5aee5fb` es su hijo directo y cambia sólo preparador/test; `b944cac` introduce exclusivamente R483; y el plan auditado es hijo directo de ese informe. El SHA-256 físico/Git de R483 es `7aa50f0fca72b3f796ddd4319f99cf2b9d60c005cbca651ac04c113d3d6be905`.

## Cierre del MEDIUM de R483

R483 dejó cuatro grupos de evidencia pendientes: mutaciones físicas específicas, casos de borde del débito, la secuencia completa posterior al débito unsigned y negativos propios del nuevo sufijo Git (`483_wave60_invalid_preparation_recovery_implementation_reaudit.md:68-95`). El plan los cubre uno por uno:

- agrega mutaciones separadas de escrow, freeze, manifest y benchmark; modos y owners tanto del draw como de archivos sensibles; symlink, hardlink, nodo especial, closed world e inventarios/firmas de primary, replay y par (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:108-143`);
- prueba individualmente todos los valores y campos del objeto de débito, versión, container, source, path/hash de amendment, cada boundary parcial receipt/attestation y la transición a ledger firmado (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:145-167`);
- exige que la secuencia empiece realmente en el origen `INVALID_PREPARATION` v1 y termine en una recuperación posterior desde el par firmado, sin admitir como sustituto el test legacy que parte de un primary ya firmado (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:169-197`);
- amplía la fixture de lineage con ambos `REVISE`, cada parent y commit exclusivo, campos de auditoría, documentos físicos/Git y la partición completa R475/recuperación (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:199-217`).

La exigencia de fallos independientes evita que una mutación temprana enmascare defensas posteriores. Es exactamente la corrección requerida por R483 y no añade un régimen de evidencia distinto.

## Ejecutabilidad de la cadena unsigned y firmada

La secuencia `60 → primary firmado → replay firmado → par → recovery posterior` es realizable con las APIs actuales. `_wave60_invalid_preparation_unsigned_debit()` autoriza únicamente la config v2, el container v1 canónico, la amendment ligada, el objeto exacto de 60 segundos y ausencia de cualquier boundary firmado (`prepare_wave56_fresh.py:6559-6619`). `wave60_prior_preparation_elapsed()` usa ese débito sólo cuando primary carece de receipt y attestation; después de firmar primary, replay hereda su acumulado, y una recuperación posterior valida primary/replay, el par firmado y devuelve `durable_seconds` (`prepare_wave56_fresh.py:6622-6674`).

La finalización productiva escribe `coordinator_budget` antes de firmar el receipt (`prepare_wave56_fresh.py:6677-6704`). En el runner, `pair_preparation_elapsed()` exige que el prior de replay sea el acumulado de primary (`run_wave60_frozen_policy_transport.py:1152-1162`) y `recovery_pair_durable_elapsed()` valida el paquete pre-truth, recompone preparación más fases y expone tanto `preparation_seconds` como `durable_seconds` (`run_wave60_frozen_policy_transport.py:4141-4183`).

La suite existente ya materializa por separado todos los tramos necesarios en el test legacy v2→v3: prepara y firma primary/replay, cierra un par pre-truth, valida su ledger durable y crea una recuperación sucesora (`tests/test_wave60_frozen_policy_transport.py:3049-3983`). La positiva de `5aee5fb` ya aporta el nuevo comienzo específico: débito unsigned `60`, primary `INVALID_PREPARATION` firmado y prior correcto de replay (`tests/test_wave60_frozen_policy_transport.py:1897-2164`). El plan exige unir ambos recorridos sin reemplazar el origen por uno genérico. No requiere una API inexistente ni un cambio de producción previo.

Los casos con sólo un miembro de la pareja receipt/attestation abortan en el boundary explícito; cuando ambos son válidos, el flujo abandona el ramal unsigned y valida el ledger firmado. Esta distinción está expresamente fijada en el plan (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:164-167,185-192`) y coincide con las ramas reales.

## Keyset, lineage y frontera científica

El keyset listado contiene exactamente 27 claves. Conserva `rejected_recovery_implementation` y su auditoría R481 `REVISE`, y añade `r481_resolution_implementation` más su auditoría R483 `REVISE`; `recovery_implementation` queda reservado exclusivamente al sucesor auditado con PASS por R485 (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:36-106,245-279`). Por lo tanto, ninguna de las dos implementaciones incompletas se borra ni se promueve retrospectivamente.

La numeración y los parents forman una secuencia acíclica: R483 → plan → R484 → implementación sucesora → R485 → amendment → R486 → config v2 → R487/HEAD (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:49-67,281-313`). Cada autoridad existe antes de ser referenciada por el siguiente artefacto. Los scopes de plan, implementación, amendment y config permanecen diferenciados y sus targets corresponden al objeto auditado.

El alcance autorizado se limita al preparador y la suite, permitiendo una corrección productiva sólo si una nueva prueba reproduce un defecto (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:219-243`). Módulo, runner y worker siguen bajo R475 y conservan sus hashes `46e31fa…`, `1c778c3e…` y `c6c5c832…`. El plan excluye cambios al estimando, draw, modelo, thresholds y source law, mantiene la config canónica sin `hard_set_tau` y no formula decisión `GO/NO-GO` (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:28-34,300-314`). No quedan findings técnicos abiertos.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R484",
  "scope": "INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "a034a9aa36e673f0376c2af0a26a54dbf463eb43",
    "plan_sha256": "0c22469829176132efb9f039ab261f8cc93821f440117f54669f20d3623bd352"
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
