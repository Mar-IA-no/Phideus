## Auditoría independiente R479

**Dictamen técnico: `REVISE` — 0 HIGH / 1 MEDIUM / 0 LOW.**

Los dos findings MEDIUM de R478 están resueltos. El plan introduce, sin embargo, una omisión nueva en la cadena de autoridad de `hard_set_tau`: el valor está ligado al snapshot físico, pero no queda exigido explícitamente el enlace transitivo con el `source_law_request.json` de v2 antes de consumirlo.

### Identidad y alcance

Se releyeron completos:

- el plan de resolución R478;
- la auditoría R478;
- el plan rechazado.

La identidad del target es exacta:

- Commit: `0c21db44ebb428647dbcd7921713b593895fc07a`.
- Parent directo: R478, `979c835bc2b2f08182e23e919f265f4fe2bc480a`.
- Único path modificado: `WAVE_60_INVALID_PREPARATION_RECOVERY_R478_RESOLUTION_PLAN.md`.
- SHA-256 físico y blob Git: `39f13cd749083ed581b967166e823ff66e872b9519013c3f1e66523ee58e9c8c`.
- Worktree limpio.

La cadena previa también coincide:

`R477 2a10b6c → plan ba193d5 → R478 979c835 → resolución 0c21db4`.

### Cierre de los findings R478

El plan resuelve correctamente ambos MEDIUM anteriores:

- Mantiene `implementation_binding` top-level en R475 y conserva byte-exactos módulo científico, runner y worker.
- Limita el parche de recuperación a `prepare_wave56_fresh.py` y su test.
- Introduce `recovery_implementation` como autoridad separada R480, con roster, hashes old/new, commit exclusivo y auditoría exclusiva.
- Conserva source law v2/R476 intacta y exige una validación real con rechazo de bindings cruzados.
- Congela el schema y keyset exactos tanto de la amendment nueva como de `attempt.recovery`.
- Liga transitivamente plan rechazado, R478, resolución, R479, implementación y R480 sin duplicarlos en la config.
- Reemplaza sólo para el nuevo schema la regla legacy `parent(amendment)=R477`, manteniendo la anterior para amendments legacy.
- La cadena `R477 → plan → R478 → resolución → R479 → implementación → R480 → amendment → R481 → config v2 → R482` es realizable, directa, exclusiva y no circular.
- El débito `60.0` tiene schema exacto, rama conservadora cerrada y ecuaciones correctas: se carga una sola vez en la primaria v2, el replay hereda su acumulado y las recuperaciones posteriores usan el ledger firmado sin volver a sumarlo.
- El origen anidado queda correctamente condicionado a la validación completa del pair terminal, ambos roots `INVALID_PREPARATION`, inventario `INITIALIZED`, directorio físico `primary/failed_preparation`, owner/modos, ausencia de aliases y mapa closed-world.
- `hard_set_tau` no entra en la config científica canónica ni mediante default: se incorpora sólo en una vista efímera inmediatamente antes del materializador y se propaga a provenance, freeze y receipts.

Los hechos físicos respaldan ese diseño:

- El pair v1 valida como `PAIR_ABORTED_PRE_TRUTH`, con ambos roots `INVALID_PREPARATION`, `any_truth_accessed=false` y `recovery_allowed=true`.
- `primary/failed_preparation` existe como directorio físico `root:root 0700`; allí viven el escrow `0600`, freeze, manifest y benchmark.
- No existen receipts de preparación en primary ni replay.
- La source law v2 real pasa `validate_source_authority()`.
- Los tres blobs reservados a R475 coinciden exactamente con sus blobs en `9f1a229…`.

### MEDIUM 1 — Falta ligar explícitamente `hard_set_contract` al request v2 antes del materializador

El plan fija correctamente:

- `hard_set_tau=0.5`;
- el path físico del snapshot Wave 59;
- su hash `f6edfd…`;
- la prohibición de defaults;
- la verificación del valor dentro del archivo.

Pero la autoridad declarada es `wave59_config_snapshot_bound_by_source_law_v2`, y el plan no exige que el runtime compruebe explícitamente, antes de construir `materializer_config`, estos dos enlaces dentro del request v2:

```text
source_law_request.source_paths["wave59_config_snapshot.json"]
  == hard_set_contract.source_path

source_law_request.source_sha256["wave59_config_snapshot.json"]
  == hard_set_contract.source_sha256
```

Esto importa porque el snapshot Wave 59 no está copiado dentro del authority root. El artefacto real sí contiene las asociaciones correctas:

- alias → path en `source_law_request.json`;
- alias → hash en ese mismo request;
- el manifest v2 liga el request por SHA-256;
- el snapshot físico da exactamente el hash declarado y `hard_set_tau=0.5`.

No hay evidencia de corrupción actual. La omisión está en el contrato ejecutable del plan: sólo se exige abrir el snapshot físico y, separadamente, llamar `validate_source_authority()` en una prueba integrada. La preflight vigente comprueba la auditoría R476 y el hash del manifest, pero no atraviesa ese request; la validación integral de source law ocurre después, en `bind_source_law()`, mientras el materializador ya consumió `hard_set_tau`.

Corrección requerida:

- Antes de materializar, validar el authority root v2 y que su manifest liga el request exacto.
- Exigir las dos igualdades alias→path→hash anteriores.
- Sólo después verificar hash físico, parsear el snapshot y exigir `hard_set_tau == 0.5`.
- Agregar tests que rechacen alias ausente, path cruzado, hash cruzado y request no ligado por el manifest.

Esta corrección permanece dentro del roster ya autorizado `preparer + test`; no requiere alterar source law v2, módulo, runner ni worker.

### Cobertura

La matriz propuesta cubre apropiadamente los dos findings R478, el origen anidado, schemas cerrados, lineage, débito one-shot, materializador real, reproducción primaria/replay y regresión Wave 56–60. Para quedar completa debe incorporar la prueba específica del enlace transitivo del snapshot con el request y manifest v2.

No se modificaron archivos ni artefactos. La auditoría fue CPU-only y no se usó ni consultó GPU.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R479",
  "scope": "INVALID_PREPARATION_RECOVERY_R478_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "0c21db44ebb428647dbcd7921713b593895fc07a",
    "plan_sha256": "39f13cd749083ed581b967166e823ff66e872b9519013c3f1e66523ee58e9c8c"
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
