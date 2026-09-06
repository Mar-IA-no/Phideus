# R487 — Auditoría independiente del plan de resolución de R486

**Dictamen técnico: `REVISE` — 0 HIGH / 1 MEDIUM / 0 LOW.**

El plan corrige de manera suficiente los dos findings de R486: incorpora R486
en una matriz real de ocho auditorías R481–R488 y redefine la prueba de
provenance como igualdad en tres artefactos más firma transitiva del receipt,
sin ampliar el schema de attestation. El keyset de 33 claves, los seis nombres
nuevos, la numeración R487–R491 y las construcciones de negativos heredadas son
coherentes y ejecutables. No obstante, el documento liga R486 a un SHA de commit
incorrecto e inexistente. Dado que esa identidad debe ingresar como binding
histórico duro de la amendment, el plan no puede recibir autoridad PASS tal como
está escrito.

## Identidad y alcance

El target auditado es exactamente:

- commit: `0be3cd6c8b2717efd5ad4b4c24c09c7b55fa1c37`;
- parent directo real: `2049eff3b411024e6b4fd444f2b975ae76c27f3e`;
- árbol Git: `623994e4ea68fdfd4cf22ff8d10689dfb0bdce8d`;
- único path introducido:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md`;
- SHA-256 físico y del blob Git:
  `1b2d64b70196ef8af46d388ddaf668c025fe4af02a5879238f2594df8eb64fbc`.

Leí completos este plan, el plan R485, la auditoría R486 y el informe R485. El
plan R485 coincide física/Git con
`cbbdbf361deaa54d9bf2461aceda7bef54412bb1fa40d821a974f2fb02520e4a`;
R486 coincide con
`677cfce51190d1d5d269e716543dfa10be70b0659838d948baf669e6b7520ffd`;
y R485 coincide con
`d58148ce731c3d06e61101f44dd7db5165b9932ce47c8acc85b45383ffe15650`.
`git diff --check` quedó limpio. No se modificó código ni se consultó GPU.

## Los dos findings R486 sí quedan resueltos

La matriz positiva enumera exactamente ocho autoridades sin saltos: R481,
R483, R485 y R486 como `REVISE`, y R482, R484, R487 y R488 como `PASS`
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md:70-86,122-125`).
Para cada una exige negativos independientes de `audit_id`, `scope`, `target`,
`technical_verdict` y `findings`, además de commit/path/SHA externos. Cuando el
campo vive sólo dentro del informe, manda crear un reporte alternativo real,
hijo del parent correcto, exclusivo de su path, con hash actualizado y una sola
diferencia semántica comprobada antes de invocar el validador
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md:88-120`). Esto
cierra expresamente la omisión de R486 detectada por su propia auditoría.

La provenance también queda formulada conforme al contrato productivo. El plan
exige igualdad del objeto completo en `generation_receipt.json`,
`preparation_freeze.json` y `preparation_receipt.json`; luego verifica la firma
Ed25519 y el record de `preparation_receipt.json` con path, bytes y SHA físicos,
y compara el SHA firmado contra el receipt que contiene la provenance
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md:39-68`). Declara
explícitamente que no se añade `recovery_provenance` a la attestation ni cambia
su schema. Esto coincide con `publish_wave60_preparation_attestation()`, cuyo
payload firma el mapa de records y no duplica la provenance
(`prepare_wave56_fresh.py:6125-6180`). La positiva usará R488 y excluirá R483,
R485 o cualquier `REVISE` como autoridad positiva.

## Keyset, lineage y ejecutabilidad

El conteo es correcto: las 27 claves fijadas por R483 más seis objetos dan 33.
Los nombres son inequívocos:

```text
r483_resolution_implementation
r483_resolution_implementation_audit
r485_resolution_plan
r485_resolution_plan_audit
r486_resolution_plan
r486_resolution_plan_audit
```

La semántica conserva `f32ba2b`/R485 y el plan R485/R486 como historia
`REVISE`; este plan y R487 quedan como resolución auditada; y
`recovery_implementation` se reserva para el sucesor aceptado por R488
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md:127-170`). No se
amplía el schema `wave60-invalid-preparation-recovery-amendment-v1` ni las doce
claves de `attempt.recovery`.

La secuencia R487 → implementación → R488 → amendment → R489 → config → R490
→ resultados R491 es acíclica: cada artefacto que será referenciado existe en
un commit padre anterior, y planes/auditorías son exclusivos de un path mientras
las implementaciones se limitan a preparer/test
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md:173-195,207-256`).

Las §§5–7 del plan R485 permanecen expresamente vigentes. La divergencia
blob/físico puede aislarse dejando A en el commit, B en el filesystem y ligando
SHA(B); `validate_wave60_bound_document()` alcanza entonces la comparación
blob↔binding después de superar path, hash físico, exclusividad y parent
(`prepare_wave56_fresh.py:460-477`). Los commits con path extra se construyen
desde el parent propio y comprueban primero el delta exacto. Las tres cadenas
secundarias mutan físicamente un source científico antes del ancla R480 y
reconstruyen el resto del sufijo con parents y deltas válidos, de modo que el
único hecho inválido final es la diferencia contra el blob R475. Estas
construcciones no requieren cambiar módulo, runner, worker ni ciencia canónica.

## Finding MEDIUM — el commit declarado para R486 no existe

El encabezado declara:

```text
2049effcdd60c3a922aee355266a87440ab97de2
```

como commit de la auditoría R486
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md:8-12`). Git no
contiene ese objeto. El informe R486 fue introducido exclusivamente por:

```text
2049eff3b411024e6b4fd444f2b975ae76c27f3e
```

que es hijo directo del plan R485 `0c07b10abf7831dc6577c39637f3b68c2a3a02b2`
y parent directo del plan auditado `0be3cd6c8b2717efd5ad4b4c24c09c7b55fa1c37`.
Su blob del informe coincide con el SHA declarado
`677cfce51190d1d5d269e716543dfa10be70b0659838d948baf669e6b7520ffd`.

No es un error editorial inocuo: `r485_resolution_plan_audit` debe preservar
exactamente el commit de R486 y el preparador autentica commits, introducción,
parent directo, exclusividad y blob. Copiar la identidad escrita produciría un
binding imposible de resolver; sustituirla tácitamente rompería la trazabilidad
del plan. La corrección requerida es reemplazar el SHA erróneo por el commit
real completo, sin cambiar las demás obligaciones. El plan corregido necesitará
otra auditoría exclusiva antes de autorizar implementación.

No se ejecutaron suites largas porque el target es un plan y el finding se
reproduce con consultas Git deterministas (`git cat-file`, `git log -- <path>` y
`git rev-parse <plan>^`). Todo el trabajo fue CPU-only; no se usó ni consultó
GPU. Este informe es el único archivo creado y no forma parte del target.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R487",
  "scope": "INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "0be3cd6c8b2717efd5ad4b4c24c09c7b55fa1c37",
    "plan_sha256": "1b2d64b70196ef8af46d388ddaf668c025fe4af02a5879238f2594df8eb64fbc"
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
