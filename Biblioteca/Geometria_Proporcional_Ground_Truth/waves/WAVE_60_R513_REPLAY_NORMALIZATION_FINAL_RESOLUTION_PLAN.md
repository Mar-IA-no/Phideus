# Ola 60 — resolución final de los findings R513

> **Estado:** `PRE-IMPLEMENTATION / R513-REVISE / PAIR-COMPLETE-IMMUTABLE / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **R512:** commit `bb300df7bfb47021a30072a209054fe4c5be4efb`, SHA-256 `faa752fa22fead8609feef293d9c791262fe59bf6a6c51b8fcd6ff173755b57e`
> **R513:** commit `b522debe93ee55b7abba9ec04e5de67669c6231b`, SHA-256 `dbe7b8cda6a29fd9c50dbe6ee12682b5d9245522828f0bd84b8b7fe0d8c7a828`, `REVISE 0/2/1`

## 1. Alcance acumulativo

Este documento hereda las reglas científicas y físicas de R510 y R512, pero
los reemplaza como autorización de implementación. Resuelve R513-01 fijando
paths, bindings y targets exactos; R513-02 mediante una autoridad normativa de
metadata para los tres manifests autorreferenciales; y R513-03 separando la
corrección candidata de su futura activación auditada.

El intento v4, la config R508 y sus ocho sources permanecen byte-exactos. No
se permite recovery, rerun, forward, training ni recomputación de métricas.

## 2. Cadena definitiva y paths exactos

```text
R513 REVISE
  -> R514 este plan, exclusivo, parent R513
  -> R515 auditoría de plan, exclusiva, parent R514
  -> R516 implementación checker+test, exclusiva, parent R515
  -> R517 auditoría de implementación, exclusiva, parent R516
  -> R518 corrección candidata JSON, exclusiva, parent R517
  -> R519 auditoría de artefacto, exclusiva, parent R518
  -> documentación y wiki
```

Los únicos paths futuros autorizados son:

```text
R515: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/515_wave60_r513_replay_normalization_resolution_plan_audit.md
R516: experiments/geometria_proporcional/adjudicate_wave60_v4_result.py
      tests/test_wave60_v4_result_adjudication.py
R517: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/517_wave60_r509_replay_normalization_implementation_audit.md
R518: Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json
R519: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/519_wave60_v4_replay_normalization_correction_audit.md
```

Cada commit debe tener como diff exactamente su pathset declarado y ser hijo
directo del eslabón anterior. Los documentos físicos deben coincidir con su
blob Git. Cero o múltiples parents, paths extra, renames, deletes, copias o
blobs divergentes son rechazo.

## 3. Autoridades JSON futuras exactas

Cada informe R515/R517/R519 contiene exactamente un bloque fenced `json`. Su
keyset superior es exactamente:

```text
schema_version, audit_id, scope, target, technical_verdict,
findings, files_modified, gpu_used_or_queried
```

En los tres casos `schema_version=wave60-audit-authority-v1`,
`technical_verdict=PASS`, findings exactos
`{high:0, medium:0, low:0}`, `files_modified=false` y
`gpu_used_or_queried=false`.

### R515

```json
{
  "audit_id": "R515",
  "scope": "R513_REPLAY_NORMALIZATION_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "<commit R514 de 40 hex>",
    "plan_sha256": "<SHA-256 físico/blob de este plan>"
  }
}
```

El keyset de `target` es exactamente `plan_commit, plan_sha256`.

### R517

```json
{
  "audit_id": "R517",
  "scope": "R509_REPLAY_NORMALIZATION_RESOLUTION_IMPLEMENTATION",
  "target": {
    "implementation_commit": "<commit R516 de 40 hex>",
    "files": {
      "experiments/geometria_proporcional/adjudicate_wave60_v4_result.py": "<SHA-256>",
      "tests/test_wave60_v4_result_adjudication.py": "<SHA-256>"
    }
  }
}
```

El keyset de `target` es exactamente `implementation_commit, files`; el keyset
de `files` son exactamente esos dos paths, ordenados canónicamente al
serializar.

### R519

```json
{
  "audit_id": "R519",
  "scope": "WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION",
  "target": {
    "artifact_commit": "<commit R518 de 40 hex>",
    "artifact_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json",
    "artifact_sha256": "<SHA-256 físico/blob>"
  }
}
```

El keyset de `target` es exactamente
`artifact_commit, artifact_path, artifact_sha256`.

El parser es fail-closed ante bloque ausente/duplicado, JSON inválido, key
faltante/extra, tipo incorrecto o divergencia entre JSON y binding físico/Git.

## 4. Keyset exacto de `authority_chain` en R518

`authority_chain` contiene exactamente estas nueve claves:

```text
r509_result_audit
r510_base_plan
r511_base_plan_audit
r512_first_resolution_plan
r513_first_resolution_plan_audit
r514_final_resolution_plan
r515_final_resolution_plan_audit
r516_implementation
r517_implementation_audit
```

Los cuatro planes/implementaciones documentales usan exactamente
`{commit,path,sha256}`. Cada auditoría usa exactamente
`{commit,path,sha256,authority_json}`; `authority_json` es el bloque completo
validado, no un resumen. `r516_implementation` usa exactamente
`{commit,files}`, donde `files` tiene los dos paths y hashes definidos arriba.

Valores históricos ya congelados:

- R509: commit `92305f4e54e72ee78924ca4b51ae5889369d805b`, path
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/509_wave60_frozen_policy_transport_v4_result_or_terminal_audit.md`, SHA
  `006a43e9257a340b162583bcf1190cf34a27085186643a1ea37d987b4fa45e28`;
- R510: commit `fa6ee25b359e06c9bef2ce4ec08768d8f3a46ff8`, path
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md`, SHA
  `31dc9a49e5b11df9d369537878651da788c0607e80c9d0e8fffd8093a584acee`;
- R511: commit `86405020425f7c2310a68d66a215e3a35a00e982`, path
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/511_wave60_r509_replay_normalization_resolution_plan_audit.md`, SHA
  `bb80e4efd3f8dc896ac20b83611ed7c46d25e2b3a229a2c358a1c694b70b789c`;
- R512: commit `bb300df7bfb47021a30072a209054fe4c5be4efb`, path
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md`, SHA
  `faa752fa22fead8609feef293d9c791262fe59bf6a6c51b8fcd6ff173755b57e`;
- R513: commit `b522debe93ee55b7abba9ec04e5de67669c6231b`, path
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/513_wave60_r511_replay_normalization_resolution_plan_audit.md`, SHA
  `dbe7b8cda6a29fd9c50dbe6ee12682b5d9245522828f0bd84b8b7fe0d8c7a828`.

Sus JSON conservan exactamente scopes, targets, verdicts y findings ya
publicados: R509 `REVISE 0/1/1`, R511 `REVISE 0/2/1`, R513 `REVISE 0/2/1`.

## 5. Binding físico completo del attempt

Los manifests internos comprometen 138 paths no autorreferenciales. Para esos
paths se exige roster, bytes, SHA-256, UID, GID y mode exactos. Este plan añade
la autoridad normativa que faltaba para los tres self-manifests:

| Path | bytes | uid | gid | mode | SHA-256 |
|---|---:|---:|---:|---:|---|
| `primary/artifact_manifest.json` | 17792 | 0 | 0 | `0444` | `a96497d5a06e8ec23b7844aa13a2ef7455ef0a8bf6b410980a96ef8ebf7ed982` |
| `replay/artifact_manifest.json` | 18047 | 0 | 0 | `0444` | `4ffabc2bd54de623820cd373b84ad4f95e49ff45b7ddc48da8eaa894f7dc7eb0` |
| `pair/artifact_manifest.json` | 2448 | 0 | 0 | `0444` | `4a51993c420a96f9f8283a3ced5e9dee279097ce37453fbd2b7480580805e686` |

El checker autentica este plan antes de usar la tabla. Luego exige:

- attempt/primary/replay/pair como directorios físicos canónicos;
- exactamente tres hijos root y 65/66/10 archivos;
- 141 archivos regulares, `st_nlink=1`, 141 pares `(st_dev,st_ino)` únicos;
- cero symlinks, nodos especiales, extras o faltantes;
- metadata y SHA exactos para los 138 paths manifestados;
- tabla exacta anterior para los tres self-manifests.

Los secretos se hashean como bytes opacos, sin interpretación semántica. Un
test modifica específicamente mode y owner/group de un self-manifest en una
copia/fixture temporal y debe ser rechazado, además de los ataques de hardlink,
symlink, nodo especial, roster y metadata ya exigidos por R512.

## 6. Contenido y estado exactos de R518

El keyset superior de R518 es exactamente:

```text
schema_version, artifact_status, activation_condition, authority_chain,
attempt_binding, config_binding, source_bindings, original_observation,
normalized_evidence, conditional_corrected_view, metrics_binding,
limitations, scientific_decision, decision_authority,
architecture_promoted, gpu_used_or_queried
```

Valores de estado:

- `schema_version=wave60-v4-replay-normalization-correction-v1`;
- `artifact_status=CANDIDATE_PENDING_R519_AUDIT`;
- `scientific_decision=null`;
- `decision_authority=user`;
- `architecture_promoted=false`;
- `gpu_used_or_queried=false`.

`activation_condition` tiene exactamente:

```json
{
  "required_audit_id": "R519",
  "required_audit_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/519_wave60_v4_replay_normalization_correction_audit.md",
  "required_scope": "WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION",
  "required_verdict": "PASS",
  "required_findings": {"high": 0, "medium": 0, "low": 0},
  "authority_effect": "ACTIVATES_CONDITIONAL_CORRECTED_VIEW"
}
```

R518 no afirma que R519 ya ocurrió. `conditional_corrected_view` contiene el
replay normalizado, condiciones y patrones `false/false` sólo como derivación
candidata. `limitations` separa `original` y `if_activated`; esta última
reemplaza `replay_exact_pending_pair_finalize` por
`replay_exact_adjudicated_by_r509_findings_resolution_chain` sin afirmar dentro
del artefacto que la condición de activación ya se cumplió. La documentación
sólo adopta esa vista después de R519 PASS.

Los subobjetos restantes tendrán keysets declarados como constantes en el
checker y serán validados por `validate_correction_payload()`. Para que esos
constantes no introduzcan criterio nuevo durante implementación, sus keysets
normativos son:

```text
attempt_binding = path, target_sha256, physical_inventory,
                  self_manifest_metadata
target_sha256 = pair/artifact_manifest.json, pair/pair_status.json,
                pair/final_analysis.json, pair/replay_comparison.json,
                primary/artifact_manifest.json, replay/artifact_manifest.json
physical_inventory = primary_files, replay_files, pair_files, total_files,
                     manifested_files, self_manifest_files, regular_files,
                     nlink_one_files, unique_device_inode_pairs,
                     metadata_and_hashes_match
self_manifest_metadata[<cada uno de tres paths>] = bytes, uid, gid, mode, sha256

config_binding = path, commit, physical_sha256, self_binding_sha256, audit
config_binding.audit = audit_id, commit, path, sha256

original_observation = status, replay_exact, mismatches, conditions, patterns,
                       limitations
normalized_evidence = historical_check_count, historical_true_count,
                      normalized_check_count, normalized_all_true,
                      local_generation_receipt_sha256,
                      preparation_freeze_sha256, generation_execution_modes,
                      generation_receipts_equal_except_execution_mode,
                      primary_scientific_hashes, replay_scientific_hashes
conditional_corrected_view = normalized_replay_exact, conditions, patterns
metrics_binding = primary_analysis_sha256, replay_analysis_sha256,
                  r509_numeric_recomputation
r509_numeric_recomputation = actions, metric_arrays, pair_tokens,
                             bootstrap_replicates, r509_report_sha256,
                             values_unchanged
limitations = original, if_activated, replaced
```

`source_bindings` es un mapa exacto de los ocho paths de
`config.source_sha256`, sin otras claves. Los valores normativos son:

| Path | SHA-256 / self-binding |
|---|---|
| `src/geometria_proporcional/wave60_frozen_policy_transport.py` | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |
| `experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py` | `b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261` |
| `experiments/geometria_proporcional/_wave60_phase_worker.py` | `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7` |
| `experiments/geometria_proporcional/prepare_wave56_fresh.py` | `0dd0f3389b2db1011ce95c916a37faf4c3898460c2d30fba8f8339c5075b92c8` |
| `tests/test_wave60_frozen_policy_transport.py` | `d8ca7d06848eb17091e743aaf025abcefa9cc333489b2c6641cd6bab9cab6960` |
| `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/475_wave60_source_law_recovery_implementation_reaudit.md` | `e5c49ca16506469ac5099c2a3f1992819c1a54fed790f1638a82e93b9b1d9996` |
| `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/476_wave60_source_law_recovery_authority_audit.md` | `497bb87f7e3677c4dae11a0281e0362fa4d1ef4d6ba60252f01cdc1f2c0a8a30` |
| `experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json` | `eab40e2d34cfcd532437c5e7567ac94b7988a46afb865bab84728fa90735e810` |

`config_binding` fija path
`experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json`,
commit `b156f6857eaa36edc8bda9e7687de7b8e1ea9721`, SHA físico
`191483d2909c3a95a1e82488e1834b55c849763f55c72549f07f2c4cf81d6416`
y self-binding `eab40e2d...`. Su subobjeto `audit` fija R508, commit
`789c4ea2298fcaba97c9bdecdd1db4360186012c`, path
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/508_wave60_frozen_policy_transport_v4_config_reaudit.md`
y SHA `bf63f5d67285fabd32a7d0adae44a47dbaaa202c33930663f8cd62ed13619f6d`.

Los tests eliminan y agregan una clave en cada nivel para probar cierre
estructural.

## 7. Regla científica y publicación heredadas

Se mantienen sin cambios las ocho guardas semánticas de R510/R512: seis hashes
target R509, config/self-binding y ocho sources R508, terminal COMPLETE, roots
EVALUATED_IMMUTABLE, manifests/firmas/receipts válidos, recomposición exacta de
36 checks con un único mismatch, enlaces locales válidos, freezes exactos,
generation receipts iguales salvo `execution_mode=recovery/replay`, y ocho
artefactos científicos exactos.

La normalización sólo cambia `replay_exact` a `true`. Los patrones permanecen
`incompatibility=false` y `harm=false` por sus respectivos controles; métricas,
intervalos y soportes quedan ligados a los analysis originales y a R509.

R518 se serializa canónicamente y se publica fuera del attempt mediante
creación exclusiva sin overwrite. El checker separa construcción read-only,
validación del payload y publicación. R519 repite la derivación desde los
estados crudos, autentica el commit exclusivo R518 y decide si activa la vista.

## 8. Presupuesto y autoridad decisional

Todo es CPU-only con `CUDA_VISIBLE_DEVICES=''`, wall/RSS/swaps medidos. Hashing,
validación estructural y agregación booleana son tareas nativas de CPU; no se
está sustituyendo training o forward de GPU. No hay promoción arquitectónica
ni `GO/NO-GO`; la autoridad científica sigue siendo de Mariano.
