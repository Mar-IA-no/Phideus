# R494 — Auditoría independiente de la amendment `INVALID_PREPARATION`

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

La amendment es JSON canónico, contiene exactamente las 43 claves autorizadas
y autentica el lineage completo sin promover ninguno de los cinco candidatos
rechazados. Sus bindings documentales, de auditoría e implementación coinciden
con los blobs Git y los archivos físicos; la autoridad final es R493. El
contrato duro, el débito conservador, el draw preservado, la población y el
inventario reproducen el intento v1 terminal pre-truth. El validador productivo
completo aceptó la amendment real en modo recovery con esta auditoría R494
ligada mediante un commit exclusivo. No encontré defectos técnicos.

## Identidad y forma canónica

El target auditado es el commit
`33d04fb6e49ed0a6891d90a92bc8c43bdc7f6d9e`, hijo directo de R493
`dba0e673f25f3a14a2db8ff17c8d39264c8c93d3`. Introduce exclusivamente:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/waves/
  WAVE_60_INVALID_PREPARATION_RECOVERY_V2_AMENDMENT.json
```

El SHA-256 del blob Git, del archivo físico y del binding auditado es
`0b470ed781a841dddda063a2ef40d4bbadea8896d052d778994184c8befaa6eb`.
El archivo tiene 30.300 bytes, parsea como un único objeto JSON y su
serialización `sort_keys=True`, indentada a dos espacios y terminada en newline
reproduce exactamente esos bytes. El keyset top-level contiene 43 claves, sin
faltantes ni extras respecto del schema productivo
`wave60-invalid-preparation-recovery-amendment-v1`
(`prepare_wave56_fresh.py:5011-5083`).

Leí completa la amendment, el validador productivo, los planes de recuperación
R478/R479, los planes R489/R491 y las auditorías R491–R493. La cadena inmediata
es lineal y exclusiva: R492 `9eeb7b0…` → implementación `7d8143d…` —sólo
preparer/test— → R493 `dba0e67…` —sólo su informe— → amendment `33d04fb…`
—sólo este JSON—.

## Lineage histórico y bindings

La amendment conserva la historia desde R477 hasta R493. Los 26 objetos
top-level que ligan explícitamente `commit + path + sha256` satisfacen en todos
los casos:

```text
sha256(blob Git del commit, path)
  == sha256(archivo físico)
  == sha256 del binding
```

Los parents directos, scopes, targets, verdicts, conteos y commits exclusivos
son recorridos por el validador, no inferidos de la prosa. El sufijo de
implementación contiene las trece auditorías R481–R493 con sus resultados
reales: siete `REVISE` y seis `PASS`. R493 es la única autoridad positiva final;
R481, R483, R485, R489 y R491 permanecen rechazadas según sus informes.

Las seis implementaciones declaradas cambian exactamente los dos paths
permitidos. En cada candidato, `old_sha256` coincide con R475 y `new_sha256`
con su propio commit. Las primeras cinco quedan históricas; el aceptado es:

```text
commit       7d8143deb283928faa82e99e390f79d16f301867
audit_commit dba0e673f25f3a14a2db8ff17c8d39264c8c93d3
audit_id     R493
scope        INVALID_PREPARATION_RECOVERY_IMPLEMENTATION
```

Sus blobs de preparer/test coinciden además con el filesystem en
`3d0532cd840461fa07ce1fccc81d5c1b085b8c00397f6162c88f435e0da23ac9`
y `12611bd90e13a5e1ff7d601b4b654802e021f90b9f3c4be8fe9f38183e178e9e`.

## Contrato duro, débito y frontera científica

`hard_set_contract` contiene exactamente las nueve claves aprobadas. La cadena
física comprobada es manifest source-law v2 → record exacto de
`source_law_request.json` → alias único `wave59_config_snapshot.json` → path y
SHA del snapshot Wave 59 → `hard_set_tau=0.5`. El validador invoca la autoridad
source-law real antes de devolver el threshold; no usa fallback ni un valor
sólo incorporado al código (`prepare_wave56_fresh.py:1434-1534`).

`unledgered_preparation_debit` es exactamente:

```text
seconds                            60.0
observed_external_wall_seconds     48.51
regime                             CONSERVATIVE_UNSIGNED_PREPARATION_DEBIT
observed_external_record_authority TRANSCRIPT_ONLY_NOT_SIGNED_LEDGER
applied_once                       true
```

Esto mantiene separado el débito conservador del tiempo externo observado y
evita presentarlo como ledger firmado. La config canónica continúa sin
`hard_set_tau`; la extensión sólo puede entrar en la vista efímera del
materializador y en la provenance autenticada.

La source law permanece anclada a R475. Módulo, runner y worker son
Git/físico-exactos en:

```text
46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa
c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7
```

No cambian draw, modelo, features, políticas, thresholds HGB, roster,
estimandos, bootstrap, penalty, seeds, splits ni presupuesto. La amendment no
decide `GO/NO-GO` científico.

## Origen físico, preservación y población

El mapa `preserved_draw_sha256` contiene exactamente 17 archivos: escrow,
pre-generation freeze, manifest y los catorce archivos declarados por el
manifest del benchmark. Cada SHA coincide con
`attempt_v1/primary/failed_preparation`; el conjunto es closed-world. Escrow,
freeze y manifest reproducen además los cuatro hashes de `escrow_origin`.

`origin_inventory` tiene exactamente 43 entradas y es idéntico, objeto por
objeto, a `physical_tree_inventory(attempt_v1/primary)`: paths, tipos, bytes,
SHA, modos y ownership coinciden. El validador confirmó ausencia de symlinks,
hardlinks y nodos adicionales, el draw anidado root-owned modo `0700`, y los
modos requeridos de escrow, freeze y manifest. El v1 no fue modificado.

La población fue recomputada desde los tres JSONL sealed. Train, val y lockbox
coinciden exactamente en:

```text
rows                                                   4992
total_unique_pair_tokens                               1152
eligible_unique_pair_tokens                             768
out_of_catalog_unique_pair_tokens                       384
noncanonical_unique_pair_tokens                         192
eligible_intersection_noncanonical_unique_pair_tokens   192
```

El predicado conserva `is_out_of_catalog=false`, población de calibración
`canonical_preserving` y filtrado de filas antes de deduplicar pair tokens.

El paquete terminal verificó `PAIR_ABORTED_PRE_TRUTH`, primary y replay
`INVALID_PREPARATION`, `truth_accessed=false`, `recovery_allowed=true` y
`last_complete_phase=INITIALIZED`, junto con las attestations Ed25519 y los
bindings de failure. Conforme al contrato ejecutable, la fuente CLI canónica
en modo recovery es la raíz `attempt_v1/primary`; el validador resuelve y
devuelve internamente su draw `failed_preparation` como fuente reutilizable
(`prepare_wave56_fresh.py:4827-5008,5826-5838`).

## Validación productiva integral

Para romper únicamente la dependencia legítima de la amendment con su propia
auditoría, escribí este informe y construí con plumbing Git un commit object
directo hijo de `33d04fb…`, sin mover HEAD, branch ni refs. Un índice temporal
bajo `/mnt/m2-1TB` partió del árbol de la amendment y añadió sólo este path. Se
verificó antes de invocar el código que:

- el parent del objeto es exactamente `33d04fb…`;
- su diff contiene exclusivamente este informe;
- el blob del objeto, el archivo físico y `amendment_audit_sha256` coinciden;
- el bloque normativo declara R494, scope y target exactos, PASS y `0/0/0`.

Con ese binding construí en memoria una config v2 con las doce claves exactas
de `attempt.recovery`, R495 reservado para la auditoría de config, source law
R475 y preparer/test R493. El `execution_contract.git_commit` quedó ligado al
commit object R494. Mediante un gitdir e índice efímeros —sin modificar el
repositorio ni sus refs— ejecuté
`_validate_wave60_invalid_preparation_recovery_amendment()` completo en modo
`recovery` sobre el intento terminal real. El recorrido aceptó la amendment y
devolvió como `reuse_source` exactamente
`attempt_v1/primary/failed_preparation`; no se limitó al helper aislado ni a la
lectura del JSON.

Todo se ejecutó CPU-only con `CUDA_VISIBLE_DEVICES=''`; no usé ni consulté GPU.
No se ejecutó draw, entrenamiento, scoring ni materialización. Los recursos
temporales fueron inventariados y eliminados por sus paths exactos. Este
informe es el único archivo creado.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R494",
  "scope": "INVALID_PREPARATION_RECOVERY_AMENDMENT",
  "target": {
    "amendment_sha256": "0b470ed781a841dddda063a2ef40d4bbadea8896d052d778994184c8befaa6eb"
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
