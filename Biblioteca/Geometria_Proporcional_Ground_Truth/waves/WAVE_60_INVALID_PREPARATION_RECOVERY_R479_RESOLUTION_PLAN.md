# Ola 60 — resolución focal de R479

> **Estado:** `PRE-IMPLEMENTATION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Plan anterior:** commit `0c21db44ebb428647dbcd7921713b593895fc07a`,
> SHA-256 `39f13cd749083ed581b967166e823ff66e872b9519013c3f1e66523ee58e9c8c`
> **Auditoría R479:** commit `3f20c79db2311cfefebc90dd049b4862d2a04a11`,
> SHA-256 `4089ef7cd4812714fa731efae45719fc1685dafb767e1e92c26574abdddb9fa4`
> **Dictamen:** `REVISE / 0 HIGH + 1 MEDIUM + 0 LOW`

## 1. Alcance de esta resolución

R479 confirmó que la resolución anterior cierra los dos findings R478: la
partición R475/recuperación es correcta, el parche puede limitarse a preparador
y test, el origen anidado es recuperable, el débito one-shot de 60 segundos está
bien definido y el lineage es realizable. Esta resolución conserva todo ese
diseño y corrige únicamente la autoridad transitiva de `hard_set_tau`.

La afirmación final que debe probarse antes de materializar es:

```text
source-law manifest v2
  → source_law_request.json exacto
  → alias wave59_config_snapshot.json
  → path canónico + SHA-256
  → snapshot físico
  → hard_set_tau = 0.5
```

Ningún eslabón puede inferirse de otro ni sustituirse por un valor incorporado
en código.

## 2. `hard_set_contract` definitivo

El objeto `hard_set_contract` de la futura amendment tendrá exactamente estas
nueve claves:

```text
authority
source_authority_path
source_authority_manifest_sha256
source_law_request_relative
source_law_request_sha256
request_alias
source_path
source_sha256
hard_set_tau
```

Sus valores serán exactamente:

```json
{
  "authority": "wave59_config_snapshot_transitively_bound_by_source_law_v2",
  "source_authority_path": "data/geometria_proporcional/wave60_frozen_policy_transport_source_law_v2",
  "source_authority_manifest_sha256": "9c69745a0661994049530e917e59e0a68b99d5f15a7c0d2bae3da66f1df43dc2",
  "source_law_request_relative": "source_law_request.json",
  "source_law_request_sha256": "983af4bb024f966b60b4e79fe753ebd38665747eab21b95db27ec3f0ab889a99",
  "request_alias": "wave59_config_snapshot.json",
  "source_path": "data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1/config.snapshot.json",
  "source_sha256": "f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6",
  "hard_set_tau": 0.5
}
```

Este objeto reemplaza el `hard_set_contract` de cuatro claves del plan anterior.
El resto de los contratos de amendment se conserva salvo los nombres necesarios
para ligar la historia adicional de R479 y esta resolución.

## 3. Orden obligatorio de validación

Antes de construir `materializer_config`, el preparador deberá:

1. resolver `source_authority_path` dentro del repositorio, sin traversal,
   symlink ni alias;
2. verificar que `source_authority_manifest.json` es el manifest físico ligado
   por `source_authority_manifest_sha256`;
3. validar la source law v2 real mediante `validate_source_authority()`, usando
   la autoridad top-level R475 de la config recuperada;
4. comprobar que el manifest tiene como registro exacto de
   `source_law_request.json` el hash y metadata físicos observados, incluido
   `source_law_request_sha256`;
5. exigir que el request físico tenga hash exacto y que:

```text
request.source_paths[request_alias] == source_path
request.source_sha256[request_alias] == source_sha256
```

6. resolver `source_path` canónicamente, comprobar `source_sha256`, parsear el
   snapshot y exigir `hard_set_tau == 0.5` finito;
7. sólo entonces copiar la config en memoria y agregar ese valor para la llamada
   al materializador.

La config canónica, su snapshot y la source law v2 no se modifican. La extensión
continúa registrada en `recovery_provenance.contract_extensions` y cubierta por
la amendment copiada y la attestation de preparación, como fijó el plan R478.

## 4. Rechazos mínimos específicos

La suite añadirá negativos independientes para:

1. alias `wave59_config_snapshot.json` ausente o adicionalmente duplicado;
2. alias que apunta a otro path físico con el mismo contenido;
3. hash del alias cruzado con otro input del request;
4. request físico válido pero no ligado por el manifest v2;
5. manifest alternativo, reempaquetado o no ligado por la config;
6. snapshot con hash distinto, path traversal/symlink o `hard_set_tau` distinto,
   ausente, no finito o de tipo no numérico.

Los fixtures de tampering usarán copias temporales. No se modifica la autoridad
v2 canónica ni el draw v1.

La positiva deberá atravesar manifest y request v2 reales antes de ejecutar el
materializador real. No podrá mockear `validate_source_authority`, el lector del
request, el hash del snapshot ni el materializador en esa misma prueba.

## 5. Schema definitivo de amendment

El schema sigue siendo:

```text
wave60-invalid-preparation-recovery-amendment-v1
```

El keyset top-level definitivo será:

```text
schema_version
status
recovery_kind
prior_attempt_container
prior_pair_failure_sha256
prior_config_audit
rejected_plan
rejected_plan_audit
r478_resolution_plan
r478_resolution_plan_audit
r479_resolution_plan
r479_resolution_plan_audit
recovery_implementation
hard_set_contract
unledgered_preparation_debit
escrow_origin
preserved_draw_sha256
population_contract
origin_inventory
```

`rejected_plan` y `rejected_plan_audit` siguen ligando `ba193d5` y R478.
`r478_resolution_plan` y `r478_resolution_plan_audit` ligan `0c21db4` y R479,
con verdict `REVISE` y findings `{high: 0, medium: 1, low: 0}`.
`r479_resolution_plan` liga este documento. `r479_resolution_plan_audit` tendrá
exactamente `commit`, `path`, `sha256`, `audit_id`, `verdict`, `findings`; su
path será
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/480_wave60_invalid_preparation_recovery_r479_resolution_plan_audit.md`
y deberá declarar `R480`, `PASS` y `{high: 0, medium: 0, low: 0}`.

`attempt.recovery` conserva sin cambios el schema y las doce claves fijadas por
la resolución anterior. La config liga todo este objeto por hash y liga su
auditoría final por los campos `amendment_audit_*`.

## 6. Autoridades futuras renumeradas

Por la incorporación de esta resolución, las autoridades pendientes pasan a
ser:

- R480: auditoría PASS de este plan;
- R481: auditoría de implementación, scope
  `INVALID_PREPARATION_RECOVERY_IMPLEMENTATION`;
- R482: auditoría de amendment, scope
  `INVALID_PREPARATION_RECOVERY_AMENDMENT`;
- R483: auditoría de config v2, scope `CONFIG`;
- R484: auditoría de resultados, si el par llega a terminal evaluable.

Los paths futuros serán:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
  480_wave60_invalid_preparation_recovery_r479_resolution_plan_audit.md
  481_wave60_invalid_preparation_recovery_implementation_audit.md
  482_wave60_invalid_preparation_recovery_amendment_audit.md
  483_wave60_frozen_policy_transport_v2_config_audit.md
  484_wave60_frozen_policy_transport_result_audit.md
```

La amendment seguirá en:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/waves/
  WAVE_60_INVALID_PREPARATION_RECOVERY_V2_AMENDMENT.json
```

## 7. Lineage definitivo

```text
R477 2a10b6c
  → plan rechazado ba193d5
  → R478 979c835
  → resolución R478 0c21db4
  → R479 3f20c79
  → esta resolución
  → R480 PASS
  → implementación preparer+test
  → R481 PASS
  → amendment
  → R482 PASS
  → config v2
  → R483 PASS / HEAD de ejecución
```

Cada commit será hijo directo y exclusivo. El validador del schema nuevo
recorrerá todos los eslabones y comprobará hashes, blobs, scopes, targets,
verdicts y conteos. La regla legacy queda intacta para el schema anterior.

## 8. Criterio de autorización

Esta resolución no habilita todavía el parche ni la recovery canónica. Requiere
R480 `PASS` con cero findings sobre el target exacto
`plan_commit + plan_sha256`. Todo el resto de la resolución R478 continúa
vigente en cuanto no sea reemplazado expresamente aquí. Ninguna auditoría
declara `GO/NO-GO` científico.
