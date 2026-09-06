# Ola 60 — resolución de R478 para recuperar `INVALID_PREPARATION`

> **Estado:** `PRE-IMPLEMENTATION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Plan rechazado:** commit `ba193d52cd46f23d57c1a0b811433d2cbcfedb6d`,
> SHA-256 `0ae881499221845d309e66dba9da42821b9e841002f8f8c4329790e87896965f`
> **Auditoría R478:** commit `979c835bc2b2f08182e23e919f265f4fe2bc480a`,
> SHA-256 `ef4be8f02f2bf40f005c63c9cf61e5932ddd52bb36fb343aab15f457dceb1eb8`
> **Dictamen R478:** `REVISE / 0 HIGH + 2 MEDIUM + 0 LOW`

## 1. Decisión de diseño

Se aceptan los dos findings de R478. La resolución adopta la variante mínima
propuesta por la auditoría:

- `src/geometria_proporcional/wave60_frozen_policy_transport.py`,
  `run_wave60_frozen_policy_transport.py` y `_wave60_phase_worker.py` permanecen
  byte a byte bajo la autoridad de implementación R475;
- la source law v2 y su auditoría R476 permanecen inalteradas;
- la corrección ejecutable se limita a
  `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- la única otra modificación de implementación será
  `tests/test_wave60_frozen_policy_transport.py`;
- `hard_set_tau=0.5` y el débito conservador de 60 segundos serán extensiones
  tipadas de la amendment, no defaults silenciosos ni campos nuevos de la config
  científica base.

Así se separan dos autoridades: R475 continúa acreditando la ley congelada, el
scorer, el runner y el worker; una auditoría nueva acreditará exclusivamente el
adaptador de recuperación del preparador y sus pruebas.

## 2. Schema exacto de amendment

La amendment usará el schema nuevo:

```text
wave60-invalid-preparation-recovery-amendment-v1
```

Su keyset top-level será exactamente:

```text
schema_version
status
recovery_kind
prior_attempt_container
prior_pair_failure_sha256
prior_config_audit
rejected_plan
rejected_plan_audit
resolution_plan
resolution_plan_audit
recovery_implementation
hard_set_contract
unledgered_preparation_debit
escrow_origin
preserved_draw_sha256
population_contract
origin_inventory
```

Los campos escalares quedan congelados así:

```text
status = APPROVED
recovery_kind = INVALID_PREPARATION
prior_attempt_container =
  data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v1
prior_pair_failure_sha256 =
  05ede417b1f856488c1796210029aa74c74211f3f14858764d9705cbb0b3563d
```

`prior_config_audit` será exactamente:

```json
{
  "commit": "2a10b6cb5a88fd2af4ca2f5f8230a296f59c8948",
  "path": "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/477_wave60_frozen_policy_transport_config_audit.md",
  "sha256": "2c0c8bd9defe0fbc4901b5bc5130e2d27de1a5a47fd61331e2eff40682f5556d"
}
```

`rejected_plan` tendrá exactamente `commit`, `path`, `sha256`; sus valores serán
los del encabezado y
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_PLAN.md`.
`rejected_plan_audit` tendrá
exactamente `commit`, `path`, `sha256`, `audit_id`, `verdict` y `findings`, con
el path
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/478_wave60_invalid_preparation_recovery_plan_audit.md`,
`R478`, `REVISE` y `{HIGH: 0, MEDIUM: 2, LOW: 0}`.

`resolution_plan` tendrá exactamente `commit`, `path`, `sha256` y apuntará a
este documento. `resolution_plan_audit` tendrá exactamente `commit`, `path`,
`sha256`, `audit_id`, `verdict` y `findings`; su path será
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/479_wave60_invalid_preparation_recovery_r478_resolution_plan_audit.md`
y deberá declarar `R479`, `PASS` y `{high: 0, medium: 0, low: 0}`.

`recovery_implementation` tendrá exactamente:

```text
commit
audit_commit
audit_path
audit_sha256
audit_id
scope
changed_sources
unchanged_source_law_sources
```

Sus valores fijos serán `audit_id=R480` y
`scope=INVALID_PREPARATION_RECOVERY_IMPLEMENTATION`. `changed_sources` contendrá
exactamente las claves `preparer` y `test`, cada una con `path`, `old_sha256` y
`new_sha256`:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

`unchanged_source_law_sources` será la lista ordenada exacta:

```text
src/geometria_proporcional/wave60_frozen_policy_transport.py
experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
experiments/geometria_proporcional/_wave60_phase_worker.py
```

El `audit_path` de R480 será
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/480_wave60_invalid_preparation_recovery_implementation_audit.md`.
Cada uno de esos tres blobs deberá coincidir con el commit R475
`9f1a229d9c0ccb5e46b921e6c92281becc317139` y con el hash de la config v1.

`hard_set_contract` tendrá exactamente:

```json
{
  "hard_set_tau": 0.5,
  "authority": "wave59_config_snapshot_bound_by_source_law_v2",
  "source_path": "data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1/config.snapshot.json",
  "source_sha256": "f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6"
}
```

El validador abrirá esa fuente pública, comprobará hash y valor `0.5`, y sólo
entonces construirá una vista efímera de config para el materializador. No se
admite `dict.get(..., 0.5)` ni ningún fallback implícito.

`unledgered_preparation_debit` tendrá exactamente:

```json
{
  "seconds": 60.0,
  "regime": "CONSERVATIVE_UNSIGNED_PREPARATION_DEBIT",
  "observed_external_wall_seconds": 48.51,
  "observed_external_record_authority": "TRANSCRIPT_ONLY_NOT_SIGNED_LEDGER",
  "applied_once": true
}
```

`escrow_origin`, `preserved_draw_sha256`, `population_contract` y
`origin_inventory` conservan la forma cerrada del schema Wave 60 anterior, pero
se calculan respecto del draw físico en `primary/failed_preparation/`; el
inventario sigue cubriendo la raíz primaria terminal completa.

La amendment vivirá exactamente en
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_V2_AMENDMENT.json`.
Su auditoría R481 vivirá en
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/481_wave60_invalid_preparation_recovery_amendment_audit.md`
y usará el bloque canónico `wave60-audit-authority-v1`, scope
`INVALID_PREPARATION_RECOVERY_AMENDMENT`, target
`{amendment_sha256: <hash físico>}`, PASS y cero findings.

## 3. Schema exacto de `attempt.recovery`

La config v2 conserva el schema existente
`wave60-pretruth-recovery-v1`. No se agregan claves al módulo científico. El
keyset seguirá siendo exactamente:

```text
schema_version
prior_attempt_container
prior_pair_failure_sha256
prior_config_audit_commit
prior_config_audit_path
prior_config_audit_sha256
amendment_path
amendment_sha256
amendment_audit_commit
amendment_audit_path
amendment_audit_sha256
preserved_draw_sha256
```

La config liga transitivamente plan, R478, resolución, R479, implementación,
R480, hard threshold y débito mediante `amendment_sha256`; liga el PASS de la
amendment mediante el trío `amendment_audit_*`. No duplica esos campos ni crea
dos fuentes de autoridad.

## 4. Partición exacta de blobs

El validador final de config sustituirá el roster monolítico por dos conjuntos:

### 4.1 Autoridad R475, inalterada

```text
src/geometria_proporcional/wave60_frozen_policy_transport.py
experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
experiments/geometria_proporcional/_wave60_phase_worker.py
```

Para cada path se exige simultáneamente hash de blob en `9f1a229…`, hash físico
y `source_sha256` de config v2 iguales. `implementation_binding` top-level sigue
siendo exactamente R475, porque esa autoridad es la que exige y reconoce la
source law v2.

### 4.2 Autoridad de recuperación R480

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

Para cada path se exige hash old contra `9f1a229…`, hash new contra el commit de
implementación, hash físico y `source_sha256` v2 iguales. El commit de
implementación no puede tocar ningún otro path y R480 será un commit exclusivo
que lo audite.

La prueba integrada deberá llamar `validate_source_authority()` con la config
v2 y source law v2 reales y demostrar que el cruce de cualquiera de ambas
autoridades se rechaza.

## 5. Resolución física de `failed_preparation`

El preparador sólo resolverá el draw archivado si se cumplen todos estos hechos:

1. `validate_pair_failure_package()` devuelve `PAIR_ABORTED_PRE_TRUTH`,
   `any_truth_accessed=false` y `recovery_allowed=true`;
2. primary y replay son `INVALID_PREPARATION`;
3. el `FAILURE.json` primario coincide con su attestation y declara
   `last_complete_phase=INITIALIZED` por su inventario;
4. `primary/failed_preparation` es un directorio físico root-owned `0700`;
5. escrow `0600`, freeze, manifest y benchmark viven allí, sin aliases,
   symlinks, hardlinks ni nodos especiales;
6. el inventario completo primario coincide con `origin_inventory` y el mapa de
   draw es closed-world respecto del manifest.

La CLI seguirá recibiendo la raíz primaria v1. `validate_invocation()` podrá
admitir el escrow anidado sólo cuando config y amendment físicas anticipen este
schema y prior exactos. Después del preflight, `validate_recovery_amendment()`
resolverá el subdirectorio y `validate_reused_escrow()` recibirá exclusivamente
esa raíz validada. La copia del benchmark conservará bytes y modos y exigirá
inodos nuevos.

## 6. Aplicación trazable de `hard_set_tau`

En recovery primaria y replay v2, inmediatamente antes de
`materialize_prepared_bundles()`, el preparador construirá:

```text
materializer_config = copia(config)
materializer_config["hard_set_tau"] =
  amendment.hard_set_contract.hard_set_tau
```

La config canónica y su snapshot no se mutan. La extensión se registrará dentro
de `recovery_provenance.contract_extensions`, repetida idénticamente en
`generation_receipt.json`, `preparation_freeze.json` y
`preparation_receipt.json`, y ligada por `recovery_amendment.json` y la
attestation de preparación. El contenido exacto será:

```json
{
  "recovery_kind": "INVALID_PREPARATION",
  "hard_set_tau": 0.5,
  "unledgered_preparation_debit_seconds": 60.0
}
```

Una preparación Wave 60 sin amendment mantiene el comportamiento v1 histórico;
no obtiene el threshold por default. Ningún otro schema recibe esta extensión.

## 7. Ledger sin doble conteo

Para la primaria recovery v2:

```text
primary_prior = 60.0
primary_cumulative = 60.0 + primary_duration
```

Para el replay v2:

```text
replay_prior = primary_cumulative
replay_cumulative = replay_prior + replay_duration
```

Para una recuperación posterior desde v2 o superior que ya posea receipts
firmados:

```text
next_prior = durable_seconds autenticados del par predecesor
```

Ese `durable_seconds` ya contiene el débito inicial porque quedó incorporado a
los ledgers v2. No se vuelven a sumar 60 segundos. La rama conservadora sólo se
activa cuando el prior firmado es exactamente el v1 indicado, ambos roots son
`INVALID_PREPARATION`, no existe ningún receipt de preparación y la amendment
nueva fija `applied_once=true`. Se rechazan cero, negativo, valor distinto de
60, receipts parciales, mezcla con el régimen firmado y segundo débito.

## 8. Lineage Git ejecutable

La cadena exacta será:

```text
2a10b6c  R477 / config v1 audit
  → ba193d5  plan rechazado
  → 979c835  R478 REVISE
  → resolution_plan_commit
  → R479 PASS resolution-plan audit
  → recovery_implementation_commit
  → R480 PASS implementation audit
  → amendment_commit
  → R481 PASS amendment audit
  → config_v2_commit
  → R482 PASS config audit / HEAD de ejecución
```

Cada flecha es parent directo y cada commit es exclusivo:

- resolución: sólo este documento;
- R479: sólo su informe;
- implementación: exactamente preparador + test;
- R480: sólo su informe;
- amendment: sólo su JSON;
- R481: sólo su informe;
- config: sólo la config canónica;
- R482: sólo su informe.

La config v2 seguirá en
`experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json`;
R482 vivirá en
`Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/482_wave60_frozen_policy_transport_v2_config_audit.md`
y conservará el scope canónico `CONFIG` con target
`config_commit + config_sha256`.

El validador nuevo comprobará la cadena completa desde R477. La regla legacy
`parent(amendment)=R477` continúa para el schema anterior; este schema nuevo la
reemplaza por `parent(amendment)=R480` y verifica los eslabones intermedios
contra los bindings de la amendment. No se relaja ningún lineage genérico.

## 9. Matriz mínima de pruebas

Además de la matriz del plan rechazado, la implementación debe cubrir:

1. source law v2 real aceptada con top-level R475 y rechazada si se cruza R475
   con R480;
2. blobs `src/runner/worker` exactos en R475 y `preparer/test` exactos en el
   commit de recuperación;
3. schema y keysets exactos de amendment y `attempt.recovery`;
4. lineage completo positivo y rechazo de cada parent saltado;
5. amendment con threshold ausente, distinto, no finito o no coincidente con la
   config Wave 59 física;
6. preparación Wave 60 real con materializador real y extensión trazada;
7. origin anidado válido y rechazo de aliases, symlinks, hardlinks, modos,
   owners, bytes, firma o inventario alterados;
8. `primary_prior=60`, replay encadenado, posterior sin doble débito y rechazo de
   mezcla de regímenes;
9. escrow/benchmark v2 byte-exactos respecto del draw v1 y con inodos nuevos;
10. v1 completo inalterado antes y después de pruebas;
11. regresión focal y Wave 56–60 CPU-only, con RSS y swap observados.

Los fixtures podrán sustituir únicamente fronteras costosas no sometidas por la
prueba concreta. La prueba positiva del defecto deberá ejecutar el
materializador real. La prueba de source law deberá usar la autoridad v2 real.

## 10. Secuencia autorizable

Este documento no autoriza todavía implementación ni recovery canónica. Primero
requiere una auditoría independiente R479 con target exacto
`resolution_plan_commit + resolution_plan_sha256`. Sólo un `PASS` con cero
findings habilita el commit de dos paths. Después se aplican sucesivamente R480,
R481 y R482. Ninguna de esas autoridades decide el resultado científico ni
`GO/NO-GO`.
