# Ola 60 — resolución de la cobertura contractual pendiente en R483

> **Estado:** `PRE-CORRECTION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Implementación auditada:** commit
> `5aee5fb6064cdd232620a5eebe9a9e856dae9f0f`
> **Auditoría R483:** commit
> `b944cac9fbcbb3f28e2d7753161c58c978868226`, informe
> `483_wave60_invalid_preparation_recovery_implementation_reaudit.md`, SHA-256
> `7aa50f0fca72b3f796ddd4319f99cf2b9d60c005cbca651ac04c113d3d6be905`
> **Dictamen:** `REVISE / 0 HIGH + 1 MEDIUM + 0 LOW`

## 1. Alcance de la resolución

R483 verificó que el defecto probatorio central de R481 quedó cerrado. La prueba
positiva atraviesa `run_preparation_transaction()`, usa el validador duro y el
materializador reales, transporta `hard_set_tau=0.5` mediante una copia efímera
de la config y liga la provenance al receipt firmado. R483 tampoco reprodujo un
defecto productivo en el cierre físico `closed-world` ni en el débito
conservador.

El único finding restante es de cobertura contractual. El plan R481 exigía
negativos independientes para cada familia física, de ledger y de lineage, más
una secuencia positiva que partiera del débito unsigned y llegara a una
recuperación posterior desde el par firmado. La suite de `5aee5fb` cubre sólo
una parte de esa matriz.

Esta resolución no modifica el estimando científico, el draw, el source law, el
modelo, los thresholds, el runner ni el worker. Autoriza únicamente:

1. completar la matriz de pruebas omitida;
2. corregir producción sólo si alguno de esos tests reproduce un defecto;
3. conservar en la autoridad final toda la historia de los dos `REVISE`;
4. renumerar amendment, config y auditoría de resultados sin reutilizar IDs.

## 2. Historia que deberá preservar la amendment

La implementación `5aee5fb` no puede aparecer como autoridad positiva. La
amendment final conservará los 23 campos ya previstos por la resolución R481 y
añadirá cuatro objetos:

```text
r481_resolution_implementation
r481_resolution_implementation_audit
r483_resolution_plan
r483_resolution_plan_audit
```

El keyset top-level final tendrá exactamente 27 claves. La secuencia será:

```text
R480 PASS
  -> e617e15 implementación rechazada
  -> R481 REVISE
  -> resolución R481
  -> R482 PASS
  -> 5aee5fb implementación incompleta
  -> R483 REVISE
  -> esta resolución
  -> R484 PASS
  -> implementación sucesora
  -> R485 PASS
  -> amendment
  -> R486 PASS
  -> config v2
  -> R487 PASS / HEAD de ejecución
```

`r481_resolution_implementation` tendrá exactamente `commit`, `parent` y
`changed_sources`, con:

```text
commit = 5aee5fb6064cdd232620a5eebe9a9e856dae9f0f
parent = f844526bc42ee8ae366adee64bc63a9f780d8093
```

Sus dos deltas compararán R475 contra `5aee5fb`:

```text
preparer
  path = experiments/geometria_proporcional/prepare_wave56_fresh.py
  old_sha256 = 7d7ead44f6d0e64802dafa585a59a20ae78f43f5e975e03c60e6bd8a1de33d66
  new_sha256 = fd3a3809fc98c825cf1c3159b6af8602db50c59a529d7a32b12f2148f687cfb5

test
  path = tests/test_wave60_frozen_policy_transport.py
  old_sha256 = 328c934c63f2cb402633b72b52699e2d48433ff7e94966169a5b1428e6f63519
  new_sha256 = d8e5c78554f5eb5ae1be5472f42ba15761e7034f305591a8f2c7931f63f7ce62
```

`r481_resolution_implementation_audit` tendrá exactamente `commit`, `path`,
`sha256`, `audit_id`, `scope`, `verdict` y `findings`. Sus valores serán el
commit exclusivo que archive R483, el path del informe, su hash ya fijado,
`audit_id=R483`, scope
`INVALID_PREPARATION_RECOVERY_IMPLEMENTATION`, `verdict=REVISE` y conteos
`0/1/0`.

`r483_resolution_plan` ligará este documento mediante `commit`, `path` y
`sha256`. `r483_resolution_plan_audit` tendrá exactamente `commit`, `path`,
`sha256`, `audit_id`, `verdict` y `findings`, con `audit_id=R484`, scope
`INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN`, `PASS` y cero findings.

El objeto `recovery_implementation` se reservará para el commit sucesor y su
auditoría R485 positiva. Los tres sources científicos seguirán bajo R475 y los
deltas finales de preparer/test continuarán calculándose desde R475, no desde
ninguna implementación intermedia.

## 3. Matriz física exhaustiva

Sobre una copia temporal independiente del origen v1, la suite deberá ejecutar
un caso positivo intacto y rechazos independientes para cada mutación siguiente:

### 3.1 Bytes y mapas

- bytes de `generation_escrow.json`;
- bytes de `pre_generation_freeze.json`;
- bytes de `benchmark/manifest.json`;
- bytes de al menos un archivo de benchmark listado por el manifest;
- entrada adicional o ausente en `preserved_draw_sha256`;
- inventario primario alterado aunque el árbol físico permanezca intacto.

### 3.2 Metadata y topología

- modo del directorio `failed_preparation`;
- owner del directorio `failed_preparation`;
- modo de un archivo sensible;
- owner de un archivo sensible;
- symlink;
- hardlink;
- FIFO u otro nodo especial;
- archivo auxiliar adicional incluso cuando la amendment actualice su propio
  inventario para incluirlo.

### 3.3 Autoridades firmadas

- attestation o binding terminal de primary;
- attestation o binding terminal de replay;
- attestation o binding del par;
- inventario firmado de primary;
- inventario firmado de replay cuando el rol sea materializado en la fixture.

Cada caso debe fallar por la defensa productiva correspondiente. No se aceptará
como sustituto una única mutación que haga fallar varias familias a la vez.

## 4. Matriz exhaustiva del débito unsigned

La suite deberá probar el caso positivo exacto y rechazar por separado:

- `seconds` igual a cero, negativo, `59.0` o cualquier valor distinto de
  `60.0`;
- `applied_once=false`;
- cambio de régimen, wall observado o autoridad del registro externo;
- versión distinta de `2`;
- `prior_attempt_container` distinto del container v1 canónico;
- source distinto de `primary` v1;
- amendment por path no canónico o hash distinto;
- sólo `preparation_receipt.json` en primary;
- sólo attestation de preparación en primary;
- sólo `preparation_receipt.json` en replay;
- sólo attestation de preparación en replay;
- receipt y attestation válidos en primary;
- receipt y attestation válidos en replay.

Los cuatro casos de frontera receipt/attestation deben demostrar que una
autoridad parcial no se interpreta como ausencia de ledger. Los casos firmados
deben demostrar que, una vez existe autoridad durable, el flujo sale del ramal
unsigned y usa exclusivamente el ledger firmado.

## 5. Secuencia positiva completa del ledger

Una prueba integrada deberá materializar esta cadena dentro de un repositorio y
árbol temporal del mismo filesystem:

```text
origen v1 sin ledger firmado
  -> recovery primary v2: prior = 60
  -> primary firmado: cumulative = 60 + duration_primary
  -> replay v2: prior = cumulative_primary
  -> replay firmado: cumulative = prior_replay + duration_replay
  -> par v2 cerrado con durable preparation_seconds = cumulative_replay
  -> recovery v3 desde primary v2/par firmado
  -> prior v3 = durable_seconds del par v2
```

La última igualdad deberá probar simultáneamente que:

- el débito `60` aparece exactamente una vez;
- el replay no vuelve a aplicarlo;
- la recuperación v3 no vuelve a entrar al ramal unsigned;
- `recovery_pair_durable_elapsed()` y los receipts firmados concuerdan;
- cualquier boundary parcial receipt/attestation aborta antes de producir un
  nuevo output.

Se permite reutilizar helpers y fixtures de la prueba legacy v2→v3, pero la
cadena nueva deberá comenzar efectivamente en el origen
`INVALID_PREPARATION` v1 y en su débito unsigned. Una prueba de ledger genérica
que parte de un primary ya firmado no satisface este requisito.

## 6. Matriz específica del sufijo Git

La fixture sintética del lineage deberá conservar una positiva y rechazar de
forma independiente:

- R481 o R483 presentados como PASS;
- findings distintos en R481 o R483;
- parent saltado en resolución R481, R482, implementación `5aee5fb`, R483,
  resolución R483, R484, implementación final o R485;
- path adicional en cada clase de commit que deba ser exclusiva;
- scope, target, audit ID, verdict o conteos alterados en cada auditoría;
- path o SHA-256 físico alterado en cada documento ligado;
- blob Git alterado respecto del hash físico declarado;
- cruce entre blobs R475 y cualquiera de los blobs de recuperación;
- mutación de uno de los tres sources científicos congelados bajo R475.

Los helpers genéricos existentes pueden usarse para evitar duplicación, pero la
nueva fixture deberá pasar cada objeto real del ensamblaje ampliado. No basta
que otro test histórico cubra aisladamente un parser o una regla de parent.

## 7. Autoridad de implementación sucesora

Después de R484 PASS se publicará un commit que modificará exactamente:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

El preparador sólo podrá cambiar para autenticar la historia adicional,
renumerar las autoridades futuras y corregir un defecto que una prueba nueva
reproduzca. Si toda defensa productiva ya es correcta, la ampliación funcional
quedará limitada a la suite y el cambio productivo será únicamente de lineage.

La auditoría R485 tendrá:

```text
scope = INVALID_PREPARATION_RECOVERY_IMPLEMENTATION
target = {implementation_commit: <commit sucesor>}
technical_verdict = PASS
findings = {high: 0, medium: 0, low: 0}
```

Además de inspección, deberá ejecutar la suite Wave 60 completa y comprobar la
regresión explícita Waves 56–59. No se publicará amendment con R485 `REVISE`.

## 8. Schema y numeración futura

El schema de amendment conserva el nombre
`wave60-invalid-preparation-recovery-amendment-v1`; la ampliación agrega
historia, no cambia la semántica de recuperación. Su keyset definitivo será:

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
rejected_recovery_implementation
rejected_recovery_implementation_audit
r481_resolution_plan
r481_resolution_plan_audit
r481_resolution_implementation
r481_resolution_implementation_audit
r483_resolution_plan
r483_resolution_plan_audit
recovery_implementation
hard_set_contract
unledgered_preparation_debit
escrow_origin
preserved_draw_sha256
population_contract
origin_inventory
```

La numeración futura queda:

- R484: auditoría de este plan;
- R485: reauditoría PASS de la implementación sucesora;
- R486: auditoría de amendment;
- R487: auditoría de config v2;
- R488: auditoría de resultados si el par llega a un estado terminal evaluable.

Los paths serán:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
  484_wave60_invalid_preparation_recovery_r483_resolution_plan_audit.md
  485_wave60_invalid_preparation_recovery_implementation_final_reaudit.md
  486_wave60_invalid_preparation_recovery_amendment_audit.md
  487_wave60_frozen_policy_transport_v2_config_audit.md
  488_wave60_frozen_policy_transport_result_audit.md
```

`attempt.recovery` mantiene exactamente sus doce claves aprobadas. El hard-set
contract, el débito, los hashes del draw, la población y la topología física no
cambian. La config canónica continúa sin `hard_set_tau`; el valor `0.5` sólo se
inyecta en la copia efímera que recibe el materializador.

## 9. Criterio de autorización

Este documento no autoriza todavía la implementación. Requiere una auditoría
independiente R484 con target exacto `plan_commit + plan_sha256`, scope
`INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN`, PASS y cero findings.

La ejecución canónica permanece prohibida hasta que existan, en una cadena de
padres directos y commits exclusivos, R485 PASS, amendment, R486 PASS, config v2
y R487 PASS como HEAD limpio. Todo el trabajo es CPU-only. Ninguna de estas
autoridades promueve la arquitectura ni decide `GO/NO-GO` científico.
