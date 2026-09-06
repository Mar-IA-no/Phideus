# Ola 60 — resolución de los dos findings contractuales de R486

> **Estado:** `PRE-CORRECTION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Plan auditado:** commit
> `0c07b10abf7831dc6577c39637f3b68c2a3a02b2`, SHA-256
> `cbbdbf361deaa54d9bf2461aceda7bef54412bb1fa40d821a974f2fb02520e4a`
> **Auditoría R486:** commit
> `2049effcdd60c3a922aee355266a87440ab97de2`, informe
> `486_wave60_invalid_preparation_recovery_r485_resolution_plan_audit.md`,
> SHA-256
> `677cfce51190d1d5d269e716543dfa10be70b0659838d948baf669e6b7520ffd`
> **Dictamen:** `REVISE / 0 HIGH + 2 MEDIUM + 0 LOW`

## 1. Alcance y relación con el plan anterior

R486 confirmó que son realizables y suficientes el diseño de mutaciones
semánticas aisladas, la divergencia blob Git/archivo físico, los commits con un
solo path extra y la cadena secundaria que altera físicamente cada source
científico. También confirmó el keyset entonces propuesto, la partición de
autoridades y la ausencia de cambio científico.

Los dos findings afectan sólo al contrato escrito:

1. la matriz de auditorías omitió R486, aunque esa autoridad ya era un objeto
   de la futura amendment;
2. se dijo que la provenance debía aparecer como objeto dentro de la
   attestation, pero el schema vigente autentica transitivamente el
   `preparation_receipt.json` que la contiene.

Este documento reemplaza únicamente §§2–4, 8–10 del plan R485 en cuanto a
lineage, matriz de auditorías, provenance y numeración. Permanecen vigentes sin
cambio sus §§5–7: divergencia blob/físico, exclusividad aislada y mutación real
de los tres sources científicos.

No se autoriza todavía implementar. No se modifica módulo científico, runner,
worker, draw, source law, modelo, thresholds, roster ni estimando.

## 2. Corrección del contrato de provenance

La fixture positiva
`test_invalid_preparation_transaction_wires_tau_and_signed_provenance` deberá
usar la futura autoridad aceptada:

```text
implementation_audit.audit_id = R488
```

La prueba exigirá exactamente:

1. igualdad del objeto completo `recovery_provenance` en
   `generation_receipt.json`, `preparation_freeze.json` y
   `preparation_receipt.json`;
2. verificación Ed25519 real de `preparation_attestation.json`;
3. presencia en su payload del record de `preparation_receipt.json` con path,
   bytes y SHA-256 físicos exactos;
4. igualdad del SHA firmado con el SHA del receipt que contiene la provenance;
5. ausencia de R483, R485 o cualquier auditoría `REVISE` como autoridad
   positiva del contexto.

No se añade una clave `recovery_provenance` a la attestation ni se cambia su
schema. La autoridad es transitiva:

```text
attestation Ed25519
  -> record(path, bytes, sha256) del preparation receipt
  -> recovery_provenance completa dentro del receipt
```

## 3. Matriz exhaustiva de auditorías corregida

La cadena sintética positiva deberá contener ocho auditorías:

```text
R481 REVISE
R482 PASS
R483 REVISE
R484 PASS
R485 REVISE
R486 REVISE
R487 PASS
R488 PASS
```

Son ocho autoridades: R481–R488 sin saltos. R486 audita el plan R485 rechazado;
R487 audita este plan corregido; R488 audita la implementación sucesora.

Para **cada una** se materializarán negativos independientes de todos los
campos semánticos del bloque machine-readable:

```text
audit_id
scope
target
technical_verdict
findings
```

Y de todos sus bindings externos:

```text
commit o audit_commit
path o audit_path
sha256 o audit_sha256
```

Cuando un objeto top-level no replica `scope`, `target` o
`technical_verdict`, la mutación se realizará dentro de un reporte alternativo
real. Ese reporte deberá:

- descender del parent correcto;
- modificar exclusivamente su path canónico;
- tener binding externo actualizado a sus bytes físicos;
- diferir de la positiva sólo en el campo bajo prueba.

Antes de invocar el validador, cada test comprobará parent, paths cambiados,
hash físico/blob y diferencia semántica única. Para R486, el parent es el commit
`0c07b10abf7831dc6577c39637f3b68c2a3a02b2` y el target exacto es
`plan_commit + plan_sha256` de ese plan. Para R487, el parent será el commit de
este documento y el target será su `plan_commit + plan_sha256`.

Los verdicts positivos y negativos no se normalizan. La positiva exige:

- R481, R483, R485 y R486: `REVISE` con sus conteos exactos;
- R482, R484, R487 y R488: `PASS` con `0/0/0`.

## 4. Historia adicional de la amendment

Las 31 claves propuestas por el plan R485 se conservan, pero ahora
`r485_resolution_plan_audit` representa correctamente R486 `REVISE`. Se añaden
dos objetos:

```text
r486_resolution_plan
r486_resolution_plan_audit
```

El keyset top-level final tendrá exactamente 33 claves. Los seis objetos que se
agregan a las 27 claves fijadas por R483 son:

```text
r483_resolution_implementation
r483_resolution_implementation_audit
r485_resolution_plan
r485_resolution_plan_audit
r486_resolution_plan
r486_resolution_plan_audit
```

Sus autoridades son:

- `r483_resolution_implementation`: `f32ba2b`, parent R484 y deltas de
  preparer/test desde R475;
- `r483_resolution_implementation_audit`: R485 `REVISE 0/1/0`;
- `r485_resolution_plan`: plan `0c07b10`, hijo de R485;
- `r485_resolution_plan_audit`: R486 `REVISE 0/2/0`;
- `r486_resolution_plan`: este documento, hijo de R486;
- `r486_resolution_plan_audit`: R487 `PASS 0/0/0`.

Cada objeto usa los mismos keysets cerrados ya aprobados:

```text
implementation = commit + parent + changed_sources
document       = commit + path + sha256
audit          = commit + path + sha256 + audit_id + verdict + findings
```

Los audits de implementación agregan además `scope`; el objeto positivo
`recovery_implementation` conserva su forma vigente completa. Los scopes y
targets se autentican desde los reportes, aunque no se dupliquen en todos los
bindings top-level.

## 5. Lineage definitivo

La secuencia futura queda:

```text
R484 PASS
  -> f32ba2b implementación incompleta
  -> R485 REVISE
  -> plan R485 0c07b10
  -> R486 REVISE
  -> este plan
  -> R487 PASS
  -> implementación sucesora
  -> R488 PASS
  -> amendment
  -> R489 PASS
  -> config v2
  -> R490 PASS / HEAD de ejecución
  -> R491 auditoría de resultados
```

Todos los commits serán hijos directos. Planes y auditorías cambian un único
path; las implementaciones cambian exactamente preparer y test.

`recovery_implementation` se reserva para el sucesor auditado por R488. Los
hashes `old_sha256` de preparer/test siguen anclados en R475. Los tres sources
científicos deben conservar en el commit final los blobs exactos de R475:

```text
module = 46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
runner = 1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa
worker = c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7
```

## 6. Autoridad de implementación y numeración

Después de R487 PASS, el sucesor puede modificar únicamente:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

Su alcance es:

- autenticar los seis objetos históricos nuevos y el keyset de 33 claves;
- renumerar autoridad final de implementación a R488;
- renumerar amendment/config/resultados a R489/R490/R491;
- comprobar blobs científicos reales contra R475;
- completar las matrices aprobadas por el plan R485 y esta resolución;
- corregir la positiva de provenance según §2.

La auditoría de implementación será R488 con scope
`INVALID_PREPARATION_RECOVERY_IMPLEMENTATION`, target exacto del commit
sucesor, PASS y cero findings. Repetirá suite Wave 60 y regresión Waves 56–59 en
CPU, registrará RSS y swaps, y comprobará que módulo/runner/worker no cambiaron.

Los paths futuros serán:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
  487_wave60_invalid_preparation_recovery_r486_resolution_plan_audit.md
  488_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md
  489_wave60_invalid_preparation_recovery_amendment_audit.md
  490_wave60_frozen_policy_transport_v2_config_audit.md
  491_wave60_frozen_policy_transport_result_audit.md
```

El schema de amendment mantiene
`wave60-invalid-preparation-recovery-amendment-v1`; amplía historia, no
semántica de recuperación. `attempt.recovery` conserva exactamente doce claves.
La config canónica permanece sin `hard_set_tau`; el valor `0.5` sólo entra en la
copia efímera del materializador.

## 7. Criterio de autorización

Este documento requiere una auditoría independiente R487 con target exacto
`plan_commit + plan_sha256`, scope
`INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN`, PASS y cero findings. Una
auditoría `REVISE` volverá a preservarse; no se la promoverá por conveniencia.

No se publica amendment ni se ejecuta el draw hasta que existan en secuencia
R488 PASS, amendment, R489 PASS, config v2 y R490 PASS como HEAD limpio. Todo el
trabajo es CPU-only. La promoción de arquitectura y cualquier `GO/NO-GO`
permanecen fuera de esta cadena técnica.
