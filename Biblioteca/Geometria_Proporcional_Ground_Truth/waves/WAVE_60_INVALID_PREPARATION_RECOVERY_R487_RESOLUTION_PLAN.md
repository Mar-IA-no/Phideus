# Ola 60 — errata de identidad y cierre del plan R486

> **Estado:** `PRE-CORRECTION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Plan auditado:** commit
> `0be3cd6c8b2717efd5ad4b4c24c09c7b55fa1c37`, SHA-256
> `1b2d64b70196ef8af46d388ddaf668c025fe4af02a5879238f2594df8eb64fbc`
> **Auditoría R487:** commit
> `50e5bd3cbfeb32491b71c24ee2b52c9b2a6325fb`, informe
> `487_wave60_invalid_preparation_recovery_r486_resolution_plan_audit.md`,
> SHA-256
> `049e7ddb3ab584cd532b7d833fed4f27cd93466de1642487cd9458d253c21113`
> **Dictamen:** `REVISE / 0 HIGH + 1 MEDIUM + 0 LOW`

## 1. Corrección única

R487 confirmó que el plan R486 resuelve los dos findings anteriores, que su
matriz de ocho auditorías es ejecutable, que la firma transitiva de provenance
coincide con el schema vigente y que el keyset, los negativos Git y la frontera
científica son coherentes. Su único finding fue una identidad de commit mal
transcrita en el encabezado.

La auditoría R486 fue introducida por el commit real:

```text
2049eff3b411024e6b4fd444f2b975ae76c27f3e
```

No por el objeto inexistente:

```text
2049effcdd60c3a922aee355266a87440ab97de2
```

El commit real es hijo directo del plan R485
`0c07b10abf7831dc6577c39637f3b68c2a3a02b2`, introduce exclusivamente
`486_wave60_invalid_preparation_recovery_r485_resolution_plan_audit.md` y su
blob coincide con SHA-256
`677cfce51190d1d5d269e716543dfa10be70b0659838d948baf669e6b7520ffd`.

Ésta es la única corrección sustantiva. Todas las obligaciones del plan R486 y
las §§5–7 vigentes del plan R485 se incorporan por referencia sin relajación.

## 2. Historia que debe conservar la amendment

El plan R486 y R487 permanecen como historia `REVISE`; no se editan ni se
reinterpretan. A las 33 claves previstas por el plan R486 se añaden:

```text
r487_resolution_plan
r487_resolution_plan_audit
```

El keyset top-level final tendrá exactamente 35 claves. Las ocho claves
añadidas sobre las 27 fijadas por R483 serán:

```text
r483_resolution_implementation
r483_resolution_implementation_audit
r485_resolution_plan
r485_resolution_plan_audit
r486_resolution_plan
r486_resolution_plan_audit
r487_resolution_plan
r487_resolution_plan_audit
```

Los nuevos bindings se interpretan así:

- `r486_resolution_plan` liga el plan `0be3cd6`;
- `r486_resolution_plan_audit` liga R487 `REVISE 0/1/0`;
- `r487_resolution_plan` liga este documento;
- `r487_resolution_plan_audit` ligará R488 `PASS 0/0/0`.

`r485_resolution_plan_audit` debe ligar el commit R486 corregido
`2049eff3b411024e6b4fd444f2b975ae76c27f3e`, con `REVISE 0/2/0`. El preparador
deberá hardcodear y verificar esa identidad exacta.

Los documentos mantienen keyset `commit/path/sha256`. Las auditorías de plan
mantienen `commit/path/sha256/audit_id/verdict/findings`, mientras scope y
target se autentican desde su bloque machine-readable. Ningún binding positivo
puede señalar a R486 o R487.

## 3. Matriz final de auditorías

La fixture positiva del sufijo contendrá nueve auditorías consecutivas:

```text
R481 REVISE 0/1/0
R482 PASS   0/0/0
R483 REVISE 0/1/0
R484 PASS   0/0/0
R485 REVISE 0/1/0
R486 REVISE 0/2/0
R487 REVISE 0/1/0
R488 PASS   0/0/0
R489 PASS   0/0/0
```

R488 audita este documento. R489 audita la implementación sucesora. Para cada
una se conservan los negativos aislados exigidos por los planes R485/R486:
campos semánticos internos, bindings externos, parent saltado, path adicional
exacto, divergencia blob/físico y conteos/verdicts propios.

La positiva de transaction/provenance usará
`implementation_audit.audit_id=R489`. La provenance completa será idéntica en
generation receipt, preparation freeze y preparation receipt; la attestation
firmará transitivamente el receipt por path, bytes y SHA-256, sin cambio de
schema.

## 4. Lineage y numeración definitiva

La secuencia desde el punto actual será:

```text
R487 REVISE
  -> este documento
  -> R488 PASS
  -> implementación sucesora
  -> R489 PASS
  -> amendment
  -> R490 PASS
  -> config v2
  -> R491 PASS / HEAD de ejecución
  -> R492 auditoría de resultados
```

Los paths futuros serán:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
  488_wave60_invalid_preparation_recovery_r487_resolution_plan_audit.md
  489_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md
  490_wave60_invalid_preparation_recovery_amendment_audit.md
  491_wave60_frozen_policy_transport_v2_config_audit.md
  492_wave60_frozen_policy_transport_result_audit.md
```

Cada commit debe ser hijo directo del anterior. Planes y auditorías son
exclusivos de un path. La implementación sucesora sólo puede modificar:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

Su auditoría R489 mantiene scope
`INVALID_PREPARATION_RECOVERY_IMPLEMENTATION`, target exacto del commit, PASS y
cero findings. Amendment/config/resultados pasan respectivamente a R490, R491
y R492.

## 5. Frontera inalterada

El schema de amendment continúa siendo
`wave60-invalid-preparation-recovery-amendment-v1`; el cambio es sólo de
historia. `attempt.recovery` conserva sus doce claves. La config canónica no
incorpora `hard_set_tau`; el valor `0.5` sigue inyectándose únicamente en la
copia efímera del materializador.

Módulo, runner y worker permanecen bajo R475 con hashes:

```text
46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa
c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7
```

No se altera draw, source law, estados, features, thresholds, roster,
estimandos, controles ni límites. Todo el trabajo continúa CPU-only.

## 6. Autorización

Este documento requiere auditoría independiente R488 con scope
`INVALID_PREPARATION_RECOVERY_R487_RESOLUTION_PLAN`, target exacto
`plan_commit + plan_sha256`, PASS y `0/0/0` antes de implementar.

No se publica amendment ni se ejecuta el draw hasta encadenar implementación,
R489 PASS, amendment, R490 PASS, config v2 y R491 PASS como HEAD limpio. Esta
cadena no promueve arquitectura ni decide `GO/NO-GO` científico.
