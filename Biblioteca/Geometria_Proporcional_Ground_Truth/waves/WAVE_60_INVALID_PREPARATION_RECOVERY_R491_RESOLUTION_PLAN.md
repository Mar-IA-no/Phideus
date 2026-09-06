# Ola 60 — resolución del ID productivo de auditoría de amendment hallado en R491

> **Estado:** `PRE-CORRECTION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Implementación auditada:**
> `4e50425bc8c33ed668da6de21dcd2a22882268fa`
> **Auditoría R491:** commit
> `413ffdb8980d8128214898797eb01f4dd8af564d`, informe
> `491_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md`,
> SHA-256
> `a0dd9503d14204710bceddc95b9430b647f78a9de6be716067ed82991cd9f241`
> **Dictamen:** `REVISE / 0 HIGH + 1 MEDIUM + 0 LOW`

## 1. Finding y frontera

R491 cerró el finding probatorio de R489: confirmó cinco implementaciones,
once auditorías, 55 negativos semánticos con
`blob Git == archivo físico == binding`, 22 transiciones en ambas matrices y
las tres ramas científicas reconstruidas. También verificó que R489 permanece
`REVISE`, R490 autentica el plan anterior, la provenance positiva apunta sólo
a R491 y módulo, runner y worker conservan sus blobs R475.

El único finding nuevo está en el ramal productivo que autentica la auditoría
de la amendment. La implementación `4e50425` todavía exige
`expected_audit_id="R490"`, aunque el plan anterior reservaba R490 para auditar
el plan R489 y R492 para la amendment. Por scope y target, R490 tampoco podría
reutilizarse legítimamente como auditoría de amendment. Una cadena futura
correcta quedaría rechazada antes de la preparación.

Como R491 es ahora autoridad `REVISE`, sus números futuros no pueden
reutilizarse como si la implementación hubiera pasado. Este plan preserva la
historia completa y desplaza otra vez las autoridades aún inexistentes.

## 2. Corrección productiva testable

Se extraerá el enlace amendment→audit a una función pública y acotada:

```python
validate_wave60_invalid_preparation_amendment_audit(
    repo_root,
    recovery,
    amendment_sha256,
    amendment_commit,
)
```

La función delegará en `validate_wave60_audit_commit` y fijará exactamente:

```text
scope             = INVALID_PREPARATION_RECOVERY_AMENDMENT
target            = {amendment_sha256: <SHA físico/canónico>}
expected_parent   = commit exclusivo de amendment
expected_audit_id = R494
```

`_validate_wave60_invalid_preparation_recovery_amendment()` deberá usar esta
función, no mantener una segunda llamada inline. Así el contrato deja de ser
una constante enterrada en un ramal difícil de alcanzar antes de publicar la
amendment.

Una prueba sintética construirá un commit exclusivo de amendment y un informe
R494 con bloque de autoridad completo. Debe aceptar R494 y rechazar, mediante
reportes físicos alternativos, al menos los IDs obsoletos R490 y R492. La
prueba conservará activas las comprobaciones reales de path, SHA, scope,
target, parent directo y exclusividad. No se reemplaza el helper de autoridad
por mocks ni por introspección textual del source.

## 3. Lineage ampliado y nombres futuros

La secuencia canónica pasa a ser:

```text
R489 REVISE
  -> plan R489
  -> R490 PASS
  -> implementación 4e50425
  -> R491 REVISE
  -> este plan
  -> R492 PASS
  -> implementación sucesora
  -> R493 PASS
  -> amendment
  -> R494 PASS
  -> config v2
  -> R495 PASS / HEAD de ejecución
  -> R496 auditoría de resultados
```

Por tanto:

- R492 audita este plan;
- R493 es la única aceptación futura de implementación;
- R494 audita la amendment;
- R495 audita la config v2;
- R496 queda reservado a resultados primario/replay.

Los paths futuros son:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
  492_wave60_invalid_preparation_recovery_r491_resolution_plan_audit.md
  493_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md
  494_wave60_invalid_preparation_recovery_amendment_audit.md
  495_wave60_frozen_policy_transport_v2_config_audit.md
  496_wave60_frozen_policy_transport_result_audit.md
```

## 4. Historia adicional en la amendment

Las 39 claves fijadas por el plan R489 se preservan. Se añaden exactamente:

```text
r489_resolution_implementation
r489_resolution_implementation_audit
r491_resolution_plan
r491_resolution_plan_audit
```

El keyset top-level final pasa a 43 claves. Los objetos quedan definidos así:

- `r489_resolution_implementation`: commit `4e50425`, parent R490 y deltas de
  preparer/test calculados contra R475;
- `r489_resolution_implementation_audit`: R491 `REVISE 0/1/0` con commit,
  path, SHA, audit ID, scope, verdict y findings;
- `r491_resolution_plan`: commit, path y SHA de este documento;
- `r491_resolution_plan_audit`: R492 `PASS 0/0/0` sobre este plan.

Los blobs rechazados de `4e50425` son:

```text
preparer
  old = 7d7ead44f6d0e64802dafa585a59a20ae78f43f5e975e03c60e6bd8a1de33d66
  new = 83525c3edd0e5b77584d1341a9b65d0d4e45996ad3b9a831f7fc09274df109b4

test
  old = 328c934c63f2cb402633b72b52699e2d48433ff7e94966169a5b1428e6f63519
  new = 11d05738e96537a0774a0a90514a615cb3c426e06c2d341fbc6585ae759de11a
```

## 5. Matrices y provenance

La positiva sintética tendrá trece auditorías R481–R493. Conserva `REVISE` en
R481, R483, R485, R486, R487, R489 y R491; conserva o incorpora `PASS` en
R482, R484, R488, R490, R492 y R493.

Las ampliaciones cuantitativas son cerradas:

- seis implementaciones contrastadas: cinco rechazadas y una aceptada;
- trece auditorías por cinco campos: 65 negativos semánticos, todos con la
  triple igualdad como precondición explícita;
- 26 transiciones en `alternate_steps` y 26 en `suffix_steps`;
- R491 y R492 reciben también tres negativos de binding externo, parent
  saltado y un único path adicional;
- las tres ramas de mutación científica reconstruyen los cuatro eslabones
  añadidos antes de alcanzar el sucesor final.

La positiva de transaction/provenance usará exclusivamente:

```text
implementation_audit.audit_id = R493
```

R489 y R491 no pueden aparecer como autoridad positiva. Generation receipt,
preparation freeze y preparation receipt conservarán igualdad del objeto de
provenance completo. La attestation Ed25519 seguirá firmando transitivamente
el receipt exacto por path, bytes y SHA-256, sin cambiar schema.

`validate_wave60_final_config_authority()` debe esperar R493 para la
implementación de recuperación y R495 para la auditoría de config. La fixture
futura correspondiente debe materializar R493/R494/R495 y mantener la
partición entre source law R475 e implementación de recuperación.

## 6. Frontera científica, pruebas y recursos

La implementación sucesora sólo puede modificar:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

Módulo, runner y worker permanecen exactamente en:

```text
46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa
c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7
```

No cambian source law, draw, modelo, features, políticas, thresholds, roster,
estimando, presupuesto, schema de amendment ni las doce claves de
`attempt.recovery`. La config canónica sigue sin `hard_set_tau`.

R493 deberá leer el cambio completo y ejecutar CPU-only la suite Wave 60 y la
regresión explícita de los nueve archivos Waves 56–59, registrando duración,
RSS máximo y swaps. Los temporales se crearán bajo `/mnt/m2-1TB`, se
inventariarán y se eliminarán por path exacto.

## 7. Autorización y orden de avance

Este documento requiere auditoría independiente R492 con scope
`INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN`, target exacto
`plan_commit + plan_sha256`, verdict `PASS` y findings `0/0/0` antes de
implementar.

Después de R493 PASS se podrá materializar la amendment de 43 claves. Su
auditoría R494 deberá ejecutar el validador canónico completo, no limitarse a
leer JSON. Sólo entonces se publicará config v2, se obtendrá R495 y se
ejecutarán preflight, primaria y replay dentro del presupuesto compartido de
900 segundos. La cadena no promueve arquitectura ni decide `GO/NO-GO`.
