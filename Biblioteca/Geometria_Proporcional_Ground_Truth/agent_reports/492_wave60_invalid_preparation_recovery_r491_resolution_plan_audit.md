# R492 — Auditoría independiente del plan de resolución de R491

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

El plan identifica con precisión el único defecto productivo hallado por R491,
lo convierte en un contrato público alcanzable por una prueba real y desplaza
sin colisiones todas las autoridades futuras. Los conteos de lineage, keyset y
matrices son exactos; la provenance positiva queda reservada a R493, los guards
de config se actualizan coherentemente y la frontera científica permanece
byte-exacta. No encontré defectos de diseño, cobertura o realizabilidad que
impidan implementar el sucesor.

## Identidad del target y antecedente

El target auditado es exactamente
`e7af03ac590e90098ff94adb09cd0181f5a083e0`, hijo directo de R491
`413ffdb8980d8128214898797eb01f4dd8af564d`. El commit introduce exclusivamente
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN.md`;
su blob Git y archivo físico coinciden en
`d699814fa56de9ed05ce4b457ee4a231af379ee3c80c815c55b9c1a2c6cc1ce5`.

R491 es hijo directo de la implementación
`4e50425bc8c33ed668da6de21dcd2a22882268fa`, introduce sólo su informe y éste
coincide con el SHA-256 declarado por el plan:
`a0dd9503d14204710bceddc95b9430b647f78a9de6be716067ed82991cd9f241`.
La implementación es hija directa de R490 `9503a697efe6c9c766fcae6fee4b4d7982ba6221`
y cambia exclusivamente preparer/test. Leí completos este plan, R491, R489 y
R490, y contrasté el código y la prueba versionados en el target.

## Finding exacto y corrección testable

El finding no es nominal ni hipotético. El ramal productivo actual llama a
`validate_wave60_audit_commit()` con scope y target correctos, pero conserva
`expected_audit_id="R490"`
(`experiments/geometria_proporcional/prepare_wave56_fresh.py:5600-5610`). R490
ya autentica el plan R489; tampoco tiene el scope ni el target de una auditoría
de amendment. R491 demostró que este valor stale rechazaría una cadena futura
correcta antes de preparar el draw.

El helper público propuesto
`validate_wave60_invalid_preparation_amendment_audit(repo_root, recovery,
amendment_sha256, amendment_commit)` encapsula exactamente ese enlace y delega
en el validador canónico existente. Fija scope
`INVALID_PREPARATION_RECOVERY_AMENDMENT`, target
`{"amendment_sha256": amendment_sha256}`, parent directo del commit exclusivo
de amendment e ID R494. La función privada debe consumir ese helper y no dejar
una segunda constante inline
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN.md:34-59`).

La prueba prescrita alcanza el código real: materializa un commit exclusivo de
amendment y reportes físicos con bloque de autoridad completo, acepta R494 y
rechaza como mínimo R490 y R492. No usa mocks ni introspección del source. Al
delegar en `validate_wave60_audit_commit()`, mantiene activas las defensas
vigentes de path canónico, SHA físico, blob Git, scope, target, parent directo y
exclusividad (`prepare_wave56_fresh.py:480-507`). Así evita repetir el hueco de
R491, cuya suite sólo construía un archivo textual R492 como antecedente de
config sin recorrer el validador completo de amendment.

## Lineage, keyset y matrices

La renumeración es lineal y acíclica:

```text
R491 REVISE → plan R491 → R492 PASS → implementación → R493 PASS
→ amendment → R494 PASS → config v2 → R495 PASS/HEAD → R496 resultados
```

Cada autoridad existe antes de ser consumida y los cinco paths futuros
enumerados corresponden exactamente a esos roles. En particular, R492 sólo
audita este plan, R493 sólo acepta la implementación sucesora, R494 audita la
amendment, R495 audita config y R496 queda reservado a resultados
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN.md:68-106`).

Las 39 claves actuales se conservan y se añaden exactamente cuatro, sin
colisión:

```text
r489_resolution_implementation
r489_resolution_implementation_audit
r491_resolution_plan
r491_resolution_plan_audit
```

El total `39 + 4 = 43` es correcto. Los dos primeros preservan
`4e50425`/R491 como implementación rechazada y autoridad `REVISE 0/1/0`; los
dos restantes ligan este plan y R492. `recovery_implementation` queda libre
para el sexto candidato, único aceptado. Los hashes de `4e50425` declarados en
el plan coinciden con Git y usan R475 como base de `old_sha256`: preparer
`7d7ead44… → 83525c3e…` y test `328c934c… → 11d05738…`
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN.md:108-138`).

Los conteos ampliados también cierran:

- seis implementaciones: las cuatro rechazadas ya históricas, `4e50425` como
  quinta rechazada y el sucesor como sexta/aceptada;
- trece auditorías R481–R493: siete `REVISE` y seis `PASS` con sus verdicts
  históricos reales;
- `13 × 5 = 65` negativos semánticos, cada uno conservando como precondición
  `blob Git == físico == binding`;
- cuatro eslabones nuevos sobre las 22 transiciones actuales, total 26 tanto
  en `alternate_steps` como en `suffix_steps`;
- R491 y R492 con sus tres drifts de binding, parent saltado y path extra;
- tres ramas científicas que reconstruyen los cuatro eslabones nuevos antes de
  llegar a la implementación final.

La estructura vigente confirma la base de esos incrementos: once entradas y
55 semánticos (`tests/test_wave60_frozen_policy_transport.py:5450-5533`), cinco
implementaciones (`tests/test_wave60_frozen_policy_transport.py:5195-5228`) y
22 pasos en ambas matrices (`tests/test_wave60_frozen_policy_transport.py:5789-6025,6097-6276`).

## Provenance, guards y frontera científica

La provenance positiva se desplaza exclusivamente a
`implementation_audit.audit_id=R493`; R489 y R491 permanecen excluidas. El
objeto completo sigue idéntico en generation receipt, preparation freeze y
preparation receipt, y la attestation conserva la firma transitiva del receipt
exacto por path, bytes y SHA-256, sin cambiar schema
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN.md:157-166`).

El guard general de config aún espera R491 para la implementación recuperada y
R493 para la auditoría final (`prepare_wave56_fresh.py:1183-1210`). El plan
ordena cambiar ambos a R493 y R495, y exige que la fixture materialice además
R494. Esto es coherente con el helper de amendment y con los parents
config←R494←amendment y R495←config; no queda una reutilización de IDs
anteriores.

La implementación sucesora queda limitada exactamente a:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

Módulo, runner y worker continúan físicamente y en R475 con SHA-256
`46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`,
`1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa` y
`c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.
El plan congela además draw, modelo, features, políticas, thresholds, roster,
estimando, presupuesto, schema, las doce claves de `attempt.recovery` y la
ausencia canónica de `hard_set_tau`.

La materialización queda correctamente bloqueada hasta R493 PASS; después la
amendment requiere validación canónica completa por R494, y config/ejecución
esperan R495 como HEAD. El plan no autoriza draw anticipado ni decide
`GO/NO-GO` científico (`WAVE_60_INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN.md:194-210`).

No ejecuté suites largas porque el target es documental y todas sus
precondiciones se verifican estáticamente contra Git, el preparer y el test
vigentes. Todo fue CPU-only con `CUDA_VISIBLE_DEVICES=''`; no usé ni consulté
GPU. Este informe es el único archivo creado.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R492",
  "scope": "INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "e7af03ac590e90098ff94adb09cd0181f5a083e0",
    "plan_sha256": "d699814fa56de9ed05ce4b457ee4a231af379ee3c80c815c55b9c1a2c6cc1ce5"
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
