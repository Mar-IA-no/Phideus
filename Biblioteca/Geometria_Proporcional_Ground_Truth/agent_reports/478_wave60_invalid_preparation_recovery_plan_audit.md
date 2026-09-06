# R478 — Auditoría independiente del plan de recuperación Wave 60

## Veredicto

**REVISE — 0 HIGH, 2 MEDIUM, 0 LOW.**

La recuperación del mismo draw después de `INVALID_PREPARATION` es científica y operacionalmente válida. El plan identifica correctamente el defecto, conserva la pregunta experimental y propone un débito conservador razonable. Sin embargo, todavía no cierra dos contratos de autoridad necesarios para implementar sin romper la source law v2 ni el lineage Git.

## Identidad del plan

- Commit: `ba193d52cd46f23d57c1a0b811433d2cbcfedb6d`.
- Parent: auditoría final de config R477 `2a10b6cb5a88fd2af4ca2f5f8230a296f59c8948`.
- Commit exclusivo: modifica solamente `WAVE_60_INVALID_PREPARATION_RECOVERY_PLAN.md`.
- SHA-256 físico y blob Git: `0ae881499221845d309e66dba9da42821b9e841002f8f8c4329790e87896965f`.
- Worktree final limpio.
- El plan fue leído completo: 198 líneas.

El hecho de que el plan sea hijo de R477 es correcto: la recuperación comienza después de la config v1 auditada y de su intento terminal, no reescribe la historia previa.

## Estado físico y validez de recuperar

El validador canónico `validate_pair_failure_package` aceptó íntegramente el intento v1.

- Pair terminal: `PAIR_ABORTED_PRE_TRUTH`.
- Primary y replay: `INVALID_PREPARATION`.
- `any_truth_accessed=false`.
- `recovery_allowed=true`.
- Pair failure: `05ede417b1f856488c1796210029aa74c74211f3f14858764d9705cbb0b3563d`.
- Pair status: `922f227c9f74ec0b917533d58205a0ebdb7147bebdee111c37d5340973b3ee78`.
- Escrow: `930c9732b2d18a22ee8654b5478202abf51fa291579e459a34d244d46e0074a3`.
- Pre-generation freeze: `88992f10ee270ff92f1ed286c1feb9418eb277fa2a11f886580d1ff52d9c4ebc`.
- Benchmark manifest: `de43c7ebbfe0d3c3cc1d5f8d62bf04d094b68f835ece16e97fadd8708238b9f2`.
- Preparation error: `7dee40f8c6ffbf9d6d317d4992c90514b2f3248a2274ce7f63a49e9ffa0d1547`.

El inventario firmado primario contiene exactamente 34 archivos físicos y todos sus hashes coinciden. No hay hardlinks. El draw está efectivamente bajo `primary/failed_preparation/`; no existen escrow ni `preparation_receipt.json` en la raíz primaria.

La implementación vigente reproduce exactamente el bloqueo descripto por el plan:

- `read_escrow(primary)` falla porque busca `primary/generation_escrow.json`.
- El lector de presupuesto falla con `Wave 60 prior preparation budget authority is absent`.
- El escrow archivado sí existe bajo `primary/failed_preparation/generation_escrow.json`.

Por tanto, no hace falta un draw nuevo. Existe evidencia suficiente para recuperar el draw original mediante validación content-blind, copia byte-exacta con inodos nuevos y reejecución determinista de los derivados.

## Invariancia científica

La explicitación `hard_set_tau=0.5` es legítima y no constituye selección post-draw:

- La config fuente Wave 59 lo fija en `0.5`.
- Ese snapshot está ligado por la source law.
- `HARD-SET` ya forma parte de la referencia congelada.
- El materializador heredado consume ese valor en `run_wave59_hgb_guard_bracket.py:305`.

Se preservan correctamente source law v2, roster, thresholds HGB, features, penalty, seeds, splits, bootstrap, estimandos y límites. No hay justificación para refit, recalibración, reselección ni GPU.

El débito de 60 segundos también es metodológicamente aceptable: excede los 48,51 segundos observados sin fingir que esa observación externa es un ledger firmado. El régimen debe quedar explícitamente separado de los elapsed exactos provenientes de receipts firmados.

## Findings

### MEDIUM 1 — Falta separar la autoridad source-law de la implementación de recuperación

El plan exige conservar la source law v2 y al mismo tiempo autoriza cambios en el módulo y runner ligados a ella ([plan:72-83](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_PLAN.md:72), [plan:145-155](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_PLAN.md:145)). También prevé actualizar hashes de implementación, pero no define qué binding autentica cada partición.

Con el código vigente, un único `implementation_binding` no puede cumplir ambos roles:

- `validate_source_authority` exige que el implementation commit/audit de la config coincida exactamente con el request y freeze de v2 ([runner:2347](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2347), [runner:2361](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2361)).
- La source law v2 está ligada a `9f1a229…` y R475.
- El validador de config final exige que los cinco blobs de implementación actuales coincidan con el commit indicado por ese mismo binding ([preparer:495](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:495)).
- Si el binding se actualiza a R479, v2 deja de validar; si se conserva R475, los archivos corregidos dejan de coincidir con el commit auditado.

Corrección requerida:

- preservar explícitamente el binding R475 como autoridad de origen de source law v2;
- introducir un binding separado para la implementación de recuperación y definir el roster de paths que autentica;
- hacer que el validador final compruebe cada path contra la autoridad correspondiente;
- añadir una prueba que valide source law v2 intacta bajo la config recuperada y detecte cualquier cruce de bindings.

La alternativa más simple es restringir el parche a `prepare_wave56_fresh.py` y tests, mantener módulo/runner/worker bajo R475 y autorizar `hard_set_tau=0.5` mediante la amendment auditada. El preparador puede pasar una vista tipada con ese valor al materializador y registrarlo en freeze/receipt de recuperación. Si se decide modificar módulo o runner, el plan debe definir expresamente la nueva partición de autoridad; regenerar source law sólo sería necesario si cambia su ley científica o su proyección, no por esta corrección de preparación.

### MEDIUM 2 — El lineage y el schema del débito no están cerrados a nivel ejecutable

La cadena prevista es correcta conceptualmente:

`R477 → plan → R478 → implementación → R479 → amendment → R480 → config → R481`.

Pero el código vigente exige que el commit de amendment sea hijo directo de la auditoría de config previa R477 ([preparer:3897](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3897)). Esa condición es incompatible con los cuatro commits intermedios previstos por el plan.

Además, el plan dice que amendment y `attempt.recovery` incorporarán el débito y los bindings de la corrección ([plan:96-110](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_PLAN.md:96)), pero no congela:

- schema y keyset exactos;
- nombres y tipos de los campos de débito;
- bindings exactos de plan/R478 e implementación/R479;
- ecuación acumulativa para primaria, replay y recuperaciones posteriores;
- regla precisa que sustituye el direct-parent actual.

Corrección requerida:

- definir el schema versionado exacto de amendment y `attempt.recovery`;
- ligar plan commit/hash, R478, implementation commit, R479 y débito `60.0`;
- exigir parents directos en cada eslabón nuevo y tratar R477 como ancestro/config de origen, no como parent inmediato de la amendment;
- fijar `primary_prior=60`, `replay_prior=60+primary_duration` y la continuidad posterior del ledger;
- probar rechazo de lineage saltado, binding cruzado, cero, doble débito, mezcla con receipt firmado y schema legacy con campos nuevos.

## Cobertura de pruebas

Las nueve familias propuestas son pertinentes y cubren el defecto físico, materializador real, integridad del draw, presupuesto, barreras pre-truth, primary/replay y regresión amplia.

Para resultar suficientes deben agregarse explícitamente:

- preservación y validación de source law v2 bajo la autoridad R475 separada;
- partición de blobs entre implementación source-law e implementación recovery;
- lineage completo R477–R481 con parents directos;
- schema/keyset exactos del débito conservador;
- prueba de que `hard_set_tau` proviene de la amendment/config auditada y queda registrado en la preparación, no de un default silencioso.

No se modificaron archivos ni artefactos. Toda la auditoría fue CPU-only; no se usó GPU.

```json
{
  "Audit-ID": "R478",
  "Scope": "RECOVERY_PLAN",
  "Target": {
    "plan_commit": "ba193d52cd46f23d57c1a0b811433d2cbcfedb6d",
    "plan_sha256": "0ae881499221845d309e66dba9da42821b9e841002f8f8c4329790e87896965f"
  },
  "Verdict": "REVISE",
  "Findings": {
    "HIGH": 0,
    "MEDIUM": 2,
    "LOW": 0
  }
}
```
