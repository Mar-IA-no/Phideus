# Wave 56 Stage 1 — Plan final de matriz de autoridad

**Estado:** `FROZEN_IMPLEMENTATION_PLAN`
**Alcance:** completar literalmente F1 de R390; sin cambios científicos

## Base

I3 fija el runner correcto; I4 fija el parser terminal; I5 fija la primera
ampliación adversarial. R390 confirmó los tres mecanismos y 56/56 tests, pero
exigió tres coberturas restantes: path individual faltante en I5, runner
modificado específicamente en I4 y matriz terminal completa end-to-end para
los tres roles. Ningún `REVISE` autoriza ejecución.

## Implementación acumulativa

El schema final registra constantes e identidades para:

- `runner_commit = 7b37b5381b0c7540e86de2d53001903475d321ab`;
- `authority_commit = 3f404103111a67721fa7a3d15cbf4ec392025e5f`;
- `coverage_commit = 68316175067419c914af584e14ec2bafa4ff550b`;
- `implementation.commit = I6`, que cambia sólo preparador+test.

Los tres commits históricos deben ser ancestrales, conservar sus diffs exactos
y mantener el runner byte-idéntico. I6 toma los blobs finales de preparador y
test; el contrato conserva exactamente preparador+runner como source deltas.

La DAG nueva es directa:

    P8 → R391(plan audit) → I6 → R392(implementation audit) → J8 → R393(final audit)

El amendment registra commit/path/hash de P8 y R391, las cuatro identidades de
implementación, path/hash de R392 y sólo el path futuro de R393. Cada commit
documental introduce un path; `HEAD == R393` y worktree limpio son obligatorios.

Paths:

    Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_AUTHORITY_MATRIX_FINALIZATION_PLAN.md
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/391_wave56_stage1_authority_matrix_finalization_plan_audit.md
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/392_wave56_stage1_authority_matrix_finalization_implementation_audit.md
    experiments/geometria_proporcional/configs/wave56_stage1_authority_matrix_finalization_amendment_v8.json
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/393_wave56_stage1_authority_matrix_finalization_final_audit.md

## Cobertura exacta

El fixture agrega I6 con preparador faltante y test faltante; crea una variante
de I4 que modifica runner; y parametriza cada rol `plan_audit`,
`implementation_audit`, `final_audit` contra decisión `REVISE`, decisión
ausente, duplicada, seguida por contenido, fence backtick y fence tilde. Cada
caso es end-to-end por `validate_recovery_amendment`, no sólo unitario.

Se conservan todos los negativos anteriores y los controles de inventario,
calibración, no-redraw, origen, manifest y replay. R392 corre la focal; R393
repite focal y suite Wave49–56.

## Ejecución

Sólo tras R393 `PASS`, recovery usa el intento fallido original; `--force`
archiva el primary v2 `PREPARED`; se ejecutan fases primary y replay exacto.
Todo es CPU-only, preservado y sin declarar `GO/NO-GO`.
