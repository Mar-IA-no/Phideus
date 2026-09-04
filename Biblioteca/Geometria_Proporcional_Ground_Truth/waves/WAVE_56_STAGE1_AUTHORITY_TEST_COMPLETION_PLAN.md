# Wave 56 Stage 1 — Plan de cierre de cobertura de autoridad

**Estado:** `FROZEN_IMPLEMENTATION_PLAN`
**Alcance:** completar los negativos exigidos por R388; sin cambio científico

## Evidencia y límite

I3 (`7b37b5381b0c7540e86de2d53001903475d321ab`) contiene el runner de
inventario visible correcto. I4 (`3f404103111a67721fa7a3d15cbf4ec392025e5f`)
contiene el parser terminal y la lineage aprobada por P6. R388 confirmó ambos
por inspección y 46/46 tests, pero concluyó `REVISE` porque faltaban negativos
explícitos de varios bordes. Este plan sólo completa esa matriz y adapta el
schema a la lineage real; no reabre runner, datos, modelo ni protocolo.

## Lineage verificable

La implementación acumulativa queda representada por:

- `runner_commit = I3`, diff exacto preparador+runner+test;
- `authority_commit = I4`, diff exacto preparador+test, runner idéntico a I3;
- `implementation.commit = I5`, diff exacto preparador+test para schema final y
  negativos faltantes.

I3 debe ser ancestro de I4, e I4 ancestro del plan. Los dos commits quedan
fijados por constantes y por el amendment. Los blobs finales de preparador/test
se toman de I5; el runner se toma de I3 y debe ser idéntico en I4 e I5. Frente
al escrow siguen existiendo sólo los deltas preparador+runner.

La nueva DAG es completamente directa desde el plan:

    P7 → R389(plan audit) → I5 → R390(implementation audit) → J7 → R391(final audit)

P7/R389/R390/J7/R391 introducen un único path; I5 cambia sólo preparador+test.
El amendment registra commit/path/hash de plan y plan-audit, las tres
identidades de implementación, path/hash de R390 y sólo el path futuro de R391.

Paths canónicos:

    Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_AUTHORITY_TEST_COMPLETION_PLAN.md
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/389_wave56_stage1_authority_test_completion_plan_audit.md
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/390_wave56_stage1_authority_test_completion_implementation_audit.md
    experiments/geometria_proporcional/configs/wave56_stage1_authority_test_completion_amendment_v7.json
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/391_wave56_stage1_authority_test_completion_final_audit.md

## Negativos obligatorios

El fixture Git separado debe rechazar:

1. runner_commit falso y runner_commit no ancestro;
2. authority_commit con preparador o test faltante, runner modificado, hash
   divergente o identidad falsa;
3. I5 con path faltante/adicional o parent distinto de R389;
4. R389 omitido, hash mutado, `PASS/REVISE` contradictorio, commit falso,
   commit no exclusivo o parent distinto de P7;
5. pérdida de preparador/runner o tercer source delta;
6. contradicciones terminales, ausencia/duplicación/contenido posterior y
   fences de backticks/tildes para R389/R390/R391.

Los informes de autoridad usan UTF-8/LF, no contienen fences/comentarios HTML y
terminan con la decisión machine-verifiable exacta concordante. La auditoría
humana completa sigue siendo obligatoria.

## Cierre y ejecución

R390 ejecuta la focal ampliada; R391 repite focal y suite Wave49–56. Sólo con
`HEAD == R391` y worktree limpio se reejecuta recovery desde el intento fallido
original. `--force` archiva el primary v2 `PREPARED`; luego se ejecutan fases
primary y replay exacto. Todo continúa CPU-only, sin redraw, sin borrado y sin
declarar `GO/NO-GO`.
