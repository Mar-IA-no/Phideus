# R314 — auditoría independiente del plan de la Ola 56

> **Agente:** `01a0672b-90d1-7ca0-bb09-992b7ce7624d` (`Kepler`)
> **Fecha:** 2026-09-03
> **Veredicto:** `REVISE`
> **Preservación:** informe del subagente, verbatim.

## Findings

1. **ALTO: el control shuffled no conserva la tarea de entrenamiento.** El modelo primario se ajusta sólo donde `a_hard != a_posterior`, pero el shuffle mueve los 24 targets completos y después los asocia con el patrón de desacuerdos del token receptor (`plan:99`, `plan:144`). En el monitor primario de Ola 55 hay `1.031` pares en desacuerdo: sólo `4/1.031` tienen target cero. Bajo el shuffle propuesto, la fracción esperada de ceros pasa a `72,07%`, porque los acuerdos del donante aportan ceros estructurales. Por tanto, el control cambia distribución y dificultad, no sólo rompe contexto-consecuencia. Además, una única permutación se vuelve criterio vinculante (`plan:183`). Debe redefinirse para preservar población y marginal del target, y usar múltiples permutaciones predeclaradas o quedar descriptivo.

2. **ALTO: Stage 0 no tiene una regla de selección ejecutable.** Se enumeran modelos y se promete elegir por regret OOF, pero faltan número y construcción de folds, grillas de regularización, solver, tolerancias, escalado fold-local y, principalmente, cómo convertir cada score OOF en overrides antes de calcular regret (`plan:23`, `plan:39`, `plan:103`). Tampoco se define si scalar, advantage-only y shuffled son candidatos seleccionables o sólo controles. Esto deja grados de libertad para selección retrospectiva encubierta. Hace falta un GroupKFold anidado o cross-fitting completamente congelado, con grillas, operating-point interno y desempates.

3. **ALTO: la cronología oracle está bien formulada conceptualmente, pero no tiene todavía un contrato ejecutable heredable.** Ola 56 exige apertura secuencial fit → select → monitor (`plan:53`), mientras Ola 55 materializa juntos los labels de selección y monitor antes del freeze de selección (`prepare_wave55_fresh.py:502`); el runner sólo demora la carga (`run_wave55_policy_bridge.py:437`). Copiar esa estructura violaría Ola 56. Deben congelarse estados y tests que prueben ausencia física del oracle/bundle siguiente antes de cada freeze. Además, el plan resuelto posterior a Stage 0 necesita una segunda auditoría independiente antes de extraer claves; la auditoría actual no puede validar una familia, config y código todavía desconocidos.

4. **MEDIO: `selector_sensitive` es vinculante pero está subdefinido.** No queda establecido si cada shard re-selecciona también scalar/advantage-only, si se compara shard A contra B o ambos contra el selector completo, cuáles son exactamente los contrastes ni la tolerancia algebraica (`plan:129`, `plan:187`). Ola 55 sí congelaba siete contrastes y tolerancia explícitos (`plan Ola 55:146`).

5. **MEDIO: excluir `policy_id` no elimina identidad ni demuestra transporte entre políticas.** El vector de cuatro utilidades es una codificación biyectiva de las 24 políticas: son todas las permutaciones de cuatro niveles (`policy_manifest.json:167`, `policy_manifest.json:173`). El split sólo por token evalúa políticas ya vistas. No es leakage para el estimando restringido a esas 24 políticas, pero contradice la justificación de transportabilidad (`nota:112`). Debe limitarse explícitamente el claim o añadir una sensibilidad leave-policy-group-out.

6. **MEDIO: faltan estados de no-evaluabilidad y artefactos ejecutables.** El plan no fija mínimos de desacuerdos, clases por fold para logística ni observaciones por shard; con una partición degenerada, regresión logística o cuantiles pueden fallar sin una salida predeclarada y el no-redraw impide corregirlo (`plan:99`, `plan:109`). También deben preservarse explícitamente pesos por token, schema/orden/dtypes de la matriz de diseño, mappings de shuffle, todas las grillas y convergencia de candidatos, método exacto de cuantiles y receipts oracle por fase.

## Veredicto

`REVISE`. No encontré criterios algebraicamente imposibles ni leakage directo en las features si se respeta la cronología, pero los dos defectos altos de selección y shuffled invalidan una ejecución prospectiva en el estado actual. La suite Wave 55 vigente pasa: `14 passed`. No se editaron archivos.
