# R318 - reauditoria focal independiente de implementacion Stage 0, Ola 56

> **Fecha:** 2026-09-03
> **Alcance:** plan Wave 56, R317, runner Stage 0, primitivas, config y tests focales.
> **Restriccion:** auditoria read-only; no se modifico codigo.
> **Veredicto:** `REVISE`

## Findings materiales

### Alta

1. **R317-3 sigue abierto en el estado real del repositorio: el entrypoint no pasa su propio freeze de fuentes.** El orden fue corregido: `require_sources_at_head` y todos los hashes upstream corren antes de `prepare_output`, por lo que el fallo ya no crea ni archiva el output (`run_wave56_retrospective.py:805-817`). Sin embargo, `git ls-files --error-unmatch` falla para los cinco paths exigidos por el preflight: runner, primitivas, plan, nota y config. Runner, primitivas y tests figuran `??`; plan, nota y config estan excluidos por `.gitignore`. Una invocacion directa aborto en `run_wave56_retrospective.py:103` y confirmo que el output permanecio ausente. Hasta integrar y congelar esos archivos en Git, Stage 0 no es ejecutable bajo el contrato implementado.

### Media

2. **R317-5 esta resuelto solo parcialmente: la sensibilidad `all-in-catalog` no repite el pipeline completo ni todos sus contrastes.** Se vuelve a ejecutar folds, seleccion de hiperparametros, operating point, evaluacion de familias, advantage-only y gate escalar sobre todos los tokens (`run_wave56_retrospective.py:840-846`). Pero los cinco shuffled null se ejecutan exclusivamente para `primary` (`run_wave56_retrospective.py:848-906`), y `delta_signs` solo compara contextual contra hard y escalar (`run_wave56_retrospective.py:1051-1079`): omite signos contra advantage-only, shuffled-promedio y pure-joint. Esto no contamina la seleccion primaria, pero no satisface la repeticion completa ni permite determinar todos los cambios de signo requeridos por el plan (`WAVE_56_CONTEXTUAL_RESIDUAL_GATE_PLAN.md:32-34`).

3. **R317-2 queda incompleto para LPGO: el calculo es reproducible por rerun, pero el artefacto no permite auditar la seleccion de cada split desde estado preservado.** Para cada grupo se guardan solo familia/params/q elegidos, modelo full final, resumen `dev_eval` y una vista reducida de candidatos (`run_wave56_retrospective.py:1001-1016`); el NPZ conserva unicamente las metricas heldout agregables (`run_wave56_retrospective.py:1017-1023`). No se preservan por grupo los OOF scores, fold-local scalers/coeficientes, objetivos de la grilla completa ni curvas de thresholds de cada familia. Es una brecha material frente al contrato general de artefactos (`WAVE_56_CONTEXTUAL_RESIDUAL_GATE_PLAN.md:95-102`) y dificulta verificar post hoc que las 8 politicas heldout no participaron sin volver a ejecutar codigo y fuentes.

## Resolucion focal de R317

1. **R317-1, resuelto.** `action_metric_arrays` produce arrays por token despues del promedio de 24 politicas; el runner toma los tokens primarios, genera una unica matriz PCG64 de 5.000 remuestras y reutiliza exactamente esos indices en todos los brazos y metricas (`run_wave56_retrospective.py:1025-1049`). Para shuffled, primero promedia por token los arrays de los cinco modelos y solo despues entra al mismo bootstrap (`run_wave56_retrospective.py:898-906`). Se preservan indices y diagnosticos de overrides (`run_wave56_retrospective.py:411-461`).

2. **R317-2, parcialmente resuelto.** Las rutas primaria, all-in-catalog y shuffled conservan inputs raw por seed, targets, design, gains, acciones, pesos, folds, OOF de cada candidato, estados fold-local/full, thresholds, curvas, mappings y metricas en JSON/NPZ (`run_wave56_retrospective.py:829-835`, `:647-701`, `:848-906`). `compare_reference` exige igualdad exacta de los tres JSON analiticos y del NPZ completo, con `equal_nan=True` solo para arrays flotantes (`run_wave56_retrospective.py:782-797`). Persiste la brecha LPGO del finding 3.

3. **R317-3, parcialmente resuelto.** El preflight ya precede al output y el test focal lo cubre (`test_wave56_contextual_gate.py:177-204`), pero las fuentes congeladas siguen fuera del indice Git; ver finding 1.

4. **R317-4, resuelto sin leakage material observado.** En cada rotacion, `params` y `q` se eligen con `dev_fit` y solo 16 politicas (`run_wave56_retrospective.py:914-954`); la familia se elige con metricas `dev_eval` restringidas a esas mismas 16 (`run_wave56_retrospective.py:955-987`); las 8 heldout entran en el resultado recien despues de fijar familia/params/q (`run_wave56_retrospective.py:988-1016`). Los pesos se recalculan como `1/d_t` dentro de las 16 politicas de fit (`run_wave56_retrospective.py:917-920`).

5. **R317-5, parcialmente resuelto.** Existe seleccion all-in-catalog real, pero faltan null shuffled y contrastes completos; ver finding 2.

6. **R317-6, resuelto.** Los hiperparametros usan tolerancia `1e-12` sobre grillas ordenadas por mayor regularizacion y menor epsilon (`run_wave56_retrospective.py:235-245`, `:374-379`); operating point y familia aplican la tolerancia congelada (`run_wave56_retrospective.py:537-541`, `:759-767`); logistica queda `NOT_EVALUABLE` si `dev_eval` tiene una clase (`run_wave56_retrospective.py:647-651`).

## Verificacion

- `pytest` focal Wave52-Wave56: **67 passed**.
- `pytest tests/test_wave56_contextual_gate.py -q`: **12 passed**.
- La ejecucion directa independiente no pudo alcanzar el end-to-end por el finding 1; no existe un output Wave56 persistido bajo el repo para inspeccionar. Por eso no se adopta como evidencia independiente el primary+replay informado por el coordinador.
- JSON estricto rechaza no finitos antes de escribir (`run_wave56_retrospective.py:79-92`), y el test dedicado paso (`test_wave56_contextual_gate.py:238-243`). Los arrays NPZ usan `allow_pickle=False` en lectura y contienen `pair_token`, mascaras y targets suficientes para recuperar el orden de la inferencia principal.

## Veredicto

`REVISE`: integrar las cinco fuentes congeladas para que el preflight de `run_wave56_retrospective.py:805` sea satisfacible; completar la sensibilidad all-in-catalog en `run_wave56_retrospective.py:840-906`; y preservar el estado de seleccion LPGO actualmente reducido en `run_wave56_retrospective.py:1001-1023`.
