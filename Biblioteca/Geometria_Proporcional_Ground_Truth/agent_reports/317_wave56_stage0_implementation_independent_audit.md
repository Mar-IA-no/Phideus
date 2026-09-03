# R317 — auditoría independiente de implementación de Stage 0, Ola 56

> **Agente:** `01a0674a-5c47-7d00-9c33-bb212b850930` (`Bohr`)
> **Fecha:** 2026-09-03
> **Veredicto:** `REVISE`
> **Preservación:** informe del subagente, verbatim.

## Alta

1. **La inferencia bootstrap está completamente ausente.** La configuración congela `5.000` remuestras por `pair_token` en `wave56_contextual_gate.json:70`, pero el runner nunca consume ese bloque ni guarda índices, IC95 o contrastes pareados. Solo produce estimaciones puntuales en `run_wave56_retrospective.py:512` y `run_wave56_retrospective.py:723`. También faltan precisión/recall y magnitud de overrides beneficiosos/perjudiciales exigidos por el plan.

2. **La serialización no permite auditar ni reproducir el fit completo.** `crossfit_fixed` descarta los escaladores y coeficientes fold-locales; la serialización elimina los scores de cada candidato y conserva solo el modelo full seleccionado en `run_wave56_retrospective.py:295`, `:530` y `:562`. Tampoco se preservan targets, inferencia raw por seed, hard/posterior actions, overrides, shuffled OOF/scores/modelos, bootstrap o schema con dtype/normalización/dominio. `compare_reference` puede declarar replay exacto comparando únicamente tres artefactos incompletos en `run_wave56_retrospective.py:473`.

3. **El entrypoint no es ejecutable en el estado actual.** Los cinco paths entregados a `require_sources_at_head` están `NOT_TRACKED`; plan, nota y config además requieren `git add -f` por `.gitignore`. El runner los rechaza en `run_wave56_retrospective.py:79`. Además crea o archiva el output antes de ese preflight en `run_wave56_retrospective.py:486`, por lo que un fallo de validación deja un directorio que bloquea el siguiente intento.

## Media

4. **La sensibilidad leave-policy-group-out no es realmente held-out.** La familia se selecciona usando resultados `dev_eval` de las 24 políticas en `run_wave56_retrospective.py:600`; luego la sensibilidad reutiliza esa familia, hiperparámetros y el cuantil elegido con las 24 políticas para evaluar los grupos supuestamente retenidos en `run_wave56_retrospective.py:674`. Esto contamina esa sensibilidad, aunque no el OOF primario.

5. **Falta completamente la sensibilidad sobre todos los tokens in-catalog.** El runner solo ejecuta `primary`; no repite selección ni compara familia/signos como exige el plan. Una reproducción manual mantuvo `ridge_contextual`, pero cambió `alpha=1` a `alpha=100`; el artefacto oficial no informa esto ni los signos.

6. **Las reglas de borde de model selection están incompletas.** Los desempates de hiperparámetros y familia usan mínimos exactos, no la tolerancia `<=1e-12`. Tampoco se marca logística `NOT_EVALUABLE` cuando `dev_eval` tiene una sola clase de gain. Los datos actuales sí tienen ambas clases, por lo que este último defecto es latente.

## Veredicto

`REVISE`. El weighting `1/d_t`, los folds por token, el escalado fold-local, el threshold estricto y el shuffle `(policy_index,d_t)` son coherentes; la corrida real tiene `95,19%` de filas shuffled movibles. Pasaron `60/60` tests Wave52–Wave56 y la selección ejecutada en memoria produjo `ridge_contextual`, pero los tests Wave56 sólo cubren primitivas, no runner, bootstrap, selección, sensitivities ni replay. No se editaron archivos.
