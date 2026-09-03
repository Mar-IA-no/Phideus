# Wave 55 implementation independent audit

## Findings

1. **HIGH: el replay puede autocertificarse sin regeneración independiente.** `--reference-dir` puede apuntar al propio output o una segunda adjudicación puede reutilizar el bundle primario. Las comparaciones no exigen paths distintos, uso conjunto de `--replay-secrets-from`, ni un receipt válido de replay de preparación. Así, `replay_exact=true` puede satisfacer el criterio 5 sin reproducir el generador con las claves preservadas. Tampoco se calcula el hash analítico canónico exigido por el plan. `prepare_wave55_fresh.py:318`, `prepare_wave55_fresh.py:446`, `run_wave55_policy_bridge.py:327`, `run_wave55_policy_bridge.py:489`.

2. **HIGH: la política de corrida primaria única y no-redraw no está ejecutablemente protegida.** El output es arbitrario; `--force` archiva uno existente y luego puede sortear claves nuevas. Tras un fallo, el intento se renombra a `.failed_*`, dejando libre el nombre original; una nueva invocación sin `--replay-secrets-from` vuelve a sortear claves. No se detectan intentos previos ni se obliga a recuperar sus secretos. `prepare_wave55_fresh.py:68`, `prepare_wave55_fresh.py:112`, `prepare_wave55_fresh.py:367`, `prepare_wave55_fresh.py:450`.

3. **MEDIUM: el diagnóstico de soporte ausente quedó incompleto.** Sólo se registra índice, conteo agregado y `EVALUABLE/NOT_EVALUABLE`; cuando alcanza `n_min=30` no se calcula desempeño ni intervalo, pese al contrato del plan. Por tanto, el estado `EVALUABLE` no viene acompañado de la evidencia que habilitaría interpretar ese soporte. `run_wave55_policy_bridge.py:435`.

4. **MEDIUM: los theta no se verifican semánticamente contra el freeze Wave 54.** Se validan los hashes independientes, pero luego se leen directamente `theta__*` de `selection_state.npz`; no se comprueba igualdad con `selected_models[*].theta` de `selection_freeze.json` ni que correspondan a la selección `primary` y no a sensibilidad global. Los hashes literales reducen el riesgo de mutación, pero no implementan la verificación cruzada exigida. `run_wave55_policy_bridge.py:350`, `run_wave55_policy_bridge.py:359`.

5. **MEDIUM: los diagnósticos de override se reportan sobre todo el monitor, no sobre la población primaria.** A diferencia de resúmenes y contrastes, `override_rate`, precisión y regret condicionado no reciben `primary_mask`. Esto mezcla estratos fuera del estimando principal y puede cambiar la lectura diagnóstica del puente. `run_wave55_policy_bridge.py:470`.

## Verificación

`venv/bin/pytest -q tests/test_wave55_policy_bridge.py`: **9 passed**. Los cuatro scripts compilan.

Los tests residuales no cubren cronología integral, no-redraw/recuperación, independencia del replay, referencias autocontenidas, validación theta-freeze, ramas de soporte `n_min`, sensibilidad completa, máscaras de override ni ejecución end-to-end. Resultado: **REVISE**, no PASS. No hice cambios.
