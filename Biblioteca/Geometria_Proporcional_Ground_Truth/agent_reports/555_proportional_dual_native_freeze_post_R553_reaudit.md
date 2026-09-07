REVISE — 2 HIGH / 1 MEDIUM / 0 LOW.

## Findings

### HIGH 1 — El checker raíz acepta falsificar el resultado K192 y promover el estado técnico a `BOTH_DESIGN_FREEZES_VALID`

`check_artifact` toma `numeric_status` del propio reporte y lo usa para recomponer R11 ([checker:1577](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1577)); luego deriva el `design_state` desde esos predicados ([checker:1596](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1596)). La inspección del NPZ sólo verifica roster, longitudes, conteos, máximo Torch–NumPy y máximo RMSE ([checker:1600](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1600)). No deriva nuevamente:

- `status` desde convergencia y umbrales;
- p99 RMSE;
- `all_canonical_converged`;
- cobertura `8..16`;
- métricas de gradiente.

Reproducción CPU en un árbol temporal:

1. Dejé intacto `fixed_depth_raw.npz`, que conserva `191/192` convergencias y RMSE fuera de umbral.
2. Cambié en ambos reportes sólo `fixed_depth_conformance.status=PASS`, R11 a PASS, conteos a `30/0` y estado a `BOTH_DESIGN_FREEZES_VALID`.
3. Recalculé los hashes deterministas de manifests y replay.
4. `check_root_artifact(...)` devolvió:

```text
{'status': 'PASS', 'reasons': []}
```

El checker raíz delega en ese chequeo incompleto y sólo recompone hashes/replay ([checker:1681](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1681)). Esto contradice la exigencia de auditoría independiente del cierre ([plan:647](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:647)) y la afirmación de R554 de que el reporte científico queda recompuesto desde el raw ([R554:36](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/554_proportional_dual_native_freeze_post_R553_analysis.md:36)).

### HIGH 2 — `51/51` no cubre aún el contrato congelado: cambios materiales adicionales pasan todos los predicados

Las 51 mutaciones sí incluyen y detectan las reproducciones concretas de R553 ([checker:1277](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1277)). Sin embargo, varios predicados sólo validan subconjuntos de las recetas completas:

- R5 no verifica `tolerance=1e-6` ([checker:413](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:413)).
- S4 no verifica solver, penalización, intercept, class weights, seed, `max_iter` o dtype de MARGINAL, ni optimizer/selection key completos de JOINT ([checker:521](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:521)).
- S5 no verifica `utility_tie`; la fixture HARD sólo diferencia códigos de set MAP/threshold, no ejecuta la selección de familia por utilidad ni su desempate ([checker:534](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:534), [checker:1085](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1085)).
- S7 no verifica `alpha`, dtype, grids de cuantiles ni la clave completa de selección ([checker:551](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:551)).
- S1 ignora `created_after_freeze_commit`, aunque la frescura es parte causal del freeze ([set config:28](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/configs/proportional_set_valued_native_freeze_v1.json:28)).
- Ningún predicado adjudica la coherencia de costo; `scientific_payload` sólo copia `projected_cost` ([checker:1480](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1480)).

Reproduje siete mutaciones independientes; todas dejaron la lista de predicados fallidos vacía:

```text
R5_TOLERANCE_CHANGED          []
S5_UTILITY_TIE_CHANGED        []
S4_MARGINAL_SOLVER_CHANGED    []
S7_RIDGE_ALPHA_CHANGED        []
S7_SELECTION_KEY_CHANGED      []
S1_FRESHNESS_DISABLED         []
SET_COST_MISCLASSIFIED        []
```

Varias alteran directamente el protocolo set-valued que hoy recibe el rótulo válido. Por ello `SET_VALUED_FREEZE_ONLY_VALID` todavía no queda suficientemente acreditado, aunque las 51 mutaciones existentes pasen.

### MEDIUM 1 — La cobertura de source bindings sigue por debajo del mínimo literal del plan

El plan exige ligar, entre otras fuentes, config/compute/runtime del smoke, schemas W49 e implementación/manifests del control matched W59 ([plan:518](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:518)). Sin embargo:

- la rama relacional liga runner y R342, pero no config, compute contract ni runtime del smoke ([relational config:6](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/configs/proportional_relational_native_freeze_v1.json:6));
- la rama set-valued comienza en W51 y termina en W57, sin W49 ni W59 ([set config:6](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/configs/proportional_set_valued_native_freeze_v1.json:6));
- `REQUIRED_BINDING_PATHS` sólo obliga plan/R547/R549/R551, por lo que C1 no detecta esas omisiones ([checker:111](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:111)).

La procedencia efectivamente ligada sí es consistente: `source_commit=dd618719...` reproduce byte a byte los seis critical implementation paths, los hashes de las tres configs coinciden y el worktree está limpio.

## Verificaciones que sí pasan

- `CUDA_VISIBLE_DEVICES='' venv/bin/python -m pytest -q tests/test_proportional_dual_native_freeze.py tests/test_proportional_graph_irls_surrogate_fidelity.py` → `16 passed in 30.43s`.
- Checker del artefacto raíz vigente → `PASS`.
- Artefacto nominal: `29 PASS / 1 FAIL`; único fallo R11; `51/51` mutaciones existentes y `4/4` fixtures pasan.
- K192 conserva correctamente `64` grafos, `192` estados, tamaños `8..16`, `191/192` convergencias y rechazo relacional.
- Hashes canónicos y replay coinciden exactamente con R554.
- `superseded_pre_R553_checker` conserva `13/13` archivos byte-exactos respecto del artefacto original de `cad2fee`; R554 lo declara explícitamente sustituido ([R554:9](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/554_proportional_dual_native_freeze_post_R553_analysis.md:9)) y el manifest raíz lo excluye del estado vigente.
- No hubo uso ni consulta de GPU. Config, runtime y reportes mantienen `gpu_used_or_queried=false`, `architecture_promoted=false`, `scientific_decision=null` y autoridad del usuario.
- No se realizaron ediciones.
