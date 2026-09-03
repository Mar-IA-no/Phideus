# R334 — Auditoría independiente del preflight de grafos proporcionales

- Fecha: 2026-09-03
- Rol: auditor independiente, sólo lectura
- Task: `task_f52702ade297`
- Dispatch: `ctx_a00a4058e486`
- Alcance: contrato, runner, configuración y tests del preflight CPU
- Resultado: `REVISE`

## Informe verbatim

Audité read-only los cuatro artefactos contra el roadmap, ejecuté la suite aislada y un diagnóstico completo de 256 masters más probes de máscaras y replay; los 7 tests existentes pasan y no modifiqué archivos. ALTO — sólo 88/320 solves Huber-IRLS convergen pero los 232 no convergidos se promedian y el informe omite failure rate (src/geometria_proporcional/proportional_graph_contract.py:460-503; experiments/geometria_proporcional/run_proportional_graph_preflight.py:127-149,168-199; roadmap:145-148,198,286-289); ALTO — replay ignora la config resuelta y el manifest no preserva entorno ni hash del constructor, agravado porque runner/contrato están untracked y HEAD no identifica su contenido (runner:45-58,230-258; roadmap:239-241,329-334); MEDIO — edge_valid y path_valid son ignorados por WLS y closure, demostrado por un edge/path inválido que aún produjo x_hat=[-33.33,33.33,0] y residual 100 (contract.py:256-261,423-435; roadmap:104-105,156-157,235-240); MEDIO — master_id es view-specific aunque el master inferencial compartido se llama lineage_id, falta target-authority table/calibration split/permutation sanity, y el único cycle diagnostic global se calcula sobre clean_log_ratio tautológicamente cerrado (contract.py:309-377; runner:104-124,153-165; roadmap:243-253,280-303); tests faltantes materiales: IRLS/convergencia, máscaras, permutación, replay/manifest, pares con ruido/amplitud realmente compartidos y cobertura de todos los artefactos (tests/test_proportional_graph_contract.py:37-55,80-127,130-153). PASS — orientación Bx=x_j-x_i, signos de paths e inversión coherente, gauge media-cero, rango conexo, lineages sin cruce de split, pairing iid/grouped, frontera de campos públicos, oracle privado y determinismo de arrays quedaron verificados; resta corregir los defects ALTO y cubrir los MEDIO antes de considerar validado este preflight.
