# R335 — Reauditoría independiente del preflight de grafos proporcionales

- Fecha: 2026-09-03
- Rol: auditor independiente, sólo lectura
- Task: `task_f502e58ceda0`
- Dispatch: `ctx_894c8a8f6122`
- Alcance: cierre de findings de R334 y diagnóstico temporal de 256 maestros
- Resultado: `REVISE`

## Informe verbatim

Audité en modo read-only el roadmap, contrato, runner, config y tests; la suite dio 10 passed y el diagnóstico canónico produjo 320 vistas desde 256 masters en 4,86 s, con 0/320 fallos IRLS, replay byte-exacto, 1927/1927 artefactos hasheados y temporales eliminados. Los findings previos quedaron resueltos en objetivo Huber fijo y convergencia (src/geometria_proporcional/proportional_graph_contract.py:524), exclusión de no convergidos y failure rate (experiments/geometria_proporcional/run_proportional_graph_preflight.py:156), replay/config/entorno/hashes (experiments/geometria_proporcional/run_proportional_graph_preflight.py:284), máscaras (src/geometria_proporcional/proportional_graph_contract.py:267 y :469), master/view/lineage y pairing (src/geometria_proporcional/proportional_graph_contract.py:372), calibration split (:334), target-authority (experiments/geometria_proporcional/run_proportional_graph_preflight.py:243), ciclos observados/limpios (:108), permutación y frontera público/privado, pero apareció un ALTO nuevo: la config canónica está ignorada por .gitignore:26 y no está versionada, mientras el runner la abre incondicionalmente en experiments/geometria_proporcional/run_proportional_graph_preflight.py:298, de modo que un clon limpio pierde la receta y falla al construir el manifest; veredicto REVISE, sin GO científico. Resta versionar explícitamente esa config mediante excepción o alta forzada, verificarla desde un checkout limpio y entonces repetir el test focal; no modifiqué archivos del repo.
