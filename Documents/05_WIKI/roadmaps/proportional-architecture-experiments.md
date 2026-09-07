---
schema_version: 1
id: phideus-proportional-architecture-experiments
kind: roadmap
page_status: current
front_status: decision_ready
architecture_status: candidate
experiment_status: dual_native_preflight_closed_set_valued_freeze_only_valid
evidence_status: the relational K192 surrogate failed its fresh canonical approximation while the set-valued freeze passed; 29/30 predicates, 61/61 mutations, exact replay, and R557 PASS 0/0/0 without promotion or GO/NO-GO
decision_status: pending_user
updated: 2026-09-07
verified_at: 2026-09-07
valid_at: 2026-09-07
recorded_at: 2026-09-07
evidence_commit: e76f7d9cdb49e262575ef234502de5e4e83ac61d
source_paths:
  - Documents/05_WIKI/concepts/ppu-geometria-armonica-natural.md
  - Documents/05_WIKI/fronts/atencion-armonica.md
  - Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_17_AUDIT_RESOLUTION.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_17_PROTOCOL_DRAFT.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/332_proportional_architecture_protocol_independent_audit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/333_proportional_architecture_protocol_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/334_proportional_graph_preflight_implementation_audit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/335_proportional_graph_preflight_implementation_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/336_proportional_graph_preflight_config_focal_audit.md
  - experiments/geometria_proporcional/run_proportional_graph_preflight.py
  - experiments/geometria_proporcional/configs/proportional_graph_preflight_v1.json
  - src/geometria_proporcional/proportional_graph_contract.py
  - data/geometria_proporcional/proportional_graph_preflight_v1/PREFLIGHT_REPORT.md
  - experiments/geometria_proporcional/configs/proportional_graph_neural_smoke_v1.json
  - experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py
  - src/geometria_proporcional/proportional_graph_neural.py
  - data/geometria_proporcional/proportional_graph_neural_smoke_v1/SMOKE_REPORT.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/338_proportional_graph_neural_smoke_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/339_proportional_graph_neural_smoke_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/340_proportional_graph_neural_smoke_final_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/341_proportional_graph_neural_smoke_closure_audit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/342_proportional_graph_neural_smoke_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_SOLVER_DISENTANGLEMENT_CPU.md
  - experiments/geometria_proporcional/configs/proportional_graph_solver_disentanglement_v1.json
  - experiments/geometria_proporcional/run_proportional_graph_solver_disentanglement.py
  - data/geometria_proporcional/proportional_graph_solver_disentanglement_v1/DISENTANGLEMENT_REPORT.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/349_proportional_solver_disentanglement_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_SOLVER_INTERFACE_DIAGNOSTIC_CPU.md
  - experiments/geometria_proporcional/configs/proportional_graph_solver_interface_diagnostic_v1.json
  - experiments/geometria_proporcional/run_proportional_graph_solver_interface_diagnostic.py
  - data/geometria_proporcional/proportional_graph_solver_interface_diagnostic_v1/SOLVER_INTERFACE_REPORT.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/352_proportional_solver_interface_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_FROZEN_ADAPTERS_CPU.md
  - experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py
  - data/geometria_proporcional/proportional_graph_frozen_adapters_v1/effects.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/353_proportional_frozen_adapters_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_IRLS_SURROGATE_FIDELITY_CPU.md
  - experiments/geometria_proporcional/run_proportional_graph_irls_surrogate_fidelity.py
  - data/geometria_proporcional/proportional_graph_irls_surrogate_fidelity_v1/summary.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/354_proportional_irls_surrogate_fidelity_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_IRLS_LOSS_CONTRAST_CPU.md
  - data/geometria_proporcional/proportional_graph_irls_loss_contrast_v1/effects.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/355_proportional_irls_loss_contrast_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_RESIDUAL_GATE_CPU.md
  - data/geometria_proporcional/proportional_graph_residual_gate_v1/effects.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/356_proportional_residual_gate_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_FRESH_MIXED_GATE_CPU.md
  - data/geometria_proporcional/proportional_graph_fresh_mixed_gate_v1/effects.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/357_proportional_fresh_mixed_gate_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_SAFE_ABSTENTION_GATE_CPU.md
  - data/geometria_proporcional/proportional_graph_safe_abstention_gate_v1/effects.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/358_proportional_safe_abstention_gate_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_SAFE_ABSTENTION_POWER_AUDIT_CPU.md
  - data/geometria_proporcional/proportional_graph_safe_abstention_power_audit_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/359_proportional_safe_abstention_power_transport_audit.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_TOPOLOGY_LOCALIZATION_GATE_CPU.md
  - data/geometria_proporcional/proportional_graph_topology_localization_gate_v1/effects.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/360_proportional_topology_localization_gate_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_CONDITIONAL_RISK_GATE_CPU.md
  - data/geometria_proporcional/proportional_graph_conditional_risk_gate_v1/effects.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/361_proportional_conditional_risk_gate_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_SIGNED_TAIL_DIAGNOSTIC_CPU.md
  - data/geometria_proporcional/proportional_graph_signed_tail_diagnostic_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/362_proportional_signed_tail_diagnostic_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_SIGNED_TAIL_GATE_CPU.md
  - data/geometria_proporcional/proportional_graph_signed_tail_gate_v1/effects.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/363_proportional_signed_tail_gate_official_analysis.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_SELECTED_ACTION_CALIBRATION_DIAGNOSTIC_CPU.md
  - data/geometria_proporcional/proportional_graph_selected_action_calibration_diagnostic_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/364_proportional_selected_action_calibration_diagnostic_analysis.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/365_proportional_selected_action_transport_power_audit_analysis.md
  - data/geometria_proporcional/proportional_graph_selected_action_transport_power_audit_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/366_proportional_equal_budget_ranking_diagnostic_analysis.md
  - data/geometria_proporcional/proportional_graph_equal_budget_ranking_diagnostic_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/367_proportional_mean_only_ranking_ablation_analysis.md
  - data/geometria_proporcional/proportional_graph_mean_only_ranking_ablation_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/368_proportional_two_stage_eligibility_diagnostic_analysis.md
  - data/geometria_proporcional/proportional_graph_two_stage_eligibility_diagnostic_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/369_proportional_mean_ranking_attribution_analysis.md
  - data/geometria_proporcional/proportional_graph_mean_ranking_attribution_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/370_proportional_mean_ranking_power_selection_audit_analysis.md
  - data/geometria_proporcional/proportional_graph_mean_ranking_power_selection_audit_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/371_proportional_mean_ranking_pareto_transport_analysis.md
  - data/geometria_proporcional/proportional_graph_mean_ranking_pareto_transport_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/372_proportional_mean_ranking_pairwise_dominance_analysis.md
  - data/geometria_proporcional/proportional_graph_mean_ranking_pairwise_dominance_v1/analysis.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/373_proportional_budget_path_typed_interface_analysis.md
  - data/geometria_proporcional/proportional_budget_path_typed_interface_v1/summary.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/374_proportional_budget_path_external_utility_port_analysis.md
  - data/geometria_proporcional/proportional_budget_path_external_utility_port_v1/summary.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_PROSPECTIVE_CLOSED.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/394_wave56_stage1_execution_results_audit.md
  - data/geometria_proporcional/wave56_contextual_gate_fresh_v1/phases/adjudicate.complete/analytics.complete/REPORT_WAVE56_STAGE1.json
  - data/geometria_proporcional/wave56_contextual_gate_fresh_v1_replay/phases/adjudicate.complete/replay_receipt.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_57_CONTEXTUAL_TAIL_GUARD_CLOSED.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/405_wave57_contextual_tail_guard_execution_results_audit.md
  - data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1/phases/adjudicate.complete/analytics.complete/REPORT_WAVE57.json
  - data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1_replay/phases/adjudicate.complete/replay_receipt.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_CLOSED.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/414_wave58_open_model_class_results_audit.md
  - data/geometria_proporcional/wave58_open_model_class_diagnostic_v1/analysis.json
  - data/geometria_proporcional/wave58_open_model_class_diagnostic_v1_replay/runtime.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_FRESH_HGB_GUARD_BRACKET_CLOSED.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/454_wave59_successor_draw_final_audit.md
  - experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket_replay_normalized.json
  - data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1/analysis.json
  - data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1_replay/replay_comparison.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_CLOSED.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/531_wave60_v4_replay_normalization_correction_audit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/533_wave60_closure_and_program_handoff_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/PROGRAM_CLOSURE_MAPPING_FEASIBILITY_AND_EXPERIMENTAL_PORTFOLIO.md
  - data/geometria_proporcional/proportional_mapping_feasibility_v1/run_a/adjudication.json
  - data/geometria_proporcional/proportional_mapping_feasibility_v1/runtime.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/544_proportional_mapping_feasibility_canonical_artifact_audit.md
  - experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md
  - experiments/geometria_proporcional/configs/proportional_dual_native_freeze_v1.json
  - data/geometria_proporcional/proportional_dual_native_freeze_v1/run_a/scientific_report.json
  - Biblioteca/Geometria_Proporcional_Ground_Truth/PROGRAM_CLOSURE_DUAL_NATIVE_FREEZES_AND_SET_RUNNER_HANDOFF.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/557_proportional_dual_native_freeze_final_independent_reaudit.md
depends_on: [ppu-natural-harmonic-geometry, front-atencion-armonica]
tangents: [phideus-evidence-regime, phideus-three-routes]
---

# Programa de arquitecturas y experimentos proporcionales

## Cambio de régimen

La campaña de investigación expansiva queda detenida. El corpus local acumulado
se usa ahora como un cuerpo cerrado de diseño: su función inmediata no es abrir
otra taxonomía, sino obligar a formular módulos neuronales concretos, compararlos
con controles fuertes y producir evidencia experimental.

Los dos primeros gates del relevo ya cerraron. `MAPPING-FEASIBILITY` mostró que
EIV, el núcleo relacional y el posterior set-valued no comparten unidad,
observación, target ni stack de decisión sin alterar su semántica. El dual
native freeze conservó esa separación y resolvió cuál de las dos ramas puede
pasar a implementación: la relacional falló su confirmación K192 y la
set-valued quedó contractualmente lista para construir su runner. El router
permanece diferido.

Esto no prohíbe recuperar un paper, descargar una fuente ya identificada o
verificar un detalle de implementación. Esas consultas quedan subordinadas a
una dependencia explícita de un experimento. Las preguntas que no bloquean la
ejecución se registran como deuda y no abren una nueva ola por sí solas.

## Lo que el corpus ya permite afirmar

La evidencia disponible no sostiene una geometría universal de las
proporciones. Sí converge en restricciones de diseño suficientemente concretas:

1. las relaciones deben ser estados de primera clase, no features incidentales;
2. gauge, unidades, dominio y equivalencia deben formar parte del contrato;
3. composición relacional y decisión puntual son operaciones distintas;
4. un output puede ser una clase o un conjunto, no necesariamente un punto;
5. executor y checker exactos deben permanecer fuera del crédito neuronal;
6. el valor de una primitive sólo se atribuye frente a baselines clásicos,
   genéricos y barajados, declarando por separado qué dimensiones de capacidad
   y cómputo quedaron efectivamente igualadas;
7. la evaluación decisiva debe retener mecanismo, cardinalidad, topología o
   régimen, no limitarse a IID aleatorio.

Atención Armónica aporta evidencia previa a favor del estado explícito de pares
y, en aquel banco, de la actualización triangular combinada con un clusterer
global bajo OOD de polifonía. No aísla una ley general de composición ni
autoriza transferir el resultado fuera de esa jurisdicción. Las Olas 49–56
aportan otra separación: representar compatibilidades, modelar el posterior
sobre conjuntos y decidir bajo utilidad son problemas distintos. El posterior
conjunto mejoró NLL y cardinalidad, pero ni una decisión bayesiana pura ni una
compuerta escalar convirtieron de manera estable esa mejora en una política
superior. La compuerta contextual fresca aisló señal de regret frente a
advantage-only y shuffled, pero perdió accuracy fuera del margen y empeoró la
cola frente a la política dura. El problema queda localizado entre propuesta de
valor y autorización de riesgo, no en la mera disponibilidad de contexto.

## Cartera arquitectónica

| Línea | Estado | Primitive puesta en riesgo | Experimento discriminante |
|---|---|---|---|
| Núcleo relacional tipado con adaptación por executor | freeze rechazado por R11; preservado, no promovido | estado de relaciones orientadas y composicionales útil más allá de WLS | reformular el surrogate o la interfaz sólo bajo un plan nuevo; no retunear K sobre la confirmación abierta |
| Posterior de conjuntos con política y guard separados | design freeze válido sólo para implementar runner; no promovido; bracket HGB/HGB cerrado como prioridad | incertidumbre conjunta y autorización de acción desacopladas | implementar y auditar el runner CPU `MARGINAL/JOINT × HARD/CONTEXTUAL` antes de abrir el draw fresco |
| Router tipado con IR, executors y checkers externos | integración condicionada | dispatch y abstención sin fusionar autoridad, solver y decisión | sólo si una primitive estrecha sobrevive al contraste |

Las tres líneas preservadas no tienen el mismo rango. `MAPPING-FEASIBILITY`
rechazó una IR operacional común bajo los contratos vigentes y validó las dos
hojas nativas. El dual native freeze posterior no produjo dos permisos de
ejecución: rechazó la rama relacional por su confirmación K192 y habilitó sólo
la implementación del runner set-valued. El lector de
espectro relativo permanece como deuda matemática fuera de la shortlist: carece
de una query externa no agotada por el eigensolver y no justifica abrir una
cuarta rama experimental.

## Arquitectura 1: núcleo local de coherencia proporcional sobre grafos

### Objeto, orientación y simetría

El banco parte de cantidades positivas latentes `q_i`. En coordenadas
logarítmicas, `x_i = log(q_i)`, una razón observada sobre una arista orientada es

```text
y_ij = x_j - x_i + ruido + corrupción.
```

El objeto identificable no es el vector absoluto `x`, sino su clase bajo
`x -> x + c`. Esa traslación global en log-espacio es la reescala común de las
cantidades originales. El modelo debe respetarla por construcción y ser
equivariante a relabeling de nodos.

Se fija una convención orientada única. Para cada arista canónica `e=(i,j)`, la
fila `e` de la matriz de incidencia `B` contiene `-1` en `i` y `+1` en `j`, de
modo que `(Bx)_e = x_j-x_i`. Cuando un camino recorre la arista al revés se usa
`y_ji=-y_ij`; no se duplica la observación como si fuera otra muestra.

### Forward mínimo

```text
batch público de relaciones observadas + máscara + covarianza autorizada
        |
        +--> track RAW: operandos observados
        |
        +--> track RAW+EXACT-CLOSURE: mismos operandos + r_ijk exacto
        |
        v
encoder compartido de aristas
        |
        +--> mixer GENERIC
        |
        +--> mixer COMPOSITION-TYPED
        |
        v
relación corregida y_tilde + peso/confianza por arista
        |
        +--> métricas pre-solver de atribución
        |
        +--> WLS congelado / Huber-IRLS congelado
                    |
                    v
             potenciales módulo gauge
                    |
                    v
             checker exacto externo
```

El experimento no atribuye automáticamente al mixer el cálculo analítico de un
residual. Separa dos factores: qué evidencia recibe y cómo la mezcla. La red
aprende una corrección de relación `y_tilde_e` y un peso positivo `w_e`; no
aprende a resolver el sistema lineal. Cada solver externo aplica la misma
interfaz a todos los brazos. Para WLS:

```text
x_hat = argmin_{mean(x)=0} ||W^(1/2) (B x - y_tilde)||².
```

`w_e = epsilon + (1-epsilon)*sigmoid(s_e)`, con `epsilon` fijado antes de test y
normalización `mean_e(w_e)=1`. El grafo público es conexo; el checker exige
`rank(B)=n-1`, registra la condición del Laplaciano ponderado y rechaza como
fallo numérico cualquier solve que no satisfaga el contrato. El segundo solver
usa Huber/IRLS con receta y tuning congelados en validación. Las salidas
pre-solver se leen antes de los resultados downstream, y la interacción
`brazo x solver` se reporta explícitamente.

### Factorial causal y bloque candidato

El contraste principal cruza dos niveles de evidencia y dos mixers:

| Factor | Nivel | Contenido |
|---|---|---|
| evidencia | `RAW` | `y_ij`, `y_ik`, `y_kj`, roles, máscara y covarianza pública; ninguna operación exacta multioperando |
| evidencia | `RAW+EXACT-CLOSURE` | lo anterior más `r_ijk = y_ij-(y_ik+y_kj)` computado externamente |
| mixer | `GENERIC` | MLP/atención libre con roles identificados |
| mixer | `COMPOSITION-TYPED` | weight sharing, orientación y antisimetría tipadas; en `RAW` no puede calcular por fórmula suma de caminos ni cierre |

Esto produce `RAW-GENERIC`, `RAW-TYPED`, `CLOSURE-GENERIC` y
`CLOSURE-TYPED`, más `EXACT-CLOSURE-ONLY` sin parámetros. Si una versión tipada
necesita recibir la suma `y_ik+y_kj` o la discrepancia exacta, pertenece al
track `CLOSURE`; el comparador genérico recibe exactamente el mismo tensor. El
efecto del residual, el efecto del mixer y su interacción se estiman por
separado. Sólo la diagonal `RAW-GENERIC` frente a `CLOSURE-TYPED` acredita el
paquete completo.

En `RAW`, el bloque tipado sólo puede imponer roles, signo bajo inversión,
weight sharing y equivariancia. Una forma mínima es:

```text
p_ijk = MLP_path([z_ij, z_ik, z_kj, orientation_roles])
a_ijk = softmax_k(score(p_ijk))
m_ij  = sum_k a_ijk * p_ijk
z'_ij = z_ij + Gate(z_ij, m_ij)
y_tilde_ij, s_ij = Heads(z'_ij)
```

En `RAW+EXACT-CLOSURE`, `r_ijk` se concatena al mismo `p_ijk` para ambos mixers.
Los shuffles de paths reasignan conjuntamente los operandos `(ik,kj)` dentro de
estratos congelados y recomputan `r_ijk`; nunca barajan sólo el residual dejando
operandos contradictorios. Este v0 sólo reclama integración **local de caminos
de dos saltos**. Grafos sin triángulos o con girth alto son controles de
cobertura; los ciclos largos requieren un mecanismo recurrente propio y otra
ablación.

### Salidas y tracks de aprendizaje

El primer corte no incluye router multigeometría, política de abstención ni
calibrador aprendido. Produce:

- relación corregida `y_tilde_e` y peso `w_e` por arista;
- probabilidad de procedencia causal de la alteración sólo en el track
  supervisado que la autoriza;
- potenciales reconstruidos módulo media cero, producidos por cada solver
  externo;
- diagnósticos de cierre, condición numérica y fallo del solve.

Hay dos tracks de entrenamiento que no comparten interpretación:

1. `RECONSTRUCTION-ONLY`: `L_quotient` a través del solver diferenciable más
   `lambda_rel*L_relation` y `lambda_cycle*||C y_tilde||_1`; no ve la máscara
   causal.
2. `CAUSAL-PROVENANCE-SUPERVISED`: añade BCE sobre la máscara que el simulador
   alteró. AP/Brier significan recuperación y calibración de esa procedencia,
   no detección ontológica de una única “arista falsa”.

`C` es una base de ciclos del grafo observado y el término de cierre se calcula
sobre `y_tilde`; calcularlo sobre `Bx_hat` sería idénticamente cero y hacerlo
sobre `y` crudo no dependería de la red. `L_relation` compara `y_tilde` con la
relación limpia sólo en el banco sintético y se elimina en una transferencia
sin esa autoridad. Los `lambda` se fijan en validación. Brier, ECE y
riesgo-cobertura se reportan, pero no se añade una `L_calibration` indefinida.

Los mundos con varias descomposiciones observacionalmente equivalentes forman
un slice `NONIDENTIFIABLE`: no reciben accuracy binaria. Se conserva su clase o
posterior certificado y se puntúa cobertura o proper scoring; los casos cuyo
checker no resuelve quedan `COMPUTATIONALLY-UNRESOLVED` y fuera del claim de
localización.

## Experimento 1: benchmark de coherencia local y gauge

### Pregunta

¿Un mixer con orientación y antisimetría tipadas corrige relaciones mejor que
un mixer genérico cuando ambos reciben la misma evidencia? ¿Qué parte de la
mejora proviene de entregar un cierre analítico exacto? ¿La salida conserva
utilidad bajo dos solvers externos y bajo un cambio de mecanismo de corrupción?

### Generación y frontera anti-leakage

Cada master contiene un grafo conexo, potenciales latentes, razones exactas y
una observación parcial contaminada. El generador conserva el estado limpio en
un sidecar privado. El batch público tiene una whitelist cerrada: índices
locales reindexados, `y_e` observado, `edge_exists`, `edge_valid`,
`path_incidence_valid` y covarianza generada independientemente de la máscara
causal. Quedan excluidos `q`, `x_true`, razones limpias, máscara causal,
mecanismo, seed, lineage, split, IDs persistentes y salidas oracle. Un test de
schema compara todos los brazos y persiste hashes del constructor y la
whitelist.

Todas las vistas, corrupciones y controles derivados de un mismo mundo
conservan una lineage única y nunca cruzan train, calibration, validation o
test. El master, no la arista ni el camino, es la unidad inferencial.

| Split | Cambio retenido |
|---|---|
| ID | tamaños, familias topológicas y mecanismo vistos |
| OOD-size | más nodos, misma familia, densidad y mecanismo |
| OOD-topology | familias generadoras completas retenidas; tamaño y corrupción igualados |
| OOD-corruption primario | dependencia de outliers IID hacia agrupada, con tasa, amplitud, tamaño y topología igualados |
| Gauge/permutation sanity | reescala latente común y relabeling pareados; no son nuevas muestras ni splits OOD |

El benchmark es sintético y sólo puede validar la mecánica. No acredita una
geometría física natural.

### Brazos y controles obligatorios

| Brazo | Función causal |
|---|---|
| mínimos cuadrados sin pesos | piso exacto no robusto |
| Huber/IRLS | baseline clásico robusto |
| factorial `RAW/CLOSURE x GENERIC/TYPED` | separa evidencia analítica, mixer e interacción |
| `EXACT-CLOSURE-ONLY` | mide cuánto agota el target la primitive exacta sin red |
| MLP por arista | evidencia local sin composición |
| pair-state sin mezcla | efecto de representar relaciones explícitamente |
| message passing genérico, igualado en parámetros/shapes/inicialización pero no en FLOPs | capacidad global sin tipado proporcional; no aísla tipado por sí solo |
| path-incidence shuffle balanceado | falsación causal de la incidencia de caminos |
| orientación coherentemente invertida | sanity de convención; debe transformar, no degradar |
| direct decoder centrado | diagnóstico pre/post-executor; no entra al contraste del mixer |
| pesos oracle | referencia privilegiada, nunca baseline deployable |

Los cuatro brazos factoriales comparten encoder, reader, ancho, profundidad,
datos y presupuesto entrenable. El manifiesto congela parámetros, FLOPs
entrenables, operaciones exactas, latencia, memoria y espacio de tuning. Los
solvers reciben la misma interfaz y se cruzan con cada brazo; ningún lift
post-solver se atribuye por reflejo al núcleo.

### Métricas y unidad inferencial

- superficie primaria de atribución: error de relación corregida y score de
  confianza, por master y autoridad del target;
- superficie primaria de utilidad: RMSE de potenciales en el cociente por
  master, reportada factorialmente por solver;
- secundarias: AP/Brier de procedencia causal, residual por longitud, error de
  ratios reconstruidos, condición numérica y failure rate;
- sanities: invariancia ante reescala latente, equivariancia ante permutación y
  orientación, y exactitud de los solvers en datos limpios;
- bootstrap pareado por master y resultados por seed antes del ensemble.

La inferencia se formula por split. `OOD-corruption` es el contraste
confirmatorio único del primer corte; ID, size y topology son secundarios y se
ajustan como familia cuando corresponda. Una mejora allí no se generaliza a
otros mecanismos ni a dominios físicos. El bootstrap es jerárquico por master;
las familias topológicas sólo sostienen un claim poblacional si su número lo
permite, y de otro modo se informan una por una.

### Escalera de ejecución y presupuesto

1. **Contrato y clásicos, CPU — ejecutado.** `256` masters de preflight, `n=8..16`:
   generador, target-authority table, checker, WLS e IRLS; verificar orientación,
   gauge, permutación, rank, condición y anti-leakage.
2. **Smoke neuronal, CPU — ejecutado.** `512/128/256` masters train/val/test, dos seeds,
   `10` épocas, ancho `64`, dos bloques y batch `64`; correr los cuatro brazos
   factoriales y controles mínimos bajo techo de `2 h` y `8 GiB` de RAM. Este
   corte sirve para depurar, no para claims.
3. **Desentrelazado de solver, CPU — ejecutado.** Reusar los estados raw para cruzar relación
   cruda/corregida y peso unidad/aprendido bajo WLS e IRLS, sin re-forward. El
   objetivo es localizar si la pérdida aparece en la corrección, en el peso o
   en su interacción con el solver robusto.
4. **Diagnóstico de interfaz por solver, CPU — ejecutado.** Seleccionar en
   validation una interfaz estática, temperar el peso, destruir sólo su
   asignación espacial y ensayar un selector lineal con observables públicos;
   transportar cada regla sin reajuste a test IID/grouped.
5. **Freeze confirmatorio.** Congelar generador, primary split, manifests,
   hiperparámetros y hashes; estimar tiempo real. Ejecutar tres seeds y reportar
   cada seed más ensemble. Si la proyección supera `12 h` CPU, detener el
   lanzamiento, informar objetivo, duración y VRAM estimadas y esperar la
   habilitación explícita del usuario antes de cargar CUDA.
6. **Transferencia de primitive.** Sólo si la composición aporta, probar el
   mismo bloque sobre agrupamiento armónico render-then-detect o sobre otro
   banco relacional ya autorizado. No redefinir el operador después de ver el
   destino.

### Resultado del primer escalón

El preflight clásico produjo `320` vistas desde `256` masters: `128` de train,
`32` de calibración, `32` de validación y `64` masters de test con vistas
pareadas `iid/grouped`. Los `320/320` solves Huber-IRLS convergieron; el máximo
residual de ciclo limpio fue `3.96e-15`, mientras la mediana observada fue
`0.394`, de modo que el checker distingue cierre exacto de observación
contaminada sin leer autoridad privada.

En RMSE de cociente, IRLS mejoró a WLS en todos los slices. En test IID pasó de
`0.202` a `0.109`; bajo corrupción agrupada pasó de `0.204` a `0.178`. La
degradación pareada IID→agrupada de IRLS muestra que el cambio de dependencia
es un desafío real para el baseline robusto, aunque estas cifras son todavía
diagnósticas y no comparan mixers neuronales. La referencia con pesos oracle,
que usa la máscara causal privada, quedó en `0.045` IID y `0.077` agrupado: mide
solvabilidad, no constituye un baseline deployable.

El artefacto oficial conserva `1.927` archivos manifestados, hashes de las
cuatro fuentes ejecutables, inputs públicos, sidecars privados, salidas crudas
de tres solvers, índices de bootstrap y replay byte-exacto. La suite focal dio
`10 passed`; la regresión del frente completo dio `173 passed`. Las auditorías
independientes R334–R336 cerraron convergencia, máscaras, replay, trazabilidad y
versionado de la configuración canónica. El estado pasa por ello de protocolo
auditado a **preflight clásico ejecutado**. Ese resultado habilitó, sin
decidir, el smoke neuronal CPU.

No se declara por adelantado un GO ni se inventa un efecto mínimo. El informe
estima el contraste primario `TYPED-GENERIC` dentro de cada nivel de evidencia,
su interacción con `RAW/CLOSURE` y la interacción `brazo x solver` en
`OOD-corruption`, con CI pareado por master. Se congelan antes de test la
dirección del contraste, el orden de lectura y la corrección de multiplicidad
para la familia secundaria. Si el path shuffle no está balanceado o no degrada,
si `EXACT-CLOSURE-ONLY` agota la tarea, o si el lift aparece sólo con un solver,
la atribución al mixer queda rechazada aunque el paquete completo funcione. La
promoción arquitectónica pertenece al usuario.

### Resultado del segundo escalón

El smoke neuronal oficial corrió los ocho brazos y dos seeds sobre un universo
común: de `1.280` vistas generadas excluyó `26` sin shuffle balanceado factible
y conservó `496` train, `127` validation y `504` test, estos últimos como `252`
masters `iid/grouped` pareados. Los `16` trainings de `10` épocas terminaron en
`1.320,29 s` y `1,047 GiB` de RSS máximo. La repetición independiente terminó en
`1.425,12 s`; los dos paquetes tienen los mismos `48` archivos, el manifest y
los `46/46` artefactos deterministas son byte-exactos, y sólo la observación de
runtime difiere como estaba predeclarado.

El efecto tipado aparece, pero no es monolítico. En RAW reduce el RMSE de
relación frente al mixer genérico en `-0,0058` IID y `-0,0040` grouped, y reduce
WLS en `-0,0056` y `-0,0015`; con CLOSURE la reducción de relación es
`-0,0065/-0,0023`, mientras WLS queda en `-0,0080` IID y `-0,0003` grouped, con
el intervalo grouped cruzando cero. El decoder directo se mueve en dirección
contraria: el tipado aumenta su error. La interacción factorial también cambia
por slice; entregar cierre exacto amplía levemente la ventaja tipada en IID,
pero la reduce o invierte bajo corrupción grouped.

Los controles localizan mejor la capacidad. Frente al path shuffle,
`CLOSURE-TYPED` reduce el error de relación en `-0,0754` IID y `-0,0733`
grouped. Frente a pair-state sin mezcla, `RAW-GENERIC` lo reduce en
`-0,0700/-0,0723`. `EXACT-CLOSURE-ONLY` no agota la reconstrucción: su WLS es
`0,2060` IID y `0,2259` grouped. Sin embargo, la mejora pre-solver no se
transporta de modo uniforme: IRLS sobre observación cruda alcanza
`0,1142/0,1825`, mientras RAW-TYPED queda en `0,1768/0,1971`. Hubo `8` fallos
IRLS sobre `11.989` evaluaciones; el estimando conservador dejó no evaluables
las comparaciones afectadas en lugar de descartarlas silenciosamente.

La lectura es por ello doble. El mixing de caminos y el tipado contienen señal
para corregir relaciones y mejorar WLS en parte del factorial, pero una única
salida de relación y confiabilidad no sirve igual a WLS, decoder directo e
IRLS. La brecha hasta pesos oracle —WLS `0,0360` IID y `0,0976` grouped— sigue
siendo amplia. Antes de un freeze confirmatorio corresponde un contraste CPU
que desacople relación cruda/corregida y peso unidad/aprendido usando los
estados ya preservados. Esto registra una alternativa solver-específica; no
promueve arquitectura ni constituye GO/NO-GO.

### Resultado del tercer escalón

El desentrelazado reutilizó los outputs congelados de los ocho brazos y dos
seeds, sin reentrenar ni ejecutar un nuevo forward. Para cada uno de los `252`
masters test pareados recombinó relación observada/corregida y peso
unitario/aprendido bajo WLS e IRLS. El promedio entre seeds precede a los
efectos por master; los intervalos marginales usan `2.000` réplicas bootstrap.
La corrida oficial y su replay reprodujeron byte-exactamente los `25/25`
artefactos deterministas. La regresión proporcional completa cerró con `70
passed`; la ejecución no importó `torch`, no vio CUDA y consumió `108.901 s`
con `0.859 GiB` de RSS máximo observado.

WLS muestra una estructura estable. En los ocho slices primarios, aplicar el
peso aprendido a la relación observada aporta el mayor efecto favorable; la
corrección relacional agrega una mejora menor y la interacción positiva entre
ambas devuelve parte de esas ganancias. El paquete completo, aun siendo
subaditivo, mejora WLS en todos los casos: de `-0.0065` a `-0.0153` IID y de
`-0.0088` a `-0.0106` grouped, con intervalos marginales por debajo de cero.

IRLS invierte el resultado y también cambia el componente crítico. En IID, la
relación corregida con peso unitario degrada el RMSE entre `+0.0684` y
`+0.0747`; el peso aprendido sobre la observación compensa cerca de `-0.010`
cuando el slice es evaluable. En grouped, la corrección aislada queda cerca de
cero, mientras el peso aprendido empeora entre `+0.0149` y `+0.0191`. El
paquete corregido×aprendido degrada los seis slices IRLS evaluables. Su
interacción con solver queda separada de cero en todos ellos.

Los controles path-shuffle, no-mix, message passing genérico y edge-MLP
reproducen el patrón WLS favorable e IRLS adverso donde el estimando permanece
evaluable. La dependencia del executor no puede atribuirse, por tanto, al
mixer tipado por sí solo. Catorce combinaciones agregadas registraron una no
convergencia IRLS; ningún WLS falló y la política estricta dejó no evaluable el
slice completo en lugar de usar supervivientes finitos.

La inferencia queda acotada al banco sintético y a estos checkpoints: una
relación corregida acompañada por confiabilidad aprendida no forma todavía un
contrato solver-agnóstico. El estado relacional puede seguir conteniendo señal
pre-solver, pero su adaptación downstream debe declarar la semántica del
executor. Se registra una candidata con heads o pérdidas solver-específicas;
su entrenamiento queda en la cola GPU suspendida. El próximo trabajo CPU puede
usar validation para seleccionar una interfaz estática por solver y
transportarla sin cambios a test, junto con un diagnóstico basado sólo en
observables públicos. Como test ya fue abierto, ese contraste será
exploratorio, no confirmatorio.

### Resultado del cuarto escalón

El diagnóstico de interfaz reutilizó los mismos estados congelados y mantuvo
todo ajuste dentro de validation. Su corrida oficial tardó `229.683 s`, el
replay `228.782 s`, y ambos igualaron byte por byte el manifest y los `22/22`
artefactos deterministas. La regresión proporcional completa cerró con `96
passed`; no hubo imports de `torch`, CUDA permaneció invisible y el pico de RSS
fue `0.171 GiB`.

La regla estática eligió `observed|learned` para WLS y `observed|unit` para
IRLS en los cuatro brazos primarios. En WLS, el peso aprendido redujo el RMSE
frente a unitario entre `-0.0130` y `-0.0228` IID y entre `-0.0095` y `-0.0112`
grouped. Omitir la corrección relacional mejoró además el paquete entregado,
especialmente en IID. Para IRLS, volver a peso unitario evitó degradaciones del
paquete entre `0.0625` y `0.0711` IID y entre `0.0150` y `0.0207` en los slices
grouped evaluables. Esta regla no constituye routing aprendido: la identidad
del solver ya estaba disponible y la decisión es global.

Los controles distinguen la semántica del peso. Ocho shuffles por vista
preservaron exactamente su distribución y destruyeron sólo su asignación a
aristas. En los ocho slices WLS, learned superó a shuffled con intervalos
marginales enteramente favorables, mientras shuffled empeoró al peso unitario.
La head contiene, por tanto, información relacional localizada útil para WLS;
no es sólo una distribución marginal conveniente. En IRLS sucede otra cosa:
el solver con base unitaria deja `0.0262` de su masa final sobre aristas
alteradas en IID y `0.0522` grouped, mientras la base aprendida eleva esos
valores a `0.0303–0.0323` y `0.0539–0.0583`. La evidencia contradice la
hipótesis de doble supresión y es más compatible con interferencia entre peso
exógeno y reponderación residual endógena.

Ni un temperado escalar ni el selector ridge de veintitrés features públicas
resolvieron ese transporte. WLS prefirió `alpha=1`; IRLS eligió `0` o `0.25`,
pero el cuarto de intensidad favorable en dos brazos IID no transportó de modo
estable a grouped. El predictor público perdió casi toda su correlación con el
beneficio fuera de validation y, bajo IRLS grouped, empeoró los cuatro brazos.
Esto rechaza esas dos reparaciones concretas, no todo routing posible.

La inferencia arquitectónica permanece candidata: un estado relacional común
podría alimentar adaptadores tipados por executor, con una salida WLS de
importancia localizada y una salida IRLS compatible con robustez residual, sin
peso exógeno por defecto. Antes de cualquier entrenamiento corresponde
inspeccionar en CPU si los checkpoints permiten congelar encoder y mixer y
aislar adaptadores pequeños. Reentrenamiento integral, seeds adicionales,
transferencia física y cualquier ejecución GPU permanecen en cola. No hubo
promoción ni decisión GO/NO-GO.

### Resultado del quinto escalón

El formato permitió congelar limpiamente el tronco y reoptimizar sólo `4.225`
parámetros de peso o `4.224` de corrección por checkpoint. Corrida y replay
cerraron en unos `304 s`, con `0.822 GiB` de RSS y `31/31` artefactos
deterministas byte-exactos.

La head WLS especializada mejoró IID entre `-0.0112` y `-0.0151`, pero no
transportó a grouped: los brazos genéricos quedaron inciertos y los tipados
cambiaron de signo, con `CLOSURE-TYPED` adverso. La corrección destinada a IRLS
redujo el error marginal de relación, pero empeoró IRLS IID entre `+0.0664` y
`+0.0763`; en grouped los intervalos cruzaron cero. El preflight de formato fue
positivo, pero la solución head-only propuesta fue negativa como interfaz
transportable. El siguiente diseño debe representar la semántica residual de
IRLS mediante un surrogate o gradiente validado contra el executor, no repetir
denoising marginal. Toda GPU permanece en cola; no hubo promoción ni GO/NO-GO.

### Resultado del sexto escalón

El surrogate diferenciable fue contrastado antes de usarlo para entrenar. La
corrida recorrió las `127` vistas IID de validation y `1.024` estados adicionales
derivados de `32` vistas, cuatro brazos, dos seeds y cuatro puntos de trayectoria.
Los `1.151/1.151` estados convergieron en el executor canónico. Una referencia
NumPy fixed-K independiente y el unroll Torch coincidieron en escala de redondeo;
para `K=64`, el máximo entre potenciales, pesos y objetivo Huber fue `3,46e-13`.

`K=64` fue la menor profundidad con auditoría de valores y gradientes que cumplió
el contrato: p99 de RMSE `2,15e-6`, máximo `3,75e-6`, coseno mediano de gradiente
`1,0`, p95 relativo `1,02e-9` y cero inversiones de signo. De `1.218`
coordenadas válidas en esa profundidad se compararon `1.197`; `21` se excluyeron
por proximidad al kink de Huber y quedaron preservadas con su motivo. `K=32`
cumplió por valor pero no fue elegido porque no estaba en la grilla de auditoría
de gradiente.

Corrida y replay reprodujeron el manifiesto y `7/7` artefactos deterministas. La
oficial tardó `109,47 s` y alcanzó `0,787 GiB`; el replay, `105,54 s` y
`0,789 GiB`. Ambos fueron CPU-only. Esto habilita `K=64` como herramienta para
un contraste de entrenamiento, no valida ese entrenamiento ni aporta evidencia
grouped, de test o física. El siguiente paso CPU compara una pérdida local contra
una pérdida de potenciales posterior al surrogate, manteniendo el executor
canónico como evaluador externo. No hubo promoción ni GO/NO-GO.

### Resultado del séptimo escalón

El contraste mantuvo tronco, `4.224` parámetros, datos, batches y updates
igualados, y ajustó en train la escala inicial de gradiente. La pérdida
post-IRLS superó a la relacional local en los ocho slices: entre `-0,0428` y
`-0,0551` IID, y entre `-0,0084` y `-0,0118` grouped, con intervalos pareados
que excluyen cero y signo consistente entre seeds. A la vez, sacrificó RMSE
relacional: la mejora downstream no equivale a denoising marginal.

El baseline más fuerte conserva el límite. Frente a `observed|unit`, post-IRLS
empeoró los cuatro brazos IID entre `+0,0222` y `+0,0259`. En grouped mejoró
tres brazos entre `-0,0051` y `-0,0084`; `closure_typed` quedó incierto. Los
`20.192/20.192` solves convergieron. Corrida y replay CPU reprodujeron `32/32`
artefactos en unos `500 s` y `0,811 GiB`. El target local estaba desalineado,
pero una corrección siempre activa tampoco constituye una interfaz universal.
El próximo diseño CPU es un gate residual con identidad exacta y controles
constantes/públicos. No hubo promoción ni GO/NO-GO.

### Resultado del octavo escalón

El gate preservó `observed` como identidad exacta y eligió entre cinco
intensidades mediante quince features públicas. En grouped mejoró identity en
los cuatro brazos entre `-0,00168` y `-0,00242`, y también superó dieciséis
shuffles matched. En IID, tres intervalos cruzaron cero y `raw_typed` fue
adverso. El alpha constante y un gate reducido a magnitud de corrección
eligieron identidad en todas las vistas.

La tasa de abstención fue menor en grouped aun sin recibir mecanismo, de modo
que existe una correspondencia pública parcial. No basta para una política
conjuntamente favorable: validation sólo contiene IID. Los `25.240` solves
convergieron y el replay reprodujo `17/17` artefactos en unos `67 s` y
`0,690 GiB`. El siguiente diseño debe crear calibración y test frescos con ambos
mecanismos representados antes de la selección. No hubo promoción ni GO/NO-GO.

### Resultado del noveno escalón

La calibración y la adjudicación se separaron en dos realizaciones nuevas de
`251` masters pareados IID/grouped cada una. El freeze se escribió antes de
materializar adjudicación y quedó ligado al commit, config, fuentes y
`22` artefactos de calibración. El gate siguió decidiendo sólo con observables
públicos; mecanismo y autoridad privada se reservaron para construir el target
de calibración y reportar slices.

En la adjudicación fresca, el gate completo mejoró identity en grouped en los
cuatro brazos entre `-0,00156` y `-0,00239`, con intervalos que excluyen cero.
En IID, las cuatro medias fueron levemente adversas, pero todos los intervalos
cruzaron cero. El promedio balanceado favoreció los cuatro brazos y quedó
resuelto en raw-generic (`-0,00098`) y raw-typed (`-0,00115`); los dos closure
quedaron inciertos.

Tres brazos grouped superaron al gate IID histórico y tres al promedio de
dieciséis shuffles. El límite principal está en el control reducido: ninguna
comparación del gate de quince features frente al gate de dos medidas de escala
excluyó cero. La calibración mixta reproduce señal pública de routing, pero no
acredita superioridad por mecanismo ni valor incremental de la representación
completa. Los `40.160/40.160` solves convergieron y el replay igualó `51/51`
artefactos. El próximo contraste CPU debe usar realizaciones nuevas y calibrar
una abstención bajo restricción de no-daño IID; no se retoca el test abierto.
No hubo promoción ni GO/NO-GO.

### Resultado del décimo escalón

La regla de abstención reutilizó sin refit los gates full, reduced y shuffled.
Policy selection abrió una realización nueva de `247` masters y exigió que cada
threshold no trivial tuviera límite superior IID simultáneo menor o igual a
cero. Ninguno cumplió: identidad quedó seleccionada en `8/8` gates full/reduced
y `64/64` shuffles. Las mejores medias IID full eran negativas, pero sus límites
superiores todavía quedaron entre `+0,000135` y `+0,000747`; la decisión fue por
incertidumbre, no necesariamente por daño puntual.

En una adjudicación posterior de `249` masters, las políticas safe fueron
identidad y no produjeron beneficio. El comparador unconstrained volvió a
mejorar grouped y el promedio balanceado en los cuatro brazos, con intervalos
favorables, mientras IID quedó incierto. La ventaja predicha correlacionó
positivamente con beneficio grouped (`0,220–0,389`) y negativamente con
beneficio IID (`-0,230` a `-0,353`); un threshold monotónico no ordena ambos
mecanismos de la misma manera.

Los `39.680/39.680` solves convergieron y el replay reprodujo `50/50`
artefactos. La abstención funcionó como firewall conservador, no como solución
de routing. No hubo promoción ni GO/NO-GO.

### Resultado del undécimo escalón

La auditoría de potencia reutilizó tres universos ya abiertos: `247` masters de
policy selection, `251` de la adjudicación R357 y `249` de R358. Bajo el
supuesto de efecto fijo, los candidatos full más fáciles de certificar
proyectaron `3.504`, `368`, `613` y `374` masters para raw-generic, raw-typed,
closure-generic y closure-typed. Reduced no ofreció ningún candidato finito:
sus medias IID de selección eran no negativas o los thresholds extremos no
intervenían.

La condición que sostiene esas proyecciones no transportó. Ninguno de los
cuatro candidatos full mantuvo un efecto IID puntual no positivo en los tres
universos, y los cuatro fueron positivos en la adjudicación más reciente,
aunque con intervalos que cruzan cero. Grouped y el promedio balanceado sí
conservaron signo favorable en ambas adjudicaciones para los cuatro brazos. La
incertidumbre simultánea cuesta potencia, pero aumentar sólo la muestra bajo un
efecto supuesto constante no resuelve la inversión observada.

La corrida y el replay CPU tardaron alrededor de `2,4 s`, usaron `0,691 GiB` y
reprodujeron `4/4` artefactos. El próximo diseño debe representar localización
topológica de la corrección y enfrentarla con un control de escala matched que
destruya esa asignación. Requiere nuevas realizaciones prospectivas; los tests
abiertos no seleccionan features ni thresholds. GPU permanece en cola. No hubo
promoción ni GO/NO-GO.

### Resultado del duodécimo escalón

El gate topológico agregó nueve features invariantes a las quince públicas y
las comparó con dieciséis controles de igual dimensión que permutan la
corrección entre aristas, además de dieciséis target-shuffles. Tres realizaciones
disjuntas aportaron `250/253/252` masters para calibración, selección y
adjudicación. La banda IID eligió identidad en los `12/12` gates principales y
los `128/128` controles; safe volvió a ser un firewall sin intervención.

El comparador topology unconstrained cambió la separación: actuó en
`23,4–31,7%` de IID y `62,7–68,7%` de grouped. Frente a identidad mejoró grouped
`4/4` entre `-0,00663` y `-0,00963`, y balanceado `4/4` entre `-0,00285` y
`-0,00407`; IID quedó incierto en los cuatro brazos. Superó a public-base y al
control topology-permuted en grouped y balanceado `4/4`, siempre con intervalos
favorables. Frente a target-shuffled, la superioridad se resolvió en tres
brazos. La asignación a nodos y vecindades recibe así valor causal parcial
dentro del generador, pero todavía no una política safe.

Los `60.400/60.400` solves convergieron y el replay reprodujo `76/76`
artefactos. El próximo contraste debe conservar la representación y cambiar
sólo la interfaz de riesgo: calibrar un límite superior de daño IID por alpha,
con public-base y topology-permuted como controles. Requiere otra adjudicación
fresca; no se ajusta sobre este test. GPU permanece en cola. No hubo promoción
ni GO/NO-GO.

### Resultado del decimotercer escalón

La interfaz condicional congeló el predictor topology y separó `251/249/250/254`
masters entre ajuste de escala IID, calibración, firewall y adjudicación. Las
familias constant, public-base, topology y dieciséis topology-permuted cambiaron
sólo `sigma`; `mu`, los cuatro alphas y el executor permanecieron iguales. El
ajuste topology del residuo absoluto mejoró a public-base sólo en raw-typed y al
promedio permutado sólo en los dos brazos raw.

Los doce gates principales pasaron formalmente el firewall, con límites muchas
veces iguales a cero. En adjudicación topology intervino, sin embargo, sólo en
`13/2.032` decisiones brazo-vista: ambos typed fueron identidad exacta. Los dos
brazos que actuaron conservaron puntos grouped favorables, pero ningún intervalo
frente a identidad excluyó cero y no apareció superioridad consistente frente a
constant, public-base o topology-permuted. La escala absoluta no transportó la
ganancia grouped de R360; la eliminó junto con casi toda acción.

Los `80.320/80.320` solves convergieron y el replay igualó `98/98` archivos
deterministas. La próxima hipótesis CPU es unilateral: diagnosticar si modelar
el cuantil superior del residuo firmado evita que una desviación favorable
engrose el margen de daño. El análisis de crudos abiertos es post hoc y sólo
puede diseñar otro contraste fresco. GPU permanece en cola. No hubo promoción
ni GO/NO-GO.

### Resultado del decimocuarto escalón

El diagnóstico unilateral no generó otra vista. Sobre los artefactos abiertos
de R361 ajustó el cuantil `0,90` del residuo firmado y reaplicó calibración,
firewall y adjudicación. Topology mejoró pinball OOF frente a constant y al
promedio topology-permuted en `4/4`, y frente a public-base en `3/4`.

Después del firewall actuó en `24/2.032` decisiones, frente a `13/2.032` de la
escala absoluta. Raw-generic resolvió grouped y balanceado frente a identidad;
closure-generic resolvió balanceado. Sólo raw-generic balanceado mejoró también
a absolute-scale y ningún intervalo frente al sham excluyó cero. Raw-typed fue
bloqueado por el firewall; sin él habría sido adverso grouped en este test.

Corrida y replay CPU reprodujeron `11/11` archivos. La señal justifica un
contraste prospectivo de cuatro roles, no una política acreditada. Fit,
calibración, selección y adjudicación deben usar realizaciones nuevas con el
mismo cuantil, grilla, controles y tie-break congelados. GPU permanece en cola.
No hubo promoción ni GO/NO-GO.

### Resultado del decimoquinto escalón

El contraste prospectivo reajustó signed-tail y absolute-scale sobre el mismo
risk fit y separó calibración, firewall y adjudicación en cuatro realizaciones
nuevas. Topology signed obtuvo menor pinball que constant, public-base y el
promedio permutado en `4/4`; la señal de representación transportó al ajuste.

La política no transportó. El firewall dejó signed topology en tres brazos y
absolute topology en dos, pero sobre adjudicación actuaron apenas `5/2.040` y
`2/2.040` veces, siempre grouped. Ningún intervalo signed frente a identidad o
absolute excluyó cero. Topology no superó al control matched y fue adverso
frente al promedio permutado en closure-typed grouped y balanceado.

Los `80.560/80.560` solves convergieron; oficial y replay igualaron `102/102`
archivos. Repetir la misma corrección simultánea tendría poco poder: paga el
máximo de cuatro residuos aunque sólo ejecutará un alpha. El diagnóstico CPU
siguiente congela primero el proposer y calibra después el residuo escalar de
la acción elegida. Esa garantía sería más estrecha pero estaría alineada con la
decisión. No hubo promoción ni GO/NO-GO; GPU permanece en cola.

### Resultado del decimosexto escalón

El diagnóstico posterior congeló primero un alpha por vista y calibró sólo el
residuo de esa acción. Como cada score seleccionado es un componente del máximo
simultáneo, `q_selected <= q_simultaneous` se verificó en todas las familias y
réplicas. Topology pasó de `5` a `14/2.040` acciones.

Closure-typed recuperó beneficio grouped y balanceado frente a identidad y al
gate simultáneo. La atribución no cerró: ningún intervalo topology frente a
constant, public-base o promedio topology-permuted excluyó cero. La cobertura
conjunta era una fuente de abstención, pero no el único cuello de política.

Oficial/replay igualaron `9/9` archivos sin nuevas vistas o solves. La candidata
selected-action queda preservada. Un nuevo protocolo fresco sólo agrega poder
si declara topology menos controles matched como estimando principal. No hubo
promoción ni GO/NO-GO; GPU permanece en cola.

### Resultado del decimoséptimo escalón

La auditoría de transporte aplicó la misma interfaz selected-action a dos
cohortes independientes ya abiertas: R361/R362 y R363/R364. La segunda
reconstrucción reprodujo exactamente R364. No hubo vistas, solves ni refit.

El firewall topology cambió de `1/4` brazos en la primera cohorte a `4/4` en la
segunda; la acción desplegada fue `12/2.032` y `14/2.040`. En el estimando
principal topology menos el promedio topology-permuted desplegado, `7/12`
celdas cambiaron de signo, `2/12` fueron adversas en ambas, `1/12` fue identidad
o cero numérico y sólo closure-generic grouped/balanceado fue favorable en
ambas. Ninguno de esos dos intervalos individuales excluyó cero.

La diferencia frente a identidad proyectaba un tamaño realizable de `255`
masters, pero el control matched elevó la proyección transport-aware a `4.993`
y `8.900`. Repetir unas `250` unidades no tiene poder diagnóstico plausible
bajo el efecto fijo observado. El siguiente contraste CPU debe compartir
espacio candidato y alpha, e igualar el presupuesto de acción entre rankings
topology/control; fijar también las vistas anularía el estimando. Oficial y replay igualaron `9/9`
archivos; no hubo promoción ni GO/NO-GO y la GPU permanece en cola.

### Resultado del decimoctavo escalón

El diagnóstico siguiente corrigió una ambigüedad del soporte común: fijar las
mismas vistas habría anulado el estimando. R366 compartió la propuesta de alpha
public-base, dejó que cada scorer ordenara el mismo universo y le asignó igual
presupuesto total en `1%, 2%, 5%, 10%, 20%, 40%`.

Topology menos el promedio permutado fue adverso en ambas cohortes en `37/72`
celdas brazo×presupuesto×slice, inestable en `22/72` y favorable en `13/72`.
Raw-generic balanceado conservó signo favorable en `1–5%`, pero ningún intervalo
excluyó cero y la curva se invirtió después. Frente a identidad, en cambio,
`64/72` celdas fueron favorables en ambas: hay oportunidades ordenables, pero
el aporte incremental de la cola topology no recibe crédito.

La propuesta compartida eligió `alpha=0,25` en `93,9–99,6%` de las vistas. El
próximo diagnóstico CPU compara ranking mean-only contra las correcciones de
cola. Si `mu` domina, la cola puede servir para incertidumbre sin ordenar la
acción. Oficial/replay igualaron `9/9` archivos; no hubo promoción ni GO/NO-GO
y la GPU permanece en cola.

### Resultado del decimonoveno escalón

R367 retiró la corrección de cola del ranking sin cambiar alpha, presupuestos o
cohortes. Mean-only superó a `mu+u_topology` en ambas realizaciones en `50/72`
celdas; la cola ganó `2/72`, cambió de signo en `19/72` y produjo una igualdad.
El promedio entre presupuestos favoreció mean-only en `10/12` brazo×slice. La
excepción raw-generic IID fue pequeña y sus dos intervalos cruzaron cero.

Mean-only mejoró identidad en ambas cohortes en `58/72` y al promedio
permuted-tail en `55/72`. La lectura arquitectónica es una separación de
responsabilidades: `mu` ordena y la cola estima incertidumbre sin reordenar.
Oficial/replay igualaron `9/9`; no hubo promoción ni GO/NO-GO y GPU sigue en cola.

### Resultado del vigésimo escalón

R368 implementó la separación propuesta: `mu` fijó el ranking y las colas sólo
pudieron excluir vistas. El filtro topology habilitó `31/2.032` decisiones en A
y `7/2.040` en B, con saturación del soporte desde `2%` y `1%` respectivamente.
Frente a mean-only fue adverso en ambas cohortes en `57/72` celdas y perdió los
`12/12` promedios brazo×slice.

El control incremental tampoco acreditó topology. Contra el promedio de
dieciséis filtros permutados, topology fue adverso en `24/72`, favorable en
`10/72`, inestable en `32/72` y cero en `6/72`; ningún intervalo calibrado
resolvió una dirección en ambas cohortes. El firewall dejó `8/24` políticas en
A y `18/24` en B, pero no corrigió el patrón. La cobertura sigue siendo
marginal selected-action: hubo acciones dañinas dentro del conjunto elegible,
por lo que la interfaz no concede seguridad condicional.

La oportunidad arquitectónica se simplifica. `mu` puede conservar el ranking y
el presupuesto/firewall agregado la decisión; la cola aprendida no merece el
camino primario mientras no muestre valor incremental. Antes de otro filtro o
freeze, el siguiente diagnóstico CPU atribuye la señal de `mu` contra medias
public-base, reduced y controles de localización permutada. Oficial/replay
igualaron `17/17`, con `3.925` arrays finitos, sin promoción, GO/NO-GO ni GPU.

### Resultado del vigesimoprimer escalón

R369 reaplicó por CPU los modelos de media R360 con alpha y presupuesto comunes.
En el stage de ranking puro, topology-mean superó reduced en `43/72` celdas,
target-shuffled en `45/72` y topology-permuted en `37/72`; frente a identidad
fue favorable en `58/72`. La señal incremental se concentró en grouped
(`16/24`) y balanceado (`14/24`), mientras IID quedó dividido.

La atribución más cercana no cerró: ningún intervalo topology−permuted se
resolvió favorablemente en ambas cohortes. El proposer común sostuvo la
dirección (`39/72` favorable), pero el firewall por política la revirtió. Con
alpha fijo, deployed quedó `11/72` favorable, `20/72` adverso, `29/72`
inestable y `12/72` cero. Esto separa dos hallazgos: la media topology contiene
señal más allá de reduced y target espurio, pero ni la localización incremental
ni su deployment están validados.

La primitive recuperable pasa a ser el ranker de media, no la cola. Antes de
otra cabeza o realización fresca, el siguiente diagnóstico CPU audita potencia
y la no linealidad de policy-selection sobre el estimando topology−permuted.
Oficial/replay igualaron `16/16`, con `5.015` arrays finitos, sin promoción,
GO/NO-GO ni GPU.

### Resultado del vigesimosegundo escalón

R370 auditó potencia y selección sin modificar R369. Las `37` celdas primarias
favorables proyectan entre `522` y `1.541.364` masters bajo efecto y varianza
fijos, con mediana `5.552`; ninguna queda en `<=500`. Al remuestrear además las
dieciséis permutaciones, ninguna celda resuelve el intervalo en ambas cohortes.

La descomposición localizó la reversión. De las 37 señales ranked, 26 dejan de
ser favorables después del deployment. Aplicar sólo el firewall topology vuelve
`31/72` celdas adversas y deja `11/72` favorables; aplicar sólo los firewalls de
control conserva `44/72` favorables. El ranker y el selector no deben recibir
un único estatuto.

No se justifica otro freeze histórico ni una cola nueva. El próximo diagnóstico
CPU debe representar, sin elegir pesos, el frente Pareto entre daño IID y
beneficio grouped de las políticas ya preservadas. Si ese frente no transporta,
la señal de ranking permanece descriptiva; si transporta, la elección de
utilidad vuelve explícitamente al usuario. Oficial/replay igualaron `8/8`, con
`2.356` arrays finitos, sin promoción, GO/NO-GO ni GPU.

### Resultado del vigesimotercer escalón

R371 representó esa tensión como frente Pareto sin fijar utilidad. Los frentes
topology seleccionados contienen de una a tres políticas: `40%` aparece en
`8/8`, `20%` en `6/8` y `10%` en `3/8`; identity y `1/2/5%` quedan fuera. La
retención selection→adjudication promedia `0,771`, pero cae hasta `0,333`, y el
Jaccard A/B de los IDs seleccionados sólo promedia `0,542`.

La envolvente topology domina por cobertura a reduced en `8/8` celdas. Frente
a public-base domina en `4/8` y queda incomparable en `4/8`. Contra las 128
arenas permutadas, topology domina 35, pierde 7 y queda incomparable en 86. El
frente es una descripción más fiel que el firewall binario, pero todavía no es
una política estable ni una atribución topológica limpia.

No corresponde elegir `40%` por repetición ni abrir otro freeze corto. El
siguiente diagnóstico CPU abre la matriz pareada de dominancia entre
presupuestos y su cambio entre selección y adjudicación; debe distinguir una
ruta de acción estable de empates bootstrap sin introducir utilidad.
Oficial/replay igualaron `8/8`, con `1.220` arrays y `4.181.792` valores
finitos, sin promoción, GO/NO-GO ni GPU.

### Resultado del vigesimocuarto escalón

R372 abrió los 21 pares de la ruta de presupuestos. Las `3.040`
comprobaciones sobre las acciones R369 confirmaron anidamiento. En selection,
la expansión topology domina `117/168` pares y compra beneficio grouped con
costo IID en `48/168`; en adjudication los conteos pasan a `135/168` y
`32/168`.

El tramo `20→40%` explica el extremo recurrente de R371: nunca queda dominado
por `20%`, sino que domina en dos celdas e intercambia IID por grouped en seis.
Pero sólo `4/8` celdas conservan el mismo estado entre roles. El acuerdo de los
168 estados selection→adjudication es `60,1%`; entre A/B promedia `67,9%` en
selection y `70,2%` en adjudication. La membresía al frente oculta cambios en
la comparación que la produce.

El análisis histórico ya agotó su ganancia razonable. La candidata
arquitectónica pasa a ser una interfaz tipada `BudgetPath` que conserve
acciones anidadas, las dos coordenadas y su incertidumbre, dejando utilidad y
elección aguas abajo. Diseñarla y chequearla por CPU no equivale a promoverla;
confirmarla exigiría el freeze prospectivo dimensionado por R370. Oficial y
replay igualaron `8/8`, con `12.768` arrays y `6.460.608` valores finitos, sin
promoción, GO/NO-GO ni GPU.

### Resultado del vigesimoquinto escalón

R373 convirtió la ruta en una interfaz tipada sin volver a cortar el
histórico. Materializó `608` `BudgetPath`: dos cohortes, dos propuestas, cuatro
brazos, dos roles, tres familias principales y dieciséis controles. Cada uno
conserva siete políticas, 21 pares, hashes hacia R369–R372, reader Pareto y
estados de autoridad separados.

El checker independiente reconstruyó `608/608` objetos desde los NPZ y rechazó
`8/8` mutaciones de schema, utilidad, ejes, acción, objetivo, frente, estado
pareado y lineage. La salida canónica no contiene política seleccionada, peso
de utilidad, score escalar ni recomendación. Declara `STRUCTURE_ONLY`,
`OPENED_POSTHOC`, autoridad física no reclamada y decisión no resuelta.

La interfaz queda disponible como candidata separada, no promovida. El próximo
trabajo CPU puede especificar el puerto de utilidad externa con fixtures
sintéticos, sin aplicarlo a R369–R372 hasta que exista una utilidad declarada
por el usuario. Una confirmación empírica sigue requiriendo el freeze
dimensionado por R370. Oficial/replay igualaron `7/7`; la regresión cerró
`214/214`, sin fit, vistas, solves ni GPU.

### Resultado del vigesimosexto escalón

R374 materializó el puerto posterior sin tocar ninguna de las `608` rutas
históricas. Dos fixtures sintéticos fijaron de antemano siete decisiones para
weighted sum, prioridad lexicográfica y restricción epsilon. Las `7/7`
coincidieron; el empate exacto devolvió los tres óptimos y el caso sin política
factible se abstuvo, sin relajar la cota.

También pasaron `3/3` propiedades metamórficas y se rechazaron `12/12` casos
inválidos de binding, scope, parámetros, candidatos y campos de selección. El
puerto revalida lineage sintético, artefacto, receipt y reader, y comprueba que
las entradas no cambien. Oficial/replay igualaron los diez productos
deterministas y el manifest; la regresión cerró `225/225` y
`gpu_queried: false`.

El resultado cierra una pregunta mecánica, no una elección. El summary declara
`historical_budget_paths_evaluated: 0`, `user_utility_status: NOT_DECLARED` y
`decision_authority: SYNTHETIC_TEST_ONLY`. La activación empírica espera una
preferencia auténtica y un freeze prospectivo dimensionado por R370. No hay
promoción ni GO/NO-GO, y toda prueba GPU permanece en cola.

### Artefactos obligatorios

Cada ejecución conserva checkpoints `last_epoch`, config resuelta, seeds,
manifiesto de lineages y splits, hash del schema público, commit y entorno. Por
master y arista guarda inputs públicos, target privado, `y_tilde`, pesos,
scores pre-solver, potenciales post-solver, diagnósticos de rank/condición,
fallos y pertenencia a slice. Se preservan índices de bootstrap y un comando de
replay. Los crudos permiten cambiar métricas, solver o bootstrap sin re-forward.

## Arquitectura 2: posterior de conjuntos y política contextual

Esta línea ya tiene componentes ejecutados:

```text
encoder DeepSets
 -> cuatro logits de compatibilidad
 -> posterior sobre 15 conjuntos no vacíos
    (unary + cardinalidad + interacciones)
 -> riesgo esperado bajo utilidad
 -> acción dura o bayesiana
 -> compuerta contextual residual
```

Su singularidad potencial no es geométrica todavía. Consiste en conservar una
región compatible y postergar la decisión hasta recibir utilidad y contexto. La
Ola 54 mostró que el posterior conjunto modela mejor dependencias y
cardinalidad. La Ola 55 mostró que un umbral escalar selecciona identidad en la
población primaria. La Ola 56 completó el contraste contextual fresco y su
replay exacto: redujo regret medio y elevó compatibilidad frente a hard, pero
perdió accuracy fuera del margen y empeoró worst regret. Frente al shuffled la
dirección fue favorable, aunque la magnitud `0.008783` quedó bajo el mínimo
`0.01`. El patrón prospectivo cerró `4/6`, no satisfecho.

La Ola 57 ejecutó esa separación. El proposer Ridge seleccionó `q=0.8` y el
guard Logistic `q=0.4`. Frente al proposer con la misma máscara, el guard mejoró
accuracy `+0.006944` y worst regret `-0.028322`, con intervalos completamente
favorables, sin empeorar materialmente regret. Frente al hard-set, en cambio,
la mejora de regret fue sólo `-0.003835`, la compatibilidad cedió `-0.002451` y
los intervalos de regret y cola cruzaron cero. Sólo dos de cinco shams
alcanzaron el Hamming ponderado mínimo, por lo que la condición causal quedó
`NOT_EVALUABLE` y el patrón terminal nulo. El replay fue exacto `23/23 + 13/13`.

La separación proposer/guard queda así como alternativa recuperable, no como
política promovida. El siguiente diseño debe construir shuffles válidos por
definición y, además, atacar la magnitud de regret y la cesión de compatibilidad
frente al hard. Reparar sólo el control no podría volver positiva esta
realización, porque sus tres primeras condiciones ya son falsas.

La Ola 58 convirtió ese próximo paso en un diagnóstico abierto de clase de
modelo. Sobre los tres splits ya inspeccionados comparó 36 IDs canónicos y 24
probes históricos. El orden congelado nominó un proposer HGB con guard HGB de
incompatibilidad posterior: en el monitor abierto mejoró compatibilidad
`+0.009123` y regret `-0.014490` frente al hard, con intervalos token-wise que
excluyen cero. Validation sólo separó compatibilidad. El replay científico fue
exacto `10/10`, el replay `LEGACY-W57` fue exacto `34/34` y R414 recomputó 120
combinaciones candidato×split sin divergencias.

La nominación no reduce el problema a un único ID. Los 21 elegibles forman el
frente Pareto completo, los 36 IDs representan 19 políticas distintas y el
JOINT nominado coincide exactamente con SEQUENTIAL. A la vez, el guard HGB de
harm conserva una mejora de worst regret en monitor de `-0.025327`, IC95
`[-0.049292,-0.002996]`, que el nominado de incompatibilidad no reproduce. El
siguiente prospectivo debe congelar un bracket reducido entre esas dos
políticas HGB/HGB, hard y controles matched, con shuffles válidos por
construcción y prioridad media/cola declarada antes del monitor.

La Ola 59 ejecutó ese bracket sobre una realización fresca y volvió concluyente
la parte que la Ola 58 sólo podía proponer. Incompatibilidad mejoró frente a
hard compatibilidad `+0,006876`, regret `-0,011941` y worst regret `-0,023102`,
pero el IC95 de regret contra cinco controles de desplazamiento condicional
máximo terminó en `+0,000843`. Pasó `7/8` condiciones y el patrón fue falso.
Harm mejoró worst regret frente a hard `-0,029978`, pero su IC95 de
compatibilidad bajó a `-0,001513` y el contraste de cola contra controles
terminó en `+0,003465`. Pasó `6/8` y también fue falso.

El replay fue exacto y una auditoría independiente recompuso `192/192` arrays,
bootstrap, resúmenes y factoriales. Por eso la salida no se atribuye a un fallo
operacional: hay mejoras locales frente a hard, pero no evidencia suficiente
para distinguir la ley aprendida de un control matched de igual desplazamiento;
el brazo de harm, además, no conserva compatibilidad. Esta observación debilita
el bracket actual sin clausurar toda arquitectura proposer/guard.

La Ola 60 congeló esas dos pipelines completas y las aplicó a otro draw sin
refit, recalibración ni reselección. Las cuatro métricas de cada política
mejoraron frente a `hard`, con intervalos favorables. La separación matched
volvió a fallar: incompatibility menos controles dio regret `-0.000623`, IC95
`[-0.003128,+0.001548]`; harm dio worst regret `-0.001661`, IC95
`[-0.014064,+0.009468]`. Ambos patrones permanecieron falsos.

R531 preservó la comparación histórica `MISMATCH 35/36` y verificó que su único
falso era un enlace operacional local entre roles `recovery/replay`; la vista
normalizada quedó `36/36`, sin alterar ciencia. El transporte refuerza una
eficacia local frente a `hard`, pero no atribuye esa eficacia al target HGB ni
al guard. El bracket deja de ser la continuación prioritaria.

## Arquitectura 3: lector de espectro relativo

Para pares SPD, un executor exacto puede construir el espectro relativo
orientado `ell(B|A) = {log lambda_i(B|A)}`. Un reader permutation-invariant
podría responder queries sobre esa relación sin reducirla de antemano a
determinante, extremos o espectros ordinarios separados.

La arquitectura es concreta, pero el experimento todavía no: falta una query
externa cuya respuesta no sea una función clásica ya entregada por el executor.
Esa carencia se registra como deuda de diseño y no justifica volver ahora a una
campaña bibliográfica.

## Arquitectura 4: router tipado con executors

El horizonte de integración es un sistema modular:

```text
contrato de objeto
 -> constructor de IR relacional
 -> router aprendido con abstención
 -> primitive estrecha
 -> executor tipado común
 -> checker y ledger externos
 -> reader de punto/clase/conjunto/certificado
```

El router sólo podría elegir entre relaciones y executors autorizados por el
contrato. No tendría permiso para inventar una geometría y validarla con su
propia salida. Esta arquitectura no se implementa hasta que una primitive
estrecha demuestre valor: construirla antes impediría atribuir cualquier
resultado.

## Orden de trabajo propuesto

1. cerrar las pruebas prospectivas de Olas 56–57, abrir sólo como diagnóstico
   el roster de Ola 58 y contrastar sus dos políticas sobre el draw fresco de
   Ola 59 — completado;
   Ola 56 cerró `4/6`; Ola 57 conserva una mejora incremental del guard frente
   al proposer, pero falla tres condiciones contra hard y deja el control sham
   `NOT_EVALUABLE`; Ola 58 nomina una política HGB/HGB de incompatibilidad para
   regret medio y preserva otra HGB/HGB de harm para cola; Ola 59 cierra ambos
   patrones falsos (`7/8` y `6/8`) porque no separan controles matched y harm
   tampoco preserva compatibilidad, con replay exacto y sin declarar techo;
2. congelar y auditar el protocolo factorial de coherencia local;
3. implementar contrato, clásicos y smoke neuronal en CPU — completado;
4. ejecutar el desentrelazado CPU de relación, peso y solver desde los crudos
   — completado;
5. ejecutar el selector estático por solver y el diagnóstico de observables
   públicos como contraste exploratorio CPU — completado;
6. inspeccionar el checkpoint y ejecutar adaptadores con encoder/mixer
   congelados — completado, con transporte negativo;
7. validar por CPU un surrogate IRLS contra el executor y diferencias finitas
   — completado; `K=64` es la menor profundidad conforme auditada;
8. comparar por CPU pérdida local contra pérdida post-surrogate con tronco
   congelado — completado; mejora relativa en ocho slices, pero no supera IID;
9. ejecutar un gate residual CPU que preserve identidad y use sólo observables
   públicos — completado; positivo grouped, no favorable IID;
10. ejecutar una realización fresca con calibración IID/grouped y test fresco
    — completado; grouped mejora en cuatro brazos y el promedio balanceado en
    dos, sin ventaja resuelta del gate completo sobre el reducido;
11. diseñar otra realización CPU para una regla de abstención calibrada bajo
    restricción de no-daño IID — completado; la regla seleccionó identidad en
    todos los gates y shuffles;
12. auditar por CPU la potencia de la selección desde los estados preservados
    — completado; las proyecciones finitas no transportan un signo IID estable;
13. diseñar un gate CPU sensible a localización topológica con control de
    escala matched y nuevas realizaciones prospectivas — completado; topology
    supera base y sham en grouped/balanceado, pero safe elige identidad;
14. diseñar una interfaz selectiva de riesgo CPU que congele la representación
    y calibre un límite superior IID por alpha con controles matched —
    completado; el residuo absoluto produce acción casi nula y ninguna ventaja
    resuelta;
15. auditar por CPU una alternativa unilateral sobre el residuo firmado antes
    de abrir otra realización prospectiva — completado; mejora OOF y acción en
    test abierto, sin superioridad frente al sham;
16. diseñar y ejecutar cuatro roles frescos para signed-tail contra
    absolute-scale y controles matched — completado; el ajuste OOF transporta,
    la acción cae a `5/2.040` y no supera controles;
17. auditar por CPU una interfaz que proponga una acción antes de calibrar su
    residuo escalar, sobre artefactos R363 ya abiertos — completado; recupera
    acción y un brazo frente a identidad, sin atribución topology-control;
18. auditar transporte y potencia de selected-action contra controles matched
    sobre dos cohortes abiertas — completado; `7/12` signos inestables y los
    dos favorables requieren `4.993–8.900` masters;
19. diagnosticar por CPU una propuesta de alpha compartida y rankings
    topology/control con presupuesto de acción igualado — completado; la cola
    topology es adversa o inestable en `59/72` celdas y sólo raw-generic deja
    una franja estrecha no resuelta;
20. comparar por CPU ranking mean-only contra topology, public-base y controles
    — completado; mean-only gana `50/72` y topology-tail sólo `2/72`;
21. diseñar y ejecutar una interfaz CPU two-stage: ranking congelado por `mu` y
    cola usada sólo para elegibilidad/no-daño — completado; topology actúa en
    `31/2.032` y `7/2.040`, pierde frente a mean-only en `57/72` y no supera al
    control permutado;
22. atribuir por CPU la señal de ranking de `mu` contra medias public-base,
    reduced y controles de localización permutada — completado; topology supera
    permuted `37/72` en ranking puro, sin intervalos primarios compartidos y con
    reversión después del firewall;
23. auditar por CPU potencia, transporte y no linealidad de policy-selection
    para topology-mean menos topology-permuted — completado; ninguna señal
    proyecta `<=500`, el bootstrap conjunto cierra `0/72` y el firewall topology
    elimina `26/37` puntos favorables;
24. mapear por CPU el frente Pareto IID-daño/grouped-beneficio sin elegir una
    utilidad antes de otra política o realización — completado; retención media
    `0,771`, Jaccard A/B `0,542` e incomparabilidad frente a permuted en
    `86/128` arenas;
25. auditar por CPU la matriz pareada de dominancia entre presupuestos y su
    transporte selection→adjudication, sin cutoff ni utilidad — completado;
    `20→40%` nunca queda dominado, pero sólo `4/8` estados transportan;
26. diseñar y chequear por CPU una interfaz tipada `BudgetPath` que separe ruta,
    incertidumbre y utilidad, sin promoverla ni rebanar otra vez el histórico —
    completado; `608/608` paths válidos y `8/8` mutaciones rechazadas;
27. especificar y probar por CPU el puerto de utilidad externa sólo con
    fixtures sintéticos — completado; `7/7` positivos, `3/3` metamórficos y
    `12/12` inválidos, con cero rutas históricas evaluadas;
28. transportar sin refit ni recalibración las dos pipelines HGB/HGB de Ola 59
    a otro draw — completado en Ola 60; las ocho métricas frente a hard son
    favorables, pero ambos contrastes matched cruzan cero y los patrones quedan
    `false/false`; replay científico normalizado `36/36` por R531;
29. cerrar la base acumulativa en tres líneas y auditar un relevo experimental
    finito — completado por la síntesis terminal y R533;
30. ejecutar por CPU `MAPPING-FEASIBILITY` — completado; M1–M4 fallaron, M5 y
    R1–R6/S1–S6 pasaron, los dos runs coincidieron en `148/148` archivos core y
    la salida técnica fue `BIFURCATE_NATIVE_CONTRASTS`;
31. diseñar y auditar dos freezes coordinados pero nativos — completado; la
    confirmación K192 rechazó el relacional y el set-valued quedó como única
    rama válida, con `29/30` predicados, `61/61` mutaciones y R557 `PASS`;
32. implementar y auditar por CPU el runner set-valued
    `MARGINAL/JOINT × HARD/CONTEXTUAL` contra fixtures y artefactos ya abiertos,
    sin crear todavía el draw fresco ni abrir el monitor;
33. mantener la aplicación empírica de `BudgetPath` inactiva hasta que exista
    una utilidad auténticamente declarada, y estudiar integración o transferencia
    sólo después de que una primitive estrecha obtenga evidencia afirmativa.

## Deudas registradas, no abiertas

- query externa no agotada por el eigensolver para el lector SPD;
- mecanismo multi-hop para ciclos largos, con ablación propia;
- transferencia desde log-razones escalares hacia grupos no conmutativos;
- criterio físico externo para pasar de coherencia interna a geometría natural;
- integración eventual entre posterior set-valued, abstención y router tipado.

Estas deudas no bloquean el primer benchmark y no autorizan nuevas olas de
investigación. El programa bibliográfico expansivo queda cerrado; sólo una
carencia concreta del nuevo experimento puede justificar una consulta o descarga
quirúrgica.
