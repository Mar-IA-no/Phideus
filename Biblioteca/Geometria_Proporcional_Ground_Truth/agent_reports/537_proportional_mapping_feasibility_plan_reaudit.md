# R537 — Reauditoría independiente del plan `MAPPING-FEASIBILITY`

## Dictamen técnico: REVISE

La revisión en `9c43659` resuelve el defecto central de R536: existe ahora una
álgebra total que separa fallas técnicas de hojas semánticas, los contratos
nativos tienen predicados cerrados, la matriz de fuentes liga los estados
necesarios y el replay ya no necesita que la decisión se hashee a sí misma.
También pude materializar por CPU las dos masas set-valued, sus dos readers y
las cuatro matrices de acciones `[384,24]` sin abrir monitor ni lockbox.

Persisten, sin embargo, **0 HIGH, 3 MEDIUM y 0 LOW**. No son observaciones de
estilo. Dos afectan pruebas obligatorias —el control target-shuffled relacional
y la invariancia contrafactual privada— y la tercera deja dos implementaciones
distintas igualmente compatibles con la descripción del reader contextual. En
ese estado, builder y checker independientes pueden divergir o el gate puede
detenerse por una incompatibilidad introducida por el propio plan.

## Identidad, independencia y alcance

- Target exacto: commit
  `9c43659f930dd66a2c834f6749bafedea1e18801`.
- Parent directo verificado:
  `65153e99f669302d80c84c551ebf417a941994c0`.
- Path auditado:
  `experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md`.
- SHA-256 del plan auditado:
  `c3d6c8ff6ca670e75269622f216188b3ab0a41b6dd5e980f117267b9a5f9b97e`.
- Git blob del plan: `b1deced5b0bf30d27ca37970198458bf8b8c802c`.
- R536 leído completo; SHA-256:
  `656b6e23ce1354eed79fc3887aee63faaf77e343838fc20a4bb8b339eccd75c6`.
- El diff `65153e9..9c43659` cambia un solo path: el plan, con `345`
  inserciones y `170` eliminaciones.
- La tarea fue recibida en una instancia configurada como `GPT-5.6-Sol`,
  esfuerzo `high`. El runtime no expone una atestación local adicional del
  modelo; no observé una discrepancia de configuración.
- No se usó ni consultó GPU. No se abrió lockbox ni
  `sealed_monitor_bundle.npz`; tampoco se abrió monitor histórico.
- No se implementó el gate ni se modificó el plan.

## Revisión finding por finding de R536

### R536-01 — RESUELTO

El plan define ahora estados técnicos separados y una tabla total que sólo
produce una hoja cuando `source_status`, `artifact_status`, `checker_status` y
`replay_status` son `PASS`
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:131-167`). Los contratos nativos
ya tienen `R1..R6` y `S1..S6`, cada uno con condición bicondicional y reason
codes cerrados (`ibid.:179-193,255-262`). Las tres hojas cuentan con fixtures
positivos, y cada predicado cuenta con una mutación aislada que exige su reason
code exacto (`ibid.:264-276`). Una falla técnica deja explícitamente
`mapping_decision:null` (`ibid.:154-165,349-354`).

### R536-02 — RESUELTO

La matriz congelada separa fuentes, rol, fase y hash; prohíbe añadir fuentes en
runtime (`ibid.:35-82`). Recalculé los 29 SHA-256: **29/29 coinciden**. La Ola 49
liga fixtures visibles y predicciones por train/val por separado de la autoridad
W50; la Ola 54 liga manifest, bundle fit/select, calibrador, selección y
primitives; el grafo liga manifest/config y los cuatro estados
`GENERIC/TYPED × seed` (`ibid.:49-79`).

La inspección del bundle confirma exactamente `384` tokens, logits `[384,4]`,
per-seed logits `[3,384,4]`, target `[384,4]` y roles `192/192`, como exige S1
(`data/geometria_proporcional/wave54_joint_set_inputs_v1/fit_select_bundle.npz::{pair_token,ensemble_logits,per_seed_logits,target,split_role}`;
`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:257`). Los cuatro NPZ grafo tienen
los keysets mixtos esperados; la allowlist de ocho arrays y la prohibición de
truth/IDs/split están definidas por contrato
(`src/geometria_proporcional/proportional_graph_contract.py:19-47,119-176,210-220`).

### R536-03 — PARCIALMENTE RESUELTO; ver R537-03

Las cuatro celdas ya están nominadas mediante funciones y fitting explícitos.
La masa marginal usa producto Bernoulli condicionado a set no vacío
(`src/geometria_proporcional/wave53_uncertainty.py:12-46`); la joint usa theta
`joint_full` y produce masa normalizada
(`src/geometria_proporcional/wave54_joint_set.py:42-84`). El hard reader depende
de la masa de cada posterior y tiene desempates totales
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:200-218`). El contextual fija
ridge, dos guards, splits, grids, desigualdades de autorización, orden de
selección y freeze (`ibid.:220-253`).

Una reconstrucción CPU sobre las fuentes ligadas produjo, para ambos
posteriors, masas `[384,15]`, hard `[384,24]`, candidato `[384,24]` y design
contextual `[384,24,17]`. Hubo soporte de fitting en ambas clases de ambos
guards:

| Posterior | filas disagreement fit | tokens fit | harm `0/1` | incompatibilidad `0/1` |
|---|---:|---:|---:|---:|
| MARGINAL | 464 | 82 | 178 / 286 | 387 / 77 |
| JOINT | 601 | 89 | 240 / 361 | 507 / 94 |

Ridge y las cuatro logísticas ajustaron estados finitos bajo scikit-learn
`1.8.0`, coincidente con el contrato
(`src/geometria_proporcional/wave57_tail_guard.py:19-31,139-176`). Esto demuestra
ejecutabilidad numérica básica, pero no elimina la doble semántica de `HARD`
descrita en R537-03.

### R536-04 — PARCIALMENTE RESUELTO; ver R537-02

La revisión sí separa preparer, builder, evaluator y checker, congela el
candidato antes de abrir desarrollo, prohíbe que el builder emita una decisión
y exige un checker que no importe builder/evaluator/helpers
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:84-129`). La extracción pública
por allowlist y la igualdad contrafactual son mejoras sustantivas
(`ibid.:95-109,288-293`).

No obstante, la prueba contrafactual no puede ejecutarse literalmente junto con
la verificación obligatoria de hashes. R537-02 conserva por eso una parte
material de este finding.

### R536-05 — RESUELTO

El plan separa clase científica determinista y telemetría, congela JSON/NPZ,
excluye runtime de los manifests y define dos fases de replay sin self-hash
(`ibid.:295-330`). `core_manifest` excluye a sí mismo, runtime y adjudicación;
la decisión final sólo aparece después de que un receipt externo pruebe igualdad
de ambos cores (`ibid.:316-329`). No encontré circularidad restante.

El presupuesto pasó de objetivo a condición terminal: watchdog global `<600 s`,
`RLIMIT_AS=2 GiB`, RSS `<2 GiB` y `mapping_decision:null` ante exceso
(`ibid.:340-354`). Como smoke de ejecutabilidad, un proceso nuevo con
`RLIMIT_AS=2 GiB` importó NumPy, SciPy y scikit-learn `1.8.0` en `0.88 s` y
alcanzó `171704 KiB` de RSS máximo. Eso no predice el costo completo, pero sí
descarta una incompatibilidad inmediata entre el límite y el runtime base.

## Findings nuevos o remanentes

### MEDIUM — R537-01: el target-shuffle relacional no define un transporte total entre grafos y rompe la unidad pareada

**Evidencia.** R6 prescribe una única permutación de masters dentro de
`(split,mechanism,n_nodes)` con `PCG64(53601)` y exige aplicar el mismo control a
las cuatro celdas (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:186-193`). Pero
los targets tienen dos geometrías: `x_true` está indexado por nodos y
`clean_log_ratio` por aristas. Los solvers exigen que `values` y weights estén
alineados exactamente con las aristas receptoras
(`src/geometria_proporcional/proportional_graph_contract.py:494-523,555-586`).

En el estado ligado hay `631` vistas y `379` masters. Los `27/27` estratos
`(split,mechanism,n_nodes)` contienen más de una cantidad de aristas. Bajo la
lectura directa `rng.permutation(indices)` con los estratos ordenados,
`536/631` asignaciones receptor→donante tienen longitudes de
`clean_log_ratio` distintas
(`data/geometria_proporcional/proportional_graph_neural_smoke_v1/raw_eval/raw_generic|seed=104729.npz::{edge_offsets,node_offsets,n_nodes,split,mechanism,master_id}`).
Además, estratificar la permutación por `mechanism` permite que las vistas IID y
grouped del mismo master reciban donantes distintos, aunque el plan declara
`master_id` como unidad primaria y las vistas como dependientes
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:181-184`).

**Impacto.** La permutación no determina un target relacional edge-aligned y no
preserva el pairing del estimando. Implementar un shuffle literal produce shape
failure; agregar `n_edges` o topología como estrato, o proyectar `x_true` del
donante sobre la incidencia receptora, son controles distintos no congelados.
Un fallo así no debe convertirse en `R6=FAIL` y luego en
`NO_EXECUTABLE_SUCCESSOR`: es una carencia del diseño del control.

**Corrección requerida.** Congelar una operación total. Una opción es permutar
donantes una sola vez a nivel master dentro de `(split,n_nodes)`, transportar el
potencial del donante al gauge canónico y recomputar el target relacional sobre
la incidencia de cada vista receptora, conservando el mismo donante para todas
las vistas del master. Otra es definir un control topológico distinto, con su
estimando y soporte explícitos. Añadir fixture con dos receptores de igual
`n_nodes` y distinto `n_edges`, y una mutación que detecte donantes discordantes
entre vistas pareadas.

### MEDIUM — R537-02: las mutaciones privadas obligatorias son bloqueadas por el mismo hash freeze que deberían atravesar

**Evidencia.** El preparer debe verificar todos los hashes antes de crear los
árboles y una divergencia de fuente produce `source_status:FAIL`
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:81-82,88-109`). Sin embargo, el
test contrafactual exige cambiar o eliminar, uno por uno, cada campo privado de
las fuentes W49/W50, W54 y grafo, regenerar el preparer y luego comparar
`prepared/public/`, candidato y adapters byte a byte (`ibid.:288-293`). Cada una
de esas mutaciones cambia necesariamente el SHA-256 de una fuente ligada, de
modo que el preparer fail-closed se detiene antes de extraer un árbol público.

**Impacto.** Las dos reglas son correctas por separado, pero no forman un test
ejecutable. Si se omite hash verification para la mutación, aparece un modo de
test no definido; si no se omite, la invariancia de dependencia privada nunca se
ejerce. El checker puede terminar validando sólo rechazo por hash, no que el
builder sea contrafactualmente independiente de truth.

**Corrección requerida.** Separar explícitamente dos suites: (a) mutaciones de
fuentes canónicas, que deben fallar por hash antes del build; y (b) fixtures
sintéticos autenticados por un trust root de tests, con dos variantes que
comparten extracción pública y difieren sólo en privado. La segunda suite debe
ejecutar preparer+builder completos y exigir igualdad byte-exacta. El plan debe
prohibir que el modo fixture reautorice las fuentes científicas o pueda producir
una hoja semántica.

### MEDIUM — R537-03: `HARD_MAP_SET` y las 17 features W56 usan dos conjuntos hard distintos

**Evidencia.** El nuevo hard reader elige el set MAP de cada posterior y deriva
la acción desde él (`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:211-218`). El
contextual declara reutilizar las 17 features W56 con la masa y los riesgos de
ese posterior (`ibid.:220-230`). Pero la primitive W56 no deriva
`hard_cardinality` ni `posterior_mass_hard_set` del hard reader recibido:
reconstruye otro set mediante `sigmoid(ensemble_logits) >= 0.5`, con fallback
argmax (`src/geometria_proporcional/wave56_contextual_gate.py:61-69,72-83,117-120`).

La diferencia ocurre en los datos ligados, no sólo en un borde hipotético. El
set MAP difiere del set threshold-W56 en `11/384` tokens MARGINAL y `57/384`
tokens JOINT; en JOINT hay `62` bits distintos
(`data/geometria_proporcional/wave54_joint_set_inputs_v1/fit_select_bundle.npz::{ensemble_logits}`;
`data/geometria_proporcional/wave54_joint_set_v1/selection_freeze.json`, theta
`selected_models.joint_full.theta`).

**Impacto.** Una implementación puede llamar `contextual_design` sin cambios y
usar features referidas al hard histórico, mientras otra puede adaptar esas dos
features al nuevo `HARD_MAP_SET`. Ambas lecturas son plausibles y producen
designs, coeficientes, thresholds y acciones distintos. La independencia del
checker no resuelve una especificación ambigua.

**Corrección requerida.** Elegir una semántica y escribir su fórmula exacta. Si
el baseline del reader es el set MAP, pasar ese set explícitamente al design y
derivar de él cardinalidad y masa. Si se desea conservar el threshold histórico
como feature inference-safe, renombrar ambas features para que no representen al
baseline, fijar `tau=0.5` y declarar que el contextual observa simultáneamente
el set histórico y el hard MAP. Añadir una mutación donde ambos sets difieren y
exigir el digest exacto del design.

## Comprobaciones consolidadas

| Check | Resultado |
|---|---|
| identidad target/parent/pathset | PASS |
| SHA-256 del plan | PASS |
| hashes declarados de fuentes | PASS, 29/29 |
| roles público/privado | PARTIAL: separación definida; suite contrafactual inconsistente |
| álgebra total y hojas | PASS |
| cuatro celdas set-valued | PASS numérico; REVISE semántica de features hard |
| checker independiente | PASS en separación; PARTIAL en prueba de no dependencia privada |
| mutaciones | REVISE |
| replay sin circularidad | PASS |
| budget terminal | PASS como contrato; smoke base compatible |
| uso o consulta de GPU | no |
| apertura de lockbox/sealed monitor | no |
| modificación del plan/código/datos | no |

```json
{
  "schema_version": "proportional-mapping-feasibility-plan-reaudit-v1",
  "audit_id": "R537",
  "target": {
    "commit": "9c43659f930dd66a2c834f6749bafedea1e18801",
    "parent": "65153e99f669302d80c84c551ebf417a941994c0",
    "file": "experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md",
    "sha256": "c3d6c8ff6ca670e75269622f216188b3ab0a41b6dd5e980f117267b9a5f9b97e",
    "git_blob": "b1deced5b0bf30d27ca37970198458bf8b8c802c"
  },
  "prior_audit": {
    "audit_id": "R536",
    "sha256": "656b6e23ce1354eed79fc3887aee63faaf77e343838fc20a4bb8b339eccd75c6"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 3,
    "low": 0
  },
  "r536_resolution": {
    "R536-01": "RESOLVED",
    "R536-02": "RESOLVED",
    "R536-03": "PARTIAL",
    "R536-04": "PARTIAL",
    "R536-05": "RESOLVED"
  },
  "source_hashes": {
    "matched": 29,
    "declared": 29
  },
  "set_valued_four_cells_numerically_executable": true,
  "relational_target_shuffle_total": false,
  "private_counterfactual_suite_executable_as_written": false,
  "replay_cycle_free": true,
  "plan_modified": false,
  "implementation_performed": false,
  "lockbox_or_sealed_monitor_opened": false,
  "gpu_used_or_queried": false
}
```
