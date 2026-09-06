# R536 — Auditoría independiente del plan `MAPPING-FEASIBILITY`

## Dictamen técnico: REVISE

El plan formula correctamente la pregunta de fondo: no presupone que EIV,
grafo relacional y posterior set-valued compartan objeto, impide convertir
WLS/IRLS en readers por renombrado y reserva promoción y `GO/NO-GO` al usuario.
También acierta al exigir un gate CPU anterior a cualquier factorial neuronal.

Sin embargo, el contrato todavía no puede producir de forma auditable una de
sus tres salidas. Encontré **1 HIGH, 4 MEDIUM y 0 LOW**. El defecto central no
es que la evidencia vigente parezca conducir a una bifurcación —la ausencia de
una unidad común puede ser un resultado legítimo—, sino que el plan no separa
validez del artefacto de resultado científico ni define predicados nativos
totales. En ese estado, un parser incompleto, una fuente sin autoridad o una
decisión elegida por el builder podrían terminar bajo la misma etiqueta que una
imposibilidad semántica real.

## Identidad y alcance

- Target: commit `eab92704c5cb71cb2ef18dbf14a6f35ddc53aac7`.
- Parent directo: `be4bac8034fc6adca2e74d27a4fc498979a9897f`.
- Pathset del target: sólo
  `experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md`.
- SHA-256 del plan auditado:
  `5068c437c5f49772dfad2ea6414f74b3d4c936b52da6f9dd7769cc1975430fca`.
- El worktree estaba limpio antes de crear este informe.
- Leí completo el plan, la síntesis terminal, R533 y R535, el schema de Ola 49,
  fixtures/predicciones/oracle de desarrollo, el cierre de Ola 49, las
  primitives, plan, preparación, bundles y cierre de Ola 54, el contrato grafo,
  sus manifests/estados raw y los contratos contextuales relevantes de Olas
  56–60.

## Findings

### HIGH — R536-01: no existe una álgebra total de validez y adjudicación; dos salidas no tienen predicados comprobables

**Evidencia.** El plan nombra los tres estados y define
`COMMON_FACTORIAL_FEASIBLE` mediante `M1..M5=PASS`
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:17-24,107-108`), pero no define
qué condiciones booleanas hacen que sobrevivan los dos contratos nativos. Las
secciones relacional y set-valued describen una comparación futura, no un test
de ejecutabilidad con campos requeridos, criterios de `PASS/FAIL` y razones
cerradas (`ibid.:110-141`). Por eso `BIFURCATE_NATIVE_CONTRASTS` y
`NO_EXECUTABLE_SUCCESSOR` no se derivan de una tabla ejecutable. El único
positivo sintético exigido prueba aceptación de un mapping común
(`ibid.:159-160`); no demuestra que el checker pueda alcanzar correctamente la
bifurcación ni el estado sin sucesor.

Además, el plan no separa `artifact_status`/`checker_status` de la salida de
mapeo. Un hash incorrecto, una fuente no parseable, un budget excedido o un
checker inválido podrían parecer un «contrato nativo fallido» y producir
`NO_EXECUTABLE_SUCCESSOR`. Eso convertiría una falla de evidencia en una
conclusión sobre la arquitectura.

**Impacto.** La salida puede quedar decidida por lógica ad hoc del builder o
por una falla operacional. En consecuencia, el gate no satisface aún la
pregunta recibida de decidir entre tres estados sin estar cableado al resultado.

**Corrección requerida.** Antes de implementar:

1. declarar predicados cerrados y machine-readable `M1..M5`,
   `RELATIONAL_NATIVE_1..n` y `SET_NATIVE_1..n`, con evidencia fuente y reason
   code para cada uno;
2. fijar la tabla total:
   `all(M) -> COMMON_FACTORIAL_FEASIBLE`; `not all(M) and all(R) and all(S) ->
   BIFURCATE_NATIVE_CONTRASTS`; `not all(M) and (not all(R) or not all(S)) ->
   NO_EXECUTABLE_SUCCESSOR`;
3. separar `artifact_status`, `source_status`, `checker_status`,
   `replay_status` y `mapping_decision`; cualquier invalidez deja
   `mapping_decision:null` y nunca cuenta como refutación;
4. añadir fixtures positivos y mutaciones que ejerzan las tres hojas, cada M y
   cada predicado nativo, verificando tanto aceptación como reason code.

### MEDIUM — R536-02: los bindings ligados no reconstruyen los contratos que la tabla les atribuye

**Evidencia.** La tabla de fuentes (`ibid.:30-41`) contiene hashes físicamente
correctos, pero varios objetos tienen otro rol:

- `wave49/predictions/train.jsonl` contiene `19.968` salidas —cuatro selectors
  por `fixture_id`— y no contiene `oracle_compatible_set`, `oracle_status`,
  `family_id` ni `pair_token`. No puede fundar el target nativo declarado en
  `ibid.:52`; la autoridad de desarrollo está en el oracle sellado y debe
  mantenerse separada de la predicción.
- `wave54_joint_set_v1/posterior_state.npz` contiene masas, acciones, riesgos,
  targets y utilidades, pero no `ensemble_logits`, `per_seed_logits`,
  `design_stratum` ni `cluster_id`. Los logits y metadata que el plan promete
  inventariar (`ibid.:53`) viven en los bundles preparados. La preparación
  muestra la lista efectiva de campos
  (`prepare_wave54_inputs.py:170-209,272-286`) y el protocolo exige ligar
  manifest y bundles (`WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:20-51`).
- Para el grafo se liga sólo `raw_typed|seed=104729.npz`. Ese archivo contiene
  los arrays públicos, pero también `x_true`, `clean_log_ratio`,
  `causal_corruption_mask`, `split`, `master_id`, `view_id` y `mechanism`; no
  representa una superficie pública. Tampoco acredita por sí solo el brazo
  `GENERIC`, el segundo seed ni la paridad WLS/IRLS. El contrato separa de forma
  explícita `PUBLIC_ARRAY_FIELDS` y `FORBIDDEN_PUBLIC_FIELDS`
  (`proportional_graph_contract.py:19-47`) y tipa la autoridad privada aparte
  (`ibid.:143-176`).

**Impacto.** `source_inventory.json` no puede reconstruir observación, target y
salidas declaradas usando sólo el mínimo normativo. Añadir archivos libremente
durante la implementación tampoco es un freeze: desplaza la elección de
autoridad al builder.

**Corrección requerida.** Sustituir la tabla mínima por una matriz congelada
`fuente × rol × campos permitidos × fase de acceso × hash`: (a) fixture público,
oracle de desarrollo y predicción EIV separados; (b) manifest y
`fit_select_bundle.npz` de Ola 54 ligados —sin abrir monitor para construir el
mapping—; (c) manifests/config/cómputo y todos los estados `GENERIC/TYPED ×
seed` necesarios para probar el contrato grafo. Toda fuente adicional debe
quedar nombrada y hasheada en el plan resuelto, no elegirse en runtime.

### MEDIUM — R536-03: el contraste `marginal/joint × hard/contextual` no identifica aún cuatro celdas ejecutables

**Evidencia.** El plan nombra `hard/contextual` (`ibid.:129-141`) pero no liga ni
identifica cuál contrato contextual usa, qué parámetros están congelados o si
el reader se ajusta una vez, por posterior o sobre una IR común. La primitive
histórica contextual no es una función ciega `scores -> action`: recibe logits
ensemble/per-seed, masa sobre conjuntos, riesgo por acción, acciones hard y
posterior y utilidad (`wave56_contextual_gate.py:72-162`). `hard`, en cambio, se
deriva directamente de los logits (`ibid.:61-69`), de modo que las dos celdas
hard serían idénticas por construcción si sólo se cambia marginal por joint.
Ola 54 también define `hard_set_policy` y los posteriors como brazos distintos,
no como el factorial ahora propuesto
(`WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:83-104`).

El término `contextual` puede además referir a la ridge de Ola 56 o a las
pipelines proposer/guard posteriores; son contratos diferentes. La síntesis
terminal exige proposer y guard separados y un lifecycle congelado
(`PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md:143-171`), mientras el
plan presente no selecciona una de esas realizaciones.

**Impacto.** No se puede decidir que el contraste set-valued «sobrevive» ni
atribuir una interacción representación × decisión: cambiar simultáneamente
posterior, features, acción candidata y entrenamiento del gate mezcla los
factores; reutilizar hard duplica una celda y vuelve estructuralmente nulo uno
de los efectos.

**Corrección requerida.** Elegir y ligar un único contrato de decisión;
materializar las cuatro celdas con funciones, inputs, parámetros, fitting y
freeze explícitos; declarar qué cantidades cambian con el posterior y cuáles
son comunes; y registrar que las celdas hard son duplicadas si ésa es la
semántica elegida. Si no existe un estimando interpretable tras esa
especificación, `SET_NATIVE=FAIL` debe surgir del predicado nativo, no ser
reparado por renombrado.

### MEDIUM — R536-04: la independencia del checker y la barrera contra leakage son declarativas

**Evidencia.** El plan exige que el checker reconstruya decisiones
(`PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:100-105`) y enumera mutaciones
contra inyección de truth (`ibid.:143-160`), pero no define la raíz de confianza,
la interfaz entre builder/evaluator/checker, los imports permitidos ni qué
proceso puede abrir fuentes privadas. Esto es material porque el NPZ grafo
ligado ya mezcla observación y truth, y porque el join Wave 49
`fixture_id -> pair_token -> oracle target` requiere autoridad que no pertenece
al adapter. Una prueba que rechaza una clave prohibida no demuestra que el
builder no haya usado esa clave antes de omitirla de su output.

**Impacto.** Builder y checker pueden concordar por helper compartido, o el
builder puede leer truth para producir un mapping aparentemente limpio. Las
mutaciones de outputs serializados no detectan dependencia contrafactual de
campos privados.

**Corrección requerida.** Definir procesos y superficies físicas: preparador
allowlist público; builder que sólo recibe esa extracción; evaluator/checker que
abre truth después del candidate freeze; checker externo con implementación de
predicados y joins independiente del builder y sin importar sus helpers. El
checker debe recomputar hashes, keysets, shapes, bijecciones y tabla de
adjudicación desde fuentes ligadas. Añadir mutaciones contrafactuales que cambien
o eliminen cada campo privado manteniendo fijo el input público y comprueben que
mapping/adapters no cambian, además de corrupciones del candidate artifact que
el checker deba rechazar.

### MEDIUM — R536-05: replay, manifest y presupuesto no tienen semántica terminal compatible

**Evidencia.** Se exige igualdad byte a byte de todo menos `runtime`
(`ibid.:162-179`), pero no se dice si `manifest.json` incluye el hash de
`runtime.json`; si lo incluye, el manifest también cambia y contradice la única
excepción. Tampoco se fijan canonical JSON, paths relativos, timestamps,
clasificación de self-manifest ni comparación de reportes. La historia inmediata
de Ola 60 demuestra que un receipt operacional no normalizado puede producir un
`MISMATCH` científico espurio; R535 conserva justamente esa distinción.

El límite de diez minutos y 2 GiB es un «objetivo» (`ibid.:181-188`), no una
guarda que anule adjudicación. No se especifican medición, enforcement ni estado
cuando build, replay o tests exceden el presupuesto.

**Impacto.** Dos ejecuciones científicamente idénticas pueden fallar el replay,
o una ejecución fuera de contrato puede publicar una de las tres decisiones.

**Corrección requerida.** Congelar un inventario de clases: artefactos
deterministas byte-exactos, runtime/telemetría no científicos y self-manifests;
usar JSON canónico, paths repo-relative y cero tiempo/UUID en la clase exacta;
declarar si runtime queda fuera del manifest científico o se compara mediante
normalización tipada; ejecutar replay en proceso/directorio independientes y
emitir un receipt por clase. Medir wall y `ru_maxrss`, imponer ambos límites y
dejar `mapping_decision:null` con `artifact_status=BUDGET_EXCEEDED` si se
incumplen.

## Respuesta a la pregunta de auditoría

Con la evidencia actual, es esperable que `M1` falle: los artefactos no ofrecen
una correspondencia total entre fixtures/pair tokens y masters de grafo, y sus
targets no comparten espacio. Eso apoya la utilidad del gate; no autoriza aún a
publicar `BIFURCATE_NATIVE_CONTRASTS`. Hasta definir y ejercer los predicados
nativos, esa etiqueta y `NO_EXECUTABLE_SUCCESSOR` dependen del criterio no
congelado del implementador.

Tras las correcciones anteriores, el diseño sí puede ser finito, CPU y
diagnóstico: la salida común requerirá los cinco predicados de mapping; la
bifurcación requerirá dos contratos nativos íntegros; y la ausencia de sucesor
será una conclusión semántica sólo después de validar fuentes, checker, replay y
presupuesto.

## Comprobaciones realizadas

| Check | Resultado |
|---|---|
| identidad target/parent/pathset | PASS |
| hashes declarados en el plan | 8/8 coinciden |
| inspección de schemas y keysets nativos | completada |
| uso o consulta de GPU | no |
| modificación del plan u otros archivos | no |

```json
{
  "schema_version": "proportional-mapping-feasibility-plan-audit-v1",
  "audit_id": "R536",
  "target": {
    "commit": "eab92704c5cb71cb2ef18dbf14a6f35ddc53aac7",
    "parent": "be4bac8034fc6adca2e74d27a4fc498979a9897f",
    "file": "experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md",
    "sha256": "5068c437c5f49772dfad2ea6414f74b3d4c936b52da6f9dd7769cc1975430fca"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 1,
    "medium": 4,
    "low": 0
  },
  "three_outcomes_total_and_unwired": false,
  "source_binding_sufficient": false,
  "checker_independence_sufficient": false,
  "implementation_specification_sufficient": false,
  "plan_modified": false,
  "gpu_used_or_queried": false
}
```
