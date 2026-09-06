# Plan CPU — `MAPPING-FEASIBILITY` para el relevo proporcional

> **Estado:** `PLAN-CANDIDATE / PRE-IMPLEMENTATION / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Autoridad de promoción:** usuario

## 1. Pregunta finita

Este gate no busca una geometría nueva ni entrena un modelo. Decide si tres
objetos ya existentes pueden entrar en un mismo contraste causal sin cambiar de
pregunta durante el adapter:

1. la referencia clásica EIV + conformal de Ola 49;
2. el núcleo relacional `GENERIC/TYPED` y sus executors WLS/IRLS;
3. el posterior `marginal/joint` y sus readers `hard/contextual`.

El gate termina en uno de tres estados:

- `COMMON_FACTORIAL_FEASIBLE`: existe una query, unidad, observación, target e
  interfaz de scores comunes y los adapters preservan información y autoridad;
- `BIFURCATE_NATIVE_CONTRASTS`: el mapeo común falla, pero sobreviven un
  contraste relacional y otro set-valued nativamente interpretables;
- `NO_EXECUTABLE_SUCCESSOR`: también falla al menos uno de los dos contratos
  nativos y hace falta rediseño antes de experimentar.

Ningún estado promueve una arquitectura ni constituye `GO/NO-GO`.

## 2. Evidencia ligada

El checker fijará por SHA-256, como mínimo:

| Fuente | SHA-256 |
|---|---|
| `PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md` | `888477866168d1f2e5dbb756835ef0ed6521a636ab8cd752aaaa6181dc490db2` |
| `src/geometria_proporcional/wave49_schema.py` | `135c3d4994019cfccf3bc2ec3b7c8c6c9b4772a898eb3a900f58c0135e183f36` |
| `data/geometria_proporcional/wave49/visible/train.jsonl` | `2670bee5cb0312d78caf65261f2bf32fc5869038d5a472e056161943d0176010` |
| `data/geometria_proporcional/wave49/predictions/train.jsonl` | `0a1c8b58c8cf6271c8702d66144e30cd63251e4464afa7ff4fb6b2d648088f2e` |
| `src/geometria_proporcional/wave54_joint_set.py` | `75b3bb5a65f145a9b7a4787580576723366ac3ca0d49cbc007394220f6d98901` |
| `data/geometria_proporcional/wave54_joint_set_v1/posterior_state.npz` | `9023e372a5fb2d9cd8dc4d4f5b4a4200792d9ece5207515f3bbb645f86cfc999` |
| `src/geometria_proporcional/proportional_graph_contract.py` | `a2d8c3ebf327aded797b89eddcdcd2cc315fc0a25bf9b49a197878a90c3f5a53` |
| `data/geometria_proporcional/proportional_graph_neural_smoke_v1/raw_eval/raw_typed\|seed=104729.npz` | `7e66148516a1a921a6832178770a392ff5fcf8d74839c6d0e0f0901353ba7a26` |

Los artefactos se leen; no se modifican, recalibran ni reejecutan. El preflight
no abre un lockbox ni usa labels privadas para construir adapters.

## 3. Inventario nativo que debe reconstruirse

El resultado no partirá de nombres de módulos, sino de contratos observados.

| Línea | Unidad nativa | Observación autorizada | Target nativo | Salida previa al reader |
|---|---|---|---|---|
| EIV/Ola 49 | fixture de pares `(x,y)` con `n∈{8,16,24}` | vectores `x`, `y`, covarianza y semántica de coordenadas | conjunto de familias funcionales compatibles entre cuatro catálogos, más OOD | scores por familia, conjunto estructural, cutoff conformal |
| posterior/Ola 54 | `pair_token` de Ola 50–53 | cuatro logits ensemble y metadata de estrato | conjunto booleano no vacío sobre cuatro familias | masa sobre quince conjuntos y riesgo por acción/utilidad |
| grafo proporcional | `master_id/view_id` con grafo de 8–16 nodos | aristas, log-ratios observadas, varianzas y caminos orientados | potenciales módulo gauge y relaciones limpias; mecanismo de corrupción sólo privado | relación corregida y confiabilidad por arista antes de WLS/IRLS |

La implementación verificará estos campos directamente en schemas y artefactos,
incluidas shapes, cardinalidades, namespaces de IDs y campos prohibidos.

## 4. Condiciones del mapeo común

Cada brazo declarará dos funciones puras:

```text
observation_to_state(public_observation) -> typed_state
state_to_compatibility_scores(typed_state) -> score_vector + uncertainty
```

El factorial común sólo es admisible si pasan conjuntamente estas condiciones:

### M1 — identidad de query y unidad

- una sola pregunta científica escrita de forma literal;
- una sola unidad independiente, no una equiparación nominal entre fixture,
  `pair_token` y vista de grafo;
- correspondencia total y verificable de IDs entre brazos.

### M2 — paridad de observación

- mismo input autorizado o proyecciones deterministas del mismo input;
- ningún brazo recibe campos privados, truth, oracle, split o mecanismo;
- ningún adapter inventa caminos, familias o covarianza ausentes en otro brazo.

### M3 — conservación del target

- un mismo espacio de verdad y una función total hacia el target común;
- ida y vuelta exacta o certificado explícito de pérdida que impida atribución
  factorial;
- una salida continua de relación no se convierte en label de familia mediante
  thresholds aprendidos del monitor.

### M4 — executor, checker y reader comunes

- el efecto representacional se mide antes del reader;
- operaciones exactas externas se dan a todos o a ninguno;
- WLS/IRLS no se presentan como readers equivalentes a hard/contextual si
  resuelven otra clase de objeto;
- EIV se separa en score, calibración, reader y abstención; si la separación
  cambia su significado, permanece referencia externa.

### M5 — autoridad y acceso

- query, schema, target y controles proceden del design freeze;
- utilidad marcada `SYNTHETIC_EXTERNAL` y ausente del encoder;
- train/calibration/selection no acceden a monitor;
- el checker puede reconstruir cada decisión sin importar el builder.

`COMMON_FACTORIAL_FEASIBLE` exige `M1..M5=PASS`. `PARTIAL` cuenta como fallo del
factorial y conserva su evidencia para diseñar la bifurcación.

## 5. Contratos nativos si el gate falla

La bifurcación no es un premio consuelo: evita una interacción causal sin objeto
común.

### 5.1 Contraste relacional

```text
representación: GENERIC vs TYPED
executor:       WLS vs IRLS
unidad:         master, con IID/grouped pareados como vistas
targets:        relación limpia y potencial módulo gauge
primarias:      relation RMSE, quotient RMSE, convergencia/fallo
```

El diseño deberá igualar input público, parámetros, inicialización, updates y
espacio de tuning. EIV podrá figurar sólo como referencia externa si no opera
sobre el mismo target.

### 5.2 Contraste set-valued

```text
posterior:      MARGINAL vs JOINT
reader:         HARD vs CONTEXTUAL
unidad:         pair_token
target:         conjunto compatible no vacío sobre cuatro familias
primarias:      proper score, cobertura/ancho, compatibilidad, regret y cola
```

Cada celda recibirá los mismos logits, utilidad contractual y controles
target-shuffled/matched. Posterior y decisión se estimarán por separado antes de
su interacción.

## 6. Mutaciones obligatorias

El checker debe rechazar, como mínimo:

1. renombrar `fixture_id`, `pair_token` o `master_id` para fingir correspondencia;
2. construir familias de Ola 49 desde truth limpia del grafo;
3. inyectar `x_true`, `clean_log_ratio`, corrupción o labels oracle al adapter;
4. convertir relación continua en cuatro labels mediante umbrales no congelados;
5. agregar caminos sintéticos sólo al brazo tipado;
6. presentar WLS/IRLS como hard/contextual sin igualdad de acción/target;
7. incorporar calibración conformal dentro del encoder EIV;
8. usar utilidad para producir el posterior representacional;
9. aceptar mapping parcial como factorial completo;
10. seleccionar una arquitectura o escribir `GO`, `NO-GO` o promoción en el
    artefacto de adjudicación.

Se añadirán positivos sintéticos mínimos para demostrar que el checker puede
aceptar un mapeo genuinamente común y no está cableado a fallar.

## 7. Artefactos y replay

La implementación producirá en
`data/geometria_proporcional/proportional_mapping_feasibility_v1/`:

- `source_inventory.json`;
- `native_contracts.json`;
- `mapping_matrix.json`;
- `mutation_results.json`;
- `adjudication.json`;
- `runtime.json`;
- `manifest.json`;
- `REPORT_MAPPING_FEASIBILITY.md`.

Un replay en directorio separado deberá igualar byte a byte todo menos runtime;
los arrays, si se usan, se compararán por dtype, shape y contenido. Cada salida
declarará `gpu_used_or_queried:false`, `architecture_promoted:false`,
`scientific_decision:null` y `decision_authority:user`.

## 8. Presupuesto y detención

- dispositivo: CPU; CUDA invisibilizada sin consultar su estado;
- objetivo: menos de 10 minutos y 2 GiB RSS para build + replay + tests;
- sin training, forward neuronal, bootstrap nuevo ni solves masivos;
- si la reconstrucción exigiera re-forward, training o una operación de muchas
  horas que GPU resolviera materialmente mejor, el gate se detiene antes de CUDA
  y se informa objetivo, duración y VRAM estimada por Telegram.

## 9. Secuencia de trabajo

1. auditar este plan con una instancia independiente;
2. resolver findings materiales y congelar plan + hashes;
3. implementar builder, checker independiente y tests de mutación;
4. ejecutar build y replay CPU;
5. auditar artefactos y adjudicación;
6. documentar una sola salida: factorial común, bifurcación nativa o rediseño.

La auditoría del plan ocurre antes de escribir el builder. La promoción de una
arquitectura y cualquier `GO/NO-GO` continúan fuera de este gate.
