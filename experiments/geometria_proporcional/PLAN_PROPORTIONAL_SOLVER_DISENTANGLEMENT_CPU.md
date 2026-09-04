# Plan CPU — desentrelazado de relación, peso y solver

**Estado:** auditado `PASS` por R346; habilitado para implementación CPU  
**Fecha:** 2026-09-03  
**Régimen:** seguimiento diagnóstico post hoc del smoke neuronal; CPU-only  
**Autoridad de promoción y GO/NO-GO:** usuario

## 1. Pregunta y alcance

El smoke neuronal proporcional produjo una separación que la salida conjunta
no permite localizar: la corrección relacional y parte de WLS mejoran, mientras
el decoder directo empeora y el Huber-IRLS aplicado a la observación cruda
supera a las salidas neuronales. Este experimento pregunta dónde aparece esa
pérdida funcional:

1. en sustituir la relación observada por la corregida;
2. en sustituir pesos unitarios por la confiabilidad aprendida;
3. en la interacción entre ambos cambios;
4. o en su dependencia del solver downstream.

La prueba es una ablación funcional de outputs congelados. No reentrena heads,
no vuelve a ejecutar el modelo y no identifica causalmente qué parte del
entrenamiento produjo cada salida. «Efecto de relación» y «efecto de peso»
designan intervenciones en inferencia sobre arrays ya materializados, no efectos
causales de módulos aprendidos.

Como la pregunta nació después de observar el smoke oficial, todo el análisis
es exploratorio y post hoc. Congelar ahora la matriz, el orden y los artefactos
reduce grados de libertad posteriores, pero no convierte la prueba en
confirmatoria ni habilita decisiones automáticas.

## 2. Fuente inmutable y frontera anti-leakage

La única fuente admitida es:

`data/geometria_proporcional/proportional_graph_neural_smoke_v1/`

El modo oficial exige antes de calcular:

- `manifest.json` con SHA-256
  `7e982a92bd366a4c22fe95c0cc9c8fd5f7a1773bd78c4e8e0781b5c2259624e1`;
- schema de artefacto `proportional-graph-neural-artifact-v1`;
- commit fuente `3a683ac9a7ef444e344b746cdd83062ff03ff30a` y estado limpio;
- schema público
  `0a219cb4991b7696291512d5aa307684bcb4960058caf88e273f8af138edf6af`;
- hash del solver `src/geometria_proporcional/proportional_graph_contract.py`
  igual al registrado por el manifest fuente;
- hashes y tamaños válidos para los dieciséis `raw_eval/<arm>|seed=<seed>.npz`,
  `raw_eval/observed_unweighted.npz`, `bootstrap_indices.npz` y
  `resolved_config.json`;
- brazos y seeds exactamente iguales a los de la configuración fuente;
- `raw_eval/observed_unweighted.npz` como fuente canónica única de todos los
  campos de `PublicGraphObservation` y de `x_true/clean_log_ratio` para scoring;
- cada raw neuronal aporta únicamente `corrected_log_ratio`, `reliability` y
  sus estados WLS/IRLS ya materializados;
- alineación exacta entre el archivo canónico y los raws neuronales de
  `view_id`, `master_id`, split, mecanismo, `n_nodes`, offsets de aristas y
  nodos, `edge_index`, `edge_valid`, observación, `x_true` y relación limpia;
- `edge_variance` de los raws neuronales debe ser exactamente la proyección
  `float32` de la varianza canónica `float64`, pero nunca se usa como input del
  solver;
- en los raws no barajados, `path_index/path_valid` deben igualar al control y
  `path_sign` debe igualar su proyección `float32`;
- los paths se validan internamente, pero quedan fuera del predicado de igualdad
  para `closure_typed_path_shuffle`, porque ese control conserva
  deliberadamente caminos intervenidos.

La reconstrucción de cada `PublicGraphObservation` usa sólo los campos
canónicos `n_nodes`, `edge_index`, `observed_log_ratio`, `edge_valid`,
`path_index`, `path_sign`, `path_valid` y `edge_variance` del control público
`observed_unweighted`. `x_true` y `clean_log_ratio` entran únicamente después
del solve para puntuar. La máscara causal no entra en relaciones, pesos ni
solver. Un test espía debe hacer verificable esta frontera.

El runner abre la fuente sólo para lectura, rechaza que el output sea la fuente
o quede dentro de ella y aborta si el destino ya existe. No ofrece `--force`.
El replay siempre escribe en un destino nuevo.

## 3. Matriz congelada

Para cada combinación `arm × seed × view` se cruzan tres factores:

| Factor | Nivel 0 | Nivel 1 |
|---|---|---|
| relación | `observed`: `observed_log_ratio` | `corrected`: `corrected_log_ratio` |
| peso base | `unit`: uno en aristas válidas | `learned`: `reliability` aprendida |
| solver | WLS | Huber-IRLS |

En IRLS, `unit/learned` describe el peso base sobre el cual actúa el reweighting
robusto; no se lo confundirá con el peso final de IRLS. Los hiperparámetros se
heredan sin tuning del smoke: `weight_floor=0.001`, `huber_delta=1.5`,
`irls_damping=1.0`, `irls_iterations=7500` y `tolerance=1e-6`, el default que
usó el runner fuente. Todos deben quedar materializados en la config resuelta.

Se procesan los ocho brazos y dos seeds del smoke. La familia principal de
lectura contiene, en el orden canónico de la configuración:

1. `raw_generic`;
2. `raw_typed`;
3. `closure_generic`;
4. `closure_typed`.

Los cuatro controles (`closure_typed_path_shuffle`, `pair_state_no_mix`,
`generic_message_passing`, `edge_mlp`) reciben la misma matriz para diagnóstico
secundario. No se abren nuevos contrastes entre brazos: ésos pertenecen al
smoke oficial. Tampoco se agregan hiperparámetros, variantes de Huber ni
umbrales después de ver los resultados.

## 4. Reuso y cómputo nuevo

Dos celdas ya existen y se reutilizan, condicionadas a la atestación de fuente:

- `observed × unit`: `raw_eval/observed_unweighted.npz`, común a todos los
  brazos y seeds;
- `corrected × learned`: el estado WLS/IRLS del archivo raw de cada
  `arm × seed`.

Se calculan por primera vez:

- `observed × learned`;
- `corrected × unit`.

WLS se recomputa para las cuatro celdas usando siempre la observación canónica
float64. Las dos celdas ya preservadas deben coincidir con su estado fuente a
tolerancia `1e-12`; una diferencia aborta la corrida. IRLS sólo se calcula para
las dos celdas nuevas; las otras dos se copian de los estados atestados. Las
posiciones `[0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 630]` de cada raw,
congeladas sin leer métricas, se recalculan para comparar `x_hat`, pesos finales,
convergencia e iteraciones contra el estado fuente. La comparación numérica usa
la observación canónica y tolerancia `1e-12`; convergencia e iteraciones deben
coincidir exactamente. Esa verificación no selecciona ni excluye vistas.

Se evalúan las `631` vistas preservadas por archivo: `127` validation y `504`
test. Validation sirve para integridad y descripción; los contrastes se
informan sobre los `252` masters test con pareja `iid/grouped`, en el mismo orden
de `bootstrap_indices.npz`.

## 5. Métricas y estimandos

Por celda y vista se preservan:

- RMSE de cociente, alineado al gauge por centrado;
- RMSE de la relación reconstruida contra la relación limpia;
- RMSE residual ponderado contra el input efectivo del solver;
- rango y condición del Laplaciano;
- convergencia e iteraciones;
- `x_hat`, relación reconstruida y pesos finales del solver.

El RMSE de cociente es la métrica primaria downstream. El RMSE de relación
reconstruida y los diagnósticos numéricos son secundarios. El error pre-solver
de la relación corregida se copia como referencia del smoke, pero no se presenta
como nuevo resultado del factorial downstream.

Sea `Q(r,w,s)` el RMSE de cociente para relación `r`, peso `w` y solver `s`.
Para cada brazo, solver y slice se calculan, con la convención de que un valor
negativo favorece la intervención nombrada:

1. **relación con peso unitario**:
   `Q(corrected,unit,s) - Q(observed,unit,s)`;
2. **peso sobre relación observada**:
   `Q(observed,learned,s) - Q(observed,unit,s)`;
3. **interacción relación × peso**:
   `Q(corrected,learned,s) - Q(corrected,unit,s)
   - Q(observed,learned,s) + Q(observed,unit,s)`;
4. **efecto total entregado**:
   `Q(corrected,learned,s) - Q(observed,unit,s)`.

La identidad `total = relación + peso + interacción` se verifica por vista y
en los agregados. Para cada estimando se calcula además la interacción con el
solver:

`estimando(IRLS) - estimando(WLS)`.

Un valor negativo allí significa que el estimando es más favorable —o menos
desfavorable— bajo IRLS. No implica que IRLS tenga menor error absoluto; los
niveles absolutos se reportan junto a los efectos.

El orden de slices es:

1. `test|iid`;
2. `test|grouped`;
3. `grouped_minus_iid`.

## 6. Unidad, seeds, bootstrap y multiplicidad

La unidad de inferencia sigue siendo el master. Primero se promedian los dos
seeds dentro de `arm × view × celda`; sólo después se forman los efectos por
master. IID y grouped permanecen pareados. Los `2.000` índices bootstrap y el
orden de los `252` masters se copian byte por byte del artefacto fuente y se
usan para todas las celdas y efectos, preservando su covarianza.

Se publican media e intervalo bootstrap marginal del 95% para todas las celdas
con estatus estricto evaluable. Para un nivel IRLS, «evaluable» exige que los
dos seeds sean finitos para cada master del slice completo; cualquier ausencia
produce `NOT_EVALUABLE_SOLVER_FAILURE` y suprime media e intervalo inferenciales.
No se usan los intervalos como tests confirmatorios, no se inventa un umbral de
efecto y no se elige una arquitectura por cruce de cero. Por tratarse de una
matriz exploratoria post hoc, no se reportan `p-values` ni una corrección de
multiplicidad que pudiera sugerir un estatus confirmatorio falso. El informe
muestra la familia completa, el orden congelado y advierte que los intervalos
son marginales, no simultáneos.

## 7. Fallos IRLS

La no convergencia es un resultado del acoplamiento, no un dato descartable.

- Cada celda informa conteo y tasa de fallo por brazo, seed, split y mecanismo.
- El estado raw conserva igualmente la última solución, pesos e iteraciones.
- El RMSE inferencial de una celda IRLS no convergente se marca `NaN`.
- Si cualquiera de los dos seeds falla para un master, el nivel
  `Q(r,w,IRLS)` seed-promediado de ese master es `NaN`; si ocurre para cualquier
  master del slice, el nivel inferencial completo queda
  `NOT_EVALUABLE_SOLVER_FAILURE`, sin CI.
- Si falta cualquier término de un efecto en cualquiera de los dos seeds para
  uno de los masters de su universo, el estimando IRLS completo del slice se
  marca `NOT_EVALUABLE_SOLVER_FAILURE`; no se usa complete-case.
- Toda interacción WLS/IRLS afectada hereda el mismo estado no evaluable.
- Los niveles finitos pueden resumirse sólo como diagnóstico descriptivo y
  deben vivir en un bloque separado con `n_total`, `n_finite` y tasa de fallo;
  nunca reemplazan el nivel o estimando estricto.
- Una excepción estructural del solver, topología desconectada o array inválido
  aborta la corrida. Sólo la no convergencia declarada sigue como outcome.

## 8. Lectura permitida

La matriz puede localizar, sin decidir arquitectura:

- corrección útil con pesos unitarios pero no con pesos aprendidos;
- peso aprendido útil sobre observación, pero no sobre relación corregida;
- interacción adversa entre las dos salidas;
- diferencia del mismo efecto entre WLS e IRLS;
- cambio de esos patrones entre IID y corrupción agrupada;
- fallos de convergencia inducidos por una celda concreta.

No puede atribuir el patrón al entrenamiento de un head aislado, acreditar una
geometría natural, demostrar transferencia física, promover una primitive ni
decidir GO/NO-GO. Una lectura compatible con heads o pérdidas
solver-específicas queda registrada como hipótesis de diseño y requeriría un
experimento nuevo; no se implementa como consecuencia automática.

## 9. Artefactos obligatorios

El paquete `proportional_graph_solver_disentanglement_v1/` debe contener:

- `resolved_config.json`;
- `source_attestation.json`, con hashes, tamaños, commit, schema y archivos
  efectivamente usados;
- `per_view_metrics.npz`, con ejes y métricas completos;
- `raw_solver/<arm>|seed=<seed>.npz`, con `x_hat`, relaciones reconstruidas,
  pesos finales, offsets e identificadores para las ocho celdas;
- `summary.json`, separado por brazo, seed, split, mecanismo, relación, peso y
  solver;
- `effects.json`, con niveles, cuatro estimandos, interacción con solver,
  estatus, conteos e intervalos;
- `failure_diagnostics.json`;
- `bootstrap_indices.npz`, copia byte-exacta de la fuente;
- `DISENTANGLEMENT_REPORT.md`;
- `replay.sh`, CPU-only y dirigido a un destino nuevo;
- `manifest.json`, que incluye hashes del plan, config, runner, solver y fuente;
- `runtime_observation.json`, excluido del perímetro byte-exacto.

Todos los JSON usan orden estable y `allow_nan=false`; `NaN/Inf` se serializan
como `null` sólo en vistas derivadas, con estatus y conteos explícitos. El
manifest enumera todos los artefactos deterministas excepto él mismo y runtime.
El replay debe reproducir byte por byte cada archivo manifestado.

En modo oficial, plan, config, runner y solver deben estar trackeados y limpios.
El manifest registra `HEAD`, branch, estado y hash del estado, además de Python,
NumPy, SciPy, plataforma, variables de threads e inventario de threadpools. Ese
commit y los hashes hacen recuperables los ejecutables mediante Git. Antes de
calcular, `replay.sh` verifica que esos cuatro archivos coincidan con los hashes
oficiales y aborta con instrucción de recuperar el commit registrado si no
coinciden; nunca ejecuta silenciosamente otro working tree.

## 10. Presupuesto y guardas CPU

- dispositivo: CPU exclusivamente;
- sin importación de `torch`, sin forward y sin carga de checkpoints;
- comando oficial y replay con `CUDA_VISIBLE_DEVICES=''`;
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` y límite
  equivalente mediante `threadpoolctl`, todos registrados en el manifest;
- techo: `7.200 s` y `8 GiB` de RSS;
- chequeo de presupuesto durante el loop, no sólo al final;
- piloto técnico previo sobre un subconjunto fijo para medir costo, sin producir
  inferencia ni modificar la matriz oficial;
- monitoreo periódico de RAM durante la corrida.

El piloto independiente R345 ejecutó `1.024` solves nuevos sobre `32` vistas
por archivo en `9,18 s`, sin fallos, con máximo de `1.999` iteraciones y
`~0,059 GiB` de RSS. Su proyección lineal para `20.192` solves es `~181 s`.
Esto acredita margen operativo, no un límite garantizado de peor caso.

Si el piloto proyecta más de dos horas, no se deriva el trabajo a GPU ni a otra
máquina: se optimiza reuso/serialización o se deja la ejecución pendiente. La
suspensión GPU prevalece sobre cualquier autorización anterior.

## 11. Implementación y pruebas antes de la corrida oficial

Se agregarán:

- `experiments/geometria_proporcional/configs/proportional_graph_solver_disentanglement_v1.json`;
- `experiments/geometria_proporcional/run_proportional_graph_solver_disentanglement.py`;
- `tests/test_proportional_graph_solver_disentanglement.py`.

La suite debe cubrir como mínimo:

1. schema estricto y hashes de fuente;
2. rechazo de destino existente, alias o descendiente de la fuente;
3. loader ragged y alineación entre archivos;
4. frontera public/private mediante un solver espía;
5. mapeo exacto de las cuatro celdas y del peso base IRLS;
6. reuso de `observed×unit` y `corrected×learned` con verificación;
7. álgebra e identidad de los cuatro estimandos;
8. promedio de seeds antes del bootstrap por master;
9. pairing IID/grouped e índices fuente byte-exactos;
10. propagación conservadora de fallos IRLS en niveles, efectos, total e
    interacciones con solver;
11. preservación de los últimos estados aun sin convergencia;
12. serialización determinista, manifest y replay a destino nuevo;
13. integración tiny CPU sin importar `torch` ni tocar CUDA;
14. fuentes nuevas trackeadas/limpias, entorno/threads persistidos y replay que
    rechaza hashes ejecutables divergentes.

Después de implementar se ejecutan suite focal, regresión proporcional y una
auditoría independiente del diff. Sólo con esos controles cerrados se lanza la
corrida oficial CPU, seguida por replay, análisis y propagación documental.

## 12. Cola GPU explícita

Quedan en cola, sin ejecución hasta una nueva orden del usuario:

1. freeze confirmatorio de más seeds si el desentrelazado justifica una pregunta
   estable y su costo excede razonablemente CPU;
2. entrenamiento de heads o pérdidas específicos por solver;
3. transferencia de la primitive a un dominio físico o a Atención Armónica si
   requiriera aceleración.

Ninguna entrada de esta cola constituye una recomendación de promoción ni una
reserva automática de GPU.
