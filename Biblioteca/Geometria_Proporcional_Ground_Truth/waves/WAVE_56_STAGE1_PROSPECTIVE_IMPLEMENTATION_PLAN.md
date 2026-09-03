# Ola 56 Stage 1 — plan de implementación prospectiva fresca

> **Estado:** `AUDITED-FROZEN-FOR-IMPLEMENTATION / PRE-KEY-DRAW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Plan científico padre:** `WAVE_56_CONTEXTUAL_RESIDUAL_GATE_PLAN.md`
> **Selección retrospectiva:** `WAVE_56_STAGE0_RETROSPECTIVE_CLOSED.md`

## Propósito

Stage 1 somete a falsación prospectiva la única receta seleccionada por Stage
0: `ridge_contextual(alpha=1.0)`. No compara nuevamente familias ni reabre la
grilla de regularización. Su pregunta es si el contexto disponible en
inferencia permite reducir regret sobre una realización fresca de la misma ley
sin perder accuracy ni compatibilidad, y si esa mejora supera al gate escalar,
al Ridge advantage-only y a targets barajados.

Toda la ejecución es CPU-only. No se modifica ni reentrena el encoder. Se
reutilizan el normalizador y los checkpoints `sigmoid_only` seeds `17/29/43`,
el posterior conjunto y las veinticuatro políticas ya congeladas.

## Freeze previo a las claves

Una config prospectiva separada fija antes de generar:

- familia primaria `ridge_contextual`, `alpha=1.0`, diecisiete features;
- control `ridge_advantage_only`, `alpha=100.0`;
- cinco seeds de shuffle `{56031,56032,56033,56034,56035}`;
- cuantiles `{0.50,0.60,0.70,0.80,0.90,0.95,0.975}` y `hard_only`;
- grilla escalar de Ola 55 `{0,0.01,0.02,0.05,0.1,0.2,0.4,hard_only}`;
- constraints, mínimos, shards y cinco mil bootstraps del plan padre;
- hashes de `selection_freeze.json` y `analysis_core.json` de Stage 0, policy
  manifest, posterior de Ola 54, normalizador y checkpoints de Ola 51;
- bootstrap PCG64 seed `5607`, `5000` réplicas e índice canónico por
  `pair_token` ordenado lexicográficamente;
- nombres canónicos de primary/replay y roles físicos
  `train→gate_fit`, `val→gate_select`, `lockbox→sealed_monitor`.

Plan, config, preparador, worker, ejecutor, primitivas y tests deben estar
versionados, limpios y auditados independientemente antes de extraer las tres
claves nuevas. El preflight corre antes de crear el output.

## Preparación ciega

`prepare_wave56_fresh.py` genera una sola realización del benchmark con claves
nuevas o, en replay, reutiliza exactamente las de primary. Después del
preflight y antes de invocar el generador, crea las tres claves en memoria y
persiste un escrow `0600`: tempfile en el mismo directorio, `fsync` del
archivo, `os.replace` y `fsync` del directorio contenedor. De inmediato publica
`pre_generation_freeze.json` con la misma secuencia atómica, reabre ambos
archivos y verifica payload, permisos, hashes y compromisos. Sólo después
invoca `generate_benchmark`. El escrow contiene las tres claves, commit, hashes
de todas las fuentes, config, bindings y compromisos; el freeze público deriva
del mismo payload sin secretos. Si el proceso cae después del escrow, toda
recuperación reutiliza esas tres claves y reconstruye/verifica el freeze antes
de generar; si cae antes, no existe ningún benchmark ni material observado.

La generación produce observaciones visibles y verdad sellada, pero no
materializa oracle, labels autorizados ni bundles analíticos. Al terminar, el
árbol `benchmark/sealed/` queda propiedad de root y modo `0700`; el escrow,
`0600`. Los visibles, normalizador y checkpoints inference-only se copian a un
staging disjunto sin symlinks. El proceso analítico no recibe la raíz del
benchmark.

Un worker inference-only recibe un staging mínimo con:

- los visibles de `train`, `val` y `lockbox`;
- protocol config;
- config prospectiva;
- normalizador;
- checkpoints inference-only seeds `17/29/43`.

El staging prohíbe paths con oracle, labels, sealed truth, optimizer, history o
symlinks. El worker corre con UID/GID sin privilegios mediante `setpriv`, con
working directory dentro del staging; un test de acceso debe demostrar
`PermissionError` al intentar abrir cualquier truth sellada. Infiere los tres
splits y preserva logits raw por seed. Sólo después se escribe
`preparation_freeze.json`, que fija hashes de visibles, logits, staging,
upstream, fuentes, commit y compromisos criptográficos. En este punto deben
seguir ausentes `authorized_labels/`, `bundles/`, `fit_freeze.json` y
`selection_freeze.json`.

Primary admite un único draw. Un fallo con escrow sólo puede recuperarse con
ese escrow y el mismo commit/config/fuentes congelados. `--force` archiva y
nunca borra. Replay vive en otro directorio y compara compromisos, visibles e
inferencia array por array.

## Máquina de estados física

`run_wave56_contextual_gate.py` es un coordinador mínimo y expone tres comandos
recuperables. No calcula métricas ni carga truth. Invoca dos ejecutables
separados: un materializador root split-scoped que sólo admite un nombre de
split de la tabla congelada, y un worker analítico sin privilegios que recibe un
staging nuevo con visibles, logits, labels autorizados y freezes ya permitidos,
pero no la raíz del benchmark. Cada worker incluye un probe negativo que
confirma que no puede abrir sellados futuros.

Cada fase escribe bajo `phases/<fase>.pending/`, registra antes y después un
inventario con hashes, hace `fsync` y sólo entonces promueve el directorio con
`os.replace` a `phases/<fase>.complete/`. La tabla de estados permitidos es:

| Estado | Puede abrir | Artefacto terminal | Próximo comando |
|---|---|---|---|
| `PREPARED` | ningún oracle | `preparation_freeze.json` | `fit` |
| `FIT_PENDING` | sólo `train` | ninguno | reanudar `fit` con mismo HEAD |
| `FIT_COMPLETE` | sólo bundle/freezes de fit | `fit_core.json` + `fit_freeze.json` | `select` |
| `FIT_NOT_EVALUABLE` | sólo material ya abierto de fit | `fit_not_evaluable.json` | terminal; monitor permanece sellado |
| `SELECT_PENDING` | `train` + sólo `val` | ninguno | reanudar `select` con mismo HEAD |
| `SELECT_COMPLETE` | fit/select autorizados | `selection_core.json` + `selection_freeze.json` | `adjudicate` |
| `SELECT_NOT_EVALUABLE` | fit/select autorizados | `selection_not_evaluable.json` | terminal; monitor permanece sellado |
| `ADJUDICATE_PENDING` | los tres splits | ninguno | reanudar con mismo HEAD |
| `COMPLETE` | paquete cerrado | `analysis_core.json` + manifest | ninguno |
| `MONITOR_NOT_EVALUABLE` | paquete cerrado sin adjudicación | `monitor_not_evaluable.json` + manifest | terminal |

Antes de reanudar un `.pending`, se validan los hashes de todos sus inputs y
outputs presentes. Un estado incompatible aborta sin abrir material nuevo.

### `fit`

1. Asserta ausencia de labels y bundles de `gate_select` y `sealed_monitor`.
2. El coordinador materializa únicamente el oracle de `train` en un staging
   `FIT_PENDING`, mediante el ejecutable split-scoped, con rol
   `gate_fit` y construye su bundle desde logits ya congelados.
3. Verifica mínimos y disjunción con Olas 54–55.
4. Ajusta escalador y Ridge contextual `alpha=1.0` sólo sobre desacuerdos
   primarios con peso `1/d_t`.
5. Ajusta Ridge advantage-only `alpha=100.0` y cinco Ridges contextuales sobre
   targets barajados dentro de `(policy_index,d_t)`.
6. Preserva modelos, escaladores, features, target, pesos, mappings y
   diagnósticos; escribe `fit_freeze.json` de modo atómico.

No calcula thresholds ni consulta `val`/`lockbox` etiquetados.

### `select`

1. Verifica hash y contenido de `fit_freeze` y que `FIT_COMPLETE` sea íntegro.
2. Asserta ausencia de labels/bundle de `sealed_monitor`.
3. Materializa sólo el oracle de `val` en `SELECT_PENDING`, rol `gate_select`, y construye su
   bundle desde inferencia congelada.
4. Selecciona por separado el cuantil del contextual, advantage-only y cada
   shuffle; re-selecciona `gamma` del gate escalar con la grilla congelada.
5. El selector contextual usa constraints de accuracy y compatibilidad y
   minimiza regret; los scores iguales al threshold no reemplazan.
6. Repite únicamente la selección contextual en los dos shards deterministas y
   la sensibilidad all-in-catalog prevista.
7. Congela thresholds numéricos, brazos, curvas y estados en
   `selection_freeze.json` de modo atómico.

No evalúa ni inspecciona el monitor.

### `adjudicate`

1. Verifica hashes de `fit_freeze` y `selection_freeze`, ambos estados
   completos, y ausencia previa de
   labels/bundle de monitor.
2. Materializa por primera vez el oracle de `lockbox`, rol `sealed_monitor`, y
   construye su bundle desde inferencia congelada.
3. Aplica sin re-selección los siete brazos congelados:
   `hard_set_policy`, `pure_joint_full`, `scalar_advantage_gate`,
   `contextual_value_gate`, `advantage_only_value_gate`,
   `contextual_shuffled_gain` y `oracle_positive_gain`.
4. Calcula métricas, overrides, curvas diagnósticas, signos, shards, soporte
   ausente y cinco mil bootstraps pareados por `pair_token`.
5. Evalúa el patrón diagnóstico prospectivo del plan padre sin convertirlo en
   GO/NO-GO científico.
6. Escribe `analysis_core.json`, `result_arrays.npz`, reporte, runtime y
   manifest integral.

`result_arrays.npz` preserva, por rol y brazo cuando corresponda: tokens
ordenados, targets, cardinalidad/estrato, logits por seed y ensemble, posterior,
diseño, disagreements, pesos, gain realizado, scores, thresholds, acciones,
overrides y métricas token×política; además modelos/escaladores, mappings y
targets de los cinco shuffles, asignación de shards e índices bootstrap. El eje
`shuffle_id` preserva explícitamente para cada réplica sus scores, threshold,
acciones, overrides y métricas token×política; el promedio usado por el
bootstrap se guarda como un array distinto y derivable.

## Poblaciones y separaciones

La población primaria sigue siendo `NEAR_RIVAL` con cardinalidad verdadera
mayor o igual que dos; la cardinalidad define población pero no feature. Las
veinticuatro políticas de un token permanecen juntas. `gate_fit`,
`gate_select` y `sealed_monitor` deben ser disjuntos por `pair_token` y no
solaparse con los bundles de Olas 54–55.

El hecho de que el generador guarde verdad sellada para los tres splits no
autoriza su lectura. Antes de cada fase, el inventario físico de labels y
bundles posteriores debe ser vacío; la inferencia aislada nunca recibe sealed
truth. Los receipts distinguen generación, inferencia, materialización de
oracle y consumo analítico.

## Fallos y no-evaluabilidad

No hay redraw. Si un fallo ocurre después del escrow u oracle, se conserva el
estado `.pending` y la recuperación usa las mismas claves, el mismo material y
el mismo HEAD. Después de abrir cualquier label queda prohibido cambiar código,
config, plan, estimando, estimador, selector o bordes para esa realización. Un
bug que exija delta de código invalida la adjudicación de ese draw; sus
artefactos se preservan como fallo y una futura repetición requiere nuevo plan
y auditoría, no una corrección guiada por resultados abiertos.

Los mínimos prospectivos del plan padre se aplican por fase. Si faltan tokens,
filas de desacuerdo o clases requeridas, se promueve atómicamente el `.pending`
al estado terminal `FIT_NOT_EVALUABLE`, `SELECT_NOT_EVALUABLE` o
`MONITOR_NOT_EVALUABLE`, con conteos, hashes y motivo; no se cambia familia,
grilla, población ni semilla. Fit o select no evaluables prohíben abrir el split
siguiente. Monitor no evaluable cierra el paquete sin adjudicación.

Si sólo falla la fracción movible, la fase conserva su estado normal
`*_COMPLETE`: queda `NOT_EVALUABLE` únicamente el brazo shuffled y el patrón
prospectivo completo registra `diagnostic_condition_4=NOT_EVALUABLE`. Continúan
los demás brazos y contrastes, pero no se emite el booleano agregado del patrón
como si las seis condiciones hubieran sido observadas.

## Replay exacto

Replay usa las mismas claves, el mismo HEAD y la misma cronología física. La
matriz de comparación es:

- byte-exact: `fit_core.json`, `selection_core.json`, `analysis_core.json` y
  `feature_schema.json`;
- array-exact con dtype, shape y `equal_nan`: `fit_arrays.npz`,
  `selection_arrays.npz` y `result_arrays.npz`;
- semántica exacta por hash de contenido: visibles, logits raw y bundles;
- excluidos del núcleo exacto: paths absolutos, timestamps, duración,
  `runtime.json`, receipts operativos y el manifest contenedor.

Los índices bootstrap se generan una sola vez en monitor mediante
`np.random.Generator(np.random.PCG64(5607))` sobre tokens primarios ordenados
lexicográficamente y se guardan en `result_arrays.npz`.

## Artefactos previstos

- `experiments/geometria_proporcional/configs/wave56_contextual_gate_fresh.json`
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`
- `experiments/geometria_proporcional/_wave56_infer_worker.py`
- `experiments/geometria_proporcional/_wave56_oracle_materializer.py`
- `experiments/geometria_proporcional/_wave56_phase_worker.py`
- `experiments/geometria_proporcional/run_wave56_contextual_gate.py`
- `tests/test_wave56_prospective.py`
- `data/geometria_proporcional/wave56_contextual_gate_fresh_v1/`
- `data/geometria_proporcional/wave56_contextual_gate_fresh_v1_replay/`

## Verificación previa al draw

1. Tests unitarios de cada transición y de ausencia de artefactos futuros.
2. Integración sintética `prepare→fit→select→adjudicate` con directorios
   temporales y oracle pequeño.
3. Replay sintético exacto y prueba de no-redraw/recovery.
4. Re-forward histórico exacto de checkpoints antes de aceptar preparación.
5. Auditoría independiente de plan, config, código y tests ya versionados.

Sólo un `PASS` independiente sin findings materiales habilita el draw fresco.
La ejecución continúa siendo un experimento prospectivo de la misma ley; no
prueba transferencia de aparato, utilidad natural, geometría física ni PPU.

## Resolución de la auditoría R321

R321 emitió `REVISE` con tres findings altos y tres medios. Esta revisión: (1)
reemplaza la ausencia nominal de labels por una frontera ejecutable con sealed
root-only, materializador split-scoped y workers analíticos sin privilegios;
(2) agrega escrow atómico completo y freeze durable antes de invocar el
generador; (3) congela una máquina de estados transaccional y prohíbe cambiar el
ejecutable después de abrir labels; (4) fija RNG, orden y matriz de replay; (5)
limita `NOT_EVALUABLE` por shuffle al control y a su condición diagnóstica; y
(6) enumera los siete brazos y el schema mínimo de arrays. Requiere reauditoría
independiente focal antes de implementar.

R322 mantuvo dos findings altos y dos medios. Esta segunda revisión: (1) hace
durables escrow y freeze mediante `fsync` de archivo y directorio, publicación
atómica y verificación por reapertura antes del generador; (2) alinea el plan
padre con la prohibición de deltas post-label; (3) declara `disagreement` como
máscara y no como una decimoctava feature; y (4) fija un eje `shuffle_id` con
scores, thresholds, acciones, overrides y métricas de las cinco réplicas además
de su agregado. Requiere una reauditoría focal final antes de implementar.

R323 confirmó esas cinco resoluciones y mantuvo un finding alto: los mínimos
podían producir `NOT_EVALUABLE`, pero la máquina física no tenía estados
terminales para representarlo. Esta revisión agrega finales distintos para
fit, select y monitor, impide abrir el split siguiente cuando corresponde y
separa esos cierres del `NOT_EVALUABLE` exclusivo del brazo shuffled. Requiere
una última reauditoría focal del plan.

R324 verificó los tres estados terminales por mínimos, la continuidad normal
cuando sólo falla el shuffled y la ausencia de regresiones en los contratos de
R321–R323. Emitió `PASS` sin findings materiales. El plan queda habilitado para
implementación; ningún draw fresco queda autorizado hasta que config, código y
tests versionados reciban otra auditoría independiente.
