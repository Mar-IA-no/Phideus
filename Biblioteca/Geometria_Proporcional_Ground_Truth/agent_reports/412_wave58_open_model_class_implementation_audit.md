# Ola 58 — auditoría independiente pre-ejecución de implementación

## Dictamen: REVISE

La implementación contiene una base funcional amplia y sus `16` tests específicos pasan en CPU, pero todavía no puede ejecutarse bajo el contrato aceptado. Hay un bloqueo mecánico de configuración y varios desvíos metodológicos en las fronteras FIT/SELECT/MONITOR, el replay `LEGACY-W57`, la autoridad HGB y el tratamiento de candidatos no evaluables. Los tests actuales no ejercen esas fronteras con suficiente fuerza.

## Alcance y evidencia ejecutada

Se leyeron completos:

- `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md`, SHA-256 `d7fbbad633a3f03d37de46ad505dbcd330ac9ffa5544838ea2c4707a3ac6ba69`;
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/411_wave58_open_model_class_diagnostic_plan_final_audit.md`, SHA-256 `b065619e8d46eeb06d800f9e4ffb8025ff22d3e210a9047e53b3f13181a35436`;
- `src/geometria_proporcional/wave58_open_diagnostic.py`, SHA-256 `5e53ce023653a3825dfe702b87d745468c0a95b0cd8f7f94d7c4c42c68ddd1b9`;
- `experiments/geometria_proporcional/run_wave58_open_diagnostic.py`, SHA-256 `f8fb2d1e815232b885413cdccbffe29f6cde8e178b4af20cbc30a9729b5cb559`;
- `experiments/geometria_proporcional/configs/wave58_open_model_class_diagnostic.json`, SHA-256 `5d69c0fbb96844f502e5da0bca70930ad29a7b4729fdeadb0c7a622c09c62d51`;
- `tests/test_wave58_open_diagnostic.py`, SHA-256 `bcfa965aa32fee8a371d81b048bfd37177e18a4550435403dfa70144ac3ba733`.

Comandos de verificación:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  venv/bin/python -m pytest -q tests/test_wave58_open_diagnostic.py
................                                                         [100%]
16 passed in 3.12s
```

El preflight real con `require_frozen_sources=True` termina actualmente en:

```text
RuntimeError: implementation commit is not frozen
```

## Findings priorizados

### P0 — la configuración ejecutable no está congelada

La config conserva `implementation_commit: "TO_BE_FROZEN"` y `source_sha256: {}` (`experiments/geometria_proporcional/configs/wave58_open_model_class_diagnostic.json:37-38`). El coordinador exige ambos antes de correr (`experiments/geometria_proporcional/run_wave58_open_diagnostic.py:135-143`), por lo que la ejecución canónica aborta hoy antes de FIT. Además, cuando el placeholder se reemplace, el código sólo comprueba que ya no sea esa cadena; no coteja `implementation_commit` con el `HEAD` real (`run_wave58_open_diagnostic.py:135-143`). El commit observado durante esta auditoría fue `94cc8a59d8fe053b81684dfe4d9783c343c9d7dd`.

Corrección requerida: después de resolver los restantes findings, congelar todos los hashes del inventario, fijar el commit y hacer que el preflight contraste también ese commit con el checkout ejecutado.

### P1 — FIT/SELECT/MONITOR existen físicamente, pero los freezes no gobiernan el acceso

El plan exige que SELECT reciba validation después del freeze FIT y que MONITOR reciba el monitor sólo después de verificar ambos freezes (`WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:359-363`). En cambio:

- SELECT abre `validation_bundle.npz` antes de cualquier comprobación y nunca lee ni valida `fit_freeze.json` (`run_wave58_open_diagnostic.py:379-385`);
- MONITOR abre validation y monitor, reconstruye scores de ambos y sólo después lee `selection_freeze.json`; nunca valida los hashes declarados por FIT o SELECT (`run_wave58_open_diagnostic.py:611-630`);
- el coordinador se limita a pasar los directorios terminados, sin verificar sus freezes antes de incorporar el input futuro (`run_wave58_open_diagnostic.py:873-905`).

Los allowlists de nombres sí están presentes (`run_wave58_open_diagnostic.py:754-777`), pero no sustituyen la verificación criptográfica y cronológica requerida. Debe validarse el freeze anterior —incluidos sus hashes contra los archivos staged— antes de abrir el bundle de la fase siguiente.

### P1 — `LEGACY-W57` no está aislado ni reproduce exactamente la grilla Wave 57

El contrato aceptado exige recalcular selección exclusivamente desde `gate_select`, congelarla y sólo entonces abrir monitor y los arrays legacy de referencia (`WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:227-235`). La implementación no produce `LEGACY-W57` en SELECT. Lo recomputa dentro de MONITOR, después de haber abierto y puntuado el monitor (`run_wave58_open_diagnostic.py:611-624,721-724`). Peor aún, `verify_legacy` carga `legacy_selection.npz` y `legacy_results.npz` antes de ejecutar `select_sequential` (`run_wave58_open_diagnostic.py:422-448`), mientras ambos archivos ya forman parte del staging MONITOR (`run_wave58_open_diagnostic.py:762-771`). La igualdad funcional observada no demuestra la separación predeclarada.

También hay una divergencia concreta de grilla. Wave 57 congeló `guard_acceptance_quantiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.8]` (`experiments/geometria_proporcional/configs/wave57_contextual_tail_guard_fresh.json:60-63`). El replay usa `Q_GUARD_8`, que añade `0.7` (`src/geometria_proporcional/wave58_open_diagnostic.py:28-29`; `run_wave58_open_diagnostic.py:448-452`). El threshold ganador coincide en los datos actuales y por eso el test pasa, pero el proceso no usa “exactamente los cuantiles” Wave 57 ni compara la grilla completa.

Corrección requerida: ejecutar y congelar el brazo legacy en una frontera previa al monitor y previa a los arrays de referencia; usar literalmente la config Wave 57, incluida su grilla de siete cuantiles; sólo después abrir referencias y monitor para verificar estados, scores, selección, máscaras, acciones, resúmenes y deltas.

### P1 — HGB usa precisamente la reconstrucción privada que el plan excluyó

El plan fija como autoridad los scores `float64`, exige refit independiente para el replay y dice expresamente que los tests no deben tratar el árbol privado de sklearn como estado reconstruible (`WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:210-218`). La implementación importa `TreePredictor` desde un módulo privado (`src/geometria_proporcional/wave58_open_diagnostic.py:15-17`), exporta `_predictors` y `_baseline_prediction` (`wave58_open_diagnostic.py:229-252`) y reconstruye inferencia desde esos nodos privados (`wave58_open_diagnostic.py:256-270`). El test HGB convierte esa conducta contraria al plan en requisito (`tests/test_wave58_open_diagnostic.py:154-170`).

Debe reemplazarse por el régimen acordado: scores por split como autoridad, metadata pública y refit HGB en el replay; no árboles privados serializados ni reconstrucción con API interna.

### P1 — `NOT_EVALUABLE` no se propaga al roster; aborta la selección

Los fits Logistic/HGB detectan una sola clase y devuelven estado `NOT_EVALUABLE` (`wave58_open_diagnostic.py:185-214,273-289`), pero `score_grid` transforma ese estado en una matriz de `NaN` (`wave58_open_diagnostic.py:307-323`). Los selectores intentan luego calcular thresholds sobre esos scores y `_threshold` lanza `ValueError` ante cualquier no finito (`wave58_open_diagnostic.py:367-405`). Por tanto, un modelo no evaluable no queda registrado como candidato conservado: detiene toda la corrida. Esto contradice `WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:203-208`.

El test actual sólo comprueba el retorno local del fit de una clase (`tests/test_wave58_open_diagnostic.py:173-183`); no prueba el recorrido completo hasta los 36 candidatos y 24 probes. Se requiere propagación explícita de estado/reason por candidato, sin poda y sin abortar candidatos independientes. La misma ruta debe cubrir no convergencia y estados/scores no finitos, actualmente no verificados de extremo a extremo.

### P1 — el worker recibe los paths originales dentro de `config.json`

El plan exige que el worker reciba copias read-only y no paths a los originales (`WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:348-352`). Sin embargo, cada worker recibe la config completa (`run_wave58_open_diagnostic.py:873-903`), cuyo bloque `inputs` contiene todos los paths originales, incluidos monitor y referencias legacy (`wave58_open_model_class_diagnostic.json:17-24`). Los originales observados son `0600 root:root`, de modo que `nobody` no puede leer esos NPZ directamente; eso reduce el riesgo práctico, pero no satisface el contrato de ausencia de paths ni su test obligatorio.

Debe stagearse una config sanitizada de fase, sin inventario de paths originales ni nombres de inputs futuros, manteniendo el inventario completo sólo del lado coordinador y en la salida final.

### P2 — preservación y reporte no coinciden por completo con el inventario prometido

`scores_and_masks.npz` recibe actions/proposals/authorized sólo para monitor (`run_wave58_open_diagnostic.py:671-719`); las máscaras de validation permanecen únicamente en `select/selection_arrays.npz`. Esto no coincide literalmente con el contrato de `scores_and_masks.npz` con scores, targets, thresholds y máscaras por brazo (`WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:325-333`). Asimismo, `analysis.json` conserva los 36+24 resultados, pero `REPORT.md` sólo informa sus conteos y detalla el candidato nominado (`run_wave58_open_diagnostic.py:563-608`), mientras el plan pide que el reporte conserve los resultados de todos los IDs y, para cada brazo, thresholds, soporte, deltas, validation, shards, monitor y overrides (`WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:307-318`).

No se perdió toda la evidencia —las grillas y arrays parciales existen en otros archivos—, pero el esquema prometido y el informe humano deben alinearse antes de congelar el corpus.

### P1 — la cobertura pasa, pero no prueba varios requisitos pre-ejecución explícitos

Los `16` tests son verdes, pero varias comprobaciones exigidas por `WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:365-388` son nominales o ausentes:

- el fixture `P1-HCT/Qg8³` sólo cuenta celdas y fuerza que gane `HARD_ONLY`; no construye ni verifica un empate activo bajo el orden total de guards (`tests/test_wave58_open_diagnostic.py:220-238`);
- JOINT y shard-robust también sólo comprueban tamaño, presencia de shards y terminal hard en un fixture de métricas idénticas (`tests/test_wave58_open_diagnostic.py:241-258`);
- el airlock se inspecciona mediante búsqueda de strings y sólo se ejecuta físicamente FIT; no hay test de rechazo y cronología de freezes para SELECT/MONITOR (`tests/test_wave58_open_diagnostic.py:300-335`);
- el supuesto replay Wave 58 “desde vacío” sólo compara dos directorios sintéticos ya poblados con el mismo contenido; no ejecuta dos corridas independientes (`tests/test_wave58_open_diagnostic.py:274-288`);
- el test legacy comprueba igualdad funcional, pero mediante referencias accesibles en el mismo proceso y sin auditar que se abran después del freeze (`tests/test_wave58_open_diagnostic.py:337-370`);
- no hay test end-to-end de no convergencia/no finitos, conservación del roster `NOT_EVALUABLE`, hashes pre/post, permisos `0444/0555`, ausencia de paths originales ni límite efectivo de threads en cada fase.

Debe añadirse cobertura conductual de esas fronteras, no sólo presencia textual de constantes.

## Aspectos conformes verificados

- Los hashes de plan, R411 e inputs están correctamente ligados y los hashes de entrada actuales coinciden (`run_wave58_open_diagnostic.py:111-134`; config `:9-24`).
- Los seis targets, su dominio y polaridad corresponden al plan (`wave58_open_diagnostic.py:115-144`).
- El roster canónico contiene 36 IDs únicos y el ledger histórico 24; las especificaciones P1–P4 están materializadas (`run_wave58_open_diagnostic.py:236-340`).
- La selección general enumera el producto k-ario completo, conserva duplicados, usa comparaciones estrictas y agrega un único terminal `HARD_ONLY` (`wave58_open_diagnostic.py:374-500`).
- El orden nominal de targets y los tie-breaks implementados para las rutas del roster son consistentes con el plan; el selector secuencial corta antes del guard cuando gana hard (`wave58_open_diagnostic.py:503-603`).
- La publicación final usa un directorio temporal en el mismo parent y `os.replace`; ante excepción elimina sólo el staging (`run_wave58_open_diagnostic.py:858-951`).
- Los workers exigen UID/GID `65534`, `CUDA_VISIBLE_DEVICES=''`, allowlist exacto y output vacío; las copias staged se fijan `0444/0555` (`run_wave58_open_diagnostic.py:749-837`).
- No hay imports CUDA/GPU; el runner fija las tres variables de threads y `threadpool_limits(limits=4)` (`run_wave58_open_diagnostic.py:783-835`).
- El comparador de replay exige igualdad byte a byte para JSON/Markdown y array por array para NPZ en los diez artefactos científicos (`run_wave58_open_diagnostic.py:840-855`).
- En los bundles reales, los `pair_token` primarios están ordenados lexicográficamente y son únicos: train `299`, validation `302`, monitor `306`; la matriz bootstrap se comparte por split en la implementación actual.

## Condición de reauditoría

No ejecutar Wave 58 todavía. Una reauditoría puede ser acotada a comprobar: config y commit realmente congelados; freezes verificados antes de cada acceso futuro; replay legacy aislado con la grilla exacta de siete cuantiles; eliminación de la reconstrucción HGB privada; propagación completa de `NOT_EVALUABLE`; config de worker sanitizada; artefactos/reporte reconciliados; y tests conductuales para esas correcciones.
