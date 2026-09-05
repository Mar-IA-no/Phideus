# Ola 58 — reauditoría independiente pre-ejecución de implementación

## Dictamen: PASS

La implementación corregida resuelve los findings de R412 y materializa el plan aceptado con separación física y cronológica FIT/SELECT/MONITOR, replay `LEGACY-W57` previo a referencias, roster completo ante `NOT_EVALUABLE`, staging sanitizado, artefactos suficientes y replay Wave 58 exacto desde vacío. La suite específica completa pasó `22/22`, incluida una primaria y un replay físicos ejecutados como `nobody`.

El estado `implementation_commit="TO_BE_FROZEN"` y `source_sha256={}` continúa siendo aceptable únicamente como secuencia prefreeze. No habilita todavía la corrida canónica: commit y hashes deben fijarse después de este PASS y el preflight ya está preparado para rechazarlos si faltan o divergen.

## Superficies auditadas

- Plan aceptado: `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md`, SHA-256 `d7fbbad633a3f03d37de46ad505dbcd330ac9ffa5544838ea2c4707a3ac6ba69`.
- Auditoría previa R412: `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/412_wave58_open_model_class_implementation_audit.md`, SHA-256 `a834f7844482eafde736f75b37b2c21cb8280b107ef59554d7b690f81667f8ce`.
- Primitivas Wave 58: `src/geometria_proporcional/wave58_open_diagnostic.py`, SHA-256 `5f795e18c492c439e7f43fe33714f07a40cb47cf907323414bd99e5ec71166e7`.
- Runner/coordinador: `experiments/geometria_proporcional/run_wave58_open_diagnostic.py`, SHA-256 `c1f80c32e84ff5440123884f2caf4787a282380093fa4074da201e4d3f64527b`.
- Config prefreeze: `experiments/geometria_proporcional/configs/wave58_open_model_class_diagnostic.json`, SHA-256 `51a8e7789d7a0af221f522e3d3abd25123d668ae8bb3fe967909338adfbe53c7`.
- Tests: `tests/test_wave58_open_diagnostic.py`, SHA-256 `24e5e1bfaeb85263dd456879b0630685b159efdf1a2d22c2cbf8a34c7e66736b`.

## Cierre de findings R412

### 1. Binding de commit y fuentes — resuelto, pendiente sólo el acto mecánico de freeze

`validate_config` rechaza el placeholder, comprueba que el commit exista mediante `git cat-file`, exige que sea ancestro de `HEAD`, verifica que el inventario de hashes coincida exactamente con los nueve `source_paths` y recalcula cada SHA (`experiments/geometria_proporcional/run_wave58_open_diagnostic.py:192-217`). Una simulación read-only con el `HEAD` actual `94cc8a59d8fe053b81684dfe4d9783c343c9d7dd` y hashes calculados para las nueve fuentes pasó el preflight completo.

La config vigente conserva deliberadamente el placeholder (`experiments/geometria_proporcional/configs/wave58_open_model_class_diagnostic.json:37-38`). Antes de ejecutar se deben escribir el commit y los nueve hashes definitivos; cualquier edición posterior del código o tests obliga a recalcularlos y volver a validar.

### 2. Cronología y verificación de freezes — resuelto

Los validadores recomputan hashes de estados/scores FIT, grillas/arrays SELECT y la ligadura del freeze SELECT al freeze FIT (`run_wave58_open_diagnostic.py:122-150`). SELECT valida FIT antes de abrir validation (`run_wave58_open_diagnostic.py:455-461`). MONITOR valida FIT+SELECT antes de abrir validation o monitor y difiere expresamente la apertura del monitor hasta esa comprobación (`run_wave58_open_diagnostic.py:777-803`). El coordinador repite esas validaciones entre invocaciones, antes de stagear la fase futura (`run_wave58_open_diagnostic.py:1067-1100`).

### 3. Replay exacto `LEGACY-W57` — resuelto

La grilla histórica exacta quedó separada como `Q_GUARD_WAVE57=(0.1,0.2,0.3,0.4,0.5,0.6,0.8)` (`src/geometria_proporcional/wave58_open_diagnostic.py:26-29`) y la config la liga explícitamente (`wave58_open_model_class_diagnostic.json:44-45`). `LEGACY-W57` se selecciona en SELECT exclusivamente sobre validation con esa grilla y se incorpora al freeze antes del monitor (`run_wave58_open_diagnostic.py:455-505`).

En MONITOR se restauran primero los thresholds congelados y se materializan las consecuencias de validation y monitor (`run_wave58_open_diagnostic.py:523-552`). Sólo después se abren `legacy_selection.npz` y `legacy_results.npz` para comparar estados, scores, thresholds, máscaras, acciones y métricas (`run_wave58_open_diagnostic.py:553-614`). El test con los artefactos reales confirma igualdad total de las comprobaciones implementadas (`tests/test_wave58_open_diagnostic.py:492-525`) y la corrida física integral vuelve a exigirla (`tests/test_wave58_open_diagnostic.py:539-566`).

### 4. Transporte HGB y autoridad científica — distinción aceptable y explícita

Se eliminó el import y todo uso de `sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor`; la búsqueda estática tampoco encontró `TreePredictor` ni el módulo privado del scorer. El transporte entre invocaciones físicas conserva arrays tipados de nodos y restringe el scorer portable a splits numéricos, rechazando cualquier split categórico (`src/geometria_proporcional/wave58_open_diagnostic.py:266-328`). El estado declara literalmente `transport_only=true` y `score_authority="preserved_float64_scores_per_split"` (`wave58_open_diagnostic.py:280-290`).

La extracción del transporte aún depende del layout fijado `_predictors`/`_baseline_prediction`, pero no se presenta como estado científico reconstruible ni como contrato cross-version: sklearn está fijado exactamente en `1.8.0`, el scorer se contrasta contra el modelo ajustado, los scores `float64` de train/validation/monitor se preservan en sus artefactos y el replay completo vuelve a ajustar HGB desde cero. Bajo esta frontera explícita, los nodos son un mecanismo operativo necesario para cruzar fases sin pickle; la autoridad de reanálisis son los scores preservados. Una deriva de versión, fuente o resultado queda interceptada por runtime, hashes y replay exacto.

### 5. Conservación del roster `NOT_EVALUABLE` — resuelto

Ridge, Logistic y HGB detectan arrays de fit inválidos/no finitos; Logistic y HGB conservan además la ruta de una sola clase (`wave58_open_diagnostic.py:153-190,205-263,331-379`). Los selectores detectan scores no finitos antes de cuantilar y producen una selección explícita `NOT_EVALUABLE` con razón y terminal hard no factible, sin abortar el roster (`wave58_open_diagnostic.py:449-509,638-659`). Estados y razones se propagan a summaries, analysis y nomination, que excluye sólo de elegibilidad sin podar IDs (`run_wave58_open_diagnostic.py:417-452,514-520,629-646,841-883`). El test verifica que siguen existiendo los 36 candidatos y que los afectados conservan estado (`tests/test_wave58_open_diagnostic.py:342-370`).

### 6. Config de worker sin paths originales — resuelto

`worker_config` construye una vista de fase por allowlist y excluye `inputs`, `source_paths`, outputs y demás paths coordinadores (`run_wave58_open_diagnostic.py:99-119`). El coordinador crea tres JSON sanitizados y entrega el correspondiente a cada worker (`run_wave58_open_diagnostic.py:1061-1099`). El test inspecciona las tres vistas y rechaza cualquier path de `data/geometria_proporcional` (`tests/test_wave58_open_diagnostic.py:429-437`). Los bundles continúan copiándose `0444`, sus directorios `0555` y la ejecución ocurre como UID/GID `65534`.

### 7. Artefactos y reporte — resuelto

`scores_and_masks.npz` parte ahora de todos los arrays `validation__*` congelados por SELECT y agrega el conjunto paralelo de monitor, targets, thresholds, acciones, propuestas, autorizaciones, métricas y bootstrap (`run_wave58_open_diagnostic.py:800-840,841-932`). `REPORT.md` enumera cada uno de los 36 candidatos canónicos y 24 probes históricos con estado, thresholds, shards, métricas, soportes y deltas; remite a los arrays y grillas para las matrices completas (`run_wave58_open_diagnostic.py:680-774`). La integración física comprueba presencia de todos los IDs y de máscaras/targets de ambos splits (`tests/test_wave58_open_diagnostic.py:554-564`).

### 8. Cobertura conductual y corrida física — resuelto

La suite amplió el fixture `P1-HCT/Qg8³` a un empate activo y verifica la elección resultante (`tests/test_wave58_open_diagnostic.py:231-254`), ejercita JOINT y shard-robust activos (`tests/test_wave58_open_diagnostic.py:257-277`), pesos idénticos en los doce fits (`tests/test_wave58_open_diagnostic.py:373-410`), config sanitizada, precedencia de freezes y replay legacy real. El último test ejecuta desde directorios vacíos una primaria y un replay completos mediante los tres workers físicos `nobody`, exige igualdad de los diez artefactos científicos y revalida ambos freezes (`tests/test_wave58_open_diagnostic.py:535-566`).

Resultado observado:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  venv/bin/python -m pytest -q tests/test_wave58_open_diagnostic.py
......................                                                   [100%]
22 passed in 109.57s (0:01:49)
```

### 9. Corrección adicional P3 — conforme

Los seis probes P3 usan ahora grillas heterogéneas `Qg8 × Qg9`, no `Qg9²`: harm recibe `Q_GUARD_8` y compatibility/posterior-incompatibility/accuracy recibe `Q_GUARD_9` (`run_wave58_open_diagnostic.py:366-371`). `select_candidate` acepta una grilla por guard y enumera su producto cartesiano completo (`wave58_open_diagnostic.py:475-510`). El test específico fija esa relación (`tests/test_wave58_open_diagnostic.py:319-340`).

## Invariantes restantes verificadas

- Los seis targets, contratos sklearn, seeds, 36 candidatos canónicos y 24 probes históricos permanecen alineados con el plan.
- Los tie-breaks, terminal hard único, conjunción k-aria, thresholds estrictos, producto completo y bootstrap compartido por split no regresaron.
- `CUDA_VISIBLE_DEVICES=''` es obligatorio; no hay imports Torch/CuPy/CUDA y el worker comprueba que los threadpools efectivos no excedan cuatro hilos (`run_wave58_open_diagnostic.py:935-980,1013-1026`).
- La salida se construye fuera del path final, verifica hashes originales pre/post, compara replay científico y se publica por `os.replace`; ante error se elimina sólo el staging (`run_wave58_open_diagnostic.py:1049-1148`).
- `git diff --check` no encontró errores en implementación, runner, config ni tests.

## Condición operativa posterior al PASS

Este PASS acepta la implementación prefreeze; no reemplaza el freeze. El próximo responsable debe fijar `implementation_commit` y los nueve `source_sha256`, ejecutar `validate_config(..., require_frozen_sources=True)` y abstenerse de editar esas superficies después. Sólo entonces queda habilitada la corrida primaria canónica seguida por el replay desde vacío.
