# Ola 54 — auditoría independiente de implementación

**Alcance.** Revisión sin edición de `WAVE_54_JOINT_SET_POSTERIOR_PLAN.md`, primitives, preparador, runner, config y tests no commiteados. Se ejecutó `venv/bin/python -m pytest -q tests/test_wave54_joint_set.py`: **9 passed**. No se ejecutó una corrida analítica ni se emite GO/NO-GO.

## Findings

### Bloqueante — el preflight permite ejecutar código y configuración no versionados

**Observación.** Ambos entrypoints definen worktree “clean” mediante `git status --porcelain --untracked-files=no`, por lo que excluyen precisamente los archivos nuevos de la implementación. En el snapshot auditado `git ls-files --error-unmatch` falla para los tres `.py` y el test de Ola 54; además, la config no está trackeada y `*.json` la ignora. Los hashes de fuentes que el runner escribe son sólo recibos del contenido que acaba de leer, no expectativas congeladas ni una comprobación de pertenencia a `HEAD`.

**Impacto.** Después de commitear el plan, el runner puede producir un resultado aparentemente congelado con primitives, runner, preparador o config no versionados y modificables sin que el guard lo detecte. Eso contradice la cronología exigida por el plan y deja sin base el replay reproducible.

**Evidencia.** [plan:30-34](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L30), [preparador:81-90](../../../experiments/geometria_proporcional/prepare_wave54_inputs.py#L81), [runner:126-135](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L126), [runner:379-400](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L379), [config:1-49](../../../experiments/geometria_proporcional/configs/wave54_joint_set.json#L1), [.gitignore:26](../../../.gitignore#L26).

### Alta — el runner lee íntegramente el bundle sellado antes del freeze de selección

**Observación.** `require_hash()` abre y recorre el archivo completo para hashearlo. El runner lo invoca sobre `sealed_monitor_bundle.npz` antes de cargar el bundle fit/select, ajustar los modelos y escribir `selection_freeze.json`. Sin embargo, el freeze declara `sealed_monitor_accessed: false`, y el receipt posterior presenta fit y monitor como si fueran los dos únicos accesos ordenados.

**Impacto.** Aunque el código actual no deserializa targets durante ese hash, la cronología declarada (“recién entonces abre el monitor”) y su evidencia de acceso son falsas en sentido estricto. Se pierde la garantía auditable de que ningún contenido sellado estuvo disponible antes de congelar selector, baseline y parámetros.

**Evidencia.** [plan:47-51](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L47), [runner:86-91](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L86), [runner:359-371](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L359), [runner:451-464](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L451), [runner:686-695](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L686).

### Alta — la barrera fit/select–monitor no verifica disjunción de `pair_token`

**Observación.** El preparador sólo prueba que `calibration_fit` y `decision_select` formen una partición del archivo `val_threshold`. Nunca comprueba que esos tokens sean disjuntos de `val_monitor`; el runner tampoco valida esa propiedad al reabrir los bundles. Tener dos `.npz` distintos no impide que una misma unidad analítica aparezca a ambos lados.

**Impacto.** Un solapamiento upstream silencioso convierte monitor en información ya vista por el ajuste o la selección, invalida los bootstrap por `pair_token` y hace que la separación física afirmada no alcance como control de leakage.

**Evidencia.** [plan:24-28](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L24), [plan:36-51](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L36), [preparador:211-229](../../../experiments/geometria_proporcional/prepare_wave54_inputs.py#L211), [runner:402-405](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L402), [runner:464-518](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L464).

### Alta — el guard de sensibilidad del selector cubre sólo tres de los contrastes primarios

**Observación.** El plan invalida una conclusión fuerte si la selección global en lugar de la primaria cambia el signo de *un contraste primario*. El runner sólo inspecciona NLL contra `best_independent`, accuracy contra duro y regret contra duro. Omite, entre otros, NLL contra `joint_unary_cardinality`, L1 de cardinalidad, tasa compatible y los dos controles de accuracy.

**Impacto.** Puede quedar `selector_not_sign_sensitive=true` y, por ende, pasar el patrón conjunto aunque el lambda global invierta el signo de un contraste que participa en sus criterios. La sensibilidad queda submedida exactamente donde pretende limitar la interpretación.

**Evidencia.** [plan:108-113](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L108), [plan:140-158](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L140), [runner:539-559](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L539), [runner:598-618](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L598).

### Alta operativa — `--force` permite borrado amplio antes de validar entradas, versionado o binding

**Observación.** El preparador acepta cualquier `--output-dir` que no solape los tres directorios externos; no prueba solapamiento con el repo, config o archivos de ejecución, y luego aplica `shutil.rmtree(output)`. Así, con fuentes externas, un `--output-dir` igual a la raíz del repo pasa esa comprobación y puede borrar el worktree. El runner sí cubre más solapamientos, pero borra el output existente antes de comprobar config, hashes de bundles y fuentes.

**Impacto.** Un argumento equivocado o un binding inválido puede destruir un bundle o resultado reutilizable y abortar después, sin producir el reemplazo. Es un riesgo de ejecución independiente de la validez científica.

**Evidencia.** [preparador:163-178](../../../experiments/geometria_proporcional/prepare_wave54_inputs.py#L163), [preparador:180-209](../../../experiments/geometria_proporcional/prepare_wave54_inputs.py#L180), [runner:340-371](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L340).

### Media — los artefactos no preservan todo lo que el plan exige para reanálisis y replay

**Observación.** `fit_diagnostics.json` preserva filas de grilla, pero `selection_state.npz` guarda parámetros sólo de los lambdas seleccionados, no de cada estructura×lambda. `posterior_state.npz` guarda masas y acciones, pero no `action_risk`, `minimum_risk`, márgenes/scores selectivos ni máscaras de cobertura; sólo se emiten resúmenes agregados. El package manifest tampoco registra runtime de Python/plataforma, pese a que el plan lo exige.

**Impacto.** No puede auditarse la grilla completa, la abstención por token ni recrearse ciertos diagnósticos sin volver a ajustar o recomputar. Esto reduce trazabilidad y debilita el requisito de replay exacto, aun si los cuatro NPZ comparados coinciden.

**Evidencia.** [plan:178-190](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L178), [runner:620-656](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L620), [runner:686-704](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L686), [runner:706-730](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L706).

## Aspectos verificados sin finding

- La reparametrización de interacciones implementa seis coeficientes con suma cero a partir de cinco contrastes; la referencia reproduce el posterior Bernoulli independiente condicionado a no vacío. [primitives:34-63](../../../src/geometria_proporcional/wave54_joint_set.py#L34), [tests:41-46](../../../tests/test_wave54_joint_set.py#L41).
- La NLL regularizada y su gradiente analítico son consistentes con la prueba de diferencias finitas, y el ajuste falla en vez de usar un fallback si L-BFGS-B no converge. [primitives:87-148](../../../src/geometria_proporcional/wave54_joint_set.py#L87), [tests:49-86](../../../tests/test_wave54_joint_set.py#L49).
- `best_independent` se selecciona en `decision_select` sobre la población primaria y con empate hacia raw, como agregó el plan. [runner:417-426](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L417), [plan:117-119](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L117).

## Riesgo residual

Los tests existentes son unitarios y no ejercen `main()` con bundles reales o sintéticos: no prueban la cronología, disjunción entre bundles, guards de versionado, destrucción por `--force`, artefactos ni replay. Deben resolverse los findings de validez y añadir pruebas de integración negativas antes de interpretar una corrida como evidencia de la Ola 54.
