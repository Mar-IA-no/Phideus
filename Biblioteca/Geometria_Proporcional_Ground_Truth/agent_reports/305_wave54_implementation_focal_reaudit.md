# Ola 54 — reauditoría focal de implementación (R305)

**Dictamen: `REVISE`.** Revisión sin editar código ni ejecutar la corrida analítica. Se leyó el plan vigente y R304, se inspeccionaron primitives, preparador, runner, config y tests, y se ejecutó `venv/bin/python -m pytest -q tests/test_wave54_joint_set.py`: **12 passed**. No se emite GO/NO-GO.

## Resoluciones de R304 verificadas

- **Cronología fit/select/monitor, incluido hashing: resuelta en código.** El runner escribe `analysis_freeze.json` con el *digest esperado* del monitor tomado de la config, pero no hashea ni abre `sealed_monitor_bundle.npz` hasta después de materializar `selection_freeze.json`; el hash y la deserialización del monitor ocurren en ese orden posterior ([runner:441-466](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L441), [runner:515-532](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L515)).
- **Disjunción por `pair_token`: resuelta.** El preparador rechaza duplicados y toda intersección entre `val_threshold` y `val_monitor`; el runner repite la comprobación al reabrir ambos bundles ([preparador:114-125](../../../experiments/geometria_proporcional/prepare_wave54_inputs.py#L114), [runner:168-179](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L168)).
- **Sensibilidad del selector: resuelta para los ocho contrastes del patrón en monitor.** Se evalúan NLL contra `best_independent`, L1 de cardinalidad, NLL contra unary+cardinality, regret, accuracy, compatible y ambos controles de accuracy, y cualquiera que invierta signo activa `selector_sensitive` ([runner:616-674](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L616)).
- **Artefactos y replay: resueltos en lo sustantivo.** La grilla completa de parámetros se conserva en `selection_state.npz`; `posterior_state.npz` incluye riesgos, mínimo riesgo, márgenes, scores y máscaras de cobertura; `metrics_state.npz` conserva métricas unitarias; y `package_manifest.json` registra Python, plataforma y paquetes ([runner:737-857](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L737)).

## Findings que requieren corrección

### F305-01 — Bloqueante de ejecución: los cinco insumos obligatorios no están trackeados ni el worktree está limpio

**Observación.** El mecanismo nuevo sí falla cerrado por fuente individual: `require_sources_at_head()` exige pertenencia a `HEAD` y ausencia de cambios ([preparador:58-81](../../../experiments/geometria_proporcional/prepare_wave54_inputs.py#L58), [runner:128-151](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L128)). Pero, al momento de esta reauditoría, fallan `git ls-files --error-unmatch` para preparador, runner, primitive, test y config; esta última sigue ignorada por `*.json`. Además, el plan está modificado y los cuatro `.py` permanecen no trackeados, por lo que ambos entrypoints abortarían antes de preparar o analizar.

**Impacto.** La protección está bien encaminada, pero el pre-requisito explícito del plan —fuentes, tests y config versionados, hashes de bundle incorporados y worktree limpio antes de ejecutar— no está satisfecho ([plan:30-34](../waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md#L30)). No hay base para correr ni para tratar el estado actual como un protocolo congelado.

**Corrección mínima.** Añadir explícitamente la excepción de la config al control de versiones, versionar los cinco archivos junto con el plan y, tras producir los bundles, incorporar sus tres hashes al binding, volver a versionar y recién ejecutar desde un worktree limpio. No sustituir esa secuencia por una relajación del guard.

### F305-02 — Alta operativa: `--force` del runner puede desplazar un árbol de fuentes propio

**Observación.** El preparador rechaza una salida que contenga una fuente de ejecución ([preparador:230-233](../../../experiments/geometria_proporcional/prepare_wave54_inputs.py#L230)), pero el runner sólo rechaza solapamiento con inputs, referencia, raíz del repo y ancestros ([runner:406-415](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L406)). En un worktree limpio, `--output-dir $REPO/src --force` no se solapa con config, bundles ni fuentes de datos, pasa esos guards y llega a `output.rename(...)` ([runner:154-165](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L154), [runner:431](../../../experiments/geometria_proporcional/run_wave54_joint_set.py#L431)); así desplaza `src/`, que contiene `PRIMITIVES_PATH`, y recrea un `src/` vacío para los resultados.

**Impacto.** Aunque el rename conserva un archivo archivado y no usa `rmtree`, un argumento equivocado puede dejar el árbol de fuentes fuera de su ruta canónica e inutilizar el worktree, contradiciendo la seguridad operativa exigida para `--force`.

**Corrección mínima.** Antes de `prepare_output_directory`, aplicar al runner la misma comprobación del preparador contra cada `execution_source` (rechazar tanto una fuente como cualquier directorio que la contenga), y cubrir `--output-dir $REPO/src --force` con una prueba negativa que confirme que no se renombra nada.

## Alcance de la suite

Los 12 tests cubren las primitives y verifican directamente la disjunción, el archivo por `--force`, la selección por empate y los rechazos de lockbox. No ejercen `main()` de ambos entrypoints con un entorno sintético completo; por eso no demuestran end-to-end la cronología de accesos ni los guards de ruta. Dado F305-01 y F305-02, esa falta de integración queda como deuda de verificación a cerrar junto con las correcciones, no como una lectura de resultados experimentales.

## Conclusión

Las resoluciones metodológicas de R304 sobre sellado, separación, sensibilidad y preservación están implementadas en el código inspeccionado. El dictamen sigue `REVISE` únicamente por el estado de versionado obligatorio y por la ruta `--force` todavía peligrosa del runner; no se evaluó ni se decide la promoción científica, arquitectura o GO/NO-GO.
