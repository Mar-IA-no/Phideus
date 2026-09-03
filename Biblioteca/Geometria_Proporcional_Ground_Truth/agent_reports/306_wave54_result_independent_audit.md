# R306 — auditoría independiente de resultados de la Ola 54

**Fecha:** 2026-09-03
**Alcance:** revisión independiente y sólo de lectura de `wave54_joint_set_v1` y su replay. No se modificaron código ni resultados.

## Veredicto de auditoría

**PASS — sin finding invalidante.** La ejecución, sus estados crudos, el replay y las superficies de resumen son coherentes con el plan congelado. Este PASS acredita trazabilidad y cálculo del resultado histórico; no promueve el posterior, no revalida los logits/encoder upstream y no constituye una decisión GO/NO-GO.

## Observaciones verificadas

### Cronología, población y selección

- El commit de ambos artefactos coincide con el `HEAD` auditado: `1c92ae1dc01d6c5f339c814ee04d94fb5a9a4050`. Los hashes del bundle fit/select, del monitor sellado y del manifest coinciden con `frozen_config.json`: `2c30…aae3`, `e28d…9388` y `5acf…4208`, respectivamente.
- La separación es efectiva: `384` tokens fit/select (`192` calibration, `192` decision), `384` monitor y `0` `pair_token` compartidos. La población primaria contiene `75` tokens en `decision_select` y `148` en monitor; el bootstrap guarda exactamente índices `(2000, 148)` con seed `5407`.
- `best_independent` fue congelado correctamente como `independent_platt`: NLL primaria en selección `1.0941164747`, frente a `1.1098979966` de raw. El runner calcula esa elección antes de abrir monitor (`experiments/geometria_proporcional/run_wave54_joint_set.py:499`); y aplica el desempate hacia raw si hubiera igualdad (`experiments/geometria_proporcional/run_wave54_joint_set.py:507`).
- La selección de regularización reproduce literalmente el mínimo de NLL primario y el desempate hacia mayor `lambda` (`experiments/geometria_proporcional/run_wave54_joint_set.py:298`). Los pares primary/global son: `joint_full` `1e-3/1e-4`, `joint_unary` `1e-4/10`, y `joint_unary_cardinality` `1e-1/1e-2`. Ninguno de los ocho contrastes de sensibilidad cambia de signo en monitor; por ello `selector_sensitive=false` está correctamente calculado (`experiments/geometria_proporcional/run_wave54_joint_set.py:633`).

### Posterior, controles y masa fuera de soporte

- La implementación usa los 12 parámetros identificados previstos: cuatro slopes, tres sesgos de cardinalidad y cinco contrastes que inducen seis interacciones de suma cero (`src/geometria_proporcional/wave54_joint_set.py:14`). Los ajustes seleccionados son finitos, con norma de gradiente máxima relevante `5.2531e-7`; las masas son finitas y sus sumas por token quedan en `[0.9999999999999983, 1.0000000000000018]`.
- Se preservan los nueve brazos del plan, incluidos `hard_set_policy`, raw/Platt, las tres estructuras conjuntas, prior empírico, control target-shuffled y oracle. El control shuffled usa exactamente la `lambda` primaria de `joint_full`, tal como implementa el ajuste (`experiments/geometria_proporcional/run_wave54_joint_set.py:376`).
- `calibration_fit` observa `10/15` conjuntos; los cinco ausentes son índices `[0,4,8,10,12]`. En `decision_select`, la masa no observada es `0.0252265` para `joint_full` frente a `0.0667006` para raw; la diferencia es `-0.0414741`, por debajo de la tolerancia configurada. El diagnóstico es por tanto correctamente positivo, sin convertir las clases ausentes en evidencia positiva.

### Contrastes, IC y los 11 checks

Las convenciones de signo son consistentes: los contrastes guardan `joint_full − comparador` (`experiments/geometria_proporcional/run_wave54_joint_set.py:305`); por tanto, un delta negativo de NLL/regret favorece al posterior conjunto y uno positivo de accuracy/compatibilidad lo favorece.

| Check congelado | Resultado auditado | Estado |
|---|---:|:---:|
| NLL vs `best_independent` | `-0.081790`, IC95 `[-0.132251, -0.036885]` | true |
| L1 de cardinalidad vs mejor independiente | reducción `0.087106` | true |
| NLL vs unary+cardinality | `-0.048799`, IC95 `[-0.092191, -0.011775]` | true |
| Regret vs hard | `-0.013021`, IC95 `[-0.037777, +0.008405]`; no alcanza `-0.02` | false |
| Accuracy vs hard | `-0.057714`; menor que el límite `-0.01` | false |
| Compatibilidad vs hard | `+0.036318` | true |
| Accuracy vs prior | `+0.140484` | true |
| Accuracy vs shuffled | `+0.038570`; no alcanza `+0.05` | false |
| Masa unseen en selección | diferencia joint−raw `-0.041474` | true |
| Selector sin cambio de signo | `false` en los ocho indicadores de cambio | true |
| Replay externo exacto | `true` en la corrida replay | true |

La tabla coincide con `summary.json`, `metrics_state.npz`, `posterior_state.npz` y los índices bootstrap. En particular, los tres `false` son literales: no hay error de signo ni de intervalo que permita reinterpretarlos como aprobados. El valor agregado `joint_set_posterior_promising=false` resulta de esos checks y coincide con el código que los compone (`experiments/geometria_proporcional/run_wave54_joint_set.py:743`).

### Replay e integridad de artefactos

- Los cuatro estados NPZ relevantes son byte-idénticos entre ejecución y replay: `posterior_state`, `bootstrap_indices`, `selection_state` y `metrics_state`. También coinciden `analysis_hash=694bc4c050d617b1c271ddd5dac1e7f51aa09c5335ba957ac0fdac7b047f1e8b`, configuración congelada, selección, diagnósticos y reporte.
- Los manifests verifican sus `13/13` archivos cada uno. Las únicas diferencias entre directorios son esperables para una segunda ejecución: ruta `superseded_output` en `analysis_freeze`, su hash en `access_receipt`, `runtime_seconds`, y `external_replay_exact`, que cambia de `false` en la corrida fuente sin referencia a `true` en la corrida que sí recibe la referencia. No afectan los arrays ni el análisis.
- `14 passed` en `tests/test_wave54_joint_set.py`; incluye gauges, referencia Bernoulli, gradiente, selección, separación de bundles y comparador de replay.

## Interpretación científica acotada

**Observación:** en la población histórica primaria de 148 tokens, el posterior conjunto mejora NLL exacto y la distribución de cardinalidad frente a Platt independiente, y supera NLL a la variante sin contrastes heterogéneos. Conserva compatibilidad mayor que el baseline duro y asigna menos masa a conjuntos no observados que raw.

**Inferencia acotada:** esos datos sostienen que, sobre estos logits congelados y el soporte histórico observado, la factorización conjunta capta estructura predictiva de conjunto que no se reduce solamente a cardinalidad. No sostienen una mejora decisional suficiente: el regret no cumple el margen/IC predefinido, la accuracy cae frente a hard y el control shuffled no alcanza el margen de accuracy exigido.

**Alcance:** no se infiere utilidad natural, geometría física, causalidad del encoder, comportamiento sobre los cinco conjuntos ausentes de fit ni transferencia a otra población. La determinación de cualquier paso posterior corresponde al responsable científico.
