# Ola 55 — cierre del puente conservador posterior-decisión

> **Estado:** `CLOSED / CPU-ONLY / FRESH-SAME-GENERATOR / EXACT-REPLAY / DEVELOPMENT-NEGATIVE / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Plan congelado:** `WAVE_55_CONSERVATIVE_POLICY_BRIDGE_PLAN.md`
> **Código ejecutado:** `cbeabeba20a9e0b2b472672e86408e83b325eb08`

## Qué se puso a prueba

La Ola 54 había separado representación y decisión. Su posterior regularizado
sobre los quince conjuntos compatibles reducía NLL, corregía cardinalidad y
capturaba interacciones que las marginales independientes perdían, pero la
acción bayesiana pura sacrificaba exactitud frente al conjunto duro. La Ola 55
probó una interfaz conservadora entre ambos niveles: mantener la acción dura y
reemplazarla por la acción bayesiana sólo cuando la reducción de riesgo estimada
por el posterior superara un umbral global `gamma`.

El contraste mantuvo congelados el generador, el encoder `sigmoid_only`, sus
tres seeds, el normalizador, los parámetros primarios del posterior conjunto,
el calibrador Platt, las veinticuatro políticas ordinales y la función de
regret. Sólo cambió la regla posterior-decisión. Se compararon el posterior
conjunto completo, su variante unary+cardinalidad, las marginales independientes
y un posterior entrenado con targets barajados, además del baseline duro, la
decisión bayesiana pura y una referencia privilegiada.

## Protocolo e integridad

La arquitectura se adjudicó sobre una realización fresca de la misma ley
sintética. `decision_select` reunió `768` tokens y seleccionó `gamma`; un
`sealed_monitor` físicamente separado reunió otros `768`. La población primaria
fue `NEAR_RIVAL` con cardinalidad mayor o igual que dos: `300` tokens para
selección y `302` para monitor. El lockbox se generó, pero su oracle no se
materializó ni se consultó.

La inferencia sobre train y val ocurrió antes de crear sus labels autorizados,
en un worker que sólo recibió observaciones visibles, normalizador y checkpoints
inference-only. El benchmark nuevo no comparte tokens con los bundles de ajuste,
selección o monitor de la Ola 54. Un re-forward histórico reprodujo exactamente
los logits de los tres checkpoints de Ola 51 antes de generar datos frescos.

La preparación replay pasó `12/12` comprobaciones; la adjudicación repitió de
forma byte-exact `analysis_core.json`, `selection_freeze.json`,
`result_arrays.npz` y `bootstrap_indices.npz`. La auditoría independiente `R313`
recalculó selección, métricas, intervalos, sensibilidad y manifests sin hallar
findings materiales. La suite relevante cerró con `104 passed`.

## Resultado primario

En `decision_select`, el posterior conjunto completo mostró el conflicto que la
compuerta debía resolver. Los umbrales bajos reducían regret, pero violaban la
restricción de no inferioridad en accuracy. `gamma=0.4` ya era factible, aunque
su regret (`0.120000`) resultó apenas peor que el baseline duro (`0.119722`). Por
eso el selector predeclarado eligió correctamente `hard_only`.

| Brazo | `gamma` seleccionado |
|---|---:|
| `bridge_joint_full` | `hard_only` |
| `bridge_joint_unary_cardinality` | `hard_only` |
| `bridge_independent_platt` | `hard_only` |
| `bridge_joint_target_shuffled` | `0.2` |

En el monitor primario, el puente conjunto completo reprodujo exactamente el
baseline duro:

| Brazo | Accuracy | Compatible | Regret | Máximo regret medio por token |
|---|---:|---:|---:|---:|
| conjunto duro | `0.831126` | `0.937362` | `0.129622` | `0.394316` |
| puente conjunto completo | `0.831126` | `0.937362` | `0.129622` | `0.394316` |
| Bayes conjunto puro | `0.781457` | `0.974614` | `0.114008` | `0.426876` |

El puente preservó accuracy frente a la decisión bayesiana pura (`+0.049669`,
IC95 `[+0.030077,+0.070778]`), pero no conservó su ventaja de regret
(`+0.015614`, IC95 `[-0.000747,+0.032631]`). Frente al control barajado, el
puente tuvo regret ligeramente mayor (`+0.004289`, IC95
`[+0.000483,+0.008578]`). El patrón predeclarado resolvió `4/9` condiciones y
quedó `false` con replay exacto.

## Qué revela la sensibilidad

La selección sobre todos los tokens in-catalog, en lugar de la población
primaria, eligió `gamma=0.2`. Aplicada al mismo monitor primario, redujo regret
frente al conjunto duro (`-0.010934`, IC95 `[-0.020925,-0.001644]`) y aumentó
compatibilidad (`+0.018350`, IC95 `[+0.011589,+0.025938]`), pero redujo accuracy
(`-0.011727`, IC95 `[-0.023593,-0.000276]`). Cambiaron de estado algebraico
`5/7` contrastes predeclarados: el resultado es selector-sensitive.

La limitación no parece ser que el posterior jamás detecte una oportunidad.
Con `gamma=0.2` realiza `353/7248` overrides token×política; `134` son
beneficiosos y `219` perjudiciales. Los beneficios tienen magnitud suficiente
para reducir regret agregado, aunque la precisión de override es sólo `37.96%`.
La correlación entre ventaja posterior estimada y mejora realizada es débil
(`r≈0.17`). Un umbral único ordena parcialmente la magnitud del riesgo, pero no
separa de forma estable cuándo conviene abandonar la acción dura.

## Implicancia arquitectónica

La Ola 55 no invalida el posterior conjunto ni toda política residual. Invalida
una interfaz más estrecha: **una compuerta global escalar basada sólo en la
ventaja esperada no convierte de manera estable la mejora representacional del
posterior en una mejora conjunta de accuracy, compatibilidad y regret** bajo
este protocolo.

El próximo contraste no debería repetir la grilla de `gamma` ni escalar el
encoder. La alternativa con mayor poder diagnóstico es una compuerta residual
condicional y de baja capacidad. Sus entradas deben existir en inferencia:
ventaja estimada, entropía y margen del posterior, cardinalidad del conjunto
duro, desacuerdo entre acción dura y bayesiana, dispersión entre seeds y
estructura de la política/utilidad. Su target de entrenamiento puede ser el
signo o la magnitud del cambio de regret observado, con particiones agrupadas
por token y las veinticuatro políticas mantenidas juntas.

Ese experimento separaría dos explicaciones. Si la heterogeneidad entre estados
y políticas es el problema, una compuerta condicionada debería mejorar el
frente regret-accuracy. Si tampoco una regla regularizada y baja en capacidad
supera al umbral escalar en un monitor fresco, la evidencia apuntaría a que el
posterior actual no ordena sus errores de manera decisionalmente utilizable bajo
esta autoridad de utilidad.

## Soporte y alcance

Los cinco conjuntos ausentes del fit de Ola 54 tuvieron `0` tokens en monitor;
el diagnóstico queda `NOT_EVALUABLE` frente al mínimo predeclarado de `30`. Su
evaluación requiere un brazo enriquecido declarado como otra ley, no una
extensión silenciosa del resultado primario.

La evidencia corresponde a una realización fresca del mismo generador, sobre
logits y posterior congelados, sin GPU. No prueba transferencia a otro aparato,
utilidad natural, geometría física ni una PPU. Tampoco promueve una arquitectura
ni decide GO/NO-GO.

## Artefactos

- `data/geometria_proporcional/wave55_policy_bridge_fresh_v1/`
- `data/geometria_proporcional/wave55_policy_bridge_fresh_v1_replay/`
- `data/geometria_proporcional/wave55_policy_bridge_results_v1/`
- `data/geometria_proporcional/wave55_policy_bridge_results_v1_replay/`
- `agent_reports/307_wave55_policy_bridge_plan_independent_audit.md`
- `agent_reports/308_wave55_policy_bridge_plan_focal_reaudit.md`
- `agent_reports/309_wave55_policy_bridge_plan_final_reaudit.md`
- `agent_reports/310_wave55_implementation_independent_audit.md`
- `agent_reports/311_wave55_implementation_focal_reaudit.md`
- `agent_reports/312_wave55_implementation_final_reaudit.md`
- `agent_reports/313_wave55_result_independent_audit.md`
