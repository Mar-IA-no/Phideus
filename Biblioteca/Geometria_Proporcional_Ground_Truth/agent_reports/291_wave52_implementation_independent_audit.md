# Auditoria independiente de implementacion - Ola 52

> Estado: `REVISE`
> Fecha: 2026-09-03
> Instancia independiente: Kierkegaard (`01a06622-38bb-7ef0-923d-4633285dd0d5`)
> Alcance: plan, config, primitivas, runner y tests previos a toda ejecucion.

## Findings verbatim

1. **HIGH: los casos fuera de catalogo validos bloquean la corrida.** El plan exige excluir targets vacios de CE y reportarlos, pero `authorized_actions` rechaza cualquier fila vacia y se invoca sin mascara durante entrenamiento y evaluacion. Correccion: aplicar CE solo a `cardinality > 0`, conservar BCE para todos en el joint y producir metricas separadas para targets vacios.

2. **HIGH: falta el preflight de unidad generativa; bootstrap y splits asumen directamente `pair_token`.** El agrupamiento se limita a `pair_token` y el bootstrap remuestrea tokens individualmente. No se verifica si comparten una unidad superior, como exige el plan. Correccion: materializar y firmar `pair_token -> cluster_id`, comprobar disjuncion de splits a ese nivel y remuestrear clusters completos.

3. **HIGH: el contraste declarado como "misma representacion, distinto reader" no fija realmente la representacion agregada.** `explicit_set_policy` compone despues de promediar logits de set entre seeds, mientras el reader aprendido se entrena con logits especificos de cada seed y promedia salidas no lineales posteriormente. Correccion: alimentar ambos readers con exactamente el mismo tensor de logits crudos agregado en el mismo punto, o evaluar ambos por seed y aplicar identica regla de ensemble.

4. **HIGH: el control contrafactual no implementa el estimando predeclarado.** Solo evalua el reader explicito ensemble, no cada checkpoint. Ademas exige que ambas predicciones sean correctas, mezclando sensibilidad causal con exactitud basal. Correccion: fijar y guardar pares de politicas con ganador distinto, evaluar por checkpoint y medir directamente `a_cf != a_original && a_cf == target_cf`; reportar la correccion conjunta por separado.

5. **HIGH: artefactos y replay incumplen el contrato.** Los NPZ guardan acciones pero no logits crudos de ambas cabezas ni scores por seed; tampoco se guardan mappings de shuffle ni acciones de train/threshold. No existe replay array-exact de la corrida antes del empaquetado. Correccion: persistir logits/scores por split-fold-seed-arm, targets derivados y mappings; anadir replay desde checkpoints que compare arrays, metricas y hash antes de emitir el paquete final.

6. **MEDIUM: `worst_restricted_regret` esta mal agregado.** Se calcula el maximo dentro de cada fold, pero despues se promedian esos maximos junto con las metricas ordinarias. El resultado no es el peor regret entre los 24 contextos. Correccion: usar el maximo de los parciales por token.

7. **MEDIUM: faltan outputs necesarios para aplicar integramente plan y criterios.** Faltan recall, ancho, incompatibilidad, AUC/AP, fallback agregado, IC de metricas, resultados por seed y politica, y costos de parametros/backprops/operaciones. El criterio de integridad solo comprueba logits exactos. Correccion: emitir esas metricas y recibos antes de evaluar `policy_transport_promising`.

8. **MEDIUM: la barrera anti-leakage/binding ocurre demasiado tarde.** Los labels se cargan antes de generar el manifest de politicas, y el snapshot de fuentes ejecutadas se valida recien despues del entrenamiento. Correccion: generar, validar y hashear politicas antes de leer labels; capturar bindings de codigo/config al inicio y ligar luego las acciones derivadas a ese hash.

## Veredicto

`REVISE`
