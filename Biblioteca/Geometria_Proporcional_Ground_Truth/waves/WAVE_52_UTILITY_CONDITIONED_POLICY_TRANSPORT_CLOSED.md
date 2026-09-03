# Ola 52 — cierre del transporte de politica ordinal

> **Estado:** `EXECUTED / REPLAYED / DEVELOPMENT-MIXED / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Codigo ejecutado:** `b088058d6b3145212258a8b17731569b05904a68`
> **Regimen:** CPU, train/val historicos ya abiertos, lockbox no leido

## Que se puso a prueba

La Ola 51 habia separado una cabeza de conjunto y otra de eleccion, pero ambas
seguian aprendiendo de una supervision que no contenia una razon externa para
preferir un miembro compatible sobre otro. La Ola 52 introdujo esa razon como
un orden de utilidad explicito. Cada `pair_token` fue evaluado bajo las 24
permutaciones posibles de cuatro niveles ordinales, repartidas en tres folds
balanceados: ocho politicas para entrenamiento, ocho para seleccion y ocho para
desplazamiento. A lo largo de los folds, cada politica actuo exactamente una vez
como politica no vista.

El contraste central mantuvo fija la representacion set-valued. El reader
explicito eligio la familia de mayor utilidad dentro del conjunto predicho; un
reader aprendido recibio los mismos logits de conjunto y la misma utilidad. En
paralelo se entrenaron selectores contextuales end-to-end, un brazo conjunto y
controles sin utilidad, con utilidad enmascarada y con intervenciones
contrafactuales. El sistema primario promedio logits crudos de tres checkpoints
antes de tomar la decision.

La poblacion decisiva quedo formada por `148` tokens `NEAR_RIVAL` con al menos
dos familias compatibles. Cada uno fue observado bajo las 24 politicas
held-out. El bootstrap uso el `pair_token` como unidad generativa, no las 3.552
evaluaciones contextuales como si fueran observaciones independientes.

## Resultado

| Sistema | Exactitud | Accion compatible | Regret restringido | Peor regret |
|---|---:|---:|---:|---:|
| `explicit_set_policy` | 0.840 | **0.938** | 0.124 | **0.369** |
| `learned_reader_same_set` | 0.847 | 0.933 | **0.114** | 0.447 |
| `direct_contextual_choice` | 0.834 | 0.883 | 0.165 | 0.565 |
| `joint_set_contextual_choice` | 0.831 | 0.881 | 0.167 | 0.578 |
| `utility_ignored` | 0.296 | 0.930 | 0.490 | 1.025 |
| `score_composition` | **0.871** | 0.905 | 0.130 | 0.477 |
| `oracle_set_then_utility` | 1.000 | 1.000 | 0.000 | 0.000 |

La utilidad hizo trabajo efectivo dentro del banco. El selector contextual supero al control que la
ignoraba en `+0.5389` de exactitud, IC95 `[+0.4980,+0.5769]`. El reader
explicito con contexto verdadero supero su version enmascarada en `+0.5518`,
IC95 `[+0.5051,+0.5940]`. El banco, por lo tanto, contiene una decision que
depende causalmente del orden de utilidad y no puede resolverse por prevalencia
de clases solamente.

La factorizacion explicita tambien produjo una ventaja concreta frente al
selector directo: elevo la tasa de accion compatible en `+0.0543`, IC95
`[+0.0296,+0.0831]`, y redujo el regret en `-0.0407`, IC95
`[-0.0684,-0.0167]`. Sin embargo, no mejoro la exactitud: el delta fue
`+0.0051`, IC95 `[-0.0152,+0.0267]`, lejos del margen predeclarado de `+0.03`.
Frente al reader aprendido sobre la misma representacion, la diferencia fue
`-0.0070`, IC95 `[-0.0265,+0.0115]`. El resultado no respalda una superioridad
general del reader explicito.

El control contrafactual completo tampoco alcanzo el criterio fijado. Cuando la
politica cambiaba de modo que otro miembro compatible pasara a ser el ganador,
el reader explicito cambio hacia ese nuevo ganador en `0.677` de los casos,
frente al minimo diagnostico de `0.80`. El patron conjunto preregistrado fue,
por lo tanto, falso.

## Que significa arquitectonicamente

La Ola 52 no devuelve la investigacion al punto anterior. Muestra que la
separacion entre identificacion y decision tiene una funcion operacional: una
politica explicita puede reutilizar una representacion congelada ante ordenes
nuevos, reducir decisiones incompatibles y limitar el costo de errores severos.
Pero tambien muestra el limite de esa modularidad cuando el conjunto de entrada
es imperfecto. El set-head obtuvo recall `0.826`, ancho medio `2.547` y al menos
un miembro incompatible en `0.276` de los tokens. Ninguna politica posterior
puede elegir una alternativa compatible que la representacion ya excluyo, ni
evitar por completo una alternativa espuria que la representacion autorizo.

El siguiente discriminante no deberia ser otra cabeza mas grande sobre el mismo
conjunto binarizado. La evidencia favorece una interfaz de decision sensible a
incertidumbre: conservar logits o una region de compatibilidad, incorporar la
utilidad mediante riesgo esperado o regret robusto y permitir abstencion cuando
la evidencia no autoriza una accion estable. Esta alternativa debe compararse
contra la composicion simple de scores, que alcanzo mayor exactitud pero peor
compatibilidad y peor regret maximo que la politica explicita. La pregunta deja
de ser si conviene aprender o programar un argmax; pasa a ser como transportar
una region identificada incompleta hacia una accion sin fingir que sus bordes
son exactos.

## Integridad y reproduccion

- corrida primaria: `98.83 s`, CUDA deshabilitada;
- replica independiente: `110.55 s`, CUDA deshabilitada;
- `31/31` NPZ presentes en ambas corridas y `497/497` arrays exactos;
- `36/36` checkpoints presentes y `1.548/1.548` tensores exactos;
- informe final byte-identico y `artifact_manifest.json` byte-identico;
- summaries semanticamente iguales al excluir `runtime`;
- `artifact_hash` identico: `5ff5333e9f6da8071a15d643538caa8828e8ea477f0d9603139cd8b1887c996f`;
- el manifiesto de paquete difiere solo en el hash de `summary.json`, porque el
  tiempo de ejecucion es distinto entre corridas independientes;
- el replay interno recargo checkpoints y reprodujo exactamente logits,
  metricas agregadas, metricas por seed y metricas por politica.

La comparacion independiente queda preservada en
`data/geometria_proporcional/wave52_policy_transport_replay_comparison.json`.

Una auditoria independiente posterior confirmo la adjudicacion del patron y la
unidad de bootstrap. Corrigio un exceso de alcance en la expresion "utilidad
real": la evidencia sostiene dependencia efectiva de una utilidad sintetica y
transporte parcial dentro del banco ordinal, no utilidad natural o de dominio.
Tambien distinguio la replica computacional exacta de una replica cientifica
independiente. El informe verbatim se conserva en
`agent_reports/296_wave52_result_independent_audit.md`.

## Alcance

Es evidencia de desarrollo sobre datos historicos abiertos, con una utilidad
ordinal construida y un mismo generador sintetico. No leyo el lockbox, no valida
una utilidad natural, no prueba una geometria fisica, no promueve una
arquitectura y no decide GO/NO-GO.
