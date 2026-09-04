# Ola 56 Stage 1 — cierre prospectivo de la compuerta contextual

## Estado

La realización prospectiva fresca y su replay terminaron las tres fases en
CPU: `PREPARED → FIT_COMPLETE → SELECT_COMPLETE → COMPLETE`. La recuperación
conservó el escrow y el freeze previos byte por byte; el intento que había
quedado `PREPARED` fue archivado como `superseded`, no eliminado. El replay
reprodujo exactamente los `23/23` compromisos de preparación y los `10/10`
artefactos analíticos deterministas exigidos por el protocolo.

La ejecución es técnicamente válida y reproducible. El patrón prospectivo
predeclarado, sin embargo, quedó `false`: se cumplieron cuatro de sus seis
condiciones y fallaron la no inferioridad de accuracy frente a la política dura
y la magnitud mínima de mejora de regret frente al control shuffled. Este cierre
no promueve una arquitectura ni resuelve un `GO/NO-GO`, cuya autoridad permanece
en el usuario.

## Recuperación y frontera de autoridad

El primer intento oficial había fallado antes de abrir labels porque el freeze
visible omitía `calibration_null.jsonl`, aunque el archivo sí pertenecía al
manifest autenticado. La corrección no reutilizó ese estado como autorización
implícita. Se construyó una cadena nueva y lineal:

1. plan final de matriz de autoridad `P8`;
2. auditoría independiente del plan `R391`;
3. implementación `I6`, ligada a las tres etapas históricas que habían
   modificado runner, parser y cobertura;
4. auditoría independiente de implementación `R392`;
5. amendment canónico v8;
6. auditoría final del paquete `R393`;
7. ejecución bajo el commit terminal exacto y worktree limpio.

El parser exige decisión terminal congruente, paths canónicos, hashes, commits
de introducción únicos, diffs exclusivos y padres directos. La suite focal cerró
`77/77` y la regresión explícita Wave 49–56 `229/229`. El amendment mantuvo sin
cambios inventario, población, aserciones de no-redraw y origen de escrow.

## Selección congelada

FIT reunió `299` pair tokens primarios y `1.114` filas de desacuerdo. SELECT
reunió `299` tokens y `1.051` filas. Sobre validation, la compuerta contextual
eligió `q=0.6`, threshold `0.1589291943` y coverage `0.0585284`; obtuvo accuracy
`0.8320792`, compatibilidad `0.9499721` y regret `0.1141885`. Los dos shards
deterministas volvieron a elegir `q=0.6` y conservaron los signos del selector
completo, de modo que `selector_sensitive=false`.

El control advantage-only eligió `q=0.9`. La compuerta escalar heredada eligió
`hard_only`. Ninguno de esos valores fue recalculado después sobre el monitor.

## Resultado en monitor

La población primaria contiene `301` pair tokens `NEAR_RIVAL` con cardinalidad
mayor o igual que dos y `1.175` decisiones donde la acción dura y la bayesiana
discrepan.

| Brazo | Accuracy | Compatible | Regret | Worst regret | Coverage |
|---|---:|---:|---:|---:|---:|
| política dura | 0.837209 | 0.930233 | 0.132521 | 0.392857 | 0 |
| compuerta contextual | 0.820598 | 0.951550 | 0.118563 | 0.420819 | 0.069906 |
| advantage-only | 0.822813 | 0.935770 | 0.133490 | 0.421650 | 0.026163 |
| promedio shuffled | 0.825941 | 0.939120 | 0.127346 | 0.436157 | — |
| posterior conjunto puro | 0.759828 | 0.964978 | 0.132740 | 0.468992 | 0.162652 |
| oracle positive-gain | 0.872785 | 0.968577 | 0.083299 | 0.314230 | 0.043605 |

La compuerta contextual efectuó `505` overrides: `199` beneficiosos, `294`
perjudiciales y `12` neutrales. La ganancia media de los overrides fue positiva
porque los beneficios tuvieron mayor magnitud, pero la frecuencia de decisiones
perjudiciales fue superior. Esta asimetría anticipa la tensión que aparece entre
el regret medio y la cola.

## Adjudicación del patrón predeclarado

| Condición | Estado | Evidencia |
|---|---|---|
| 1. Reducir regret frente a hard | cumple | Δ `-0.013958`, IC95 `[-0.026359,-0.002757]` |
| 2. Accuracy y compatibilidad no inferiores | no cumple | compatibilidad mejora; accuracy Δ `-0.016611`, IC95 inferior `-0.028931 < -0.01` |
| 3. Reducir regret frente a scalar y advantage-only | cumple | magnitudes e IC95 satisfacen ambos umbrales |
| 4. Reducir regret al menos `0.01` frente a shuffled | no cumple | Δ `-0.008783`, IC95 `[-0.017543,-0.000724]` |
| 5. Superar a pure joint en accuracy y regret | cumple | ambos IC95 satisfacen el criterio |
| 6. Selector estable y replay exacto | cumple | `selector_sensitive=false`; replay `10/10` |

El resultado no equivale a ausencia de señal contextual. Frente a
advantage-only, el contextual reduce regret `-0.014927` con IC95 completamente
negativo; frente al shuffled también conserva dirección e intervalo favorables.
Lo que no transporta es el patrón conjunto con sus márgenes fijados. Frente a
hard, la mejora media de regret convive con una pérdida de accuracy mayor que la
permitida y con un empeoramiento de worst regret de `+0.027962`, IC95
`[+0.001938,+0.052602]`.

## Soporte y alcance

Los cinco conjuntos ausentes heredados de la Ola 54 tuvieron soporte `0`, por
debajo del mínimo `30`, y quedaron `NOT_EVALUABLE`. No se les imputan resultados
de los conjuntos presentes. Los splits históricos, FIT, SELECT y monitor fueron
disjuntos por pair token, con overlap `0` en las seis comparaciones.

El alcance es una realización fresca de la misma ley sintética y un catálogo
fijo de veinticuatro utilidades ordinales. No prueba transporte a otra ley,
utilidad natural, autoridad física, geometría proporcional ni PPU.

## Lectura experimental

**Observación.** La información contextual permite reducir regret medio frente
a controles que conservan sólo advantage o destruyen la correspondencia entre
features y gain. Esa reducción no preserva al mismo tiempo accuracy ni la cola
de regret frente a la política dura.

**Inferencia acotada.** El límite ya no puede atribuirse a una ausencia total de
señal ni resolverse repitiendo un umbral escalar. La regresión de gain medio no
distingue con suficiente seguridad la mayor cantidad de overrides perjudiciales
ni protege los peores casos.

**Hipótesis siguiente.** Una arquitectura de decisión más informativa separaría
valor esperado y riesgo de daño: un predictor de gain puede proponer el cambio,
mientras un segundo estimador de probabilidad o cuantil de pérdida decide si la
acción queda autorizada. El contraste pertinente debe usar una nueva realización
fresca, comparar contra un control de capacidad igualada y preregistrar
simultáneamente regret medio, accuracy y worst regret. Es una alternativa
experimental CPU de baja capacidad, no una promoción silenciosa del posterior
ni de la compuerta contextual.

## Artefactos y auditoría

- Primario: `data/geometria_proporcional/wave56_contextual_gate_fresh_v1/`
- Replay: `data/geometria_proporcional/wave56_contextual_gate_fresh_v1_replay/`
- Reporte canónico: `phases/adjudicate.complete/analytics.complete/REPORT_WAVE56_STAGE1.json`
  con SHA-256 `856f954f724e6858766f5dd20fc469305180ba4536dcd5ba8eaba82c31d49bae`.
- Arrays finales: `result_arrays.npz`, SHA-256
  `4608180ca1ba34601103b4abeb9e2b8bb45b7db8ea0fc07ad11ada47bd1bd051`.
- Replay receipt: `phases/adjudicate.complete/replay_receipt.json`, SHA-256
  `279461b993ca760a212dd70ae6b8d5889a762391c05f617af728b62850ed77b9`.
- Auditoría independiente de resultados: `R394`, dictamen
  `PASS-CON-RIESGOS`, SHA-256
  `bccbe20cfd2363178d5cf59e3d1fb836b2ac451dd27726b188b1c8c0b25e681b`.

Toda la ejecución fue CPU-only. No se consultó ni utilizó GPU.
