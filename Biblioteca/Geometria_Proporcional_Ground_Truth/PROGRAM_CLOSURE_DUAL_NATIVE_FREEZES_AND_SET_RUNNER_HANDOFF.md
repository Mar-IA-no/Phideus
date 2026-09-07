# Cierre del dual native freeze y relevo al runner set-valued

**Fecha:** 2026-09-07  
**Estado:** `SET_VALUED_FREEZE_ONLY_VALID`  
**Régimen:** diseño, implementación contractual y preflight CPU; sin training, GPU, promoción arquitectónica ni GO/NO-GO

## Pregunta cerrada

Después de que `MAPPING-FEASIBILITY` descartara un factorial común, este goal
debía convertir sus dos hojas nativas en protocolos ejecutables y auditables:

1. `GENERIC/TYPED × WLS/IRLS` para el núcleo relacional;
2. `MARGINAL/JOINT × HARD/CONTEXTUAL` para el sistema set-valued.

Las ramas comparten procedencia, fases, replay y reglas de preservación, pero
no comparten unidad, observación, target, métrica ni una función de mérito. El
goal no abrió draws experimentales ni entrenó modelos.

## Adjudicación técnica

El checker recompone treinta predicados desde las configs y, para el surrogate
relacional, desde los estados numéricos crudos. El resultado final es:

- `29 PASS / 1 FAIL`;
- `R11_BASE_WEIGHTED_K192_CONFORMANCE=FAIL`;
- `SET_VALUED_FREEZE_ONLY_VALID`.

La confirmación fresca K192 usó `64` grafos y `192` estados con tamaños
`8..16`. Torch y NumPy fixed-depth coincidieron hasta
`1.1368683772161603e-13`; ambas familias de gradientes conservaron conformidad.
El freeze relacional falla, sin embargo, frente al executor canónico: hubo
`191/192` convergencias, RMSE canónico p99
`0.0017497835855976105` y máximo `0.12815754567975435`, por encima de los
umbrales predeclarados `1e-4/1e-3`. Conforme al protocolo no se probó otro K ni
se habilitó el runner relacional.

La rama set-valued pasó sus doce predicados. Esto acredita únicamente que el
contrato `MARGINAL/JOINT × HARD/CONTEXTUAL`, sus fases, controles, schemas,
estimandos y presupuesto están suficientemente congelados para implementar su
runner. No acredita resultados empíricos de la corrida aún inexistente.

## Integridad y replay

- fixtures materiales: `4/4 PASS`;
- mutaciones negativas: `61/61 PASS`;
- tests focales: `17/17 PASS`;
- replay científico: byte-exacto entre `run_a` y `run_b`;
- checker de raíz: `PASS`;
- runtime conjunto de los dos preflights: `29.285093147307634 s`;
- RSS pico observado: `612347904 bytes`;
- GPU consultada o usada: `false`.

R557 repitió de manera independiente el cálculo K192 contra el código ligado:
las quince arrays fueron idénticas al NPZ canónico. También recompuso el
resumen desde raw, verificó las siete variaciones antes omitidas, los bindings
literales, los manifests y el estado final. Veredicto: `PASS`, sin findings
altos, medios ni bajos.

Hashes canónicos:

- coordinator config: `a80051c102bf873e977794ce850339c887ea039ec65384bd11d383af54499f9c`;
- `run_a/scientific_report.json`: `4c213f2eab6e38f0f230a7510c5b49bcaf97c46539e9dea46eb984026723a705`;
- `run_a/fixed_depth_raw.npz`: `104409ef05add1c2710ceb7d05b55956814ef2c9ad00d794991684de2e91c7c7`;
- `replay_comparison.json`: `6751bf95bbfcf618b2667dfb00b7dc89557cf790bc77126d63a3251b3f08818e`;
- root manifest: `e794c541214a02dac5a43e855318cca72eadcad96f7e0426d3b95b388cba3a81`.

## Relevo finito

La rama relacional queda preservada como alternativa rechazada por este
surrogate y esta confirmación, no como techo general de arquitecturas
relacionales. La continuación proporcionada por CPU es implementar y auditar
el runner set-valued contra fixtures y artefactos ya abiertos, sin crear aún
el draw fresco ni abrir el monitor. Sólo después de esa auditoría podrá
considerarse una ejecución experimental bajo el freeze.

El costo set-valued permanece proyectado en `120/420/1800 s` y menos de
`1.5 GiB`; todavía no es runtime medido de training. La GPU sigue suspendida
hasta habilitación explícita del usuario. Este cierre no promueve una
arquitectura ni decide GO/NO-GO.

## Fuentes inmediatas

- `experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md`
- `experiments/geometria_proporcional/configs/proportional_dual_native_freeze_v1.json`
- `data/geometria_proporcional/proportional_dual_native_freeze_v1/`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/556_proportional_dual_native_freeze_raw_derived_analysis.md`
- `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/557_proportional_dual_native_freeze_final_independent_reaudit.md`
