# R556 — Dual native freeze con adjudicación derivada del raw

**Fecha:** 2026-09-07

**Estado:** `SET_VALUED_FREEZE_ONLY_VALID / PENDING-FINAL-INDEPENDENT-REAUDIT`

**Régimen:** design freeze y preflight CPU; no training, GPU, promoción ni GO/NO-GO

## Corrección de autoridad

R555 invalidó la acreditación de R554 al demostrar que el checker aceptaba
reescribir el status K192 y promover ambos freezes sin modificar el NPZ
contradictorio. R556 reemplaza a R554 como análisis vigente. Los artefactos de
R554 se preservan en
`proportional_dual_native_freeze_v1_history/superseded_R555_raw_recomposition_gap/`.

El checker ya no toma `fixed_depth_conformance.status` del reporte como fuente.
El NPZ conserva ahora node count y mecanismo por estado, errores fixed y
canónicos, convergencia, iteraciones y los vectores analíticos/numéricos de
ambas familias de gradientes con offsets y exclusiones. Desde esos estados
recompone el resumen completo: cobertura `8..16`, convergencias, estadísticos
de iteraciones, máximo Torch–NumPy, p99/máximo canónicos, métricas de gradiente
y status. Sólo después vuelve a adjudicar R11, conteos y design state. El test
adversarial reproduce el ataque de R555, actualiza hashes internos y exige su
rechazo por tres vías: resumen numérico, predicados y estado.

## Freeze exhaustivo y procedencia

Los predicados validan ahora las recetas congeladas completas de generador,
arms, training, executors, conformance K192, controles, inferencia, fresh draw,
MARGINAL/JOINT, HARD, CONTEXTUAL, target shuffle, matched controls y costo
proyectado. La suite agrega diez negativos a los 51 anteriores: tolerancia
IRLS, solver marginal, utility tie, alpha Ridge, selection key, frescura, clase
de costo y tres omisiones de bindings. El resultado es `61/61 PASS`.

La cobertura literal de fuentes agrega:

- config, compute contract y runtime del smoke relacional;
- schema W49;
- runner, artifact manifest, source bindings y config snapshot del matched
  control W59.

`source_commit=f9bba63e2e744b84873bcf6e6a06eb845683d511` reproduce los
seis paths críticos. El coordinador liga las configs relacional y set-valued
por SHA-256, respectivamente
`57c5ab901ec292c87af3c2b051e6fd6b8b3a2be3a0787ccf94fa85e68ee5393f`
y `41b204789d8ba3cbdff057cf6a10f73bb39fe9c1a7b44233f2b0b78181e47409`.

## Resultado CPU reproducido

- tests focales: `17/17 PASS`;
- predicados: `29 PASS / 1 FAIL`; el único fallo es R11;
- fixtures materiales: `4/4 PASS`;
- mutaciones: `61/61 PASS`;
- K192: 64 grafos, 192 estados, tamaños `8..16`, 191 convergencias y 1
  fallo; error máximo Torch–NumPy `1.1368683772161603e-13`; RMSE canónico p99
  `0.0017497835855976105` y máximo `0.12815754567975435` frente a umbrales
  `1e-4/1e-3`;
- replay científico byte-exacto y checker raíz `PASS`;
- dos runs en `29.285093147307634 s`, pico observado `612347904 bytes`,
  `CUDA_VISIBLE_DEVICES=''`, sin uso ni consulta de GPU.

La adjudicación técnica permanece `SET_VALUED_FREEZE_ONLY_VALID`. El freeze
relacional queda rechazado por la confirmación predeclarada y no habilita su
runner. El freeze set-valued habilita únicamente implementar y auditar un
runner CPU antes de crear o abrir datos experimentales. Su costo sigue siendo
una proyección, `120/420/1800 s` y `<1.5 GiB`, no runtime medido de training.

## Hashes canónicos

- coordinator config:
  `a80051c102bf873e977794ce850339c887ea039ec65384bd11d383af54499f9c`;
- `run_a/scientific_report.json`:
  `4c213f2eab6e38f0f230a7510c5b49bcaf97c46539e9dea46eb984026723a705`;
- `run_a/fixed_depth_raw.npz`:
  `104409ef05add1c2710ceb7d05b55956814ef2c9ad00d794991684de2e91c7c7`;
- `replay_comparison.json`:
  `6751bf95bbfcf618b2667dfb00b7dc89557cf790bc77126d63a3251b3f08818e`;
- root manifest:
  `e794c541214a02dac5a43e855318cca72eadcad96f7e0426d3b95b388cba3a81`.

La promoción arquitectónica y toda decisión científica permanecen bajo
autoridad del usuario.
