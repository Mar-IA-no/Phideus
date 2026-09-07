# R554 — Dual native freeze: análisis corregido posterior a R553

**Fecha:** 2026-09-07

**Estado:** `SET_VALUED_FREEZE_ONLY_VALID / PENDING-INDEPENDENT-REAUDIT`

**Régimen:** design freeze y preflight CPU; no training, GPU, promoción ni GO/NO-GO

## Alcance y corrección de registro

Este informe sustituye como análisis vigente a R552. R553 demostró que R552
sobreafirmaba la cobertura del checker: sus 32 mutaciones no incluían una
mutación de unión de soportes ni una ejecución material del path-control bajo
mutaciones privadas. El artefacto de R552 se conserva íntegro en
`superseded_pre_R553_checker/`; no se usa para acreditar el estado actual.

## Cambios de acreditación

- `source_commit=dd6187192efaa307f50efd125d2169cb437eca7f` autentica por contenido
  checker, runners, primitiva neural y tests críticos. Las ramas ligan R547,
  R549, R551 y las fuentes declaradas; el coordinador liga por SHA-256 ambas
  configs de rama.
- El checker ejecuta 30 mutaciones nominales y 21 suplementarias. Las `51/51`
  son detectadas. Entre las suplementarias están commit existente pero ajeno,
  target/utility fuera de fase, confusión entre peso crudo y normalizado,
  K distinto de 192, reutilización de calibración, retuning, ranking cruzado,
  `READY_FOR_EXECUTION`, GPU/GO, matching que lee target, soporte unión,
  identidad, cruce de folds y mapa no canónico.
- La fixture material `path_private_invariance` muta por separado los 12 campos
  de la autoridad privada —incluidos `master_id`, `corruption_mechanism` y
  `seed`—, vuelve a ejecutar `observation_tensors` y
  `shuffled_path_tensors`, y obtiene path index/sign/valid idénticos bajo seed
  derivada sólo de estructura pública.
- La fixture `posterior_hard_map` normaliza una masa de 15 estados y prueba un
  caso en que `HARD_MAP_SET=3` y la regla histórica por threshold produce 9.
- Cada run exige roster exacto, hashes y tamaños; recompone hashes de configs,
  predicados, fixtures, mutaciones, conteos, design state y consistencia mínima
  del raw NPZ. El manifest raíz autentica ambos manifests y el replay, declara
  de forma exacta los cuatro árboles históricos excluidos y vuelve a chequear
  ambos runs.

## Resultado reproducido

- tests focales CPU: `16/16 PASS`;
- predicados: `29 PASS / 1 FAIL`; la única falla es
  `R11_BASE_WEIGHTED_K192_CONFORMANCE`;
- K192: 64 grafos, 192 estados, 191 convergencias y 1 fallo; error máximo
  Torch–NumPy `1.1368683772161603e-13`; RMSE canónico p99
  `0.0017497835855976105` y máximo `0.12815754567975435`, por encima de
  `1e-4/1e-3`;
- fixtures `4/4 PASS`; mutaciones `51/51 PASS`;
- replay científico byte-exacto y checker de raíz `PASS`;
- runtime de los dos runs: `29.585947770625353 s`; pico observado
  `612368384 bytes`; `CUDA_VISIBLE_DEVICES=''`, sin uso ni consulta de GPU.

La adjudicación técnica sigue siendo `SET_VALUED_FREEZE_ONLY_VALID`: el freeze
relacional no supera la conformance predeclarada y no habilita runner; el
set-valued queda habilitado únicamente para implementar su runner CPU y volver
a auditarlo antes de abrir un draw experimental. La proyección de ese futuro
runner permanece en `120/420/1800 s`, pico superior proyectado `<1.5 GiB`,
clase `PROJECTED_CPU_PROPORTIONATE`; no es tiempo medido de entrenamiento.

## Trazabilidad canónica

- coordinador config SHA-256:
  `0c1952c0215db21b6dd077aee531f9bc0f6121260b87ca0f3dfb6aee143bae4b`;
- relational config SHA-256:
  `96bedebcf59027f8ff00c9692d628a6c0dca45c7efc3272900d2844a322c18c1`;
- set-valued config SHA-256:
  `f3a2431d325096b9facb00c4728df131a45fa73acf4dd3f367df0e1810c6e0aa`;
- `run_a/scientific_report.json`:
  `42d3dc53a394668b7d398a33ca023fce7708cd7283677bf745608e81287391be`;
- `run_a/fixed_depth_raw.npz`:
  `0632663e032c6e71f328c6651c49c303cd372633ee2b039328271970e35b3c08`;
- `replay_comparison.json`:
  `3323b0f1fd781ec099c96f8549cd92e5753f5fc2448928efd72fd79e6ce3bdc2`;
- manifest raíz:
  `33cf87fc7133592b87b5e1107eb064a2963a86aa0a947dd098383dfbc7e993f6`.

La decisión científica y cualquier promoción arquitectónica permanecen bajo
autoridad del usuario.
