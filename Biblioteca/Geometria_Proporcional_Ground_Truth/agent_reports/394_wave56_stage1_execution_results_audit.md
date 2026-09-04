# Ola 56 Stage 1 — auditoría independiente de ejecución y resultados

**Artefacto primario:** `data/geometria_proporcional/wave56_contextual_gate_fresh_v1`  
**Replay:** `data/geometria_proporcional/wave56_contextual_gate_fresh_v1_replay`  
**Commit observado antes de este informe:** `4d53138d152365460dba36a883f956dfdb349b90`  
**Dictamen:** `PASS-CON-RIESGOS`

## Síntesis ejecutiva

La ejecución de Stage 1 y su replay forman una cadena técnicamente consistente.
El primario transitó `PREPARED → FIT_COMPLETE → SELECT_COMPLETE → COMPLETE`;
cada fase recibió únicamente su split autorizado, los tres workers analíticos
corrieron como UID/GID `65534`, sin capabilities efectivas ni acceso al root del
benchmark, y los probes de truth sellada fueron denegados. El replay reproduce
exactamente los 23 compromisos de preparación declarados y los 10 artefactos
analíticos deterministas que su receipt obliga a comparar. Los nueve NPZ de
logits son byte-exactos entre primario y replay.

La recomputación independiente desde los arrays congelados coincide con el
reporte: 24 resúmenes de brazos con error máximo `0`, cuatro promedios de
shuffles con error máximo `1.11e-16`, 80 escalares de contrastes/bootstrap con
error máximo `0` y 104 métricas por slice/política con error máximo `0`. Los
arrays de bootstrap tienen forma `5000 × 301` y usan únicamente índices
`0..300`. Los archivos `fit_arrays.npz`, `selection_arrays.npz` y
`result_arrays.npz` son array-exactos entre primario y replay en sus `61`, `67`
y `303` arrays, respectivamente; además, los archives de FIT y SELECT embebidos
en el resultado final coinciden array por array con sus artefactos de origen.

El resultado científico predeclarado no es positivo: el patrón prospectivo es
`false`. Se satisfacen las condiciones 1, 3, 5 y 6, pero no la 2 ni la 4. El
gate contextual reduce regret agregado frente al hard-set en `0.013958` con
IC95 `[-0.026359, -0.002757]` y mejora compatibilidad en `0.021318`, pero pierde
accuracy en `0.016611`; el límite inferior de ese contraste de accuracy es
`-0.028931`, peor que el margen de no inferioridad `-0.01`. Frente al promedio
de cinco shuffles, la reducción de regret es estadísticamente orientada en la
dirección esperada, IC95 `[-0.017543, -0.000724]`, pero su magnitud puntual es
`0.008783`, menor que el mínimo predeclarado `0.01`.

Esto no exige reparar artefactos ni invalida la ejecución: es un resultado
negativo/parcial del contraste preregistrado. Sí impide presentar Stage 1 como
validación del patrón completo o como promoción de arquitectura. La autoridad
de cualquier decisión científica sigue siendo del usuario.

## Alcance, límites y régimen de lectura

Audité config, contratos y planes de Ola 56, código de preparación/ejecución y
workers, receipts públicos, freezes, manifiestos, JSON analíticos y NPZ de
logits/resultados. Cargué NPZ únicamente para verificaciones numéricas concretas;
no volqué sus contenidos ni generé derivados persistentes.

No abrí `benchmark/sealed/**`, `generation_escrow.json`, ningún JSONL bajo
`authorized_labels/`, credenciales ni secretos. Para esos objetos sólo usé
metadata y hashes publicados por manifests/receipts. No lancé prepare,
recovery, replay, inferencia ni ninguna fase experimental. No consulté ni usé
GPU. El único archivo creado por esta auditoría es este informe.

La evaluación cubre integridad, reproducibilidad y consistencia interna de este
monitor sintético fresco bajo la misma ley generadora. No acredita transporte a
otra ley, utilidad natural, autoridad física, PPU validada ni techo de la
hipótesis.

## Fuentes canónicas y hashes

Los números de las secciones siguientes están anclados a estas fuentes:

| Fuente | SHA-256 | Uso |
|---|---|---|
| `experiments/geometria_proporcional/configs/wave56_contextual_gate_fresh.json` | `7b5103304ff7bc8c8e56c9b26a45f857ab6e751fe147849fb6470d33cf48d399` | población primaria, selector, márgenes, bootstrap y criterios |
| `.../phases/fit.complete/analytics.complete/fit_core.json` | `902be66b592e7ddd646cc7dc69a0ff0ecf1eaf8bc7238f95a10a49c3847727ae` | FIT, modelos y controles shuffled |
| `.../phases/select.complete/analytics.complete/selection_core.json` | `da7ab1aa89b6791a4c908ce04fcdbc5e523aa28edd806af0dfb6a93a38a35fde` | selección y disjointness |
| `.../phases/adjudicate.complete/analytics.complete/analysis_core.json` | `856f954f724e6858766f5dd20fc469305180ba4536dcd5ba8eaba82c31d49bae` | resúmenes, contrastes, slices y condiciones |
| `.../phases/adjudicate.complete/analytics.complete/result_arrays.npz` | `4608180ca1ba34601103b4abeb9e2b8bb45b7db8ea0fc07ad11ada47bd1bd051` | recomputación independiente y diagnóstico por seed |
| replay `.../phases/adjudicate.complete/replay_receipt.json` | `279461b993ca760a212dd70ae6b8d5889a762391c05f617af728b62850ed77b9` | exactitud analítica 10/10 |
| replay `.../preparation_replay.json` | `d7e56be9f544088e92166018da6665b7fa6a475663ee069be6d98635046e762e` | exactitud de preparación 23/23 |
| replay `.../phases/adjudicate.complete/diagnostic_outcome.json` | `272619de310e51ef74fbd1ab96b32182585e222104536c86a2de2eeafd2f6fe0` | cierre conjunto con replay |
| `.../benchmark/attestations/semantic_root.json` | `0ea98a1c263c47bed1626268559588a39d88841a34aa837dfc66e3bcfe1506f0` | población y compromisos públicos |
| `.../phases/adjudicate.complete/runtime.json` | `95f7b27b3d5c5edcd497a7149c623686b000b7f81d463a8a19823887c0cfd4f1` | runtime CPU y versiones |

En esta tabla, `...` refiere al root primario indicado al comienzo. El
`REPORT_WAVE56_STAGE1.json` es byte-idéntico a `analysis_core.json` y comparte
su SHA-256 `856f954f...`.

El amendment materializado en primario y replay es byte-idéntico al canónico
`wave56_stage1_authority_matrix_finalization_amendment_v8.json`, SHA-256
`76de3fd1e20ed57914d371696bf390f1c05c881898caeef5a99b36719c6c798c`.

## Cadena transaccional y aislamiento

Los receipts registran el siguiente orden UTC:

| Corrida | PREPARED | FIT/train | SELECT/val | COMPLETE/lockbox |
|---|---:|---:|---:|---:|
| Primario | `19:54:19` | `19:55:08` | `19:55:38` | `19:56:08` |
| Replay | `19:56:59` | `19:57:26` | `19:57:56` | `19:58:26` |

Cada materialización declara `4992` filas y
`other_splits_materialized:false`. Los access receipts declaran
`benchmark_root_received:false`, UID/GID efectivo `65534`,
`no_new_privileges:1`, capabilities `0000000000000000` y tres probes sellados
denegados por fase. Los hashes de módulos registrados por los workers coinciden
con el worktree auditado, entre ellos:

- runner transaccional `run_wave56_contextual_gate.py`: `a9f2cd4e...`;
- cálculo analítico `run_wave56_retrospective.py`: `51703287...`;
- phase worker: `581800f9...`;
- infer worker: `34a84c3c...`;
- oracle materializer: `f214db80...`;
- feature builder contextual: `474d3274...`.

La separación declarada también es materialmente coherente en los artefactos:
FIT usa train, SELECT usa val y ADJUDICATE usa lockbox. El chequeo congelado de
pair tokens informa `768` tokens en FIT, `768` en SELECT y `1920` históricos,
con seis comparaciones de intersección y overlap `0` en todas.

El intento PREPARED reemplazado quedó preservado como
`wave56_contextual_gate_fresh_v1.superseded_20260904T195348106985Z`; el receipt
primario lo referencia explícitamente. El origen fallido comprometido también
permanece bajo el basename declarado. Esta auditoría no abrió su contenido
sellado.

## Reproducibilidad del replay

El replay receipt final da `all_exact:true` para los diez objetos obligatorios:
core y arrays de FIT, schema y bundle de FIT; core, arrays y bundle de SELECT;
y core, result arrays y sealed-monitor bundle de ADJUDICATE. La verificación
independiente no se limitó al receipt: comparó arrays cargados y obtuvo cero
diferencias en `61/61`, `67/67` y `303/303` arrays.

La preparación declara `all_exact:true` en `23/23` checks, incluidos benchmark
público, freezes, amendment y los nueve logits. Los hashes byte-exactos de los
nueve logits entre primario y replay son:

| Seed | train | val | lockbox |
|---:|---|---|---|
| 17 | `fda8cabc...` | `48a516ef...` | `e4bdb3a6...` |
| 29 | `5409fd1d...` | `d8417d7b...` | `a9f7a10d...` |
| 43 | `bc8bc8ff...` | `c5b817a7...` | `97875f3e...` |

Los manifiestos finales no deben ser byte-idénticos porque incorporan receipts
contextuales y el replay agrega sus dos comprobantes. En el primario verifiqué
hash directamente para 52 de 56 entradas y en el replay para 54 de 58, sin
discrepancias. Las cuatro entradas restantes por corrida —el archivo de
commitments semánticos y los tres JSONL de labels autorizados— se limitaron a
metadata de manifest; no abrí ni hasheé su contenido. Los paths sellados y el
escrow están excluidos explícitamente del manifest final.

## Selección congelada

FIT informa `299` tokens primarios y `1114` filas de desacuerdo; SELECT,
`299` y `1051`. En validation, el gate contextual elige `q=0.6`, threshold
`0.1589291943`, coverage `0.0585284`, accuracy `0.8320792`, compatibilidad
`0.9499721` y regret `0.1141885`. El brazo advantage-only elige `q=0.9`; el
escalar elige `hard_only`. Los dos shards vuelven a elegir `q=0.6` y conservan
los mismos signos monitor que el selector completo, de modo que
`selector_sensitive:false`.

Los cuatro resúmenes seleccionados de cada familia recalculados desde
`selection_arrays.npz` coinciden exactamente con `selection_core.json`. Esto
comprueba la materialización de la selección; no convierte la estabilidad de
dos shards de un único draw en estabilidad externa.

## Resultado agregado en monitor

La población primaria predeclarada contiene `301` pair tokens
`NEAR_RIVAL` con cardinalidad `>=2` y `1175` decisiones hard/posterior en
desacuerdo.

| Brazo | Accuracy | Compatible | Regret | Worst regret | Coverage |
|---|---:|---:|---:|---:|---:|
| hard-set | 0.837209 | 0.930233 | 0.132521 | 0.392857 | 0 |
| scalar advantage | 0.837209 | 0.930233 | 0.132521 | 0.392857 | 0 |
| contextual | 0.820598 | 0.951550 | 0.118563 | 0.420819 | 0.069906 |
| advantage-only | 0.822813 | 0.935770 | 0.133490 | 0.421650 | 0.026163 |
| promedio shuffled | 0.825941 | 0.939120 | 0.127346 | 0.436157 | — |
| pure joint | 0.759828 | 0.964978 | 0.132740 | 0.468992 | 0.162652 |
| oracle positive-gain | 0.872785 | 0.968577 | 0.083299 | 0.314230 | 0.043605 |

El contextual realiza `505` overrides sobre `301 × 24 = 7224` decisiones:
`199` beneficiosos, `294` perjudiciales y `12` neutrales. La precisión entre
overrides no neutrales es `0.403651`, el recall de overrides beneficiosos es
`0.631746` y la ganancia media sobre todos los overrides es `0.199670`. Este
último promedio positivo convive con más overrides perjudiciales que
beneficiosos porque la magnitud media beneficiosa (`1.109296`) supera la
magnitud media del daño (`0.407880`). No corresponde convertir esa asimetría de
magnitudes en una afirmación causal fuera del monitor.

Los contrastes pareados principales, contextual menos referencia, son:

| Referencia | Δ accuracy [IC95] | Δ compatible [IC95] | Δ regret [IC95] | Δ worst regret [IC95] |
|---|---:|---:|---:|---:|
| hard-set | -0.016611 [-0.028931, -0.004291] | +0.021318 [+0.011905, +0.032254] | -0.013958 [-0.026359, -0.002757] | +0.027962 [+0.001938, +0.052602] |
| advantage-only | -0.002215 [-0.014812, +0.010659] | +0.015781 [+0.007060, +0.025886] | -0.014927 [-0.026636, -0.004406] | -0.000831 [-0.027132, +0.024917] |
| shuffle promedio | -0.005343 [-0.014203, +0.003488] | +0.012431 [+0.005509, +0.019934] | -0.008783 [-0.017543, -0.000724] | -0.015338 [-0.035494, +0.003654] |
| pure joint | +0.060770 [+0.047339, +0.074612] | -0.013427 [-0.020349, -0.007060] | -0.014177 [-0.023614, -0.004775] | -0.048173 [-0.079464, -0.017435] |

La mejora de regret medio frente al hard-set no domina el riesgo: `worst
regret` empeora en `0.027962` y su IC95 queda completamente sobre cero. Esa
tensión media/cola es una observación del monitor y debe mantenerse visible en
cualquier lectura posterior.

## Patrón diagnóstico predeclarado

La evaluación final del replay registra:

| Condición | Estado | Evidencia relevante |
|---|---|---|
| 1. Regret vs hard-set | `true` | Δ `-0.013958`, IC95 superior `-0.002757` |
| 2. Accuracy y compatibilidad no inferiores vs hard | `false` | compatibilidad pasa; accuracy tiene IC95 inferior `-0.028931 < -0.01` |
| 3. Regret vs scalar y advantage-only | `true` | ambas magnitudes y límites superiores satisfacen criterio |
| 4. Regret vs shuffled | `false` | IC95 superior `<0`, pero reducción `0.008783 < 0.01` |
| 5. Accuracy y regret vs pure joint | `true` | ambos IC95 satisfacen criterio |
| 6. Selector estable + replay exacto | `true` | `selector_sensitive:false`; replay `10/10` exacto |

Por tanto, `prospective_pattern_observed:false`. El `diagnostic_outcome.json`
del primario conserva correctamente la condición 6 como pendiente, mientras
el del replay la cierra. Para consumo final deben leerse conjuntamente el
outcome y el replay receipt del replay; el primario aislado no contiene el
cierre terminal.

Los cinco conjuntos ausentes heredados de Ola 54 tienen soporte `0`, debajo del
mínimo `30`, y permanecen `NOT_EVALUABLE`. No hay base para una afirmación sobre
ellos ni para imputar su comportamiento desde los slices presentes.

## Diagnóstico por seed

Stage 1 decide con el promedio exacto de logits de seeds 17/29/43; el error
máximo de `ensemble_logits - mean(per_seed_logits)` es `0`. El protocolo no
define una política contextual final separada por seed. Para cubrir la
verificación por seed sin inventar un endpoint confirmatorio, calculé sólo un
diagnóstico suplementario de reconstrucción del hard-set con el mismo
`tau=0.5` y el mismo fallback no vacío que usa el código:

| Logits | Exact-set, 768 tokens | Hamming | Exact-set, 301 primarios | Hamming primario |
|---|---:|---:|---:|---:|
| seed 17 | 0.536458 | 0.826172 | 0.631229 | 0.881229 |
| seed 29 | 0.545573 | 0.829753 | 0.564784 | 0.868771 |
| seed 43 | 0.520833 | 0.818685 | 0.561462 | 0.861296 |
| ensemble | 0.537760 | 0.827148 | 0.594684 | 0.875415 |

Estos números provienen de `result_arrays.npz` SHA `4608180c...`; son una
auditoría descriptiva no preregistrada, sin IC, y no deben presentarse como
performance por seed del gate contextual.

## Diagnóstico por slice

Los slices siguientes son descriptivos, all-in-catalog y no intervienen en la
selección. No traen intervalos propios, por lo que diferencias entre celdas no
deben interpretarse como efectos confirmados.

| Slice | n | Accuracy | Compatible | Regret | Worst regret |
|---|---:|---:|---:|---:|---:|
| FAR, card 1 | 146 | 0.710616 | 0.710616 | 0.361729 | 0.496575 |
| FAR, card 2 | 84 | 0.569444 | 0.762401 | 0.391576 | 1.032738 |
| FAR, card 3 | 82 | 0.831301 | 0.945122 | 0.117547 | 0.531504 |
| FAR, card 4 | 72 | 0.958912 | 1.000000 | 0.013696 | 0.060185 |
| NEAR, card 1 | 83 | 0.398092 | 0.398092 | 0.752385 | 0.858434 |
| NEAR, card 2 | 88 | 0.679924 | 0.883996 | 0.243490 | 0.762311 |
| NEAR, card 3 | 64 | 0.791667 | 0.931641 | 0.143609 | 0.649740 |
| NEAR, card 4 | 149 | 0.916107 | 1.000000 | 0.034023 | 0.120805 |

La heterogeneidad por cardinalidad es grande. En particular, FAR/card 2 tiene
worst regret `1.032738`, y NEAR/card 1 regret `0.752385`; pero la población
primaria es sólo NEAR con cardinalidad 2–4. Esos slices externos al primario
sirven como mapa de riesgo, no como refutación o confirmación adicional del
endpoint principal.

## Findings y riesgos

### Alta severidad

Ningún finding de integridad o reproducibilidad de alta severidad.

### Media severidad

1. **El patrón prospectivo no se satisface.** Fallan dos de seis condiciones
   predeclaradas. Es un resultado experimental, no un bug, pero bloquea toda
   formulación de validación conjunta.
2. **Trade-off media/cola frente al hard-set.** El regret medio mejora, pero la
   accuracy empeora más allá del margen preregistrado y el worst regret empeora
   con IC95 completamente positivo. Una síntesis centrada sólo en regret medio
   sería materialmente incompleta.
3. **Validez externa no demostrada.** Es un fresh draw de la misma ley sintética,
   con catálogo fijo de 24 políticas. No autoriza transporte a otra ley,
   utilidad natural ni promoción de una PPU.

### Baja severidad

1. **Cierre terminal distribuido.** El primario conserva estado pre-replay; el
   resultado final vive en el outcome y receipt del replay. Es coherente con la
   inmutabilidad del primario, pero un consumidor que lea sólo el directorio
   primario verá `PENDING_EXACT_REPLAY`/`null`.
2. **Ausencia no evaluable.** Los cinco conjuntos ausentes tienen soporte cero;
   no se puede trasladar hacia ellos ningún resultado de Stage 1.
3. **Slices sin inferencia propia.** Las celdas descriptivas muestran
   heterogeneidad importante pero no incluyen intervalos por slice; deben
   conservarse como diagnóstico y no como ranking estable.

## Comandos CPU y controles realizados

Se usaron únicamente comandos read-only (`jq`, `sha256sum`, `cmp`, `find`,
`git show/status`) y scripts Python inline con NumPy para:

- recomputar promedios de métricas desde los arrays por token;
- regenerar los 5.000 bootstraps pareados desde `bootstrap_indices` y comparar
  media, percentiles 2.5/97.5 y fracción positiva;
- recomputar métricas de 8 slices y 24 políticas;
- verificar el ensemble como media exacta de los tres seeds;
- comparar los arrays primario/replay con soporte dtype-safe para NaN;
- verificar que los archives FIT/SELECT embebidos son idénticos a sus fuentes;
- verificar tamaños y hashes permitidos contra ambos manifests, excluyendo de
  lectura los cuatro objetos sensibles indicados arriba.

También se inició, con `CUDA_VISIBLE_DEVICES=''`, un pytest focal de
`test_wave56_contextual_gate.py` y `test_wave56_prospective.py`. Se interrumpió
para no prolongar una comprobación no material después de `13 passed in
109.44s`; exit `2` por `KeyboardInterrupt`. Ese intento parcial se descarta y
no se usa como evidencia de suite. No queda proceso activo.

El runtime congelado declara Python `3.13.5`, NumPy `2.3.5`, SciPy `1.17.0`,
scikit-learn `1.8.0` y `device:"cpu"`, exacto entre primario y replay.

## Dictamen

`PASS-CON-RIESGOS`: los artefactos son internamente consistentes, la cadena de
fases respeta la autoridad declarada, la recomputación numérica coincide y el
replay exacto cierra correctamente. Los riesgos no exigen rehacer la ejecución,
pero sí obligan a registrar el resultado prospectivo como no satisfecho, el
trade-off de accuracy/worst-regret, la ausencia no evaluable y el alcance
sintético limitado. Este dictamen no constituye `GO/NO-GO`, no promueve una
arquitectura y no abre una fase nueva.

