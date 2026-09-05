# Ola 58 — auditoría independiente de resultados canónicos

## Dictamen: PASS

Los resultados primario y replay de Wave 58 satisfacen el protocolo aceptado. Los bindings, manifests, freezes, rosters, replay legacy, replay científico, métricas, soportes, intervalos y nominación recomputan sin divergencias. No hay decisión científica ni promoción arquitectónica encubierta.

El PASS no convierte el diagnóstico en evidencia prospectiva. La auditoría encontró degeneraciones interpretativas importantes —en particular, el ID nominado `JOINT` es exactamente la misma política que su variante `SEQUENTIAL`— que acotan qué arquitectura puede proponerse para una realización futura. Esas observaciones no invalidan la ejecución y quedan documentadas aquí.

## Corpus y bindings auditados

- Plan: `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md`, SHA-256 `d7fbbad633a3f03d37de46ad505dbcd330ac9ffa5544838ea2c4707a3ac6ba69`.
- Reauditoría de implementación R413: `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/413_wave58_open_model_class_implementation_reaudit.md`, SHA-256 `e44a6b9aff8ea72c72ed0d38fc02b62e893f00c1f2560379bdabcd751d0fdb9f`.
- Config congelada y copia idéntica en ambos outputs: SHA-256 `1f822c1161a5da4edf5039d6ad464d9ee1a39cdde69c7226460def2b4c854864`.
- Primaria: `data/geometria_proporcional/wave58_open_model_class_diagnostic_v1/`.
- Replay: `data/geometria_proporcional/wave58_open_model_class_diagnostic_v1_replay/`.

La config liga el plan y R411, los siete inputs, nueve fuentes y el commit de implementación `d558d98d14aa574254b1f66f4b0ca9c366744ca9` (`config.json:9-48`). El commit existe y es ancestro del `HEAD` auditado `41a333ccf0112bbf6c7cdafefa36b9d56108f045`; los nueve hashes de fuente actuales igualan la config y ambos manifests. Los siete hashes de input actuales igualan config, `input_sha256_before` e `input_sha256_after` (`manifest.json:59-105`).

Para cada output se recalcularon SHA-256 y tamaño de todos los archivos listados: los once registros del manifest coinciden con los once archivos distintos del propio manifest, sin faltantes ni extras. Los hashes del plan y auditoría están ligados transitivamente por `config.json`, cuyo hash está registrado en cada manifest.

## Integridad de fases y separación observable

El freeze FIT declara `fit-complete-before-validation-access`, liga el train exacto y reproduce los hashes de `model_states.json` y `model_scores.npz` (`fit/fit_freeze.json:2-6`). El freeze SELECT contiene una copia exacta de ese freeze, liga su SHA, reproduce los hashes de `selection_grids.json` y `selection_arrays.npz` y declara `selection-complete-before-open-monitor-access` (`select/selection_freeze.json:2290-2300,3639-3641`).

La secuencia también es observable en los metadatos de creación de la primaria: FIT `22:18:50`, SELECT `22:19:22`, análisis final `22:19:40` del 2026-09-04. Esos mtimes son evidencia auxiliar, no autoridad criptográfica; la autoridad son los freezes ligados y el replay idéntico. Los runtimes registran tres fases separadas, UID/GID `65534`, cuatro hilos y CPU: primaria `fit=2.104 s`, `select=31.504 s`, `monitor=18.317 s`, RSS máxima `1,067,512 KiB`; replay `2.103/31.246/18.268 s`, RSS `1,066,704 KiB`. Ambos quedan dentro de los límites planificados.

## Replay e inventarios

El replay declara y esta auditoría recomputó igualdad exacta de los diez artefactos científicos:

1. `config.json`;
2. `fit/model_states.json`;
3. `fit/fit_freeze.json`;
4. `fit/model_scores.npz`;
5. `select/selection_grids.json`;
6. `select/selection_freeze.json`;
7. `select/selection_arrays.npz`;
8. `analysis.json`;
9. `REPORT.md`;
10. `scores_and_masks.npz`.

Los siete documentos de texto son byte-identical y los tres NPZ son idénticos clave por clave y array por array. `runtime.json` y `manifest.json` son las únicas exclusiones declaradas (`replay/runtime.json:9-22`; ambos `manifest.json:78-93`).

`model_states.json` contiene los doce modelos esperados y todos están `PASS`: Ridge gain, HGB gain, ocho Logistic y dos HGB guards. Los tres HGB declaran `transport_only=true`, `score_authority=preserved_float64_scores_per_split`, `n_iter=100`; sus hashes de score train coinciden con los arrays preservados. Ningún NPZ contiene dtype object.

El corpus conserva 36/36 candidatos canónicos y 24/24 probes históricos, todos `PASS` (`analysis.json:candidate_counts`, `canonical_candidates[*].status`, `historical_probes[*].status`; `REPORT.md:7-11`). Las grillas completas suman `5,976` celdas canónicas y `5,939` históricas. Sus cardinalidades corresponden exactamente a cada selector y producto: canónico `10/64/82/568` según secuencial/joint y uno/dos guards; histórico `9/65/513` para P1, `57` para P2, `505` para P3 y `64/568` para P4. Cada grilla tiene un único terminal hard y cada selección congelada coincide con una celda de su grilla.

## `LEGACY-W57`

El replay legacy conserva la grilla de siete cuantiles `[0.1,0.2,0.3,0.4,0.5,0.6,0.8]`, más un único terminal, y el proposer conserva sus siete cuantiles más terminal. La selección congelada usa proposer `q=0.8`, threshold exacto `0.37195414707872076`, guard `q=0.4`, threshold `0.27995488763446674`, soporte de propuesta `210 filas/53 tokens` y autorización `84 filas/25 tokens` (`analysis.json:legacy_wave57.selection`).

Las `34/34` comprobaciones archivadas son verdaderas (`REPORT.md:11`; `analysis.json:legacy_wave57.checks`). Esta auditoría volvió a ejecutar `verify_legacy` desde los estados/scores publicados y las referencias Wave 57: reprodujo `34/34`, incluidos estados, scores de ambos splits, thresholds, propuestas, autorizaciones, acciones y seis arrays de métricas por split.

## Recomputación de métricas, soportes, IC y nominación

Se recorrieron los 60 brazos por dos splits, `120` combinaciones candidato×split. Para cada una se verificó:

- `actions == where(authorized, posterior_actions, hard_actions)`;
- soporte token-wise desde la máscara `authorized` y población primaria;
- medias de las métricas archivadas;
- delta pareado contra hard;
- IC percentil `[2.5,97.5]` mediante las matrices bootstrap preservadas.

No hubo divergencias. Validation usa una única matriz `(5000,302)` con índices en `[0,301]`; monitor una `(5000,306)` con índices en `[0,305]`. En ambos casos los tokens son únicos y lexicográficamente ordenados. Hard y oracle-positive-gain también recomputan exactamente en ambos splits.

La auditoría reconstruyó independientemente las reglas de soporte, márgenes, dominancia de ocho coordenadas y orden total de nominación. Obtuvo los mismos `21` elegibles, el mismo frente Pareto de `21` integrantes y el mismo nominado:

`C-HGB-HGB-INCOMPATIBILITY-JOINT`.

Sus valores exactos son:

| Split | Soporte | Accuracy | Compatibilidad | Regret | Worst regret |
|---|---:|---:|---:|---:|---:|
| validation | 45 | 0.8421633554 | 0.9456401766 | 0.1145833333 | 0.3750000000 |
| monitor abierto | 55 | 0.8476307190 | 0.9494825708 | 0.1081268155 | 0.3828976035 |

Contra hard, los deltas e IC95 token-wise relevantes son:

| Split | Métrica | Delta | IC95 |
|---|---|---:|---:|
| validation | accuracy | -0.0035871965 | [-0.0117273731, 0.0040011038] |
| validation | compatibilidad | +0.0049668874 | [+0.0001379691, +0.0100717439] |
| validation | regret | -0.0054267844 | [-0.0123137417, +0.0012877116] |
| validation | worst regret | +0.0080022075 | [-0.0074503311, +0.0234547461] |
| monitor abierto | accuracy | +0.0063997821 | [-0.0023148148, +0.0152505447] |
| monitor abierto | compatibilidad | +0.0091230937 | [+0.0009497549, +0.0166122004] |
| monitor abierto | regret | -0.0144902869 | [-0.0238181963, -0.0051059255] |
| monitor abierto | worst regret | -0.0024509804 | [-0.0190631808, +0.0136165577] |

Son intervalos `CONDITIONAL / ADAPTIVE / POST-SELECTION`; no miden variación entre draws ni arbitran la nominación (`analysis.json:bootstrap`, `config.json:71-79`).

## Findings interpretativos priorizados

No hay findings P0 ni P1.

### P2 — el selector del ID nominado no está identificado

`C-HGB-HGB-INCOMPATIBILITY-JOINT` y `C-HGB-HGB-INCOMPATIBILITY-SEQUENTIAL` seleccionaron exactamente los mismos thresholds (`q_p=0.8`, `p=0.37786739818476994`, `q_g=0.9`, `g=0.05031062804095663`) y tienen propuestas, autorizaciones, acciones, métricas, soportes e IC idénticos en ambos splits. Sólo difieren el selector nominal y el `cell_index` de su grilla. Ambos pertenecen al frente Pareto; `JOINT` gana finalmente por el desempate lexicográfico del ID, no por evidencia de superioridad del selector.

Por tanto, la inferencia admisible no es “JOINT superó a SEQUENTIAL”. El diagnóstico nomina una política compartida de proposer HGB + guard HGB de incompatibilidad posterior; deja sin identificar cuál de esos dos procedimientos de selección conviene congelar prospectivamente.

### P2 — el frente Pareto no redujo los elegibles y existen degeneraciones extensas

Los `21` candidatos elegibles son también los `21` integrantes del frente: la dominancia no eliminó ninguno. La nominación proviene enteramente del orden total posterior, comenzando por regret de monitor abierto. Entre los 36 IDs hay sólo `19` firmas distintas de propuestas+autorizaciones+acciones sobre ambos splits; `17` IDs son redundantes con otro. Nueve candidatos de incompatibility colapsan en una única política `HARD_ONLY`, y hay otras siete clases de equivalencia multi-ID de tamaños `3,3,2,2,2,2,2`.

Esto no es un bug: los terminals y empates estaban protocolizados. Sí impide interpretar cada ID como una arquitectura empíricamente distinta y aconseja que el prospectivo evite multiplicar brazos que materializan la misma política.

### P2 — la señal arquitectónica más defendible es una interacción acotada, no una victoria global de HGB

En el guard-set puro `INCOMPATIBILITY`, las tres variantes HGB-proposer/Logistic-guard, las tres Ridge-proposer/HGB-guard y las tres Ridge/Logistic terminan `HARD_ONLY`; sólo HGB-proposer + HGB-guard produce políticas activas. Dentro de este draw abierto, eso localiza la no identidad en la combinación de no linealidad del proposer y del estimador de incompatibilidad, no en uno de ellos por separado.

La lectura tampoco es “esa combinación domina toda alternativa”. El frente conserva 21 candidatos. Por ejemplo, `C-HGB-HGB-HARM-JOINT` tiene peor regret medio de monitor que el nominado (`0.1092842` frente a `0.1081268`), pero una mejora de worst regret mucho mayor (`-0.0253268`, IC95 `[-0.0492919,-0.0029956]`) mientras el IC de worst regret del nominado cruza cero. La nominación sigue correctamente el criterio congelado de regret medio; el trade-off de cola permanece real.

La propuesta prospectiva con mayor poder diagnóstico sería contrastar al menos la política HGB/HGB-incompatibility compartida por JOINT/SEQUENTIAL contra una candidata HGB/HGB-harm orientada a cola, sin afirmar todavía promoción arquitectónica ni éxito científico.

### P3 — los hashes de plan/auditoría están ligados de forma transitiva, no duplicados como campos top-level del manifest

Cada manifest registra el hash de `config.json`, y esa config contiene los hashes y paths de plan y R411. La cadena es verificable y no deja un hueco de integridad. Sin embargo, el manifest no repite esos dos hashes como campos top-level, pese a la formulación literal del plan (`WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:354-357`). Es una diferencia menor de esquema, no una razón para invalidar resultados ya ligados criptográficamente.

## Frontera de autoridad

`analysis.json` conserva `scientific_decision=null`, `architecture_promoted=false`, estado `COMPLETE` y scope literal `OPEN-DATA / ADAPTIVE / SELECTED-AFTER-MONITOR-INSPECTION` (`analysis.json:2,9569-9636`). `REPORT.md:3-5,100-102` repite que el monitor estaba abierto y que el resultado no valida ni promueve una arquitectura ni declara `GO/NO-GO`.

La ejecución es válida como diagnóstico retrospectivo y generador de la candidata/familia a probar. La elección científica y cualquier promoción siguen perteneciendo al usuario después de una realización futura independiente.
