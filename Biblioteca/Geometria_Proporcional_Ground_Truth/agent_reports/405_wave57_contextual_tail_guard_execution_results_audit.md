# Ola 57 — auditoría de ejecución y resultados del proposer con guard de daño

**Artefacto primario:** `data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1`

**Replay:** `data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1_replay`

**Commit de autoridad observado:** `fad81bb28c4c5720854dc57976a4145c6b80a130`

**Dictamen técnico:** `PASS-CON-RIESGOS`

**Decisión científica:** reservada al usuario

## Síntesis ejecutiva

La recuperación pre-oráculo de la Ola 57, su ejecución primaria y el replay
forman una cadena técnicamente consistente. El primario y el replay transitaron
`PREPARED -> FIT_COMPLETE -> SELECT_COMPLETE -> COMPLETE` con
`device="cpu"`. La preparación del replay reprodujo exactamente `23/23`
compromisos; el cierre analítico reprodujo `13/13` objetos. Una comparación
adicional cargó `519` arrays distribuidos en quince NPZ de inferencia, FIT,
SELECT y ADJUDICATE y no encontró diferencias entre primario y replay. Los `70`
arrays de FIT y los `70` de SELECT embebidos en el resultado final también
coinciden con sus fuentes.

La recomputación numérica desde `result_arrays.npz` coincide con el reporte:
conteos exactos, `58` escalares de resúmenes, contrastes y calibración con error
máximo `0`, y bootstrap `5000 x 306` limitado a índices `0..305`. No hay un
finding de integridad, aislamiento o reproducibilidad que obligue a repetir la
corrida.

El resultado experimental exige una lectura más precisa. Separar proposer y
guard sí aporta valor incremental respecto del proposer solo: mejora accuracy
en `+0.006944`, IC95 `[+0.001362,+0.012527]`, y worst regret en `-0.028322`,
IC95 `[-0.041394,-0.016340]`, sin empeorar regret más allá del margen
predeclarado. Sin embargo, la interfaz completa no desplaza al hard-set con la
magnitud y seguridad exigidas: regret mejora sólo `-0.003835` con IC95 que cruza
cero, compatibilidad cae `-0.002451` con límite inferior `-0.008442`, y worst
regret mejora `-0.009532` pero también con IC95 que cruza cero. Por eso fallan
las condiciones 1, 2 y 3; la condición 4 pasa y la 6 pasa.

La condición 5 no puede adjudicarse. Sólo dos de cinco controles shuffled
alcanzaron simultáneamente los mínimos de Hamming global y ponderado. El
artefacto registra correctamente `NOT_EVALUABLE`, no promedia las dos réplicas
supervivientes y deja `prospective_pattern_observed:null`. Como ya existen tres
condiciones falsas, esta indeterminación no puede convertir la conjunción del
draw actual en un resultado positivo. Tampoco debe deformarse en un resultado
global `false`: el contrato preserva explícitamente la diferencia entre
falsedad y falta de evaluación.

## Alcance y régimen de lectura

Se revisaron el plan y la configuración congelados, el amendment de
recuperación, los receipts y freezes públicos, los estados de FIT/SELECT, el
reporte final, los arrays analíticos, los outcomes terminales y los manifests
de replay. La recomputación usó los arrays autorizados ya publicados; no abrió
`benchmark/sealed/**`, `generation_escrow.json`, claves privadas ni JSONL de
labels autorizados.

El alcance científico sigue siendo una realización fresca de la misma ley
sintética y un catálogo fijo de 24 políticas. El resultado no autoriza
transporte a otra ley, utilidad natural, autoridad física, una PPU ni una
promoción arquitectónica.

## Fuentes y hashes

| Fuente | SHA-256 | Función |
|---|---|---|
| `WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md` | `db0d69ba197bfa0dcf853c9cecc1f848487ccc25910424036ebcde745a1ccedd` | hipótesis, métricas y patrón preregistrado |
| `wave57_contextual_tail_guard_fresh.json` | `fb21a43cb4e356a7f10c293e5cb0c0fc037a8ecc206883e99122115b18812a83` | modelos, mínimos, grillas y criterios |
| `wave57_preoracle_pair_token_recovery_amendment_v1.json` | `8cfcb0453bd1cd19b41db518ffca6f702cee03c31966f137cd8204fbad248fc5` | autoridad de recuperación del draw único |
| primario `fit_core.json` | `188b1b9f2bb6abb133e3355d0665a0604431538d933428e008f27f804f712829` | ajuste de proposer, guard y shams |
| primario `selection_core.json` | `6802ecee56ab9fc7ad833c70500974029d1d37ac12c568f888d397474efba63e` | selección y shards |
| primario `analysis_core.json` | `f3a49287a46b13664c718aaab98f38b8ef0f198ee141dd2484214b206dd3a764` | resultados y condiciones pre-replay |
| primario `result_arrays.npz` | `8529c7787e74ad64de275d32cc8c52362283dcb41f7d81bae5ce027ba04e5064` | recomputación numérica |
| replay `preparation_replay.json` | `d7e56be9f544088e92166018da6665b7fa6a475663ee069be6d98635046e762e` | exactitud `23/23` |
| replay `replay_receipt.json` | `fc4380e37c42eb68b1b21b660d9dc8558c4801892e7ad24376cea72ccff72136` | exactitud analítica `13/13` |
| replay `diagnostic_outcome.json` | `8a378b472baf0b43bf0b33c5f6b4b02229749d789f4352a963614d1c0711c6ce` | cierre terminal de condiciones |

`REPORT_WAVE57.json` es byte-idéntico a `analysis_core.json` y comparte SHA-256
`f3a49287...`.

## Recuperación, aislamiento y replay

El draw de origen permanece preservado como
`wave57_contextual_tail_guard_fresh_v1.failed_20260904T215941566314Z`. La
recuperación usó el mismo escrow `d489f443...`, el mismo manifest de benchmark
`3d444db8...` y el amendment auditado `8cfcb045...`; no redibujó claves ni
poblaciones. Cada split contiene `4992` filas, `1152` pair tokens totales y
`768` elegibles.

Los workers de FIT, SELECT y ADJUDICATE registran UID/GID efectivo `65534`, no
reciben el root del benchmark, corren con `no_new_privileges=1`, capabilities
efectivas nulas y probes denegados contra los tres archivos sellados. FIT abre
sólo train, SELECT sólo val y ADJUDICATE sólo lockbox. Los hashes de los módulos
staged coinciden con las fuentes declaradas por el freeze.

El replay de preparación declara `all_exact:true` en `23/23` checks, incluidos
los nueve NPZ de logits. El receipt final declara `all_exact:true` para los
trece objetos analíticos obligatorios. La comprobación dtype-safe adicional
comparó `45` arrays de logits, `70` de FIT, `20` del bundle FIT, `70` de SELECT,
`21` del bundle SELECT, `273` de resultados y `20` del bundle monitor: `519/519`
sin diferencias.

## Selección congelada

El proposer seleccionó `q=0.8`, threshold `0.3719541471`, con `210` filas y
`53` tokens propuestos. El guard verdadero seleccionó `q=0.4`, threshold
`0.2799548876`, con `84` filas y `25` tokens autorizados. El control escalar
eligió `gamma=0.4`; advantage-only eligió `hard_only`.

Ambos shards seleccionaron proposer y guard no triviales y conservaron los
signos monitor del selector completo. Esa estabilidad es interna a dos shards
del mismo draw; no acredita estabilidad entre nuevas realizaciones.

Los cinco shams conservaron fracción permutable `0.950920`. Sus resultados
fueron:

| Sham | Hamming global | Hamming ponderado | Estado |
|---:|---:|---:|---|
| 0 | 0.298569 | 0.255173 | `PASS` |
| 1 | 0.271984 | 0.226043 | `NOT_EVALUABLE` |
| 2 | 0.278119 | 0.241843 | `NOT_EVALUABLE` |
| 3 | 0.294479 | 0.248224 | `NOT_EVALUABLE` |
| 4 | 0.302658 | 0.255320 | `PASS` |

Los tres fallos provienen del Hamming ponderado `<0.25`; no hubo relajación
post-hoc ni reselección.

## Resultado en monitor

La población primaria contiene `306` tokens, `1183` filas de desacuerdo y
`183` tokens con algún desacuerdo. Entre ellos, `135` aportan algún caso
perjudicial y `67` algún caso no perjudicial.

| Brazo | Accuracy | Compatible | Regret | Worst regret |
|---|---:|---:|---:|---:|
| hard-set | 0.841231 | 0.940359 | 0.122617 | 0.385349 |
| proposer solo | 0.839325 | 0.942947 | 0.118657 | 0.404139 |
| proposer + guard | 0.846269 | 0.937908 | 0.118782 | 0.375817 |
| advantage-only | 0.841231 | 0.940359 | 0.122617 | 0.385349 |
| pure joint | 0.772467 | 0.969227 | 0.120983 | 0.439270 |
| oracle positive-gain | 0.881264 | 0.975490 | 0.072576 | 0.291394 |

El guard ejecutó `115` overrides: `69` beneficiosos, `44` perjudiciales y `2`
neutrales. Su precisión beneficiosa fue `0.610619` y su recall beneficioso
`0.212308`. Frente al proposer, retiró suficiente daño para mejorar accuracy y
cola. Frente al hard-set, esa ganancia incremental no alcanza una mejora
conjunta robusta.

Los contrastes pareados principales son:

| Referencia | Δ accuracy [IC95] | Δ compatible [IC95] | Δ regret [IC95] | Δ worst regret [IC95] |
|---|---:|---:|---:|---:|
| hard-set | +0.005038 [+0.000408,+0.009668] | -0.002451 [-0.008442,+0.000953] | -0.003835 [-0.009021,+0.002565] | -0.009532 [-0.023427,+0.004902] |
| proposer | +0.006944 [+0.001362,+0.012527] | -0.005038 [-0.008306,-0.002315] | +0.000125 [-0.003858,+0.004676] | -0.028322 [-0.041394,-0.016340] |

La segunda fila identifica la contribución específica del guard: mejora
accuracy y worst regret sobre la máscara del proposer, pero cede
compatibilidad. La primera fila muestra que esa operación incremental todavía
no produce una política globalmente superior al hard-set bajo los márgenes
congelados.

## Calibración y soporte

La cabeza de daño obtiene Brier `0.196517` y log-loss `0.583967` sobre `1183`
filas. La frecuencia de daño crece en términos generales con el score, pero el
último decil cae a `0.7797` después de `0.9068` en el anterior. Esa forma es un
diagnóstico de calibración del draw, no evidencia de monotonía poblacional.

Los cinco support sets heredados (`0,4,8,10,12`) contienen `0` tokens de la
población primaria frente al mínimo `30`; permanecen `NOT_EVALUABLE` y no
autorizan imputación.

## Patrón preregistrado

| Condición | Estado | Evidencia decisiva |
|---|---|---|
| 1. Regret vs hard | `false` | mejora `0.003835 < 0.01`; IC95 cruza cero |
| 2. Accuracy y compatibilidad no inferiores vs hard | `false` | accuracy pasa; compatibilidad tiene límite inferior `-0.008442 < 0` |
| 3. Worst regret vs hard | `false` | media favorable; límite superior `+0.004902 > 0` |
| 4. Guard incremental vs proposer | `true` | accuracy y worst regret mejoran; regret queda dentro de margen |
| 5. Regret y worst regret vs cinco shams | `NOT_EVALUABLE` | sólo `2/5` shams pasan mínimos |
| 6. Shards estables + replay exacto | `true` | ambos shards pasan y replay `13/13` |

El outcome terminal conserva `prospective_pattern_observed:null`. No debe
publicarse como validación; tampoco como una medición completa del contraste
contra shuffled. Las tres condiciones falsas ya impiden que este draw sostenga
la conjunción positiva, aun si el control ausente hubiera sido evaluable.

## Findings y riesgos

### Alta severidad

No se encontraron fallos de integridad, aislamiento o reproducibilidad.

### Media severidad

1. **Control causal incompleto.** Tres shams fallan el mínimo ponderado. No hay
   estimación autorizada de main contra el promedio de cinco controles.
2. **Mejora incremental sin superioridad global.** El guard corrige al proposer,
   pero no alcanza los márgenes de regret, compatibilidad y cola frente al hard.
3. **Validez externa no demostrada.** Es una sola realización fresca de la
   misma ley sintética, con catálogo y modelos lineales fijos.
4. **Calibración imperfecta en la cola alta.** El último decil no conserva el
   crecimiento observado en los anteriores; no debe interpretarse como una
   probabilidad de daño externamente calibrada.

### Baja severidad

1. El cierre terminal vive en el outcome del replay; el primario conserva por
   diseño la condición 6 pendiente.
2. Los cinco conjuntos ausentes tienen soporte cero y no admiten lectura.
3. El bootstrap condiciona en un único FIT/SELECT y cinco permutaciones
   observadas; el replay verifica determinismo, no variación estadística.
4. Los metadatos congelados de preparación conservaron por herencia las fases
   `wave56-preparation-exact-replay` y
   `wave56-stage1-preparation-complete`. Es una inconsistencia nominal de
   linaje: no altera los `23/23` checks, hashes ni resultados. Los artefactos
   quedan intactos; el emisor se corrigió después del draw para etiquetar como
   Wave 57 sólo los replays futuros y conservar las etiquetas históricas de
   Wave 56.

## Oportunidad arquitectónica y próximo discriminante

La evidencia sí justifica preservar la separación entre propuesta de valor y
autorización por daño como una alternativa experimental: el guard mejora de
manera clara accuracy y worst regret respecto del proposer fijo. No justifica
promoverla como política frente al hard-set.

Un siguiente contraste tendría más poder diagnóstico si reemplaza el shuffle
aleatorio con rechazo posterior por una permutación condicional construida para
satisfacer por diseño los mínimos global y ponderado, manteniendo intactos los
estratos y la capacidad del modelo. Ese cambio debe congelarse en un draw nuevo;
no puede aplicarse retrospectivamente a la Ola 57. Además, como las condiciones
1–3 ya fallan, reparar sólo el control sham no basta: la siguiente arquitectura
debería atacar explícitamente el pequeño beneficio medio y la cesión de
compatibilidad frente al hard, por ejemplo mediante una pérdida o selector de
riesgo conjunto preregistrado. Esta es una candidata de diseño, no una
promoción ni una decisión `GO/NO-GO`.

## Dictamen

`PASS-CON-RIESGOS`: la cadena recuperada es íntegra, aislada y exactamente
reproducible; la recomputación concuerda con los artefactos. La separación
proposer/guard muestra una contribución incremental real dentro del draw, pero
la política completa no satisface tres condiciones frente al hard y el control
shuffled queda incompleto. El estado correcto es evidencia mixta e
indeterminación parcial, sin promoción arquitectónica ni decisión científica.
