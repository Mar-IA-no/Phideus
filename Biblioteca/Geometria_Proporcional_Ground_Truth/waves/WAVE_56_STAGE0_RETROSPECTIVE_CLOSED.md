# Ola 56 Stage 0 — cierre retrospectivo de la compuerta residual contextual

> **Estado:** `CLOSED / RETROSPECTIVE-OPENED-DEVELOPMENT / EXACT-REPLAY / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Plan:** `WAVE_56_CONTEXTUAL_RESIDUAL_GATE_PLAN.md`
> **Código ejecutado:** `9d6f351fe5167207907324f7ca076c8a97fb4bf4`

## Qué se puso a prueba

La Ola 55 mostró que un umbral escalar sobre la ventaja esperada no separaba de
forma estable cuándo convenía abandonar la acción dura y adoptar la acción
bayesiana del posterior conjunto. Stage 0 de la Ola 56 preguntó si una
compuerta residual regularizada y de baja capacidad podía estimar ese beneficio
usando contexto disponible en inferencia: riesgo y margen de las acciones,
entropía y concentración del posterior, cardinalidad, masa del conjunto duro,
dispersión entre seeds y utilidades de la política.

El análisis fue retrospectivo sobre material de desarrollo ya abierto. Usó
`decision_select` de la Ola 55 como `dev_fit` y su monitor abierto como
`dev_eval`, sin mezclarlos para entrenar. Comparó Ridge, Huber y logística
contextuales, un Ridge que sólo veía advantage, la compuerta escalar anterior y
cinco controles con targets barajados dentro de `(policy_index,d_t)`. La unidad
inferencial fue el token y las veinticuatro políticas permanecieron juntas en
folds y bootstrap.

## Integridad y reproducción

La corrida primaria y el replay coincidieron exactamente en
`analysis_core.json`, `selection_freeze.json`, `feature_schema.json` y
`result_arrays.npz`. Se preservaron inputs, features, folds, pesos, candidatos,
modelos fold-locales y full, curvas, acciones, nulls, leave-policy-group-out e
índices de bootstrap. La auditoría independiente R320 reprodujo selección,
constraints, cinco mil bootstraps y sensibilidades, y emitió `PASS` sin
findings materiales.

La población primaria reunió `300` tokens y `1082` filas de desacuerdo en
`dev_fit`, y `302` tokens con `1031` filas en `dev_eval`. Los mínimos
predeclarados se cumplieron.

## Resultado principal

La selección retrospectiva eligió `ridge_contextual`, `alpha=1.0`, con
`q=0.70`. En `dev_eval` obtuvo:

| Métrica | Compuerta contextual | Política dura | Delta contextual−dura |
|---|---:|---:|---:|
| Accuracy | `0.826987` | `0.831126` | `-0.004139` |
| Compatibilidad | `0.949917` | `0.937362` | `+0.012555` |
| Regret | `0.117883` | `0.129622` | `-0.011739` |

La reducción de regret frente a hard fue estadísticamente compatible con una
mejora: IC95 `[-0.022260,-0.002414]`. La compatibilidad también aumentó, IC95
del delta `[+0.005791,+0.021109]`. El delta de accuracy tuvo IC95
`[-0.015315,+0.006485]`: el punto respetó la tolerancia retrospectiva, pero la
incertidumbre todavía incluye una pérdida mayor que la no-inferioridad
prospectiva fijada para Stage 1.

## Lo que el resultado no identifica

La compuerta contextual no superó de manera concluyente sus controles más
informativos:

| Contraste de regret | Delta | IC95 |
|---|---:|---:|
| contextual − advantage-only | `-0.006921` | `[-0.016303,+0.001782]` |
| contextual − promedio shuffled | `+0.000676` | `[-0.004857,+0.006117]` |

El punto frente al null barajado es incluso levemente peor, aunque indistinguible
de cero. De los `302` overrides de `dev_eval`, `144` fueron beneficiosos, `157`
perjudiciales y uno neutro. La precisión de override fue `47.84%`; las
correlaciones score-gain fueron modestas (`Pearson=0.214`, `Spearman=0.176`).
Los coeficientes muestran que Ridge usa estructura contextual además de
advantage, pero eso no prueba que esa estructura sea target-specific.

La sensibilidad all-in-catalog conservó la familia Ridge, cambió su
regularización de `alpha=1` a `alpha=100` e invirtió `4/10` signos
predeclarados. El efecto observado depende de la población `NEAR_RIVAL` y de la
composición de políticas; no debe generalizarse al catálogo completo.

## Implicancia experimental

Stage 0 resolvió una función limitada pero útil: eligió una única receta
prospectiva suficientemente barata para intentar falsarla con datos frescos.
No promovió una arquitectura. La prueba siguiente mantiene congelados
`ridge_contextual`, `alpha=1.0`, el schema de diecisiete features y el protocolo
de decisión; separa físicamente `gate_fit`, `gate_select` y `sealed_monitor`.

La lectura prospectiva debe ser exigente. Para distinguir valor contextual de
una mejora genérica o accidental, la candidata necesita superar no sólo a la
política dura, sino también al gate escalar, al Ridge advantage-only y al null
barajado, preservando accuracy y compatibilidad. Si no lo hace, la explicación
más económica será que el contexto disponible no ordena de manera estable el
beneficio decisional bajo esta ley y esta representación.

## Alcance

La evidencia es retrospectiva, CPU-only y pertenece a la misma ley sintética y
al catálogo fijo de políticas de las Olas 54–55. No prueba transferencia de
aparato, utilidad natural, geometría física ni una PPU. Toda promoción y todo
GO/NO-GO científico permanecen bajo decisión del usuario.

## Artefactos

- `data/geometria_proporcional/wave56_contextual_gate_retrospective_v1/`
- `data/geometria_proporcional/wave56_contextual_gate_retrospective_v1_replay/`
- `agent_reports/317_wave56_stage0_implementation_independent_audit.md`
- `agent_reports/318_wave56_stage0_implementation_focal_reaudit.md`
- `agent_reports/319_wave56_stage0_implementation_final_audit.md`
- `agent_reports/320_wave56_stage0_result_independent_audit.md`
