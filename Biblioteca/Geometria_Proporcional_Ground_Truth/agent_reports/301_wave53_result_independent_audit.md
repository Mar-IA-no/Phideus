# R301 — auditoría independiente del resultado de Ola 53

> **Rol:** auditor independiente con contexto mínimo
> **Fecha:** 2026-09-03
> **Resultado:** `REVISE` interpretativo; ejecución y patrón válidos

## Informe verbatim

**Veredicto: REVISE**
La ejecución, trazabilidad y adjudicación `false` son correctas. La revisión requerida afecta dos interpretaciones materiales, no invalida el experimento.

**Findings**
1. **MEDIA: “raw supera marginal en regret” es demasiado fuerte.** Raw tiene mejores estimaciones puntuales: regret `0.11841` vs `0.12235` y accuracy `0.78238` vs `0.77675`. Pero sólo la ventaja de accuracy queda respaldada por el bootstrap: marginal−raw `-0.00563`, IC95 `[-0.01014,-0.00113]`. Para regret, marginal−raw es `+0.00394`, IC95 `[-0.00038,+0.00887]`; el intervalo cruza cero. Debe decirse: **raw mejora accuracy y presenta menor regret puntual, sin diferencia concluyente de regret**. Locator: `summary.json`, JSONPath `$.contrasts.marginal_minus_raw`.

2. **MEDIA: `worst_regret` cambia de estimando sin cambiar de nombre.** La tabla principal reporta la **media entre tokens del máximo sobre 24 políticas**, mientras la sección selectiva reporta el **máximo global entre tokens**. Por eso aparecen `0.3694/0.4454` en la tabla, pero `1.25` a cobertura completa. El rótulo “peor regret” es incorrecto para la primera magnitud. Además, marginal empeora frente a hard el promedio del peor regret por token en `+0.0760`, IC95 `[+0.0214,+0.1256]`; esto debe acompañar la lectura de compatibilidad.

3. **BAJA: `utility_masked falla fuerte` necesita dimensión explícita.** Falla fuertemente en accuracy (`-0.4496` respecto de marginal) y regret (`+0.3144`), pero obtiene mayor compatibilidad (`0.9865` vs `0.9637`). No es un fracaso uniforme.

**Patrón Y Validez**
El `diagnostic_pattern=false` es correcto. Fallan exactamente tres condiciones preregistradas:

- Regret marginal−hard `-0.00148`, IC95 `[-0.02318,+0.01816]`: no alcanza `-0.02` ni excluye cero.
- Accuracy marginal−hard `-0.06278`: excede ampliamente la tolerancia `-0.01`.
- Accuracy marginal−shuffled `+0.04054`: positiva, pero menor que `+0.05`.

Pasan compatibilidad frente a hard (`+0.02618`), criterio Platt sólo por NLL, tolerancia de regret frente a raw, abstención 75%, `utility_masked` y replay externo.

La abstención sí ordena riesgo empíricamente: cobertura efectiva `0.7162`, regret `0.07983` frente a `0.12235` completo. No es garantía conformal. En monitor primario, las correlaciones máximas son `0.3804` raw y `0.3725` Platt; el mismatch de cardinalidad L1 permanece alto, `0.4424/0.4256`. “Hasta ~0.38” es correcto si se restringe a monitor primario; en `decision_select` llega a `0.4227`.

Trazabilidad y no-leakage pasan: hashes íntegros, splits únicos `192/192/384`, cero solapamientos también por `cluster_id`, bootstrap `2000×148` por `pair_token`, ninguna lectura de lockbox y replay exacto de 5 NPZ más 13 artefactos byte-exact.

**Siguiente Discriminante**
El contraste más informativo es mantener encoder, logits, splits, utilidades y pérdida congelados, pero reemplazar el producto Bernoulli por un **posterior conjunto regularizado sobre los 15 conjuntos no vacíos**: potenciales unary derivados de logits, seis interacciones por pares y sesgos de cardinalidad. Ajustar en `calibration_fit`, seleccionar regularización en `decision_select` y evaluar una sola vez en monitor contra raw independiente, Platt independiente y hard.

Esto discrimina directamente si la pérdida decisional proviene de la factorización independiente, sin confundirla con representación o entrenamiento. Las métricas primarias deberían incluir NLL del conjunto verdadero, cardinalidad L1, regret y accuracy. También evita calibrar marginales y luego condicionarlos a no vacío, operación que altera esos marginales.

No se editaron archivos ni se ejecutaron tests que escribieran cachés.

## Resolución del coordinador

Se corrige el contrato de nombres del runner y se regeneran primaria y replay. La lectura final distingue mejora concluyente de accuracy raw, diferencia no concluyente de regret raw y el comportamiento dimensional del control `utility_masked`. No se altera ninguna acción, pérdida, población ni criterio preregistrado.
