# Informe: Correcciones al libro HIT derivadas de la auditoría de trazabilidad

Fecha: 2026-04-03
Origen: Auditoría de trazabilidad numérica (Claude LOCAL + Codex + Claude UNC)
Archivo del libro MD: `/mnt/m2-1TB/harmonic-information-theory/Harmonic_Information_Theory_Foundations.md`

---

## Contexto

La auditoría de trazabilidad de Phideus (55 claims verificados, 53 PASS, 1 WARN, 1 STALE) identificó varias correcciones necesarias en el libro HIT. Este informe cubre exclusivamente el Markdown del libro. El LaTeX se sincronizará después.

Los agentes exploraron exhaustivamente todo el archivo (2700+ líneas). Los resultados experimentales de Phideus aparecen en exactamente 3 ubicaciones:

| Ubicación | Líneas | Contenido |
|-----------|--------|-----------|
| Ch5 Table 5.1 | ~457 | Única referencia forward: "S = 84.1%" |
| Ch11 §11.5-11.6 | 1068-1107 | Claims principales + Tables 11.3a, 11.3b |
| Appendix B | 2025-2144 | 8 tablas con inventario cuantitativo completo |

No hay claims numéricos de Phideus en ningún otro capítulo (verificado por búsqueda exhaustiva).

---

## CORRECCIÓN 1 — CRÍTICA: d4a4 "five independent seeds" (Ch11)

### Problema

La auditoría forense demostró que d4a4 nunca tuvo training multi-seed. Lo que se reportó son 5 evaluaciones (eval-seed) sobre un único checkpoint (seed=42). Los otros 3 arms (D0, a4r, d4-a4r) SÍ tienen 5 trainings independientes reales (en UNC).

Los valores individuales documentados (83.6, 86.4, 84.0, 82.0, 84.4) son confabulación — no existen en ningún artefacto. Los valores reales del eval-seed son 83.6, 88.4, 83.0, 82.6, 82.8 (seeds 42, 123, 456, 789, 2026).

El mean 84.1% y std ±2.3pp SÍ coinciden con los eval-seed reales. El rango y los estadísticos de comparación NO.

Un training multi-seed real de d4a4 está corriendo en Mendieta (Job 1146677, 4 seeds nuevas). Cuando termine, los valores se actualizarán.

### Ubicaciones a corregir

**Línea ~1070 (Ch11 §11.5, texto principal)**

Dice:
> Across five independent seeds, d4a4 reaches a mean S of 84.1 percent with a standard deviation of 2.3 percentage points [...] The effect size is large, Cohen d = 4.50 [...] the weakest guided seed remains above the strongest unguided one

Corrección:
> Across five evaluation seeds drawn from a single trained model, d4a4 reaches a mean S of 84.1 percent with a standard deviation of 2.3 percentage points [...] the weakest evaluation seed (82.6 percent) remains above the strongest unguided training seed (77.4 percent)

Sobre el Cohen d: Las otras 3 arms (D0, a4r, d4-a4r) sí tienen training-seed real. El Cohen d de d4a4 vs D0 fue computado con valores confabulados y no es reproducible. Opciones:

- **Opción A**: Eliminar Cohen d y p-value de d4a4 hasta que el training multi-seed real termine. Mantener Cohen d para d4-a4r y a4r que sí son válidos.
- **Opción B**: Recalcular Cohen d usando los eval-seed reales (83.6, 88.4, 83.0, 82.6, 82.8) vs D0 training-seed (74.0, 77.4, 76.0, 71.8, 76.8). Esto es metodológicamente impuro (compara varianzas de distinta naturaleza) pero da una cota.
- **Opción C**: Reportar solo el single-seed d4a4=83.6% vs D0 mean=75.2% (diferencia de +8.4pp) sin estadístico inferencial.

**Recomendación**: Opción A. Es la más honesta. El training real está corriendo y reemplazará estos valores pronto.

**Línea ~1092-1099 (Table 11.3b)**

Caption dice: "Multi-seed replication of cross-modal retrieval, Escalon 1 (n = 5 seeds)"

Agregar nota al pie de la tabla:
> For D0, a4r, and d4-a4r, each seed corresponds to a fully independent training run. For d4a4, seeds correspond to evaluation pool randomization over a single trained model (eval-seed). A complete training-seed replication for d4a4 is in progress; values will be updated upon completion.

Fila d4a4 (línea ~1096):

| Campo | Valor actual (INCORRECTO) | Valor correcto |
|-------|--------------------------|----------------|
| Range | 82.0–86.4 | 82.6–88.4 (eval-seed) |
| Cohen d | 4.50 | eliminar o marcar "pending" |
| p-value | p < 0.001 | eliminar o marcar "pending" |
| Mean | 84.1% | 84.1% (correcto) |
| SD | 2.3pp | 2.3pp (correcto) |
| Delta | +8.9pp | +8.9pp (correcto) |

**Línea ~1084 (Table 11.3a)**

"Multi-seed replication | 84.1% +/- 2.3pp | ..."

Agregar calificación: "84.1% ±2.3pp (eval-seed; training-seed replication in progress)"

O alternativamente, si Cohen d se elimina de la tabla 11.3b, también eliminar de la 11.3a la referencia a "Cohen d = 4.50" si aparece ahí.

### Referencia forward

**Línea ~457 (Ch5 Table 5.1)**

Dice: `| H3 | ... | Escalon 1: S = 84.1% |`

El valor 84.1% se mantiene (es correcto tanto para eval-seed como probablemente para el training-seed que viene). No requiere cambio.

---

## CORRECCIÓN 2 — MODERADA: Gate 10 Table B.6, dos valores incorrectos

### Problema

Dos valores en Table B.6 (Gate 10 mechanism sweep) no coinciden con los JSONs crudos (`results_unc/gate10_mechanism_sweep/`):

| Arm | Libro dice | JSON dice | Diferencia |
|-----|-----------|-----------|------------|
| a10d-ab | 57.2% @e30 | **57.4%** @e28 | +0.2pp, epoch incorrecto |
| a7-pca | 71.8% @e30 | **71.6%** @e29 | -0.2pp, epoch incorrecto |

### Ubicación

Appendix B, Table B.6, líneas ~2104 y ~2106.

### Corrección

- Línea ~2104: a7-pca: 71.8% → **71.6%**, epoch 30 → **29**
- Línea ~2106: a10d-ab: 57.2% → **57.4%**, epoch 30 → **28**

### Fuente de verdad

- `results_unc/gate10_mechanism_sweep/a7-pca_seed42/final_results.json` → best_S=0.716, best_epoch=29
- `results_unc/gate10_mechanism_sweep/a10d-ab_seed42/final_results.json` → best_S=0.574, best_epoch=28

### Impacto en texto

Línea ~2109 dice: "concat > FiLM/pca (+2pp) >> attention bias (+16pp)"

Con a7-pca corregido a 71.6%: best concat (76.4) - worst pca (71.6) = 4.8pp, no 2pp.

Verificar si el "+2pp" se refiere a mean-to-mean o a un par específico:
- Mean concat: (76.4+75.6+75.4)/3 = 75.8
- Mean pca: (74.0+73.2+71.6)/3 = 72.9
- Gap mean-to-mean: 2.9pp → "+2pp" es aproximación razonable pero debería decir "+3pp"

El "+16pp" probablemente es: best concat (76.4) - best ab (59.6) = 16.8pp → "+16pp" es OK como aproximación.

---

## CORRECCIÓN 3 — MODERADA: Gate 8 pca epoch incorrecto (Appendix B)

### Problema

Table B.4 (Gate 8 conditioned projections) dice pca best epoch = 30. El JSON dice epoch 25.

### Ubicación

Appendix B, Table B.4, línea ~2074.

### Corrección

- pca best epoch: 30 → **25**

### Fuente de verdad

`results_unc/gate8_conditioned_projections/a4r-pca_seed42/final_results.json` → best_S=0.826, best_epoch=**25**

### Nota

Esto coincide con la inconsistencia ya identificada en la auditoría de Phideus (`Proyecto_Estado_Actual.md` decía @e30, `BITACORA_UNC.md` decía @e25). Ahora confirmado por JSON crudo.

---

## CORRECCIÓN 4 — BAJA: Test02 shuffled sin asterisco (Appendix B)

### Problema

Table B.2 (línea ~2056) reporta shuffled = 73.6% sin nota. El JSON crudo (`results_unc/gate5b_param_matched/shuffled/final_results.json`) da best_S=0.732 (73.2%). La documentación interna de Phideus usa 73.6%* con asterisco indicando "valor operativo de convergencia temprana".

### Ubicación

Appendix B, Table B.2, línea ~2056.

### Opciones

- **Opción A**: Cambiar a 73.2% (valor raw del JSON) — más limpio para publicación.
- **Opción B**: Mantener 73.6% con nota al pie explicando la convención de convergencia temprana.
- **Opción C**: Dejar como está — la diferencia es 0.4pp y no afecta conclusiones (sigue cayendo en la banda de baseline).

**Recomendación**: Opción A (73.2%) es lo más limpio. El shuffled igual cae en banda baseline y la conclusión no cambia.

---

## PENDIENTE DE INVESTIGACIÓN: D0 hard-negative accuracy = 80.4%

### Problema

Table B.3 (línea ~2062) reporta D0 hard-negative accuracy = 80.4%. Todos los JSONs conocidos de D0 dan hard-negative accuracy > 94%. El valor 80.4% no matchea ningún artefacto disponible.

### Hipótesis

Podría venir de un baseline histórico anterior (Gate 2 foundation, pre-Gate 4.3 scratch) con un protocolo de evaluación diferente. No hay suficiente evidencia para corregirlo ni para confirmarlo.

### Recomendación

**No corregir hasta identificar la fuente.** Marcar internamente como "pending verification". Si no se encuentra fuente antes de publicación, considerar eliminar la columna hard-negative de Table B.3 o reemplazar el valor D0 con "n/r" (not reported under current protocol).

---

## CORRECCIÓN 5 — BAJA: Escalón 2 "twelve conditions" vs conteo real

### Contexto (no es error, pero vale aclarar)

Ch11 línea ~1107 dice "Twelve conditions were tested." Table B.8 muestra 12 arms descriptores + D0 = 13 filas. El "twelve" excluye D0 correctamente (D0 es baseline, no condición). Consistente. La memoria del proyecto dice "15/15 conditions ≈ D0" — esto incluye las 3 pca de P2.5b que se agregaron después del conteo original.

### Recomendación

No corregir. El "twelve" del texto es correcto para el momento en que se cerró Escalón 2. Si se quiere ser exhaustivo, cambiar a "Fifteen conditions" para incluir P2.5b, y actualizar el texto para reflejar el cierre completo.

---

## RESUMEN DE CORRECCIONES PARA CODEX

### Obligatorias antes de publicación

| # | Archivo | Línea(s) | Qué cambiar | Severidad |
|---|---------|----------|-------------|-----------|
| 1a | MD libro | ~1070 | "five independent seeds" → "five evaluation seeds drawn from a single trained model" | CRÍTICA |
| 1b | MD libro | ~1096 | Rango d4a4: 82.0–86.4 → 82.6–88.4 | CRÍTICA |
| 1c | MD libro | ~1096 | Cohen d=4.50 y p<0.001 para d4a4: eliminar o marcar "pending recalculation" | CRÍTICA |
| 1d | MD libro | ~1092 | Agregar nota al pie sobre eval-seed vs training-seed | CRÍTICA |
| 1e | MD libro | ~1084 | Calificar "84.1% ±2.3pp" como eval-seed | CRÍTICA |
| 2a | MD libro | ~2104 | a7-pca: 71.8% → 71.6%, epoch 30 → 29 | MODERADA |
| 2b | MD libro | ~2106 | a10d-ab: 57.2% → 57.4%, epoch 30 → 28 | MODERADA |
| 3 | MD libro | ~2074 | Gate 8 pca: epoch 30 → 25 | MODERADA |

### Recomendadas

| # | Archivo | Línea(s) | Qué cambiar | Severidad |
|---|---------|----------|-------------|-----------|
| 4 | MD libro | ~2056 | shuffled: 73.6% → 73.2% (o agregar nota) | BAJA |

### Pendientes de investigación

| # | Archivo | Línea(s) | Qué investigar |
|---|---------|----------|----------------|
| 5 | MD libro | ~2062 | D0 hard-neg 80.4%: identificar fuente o marcar n/r |

### NO requieren cambio

- Ch5 Table 5.1 (L457): "S = 84.1%" — correcto
- Table B.1 (L2043-2046): Aggregados multi-seed — correctos (mean y std coinciden)
- Tables B.5 (Gate 9/A10): Todos verificados PASS
- Table B.6 (Gate 10): 7/9 correctos, 2 corregidos arriba
- Tables B.7, B.8 (Escalón 2): Todos correctos
- Todo Escalón 3: No aparece en el libro (marcado "in progress" en Table 11.1)

---

## Nota sobre actualización futura

Cuando el training multi-seed real de d4a4 termine en Mendieta (Job 1146677, estimado ~36h):

1. Se actualizarán los valores individuales con los training-seed reales
2. Se recalcularán Cohen d y p-value con datos genuinos
3. Se eliminará la calificación "eval-seed" y la nota transitoria
4. Se actualizará el rango en Table 11.3b
5. Se sincronizará al LaTeX

Hasta entonces, las correcciones de este informe hacen que el libro sea **honesto sobre lo que sabemos** sin debilitar las conclusiones (que son genuinas: d4a4 supera a D0 por >8pp incluso en single-seed).
