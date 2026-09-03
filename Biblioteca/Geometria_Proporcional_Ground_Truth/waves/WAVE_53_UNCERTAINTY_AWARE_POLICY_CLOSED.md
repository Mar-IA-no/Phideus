# Ola 53 — cierre de la política ordinal sensible a incertidumbre

> **Estado:** `EXECUTED / EXTERNAL-REPLAY-EXACT / DIAGNOSTIC-PATTERN-FALSE / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Código ejecutado:** `d6d9b2b16a90ab8952a830e2b2c86a5435966e0c`
> **Régimen:** CPU, logits históricos abiertos, sin re-forward, sin GPU, lockbox no leído

## Qué se puso a prueba

La Ola 52 había mostrado que una política explícita puede usar una utilidad
ordinal para reducir decisiones incompatibles, pero recibe un conjunto binario
cuyas omisiones y falsos positivos ya no puede reparar. La Ola 53 conservó la
incertidumbre anterior al threshold. Sobre los logits ensemble de tres seeds
ajustó un calibrador Platt compartido, construyó una masa sobre los quince
conjuntos no vacíos bajo una hipótesis Bernoulli marginal independiente y eligió,
para cada una de las 24 políticas ordinales, la acción de menor regret esperado.

El experimento mantuvo separados `calibration_fit`, `decision_select` y
`val_monitor` en `192/192/384` tokens. La población primaria de monitor reunió
`148` tokens `NEAR_RIVAL` con al menos dos familias compatibles. Calibrador,
peso del baseline continuo y fronteras de abstención fueron fijados antes de
leer los outcomes de monitor. El bootstrap usó `pair_token` como unidad y
conservó juntas sus 24 políticas.

## Resultado

| Sistema | Accuracy | Acción compatible | Regret | Promedio del peor regret por token |
|---|---:|---:|---:|---:|
| `hard_set_policy` | **0.8395** | 0.9375 | 0.1238 | **0.3694** |
| `marginal_expected_regret` | 0.7767 | **0.9637** | 0.1223 | 0.4454 |
| `raw_expected_regret` | 0.7824 | 0.9626 | **0.1184** | 0.4420 |
| `score_composition` | **0.8713** | 0.9048 | 0.1304 | 0.4775 |
| `probability_shuffled` | 0.7362 | 0.9288 | 0.1690 | 0.5259 |
| `utility_masked` | 0.3271 | 0.9865 | 0.4367 | 1.0034 |

La política marginal calibrada elevó compatibilidad frente al baseline duro en
`+0.0262`, IC95 `[+0.0118,+0.0422]`. Esa seguridad local tuvo un costo: perdió
`-0.0628` de accuracy, IC95 `[-0.0884,-0.0372]`, y no produjo la reducción de
regret predeclarada (`-0.0015`, IC95 `[-0.0232,+0.0182]`). También empeoró el
promedio del peor regret por token en `+0.0760`, IC95
`[+0.0214,+0.1256]`. El resultado no es una dominancia de la regla marginal:
cambia la clase de error, pero no mejora el sistema conjunto.

Platt redujo levemente NLL, de `0.2692` a `0.2657`, sin mejorar Brier. Frente a
la misma regla con sigmoid crudo, perdió `-0.0056` de accuracy, IC95
`[-0.0101,-0.0011]`. El regret raw fue menor en el punto (`0.1184` frente a
`0.1223`), pero su contraste no fue concluyente: marginal−raw `+0.0039`, IC95
`[-0.0004,+0.0089]`. Calibrar mejor una marginal no garantizó decidir mejor.

Los controles separan dos dependencias. Barajar probabilidades redujo accuracy,
pero la diferencia `+0.0405` quedó por debajo del mínimo diagnóstico `+0.05`;
por eso ese criterio falló aunque el IC excluyera cero. Enmascarar utilidad
destruyó accuracy y elevó regret, mientras aumentó compatibilidad: la utilidad
es causal para elegir y ordenar costos, pero no para maximizar por sí sola cada
métrica.

## Abstención y dependencia

El score de riesgo sí ordenó dificultad empírica. La frontera nominal `75%`
retuvo `0.7162` de la población primaria y redujo el regret marginal de `0.1223`
a `0.0798`. Esto habilita triage o rechazo selectivo, pero no constituye una
garantía conformal.

La limitación más informativa aparece en la estructura conjunta. En monitor
primario, las correlaciones residuales entre familias alcanzaron `0.3804` con
sigmoid y `0.3725` con Platt. El error L1 de distribución de cardinalidad siguió
alto (`0.4424` y `0.4256`). El producto de cuatro Bernoulli independientes
produce así una distribución de conjuntos que no respeta plenamente las
dependencias y cardinalidades observadas. Condicionarla a conjunto no vacío
altera además las marginales que acababan de calibrarse.

## Adjudicación y siguiente discriminante

El patrón conjunto fue `false`: fallaron ventaja de regret frente al baseline
duro, no inferioridad de accuracy y magnitud mínima frente al shuffle. Pasaron
compatibilidad, el criterio de calibración por NLL, la tolerancia de regret
frente a raw, la selección al `75%`, el control de utilidad y el replay externo.

La incertidumbre no debe descartarse; debe cambiar de objeto. El siguiente
discriminante mantiene encoder, logits, splits, utilidades y pérdida congelados,
pero reemplaza la factorización marginal por un posterior regularizado sobre
los quince conjuntos no vacíos. Potenciales unary derivados de logits,
interacciones entre familias y sesgos de cardinalidad permitirán preguntar si
la pérdida decisional proviene de suponer independencia, sin confundirla con un
nuevo representation learner. Esta alternativa queda candidata, no promovida.

## Integridad y alcance

- análisis hash común: `2a3623b8614bdab43639363746ec5a6e6053b496eb1e5767d28f9808ab4408ec`;
- replay externo exacto en `5/5` NPZ y `13/13` artefactos textuales;
- `2.000 × 148` índices de bootstrap pareado preservados;
- `64.512` filas token×política y `27.648` filas token×política×seed;
- JSON estricto, sin `NaN` o infinitos no estándar;
- auditoría independiente R301 validó ejecución, trazabilidad y patrón, y
  corrigió el alcance de dos formulaciones interpretativas.

Es evidencia de desarrollo sobre un banco sintético y datos históricamente
abiertos. No usa utilidad natural, no prueba una geometría física, no promueve
una arquitectura y no decide GO/NO-GO.
