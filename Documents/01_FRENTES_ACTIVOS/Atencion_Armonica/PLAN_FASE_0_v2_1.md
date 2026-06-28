# Plan — Atención Armónica Fase 0 v2.1

> Amendment del diseño original de `Fase 0`. Este documento reemplaza a la formulación previa para el trabajo local del frente, pero **no** implica todavía propagación al troncal. `PLAN_FASE_0_v1_superseded.md` se conserva por trazabilidad.

> **Estado operativo al 2026-06-28**: el sweep de calibración pasó, la combo quedó congelada, el `final_pool` pasó el gate final, el smoke supervisado de `A-rich` confirmó aprendibilidad sin saturación y el training decisivo de `Fase 0` cerró `54/54`. El resultado es dual: pair-state fuerte, `triangle` positivo en `OOD-poly` threshold-free y calibración `ARI@τ_val` pendiente.

## Contexto

La `Fase 0` original quedó invalidada por dos artefactos sucesivos del generador:

1. con parciales exactos y armónicos enteros (`beta=0`), las pair features cerradas resolvían la tarea casi solas;
2. al introducir inarmonicidad (`beta>0`), la envolvente determinística `amp=1/n` seguía filtrando el índice armónico y dejaba a un `PairMLP` per-par casi al techo.

La versión `v2.1` rediseña el generador y la auditoría para romper ambos canales cerrados antes de cualquier training GPU.

## Qué no cambia

La arquitectura y el stack ya auditados permanecen sin cambios:

- `peak_tokens.py`
- `grouping_dataset.py`
- `pairformer.py`
- `harness.py`
- `1_train_grouping.py`
- `1_report.py`

También se conservan:

- la tarea `same-source` sobre matriz de pares,
- los 6 modelos (`A-naive`, `A-rich`, `B`, `B-local`, `B-minus`, `B-shuffle`),
- los contrastes y métricas ya definidos,
- los regímenes `easy/hard`,
- `K=8`, `f0` en `100-500 Hz`, polifonía `1/2/3`.

## Qué cambia

### 1. Inarmonicidad per-source

Cada fuente usa:

`f_n = n * f0 * sqrt(1 + beta * n^2)`

con `beta>0` per-source, sampleado log-uniformemente dentro de un rango que se congela tras el sweep de calibración.

### 2. Amplitud randomizada per-source

Cada parcial usa:

`amp_n = (1 / n^alpha) * exp(epsilon_n)`

donde:

- `alpha ~ Uniform[alpha_lo, alpha_hi]` per-source
- `epsilon_n ~ Normal(0, sigma_amp^2)` iid por parcial

Esto reemplaza el `amp=1/n` determinístico y rompe el leak `log_amp_diff ≈ dlogf`.

### 3. Orden exacto de operaciones por fuente

1. samplear `beta` y construir frecuencias inarmónicas;
2. samplear `amp_raw`;
3. aplicar dropout iid;
4. restaurar por amplitud si quedaron menos de `min_partials=4`;
5. normalizar energía solo al final, sobre survivors.

### 4. Defaults no legacy

El modo por defecto del generador ya es `v2.1`. El modo legacy (`beta=0`, `alpha=1`, `sigma_amp=0`) queda solo para tests de regresión del gate.

## Gate obligatorio antes de GPU

El pool sintético no pasa a entrenamiento real si no cumple, sobre las celdas decisivas `poly2_easy`, `poly2_hard`, `poly3_easy`, `poly3_hard`:

### Headroom

- single features: `max(AUC, 1-AUC) < 0.90`
- `LogReg` sobre todo lo que recibe `A-rich`: `< 0.90`
- `PairMLP` espejo de `A-rich`: `< 0.90`

El `PairMLP` es el criterio duro del gate.

### Solvabilidad

- `oracle_privileged_upper_bound`: `min-cell ARI > 0.80`

Además se reporta, como diagnóstico no bloqueante:

- `oracle_unpriv_f0only`, que ignora `beta` a propósito y funciona como lower bound.

## Calibración

La calibración y el pool final quedan separados:

1. `calibration_pool` chico, con seed propia.
2. sweep de `16` combos:
   - `beta-center ∈ {1e-3, 3e-3}`
   - `alpha-range ∈ {[0.5,1.5], [0.5,2.5]}`
   - `sigma_amp ∈ {0.5, 1.0}`
   - `p_drop ∈ {0.15, 0.30}`
3. elegir combo por peor celda decisiva:
   - elegible si `max_cell PairMLP < 0.88` y `min_cell oracle_priv > 0.85`
   - desempate por cercanía a `PairMLP=0.83`, luego mayor `oracle_priv`, luego menor distorsión
4. congelar la combo
5. generar `final_pool` con seed distinta
6. correr gate final solo como `PASS/ABORT`

Si ninguna combo es elegible, la `Fase 0` vuelve a rediseño antes de GPU.

### Resultado real de la calibración v2.1

El sweep ya cerró y eligió, por la regla congelada, la combo:

- `beta-center = 1e-3`
- `alpha-range = [0.5, 1.5]`
- `sigma_amp = 0.5`
- `p_drop = 0.3`

Lectura útil:

- `16/16` combos quedaron elegibles;
- la elegida cayó en el punto objetivo de ambigüedad (`PairMLP ≈ 0.83`);
- `oracle_priv = 1.0` en todas las celdas decisivas;
- `oracle_unpriv_f0only ≈ 0.07` dejó un caveat explícito: la tarea depende de modelar más que `f0` solo.

El `final_pool` regenerado con seed distinta volvió a pasar el gate final. Con eso, el diseño `v2.1` quedó habilitado para training real.

## Smoke supervisado previo al training

Antes de comprometer GPU, `A-rich` corrió el diagnostic smoke congelado sobre la combo elegida. Lo que debía demostrar era:

1. aprender por encima de chance;
2. no saturar;
3. mantener `poly3_hard` en régimen aprendible.

Eso quedó cumplido. El smoke no cerró la comparación `A-naive vs A-rich`; solo habilitó el paso siguiente: el training completo con el `final_pool` congelado.

## Estado actual del experimento

El training real de `Fase 0` cerró sobre:

- `6` modelos
- `3` seeds
- `3` runs (`ID`, `OOD-poly`, `OOD-regime`)

La lectura final de `Fase 0` separa tres efectos:

- `B-minus ≫ A-rich`: el pair-state explícito aporta el salto grande.
- `B ≫ B-shuffle`: la estructura del triángulo hace trabajo real, no solo capacidad.
- `B vs B-local`: el `triangle` no gana universalmente; queda neutro o levemente por debajo en `IID`/`OOD-regime`, pero supera a la mezcla local param-matched en `OOD-poly` bajo `AUC/AP` threshold-free (`ΔAUC +0.053`, `ΔAP +0.093`, CI excluye 0).

El caveat de cierre es que `ARI@τ_val` no transfiere para `B` en `OOD-poly`: el ranking de pares mejora, pero el clustering con umbral heredado de validación se rompe. Por eso se abre `Fase 0.5` como post-audit de calibración antes de pasar a CQT/audio.

El orden de lectura quedó congelado antes del cierre y ya fue aplicado:

1. `B vs B-local`
2. `B-shuffle`
3. `B vs A-rich`
4. `B vs B-minus`

## Archivos involucrados

### Modificados en v2.1

- `src/atencion_armonica/harmonic_synth.py`
- `experiments/atencion_armonica/0_generate.py`
- `experiments/atencion_armonica/0_audit_pool.py`

### Intocados en v2.1

- `src/atencion_armonica/peak_tokens.py`
- `src/atencion_armonica/grouping_dataset.py`
- `src/atencion_armonica/pairformer.py`
- `experiments/atencion_armonica/harness.py`
- `experiments/atencion_armonica/1_train_grouping.py`
- `experiments/atencion_armonica/1_report.py`

## Verificación esperada

1. El leak de amplitud debe bajar de forma visible respecto del modo legacy.
2. El sweep debe producir tabla por celda decisiva, no solo agregados globales.
3. El modo legacy debe seguir disparando el gate como regresión del bug.
4. El `final_pool` debe pasar el gate antes de cualquier training. Cumplido.
5. El smoke de `A-rich` debe confirmar aprendibilidad sin saturación. Cumplido.
6. El frente puede propagarse al troncal como `Fase 0` cerrada con `GO` acotado: resultado atribuible en `AUC/AP`, caveat de calibración y validación fuera del sintético todavía pendiente.
