# Atención Armónica

> Frente nuevo en incubación local que prueba si una representación explícita de pares con actualización triangular puede capturar estructura armónica global de una mezcla polifónica mejor que un backbone token-only con features armónicas inyectadas.

## Estado actual: gate v2.1 PASS, final_pool congelado y training decisivo en curso (2026-06-27)

Este frente **todavía no debe leerse como frente canónico del programa**. Ya dejó atrás la etapa de “rediseño solamente”: el sweep de calibración `v2.1` pasó, el `final_pool` quedó congelado con gate `PASS`, el smoke supervisado de `A-rich` confirmó aprendibilidad sin saturación, y el training decisivo de `Fase 0` ya está corriendo. Lo que sigue sin estar cerrado no es la viabilidad del dataset, sino la atribución causal del eventual lift: si viene de transitividad/triangle o de pair-state genérico.

La razón de esa cautela es metodológica. La pregunta del frente no es si una red cualquiera puede agrupar parciales. La pregunta es más precisa: **si la maquinaria pair-state + transitividad + triangle update aporta algo por encima de un baseline con las mismas features armónicas cuando la evidencia per-par es genuinamente ambigua**. Si el dataset deja que una feature cerrada resuelva la tarea sola, el contraste `B vs A-rich` queda anulado por construcción.

## Qué pasó hasta acá

### v1: parciales exactos armónicos, artefacto de ratios

La primera formulación de `Fase 0` trabajaba con parciales exactos y armónicos enteros (`beta=0`). La auditoría del pool encontró enseguida que `common_f0_residual` y `ratio_residual` separaban `same-source` con `AUC≈1.0`. Eso volvía trivial la tarea para `A-rich`: el baseline decisivo recibía en sus pair features una respuesta casi cerrada.

### v2: inarmonicidad, pero leak por amplitud

La segunda formulación introdujo `beta>0` per-source y dropout de parciales. Eso rompió el oráculo de ratios enteros, pero apareció otro canal cerrado: la envolvente determinística `amp = 1/n` filtraba el índice armónico. Para `same-source`, `log_amp_diff` quedaba casi igual a `dlogf`, y un `PairMLP` chico seguía separando la tarea casi al techo.

### v2.1: inarmonicidad + amplitud randomizada + gate fuerte

La versión vigente rompe ambos canales cerrados a la vez:

- `beta>0` per-source para desarmar la armonicidad exacta.
- amplitud randomizada per-source `amp_n = (1 / n^alpha) * exp(epsilon_n)` para romper el leak `log_amp_diff ≈ dlogf`.
- dropout de parciales con `min_partials=4` y restauración determinística por amplitud.
- gate obligatorio de **feature-triviality** antes de cualquier training GPU.

### Sweep y final_pool: la etapa de dataset ya quedó resuelta

El sweep `v2.1` sobre `calibration_pool` encontró `16/16` combos elegibles bajo la regla congelada. La combo elegida por desempate determinístico fue:

- `beta-center = 1e-3`
- `alpha-range = [0.5, 1.5]`
- `sigma_amp = 0.5`
- `p_drop = 0.3`

La lectura útil de esa calibración fue doble:

- **headroom real**: `PairMLP` quedó en la banda `0.79-0.83`, lejos del techo trivial;
- **solvabilidad upper-bound**: `oracle_priv = 1.0` en todas las celdas decisivas.

El caveat importante quedó explícito desde el sweep: `oracle_unpriv_f0only` colapsa a `~0.07`. Eso no bloquea el frente, pero sí obliga a decir con precisión qué demostró el gate. El dataset ya no es feature-trivial; no quedó probado todavía que cualquier aproximación simple pueda recuperar la estructura sin supervisión.

Después de congelar la combo, el `final_pool` se regeneró con seed distinta y volvió a pasar el gate. Eso clausura la discusión “¿el dataset deja headroom real?” para esta fase.

## Tesis y contraste

La tesis fuerte del frente no es “inyectar armonía en un backbone genérico”, sino probar si una arquitectura con estado de pares y actualización triangular puede operar dentro de una geometría armónica donde la consistencia global importa. El contraste decisivo sigue siendo el mismo:

- `A-naive`: token attention + bias relativo, sin pair features explícitas.
- `A-rich`: mismo backbone token-only, pero con las mismas pair features que `B`.
- `B`: Harmonic Pairformer completo.
- `B-local`: control param-matched que aísla la suma sobre `k`.
- `B-minus`: ablación sin triangle.
- `B-shuffle`: control negativo parcial.

Si `A-rich` ya resuelve la tarea al techo, el frente no puede contestar su propia pregunta. Por eso el gate del dataset es parte constitutiva del experimento, no un extra operativo.

## Protocolo vigente

Ningún pool pasa a GPU sin cumplir dos condiciones sobre las celdas decisivas `poly2/3 × easy/hard`:

1. **Headroom real**: single features, `LogReg` y `PairMLP` sobre TODO lo que recibe `A-rich` deben quedar por debajo del umbral de feature-triviality.
2. **Solvabilidad real**: un `oracle_privileged_upper_bound` debe mostrar que la estructura todavía es recuperable globalmente.

La calibración actual se hace sobre un `calibration_pool` separado del `final_pool`. En `v2.1`, ese paso ya quedó cumplido: el `final_pool` vigente pasó el gate y es el único pool habilitado para el training de `Fase 0`.

## Estado experimental vivo

### Diagnostic smoke de A-rich

Antes de comprometer GPU, el frente corrió el smoke que faltaba: `A-rich` sobre la combo elegida, en CPU y con protocolo acotado. El resultado importante no fue una F1 alta, sino otra cosa:

- `A-rich` aprende por encima de chance en todas las celdas decisivas;
- no satura;
- `poly3_hard` sigue siendo aprendible.

Eso confirmó que el problema ya no es ni trivial ni imposible desde el punto de vista supervisado. Era la última condición para habilitar el training real.

### Training decisivo en curso

El run completo de `Fase 0` ya arrancó sobre el `final_pool`, con los 6 modelos, `3` seeds y `3` runs (`ID`, `OOD-poly`, `OOD-regime`).

Lo que ya apareció como señal parcial es prometedor pero **todavía no cierra la hipótesis**:

- `B` ya mostró una ventaja muy grande sobre `A-rich` en `ID`, especialmente en `poly3_hard`.

Eso es compatible con la tesis del frente, pero no la identifica todavía. Para atribuir ganancia al triángulo/transitividad siguen faltando los contrastes que aíslan la causa:

- `B vs B-local`
- `B vs B-minus`
- `B vs B-shuffle`

Hasta ver esos controles, la lectura correcta sigue siendo “resultado parcial fuerte, atribución pendiente”.

## Documentación local de incubación

- Roadmap del frente: `./ROADMAP_ATENCION_ARMONICA.md`
- Plan vigente de `Fase 0 v2.1`: `./PLAN_FASE_0_v2_1.md`
- Plan anterior preservado por trazabilidad: `./PLAN_FASE_0_v1_superseded.md`

## Regla de propagación

Este frente **todavía no se propaga a `Documents/00_TRONCAL/`**. Aunque ya tiene `gate PASS`, `final_pool` congelado y training en curso, la pregunta de fondo del frente no está cerrada mientras falten los controles que separan:

- pair-state genérico,
- transitividad/triangle,
- posible confound de capacidad.

La capa documental correcta, por ahora, sigue siendo esta carpeta local de incubación.
