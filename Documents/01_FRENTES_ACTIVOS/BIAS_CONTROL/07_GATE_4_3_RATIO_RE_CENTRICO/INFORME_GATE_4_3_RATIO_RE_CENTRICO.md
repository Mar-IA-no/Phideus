# Gate 4.3 Ratio Re-Centrico (Version Bifurcada)

Fecha: 2026-02-14 UTC  
Estado al corte: Gate 4.3 en ejecucion (`D0` y `D4` cerrados; `A4` en curso; `A7` y duales pendientes).

---

## 1) Cambio de marco

Este informe reemplaza el framing anterior de "Gate 4.3 = barrido D0..D10".

Decision vigente:

1. **Gate 4.2** conserva el run extendido de `D4` (8 epocas) dentro de la misma fase y ya quedó cerrado.
2. `D4` sostuvo y confirmó mejora sobre foundation; por lo tanto queda abierta la ejecucion de **Gate 4.3**.
3. **Gate 4.3** pasa a ser un bloque causal corto y bifurcado (no un barrido masivo).
4. **Gate 4.4** queda como barrido amplio posterior.

---

## 2) Bifurcacion epistemologica explicita

A partir de este punto, la exploracion de descriptores se divide en tres lineas:

1. **MIDI temperado**: descriptores sobre eventos MIDI discretos (paradigma 12-TET).
2. **Audio armonia natural**: descriptores sobre estructura espectral continua/no temperada.
3. **Dual**: inyeccion simultanea MIDI+Audio para medir sinergia.

Esta separacion corrige una mezcla conceptual que venia de fases previas: no confundir "ratios sobre MIDI" con "ratios fisicos continuos del audio".

---

## 3) Gate 4.3 (bloque focal)

Objetivo: responder primero la pregunta central con el menor numero de brazos que preserve inferencia causal.

### 3.1 Brazos acordados (todos fresh, 5 epocas)

1. `D0` (control, sin descriptor)
2. `D4-only` (MIDI-only, temperado)
3. `A4-only` (Audio-only, log-freq local)
4. `A7-only` (Audio-only, rational-attractor)
5. `D4+A4` (Dual)
6. `D4+A7` (Dual)

Etapa inmediata previa al 5ep:
- pilotos de 1 epoca/100 batches para `a4`, `a7`, `d4a4`, `d4a7`.

### 3.2 Regla de comparabilidad

- Todos los brazos desde `foundation_locked_e25.pt`.
- No usar `--resume` para comparar brazos en Gate 4.3.
- Metrica primaria: `S=min(A2M, M2A)`.
- Metrica de robustez: `hard_neg`.

### 3.3 Criterio de promocion

Carril A (performance):
- `S_best5(Dx) - S_best5(D0) >= +0.5pp`
- `hard_neg(Dx) >= hard_neg(D0) - 1pp`

Carril B (potencial):
- no colapso (`S_e5 >= S_e1 - 1.0pp`)
- y pendiente positiva (`S_e5 - S_e3 >= +0.6pp`)

Promocion a Gate 4.4 ampliado si cumple A o B.

---

## 4) Gate 4.4 (barrido amplio posterior)

El barrido creativo se mueve a Gate 4.4, manteniendo la bifurcacion.

### 4.1 Orden MIDI (temperado)

`D0` control +

1. `D3`
2. `D8`
3. `D9`
4. `D10`
5. `D2`
6. `D5`
7. `D6`
8. `D7`

Nota: `D1` queda documentado como probado en Gate 4.2.

### 4.2 Orden Audio (armonia natural)

1. `A1`
2. `A2`
3. `A3`
4. `A5`
5. `A6`

---

## 5) Relacion con historico Roseta/UOEMD

Los descriptores historicos (histogramas, constellations, hashes, Route A/B) quedan como fuente de ideas y reciclaje tecnico, pero la ejecucion actual se ordena por paradigma:

- lo que dependa de eventos MIDI discretos entra en rama MIDI,
- lo que dependa de estructura espectral continua entra en rama Audio,
- las combinaciones se tratan como rama Dual.

---

## 6) Documentos vinculados

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/plan_gate_4.3.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_BIFURCACION_RATIO/plan_gate_4.4.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md`

---

## 7) Corte de ejecucion real (2026-02-14 14:45 UTC)

Run activo: `data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000`.

### 7.1 D0 y D4 cerrados (5 epocas)

| Brazo | Best S | Best ep | hard_neg (best) | Lectura |
|-------|--------|---------|-----------------|---------|
| D0 | 60.2% | e3 | 90.0% | Control reproducible, pico en mitad de schedule |
| D4 | 63.6% | e5 | 91.2% | Mejora robusta sobre D0 en mismo régimen |

Delta principal del corte:
- `D4 - D0 = +3.4pp` en `S` (best-to-best).

### 7.2 A4 en recovery fuerte (e1-e3)

| Epoch | S | A2M R@10 | M2A R@10 | MRR_avg | R@1_avg | R@20_avg | hard_neg |
|------:|---:|---------:|---------:|--------:|--------:|---------:|---------:|
| e1 | 35.4% | 35.4% | 40.4% | 0.149 | 4.4% | 59.9% | 85.8% |
| e2 | 51.2% | 51.2% | 54.0% | 0.219 | 8.5% | 71.1% | 86.8% |
| e3 | 61.0% | 61.0% | 61.2% | 0.260 | 11.8% | 79.6% | 89.8% |

Lectura técnica:
1. `A4` corrige la perturbación inicial en 3 épocas (`+25.6pp` en `S` de e1 a e3).
2. Al e3, `A4` entra en zona competitiva con `D0` en `S`, pero aún por debajo en precisión de ranking fino (`MRR`, `R@1`).
3. El cierre de e4-e5 de `A4` es crítico para decidir si la rama audio concat promueve a Gate 4.4 como candidata fuerte.

### 7.3 Cola de ejecución inmediata

1. Cerrar `A4` (e4-e5).
2. Ejecutar `A7`.
3. Intervenir la secuencia del script actual (orden viejo) al terminar `A4`.
4. Relanzar desde `A7` con orden corregido: `A7 -> A4x -> A7x -> D4+A4 -> D4+A7`.
5. Cerrar duales una vez resueltas las comparaciones directas `A4 vs A4x` y `A7 vs A7x`.
