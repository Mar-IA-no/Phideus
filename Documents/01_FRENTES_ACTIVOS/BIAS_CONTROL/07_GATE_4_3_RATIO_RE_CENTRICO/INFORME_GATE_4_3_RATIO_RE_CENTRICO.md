# Gate 4.3 Ratio Re-Centrico (Version Bifurcada)

Fecha: 2026-02-14 UTC  
Estado al corte: Gate 4.2 cerrado (`D4 8ep` best `S=64.2%`, `hard_neg=91.6%`).

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
