# Plan Gate 4.3 (Bifurcacion MIDI/Audio)

Fecha de corte: 2026-02-14

## Precondicion

- Gate 4.2 ya cerró con extension de `D4` a 8 epocas.
- Resultado de cierre: `S_best=64.2%` (epoch 7), `hard_neg=91.6%`.
- Gate 4.3 queda habilitado.

## Paradigmas

1. Linea MIDI (temperada): descriptores sobre eventos MIDI discretos (12-TET).
2. Linea Audio (armonia natural): descriptores espectrales continuos/no temperados.
3. Linea Dual: combinacion de ambas.

## Brazos (todos fresh, 5 epocas)

1. `D0` control (sin descriptor).
2. `D4-only` (MIDI-only).
3. `A4-only` (Audio-only).
4. `A7-only` (Audio-only, rational-attractor).
5. `D4+A4` (Dual).
6. `D4+A7` (Dual).

## Etapa de arranque recomendada

Antes del barrido completo de 5 epocas:

1. Pilot `a4` (1 ep / 100 batches)
2. Pilot `a7` (1 ep / 100 batches)
3. Pilot `d4a4` (1 ep / 100 batches)
4. Pilot `d4a7` (1 ep / 100 batches)

## Criterios de promocion a Gate 4.4

- Carril A (performance):
  - `S_best5(Dx) - S_best5(D0) >= +0.5pp`
  - `hard_neg(Dx) >= hard_neg(D0) - 1pp`
- Carril B (potencial):
  - no colapsa (`S_e5 >= S_e1 - 1.0pp`)
  - y pendiente positiva (`S_e5 - S_e3 >= +0.6pp`)

## Regla de control

- No usar `--resume` para comparar brazos de Gate 4.3.
- Metrica primaria: `S = min(A2M, M2A)` con protocolo canonico.
