# Plan Gate 4.3 — Cierre y Salida

Fecha de corte: 2026-02-17  
Estado: COMPLETADO

## 1) Alcance ejecutado

Gate 4.3 quedó ejecutado en cinco bloques:

1. Fase 0: `D0`, `D4`, `A4` (concat/base).
2. Fase 1: `A7`, `A4x`, `A7x`.
3. Fase 2: `D4x`.
4. Fase 3: `d4a4` (dual same-mod) + `d4a4cm` (dual cross-modal).
5. Fase 5: `A4r`, `D4r`, `A8`, `A9` (corrida UNC).

Además, se ejecutó run largo `d4a4-scratch` 30ep.

## 2) Resultado de decisión del plan

- Ganador 5ep: `d4a4` con `S=69.8%`.
- Mejor single-descriptor: `A4r` con `S=68.6%`.
- Mejor run largo: `d4a4-scratch e30` con `S=83.6%`.

Conclusión metodológica del gate:

1. `concat` es competitivo y robusto para descriptores fuertes.
2. Reverse cross-attention supera a cross-att regular para single-descriptor.
3. Inyección cross-modal temprana (`d4a4cm`) no es mecanismo de continuidad.

## 3) Criterios de salida (cumplidos)

1. Matriz descriptor × mecanismo suficientemente cubierta para decidir continuidad.
2. Comparabilidad canónica mantenida (`pool=256`, `queries=500`, `seed=42`).
3. Resultado de run largo obtenido en el mejor brazo dual (`d4a4-scratch`).

## 4) Backlog transferido a siguiente fase

1. `a4r-scratch` y `d4a4r-scratch` 30ep (cola UNC) para contrastar simplicidad vs dual reverse.
2. Gate 4.4: third tower + FiLM + MoE (arquitecturas mayores).
3. Gate 4.5: optimización de scheduler/LR sobre runs extendidos (50ep/60ep).
4. Gate 5A: barrido descriptor x mecanismo + cross-modal (replanteado con learnings 4.3).
5. Gate 5B: validación científica (13 tests) sobre best model final.

## 5) Artefactos canónicos de Gate 4.3

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/README.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME COMPLETO: d4a4-scratch 30 epochs.md`
- `data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000/`
- `data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/`
