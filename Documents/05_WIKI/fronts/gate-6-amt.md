---
schema_version: 1
id: front-gate-6-amt
kind: front
page_status: current
front_status: residual_active
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 480c7ef4dddbfe8dfe92459e80ee0ae97b765f8c
source_paths:
  - Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md
depends_on: [front-escalon-1-bias-control]
tangents: [front-escalon-2]
architecture_status: not_applicable
experiment_status: running
evidence_status: mixed_completed_and_open
decision_status: pending_user
---

# Gate 6 AMT

## Resumen

Gate 6 pregunta si la ventaja geométrica de Escalón 1 se traduce a Automatic
Music Transcription. Es una validación downstream, no una reapertura del cierre
de BIAS_CONTROL.

## Estado real

| Experimento | Estado | Lectura |
|---|---|---|
| Exp 0 | completo | Transkun es un baseline sano |
| Exp A | cerrado negativo | `Transkun+A4` no supera el control |
| Exp B | cerrado negativo | degradación no abre una ventaja para A4 |
| Exp C | activo residual | decoder serio sobre features VICReg congeladas |

## Pregunta abierta

¿Las representaciones descriptor-guided conservan información útil para AMT
cuando se las lee con un decoder suficientemente capaz, o su ventaja permanece
confinada a la geometría de retrieval?

## Límite de interpretación

Un resultado negativo en AMT no refuta la reorganización geométrica observada en
Escalón 1. Un resultado positivo tampoco prueba la tesis fuerte de armonía
natural: demostraría utilidad downstream dentro de este dominio y receta.

## Fuentes

- [README Gate 6](../../01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md)
- [Escalón 1](escalon-1-bias-control.md)
