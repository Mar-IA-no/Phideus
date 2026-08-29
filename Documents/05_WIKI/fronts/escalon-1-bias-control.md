---
schema_version: 1
id: front-escalon-1-bias-control
kind: front
page_status: current
front_status: closed
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 480c7ef4dddbfe8dfe92459e80ee0ae97b765f8c
source_paths:
  - Documents/01_FRENTES_ACTIVOS/ESCALON_1/INDICE_ESCALON1_COMPLETO.md
  - Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md
depends_on: []
tangents: [front-gate-6-amt, front-escalon-2, front-voz-expresiva]
architecture_status: baseline
experiment_status: phase_closed
evidence_status: multi_seed_controlled
decision_status: decided
---

# Escalón 1 y BIAS_CONTROL

## Resumen

Escalón 1 es el cierre fundacional de la vía descriptor-guided en Audio↔MIDI.
Reúne el brazo Shazam, el resultado negativo de DANN y el brazo neural
BIAS_CONTROL. Su resultado canónico es `d4a4=84.0%±2.7pp` sobre cinco training
seeds, frente a `D0=75.2%±2.3pp`.

## Estado real

El tronco está cerrado. Gate 8 y Gate 10 también están cerrados. La única
pregunta downstream que permanece abierta vive fuera del cierre principal, en
[Gate 6 Exp C](gate-6-amt.md).

## Observaciones

- La mejora descriptor-guided es causal bajo los controles de Gate 5B.
- El efecto principal es una reorganización geométrica del espacio latente, no
  una mejora equivalente de decodificabilidad local.
- Gate 8 mostró que conditioned projections también pueden preservar señal.
- Gate 10 mostró que, en su rama retrospectiva, el mecanismo pesó más que el
  descriptor: `concat > FiLM/pca >> attn_bias`.

## Alcance

Escalón 1 valida la mecánica de inyección y la reorganización geométrica. No
prueba por sí solo la tesis fuerte de armonía natural porque mezcla relaciones
musicales, temperamento y descriptores útiles con una ontología física más
amplia.

## Relaciones

- [Gate 6 AMT](gate-6-amt.md)
- [Escalón 2](escalon-2.md)
- [Voz Expresiva](voz-expresiva.md)
- [Tres vías de investigación](../concepts/tres-vias-de-investigacion.md)

## Fuentes

- [Índice completo de Escalón 1](../../01_FRENTES_ACTIVOS/ESCALON_1/INDICE_ESCALON1_COMPLETO.md)
- [Roadmap BIAS_CONTROL](../../01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md)
