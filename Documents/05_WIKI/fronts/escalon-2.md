---
schema_version: 1
id: front-escalon-2
kind: front
page_status: current
front_status: focus_active
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 480c7ef4dddbfe8dfe92459e80ee0ae97b765f8c
source_paths:
  - Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md
depends_on: [front-escalon-1-bias-control]
tangents: [front-voz-expresiva, front-escalon-4]
architecture_status: baseline
experiment_status: analysis_pending
evidence_status: controlled_null_two_encoder_regimes
decision_status: pending_analysis
---

# Escalón 2: Speech ↔ EGG

## Resumen

Escalón 2 es el foco principal porque lleva la hipótesis de armonía natural a
dos mediciones físicas del mismo oscilador vocal. Separa dinámica temporal de
F0, estructura armónica intra-frame, controles no-ratio y variantes
perceptuales.

## Estado real

P0, P1, P2-control, P2-main, P2.5, P2.5b y la primera pasada de P3 están
completos. `concat`, `attn_bias`, `xattn` y `pca` no produjeron lift
descriptor-guided defendible sobre sus baselines. WavLM-Large frozen elevó
levemente el baseline, pero no cambió esa lectura.

## Observación

El frente sostiene hasta ahora un null descriptorial bajo dos regímenes de
encoder: from-scratch pequeño y foundation frozen. Algunas interacciones son
informativas porque ciertos descriptores y mecanismos degradan el resultado;
no es un null que autorice indiferencia metodológica.

## Próximo experimento discriminante

Comparar P2 y P3 mediante CKA, probes lineales y análisis representacional para
separar cuánto del null depende del encoder, cuánto del descriptor y cuánto de
la relación física entre las modalidades. No está justificada otra campaña de
training ciega antes de ese diagnóstico.

## Relaciones

- [Escalón 1](escalon-1-bias-control.md): fuente de mecánica y controles.
- [Voz Expresiva](voz-expresiva.md): reutiliza voz y mecanismos, pero cambia la
  tarea y el régimen de generalización.
- [Escalón 4](escalon-4.md): horizonte fisiológico posible.

## Fuentes

- [README canónico](../../01_FRENTES_ACTIVOS/ESCALON_2/README.md)
- [Roadmap local](../../01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md)
- [Prerregistro P2.5](../../01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md)
