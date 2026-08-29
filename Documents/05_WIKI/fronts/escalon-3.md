---
schema_version: 1
id: front-escalon-3
kind: front
page_status: current
front_status: reopenable
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 480c7ef4dddbfe8dfe92459e80ee0ae97b765f8c
source_paths:
  - Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_3/ROADMAP_ESCALON_3.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_3/Resultados_E3_P5_P6.md
depends_on: []
tangents: [front-atencion-armonica, ppu-natural-harmonic-geometry]
architecture_status: candidate
experiment_status: phase_closed
evidence_status: synthetic_iid_ood
decision_status: pending_user
---

# Escalón 3: Audio XY ↔ Lissajous

## Resumen

Escalón 3 es un banco sintético con ground truth determinista donde un ratio
organiza simultáneamente sonido, trayectoria visual y parámetros generativos.
Permite estudiar IID, ratio-OOD, scale-OOD y equivalence-OOD sin ambigüedad de
etiqueta.

## Estado real

P0, P1, P2, P4, P5 y P6 completaron su primera ola. P3
descriptor×mecanismo permanece abierto. `P2-flat` es el baseline general de
IID; `P2-cqtshift` es la referencia ratio-aware; `P5-cqtshift` es el mejor
brazo geométrico/OOD actual. P6 toroidal puro organiza el espacio, pero no
supera a P5 en las métricas OOD primarias.

## Inferencia acotada

La receta toroidal pura probada no gana. Eso no refuta toda geometría no plana.
El frente mostró además que storage, retrieval y activation deben analizarse
como operaciones distintas: una geometría organizada puede requerir otra regla
de lectura para demostrar utilidad.

## Puntos de reentrada

- replicación desde el mejor brazo actual;
- P3 descriptor×mecanismo;
- P7 activation arena;
- P8 transferencia física y convergencia con Beacon.

## Relaciones

- [Atención Armónica](atencion-armonica.md)
- [PPU / Natural Harmonic Geometry](../concepts/ppu-geometria-armonica-natural.md)

## Fuentes

- [README canónico](../../01_FRENTES_ACTIVOS/ESCALON_3/README.md)
- [Roadmap local](../../01_FRENTES_ACTIVOS/ESCALON_3/ROADMAP_ESCALON_3.md)
- [Resultados P5/P6](../../01_FRENTES_ACTIVOS/ESCALON_3/Resultados_E3_P5_P6.md)
