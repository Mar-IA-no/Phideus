---
schema_version: 1
id: front-voz-expresiva
kind: front
page_status: current
front_status: decision_ready
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 480c7ef4dddbfe8dfe92459e80ee0ae97b765f8c
source_paths:
  - Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md
  - Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md
depends_on: [front-escalon-1-bias-control]
tangents: [front-escalon-2, front-escalon-4]
architecture_status: candidate
experiment_status: phase_closed
evidence_status: cross_language_mixed
decision_status: pending_user
---

# Voz Expresiva Phideus

## Resumen

Este frente prueba si descriptores ratio-based y mecanismos de inyección
transferidos desde música aportan sobre un backbone vocal SSL y si esa ventaja
resiste un cambio de lengua y de hablante.

## Estado real

Fases 0A, 0B y 1 EN/ZH están cerradas. En `N-adapt`, `concat` y FiLM replican
cross-language; en `N-strict`, el lift inglés no transfiere de forma robusta y
FiLM/xattn son negativos en ZH. El resultado es una disociación, no un “replica”
o “no replica” global.

## Decisión pendiente

| Opción | Pregunta |
|---|---|
| Cerrar Fase 1 | ¿La disociación ya delimita suficientemente el alcance? |
| Fase 1.2 | ¿El cuello de N-strict está en descriptor, normalización o corpus? |
| Habla naturalista | ¿El patrón sobrevive fuera de ESD actuado, por ejemplo en MSP-Podcast? |

La elección pertenece al usuario.

## Genealogía

`EIR-EMR` fue el antecedente conceptual. Su nomenclatura fue descartada como
marca maestra por anticipar invariantes antes de contar con evidencia; sus
preguntas físicas fueron reformuladas dentro de este frente.

## Relaciones

- [Escalón 2](escalon-2.md)
- [Escalón 4](escalon-4.md)
- [Líneas preservadas](lineas-preservadas.md)

## Fuentes

- [README canónico](../../01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md)
- [Roadmap local](../../01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md)
