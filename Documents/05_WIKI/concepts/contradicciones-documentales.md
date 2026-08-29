---
schema_version: 1
id: phideus-documentary-contradictions
kind: concept
page_status: current
front_status: transversal
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 480c7ef4dddbfe8dfe92459e80ee0ae97b765f8c
source_paths:
  - README.md
  - MARCO_EPISTEMOLOGICO_PHIDEUS.md
  - Documents/00_TRONCAL/Proyecto_Estado_Actual.md
  - Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_1/INDICE_ESCALON1_COMPLETO.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md
  - Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md
depends_on: [phideus-llm-context]
tangents: [phideus-evidence-regime]
---

# Registro de tensiones documentales

## Función

Este registro preserva desacoples entre documentos sin convertirlos
automáticamente en errores. Algunos son estados históricos válidos; otros son
deuda de sincronización. La autoridad se resuelve por alcance, fecha y régimen
de evidencia, no por ubicación física ni por el título del archivo.

| ID | Tensión | Lectura vigente | Tipo |
|---|---|---|---|
| `DOC-T01` | El roadmap Triplescaloneta v1.1 se presentaba como operativo, pero varias fases futuras allí ya fueron ejecutadas | Esqueleto conceptual e histórico; el portafolio vivo está en esta wiki y en el estado ejecutivo | recontextualización |
| `DOC-T02` | Escalón 1 conserva `84.1%±2.3pp` por eval-seeds y el cierre actual usa `84.0%±2.7pp` por training seeds | Ambos valores son válidos en regímenes distintos; el segundo es el cierre canónico multi-training-seed | separación de evidencia |
| `DOC-T03` | `ROADMAP_ESCALON_2.md` conserva una fase P3 como futura o running | El addendum y el README local, posteriores, registran P3 completo y el diagnóstico P2 vs P3 como tarea viva | snapshot histórico |
| `DOC-T04` | Escalón 1, EIR-EMR, Gate 8 y Gate 10 pueden parecer activos por su carpeta o sección | Escalón 1, Gate 8 y Gate 10 están cerrados en su alcance; EIR-EMR fue absorbido por Voz Expresiva; sólo Gate 6 Exp C queda como rama residual | deriva de navegación |
| `DOC-T05` | El marco epistemológico conserva una formulación anterior del programa y una métrica eval-seed de Escalón 1 | Se conserva como etapa real de la formulación; el estado cuantitativo actual debe leerse en el estado ejecutivo y el frente | evolución conceptual |
| `DOC-T06` | El informe histórico de ratios describe Voz Expresiva antes del cierre EN-ZH | Es genealogía válida, no sustituto del README y roadmap actuales del frente | snapshot histórico |
| `DOC-T07` | `Gate 6` nombra un diagnóstico histórico y también el frente AMT posterior | Son unidades de trabajo distintas con un alias humano compartido; esta wiki usa `front-gate-6-amt` para la rama AMT | colisión de alias |
| `DOC-T08` | Los contratos metodológicos del roadmap existen como especificación, pero no todos como schemas ejecutables | Tratar la especificación como diseño pendiente de materialización, no como infraestructura ya disponible | implementación pendiente |

## Regla de resolución

1. Para un resultado, prevalece el artefacto o informe del experimento y su
   régimen de evidencia.
2. Para el estado de un frente, prevalecen su README y roadmap actuales.
3. Para el estado global, prevalecen `Proyecto_Estado_Actual.md` y el README
   público.
4. Para genealogía, se conserva el documento histórico con su fecha.
5. Ninguna tensión autoriza a reescribir el pasado para que parezca una
   predicción correcta del presente.

## Fuentes de contraste

- [Estado ejecutivo](../../00_TRONCAL/Proyecto_Estado_Actual.md)
- [Roadmap Triplescaloneta v1.1](../../00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md)
- [Índice de Escalón 1](../../01_FRENTES_ACTIVOS/ESCALON_1/INDICE_ESCALON1_COMPLETO.md)
- [Roadmap de Escalón 2](../../01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md)
- [Marco epistemológico](../../../MARCO_EPISTEMOLOGICO_PHIDEUS.md)
