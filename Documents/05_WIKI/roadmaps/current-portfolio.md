---
schema_version: 1
id: phideus-current-portfolio
kind: roadmap
page_status: current
front_status: transversal
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 4ea32c98ccc74abbd27993e57e1cd9214230bcce
source_paths:
  - README.md
  - Documents/00_TRONCAL/Proyecto_Estado_Actual.md
  - Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md
  - Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md
  - Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md
depends_on: []
tangents: [phideus-three-routes]
---

# Mapa vigente de portafolio y roadmaps

## Función

Este documento salda la ausencia de un roadmap de portafolio posterior a la
ejecución de Escalón 2 P3, Escalón 3 P5/P6, Voz Expresiva EN↔ZH y Atención
Armónica 0.6. No reemplaza los roadmaps locales ni fija unilateralmente una
prioridad: muestra cómo se relacionan y dónde se necesita una decisión.

## Capas del roadmap

| Capa | Documento | Uso actual |
|---|---|---|
| Programa público | `README.md` | Estado sintético y framing |
| Estado ejecutivo | `Proyecto_Estado_Actual.md` | Corte consolidado |
| Portafolio | Este documento | Dependencias y decisiones entre frentes |
| Roadmap local | README/roadmap de cada frente | Secuencia interna y evidencia local |
| Roadmap histórico | `PLAN_AVANCE_TRIPLESCALONETA_v1.1.md` | Hipótesis H3a–H3d y protocolo común; no secuencia vigente |

## Flujo inmediato documentado

| Orden lógico | Unidad | Condición de entrada | Salida esperada | Quién decide continuación |
|---:|---|---|---|---|
| 1 | Escalón 2: P2 vs P3 | P2/P3 completos | Diagnóstico CKA/probes y lectura del null | Usuario |
| paralelo | Gate 6 Exp C | Artefactos downstream disponibles | Evidencia de utilidad de features congeladas | Usuario |
| decisión | Voz Expresiva | Cierre EN↔ZH completo | Elegir Fase 1.2, dominio naturalista o cierre | Usuario |
| decisión | Atención Armónica | Fases 0–0.6 completas | Elegir Stage B o CQT | Usuario |
| reactivación | Escalón 3 | P0, P1, P2, P4, P5 y P6 completos; P3 abierto | Elegir P3, replicación, activation o transferencia | Usuario |
| proyección | Escalón 4 | Método transferible y diseño aprobado | Abrir protocolo fisiológico | Usuario |
| investigación transversal | Ground truth proporcional | Cinco olas y doce investigaciones integradas | Diseñar el primer banco exacto dimensional/composicional/conservativo/local-global | Usuario |

Sólo Escalón 2 está declarado como foco principal. Las demás filas son ramas
paralelas o bifurcaciones preservadas, no una cola obligatoria.

## Dependencias y tangencias

| Origen | Destino | Qué se transfiere | Qué no se puede asumir |
|---|---|---|---|
| Escalón 1 | Escalón 2 | controles causales, mecanismos y lectura geométrica | que el descriptor musical funcione en el oscilador glotal |
| Gate 8 / 10 | E2 y Voz | conditioned projection, concat, FiLM, atención | que un ranking de mecanismos sea universal entre backbones |
| Escalón 2 | Voz Carril B | descriptores físicos Speech↔EGG | que SER y alineación cross-modal sean la misma tarea |
| Voz / E2 | Escalón 4 | puente hacia señales fisiológicas | que expresión vocal pruebe acoplamiento ECG↔PPG |
| Escalón 3 | Atención Armónica | bancos sintéticos, OOD y geometría controlada | que una geometría de Lissajous sea la geometría del grouping armónico |
| E3 + AA | PPU/NHG | storage, retrieval, composición y partición | que ya exista una arquitectura proporcional general |
| Ground truth proporcional | PPU/NHG | estratos de evidencia, equivalencias, bancos y adjudicación | que exista una métrica proporcional universal |

## Criterios de reapertura

| Frente | Reentrada mínima coherente | Evitar |
|---|---|---|
| E2 | análisis representacional P2/P3 | otro factorial ciego de training |
| Gate 6 | hipótesis explícita sobre Exp C | insistir con `Transkun+A4` ya cerrado |
| Voz | decisión entre N-strict y habla naturalista | presentar ESD como habla espontánea |
| E3 | mejor brazo o experimento discriminante nuevo | repetir P6 puro sin cambio de hipótesis |
| AA | cabeza de partición o CQT con gate de validez | volver a tuning de τ ya falsado |
| PPU/NHG | tracks exactos separados de unidades/Buckingham, composición/ciclos, conservación/equivalencia y compatibilidad local-global antes de la mega-arquitectura | confundir adimensionalidad, similitud, balance, constitución, residual, obstrucción, autosimilitud y alometría |
| E4 | dataset, modalidades, baseline y controles predeclarados | abrir por analogía sin ground truth adecuado |

## Tensión no resuelta del portafolio

El programa ya tiene una vía descriptorial madura y una vía arquitectónica en
incubación, pero todavía no tiene un experimento que las confronte bajo un mismo
dominio y protocolo. Esa convergencia es una oportunidad arquitectónica futura:
comparar una red que recibe proporciones como descriptor con otra cuyo sesgo
inductivo representa relaciones proporcionales de manera nativa, manteniendo
dataset, presupuesto y evaluación comparables.
