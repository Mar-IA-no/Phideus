---
schema_version: 1
id: phideus-current-portfolio
kind: roadmap
page_status: current
front_status: transversal
updated: 2026-09-03
verified_at: 2026-09-03
valid_at: 2026-09-03
recorded_at: 2026-09-03
evidence_commit: cbeabeba20a9e0b2b472672e86408e83b325eb08
source_paths:
  - README.md
  - Documents/00_TRONCAL/Proyecto_Estado_Actual.md
  - Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md
  - Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md
  - Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_55_CONSERVATIVE_POLICY_BRIDGE_CLOSED.md
  - Documents/05_WIKI/roadmaps/proportional-architecture-experiments.md
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
| foco arquitectónico experimental | Geometría proporcional / PPU | El núcleo local candidato quedó reformulado como factorial `RAW/CLOSURE × GENERIC/TYPED`. El contrato anti-leakage y los baselines clásicos ya fueron ejecutados sobre `256` masters: `320/320` IRLS convergieron, replay y artefactos pasaron auditoría; todavía no se probó el mixer neuronal | Implementar el smoke neuronal de dos seeds en CPU. Avisar antes de un eventual contraste GPU | Codex; promoción y GO/NO-GO: usuario |
| cierre experimental separado | Ola 56 prospectiva | Stage 0 retrospectivo seleccionó una compuerta Ridge contextual, pero no aisló valor contextual frente a advantage-only/shuffle. La recuperación pre-oracle quedó implementada y pusheada sin inferencia oficial | Completar auditoría y cierre de recuperación; luego ejecutar el protocolo fresco ya congelado, sin abrir otra ola bibliográfica | Codex |

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
| Ground truth proporcional | PPU/NHG | estratos de evidencia, equivalencias, conjuntos identificados, bancos, adjudicación, firewall de crédito y ejecutores algebraicos tipados | que exista una métrica proporcional universal, que toda respuesta deba ser puntual, que Clifford aporte una primitive irreductible o que una salida válida pertenezca necesariamente al proposer |

## Criterios de reapertura

| Frente | Reentrada mínima coherente | Evitar |
|---|---|---|
| E2 | análisis representacional P2/P3 | otro factorial ciego de training |
| Gate 6 | hipótesis explícita sobre Exp C | insistir con `Transkun+A4` ya cerrado |
| Voz | decisión entre N-strict y habla naturalista | presentar ESD como habla espontánea |
| E3 | mejor brazo o experimento discriminante nuevo | repetir P6 puro sin cambio de hipótesis |
| AA | cabeza de partición o CQT con gate de validez | volver a tuning de τ ya falsado |
| PPU/NHG | tracks exactos separados de unidades/Buckingham, composición/ciclos, conservación/equivalencia, compatibilidad local-global, geometrías estadísticas/de medidas, ley/solver, wiring/constitución, identificabilidad/certificación, projectivity, semiring, cociente de forma, realizabilidad conforme, autoridad set-valued y transformación garantizada; Olas 50–51 separan conjunto identificado y decisión, pero descartan como prioritaria la factorización two-stage simple sin autoridad adicional | confundir adimensionalidad, similitud, balance, constitución, residual, obstrucción, Fisher/Aitchison, costo/mecanismo, energía/simplécticidad, wiring/semántica, predicción/causalidad, equivariance/projectivity, PWL/tropicalidad, distancia/correspondencia, invariante local/realizabilidad global, conjunto poblacional/aproximación/inferencia, soundness/precisión, output/autoridad o proposal/repair |
| E4 | dataset, modalidades, baseline y controles predeclarados | abrir por analogía sin ground truth adecuado |

## Tensión no resuelta del portafolio

El programa ya tiene una vía descriptorial madura y una vía arquitectónica en
incubación. El nuevo régimen convirtió la segunda en trabajo experimental
concreto: el contrato y los clásicos ya establecieron una base reproducible y
un cambio de corrupción que desafía al baseline robusto. Ahora se aísla la
primitive de composición local sobre grafos de razones, separando evidencia
analítica, mixer y solver; sólo si aporta frente a controles clásicos y
genéricos se la transfiere
a un dominio donde pueda compararse con inyección explícita de descriptores bajo
dataset, presupuesto y evaluación comunes.
