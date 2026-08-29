---
schema_version: 1
id: phideus-human-visual-map
kind: map
page_status: current
front_status: transversal
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 626758f2ba21bdaf2270d012535139b9fc477255
source_paths:
  - README.md
  - Documents/00_TRONCAL/Proyecto_Estado_Actual.md
  - Documents/01_FRENTES_ACTIVOS/
  - Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md
depends_on: []
tangents: [phideus-three-routes]
---

# Mapa visual del programa Phideus

## Leyenda

| Etiqueta | Estado |
|---|---|
| `FOCO` | foco activo |
| `RESIDUAL` | rama residual o decisión pendiente |
| `REACTIVABLE` | fase cerrada con punto de reentrada |
| `INCUBACION` | incubación arquitectónica |
| `CERRADO` | cerrado, superseded o histórico |
| `PROYECCION` | horizonte sin campaña activa |

## El programa de un vistazo

```mermaid
flowchart LR
    E1["CERRADO: Escalón 1<br/>mecánica descriptor-guided"]
    G6["RESIDUAL: Gate 6 Exp C<br/>utilidad downstream"]
    E2["FOCO: Escalón 2<br/>Speech ↔ EGG"]
    VE["DECISION: Voz Expresiva<br/>decisión estratégica"]
    E3["REACTIVABLE: Escalón 3<br/>Lissajous y geometría"]
    AA["INCUBACION: Atención Armónica<br/>pair-state + triangle"]
    E4["PROYECCION: Escalón 4<br/>ECG ↔ PPG"]
    PPU["PPU / Natural Harmonic Geometry"]
    GT["INVESTIGACION: ground truth proporcional<br/>oráculos → simulación → cámara → transferencia"]

    E1 -->|mecanismos| E2
    E1 -->|mecanismos| VE
    E1 -->|downstream| G6
    E2 -->|voz física y fisiología| E4
    E2 -.->|correlatos expresivos| VE
    E3 -->|geometría latente| PPU
    AA -->|razonamiento relacional| PPU
    GT -->|base estratificada| PPU
    AA -.->|CQT / audio real| E3
```

## Estado real de los frentes

| Frente | Qué ya sabemos | Qué queda vivo | Estado |
|---|---|---|---|
| Escalón 1 | La inyección descriptor-guided puede reorganizar causalmente la geometría latente | Sólo reutilización y downstream | Cerrado |
| Gate 6 Exp C | A4 no mejoró Transkun en Exp A/B | Decoder serio sobre features VICReg congeladas | Residual |
| Escalón 2 | P2 y P3 sostienen un null descriptor-guided | Diagnóstico representacional P2 vs P3 | Foco |
| Voz Expresiva | N-adapt transfiere EN↔ZH; N-strict no | Diagnosticar N-strict o pasar a habla naturalista | Decisión |
| Escalón 3 | P5-cqtshift es el mejor brazo OOD actual; P6 puro no gana | Replicación, activation o transferencia | Reactivable |
| Atención Armónica | Pair-state importa; triangle ayuda OOD-poly con clusterer global | Mejor estimación de k/partición o CQT | Incubación |
| Escalón 4 | Existe como hipótesis fisiológica | Falta diseño experimental | Proyección |

## Dos vías científicas y una capa contextual

| Vía | Unidad experimental | Pregunta | Frentes |
|---|---|---|---|
| Descriptores | Features proporcionales + mecanismo de inyección | ¿La proporción explícita reorganiza y transfiere? | E1, E2, Voz, Gate 6 |
| Arquitectura nativa | Geometrías, pares, triángulos, particiones | ¿La red puede razonar proporcionalmente por construcción? | E3, Atención Armónica |
| Contexto agente | Wiki, memoria, relaciones entre evidencia y alternativas | ¿El conocimiento acumulado mejora la experimentación futura? | Capa metodológica transversal |
| Base de verdad | Invariantes, simuladores, cámaras y adjudicación | ¿Qué evidencia permite falsar una capacidad proporcional? | Investigación transversal PPU/NHG |

La tercera fila es una vía programática de trabajo, no evidencia científica ni
una afirmación ontológica sobre el mundo.

## Las bifurcaciones actuales

```mermaid
flowchart TD
    NOW["Estado actual"]
    NOW --> E2D["Cerrar diagnóstico P2 vs P3"]
    NOW --> G6C["Completar o cerrar Gate 6 Exp C"]
    NOW --> VED{"Voz: ¿diagnóstico o dominio naturalista?"}
    NOW --> AAD{"AA: ¿cabeza k/partición o CQT?"}
    NOW --> E3D{"E3: ¿replicación, activation o transferencia?"}

    VED --> VE12["Fase 1.2 N-strict"]
    VED --> MSP["MSP-Podcast / habla naturalista"]
    AAD --> KB["Stage B: predecir k o partición"]
    AAD --> CQT["Fase 1a: render → CQT → picos"]
    E3D --> REP["Replicar mejor brazo"]
    E3D --> ACT["Activation arena"]
    E3D --> PHY["Transferencia física"]
```

## Qué no está activo aunque parezca activo

| Línea | Por qué puede confundirse | Lectura correcta |
|---|---|---|
| Escalón 1 | Vive bajo `01_FRENTES_ACTIVOS/` | Tronco cerrado; Gate 6 Exp C es la excepción |
| EIR-EMR | Conserva README y roadmap propios | Antecedente superseded por Voz Expresiva |
| P6 toroidal | Tiene implementación y resultado | Hipótesis interesante, no ganadora en su receta actual |
| UOEMD/Rosetta | Conserva mucha documentación y código | Frente histórico cerrado |
| Roadmap Triplescaloneta v1.1 | Se presentaba como operativo | Esqueleto conceptual superado por la ejecución |

## Cómo navegar

- Para el contexto completo: [LLM_CONTEXT.md](LLM_CONTEXT.md).
- Para secuencia y dependencias: [current-portfolio.md](roadmaps/current-portfolio.md).
- Para cada frente: [index.md](index.md#frentes).
