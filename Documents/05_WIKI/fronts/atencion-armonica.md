---
schema_version: 1
id: front-atencion-armonica
kind: front
page_status: current
front_status: incubated
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 480c7ef4dddbfe8dfe92459e80ee0ae97b765f8c
source_paths:
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/ROADMAP_ATENCION_ARMONICA.md
depends_on: []
tangents: [front-escalon-3, ppu-natural-harmonic-geometry]
architecture_status: incubated
experiment_status: phase_closed
evidence_status: multi_seed_iid_ood
decision_status: pending_user
---

# Atención Armónica

## Resumen

Atención Armónica ensaya una arquitectura relacional para agrupar parciales de
fuentes polifónicas. Los picos son nodos; los estados de par describen posible
pertenencia común; el triangle update propaga consistencia a través de terceros
picos; un clusterer convierte la matriz de relaciones en una partición.

## Estado real

Fases 0, 0.5 y 0.6 están cerradas. El pair-state es el salto principal.
Comparado con B-local param-matched, el triangle no domina IID ni OOD-regime,
pero mejora OOD-poly. B-shuffle confirma que la estructura del triángulo importa.

Fase 0.5 mostró que el cuello no era el umbral `τ`, sino
connected-components. Fase 0.6 mostró que spectral y agglomerative con `k`
estimado extraen de forma deployable parte de la ventaja de B. El estimador
subestima `k`, por lo que la partición todavía no está resuelta.

## Bifurcación pendiente

| Camino | Qué aísla |
|---|---|
| Stage B: cabeza de `k/partición` | Si el cuello residual puede aprenderse sobre Pairformer congelado |
| Fase 1a: render→CQT→picos | Si la ventaja sobrevive a errores de detección manteniendo GT exacto |

## Alcance

El resultado sostiene una ventaja específica de generalización OOD-poly, no la
afirmación de que el triangle gane universalmente ni que ya exista una geometría
armónica completa.

## Relaciones

- [Escalón 3](escalon-3.md)
- [PPU / Natural Harmonic Geometry](../concepts/ppu-geometria-armonica-natural.md)

## Fuentes

- [README canónico](../../01_FRENTES_ACTIVOS/Atencion_Armonica/README.md)
- [Roadmap local](../../01_FRENTES_ACTIVOS/Atencion_Armonica/ROADMAP_ATENCION_ARMONICA.md)
- [Explicación de Fase 0.6](../../01_FRENTES_ACTIVOS/Atencion_Armonica/Explicacion_fase_0_6_clusterer_deployable_codex.md)
