---
schema_version: 1
id: front-atencion-armonica
kind: front
page_status: current
front_status: focus_active
updated: 2026-09-07
verified_at: 2026-09-07
valid_at: 2026-09-07
recorded_at: 2026-09-07
evidence_commit: 445847f6b43de1e54afa2c84c0ca3c7ca68e2e69
source_paths:
  - Documents/00_TRONCAL/ROADMAP_GENERAL/PROGRAMA_GEOMETRIA_ARMONICA_COMPUTABLE.md
  - experiments/atencion_armonica/PLAN_GEOMETRIC_RESEARCH_ACTION.md
  - experiments/atencion_armonica/RESULTS_ENERGY_PARTITION_AUDIT.md
  - experiments/atencion_armonica/PLAN_SHARED_PARTIAL_COMPATIBILITY.md
  - experiments/atencion_armonica/RESULTS_SHARED_PARTIAL_PREFLIGHT.md
  - experiments/atencion_armonica/RESULTS_SHARED_PARTIAL_GPU_PROFILE.md
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/ROADMAP_ATENCION_ARMONICA.md
depends_on: []
tangents: [front-escalon-3, ppu-natural-harmonic-geometry]
architecture_status: candidate
experiment_status: mixed
evidence_status: historical_multi_seed_iid_ood_and_cpu_diagnostics_not_new_neural_result
decision_status: pending_analysis
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

## Rebase vigente: geometría, arquitectura y pérdida

El [nuevo programa](../../00_TRONCAL/ROADMAP_GENERAL/PROGRAMA_GEOMETRIA_ARMONICA_COMPUTABLE.md)
toma este frente como banco inicial, no como arquitectura promovida. El
[diagnóstico CPU](../../../experiments/atencion_armonica/RESULTS_ENERGY_PARTITION_AUDIT.md)
recuperó una partición única y correcta por amplitudes solas en las ocho
mezclas seleccionadas de dos fuentes y las ocho de tres, también desde
log-amp float32. Los ocho casos de una fuente se cuentan aparte. Es una
muestra histórica no aleatoria, no atribución de ese uso a las redes.
Las métricas históricas se conservan; su gate per-par no excluía este atajo
global. El [contraste fijado](../../../experiments/atencion_armonica/PLAN_SHARED_PARTIAL_COMPATIBILITY.md)
retira amplitudes y compara pérdidas con red/descriptores comunes:
compatibilidad de una familia espectral, BCE sola, pesos desacoplados de
los triples y transitividad genérica. Añade baseline token-only y heurística
analítica. El núcleo está implementado y auditado; el
[preflight CPU](../../../experiments/atencion_armonica/RESULTS_SHARED_PARTIAL_PREFLIGHT.md)
completó sus guardas con descriptor no constante y gradiente físico no nulo
sin dominar BCE en el batch diagnóstico. El
[perfil GPU](../../../experiments/atencion_armonica/RESULTS_SHARED_PARTIAL_GPU_PROFILE.md)
ya midió recursos; la celda de entrenamiento está auditada. Falta terminar
coordinador/evaluación y ejecutar el contraste. No hay resultado neuronal
nuevo ni promoción.

## Bifurcaciones preservadas, no secuencia obligatoria

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
