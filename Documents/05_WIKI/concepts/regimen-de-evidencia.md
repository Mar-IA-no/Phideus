---
schema_version: 1
id: phideus-evidence-regime
kind: concept
page_status: current
front_status: transversal
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 480c7ef4dddbfe8dfe92459e80ee0ae97b765f8c
source_paths:
  - AGENTS.md
  - MARCO_EPISTEMOLOGICO_PHIDEUS.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md
depends_on: []
tangents: [phideus-three-routes]
---

# Régimen de evidencia de Phideus

## Principio

La wiki debe mantener separadas observación, hipótesis, inferencia y decisión.
Una arquitectura plausible no es un resultado; un resultado en un slice no es
una ley general; una decisión de prioridad no convierte una hipótesis en verdad.

## Jerarquía práctica

| Régimen | Qué acredita | Qué no acredita por sí solo |
|---|---|---|
| Test mecánico | integridad, forma, determinismo, invariantes del código | validez científica |
| Single run/checkpoint | observación exploratoria | estabilidad |
| Multi-seed pareado | estabilidad dentro del protocolo | transferencia de dominio |
| OOD predeclarado | generalización al shift medido | universalidad |
| Cross-domain/language | recurrencia entre dominios concretos | identidad del mecanismo causal |
| Ablación param-matched | atribución más limpia | ontología fuerte |
| Lectura cualitativa completa | coherencia semántica/contextual | métrica objetiva |

## Nulls

Un null es informativo cuando el baseline es sano, la intervención tuvo una ruta
real de efecto, el presupuesto fue comparable, el protocolo fue suficientemente
estable y las métricas tenían rango. Escalón 2 cumple buena parte de estas
condiciones, pero su diagnóstico P2/P3 sigue abierto precisamente para localizar
el null antes de generalizarlo.

## Resultados negativos

Las líneas negativas se preservan con su alcance. DANN, Shazam, Transkun+A4,
P6 toroidal puro o connected-components no se convierten en descartes de toda
su familia conceptual: delimitan recetas, mecanismos y condiciones específicas.

## Autoridad

GO/NO-GO, promoción y prioridad pertenecen al usuario. La wiki registra esas
decisiones cuando existen; no las fabrica a partir de umbrales retrospectivos.
