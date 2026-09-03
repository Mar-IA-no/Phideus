# Reauditoria focal de replay - Ola 52

> Estado: `REVISE`
> Fecha: 2026-09-03
> Instancia independiente: Kant (`01a06641-334a-71b3-93bd-18f0ecb47f08`)

## Finding verbatim

**HIGH:** el replay sigue incompleto respecto del contrato enmendado. Las
acciones per-seed se reconstruyen pero se descartan al retornar solo acciones
ensemble; por eso el replay de metricas cubre unicamente agregados ensemble.
`seed_sensitivity` y `metrics_by_policy.jsonl` no se recomponen ni comparan
desde checkpoints, y el contrato solo exige conteo de filas mas flags globales.
Esto contradice el replay ensemble y por seed comprometido en el plan.

`tau` sobre ensemble token-level y el IC de `worst_restricted_regret` quedaron
confirmados como resueltos. Veredicto: `REVISE`.
