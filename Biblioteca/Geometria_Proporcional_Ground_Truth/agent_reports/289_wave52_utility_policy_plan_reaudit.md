# Reauditoría independiente del plan revisado de Ola 52

> **Auditor:** instancia independiente `Averroes`
> **Fecha:** 2026-09-03
> **Objeto:** plan revisado de transporte de política ordinal
> **Veredicto:** `REVISE`

## Findings sustantivos

1. **Alto: posible leakage en selección de readers.** `policy_val` selecciona
   readers, pero no se define qué `pair_token` usa ni se exige su disjunción
   respecto de `val_monitor`. Separar políticas no evita adaptación a las mismas
   unidades observacionales. Debe fijarse una partición de tokens independiente
   y verificable por hash.

2. **Alto: el control negativo del criterio está confundido y resulta casi
   trivial.** `explicit_set_policy` opera sobre el conjunto predicho, mientras
   `oracle_set_wrong_context` usa el conjunto verdadero. El criterio los llama
   “el mismo reader”, pero cambia simultáneamente contexto y conjunto. Además,
   como el contexto deranged garantiza otro ganador conservando el target
   original, el control oracle tendrá exactitud nula por construcción en la
   población elegible. No demuestra sensibilidad contextual del sistema
   evaluado.

## Revisión de la primera ronda

Los `10/10` findings de R288 quedaron resueltos en su formulación original:

| Finding | Estado |
|---|---|
| atribución causal/factorización | `RESUELTO` |
| regret incompatible | `RESUELTO` |
| población policy-sensitive | `RESUELTO` |
| intervención de utilidad en evaluación | `RESUELTO` |
| comparabilidad y normalización BCE/CE | `RESUELTO` |
| threshold independiente de política | `RESUELTO` |
| folds, seeds y unidad inferencial | `RESUELTO` |
| selección post hoc de reader | `RESUELTO` |
| definición de `sigmoid_only@60` | `RESUELTO` |
| alcance ordinal y partición `8/8/8` | `RESUELTO` |

## Cierre

La revisión resolvió la primera ronda, pero los dos findings nuevos requieren
corrección antes de implementar. `REVISE`.
