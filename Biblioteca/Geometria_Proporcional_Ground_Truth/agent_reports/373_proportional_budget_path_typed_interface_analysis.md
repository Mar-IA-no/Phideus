# R373 — Interfaz tipada `BudgetPath`

**Fecha:** 2026-09-04
**Estado:** oficial y replay exacto completados
**Régimen:** materialización checker-only sobre evidencia abierta R369–R372
**Arquitectura:** candidata separada; baseline sin cambios
**Autoridad:** acredita estructura y lineage; no utilidad, seguridad, generalización, promoción ni GO/NO-GO

## Problema

R371 conservó un frente de políticas y R372 mostró que su membresía podía
ocultar cambios en la relación que la producía. Una lista de IDs no basta para
transportar esa evidencia a otra capa arquitectónica: debe conservar acciones,
coordenadas, incertidumbre, relaciones pareadas y frontera de autoridad sin
convertirlas en una recomendación.

R373 materializa esa unidad como `BudgetPath`. No vuelve a analizar el
histórico ni aprende un selector. Convierte los estados ya abiertos en un
objeto recuperable y chequeable.

## Cobertura y separación

Se construyeron `608` artefactos:

```text
2 cohortes × 2 propuestas × 4 brazos × 2 roles ×
(3 familias principales + 16 réplicas permutadas)
```

La distribución es `32` reduced, `32` public-base, `32` topology y `512`
topology-permuted. Cada artefacto contiene siete políticas y 21 pares, con
hashes que enlazan acciones R369, objetivos/frentes R371 e incrementos R372.
Los arrays pesados permanecen en sus fuentes y no se duplican.

Builder y checker están separados. El builder ensambla los tres informes; el
checker no importa el builder ni la lógica Pareto/dominancia previa. Reabre los
NPZ y reconstruye independientemente orden, ejes, anidamiento, hashes, medias,
frente, frecuencias bootstrap, pares y reader.

Los `608/608` receipts quedaron `VALID`. Eso significa que el objeto representa
fielmente la evidencia fuente bajo el schema declarado. No significa que las
608 políticas o frentes sean científicamente válidos.

## Frontera de decisión

La salida canónica no contiene `selected_policy_id`, `utility_weight`,
`scalar_score` ni `recommendation`. Declara explícitamente:

```text
artifact_status: CHECKABLE_CANDIDATE
formal_claim_status: STRUCTURE_ONLY
empirical_claim_status: OPENED_POSTHOC
physical_authority_status: NOT_CLAIMED
decision_status: UNRESOLVED
utility_status: ABSENT_EXTERNAL_REQUIRED
```

El reader devuelve `PARETO_SET`. Una utilidad futura deberá entrar por otro
contrato y dejar trazabilidad propia; no puede reescribir el artefacto ni
presentarse como si hubiera estado implícita.

## Suite adversarial

Las ocho mutaciones `PROTOCOL_INVALID` fueron rechazadas:

| Mutación | Señal de rechazo |
|---|---|
| política seleccionada inyectada | schema extra |
| utilidad embebida | frontera de utilidad |
| ejes IID/grouped alterados | contrato de ejes |
| hash de acción alterado | fuente de acción |
| media de objetivo alterada | recomputación de objetivo |
| membresía al frente alterada | recomputación Pareto |
| estado pareado alterado | recomputación de dominancia |
| manifest fuente alterado | lineage |

Una auditoría adicional endureció tipos: `false` ya no puede hacerse pasar por
la fracción numérica cero ni `1` por un booleano de adyacencia. La regresión
incluye además una ruta sintética no anidada.

## Integridad

Diseño `443760b`, implementación `239d363`. Oficial y replay terminaron en
`163,513/162,829 s`, con `0,691/0,691 GiB`, un thread y CUDA invisible. No hubo
fit, vistas, solves ni bootstraps nuevos; `gpu_queried: false`. El manifest
byte-idéntico es:

```text
2e767e7e1a67afa97ac8295429e2a157452363f0fbd54a1ea42904e72b50886c
```

Coinciden `7/7` deterministas, `608` líneas de artefactos y `608` receipts. La
regresión ampliada cerró `214/214`.

## Lectura

**Observación.** La evidencia R369–R372 puede representarse sin pérdida
mecánica como una ruta tipada, set-valued y sin utilidad embebida. Los tamper
tests muestran que el checker distingue alteraciones de estructura, lineage y
decisión.

**Hipótesis.** `BudgetPath` es una frontera arquitectónica más honesta que un
selector binario: separa el proposer de rutas, el reader Pareto y una futura
política externa. También permite que otra realización cambie relaciones o
frente sin cambiar la semántica del objeto.

**Inferencia acotada.** R373 acredita una interfaz, no la primitive empírica que
transporta. No promueve topology, no elige `40%` y no resuelve la inestabilidad
R372. El siguiente trabajo CPU admisible es especificar y probar con fixtures
sintéticos el puerto de utilidad externa, sin aplicarlo al histórico hasta que
el usuario declare una utilidad. La confirmación empírica requiere un freeze
prospectivo dimensionado por R370. Toda ejecución GPU permanece en cola.

Artefactos: schema
`src/geometria_proporcional/budget_path_schema.py`, checker
`src/geometria_proporcional/budget_path_checker.py`, plan
`experiments/geometria_proporcional/PLAN_PROPORTIONAL_BUDGET_PATH_TYPED_INTERFACE_CPU.md`,
runner
`experiments/geometria_proporcional/run_proportional_budget_path_typed_interface.py`,
oficial
`data/geometria_proporcional/proportional_budget_path_typed_interface_v1/` y
replay con sufijo `_replay`.
