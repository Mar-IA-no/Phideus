# R374 — Puerto externo de utilidad para `BudgetPath`

**Fecha:** 2026-09-04
**Estado:** oficial y replay exacto completados
**Régimen:** composición checker-only sobre fixtures sintéticos
**Arquitectura:** candidata separada; baseline y rutas históricas sin cambios
**Autoridad:** prueba mecánica sintética; no utilidad del usuario, decisión empírica, promoción ni GO/NO-GO

## Problema

R373 terminó deliberadamente con una ruta set-valued y una decisión no
resuelta. Esa separación sólo resulta útil si una preferencia posterior puede
componerse sin reescribir el artefacto, fingir que la utilidad ya estaba
presente o escoger silenciosamente entre empates. R374 construye y verifica ese
puerto. Como no existe una utilidad declarada por el usuario, no lo aplica a
ninguno de los `608` `BudgetPath` históricos: los siete casos de elección viven
en dos fixtures sintéticos con respuesta esperada predeclarada.

## Contrato y binding

Cada declaración enlaza el SHA-256 canónico del artefacto y el SHA-256 de un
receipt checker-valid. El puerto vuelve a verificar que el lineage sea
`SYNTHETIC_FIXTURE`, que los ejes sean exactamente IID/grouped y que artefacto,
receipt y conjunto candidato pertenezcan a la fixture canónica. De este modo,
un artefacto histórico no puede adquirir jurisdicción sintética por el solo
expediente de acompañarlo con un receipt fabricado.

La evaluación produce un objeto separado; los hashes de entrada se verifican
antes y después. El scope aceptado es únicamente `SYNTHETIC_FIXTURE`, la
autoridad es `SYNTHETIC_TEST_ONLY` y la política de empate es siempre
`RETURN_ALL_OPTIMA`.

## Familias y decisiones positivas

La suite cubre tres familias monotónicas:

| Caso | Declaración | Resultado esperado y observado |
|---|---|---|
| P01 | weighted IID/grouped `0,8/0,2` | `budget_0.05` |
| P02 | weighted `0,5/0,5` | empate exacto: `budget_0.10`, `budget_0.20`, `budget_0.40` |
| P03 | lexicográfica, IID primero | `budget_0.05` |
| P04 | lexicográfica, grouped primero | `budget_0.40` |
| P05 | epsilon factible | `budget_0.20` |
| P06 | epsilon sin candidato factible | `ABSTAIN_NO_FEASIBLE_POLICY` |
| P07 | frente singleton | `budget_0.10` |

Las siete decisiones coincidieron con lo predeclarado. Weighted sum exige
pesos finitos, positivos, completos y normalizados; lexicographic exige una
permutación exacta de ejes; epsilon conserva una cota finita y no la relaja si
el conjunto factible queda vacío. Los candidatos proceden exclusivamente del
reader `PARETO_SET`.

## Controles y adversariales

Las tres propiedades metamórficas pasaron:

1. permutar el orden del reader conserva el conjunto elegido;
2. agregar o alterar un decoy dominado fuera del reader no cambia la decisión;
3. artefacto y receipt permanecen byte-inmutables.

Las `12/12` declaraciones `PROTOCOL_INVALID` fueron rechazadas: hash de
artefacto o receipt incorrecto, receipt inválido, scope que fingía autoridad
del usuario, pesos nulos/negativos/incompletos/no normalizados, prioridad
lexicográfica duplicada, ejes epsilon coincidentes o coordenada/cota no finita,
tie-break oculto, selección inyectada, y candidato desconocido. La regresión
unitaria también cubre por separado los subcasos reunidos bajo una misma
familia adversarial.

## Integridad y recursos

Diseño `e089c6e`, implementación `3b94b1e`. La regresión ampliada cerró
`225/225`. Oficial y replay terminaron en `0,0266/0,0278 s`, con
`0,692/0,691 GiB` de RSS máximo, un thread y CUDA invisible. Coinciden byte por
byte los diez productos deterministas y el manifest; este último tiene
SHA-256:

```text
d9fe25b8d816b2a1bc8eef528bcfae2b409787542bf89bdba20f397eb0014876
```

El output conserva dos fixtures, siete declaraciones y siete decisiones. Su
summary registra `historical_budget_paths_evaluated: 0`,
`user_utility_status: NOT_DECLARED` y `gpu_queried: false`.

## Lectura

**Observación.** Una preferencia externa puede componerse con una ruta
set-valued sin mutarla, devolver todos los empates y abstenerse cuando una
restricción epsilon no admite políticas. La suite distingue además bindings,
scopes y declaraciones inválidas.

**Hipótesis.** La distribución más limpia de responsabilidades mantiene
`BudgetPath` como evidencia estructural inmutable y aloja declaración,
evaluación y receipt de decisión en una capa posterior. Si alguna vez se
activa empíricamente, esa capa necesitará ownership y attestation explícitos,
además del freeze prospectivo dimensionado por R370.

**Inferencia acotada.** R374 cierra la incertidumbre mecánica del puerto, no la
incertidumbre normativa. No acredita que weighted, lexicographic o epsilon
representen la utilidad del usuario, no permite seleccionar una política en
R373 y no vuelve estable la relación abierta por R372. Fabricar una preferencia
para avanzar produciría una decisión formalmente válida pero sin autoridad.
Por eso la activación empírica queda pendiente de una declaración auténtica;
las pruebas que requieran GPU permanecen en cola.

Artefactos: plan
`experiments/geometria_proporcional/PLAN_PROPORTIONAL_BUDGET_PATH_EXTERNAL_UTILITY_PORT_CPU.md`,
implementación `src/geometria_proporcional/budget_path_utility.py`, runner
`experiments/geometria_proporcional/run_proportional_budget_path_external_utility_port.py`,
oficial
`data/geometria_proporcional/proportional_budget_path_external_utility_port_v1/`
y replay con sufijo `_replay`.
