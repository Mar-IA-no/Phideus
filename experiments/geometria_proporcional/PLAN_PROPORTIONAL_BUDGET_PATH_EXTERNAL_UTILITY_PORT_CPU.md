# Plan CPU — puerto externo de utilidad para `BudgetPath`

**Fecha:** 2026-09-04
**Estado:** implementación, corrida oficial y replay exacto completados
**Régimen:** contrato checker-only sobre fixtures sintéticos
**Fuente arquitectónica:** R373, manifest
`2e767e7e1a67afa97ac8295429e2a157452363f0fbd54a1ea42904e72b50886c`
**Autoridad:** prueba composición y aislamiento; no declara utilidad del usuario, política empírica, promoción ni GO/NO-GO

## Pregunta

`BudgetPath` termina donde empieza una preferencia. R374 prueba que una utilidad
puede conectarse después, sin mutar la ruta ni aparecer retroactivamente en sus
coordenadas. Como el usuario no declaró una utilidad, ningún caso usa R369–R373
como arena de decisión: todas las elecciones se realizan sobre fixtures
sintéticos con resultado esperado predeclarado.

## Binding y jurisdicción

Una declaración externa enlaza:

- `artifact_id` y SHA-256 canónico del `BudgetPath`;
- SHA-256 del receipt checker-valid;
- `utility_kind` y parámetros explícitos;
- `scope: SYNTHETIC_FIXTURE`;
- `tie_policy: RETURN_ALL_OPTIMA`;
- `decision_authority: SYNTHETIC_TEST_ONLY`.

La versión v1 rechaza `USER_DECLARED`, `EMPIRICAL` y equivalentes. Aceptarlos
sin una declaración real sería fabricar autoridad. El artefacto y el receipt
se hashean antes y después de evaluar; la salida vive en un objeto de decisión
separado.

## Utilidades monotónicas soportadas

1. `WEIGHTED_SUM_MINIMIZE`: pesos finitos, estrictamente positivos, completos y
   normalizados para IID/grouped;
2. `LEXICOGRAPHIC_MINIMIZE`: permutación explícita de los dos ejes;
3. `EPSILON_CONSTRAINT_MINIMIZE`: eje primario, eje restringido y cota finita.

Los candidatos son exactamente los IDs del reader `PARETO_SET`. Weighted y
lexicographic devuelven todos los óptimos dentro de `1e-12`. Epsilon devuelve
todos los óptimos factibles o `ABSTAIN_NO_FEASIBLE_POLICY`; no relaja la cota.

Estas familias no pretenden agotar toda utilidad posible. Sólo prueban que el
puerto admite tres mediaciones habituales sin esconder tie-breaks.

## Fixtures y propiedades

La suite materializa geometrías sintéticas con:

- frente de tradeoffs y decoys dominados;
- óptimo único por weighted sum;
- empate exacto de varios candidatos;
- prioridad lexicográfica;
- restricción epsilon factible;
- restricción epsilon sin solución;
- frente singleton.

Se comprueba además:

1. permutar el orden del reader no cambia el conjunto elegido;
2. alterar una política fuera del reader no cambia la decisión;
3. cada selección pertenece al reader;
4. artifact y receipt son byte-inmutables;
5. los empates no se rompen por presupuesto, orden ni ID.

## Suite adversarial congelada

Casos `PROTOCOL_INVALID`, todos con rechazo esperado:

1. hash de artefacto incorrecto;
2. hash de receipt incorrecto;
3. receipt no válido;
4. scope que finge utilidad del usuario;
5. peso negativo o cero;
6. pesos incompletos/no normalizados;
7. prioridad lexicográfica duplicada;
8. ejes epsilon idénticos o cota no finita;
9. tie-break oculto distinto de `RETURN_ALL_OPTIMA`;
10. campo `selected_policy_id` inyectado en la declaración;
11. reader con candidato desconocido;
12. coordenada no finita.

## Salida y estados

La decisión declara `SYNTHETIC_FIXTURE_ONLY`, conserva candidatos, óptimos y
evidencia de evaluación, y nunca se registra como recomendación. Un receipt
checker-invalid invalida la composición, no refuta la proposición empírica.

Output canónico:
`data/geometria_proporcional/proportional_budget_path_external_utility_port_v1/`.
Conservará fixtures, declarations, decisions, metamorphic checks, mutaciones,
summary, config, entorno, manifest y replay.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `2 min` y `2 GiB`.
No consulta ni usa GPU; todo trabajo CUDA permanece en cola.

## Ejecución cerrada

El diseño quedó fijado en `e089c6e` y la implementación en `3b94b1e`. La
regresión ampliada cerró `225/225` tests. La corrida oficial y su replay
terminaron en `0,0266/0,0278 s`, con `0,692/0,691 GiB` de RSS máximo. Los diez
productos deterministas y el manifest coincidieron byte por byte; el SHA-256
del manifest es:

```text
d9fe25b8d816b2a1bc8eef528bcfae2b409787542bf89bdba20f397eb0014876
```

Los resultados observados fueron `7/7` decisiones positivas iguales a las
esperadas, `3/3` propiedades metamórficas satisfechas y `12/12` declaraciones
inválidas rechazadas. El summary conserva
`historical_budget_paths_evaluated: 0`,
`user_utility_status: NOT_DECLARED`,
`decision_authority: SYNTHETIC_TEST_ONLY` y `gpu_queried: false`.

Este cierre acredita la mecánica del puerto sobre fixtures, no una utilidad
real ni una elección para las rutas R373. Activar la capa empírica requiere una
declaración externa auténtica y un freeze prospectivo; no corresponde inferir
ninguna de las dos cosas desde esta suite.
