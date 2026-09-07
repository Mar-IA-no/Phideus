# Reauditoría focal del plan corregido

## Dictamen: REVISE

Conteo residual: **1 HIGH / 2 MEDIUM / 0 LOW**.

Los once findings de R547 fueron sustancialmente corregidos. H1–H4, H6 y M1–M5 quedan resueltos; H5 está resuelto en su diseño target-blind, pero conserva una ambigüedad de soporte al agregar los cinco controles. La arquitectura macro de ambos factoriales es ahora válida, pero el finding HIGH impide congelar todavía la rama relacional.

## Findings residuales

### HIGH — El path-shuffle usa campos privados para construir un input causal

El plan declara `master_id` y `mechanism` dentro del sidecar privado, mientras el modelo sólo puede recibir topología, observación, máscaras, paths y varianza públicos ([plan:111-114](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:111)). Sin embargo, la seed del path-shuffle depende de ambos campos privados:

```text
SHA256("rel-path-v1" || master_id || mechanism || training_seed)
```

([plan:130-136](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:130)).

Aunque el preparer materializara el shuffle antes del trainer, el tensor público del control cambiaría al mutar `mechanism`. Eso crea una dependencia private→public y contradice `R1_PUBLIC_PRIVATE_SCHEMA`, la invariancia privada y la afirmación de que el control se construye sin autoridad causal privada. El checker propuesto tampoco enumera esta mutación específica ([plan:502-511](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:502)).

Corrección concreta: derivar la seed sólo de información pública y estable, por ejemplo:

```text
SHA256("rel-path-v1" || public_lineage_pseudonym || path_structure_digest || training_seed)
```

o reutilizar la lógica vigente basada en el digest estructural que excluye la relación observada ([run_proportional_graph_neural_smoke.py:379-386](experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py:379)). Añadir mutaciones de `master_id`, `mechanism` y demás campos privados que exijan byte-identidad del roster y tensors públicos.

### MEDIUM — El algoritmo de target-shuffle set-valued aún no está definido de forma única

La nueva estratificación `(fold_id, design_stratum, cardinality)` y el umbral permutable corrigen la imposibilidad de R547 ([plan:349-357](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:349)). Pero “una asignación de costo por hash” no fija:

- bytes exactos que se hashean;
- matriz de costos;
- algoritmo de asignación;
- desempate;
- relación efectiva entre `PCG64(53602)` y el costo hash.

Dos implementaciones pueden producir mapas distintos y seguir el texto.

Corrección concreta: congelar pseudocódigo total o nombrar una función nueva con serialización, matriz, solver y desempate exactos. La fixture debe incluir un estrato con múltiples derangements óptimos y exigir un SHA-256 único.

### MEDIUM — `READER_CONTROL` no define soporte común entre cinco matchings

El matching target-blind por posterior/seed/token está ahora bien especificado y se congela antes de abrir targets ([plan:391-418](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:391)). Sin embargo, cada uno de los cinco controles puede marcar `MATCH_IMPOSSIBLE` en tokens diferentes. La tabla luego pide “media por token de los cinco matched controls” ([plan:462](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:462)), pero sólo exige cobertura `>=0.8` separadamente en cada seed ([plan:415-417](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:415)). Eso no determina si se usa intersección, unión o promedio con cantidad variable de controles.

Corrección concreta: fijar como soporte confirmatorio la intersección de tokens con `k>0` y match válido en los cinco controles, persistir las cinco máscaras y la máscara común, y aplicar el `>=0.8` a esa intersección respecto de los tokens verdaderos con `k>0`. Alternativamente, definir cinco contrastes por separado y una agregación predeclarada, sin promediar faltantes.

## Estado de R547

| Finding R547 | Estado |
|---|---|
| H1 pesos crudos/normalizados incompatibles | Resuelto en [plan:150-165](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:150) |
| H2 K64 unit-base presentado como base-weighted | Resuelto mediante conformance nueva en [plan:174-196](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:174) |
| H3 shuffle por `cluster_id` imposible | Resuelto en [plan:349-357](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:349), sujeto al finding MEDIUM de algoritmo |
| H4 tabla de decisión incompleta | Resuelto en [plan:267-286](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:267) y [plan:448-470](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:448) |
| H5 matching post-target/subespecificado | Leakage resuelto; queda ambigüedad MEDIUM de soporte |
| H6 `READY_FOR_EXECUTION` indebido | Resuelto como `READY_FOR_RUNNER_IMPLEMENTATION` en [plan:37-53](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:37) |
| M1 falsa neutralidad `0.5/0.5` | Resuelto en [plan:163-165](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:163) |
| M2 incertidumbre de seeds | Resuelto en [plan:249-255](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:249) |
| M3 denominadores WLS/IRLS | Resuelto en [plan:215-230](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:215) |
| M4 receta MARGINAL/JOINT | Resuelto en [plan:323-347](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:323) |
| M5 roster path-shuffle | Resuelto en soporte/elegibilidad en [plan:130-146](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:130), salvo la dependencia privada HIGH |

Con esas tres correcciones, el plan queda en condiciones de una reauditoría focal corta y no requiere reabrir ninguna fuente sellada ni ampliar el programa.
