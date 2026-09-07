# Auditoría independiente — `PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md`

## Dictamen: REVISE

Conteo: **6 HIGH / 5 MEDIUM / 0 LOW**.

La bifurcación en dos factoriales nativos es conceptualmente consistente con `MAPPING-FEASIBILITY`: no vuelve a fusionar unidades, targets ni métricas incompatibles. Sin embargo, el freeze todavía no es materializable ni permite adjudicar `READY_FOR_EXECUTION`. Hay dos contradicciones matemáticas/algorítmicas en la rama relacional, un control set-valued imposible bajo el schema vigente y varias reglas confirmatorias o de control aún indeterminadas.

No se consultó GPU, monitor, `sealed_monitor` ni lockbox; no se modificaron archivos.

## Findings HIGH

### H1 — El contrato de `reliability` es matemáticamente degenerado

El plan exige simultáneamente:

- `0 < w <= 1`;
- `mean(valid w)=1` después de normalización.

Eso implica necesariamente `w=1` para toda arista válida, eliminando cualquier reliability aprendida ([plan:134-139](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:134)). La implementación vigente confirma que la normalización a media uno puede producir pesos mayores que uno: recorta por piso y divide por la media ([proportional_graph_contract.py:513-523](src/geometria_proporcional/proportional_graph_contract.py:513)). La cabeza neuronal, en cambio, produce reliability cruda dentro de `(floor,1)` antes del executor ([proportional_graph_neural.py:345-358](src/geometria_proporcional/proportional_graph_neural.py:345)).

Corrección obligatoria: congelar dos objetos distintos:

- `raw_reliability ∈ [weight_floor,1]`;
- `normalized_executor_weight > 0`, con media válida uno y sin cota superior uno.

Persistir ambos y denominarlos pesos relativos, no reliability calibrada.

### H2 — R354 no certificó el surrogate K64 con reliability aprendida

El plan afirma que WLS y K64-IRLS reciben la misma reliability y atribuye la validez de esa operación a R354 ([plan:142-154](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:142)). Eso no coincide con la evidencia:

- R354 comparó IRLS unit-base; sus tres objetos fueron executor convergido, referencia fixed-K y surrogate fixed-K ([R354:27-35](Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/354_proportional_irls_surrogate_fidelity_official_analysis.md:27)).
- El surrogate vigente no acepta `base_weights` y comienza con unos ([proportional_graph_neural.py:575-612](src/geometria_proporcional/proportional_graph_neural.py:575)).
- El executor canónico sí normaliza `base_weights` aprendidos y los multiplica por los pesos Huber ([proportional_graph_contract.py:555-583](src/geometria_proporcional/proportional_graph_contract.py:555), [proportional_graph_contract.py:619-629](src/geometria_proporcional/proportional_graph_contract.py:619)).
- R354 limitó explícitamente la conformidad a validation IID y a un modelo no entrenado mediante el surrogate ([R354:104-118](Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/354_proportional_irls_surrogate_fidelity_official_analysis.md:104)).

Corrección obligatoria: implementar y certificar independientemente un `fixed_K64_IRLS(base_weights=raw_reliability)` que replique normalización, objetivo y update canónicos. Validar valor y gradiente con pesos no unitarios, en el dtype real de entrenamiento y sobre el rango de grafos congelado. Hasta entonces, R354 no puede ligarse como evidencia suficiente de `R4_DUAL_SOLVER_LOSS`.

### H3 — `JOINT-TARGET-SHUFFLED` es imposible con el estrato declarado

El plan exige derangement dentro de `(cluster_id, cardinality)` ([plan:257-259](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:257)). En la frontera Wave 59 que pretende reutilizar, `cluster_id` se materializa exactamente como `pair_token` ([\_wave56_phase_worker.py:451-478](experiments/geometria_proporcional/_wave56_phase_worker.py:451)), y `pair_token` es único. Por tanto, cada estrato `(cluster_id, cardinality)` es singleton y no admite derangement.

Además, el contrato adjudicado de `MAPPING-FEASIBILITY` usaba `(split_role, design_stratum, cardinality)`, no `cluster_id` ([PLAN_MAPPING_FEASIBILITY:306-313](experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md:306)).

Corrección obligatoria: restaurar un estrato materializable, preferiblemente incluyendo `fold_id` para impedir que un target donante atraviese folds: `(fold_id, design_stratum, cardinality)`. Congelar una misma permutación para ambos posteriors, registrar singletons/fracción permutable y declarar `NOT_EVALUABLE` si no alcanza el soporte mínimo.

### H4 — Los rótulos confirmatorios no tienen una tabla de decisión ejecutable

`PATTERN_PRESENT` depende de expresiones no definidas numéricamente:

- “sin desigualdad de fallos” ([plan:209-213](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:209));
- “Brier no adverso” y “separación del target-shuffled”;
- “compatibilidad no adversa”, “worst regret no adverso” y “separación de controles matched” ([plan:332-337](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:332)).

No se congelan márgenes, CI aplicable, igualdad estricta/no estricta, multiplicidad ni propagación de `NOT_EVALUABLE`. Así, dos implementaciones conformes pueden emitir rótulos distintos.

Corrección obligatoria: añadir una tabla total por estimando con orientación, estadístico, intervalo, margen de no inferioridad, soporte, regla de faltantes, familia confirmatoria y razón cerrada para cada salida.

### H5 — El control matched de acciones de monitor está subespecificado y permite selección post-target

Se exige igual número de overrides y Hamming no menor, con cobertura del 80%, pero no se define algoritmo, semilla, universo, estrato, desempate, referencia del Hamming ni fase exacta de construcción ([plan:298-302](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:298)). Como la separación de ese control es condición del patrón ([plan:332-337](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:332)), elegir entre matchings después de ver targets permitiría leakage o cherry-picking.

Corrección obligatoria: congelar un algoritmo determinista y target-blind ejecutado por `monitor applier`, usando únicamente acciones/scores públicos congelados. Fijar seeds, count matching por token/policy, objetivo Hamming, desempate, mapeo persistido y razón cerrada de imposibilidad.

### H6 — El checker no puede demostrar `READY_FOR_EXECUTION` con los artefactos previstos

Este goal declara que no implementará ni ejecutará training/forward/draw ([plan:67-70](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:67)). Su inventario incluye configs, checker y preflight, pero ningún runner de las dos ramas ([plan:437-450](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:437)). Fixtures declarativas pueden validar schemas y álgebra; no pueden probar que un runner inexistente respete pérdidas, accesos de fase, controles, preservación o fallos.

Corrección obligatoria: o bien:

- renombrar los estados a `DESIGN_FREEZE_VALID / READY_FOR_RUNNER_IMPLEMENTATION`; o
- incluir executors/dry-runs mínimos de ambas ramas, con barreras físicas de acceso y controles ejercitados sobre datos sintéticos, antes de emitir `READY_FOR_EXECUTION`.

Esto puede hacerse sin abrir monitor.

## Findings MEDIUM

### M1 — Coeficientes `0.5/0.5` no prueban neutralidad entre solvers

El plan justifica pesos iguales como forma de “no privilegiar WLS” ([plan:141-156](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:141)). Igual coeficiente no implica igual contribución por diferencias de escala y gradiente. R355 ya mostró que, aun igualando inicialmente normas, las trayectorias divergieron (`0,0616` frente a `0,0331`) ([R355:74-83](Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/355_proportional_irls_loss_contrast_official_analysis.md:74)).

Corrección: describirlo como objetivo compuesto fijo, no solver-neutral. Congelar y reportar por batch valor y norma de gradiente de cada término; si se recalibran coeficientes, hacerlo sólo con train y antes del freeze.

### M2 — La inferencia sobre seeds es ambigua y puede subestimar incertidumbre

El promedio entre seeds precede al delta y al bootstrap por master ([plan:193-197](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:193)), pero seeds individuales y “ensemble” aparecen como secundarios ([plan:199-207](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:199)). Bootstrappear sólo masters condiciona la inferencia a tres inicializaciones fijas; no estima variabilidad de entrenamiento.

Corrección: elegir explícitamente entre:

- estimando de ensemble congelado, combinando outputs antes del executor; o
- efecto medio sobre inicializaciones, con inferencia jerárquica/two-way o requisito de dirección por seed.

No presentar el CI por masters como generalización sobre seeds.

### M3 — La regla de soporte mezcla indebidamente WLS, IRLS e interacción

El soporte común de cuatro celdas y el umbral del 99% se aplican al “contraste post-solver” completo ([plan:170-176](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:170)). Un fallo IRLS no debería retirar ese master del contraste WLS, que sigue definido.

Corrección: separar denominadores:

- WLS `TYPED-GENERIC`: todos los masters válidos;
- IRLS: soporte convergido común y failure outcome completo;
- interacción WLS/IRLS: soporte de las cuatro celdas.

Definir exactamente denominador del 99% y criterio de “desigualdad de fallos”.

### M4 — El posterior MARGINAL y su regularización no están congelados con precisión suficiente

El plan dice “cuatro calibradores Platt” y una grilla común con JOINT ([plan:245-255](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:245)). La implementación vigente de W53 ajusta un solo LogisticRegression pooled sobre todas las familias ([run_wave53_uncertainty_policy.py:286-310](experiments/geometria_proporcional/run_wave53_uncertainty_policy.py:286)); JOINT usa directamente un coeficiente `regularization` en su objetivo ([wave54_joint_set.py:87-127](src/geometria_proporcional/wave54_joint_set.py:87)). No está definido si la nueva grilla representa `C`, `lambda`, su inversa, ni qué ocurre ante clase ausente o no convergencia por familia/fold.

Corrección: fijar por escrito objetivo, parametrización, solver, tolerancias, intercept, penalización, folds, agregación Brier/NLL y fallos. La grilla puede compartir seis niveles, pero debe mapearse explícitamente a una fuerza regularizadora comparable.

### M5 — El path-shuffle relacional no está materializado para el roster completo

El plan declara derangement “por master y estrato” sin definir estrato ni algoritmo ([plan:126-130](experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:126)). El helper vigente opera dentro de cada vista y puede devolver `path_shuffle_eligible=false` ([proportional_graph_neural.py:79-133](src/geometria_proporcional/proportional_graph_neural.py:79)); el smoke anterior excluyó 26 vistas por esta causa ([R342:11-17](Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/342_proportional_graph_neural_smoke_official_analysis.md:11)). El nuevo freeze, en cambio, promete conteos exactos de masters sin regla de exclusión.

Corrección: congelar el algoritmo exacto, unidad y estratos del shuffle, demostrar que conserva tensor/budget en todos los masters o declarar previamente cómo se forma un soporte común sin alterar los splits.

## Condición para PASS en reauditoría

Resolver H1–H6 en el texto y trasladar sus invariantes a configs/mutaciones. Después, verificar focalmente:

1. surrogate K64 base-weighted contra referencia independiente;
2. shuffle set-valued con fracción permutable positiva;
3. tabla total de rótulos;
4. control matched target-blind;
5. semántica honesta de readiness;
6. denominadores e inferencia de seeds congelados.

La arquitectura macro de dos factoriales separados puede conservarse; los findings no exigen reabrir el factorial común ni consultar monitor/GPU.
