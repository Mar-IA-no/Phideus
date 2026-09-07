# Auditoría independiente del plan del runner CPU set-valued nativo

Fecha: 2026-09-04  
Objeto auditado: `experiments/geometria_proporcional/PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md`  
Modalidad: lectura completa del plan vigente y contraste mínimo-contexto contra freeze, plan dual, plan de mapping, código y artefactos abiertos ligados explícitamente  
Restricciones respetadas: sin GPU/CUDA; sin abrir fuentes, bundles ni contenidos de monitor, lockbox o sealed

## Veredicto

**REVISE — 2 HIGH / 5 MEDIUM / 0 LOW.**

El plan es fuerte en separación lógica de fases, recipes de los cuatro brazos, serialización portable, replay y límites de claims. Sin embargo, todavía no determina una implementación fiel al freeze. Hay dos defectos de contrato centrales: JOINT-SHUFFLED recibe una decisión de hiperparámetro elegida con el target real, contradiciendo la receta congelada del control, y el supuesto runner completo no implementa los estimandos, bootstrap ni filas de decisión que el freeze exige. Cinco ambigüedades adicionales afectan unicidad del matching, diversidad efectiva de controles, independencia del checker, adjudicación de presupuesto y la promesa de ejecución sin importar `torch`.

Este veredicto no es un GO/NO-GO científico ni promueve una arquitectura. Indica que el plan necesita correcciones antes de autorizar su implementación como materialización completa del freeze.

## Findings

### H1 — JOINT-SHUFFLED hereda una lambda elegida con el target real y deja de ejecutar la receta JOINT congelada

**Evidencia.** El brazo JOINT real selecciona entre seis lambdas con cuatro folds y una clave OOF, y luego reajusta sobre las 192 filas (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:136-154`). Pero el control declara expresamente que `JOINT-SHUFFLED` conserva la lambda elegida por el brazo real y no reabre la selección (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:165-167`). El freeze vigente define la representación JOINT mediante grid, folds y clave de selección, sin una excepción para el shuffled (`configs/proportional_set_valued_native_freeze_v1.json:41-44`). El plan dual es aún más explícito: JOINT selecciona su lambda dentro de `posterior_fit` (`PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:348-360`), y el mismo target permutado se usa durante validación interna y refit (`PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:362-390`).

**Impacto.** El control ya no es la misma tubería de estimación con el target reemplazado. Transporta al brazo shuffled una decisión de tuning obtenida del target real. Esto define, como máximo, un control condicionado a la lambda del brazo real; no es el `JOINT-JOINT_SHUFFLED` congelado. La comparación podría incorporar una diferencia de protocolo de selección además de la pérdida de asociación target-logit.

**Corrección requerida.** Ejecutar para JOINT-SHUFFLED la grilla completa de seis lambdas sobre el target permutado, con los mismos folds target-blind, clave OOF, optimizer y budget, y reajustar la lambda seleccionada por el propio control. Persistir por separado grid OOF, lambda elegida y estado final del brazo real y del shuffled. Agregar una fixture donde las lambdas óptimas real y shuffled sean distintas y una mutación que intente reutilizar la lambda real. Si se desea deliberadamente el estimando condicionado a la lambda real, debe enmendarse primero el freeze y rotularse como otro control; no puede introducirse silenciosamente en este runner.

### H2 — El plan declara una implementación completa, pero no materializa el contrato de estimandos y decisión del freeze

**Evidencia.** El propósito dice materializar el diseño completo y cerrar sólo con una “implementación completa” (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:5-23`). No obstante, el árbol de artefactos termina en `diagnostic_metrics.json` y `diagnostic_arrays.npz` (`:293-334`), y `P11_CELL_AND_ESTIMAND_PARITY` sólo menciona igualdad de unidades, utility, penalty, targets y losses (`:358-373`). No se especifican el algoritmo ni los artefactos para:

- los 5000 bootstraps pareados por `pair_token`;
- los deltas con orientación primer término menos segundo;
- los percentiles 2.5/97.5;
- las ocho filas `SET_JOINT_NLL`, `SET_JOINT_BRIER`, `SET_SHUFFLE`, `READER_REGRET`, `READER_COMPAT`, `READER_WORST`, `READER_CONTROL` y `FACTOR_INTERACTION`;
- la precedencia `NOT_EVALUABLE > ADVERSE > NOT_RESOLVED` donde corresponde;
- `JOINT_PATTERN_PRESENT` y `CONTEXTUAL_PATTERN_PRESENT`.

Estos elementos son parte expresa del freeze actual (`configs/proportional_set_valued_native_freeze_v1.json:85-87`) y están definidos operacionalmente en el plan dual (`PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:466-516`). El freeze también exige conservar métricas y bootstrap en raw (`PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:581-589`). La exclusión de evidencia prospectiva o variabilidad de entrenamiento no justifica omitirlos: el bootstrap por token puede y debe ejecutarse como diagnóstico de implementación sobre el fixture abierto, sin transformarlo en evidencia nueva.

**Impacto.** Un artefacto podría alcanzar `RUNNER_PREFLIGHT_VALID` sin demostrar que el análisis final requerido existe, usa el soporte correcto o reproduce la tabla de decisión. “Cell and estimand parity” no verifica completitud ni semántica del estimando. Eso deja el mayor riesgo de implementación para después del preflight que pretende cerrarlo.

**Corrección requerida.** Añadir al plan raw por token y celda, índices o estado reproducible de los 5000 bootstraps, tabla completa de estimandos, CIs, filas de decisión y patrones. `P11` debe reconstruirlos independientemente y fallar por fila ausente, orientación invertida, unidad de bootstrap incorrecta, soporte incorrecto, media matched mal construida o precedencia alterada. Todos los resultados deben conservar la etiqueta `OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC`. Alternativamente, si este hito se limita deliberadamente a primitives, debe dejar de presentarse como implementación completa del runner congelado y registrar explícitamente el runner de estimandos/decisión como deuda bloqueante antes de `RUNNER_PREFLIGHT_VALID`.

### M1 — El desempate por suma de ranks de arista no induce una asignación total única

**Evidencia.** El matching maximiza Hamming y usa como segundo término la suma de `hash_rank` únicos por arista mediante `linear_sum_assignment` (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:246-262`). Que cada arista tenga un rank distinto no implica que dos matchings tengan sumas distintas. Contraejemplo de tres filas con igual costo Hamming primario: una matriz de ranks

```text
[[3,0,1],
 [2,7,4],
 [5,6,8]]
```

tiene dos derangements, `(1,2,0)` y `(2,0,1)`, ambos con suma 9. Por tanto, el objetivo entero todavía puede empatar. El plan no liga una versión de SciPy ni define un desempate a nivel de permutación. La estabilidad ante reordenar filas (`:399-402`) no basta porque la entrada se recanonicaliza y un empate agregado puede seguir resolviéndose por comportamiento interno no contractual de la biblioteca.

**Impacto.** Dos implementaciones conformes, o dos versiones de SciPy, pueden producir mapas distintos con el mismo costo. Eso rompe la pretensión de mapa único y puede quebrar replay futuro o comparabilidad de controles.

**Corrección requerida.** Definir un tercer criterio lexicográfico a nivel de asignación completa, por ejemplo el vector de donors en orden canónico, y especificar un algoritmo exacto que encuentre el mínimo/máximo lexicográfico dentro del óptimo primario y secundario. Ligar NumPy/SciPy en config y receipt. Añadir una fixture con empate de suma —como la anterior— y exigir exactamente el mismo mapa esperado, no sólo optimalidad del costo.

### M2 — Cinco seeds no garantizan cinco controles distintos y el plan no detecta duplicados

**Evidencia.** El plan registra cinco seeds y todos los mapas (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:239-281`), pero no exige cinco digests distintos. Como el RNG sólo desempata entre óptimos Hamming, un óptimo primario único hace que distintos seeds produzcan exactamente el mismo mapa; aun con empates, dos seeds pueden colisionar. La referencia W59 ligada por el propio plan sí rechaza una familia si no existen cinco `mapping_sha256` y cinco `target_sha256` distintos (`src/geometria_proporcional/wave59_hgb_guard_bracket.py:590-603`). Las mutaciones previstas comprueban seed omitido o promedio de menos de cinco, pero no cinco entradas nominales duplicadas (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:392-394`).

**Impacto.** `READER_CONTROL` podría promediar cinco copias de uno o pocos controles y presentarlas como cinco realizaciones matched. El error no queda cubierto por `support` ni por `match_valid`.

**Corrección requerida.** Exigir y chequear diversidad de `mapping_sha256` y del triplete de targets transportados entre las cinco realizaciones, o definir explícitamente que las colisiones invalidan la familia con un reason code cerrado. Antes de implementar, probar factibilidad de cinco mapas distintos sobre el fixture abierto con el algoritmo final; no introducir retries o seeds nuevos sin congelar su regla. Añadir mutaciones de seed duplicado, mapa duplicado y target-triplet duplicado.

### M3 — La independencia del checker se afirma, pero no se convierte en una frontera de importación verificable

**Evidencia.** El plan dice que el checker reconstruye estados y resultados desde config, sources y raw arrays (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:352-356`), pero no prohíbe que importe el nuevo módulo de primitives o el runner bajo prueba. El freeze dual sí fija esta separación: “El checker independiente no importa el futuro runner de entrenamiento” (`PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:537-541`). Tampoco hay una mutación que demuestre que un helper defectuoso compartido por runner y checker sea detectado.

**Impacto.** Runner y checker pueden compartir exactamente el bug que el checker debería descubrir —en folds, selección, set MAP, features, assignment, matching o estimandos— y producir un falso PASS.

**Corrección requerida.** Prohibir explícitamente imports desde `proportional_set_valued_native.py` y desde el runner en el checker. El checker puede usar fuentes históricas congeladas cuando actúan como referencia, pero debe recomponer de forma independiente los predicados críticos y comparar contra raw. Añadir una prueba que monkeypatchee o corrompa cada helper del runner manteniendo raw coherente con el bug y verifique que el checker externo falla.

### M4 — El contrato de costo permite pasar por encima del upper/RSS si existe una explicación

**Evidencia.** `P14` acepta estar dentro de presupuesto “o exceso explícito” (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:358-373`), y la sección de presupuesto sólo produce FAIL por exceder `1800 s` o `1.5 GiB` **sin explicación** (`:417-423`). Una explicación narrativa no cambia el consumo medido. El freeze de origen clasifica esos números como costo proyectado y declara que aún no hay runtime medido (`configs/proportional_set_valued_native_freeze_v1.json:88`); no autoriza convertir una explicación en excepción de PASS.

**Impacto.** El mismo runtime fuera de contrato puede recibir PASS o FAIL según exista texto explicativo, y `RUNNER_PREFLIGHT_VALID` deja de tener una semántica única.

**Corrección requerida.** Separar medición, clasificación y explicación. Si `upper_seconds` y `peak_rss_upper_bytes` son límites duros, cualquier exceso debe producir `COST_CONTRACT_EXCEEDED`; la explicación se registra pero no revierte el estado. Si son sólo proyecciones, fijar categorías cerradas —por ejemplo `WITHIN_PROJECTED_RANGE` y `ABOVE_PROJECTED_RANGE`— y declarar explícitamente si la segunda bloquea el preflight. Precisar si el presupuesto cubre primario+replay únicamente o también checker y mutaciones.

### M5 — “Reutiliza W52–W54” y “no se importa torch” son incompatibles sin una regla explícita de importación

**Evidencia.** El nuevo módulo se declara autocontenido pero reutiliza primitives matemáticas W52–W54 cuando coinciden (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:114-120`). La ejecución declara categóricamente “No se importa torch” (`:341-344`). Sin embargo, `src/geometria_proporcional/wave52_policy.py` importa `torch` y `torch.nn` al cargar el módulo (`wave52_policy.py:1-9`), aunque las functions requeridas para acciones y regret sean NumPy (`:48-59`, `:102-121`).

**Impacto.** Un implementador puede interpretar “reutiliza” como import directo de W52 y violar el contrato CPU/no-`torch`; otro puede copiar la lógica y obtener una dependencia y superficie de auditoría distintas. El plan no determina cuál es conforme.

**Corrección requerida.** Declarar que el runtime no puede importar `wave52_policy.py`. Extraer sus primitives NumPy necesarias a un módulo torch-free ligado por hash, o reimplementarlas localmente con paridad independiente y fixtures. Añadir un test del grafo/import runtime que falle si `torch` aparece en `sys.modules` o en el árbol de imports del runner/checker.

## Aspectos que sí quedan adecuadamente fijados

- Las 15 fuentes explícitas existen y sus SHA-256 actuales coinciden exactamente con la tabla del plan. El commit base `a49fe9b503a5ff366276568c4eded3645e7c3401` existe; entre ese commit y HEAD sólo se añadió este plan.
- Las tres poblaciones abiertas observadas tienen 192, 768 y 768 tokens, respectivamente; no hay solapamiento de `pair_token` entre ellas ni con las otras 192 filas W54. El pool binario W54 contiene 290 ceros y 478 unos. Esto confirma los conteos del plan, pero no constituye ejecución del futuro runner.
- La separación lógica `posterior_fit_truth` / `policy_fit_truth` / `decision_select_public` / `decision_select_truth`, junto con APIs separadas para applier y evaluator, es coherente con el alcance de preflight (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:75-112`). El propio plan reconoce correctamente que esto no acredita aislamiento físico.
- La receta MARGINAL coincide con el freeze. La receta JOINT real conserva folds target-blind, grid, clave OOF, optimizer y refit. HARD está ligado al set MAP con los dos tie-breaks exigidos. CONTEXTUAL define las 17 features, weights por token, Ridge, guards, grilla `343 + HARD_ONLY`, comparación estricta y clave de selección con suficiente precisión (`:122-154`, `:170-237`). H1 se limita a la variante JOINT-SHUFFLED.
- La aplicación matched usa `k` verdadero congelado, orden target-blind, masks comunes y prohíbe promediar controles faltantes. La frontera de claims es correcta: diagnóstico de implementación abierto, sin draw fresco, GPU, promoción ni decisión científica (`:25-44`, `:275-282`).
- La serialización canónica, conservación raw y replay byte-exacto están bien orientados (`:293-350`), sujetos a completar los estimandos y a reforzar la independencia del checker.

## Comprobaciones realizadas

1. Lectura completa de las 445 líneas del plan auditado.
2. Lectura completa del freeze JSON vigente y contraste focal con los planes de mapping y dual freeze ligados.
3. Inspección de las primitives W52, W53, W54, W56 y W59 citadas, incluidas las condiciones de diversidad de controles de W59.
4. Recomputación SHA-256 de las 15 fuentes de la tabla: 15/15 coinciden.
5. Verificación de existencia del commit base y diff contra HEAD: un único archivo añadido, el plan auditado.
6. Inspección read-only de schemas, shapes, roles, clases y tokens de los NPZ abiertos explícitamente permitidos; recomputación de disyunción de poblaciones y del pool binario.
7. Construcción CPU de un contraejemplo exhaustivo de `n=3` que demuestra empate entre dos assignments pese a ranks de arista únicos.
8. Verificación del entorno local relevante: scikit-learn `1.8.0`, SciPy `1.17.0`; el plan sólo fija el primero.

No se ejecutaron tests del runner porque el runner aún no existe. No se consultó, inicializó ni usó GPU/CUDA. No se abrió ninguna fuente o contenido de monitor, lockbox o sealed; sólo se inspeccionaron las fuentes y artefactos abiertos nombrados expresamente por el plan.

## Condición para reauditoría

La reauditoría debe comprobar una versión completa del plan que resuelva H1 y H2 y cierre los cinco findings medios sin trasladar las decisiones al implementador. En particular debe poder responder de forma única: qué lambda usa JOINT-SHUFFLED y cómo fue elegida; cuáles son todos los artefactos/algoritmos de estimandos y decisión; cuál es el mapa matched ante un empate agregado; qué constituye cinco controles distintos; qué código puede importar el checker; qué ocurre al exceder costo/RSS; y cómo se reutiliza la semántica W52 sin cargar `torch`.
