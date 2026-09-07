# Reauditoría independiente del plan del runner CPU set-valued nativo

Fecha: 2026-09-04  
Commit auditado: `b8e2a8066ff22c31197b0290dc99ae38363528b8`  
Objeto: `experiments/geometria_proporcional/PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md`  
Antecedente: R560, `REVISE — 2 HIGH / 5 MEDIUM / 0 LOW`

## Veredicto

**PASS — 0 HIGH / 0 MEDIUM / 0 LOW.**

La versión vigente resuelve de manera sustantiva y verificable los siete findings de R560. No quedan decisiones semánticas centrales trasladadas al implementador en los puntos auditados: JOINT-SHUFFLED hace selección y refit propios; el runner conserva el análisis diagnóstico completo; el assignment tiene desempate total; las cinco realizaciones matched deben ser distintas y son factibles sobre los artefactos abiertos; el checker tiene una frontera de importación explícita; los límites de costo son duros; y W52 queda como autoridad semántica sin ser importable por runtime o checker.

El PASS corresponde exclusivamente al plan. No anticipa el resultado de la auditoría final de código, tests, artefactos, checker ni replay, no convierte el fixture histórico en evidencia prospectiva y no constituye GO/NO-GO científico.

## Estado de los findings de R560

### R560-H1 — RESUELTO

JOINT-SHUFFLED ya no recibe la lambda del brazo real. El plan exige que vuelva a recorrer las seis lambdas usando el target permutado, los mismos folds target-blind, métricas OOF, clave, optimizer y budget, y que reajuste su propia lambda. También separa grid, lambda y estado final de ambos brazos y exige una fixture donde las lambdas elegidas difieran, más una mutación que copie la lambda real al control (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:164-176`, `:468-470`). Esto coincide con la validación interna y refit propios requeridos por el freeze dual (`PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:348-390`).

Como comprobación de factibilidad, reconstruí los folds y ejecuté independientemente las seis lambdas sobre las 192 filas abiertas W54. El target real y el target derangeado seleccionaron ambos `lambda=0.001` en esta población concreta. Esa coincidencia empírica no debilita el contrato: confirma por qué la fixture sintética que obliga lambdas diferentes es necesaria para demostrar que no hay reutilización silenciosa.

### R560-H2 — RESUELTO

El plan incorpora una sección analítica completa con raw por `pair_token` y celda, 5000 resamples pareados, seeds y soportes separados, matrices `int64`, orden y SHA-256, percentiles 2.5/97.5, orientación de deltas, las ocho clases de fila congeladas, reglas de adjudicación, precedencias, patterns, sensitivities por checkpoint y duplicaciones (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:304-355`). El árbol añade `bootstrap_indices.npz`, `estimand_table.json`, `sensitivity_arrays.npz` y `cell_duplications.json` (`:366-410`). `P11` y las mutaciones cubren unidad de bootstrap, seed/soporte, orientación, filas, precedencia y patterns (`:443-486`).

Las condiciones de soporte siguen siendo las congeladas: permutable `>=0.8` para shuffle y cobertura común `>=0.8` para `READER_CONTROL`, incorporadas por la frase “reglas congeladas” (`:336-342`) y resueltas sin ambigüedad por la config ligada (`configs/proportional_set_valued_native_freeze_v1.json:45-56`, `:72-87`) y la tabla dual (`PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:500-515`). Los resultados permanecen anidados bajo `OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC`.

### R560-M1 — RESUELTO

La suma de ranks por arista fue reemplazada por un desempate lexicográfico a nivel de permutación completa. Primero se obtiene el máximo Hamming; luego, en orden canónico de receiver, se fija el primer donor de su orden seeded que permite conservar el óptimo en el problema restante. Esto produce una solución total incluso cuando dos assignments tienen la misma suma secundaria. NumPy `2.3.5` y SciPy `1.17.0` quedan ligados, y una fixture reproduce expresamente el contraejemplo de tres filas de R560 con mapa literal esperado (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:255-279`).

Reimplementé el algoritmo por fuera del futuro runner. En los diez assignments de población abierta —cinco seeds por cada uno de los dos posteriors— cada solución preservó exactamente el máximo Hamming del posterior; el score fue 3671 para MARGINAL y 3955 para JOINT en los cinco seeds de cada familia. No apareció dependencia del assignment arbitrario devuelto por SciPy.

### R560-M2 — RESUELTO

El plan exige cinco `mapping_sha256` y cinco digests del triplete transportado distintos por posterior. Una colisión produce `NOT_EVALUABLE_CONTROL_DIVERSITY`, sin retries ni sustitución de seeds (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:281-289`); las mutaciones incluyen mapa y target-triplet duplicados (`:478-480`). Esto preserva la condición de diversidad de la referencia W59 (`src/geometria_proporcional/wave59_hgb_guard_bracket.py:590-603`) sin fingir cinco realizaciones cuando existen duplicados.

La factibilidad quedó comprobada con una reconstrucción CPU independiente desde las fuentes abiertas W54/W59 y el catálogo de 24 utilities:

| Posterior | Filas disagreement | Estratos | Singletons | Fracción permutable | Mapas distintos | Tripletes distintos |
|---|---:|---:|---:|---:|---:|---:|
| MARGINAL | 2094 | 215 | 47 | 0.977555 | 5/5 | 5/5 |
| JOINT | 2312 | 253 | 65 | 0.971886 | 5/5 | 5/5 |

Los prefijos de los digests de contenido obtenidos para los mapas MARGINAL fueron `35e2217e6249`, `25a07d002e1b`, `c3ed7f00cc47`, `0ed55756bc5b` y `e411364ed648`; para JOINT, `51538d8957fe`, `05210c226721`, `346df1849df9`, `19011cb20bb3` y `f215f5f8ed0b`. Los cinco digests de triplete también fueron distintos en cada posterior. Estos digests de reauditoría prueban diversidad de contenido; no pretenden sustituir el esquema canónico y hashes que producirá el runner.

### R560-M3 — RESUELTO

El checker no puede importar `proportional_set_valued_native.py`, el runner ni helpers definidos por ellos. Debe recomponer independientemente folds, masas, MAP, features, modelos lineales, assignment, matching, métricas, bootstrap y estimandos desde config, fuentes congeladas y raw. Una prueba corrompe un helper y raw coherente con ese bug para exigir detección externa (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:432-441`). La suite también muta específicamente el checker para que importe el código bajo prueba (`:484-486`). Esto satisface la separación requerida por el freeze dual (`PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md:537-541`).

### R560-M4 — RESUELTO

Los límites ya no son revocables mediante explicación. Primario más replay conservan `1800 s` y `1.5 GiB`; checker doble más mutaciones tienen `900 s` auxiliares y el mismo RSS por proceso. Cualquier exceso produce siempre FAIL, aunque se preserve la explicación. El central sólo agrega `ABOVE_CENTRAL_ESTIMATE` (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:493-516`). `P14` refleja los dos budgets y los trata como límites duros (`:443-458`).

### R560-M5 — RESUELTO

El runtime sólo reutiliza W53–W54. W52 permanece ligado como autoridad semántica, pero no se importa; utilities, acción autorizada y regret se reimplementan en NumPy y se contrastan con fixtures independientes (`PLAN_PROPORTIONAL_SET_VALUED_NATIVE_RUNNER_CPU.md:114-123`). Ni runtime ni checker pueden importar W52 ni cargar `torch`; un subprocess fresco debe terminar sin `torch` en `sys.modules`, y la suite muta esa condición (`:418-424`, `:484-486`). La inspección de fuentes confirma que W53 y W54 no importan W52 ni `torch`, mientras que W52 sí carga `torch` al importar (`src/geometria_proporcional/wave52_policy.py:7-9`), por lo que la nueva frontera es necesaria y suficiente a nivel de plan.

## Búsqueda de findings nuevos

No encontré findings nuevos altos, medios ni bajos. En particular:

- El contrato de bootstrap queda determinado por seeds PCG64, shapes, unidad y soportes persistidos, versiones ligadas y la primitive W53 congelada; el checker debe recomponerlo en vez de confiar en el reporte.
- La multiplicidad “por posterior” de `SET_SHUFFLE` y de las cuatro filas `READER_*` está declarada, mientras los ocho IDs permanecen como clases de fila del freeze. No hay pérdida de celdas.
- `FACTOR_INTERACTION` conserva la orientación exacta y carácter descriptivo; los patterns no generan ranking ni promoción.
- La separación lógica de preparación/aplicación/evaluación y la admisión explícita de que no prueba aislamiento físico siguen siendo correctas.
- El assignment corregido fue rápido en la reproducción abierta: aproximadamente 0.54 s por familia completa de cinco seeds, muy por debajo de los límites fijados. Esto es una observación local de factibilidad, no una promesa de runtime final.

## Comprobaciones realizadas

1. Lectura completa de las 537 líneas del plan vigente, no sólo del diff de corrección.
2. Lectura del freeze JSON vigente y contraste focal con el plan dual, el plan de mapping y R560.
3. Verificación de HEAD exacto `b8e2a8066ff22c31197b0290dc99ae38363528b8` y del diff desde la base declarada.
4. Recomputación SHA-256 de las 15 fuentes ligadas: 15/15 coinciden con la tabla. El plan vigente tiene SHA-256 `b9394277e13a6d9a4d83a678fd14454025f586444cb1bdeeb4c5d4aa33396396`.
5. Reconstrucción CPU de MARGINAL pooled Platt y selección JOINT de seis lambdas sobre las 192 filas `calibration_fit` abiertas W54.
6. Reconstrucción separada de JOINT sobre el target real y el target derangeado, con sus propias grillas OOF; ambos eligieron `0.001` en este fixture.
7. Aplicación de ambos posteriors a las 768 filas abiertas W59, derivación independiente de set MAP, acciones HARD, mínimo riesgo, gain, harm, incompatibility y disagreement.
8. Reimplementación independiente del assignment Hamming con desempate lexicográfico de factibilidad para los seeds `53611,53617,53623,53629,53633` y verificación de 5/5 mapas y 5/5 tripletes distintos por posterior.
9. Revisión de la frontera de imports W52/W53/W54/W56 y de los predicados/mutaciones del checker.
10. Revisión de unidad de bootstrap, soportes, ocho IDs, deltas, CIs, precedencias, patterns, sensitivities, raw/replay, budgets y claims.

Todas las ejecuciones fueron tabulares y CPU con `CUDA_VISIBLE_DEVICES=''` y un thread BLAS/OpenMP. No se usó, inicializó ni consultó GPU/CUDA. No se abrió ninguna fuente, bundle o contenido de monitor, lockbox o sealed; sólo se leyeron los artefactos abiertos nombrados expresamente por el plan.

## Condición posterior

El plan queda apto para implementación. La auditoría final todavía debe verificar sobre el estado real que código, config, tests, artefacto primario, replay y checker cumplan literalmente este contrato, incluidas las fixtures de lambda distinta y empate de assignment, la diversidad 5/5, la independencia del checker, la ausencia de W52/`torch`, los 5000 bootstraps y los límites duros de costo.
