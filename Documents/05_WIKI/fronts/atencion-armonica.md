---
schema_version: 1
id: front-atencion-armonica
kind: front
page_status: current
front_status: focus_active
updated: 2026-09-14
verified_at: 2026-09-14
valid_at: 2026-09-14
recorded_at: 2026-09-14
evidence_commit: 136151c75f08a13b1c4a67f26313a0d2fd900680
source_paths:
  - Documents/00_TRONCAL/ROADMAP_GENERAL/PROGRAMA_GEOMETRIA_ARMONICA_COMPUTABLE.md
  - experiments/atencion_armonica/PLAN_GEOMETRIC_RESEARCH_ACTION.md
  - experiments/atencion_armonica/RESULTS_ENERGY_PARTITION_AUDIT.md
  - experiments/atencion_armonica/PLAN_SHARED_PARTIAL_COMPATIBILITY.md
  - experiments/atencion_armonica/RESULTS_SHARED_PARTIAL_PREFLIGHT.md
  - experiments/atencion_armonica/RESULTS_SHARED_PARTIAL_GPU_PROFILE.md
  - experiments/atencion_armonica/RESULTS_SHARED_PARTIAL_STUDY.md
  - experiments/atencion_armonica/PLAN_SHARED_SOURCE_COHERENCE.md
  - experiments/atencion_armonica/RESULTS_SHARED_SOURCE_COHERENCE.md
  - experiments/atencion_armonica/PLAN_SOURCE_STRUCTURED_READER.md
  - experiments/atencion_armonica/RESULTS_SOURCE_STRUCTURED_READER.md
  - experiments/atencion_armonica/PLAN_LEARNED_PARTITION_READER.md
  - experiments/atencion_armonica/PROTOCOL_LEARNED_PARTITION_READER.md
  - experiments/atencion_armonica/test_learned_partition_training.py
  - experiments/atencion_armonica/test_learned_partition_data.py
  - experiments/atencion_armonica/test_learned_partition_resources.py
  - src/atencion_armonica/learned_partition_supervisor.py
  - src/atencion_armonica/learned_partition_campaign.py
  - src/atencion_armonica/learned_partition_selection.py
  - src/atencion_armonica/learned_partition_test.py
  - experiments/atencion_armonica/test_learned_partition_campaign.py
  - experiments/atencion_armonica/test_learned_partition_budget.py
  - experiments/atencion_armonica/test_learned_partition_inputs.py
  - experiments/atencion_armonica/test_learned_partition_selection.py
  - src/atencion_armonica/learned_partition_validation.py
  - experiments/atencion_armonica/test_learned_partition_validation.py
  - experiments/atencion_armonica/test_learned_partition_profile.py
  - src/atencion_armonica/learned_partition_resources.py
  - experiments/atencion_armonica/RESULTS_LEARNED_PARTITION_RESOURCE_PROFILE.md
  - experiments/atencion_armonica/RESULTS_LEARNED_PARTITION_IID.md
  - experiments/atencion_armonica/RESULTS_LEARNED_PARTITION_READER.md
  - experiments/atencion_armonica/RESULTS_LEARNED_PARTITION_MEANS.csv
  - experiments/atencion_armonica/PLAN_OBSERVABLE_SOURCE_RIVALS.md
  - experiments/atencion_armonica/PROTOCOL_OBSERVABLE_SOURCE_RIVALS.md
  - experiments/atencion_armonica/RESULTS_OBSERVABLE_SOURCE_RIVALS.md
  - experiments/atencion_armonica/PLAN_GENERATIVE_EVIDENCE_READER.md
  - experiments/atencion_armonica/PROTOCOL_GENERATIVE_EVIDENCE_READER.md
  - experiments/atencion_armonica/STATUS_GENERATIVE_EVIDENCE_READER.md
  - experiments/atencion_armonica/RESULTS_GENERATIVE_EVIDENCE_READER.md
  - experiments/atencion_armonica/RESULTS_GENERATIVE_EVIDENCE_METRICS.csv
  - experiments/atencion_armonica/PLAN_OPERATOR_OBJECTIVE_ALIGNMENT.md
  - experiments/atencion_armonica/PROTOCOL_OPERATOR_OBJECTIVE_ALIGNMENT.md
  - experiments/atencion_armonica/STATUS_OPERATOR_OBJECTIVE_ALIGNMENT.md
  - experiments/atencion_armonica/RESULTS_OPERATOR_OBJECTIVE_PROFILE.md
  - experiments/atencion_armonica/RESULTS_OPERATOR_OBJECTIVE_CACHE_PROFILE.md
  - experiments/atencion_armonica/RESULTS_OPERATOR_OBJECTIVE_FAST_PROFILE.md
  - experiments/atencion_armonica/RESULTS_OPERATOR_OBJECTIVE_WORKLOAD.md
  - experiments/atencion_armonica/AMENDMENT_OPERATOR_OBJECTIVE_EXECUTION.md
  - experiments/atencion_armonica/RESULTS_OPERATOR_OBJECTIVE_ALIGNMENT.md
  - experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_ENERGY.md
  - experiments/atencion_armonica/PROTOCOL_GEOMETRIC_DECISION_ENERGY.md
  - experiments/atencion_armonica/STATUS_GEOMETRIC_DECISION_ENERGY.md
  - experiments/atencion_armonica/RESULTS_GEOMETRIC_DECISION_PROFILE.md
  - src/atencion_armonica/generative_evidence_model.py
  - experiments/atencion_armonica/AMENDMENT_LEARNED_PARTITION_NUMERICAL_GUARD.md
  - src/atencion_armonica/learned_partition_reuse.py
  - experiments/atencion_armonica/test_learned_partition_reuse.py
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/ROADMAP_ATENCION_ARMONICA.md
depends_on: []
tangents: [front-escalon-3, ppu-natural-harmonic-geometry]
architecture_status: candidate
experiment_status: mixed
evidence_status: synthetic_loss_and_reader_tests_including_matched_generative_evidence_with_exact_replays
decision_status: pending_analysis
---

# Atención Armónica

## Resumen

Atención Armónica ensaya una arquitectura relacional para agrupar parciales de
fuentes polifónicas. Los picos son nodos; los estados de par describen posible
pertenencia común; el triangle update propaga consistencia a través de terceros
picos; un clusterer convierte la matriz de relaciones en una partición.

## Estado real

Fases 0, 0.5 y 0.6 están cerradas. El pair-state es el salto principal.
Comparado con B-local param-matched, el triangle no domina IID ni OOD-regime,
pero mejora OOD-poly. B-shuffle confirma que la estructura del triángulo importa.

Fase 0.5 mostró que el cuello no era el umbral `τ`, sino
connected-components. Fase 0.6 mostró que spectral y agglomerative con `k`
estimado extraen de forma deployable parte de la ventaja de B. El estimador
subestima `k`, por lo que la partición todavía no está resuelta.

## Rebase vigente: geometría, arquitectura y pérdida

El [nuevo programa](../../00_TRONCAL/ROADMAP_GENERAL/PROGRAMA_GEOMETRIA_ARMONICA_COMPUTABLE.md)
toma este frente como banco inicial, no como arquitectura promovida. El
[diagnóstico CPU](../../../experiments/atencion_armonica/RESULTS_ENERGY_PARTITION_AUDIT.md)
recuperó una partición única y correcta por amplitudes solas en las ocho
mezclas seleccionadas de dos fuentes y las ocho de tres, también desde
log-amp float32. Los ocho casos de una fuente se cuentan aparte. Es una
muestra histórica no aleatoria, no atribución de ese uso a las redes.
Las métricas históricas se conservan; su gate per-par no excluía este atajo
global. El [contraste fijado](../../../experiments/atencion_armonica/PLAN_SHARED_PARTIAL_COMPATIBILITY.md)
retira amplitudes y compara pérdidas con red/descriptores comunes:
compatibilidad de una familia espectral, BCE sola, pesos desacoplados de
los triples y transitividad genérica. Añade baseline token-only y heurística
analítica. El [resultado neuronal](../../../experiments/atencion_armonica/RESULTS_SHARED_PARTIAL_STUDY.md)
completa quince trainings y cinco tests con lectores elegidos sólo en
validación. En OOD beta, compatibilidad aumenta Brier frente a BCE en las
tres semillas; mejora el promedio frente a sham/transitividad, pero no
uniformemente entre semillas. Pares+BCE supera token-only en Brier medio
en los cinco slices; la heurística obtiene mejores particiones en OOD beta.
El [diagnóstico posterior](../../../experiments/atencion_armonica/RESULTS_SHARED_SOURCE_COHERENCE.md)
completó 96 escenas y replay. La presión física final es pequeña frente a
BCE; el fit de parámetros compartidos no identifica pertenencia y algunos
grupos mixtos tienen residual pequeño. El [lector estructurado ejecutado](../../../experiments/atencion_armonica/RESULTS_SOURCE_STRUCTURED_READER.md)
completó cuatro tests frescos y replay. El factor conjunto no muestra ventaja
clara en el primario de polifonía frente a controles con pool común; pierde
ARI medio en inarmonicidad y familia deformada y aumenta fragmentación
frente a Pares. El sham elegido tiene γ=0; su soporte potencial es parcial
en familia deformada. La auditoría final cerró sin hallazgos materiales. La evidencia sigue
siendo sintética y de inferencia, no identificabilidad ni geometría aprendida.

El [lector aprendido de particiones](../../../experiments/atencion_armonica/RESULTS_LEARNED_PARTITION_READER.md)
completó 36 entrenamientos y cuatro tests nuevos con replay exacto de los ocho
payloads por escenario. Conserva los backbones congelados y el mismo algoritmo
de candidatos; la selección se fijó sólo en calibración. En el primario de
polifonía, Compartida mejora ARI frente a Pares y Desacoplada, pero no muestra
ventaja clara frente a Local. Bajo familia deformada pierde frente a los tres
controles aprendidos. Local conserva menor error absoluto de k y menor masa
de grupos pequeños en los cuatro escenarios. La evidencia es mixta y no
promueve una arquitectura ni acredita nueva geometría latente.

Las auditorías finales de evidencia y alineación cerraron sin hallazgos
materiales abiertos. El [diagnóstico de fuentes rivales](../../../experiments/atencion_armonica/RESULTS_OBSERVABLE_SOURCE_RIVALS.md)
completó 96 escenas y replays. En deformación, el residual mediano baja de
10.376 a 1.454 cents, mientras las particiones exactas pasan de 18 a 19 entre
23 escenas con candidatos; una de las 24 sigue sin salida. En mayor beta,
la referencia plantada ajusta mejor en las 24, pero falta entre candidatos
en dos. Esto distingue ajuste, cobertura e identidad, no certifica unicidad
ni no-identificabilidad. Los
[recursos y recuperaciones](../../../experiments/atencion_armonica/RESULTS_LEARNED_PARTITION_RESOURCE_PROFILE.md)
conservan los intentos incompletos, las correcciones de validación y la enmienda
explícita de RAM, sin cambiar modelos, selección ni muestras. El próximo
movimiento debe distinguir ambigüedad de la observación y limitaciones del
lector: un oracle dentro del pool no demuestra aprendibilidad, y un residual
pequeño no identifica una fuente física. La ablación posterior mantuvo
cabeza y pérdida comunes y varió evidencia generativa correcta, ausente o
desacoplada. El ajuste clásico es una referencia de sistema, no un contraste
causal de arquitectura; rangos y cardinalidades del sampler no son invariantes
físicos ni validación de HIT. El [diseño inicial](../../../experiments/atencion_armonica/PLAN_GENERATIVE_EVIDENCE_READER.md)
ya se desarrolló en un [protocolo separado](../../../experiments/atencion_armonica/PROTOCOL_GENERATIVE_EVIDENCE_READER.md).
El [resultado generativo](../../../experiments/atencion_armonica/RESULTS_GENERATIVE_EVIDENCE_READER.md)
completó 27 entrenamientos y cuatro tests con replay. La selección OPEN eligió
época 50 por brazo, sin escoger checkpoint o semilla ganadores. El primario
deformado favorece Generativa frente a Desacoplada en ARI: +0.006858,
IC97.5% [0.002706, 0.011171]. Frente a Local, +0.002957
[−0.000668, 0.007134] no establece una ventaja clara. En polifonía,
Generativa pierde frente a Desacoplada. Los intervalos son pareados por escena,
condicionados a las nueve celdas entrenadas; no son nueve réplicas independientes.
La cobertura es 505/512, 497/512, 451/512 y 484/512, sin ARI imputado a ausencias.
El [estado operativo](../../../experiments/atencion_armonica/STATUS_GENERATIVE_EVIDENCE_READER.md)
conserva fallos, enmienda JSON posterior al draw, pausa y 7101.919 segundos
acumulados del supervisor. Los cuatro tests y las auditorías finales ya cerraron,
sin hallazgos materiales abiertos dentro del alcance revisado. No hay promoción
ni autoridad física añadida por draws sintéticos.

El [diagnóstico de operación y objetivo](../../../experiments/atencion_armonica/RESULTS_OPERATOR_OBJECTIVE_ALIGNMENT.md)
completó 2048 escenas y replay sin entrenamiento ni GPU. En familia deformada,
Generativa mejora τ-b frente a Extendida (0.434221 frente a 0.240753), pero
tiene mayor regret VI (0.082377 frente a 0.030361): ordenar mejor el conjunto
no garantiza elegir mejor su mínimo. Soporte primario: 484/512 escenas.
Los conjuntos óptimos VI/ARI se intersectan en 479/484; esa discrepancia
observada no explica la brecha de decisión. La inversión del regret dentro
de estratos de tamaños/disponibilidad/rama es descriptiva, no causal.

El [estado operativo](../../../experiments/atencion_armonica/STATUS_OPERATOR_OBJECTIVE_ALIGNMENT.md)
conserva los perfiles, la enmienda explícita y 1350.970249 s acumulados hasta
la auditoría técnica final, sin reiniciar el ledger. Se autenticó el cierre
completo y se reextrajeron 16 escenas seleccionadas de antemano; no es una
segunda extracción independiente de todo el corpus. El siguiente contraste
debe separar operación geométrica y aprendizaje de la decisión con muestras
nuevas y controles fuertes. El diagnóstico no identifica representación,
optimización o loss como causa ni promueve arquitectura.

Las auditorías finales técnica y de interpretación/alineación cerraron sin
hallazgos materiales abiertos dentro de sus alcances.

El [plan de energía geométrica para la decisión](../../../experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_ENERGY.md)
abre el sucesor: separar operación/interfaz y loss, con controles de inyección,
correspondencia desacoplada, Local y referencia clásica. El
[protocolo concreto](../../../experiments/atencion_armonica/PROTOCOL_GEOMETRIC_DECISION_ENERGY.md)
y el núcleo aritmético, el adaptador y el runner recuperable de una celda ya
fueron auditados. La preparación OPEN ya se ejecutó completa, sin refits ni
forwards; el selector está implementado y auditado. Los
[perfiles CPU/CUDA](../../../experiments/atencion_armonica/RESULTS_GEOMETRIC_DECISION_PROFILE.md)
también están completos, con recuperación exacta CUDA comprobada y costos
separados de cabeza, fitter y carga. El
[estado](../../../experiments/atencion_armonica/STATUS_GEOMETRIC_DECISION_ENERGY.md)
registra los 72 entrenamientos completos en CUDA y la selección CPU sobre
720 calibraciones, mediante operadores previamente auditados. Se mantiene una
época por variante y las nueve celdas por variante; no se elige una semilla
ganadora. El entrenamiento consumió 4912,186859 s y la selección 109,324164 s.
El ensamblado y almacén observables también están auditados: fits recuperables
desde factores, entradas compartidas y linaje autenticado original–roundtrip.
Sus pruebas CPU y las del ensamblador recuperable, exclusiones y archivo de
cabezas no sustituyen el freeze ni la evaluación prospectiva. Los defectos
de identidad detectados en la preparación se corrigieron y reauditaron antes
de usarla con tests nuevos.
El operador CPU posterior conservó los 144 estados iniciales/seleccionados y
construyó el inventario de exclusiones en 9,938720 s; su cierre está enlazado
en el mismo estado. Los puertos de producción única y predicción también
pasaron auditoría, con recuperación de transportes sin repetir el modelo.
El perfil del recorrido observable también terminó, con 16 TRAIN conocidas,
cuatro roundtrips y recuperación exacta de los 144 estados sin modelos ni
ajustes. Consumió 113,436136 s. El perfil CPU posterior midió productor/IO de
las mismas tuplas conocidas, métricas, replay, bootstrap diagnóstico e inventario
en 25,693755 s. Los perfiles acumulan 215,634043 / 600 s. El sello global y la
evaluación por identidad de evento están implementados y auditados; la
proyección completa revela que la asignación original de evaluación/replay
excede su reserva restante. Falta cerrar la revisión operativa y auditar el
supervisor integrado, sin reducir muestras ni controles. Estos perfiles no
autorizan por sí solos nuevas observaciones.
Tests, probes y evaluación prospectiva siguen pendientes. No hay ventaja
experimental demostrada ni promoción arquitectónica.

## Bifurcaciones preservadas, no secuencia obligatoria

| Camino | Qué aísla |
|---|---|
| Stage B: cabeza de `k/partición` | Si el cuello residual puede aprenderse sobre Pairformer congelado |
| Fase 1a: render→CQT→picos | Si la ventaja sobrevive a errores de detección manteniendo GT exacto |

## Alcance

El resultado histórico sostiene una ventaja específica de generalización OOD-poly, no la
afirmación de que el triangle gane universalmente ni que ya exista una geometría
armónica completa.

## Relaciones

- [Escalón 3](escalon-3.md)
- [PPU / Natural Harmonic Geometry](../concepts/ppu-geometria-armonica-natural.md)

## Fuentes

- [README canónico](../../01_FRENTES_ACTIVOS/Atencion_Armonica/README.md)
- [Roadmap local](../../01_FRENTES_ACTIVOS/Atencion_Armonica/ROADMAP_ATENCION_ARMONICA.md)
- [Explicación de Fase 0.6](../../01_FRENTES_ACTIVOS/Atencion_Armonica/Explicacion_fase_0_6_clusterer_deployable_codex.md)
