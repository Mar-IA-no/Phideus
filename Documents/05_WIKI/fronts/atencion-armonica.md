---
schema_version: 1
id: front-atencion-armonica
kind: front
page_status: current
front_status: focus_active
updated: 2026-09-09
verified_at: 2026-09-09
valid_at: 2026-09-09
recorded_at: 2026-09-09
evidence_commit: 207e6ba0d8fad1fc49ff7f8d50aec1bac02cf3fd
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
evidence_status: synthetic_loss_reader_tests_and_retrospective_rival_diagnostic_with_exact_replays
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
pequeño no identifica una fuente física. La siguiente ablación debe mantener
cabeza y pérdida comunes y variar evidencia generativa correcta, ausente o
desacoplada. El ajuste clásico es una referencia de sistema, no un contraste
causal de arquitectura; rangos y cardinalidades del sampler no son invariantes
físicos ni validación de HIT. El [diseño inicial](../../../experiments/atencion_armonica/PLAN_GENERATIVE_EVIDENCE_READER.md)
ya se desarrolló en un [protocolo separado](../../../experiments/atencion_armonica/PROTOCOL_GENERATIVE_EVIDENCE_READER.md).
El [estado de ejecución](../../../experiments/atencion_armonica/STATUS_GENERATIVE_EVIDENCE_READER.md)
separa la implementación de sus resultados: el perfil CPU/GPU y de
almacenamiento, la preparación de train/calibración y la carga real del
corpus entregado ya terminaron. La proyección de recursos fue comprobada
y los 27 entrenamientos de 50 épocas terminaron. La selección por calibración
cerró y fue auditada: época 50 para los tres brazos, sobre soporte común
503/512 y sin escoger un checkpoint o una semilla ganadores. El freeze cerró
antes de generar las 512 escenas IID, pero la primera inferencia se detuvo
por un error de integración del almacén. Su recuperación auditada preservó
esas escenas y permitió guardar features y tres forwards; después se detuvo
por una comparación incorrecta de tuplas en memoria con listas JSON. La enmienda
explícita de serialización ya fue auditada, con 45 pruebas CPU independientes,
y la inferencia IID se reanudó desde ese prefijo. IID, mayor inarmonicidad y
polifonía completaron predicción, evaluación y replay; familia deformada y las
auditorías finales siguen pendientes. La selección OPEN y
las pruebas mecánicas no acreditan una ventaja de generalización.

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
