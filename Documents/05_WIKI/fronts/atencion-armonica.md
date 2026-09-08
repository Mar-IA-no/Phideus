---
schema_version: 1
id: front-atencion-armonica
kind: front
page_status: current
front_status: focus_active
updated: 2026-09-08
verified_at: 2026-09-08
valid_at: 2026-09-08
recorded_at: 2026-09-08
evidence_commit: f1e7126724442f27cccf5b2c8006986f6418bdc4
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
  - experiments/atencion_armonica/AMENDMENT_LEARNED_PARTITION_NUMERICAL_GUARD.md
  - src/atencion_armonica/learned_partition_reuse.py
  - experiments/atencion_armonica/test_learned_partition_reuse.py
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/ROADMAP_ATENCION_ARMONICA.md
depends_on: []
tangents: [front-escalon-3, ppu-natural-harmonic-geometry]
architecture_status: candidate
experiment_status: mixed
evidence_status: synthetic_loss_contrast_and_fresh_structured_reader_tests_independently_audited
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

El siguiente contraste es un [lector aprendido de particiones](../../../experiments/atencion_armonica/PLAN_LEARNED_PARTITION_READER.md)
sobre el mismo algoritmo de candidatos y redes congeladas, con datos nuevos.
Su [protocolo ejecutable](../../../experiments/atencion_armonica/PROTOCOL_LEARNED_PARTITION_READER.md)
está auditado. El ejecutor implementa datos por shards, normalización train-only,
entrenamiento recuperable, selección y evaluación. Las
[pruebas mecánicas](../../../experiments/atencion_armonica/test_learned_partition_campaign.py)
comparan la trayectoria continua con interrupciones y conservan el prefijo de
calibración. Las reauditorías focales verificaron la recuperación, las métricas
de intervenciones y el almacenamiento normalizado empaquetado. El cálculo de
recursos ya integra las lecturas de archivos y checkpoints con el costo de
validación por etapa. El [primer perfil geométrico](../../../experiments/atencion_armonica/RESULTS_LEARNED_PARTITION_RESOURCE_PROFILE.md)
terminó, pero la proyección de validación sola excede el límite por celda.
La corrección auditada de validación y medición por bloques permitió completar
un segundo corte con perfiles CPU y 3090. Todavía excede topes; la autorización
detectó además una discrepancia de identidad de runtime. Las correcciones de
runtime y soporte fueron auditadas; el tercer corte completó los tres perfiles
y la autorización se detuvo por presupuesto. La regla prefijada favorece CPU
para las cabezas pequeñas, con forwards congelados GPU separados. La continuidad
incorpora una enmienda aprobada de tiempos, sin cambiar receta ni controles:
20 minutos por entrenamiento/forward/inferencia, 40 por score y 12 horas de
entrenamiento acumulado. El cuarto corte completó esos perfiles y obtuvo
autorización de train/calibración; la preparación de ambos conjuntos terminó.
La regla de costo volvió a seleccionar CPU para las cabezas. El primer intento
se detuvo por un falso rechazo de suma de incidencias en `float32`, con datos,
snapshot inicial y tiempo consumido preservados. La continuidad requiere
una [enmienda numérica explícita](../../../experiments/atencion_armonica/AMENDMENT_LEARNED_PARTITION_NUMERICAL_GUARD.md),
ya implementada y auditada, sin cambiar receta ni reemplazar escenas.
El quinto corte completó perfiles y la importación de los datos existentes;
su copia fue auditada, pero una prueba posterior detectó un error de recuperación
bajo una reserva activa antes de lanzar training. La corrección mantiene
el presupuesto estricto y fue verificada con un hijo real. El sexto corte
completó perfiles y una importación nueva, auditada, de la misma cohorte.
El entrenamiento CPU completó las 36 corridas de los cuatro brazos, con
checkpoints y las diez calibraciones por celda preservados. La selección
de un epoch común por brazo quedó congelada y pasó la auditoría independiente.
La evaluación IID completó datos, forwards y scoring; la normalización se
detuvo por el límite de memoria CPU antes de producir predicciones o métricas.
La recuperación versionada, revisada independientemente, completó después la
normalización dentro del límite original, con los arrays conservados exactos
y sin reetiquetar fuentes ni repetir entrenamientos. Ese stage pasó su auditoría.
El [balance IID parcial](../../../experiments/atencion_armonica/RESULTS_LEARNED_PARTITION_IID.md)
completó inferencia, evaluación y replay CPU; los ocho payloads coinciden
exactamente y la revisión independiente cerró sin hallazgos materiales en IID.
Mayor inarmonicidad completó también evaluación y replay. La inferencia del
primario de polifonía se detuvo por un falso rechazo de redondeo en la
validación del soporte. La recuperación pasó su auditoría y completó la
inferencia de polifonía, con los once archivos previos exactamente preservados;
la evaluación posterior se detuvo por RAM y está en diagnóstico. No cambian
modelos ni muestras. Familia deformada sigue pendiente. No hay cierre del
contraste ni promoción arquitectónica.

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
