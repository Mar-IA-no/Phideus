# Coherencia de fuente y presión de la pérdida

Fecha: 2026-09-08. Estado: diagnóstico v2 completo, con primaria, replay y
evidencia reauditorada. Sin promoción arquitectónica ni GO/NO-GO.

El diagnóstico no identifica una causa del resultado neuronal anterior.
La penalización física conserva masa sobre triples de una misma fuente,
pero su presión medida en los logits finales es pequeña frente a BCE.
Exigir parámetros compartidos detecta ajustes pobres en algunas particiones,
pero también hay grupos de fuentes mezcladas con residual pequeño. Un buen
ajuste a la familia no identifica por sí solo la pertenencia generativa.

## Qué se ejecutó

El [plan fijado](PLAN_SHARED_SOURCE_COHERENCE.md) reutilizó IDs 0–31 de
validación, OOD beta y OOD polifonía: 96 escenas, cuatro brazos de pares por
tres semillas y una heurística, con las particiones ya guardadas. No hubo
nuevos datos, forwards, entrenamiento, GPU ni ajuste de lectores.
Validación es `READER_SELECTION_IN_SAMPLE`; los otros dos slices son
`OPEN_TEST_POSTHOC`. Ninguno constituye confirmación independiente.

La presión se calcula en coordenadas de logit simétrico, no sobre parámetros
de la red. El ajuste de cada grupo recibe sólo frecuencias observadas y
miembros, busca índices distintos de 1 a 8 y un beta compartido, y perfila
el offset. Usa grillas anidadas de 1.025 y 257 valores. La pureza se evalúa
después; los grupos verdaderos se conservan como referencia privilegiada.

## Presión: masa de pérdida no equivale a fuerza relativa ni a causalidad

P/B divide la suma de presión física ponderada por lambda=0,1 entre la suma
absoluta de derivadas BCE de esa clase. La tabla promedia primero las tres
semillas dentro de escena y después las 32 escenas; todas son elegibles.
`F_true` es la fracción de masa de pérdida física sobre triples de una fuente,
no una tasa de clasificación ni una fracción de gradiente de parámetros.

| Slice | Logits del brazo | P/B positivas | P/B negativas | F_true |
|---|---|---:|---:|---:|
| Validación | BCE | 0,002931 | 0,000697 | 0,857321 |
| Validación | Compatibilidad | 0,002868 | 0,000710 | 0,860431 |
| OOD beta | BCE | 0,000484 | 0,001020 | 0,814474 |
| OOD beta | Compatibilidad | 0,000493 | 0,000762 | 0,835765 |
| OOD polifonía | BCE | 0,001620 | 0,002249 | 0,365629 |
| OOD polifonía | Compatibilidad | 0,001487 | 0,002148 | 0,408724 |

Los cocientes son adimensionales: 0,002931 equivale aproximadamente al
0,293% de la magnitud BCE definida, no al 29,3%. En OOD beta, los cuatro
brazos tienen P/B positiva media entre 0,000463 y 0,000494. Estos estados
finales no reconstruyen la trayectoria del entrenamiento ni permiten
descartar efectos acumulados o mediados por el Jacobiano de la red.

Al sustituir las probabilidades por pertenencia verdadera, la pérdida
física media sigue siendo positiva: 0,004455 en validación, 0,004783 en
OOD beta y 0,001319 en OOD polifonía. Esto caracteriza la penalización sobre
observaciones ruidosas; no adjudica cuánto causó del deterioro anterior.
La dirección no negativa de la derivada era conocida por la fórmula, no
un descubrimiento experimental. Física y sham se calcularon sobre los
mismos logits en todos los brazos; sus estimandos completos están en el
[resumen por lector y brazo](../../data/atencion_armonica/source_coherence_v2/summary.json).

## Ajustar una fuente común no resuelve la pertenencia

El RMS perfilado de los grupos verdaderos, resumido por escena y tamaño,
queda entre 1,292 y 1,794 cents en validación; entre 1,464 y 1,709 en OOD
beta; y entre 1,420 y 1,649 en OOD polifonía. Son rangos de cinco medias
por tamaño, no rangos de todos los grupos ni intervalos inferenciales.

La heurística ofrece una ilustración de la distinción en OOD polifonía.
Cada celda promedia por igual grupos del mismo tamaño/categoría dentro
de escena y después escenas elegibles. Entre paréntesis figura la cantidad
de escenas sobre 32; no se mezclan tamaños ni se trata cada grupo como réplica.

| Miembros | Grupos puros: RMS cents (escenas) | Grupos mixtos: RMS cents (escenas) |
|---:|---:|---:|
| 3 | 1,251 (10) | No evaluable (0) |
| 4 | 1,190 (12) | No evaluable (0) |
| 5 | 1,283 (9) | No evaluable (0) |
| 6 | 1,564 (13) | 1,492 (2) |
| 7 | 1,533 (17) | 3,698 (2) |
| 8 | 1,590 (12) | 298,776 (2) |

El caso de seis miembros impide convertir residual pequeño en garantía de
fuente única. El de ocho muestra que algunos grupos predichos ajustan muy
mal sobre esta grilla, pero un residual alto es una cota superior del mejor
ajuste continuo, no un certificado de incompatibilidad. La tabla es una
ilustración retrospectiva, no una comparación causal entre métodos.

El resumen conjunto de las tres semillas pierde cobertura en varios
estratos mixtos. Por ejemplo, pares+BCE en OOD beta tiene cero escenas con
las tres semillas elegibles para grupos mixtos de tamaños 3–7, y sólo una
para tamaño 8. No se sustituye ese vacío por el promedio de las semillas
disponibles. Los trece lectores individuales, todos los estratos, las
medianas/máximos endpoint y las coberturas permanecen en el resumen completo.
El residual endpoint local y el RMS conjunto son estimandos distintos:
su diferencia numérica no demuestra una brecha local-global certificada.

## Cobertura y reproducibilidad

Conteos de grupos en las 13 particiones por escena; son inventario de
lecturas repetidas, no tamaño muestral independiente. Los grupos menores
de tres y mayores de ocho se preservan sin asignarles un RMS comparable.

| Slice | Grupos predichos | Ajuste 3–8 | Menores de 3 | Mayores de 8 | Referencias verdaderas |
|---|---:|---:|---:|---:|---:|
| Validación | 1.072 | 1.008 | 45 | 19 | 81 |
| OOD beta | 1.368 | 1.024 | 333 | 11 | 77 |
| OOD polifonía | 1.691 | 1.464 | 120 | 107 | 128 |

Se conservaron 4.417 registros de grupo, 1.152 registros de presión y 942
claves de grupo observable, distinguiendo escena y miembros; 653 admiten el
ajuste por cardinalidad. La grilla fina no empeoró ningún mínimo: mejora
entre 0 y 0,780782 cents frente a la gruesa. No aparecieron co-mínimos
dentro de 1e-9 cents en esos 653 fits; eso no demuestra identificabilidad
continua de índices, beta u offset.

Primaria: 5,848 s y 88,53 MiB RSS. Replay: 5,886 s y 88,42 MiB. Los 1.349
artefactos científicos son idénticos byte a byte; tiempo y memoria se
conservan aparte. El preflight conservador había proyectado 279,25 s por
corrida sin deduplicación. Las 25 fuentes congeladas del experimento previo
permanecieron intactas. Las pruebas mecánicas sumaron 24 casos aprobados.

La primera versión conservó todo el crudo, pero omitió del resumen los
endpoints disponibles en 137 grupos mayores de ocho. V2 corrige su
elegibilidad por métrica: mantiene RMS conjunto no evaluable y muestra
mediana/máximo local. Sólo cambian configuración, escenas y resumen;
los otros 1.346 artefactos científicos son idénticos a v1. La versión
original permanece preservada, no reescrita.

- [Manifest primario](../../data/atencion_armonica/source_coherence_v2/manifest.json).
- [Manifest de replay](../../data/atencion_armonica/source_coherence_v2_replay/manifest.json).
- [Configuración y hashes de inputs](../../data/atencion_armonica/source_coherence_v2/config.json).
- [Grupos y witnesses](../../data/atencion_armonica/source_coherence_v2/groups.jsonl).
- [Estimandos por escena](../../data/atencion_armonica/source_coherence_v2/scenes.jsonl).
- [Preflight mecánico](../../data/atencion_armonica/source_coherence_preflight_v1/manifest.json).

Los artefactos pesados son locales. El manifest primario tiene SHA-256
`61aa27c523959556dd611191fa204d6d94d66978715658fc57b434973f9fb06b`.

## Balance geométrico

Se construyó una operación computable de coherencia conjunta para un grupo
candidato bajo la familia de fuente, manteniendo la incertidumbre de
pertenencia separada del ajuste.
Eso acerca el diseño al objeto físico, pero todavía sólo explica estados
ya producidos: no mejora una inferencia ni demuestra generalización.

El siguiente contraste debe hacer actuar esa información dentro de un
sistema que produzca particiones, frente a controles con igual acceso y
búsqueda. No queda justificado endurecer un umbral físico, retocar lambda
en estos tests ni asumir que sustituir pares por fuentes latentes resolverá
la ambigüedad. La nueva pregunta requiere protocolo previo y datos frescos;
el baseline descriptor-guided, los controles genéricos y las alternativas
de audio observado permanecen disponibles.

El [diseño de lector estructurado](PLAN_SOURCE_STRUCTURED_READER.md) propone
ese contraste sobre candidatos comunes y checkpoints BCE congelados:
fuente compartida frente a energía de pares, costos desacoplados y
compatibilidad local. Es un plan reauditorado, todavía no implementado.
