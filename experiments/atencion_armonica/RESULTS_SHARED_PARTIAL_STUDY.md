# Compatibilidad entre parciales: resultado del contraste neuronal

Fecha: 2026-09-07. Estado: experimento completo; evidencia, interpretación y
límites revisados independientemente. No hay promoción arquitectónica ni GO/NO-GO.

La pérdida de compatibilidad física no mejoró el escenario primario frente
a la misma red entrenada sólo con BCE: aumentó Brier en las tres semillas.
Su promedio fue mejor que el de los controles regularizados, pero esa
ventaja cambió de signo en una semilla. El resultado cuestiona esta pérdida
y receta concretas, no toda geometría armónica ni toda regularización física.

## Qué se comparó

El [protocolo congelado](PLAN_SHARED_PARTIAL_COMPATIBILITY.md) retiró las
amplitudes y entregó las mismas frecuencias observadas y descriptores a todos
los brazos. Cuatro comparten estados de pares B-local y difieren sólo en la
pérdida: BCE, compatibilidad física, pesos desacoplados de los triples
(sham) y transitividad genérica. El quinto es token-only con descriptores.
La referencia analítica usa soporte medio de triples y el mismo lector de
particiones; es una heurística pertinente, no un estado del arte clásico.

Se completaron cinco brazos × tres semillas × 50 épocas, con 8.192 escenas
de training. Los dieciséis lectores se eligieron exclusivamente sobre 1.024
escenas de validación y se congelaron antes de generar los cinco tests de
1.024 escenas cada uno. No se ajustaron checkpoints, pérdidas ni umbrales
por lo observado en test. El contraste principal atribuye objetivos con
arquitectura común; no prueba una nueva arquitectura triangular.

## Contraste primario: inarmonicidad fuera del rango de training

Delta = Brier de compatibilidad menos Brier del control; negativo favorece
compatibilidad. La unidad de remuestreo es la escena. El resumen conjunto
promedia primero los tres deltas dentro de cada escena y después realiza
2.000 remuestras pareadas. Sus intervalos del 95% son descriptivos, no
ajustados por multiplicidad ni representativos de nuevos corpus de training.

| Control | Delta medio | Intervalo conjunto | Deltas por semillas 721 / 722 / 723 |
|---|---:|---:|---|
| BCE sola | +0,003579 | [+0,002849; +0,004316] | +0,000960 / +0,003362 / +0,006415 |
| Pesos desacoplados | −0,002478 | [−0,003166; −0,001755] | −0,001831 / −0,007606 / +0,002003 |
| Transitividad | −0,002575 | [−0,003292; −0,001871] | −0,003336 / −0,008582 / +0,004193 |

Las semillas completas son `2026090721/22/23`. Transitividad es el control
obligatorio de atribución, no una comparación omitible por su resultado.
El promedio favorable frente a regularizadores no basta para concluir
utilidad incremental frente al baseline sin regularizar.

## Los cinco escenarios

Brier: media de métricas por escena, promediada entre tres semillas; la
heurística tiene un único lector. Menor es mejor. Estos slices secundarios
no sustituyen el contraste primario ni una evaluación de nueva muestra.

| Método | IID | OOD beta | OOD polifonía | OOD ruido | Familia deformada |
|---|---:|---:|---:|---:|---:|
| Pares + BCE | 0,011879 | 0,090262 | 0,048621 | 0,077725 | 0,087006 |
| Pares + compatibilidad | 0,012027 | 0,093841 | 0,046904 | 0,078594 | 0,088867 |
| Pares + pesos desacoplados | 0,011889 | 0,096319 | 0,044756 | 0,078861 | 0,088496 |
| Pares + transitividad | 0,011756 | 0,096416 | 0,047761 | 0,078105 | 0,088914 |
| Tokens + descriptores | 0,060231 | 0,112960 | 0,073686 | 0,117473 | 0,116219 |
| Soporte analítico | 0,140484 | 0,141320 | 0,125518 | 0,179479 | 0,162172 |

Los estados de pares con BCE tienen menor Brier medio que token-only en
los cinco slices bajo esta receta. Eso no aísla un único componente interno
ni valida universalmente Pairformer. En mayor polifonía, compatibilidad
mejora frente a BCE, pero sham obtiene un Brier medio menor: tampoco allí
puede atribuirse automáticamente la mejora al significado físico del peso.

Brier y partición no ordenan igual los métodos. En OOD beta, la heurística
obtiene ARI 0,851012 y partición exacta 0,513672, frente a 0,800651 y
0,311523 de pares+BCE, pese a su peor Brier. En familia deformada también
tiene mayor ARI medio (0,778232 frente a 0,770822), pero menor exactitud de
partición (0,332031 frente a 0,360352). No corresponde elegir la métrica
después para declarar ganador; la disociación merece una pregunta separada.

El slice ≤10 cents conserva 210/285/515/229/226 escenas elegibles en
IID/beta/polifonía/ruido/deformada, respectivamente. Todos esos pares tienen
target negativo en estos draws; AP, AUC y recall positivo quedan no
evaluables. Un Brier near-collision pequeño no demuestra separación de
picos fusionados: el productor conserva dos eventos observados distintos.

## Alcance y continuidad

La evidencia es sintética, de un corpus de training y tres inicializaciones.
No hay audio detectado, prueba de identificación global de fuentes ni
certificación de que todos los triples compatibles compartan parámetros.
La equivariancia a permutación y el centrado de escala se comprobaron como
propiedades mecánicas; no constituyen una geometría aprendida demostrada.
El valor de lambda y la forma de penalización no se optimizaron después.

El siguiente paso debe examinar la correspondencia entre pertenencia
predicha y una fuente con parámetros compartidos, junto con el efecto de
la penalización sobre relaciones verdaderas. Reutilizar los estados ya
guardados permite un diagnóstico post-hoc generador de hipótesis, sin
reentrenar ni volver al tuning de umbrales. Ni test ni validación, ya usada
para lectores, recuperan independencia por ese reanálisis. Una confirmación
posterior requerirá protocolo congelado y muestra fresca. La pregunta se
concreta en el [plan de coherencia de fuente](PLAN_SHARED_SOURCE_COHERENCE.md),
un goal separado y finito cuya implementación depende de su auditoría.

## Evidencia recuperable

- [Freeze prospectivo](../../data/atencion_armonica/shared_partial_evaluation_freeze_v1.json): fuentes, checkpoints, lectores y auditoría anteriores a test.
- [Training](../../data/atencion_armonica/shared_partial_training_v1/manifest.json) y [selección en validación](../../data/atencion_armonica/shared_partial_validation_selection_v1/manifest.json).
- [Logits de test](../../data/atencion_armonica/shared_partial_test_logits_v1/manifest.json): 75 archivos NPZ, con 1.024 matrices por archivo (una por escena); 76.800 matrices en total, con identidad por observación.
- [Reporte completo](../../data/atencion_armonica/shared_partial_test_analysis_v1/report.json) y [manifest de análisis](../../data/atencion_armonica/shared_partial_test_analysis_v1/manifest.json): 80 lecturas, 25 archivos de desacuerdo entre semillas y los deltas por escena; 107 artefactos hasheados más el manifest (108 archivos).

El manifest de análisis tiene SHA-256
`62eb10e502bfd32b99ec4a5acd13f6777999b08f4266fedf4f8744bd98a4c05b`.
Se preservan checkpoints 10/25/50 y last, optimizador, RNG, curvas, datos y
logits float32. Training duró 1.461,24 s y reservó como máximo 1,64 GiB;
el forward de test duró 17,16 s y reservó 358 MiB. El análisis posterior
duró 146,23 s por CPU. El presupuesto acumulado cargó 1.488,64 s incluyendo
perfil y overhead de jobs, sin reserva activa ni sobrepaso; no es sólo
tiempo de kernels. Los artefactos pesados son locales, no se incorporan a Git.
