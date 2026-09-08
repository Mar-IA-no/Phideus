# Coherencia de fuente y presión de la pérdida

Estado: diseño reauditorado independientemente; habilita implementación y
preflight CPU. No cambia el protocolo ni los resultados ya cerrados.

## Pregunta y motivo

El [contraste anterior](RESULTS_SHARED_PARTIAL_STUDY.md) no sostuvo la
ventaja de la pérdida física sobre BCE en OOD beta. A la vez, Brier y
partición ordenaron de manera diferente red y heurística. Antes de otra
campaña neuronal, este goal examina dos mecanismos concretos: cuánto
penaliza la pérdida relaciones verdaderas y si los grupos predichos admiten
un único conjunto de parámetros de la familia, en lugar de witnesses
independientes para cada triple. No se asume que ninguno explique el daño.

La correspondencia interrogada es familia espectral → conjunto de parciales
con parámetros compartidos → pertenencia aprendida. Se conserva la familia
`f_n=n*f0*sqrt(1+beta*n²)` y el antecedente de fuente como objeto latente del
frente histórico. La recuperación primaria dirigida de Hodgkinson et al.
ya está archivada; no hace falta otra ola bibliográfica. Una carencia puntual
de implementación se investigará sólo si bloquea este diagnóstico.

## Acceso y muestra fijada antes del diagnóstico

Usar artefactos de la campaña cerrada, sin nuevos draws, forwards ni training.
Tomar IDs 0..31 de validation, ood_beta y ood_polyphony: 96 escenas en total.
Es una muestra determinística diagnóstica de corpus abiertos, no un test
prospectivo ni una estimación poblacional. No elegir casos por error, score,
residual o partición. Conservar también los casos sin grupos evaluables.
Validation queda rotulada `READER_SELECTION_IN_SAMPLE`, porque intervino
en los umbrales; los otros dos slices son `OPEN_TEST_POSTHOC`. No combinar
los tres splits ni presentar ninguno como confirmación independiente.

Lectores: cuatro brazos de pares × tres semillas y la heurística analítica,
13 por escena; umbrales y particiones ya guardados, nunca reajustados.
La comparación token-only del goal anterior permanece conservada, pero no
se usa para atribuir la regularización con arquitectura común en éste.
Cargar logits sólo para los doce lectores neuronales. Inferencia geométrica
recibe exclusivamente q float32 observado y los miembros del grupo a
examinar. Etiquetas, índices verdaderos, f0 y beta de sidecars quedan fuera
del fitter; source_ids se usan después para evaluar y para una referencia
diagnóstica privilegiada sobre grupos verdaderos, rotulada como tal.

## Diagnóstico de presión sobre relaciones

Recalcular en float64 las probabilidades de los logits conservados y usar
los pesos físicos/sham cuantizados float32 como en training. Es análisis
numérico posterior, no replay bitwise de entrenamiento. Verificar simetría
y forma antes de usar una arista no ordenada como coordenada.

Para una escena de N eventos, E=N(N−1)/2 aristas y T=N(N−1)(N−2)/6 triples,
`g_BCE(e)=(p_e-y_e)/E` es la derivada respecto del logit de arista simétrico.
La coordenada es `s_e=z_ij=z_ji`: se perturban ambas entradas a la vez.
Por tanto equivale a sumar las dos derivadas del BCE sobre el tensor lleno,
no a tomar una sola de sus orientaciones.
Para `L=sum_t w_t product_(e in t)p_e/T`,
`g_L(e)=p_e(1-p_e)*sum_(t contains e)w_t product_(other edges)p/T`.
La magnitud ponderada usa lambda=0.1, sin seleccionarla. Calcular física y
sham en los mismos logits, no sólo la pérdida nativa de cada brazo.

Conservar por escena y tipo de relación positiva/negativa: conteos, suma
absoluta BCE, suma de presión física/sham ponderada, cociente entre esas
sumas cuando BCE sea no nula y estado no evaluable en otro caso. También
guardar medias de peso en triples de una fuente y mixtos, y masa de pérdida
que recaería sobre la pertenencia verdadera. No convertir una derivada
respecto de logits en gradiente de parámetros o en causa del entrenamiento.
La dirección no negativa de esta penalización es algebraica; el experimento
mide su magnitud y localización, no presenta el signo como descubrimiento.

Estimandos exactos por clase c∈{0,1}: `B_c=sum_(y_e=c)|g_BCE(e)|`,
`P_c=0.1*sum_(y_e=c)g_fis(e)`, `S_c=0.1*sum_(y_e=c)g_sham(e)`.
Los cocientes son P_c/B_c y S_c/B_c; clase vacía o B_c=0 produce null con
estado explícito. Para `a_t=w_t*product_(e in t)p_e`, guardar
`L=sum_t a_t/T`, `F_true=sum_(t misma fuente)a_t/sum_t a_t` (null si suma=0)
y `L_truth=sum_(t misma fuente)w_t/T`, que sustituye probabilidades por
pertenencia verdadera binaria. Repetir los tres para sham en los mismos
logits. `L_truth` es referencia privilegiada, no loss usada para inferencia.
Las medias de peso true/mixed dividen por el número de triples de cada
clase, con null si falta. Sham no evaluable conserva pesos cero del cache,
estado propio y BCE intacta; no se presenta su cero como mejora física.

## Ajuste conjunto de una fuente: witness aproximado, no certificado

Para un grupo de m miembros, ordenar q por frecuencia y considerar todos
los subconjuntos crecientes de m índices distintos de 1..8. Con beta fijo,
`u_i=q_i-log(n_i)-0.5*log1p(beta*n_i²)`; el offset óptimo es la media de u.
Minimizar RMS de `u-mean(u)` sobre esas asignaciones y las grillas fijadas
abajo. `residual_cents=(1200/log(2))*sqrt(mean((u-mean(u))²))`.
Reportar residual en cents, índices propuestos, beta y offset testigo.

Es un mínimo sobre grilla, no el mínimo continuo: el residual es una cota
superior del mejor ajuste, y un residual alto no certifica incompatibilidad.
No introducir un umbral de aceptación física. Repetir cada ajuste con la
grilla anidada de 1025 valores y preservar ambos residuales y su diferencia;
no seleccionar resolución por el resultado. Crear una sola grilla fina
float64 `np.geomspace(1e-5,0.02,1025)`; la gruesa es exactamente `fine[::4]`.
Precomputar por tamaño las plantillas centradas de log(n)+0.5log1p(beta*n²),
evaluar una vez la fina y obtener la gruesa de esas mismas filas. Persistir
grid/version/hash. Enumerar asignaciones de índices lexicográficamente.

Para cada resolución, guardar mínimo, número de celdas (asignación,beta)
con residual ≤mínimo+1e-9 cents y gap al siguiente valor fuera de esa banda
(null si no existe). Elegir entre esos co-mínimos el witness de menor tupla
de índices y luego menor beta, sólo para serializar; guardar también su
residual, que puede diferir del mínimo hasta la tolerancia. Declarar
`index_authority=WITNESS_NOT_IDENTIFIED` y
`offset_authority=CENTERED_LOG_GAUGE`: el offset no es f0 en Hz recuperado.
Exigir `coarse_min-fine_min >= -1e-9 cents`; esa tolerancia es numérica,
no un umbral físico. Reutilizar cada grupo observable
idéntico entre lectores dentro de una escena para evitar cómputo redundante.
Clave de cache: `(split,scene_id,sorted_member_indices,grid_hash)`, con
conteos de hits/misses; no reutilizar entre observaciones distintas.

Grupos m<3: `UNDERCONSTRAINED`, sin puntaje de coherencia comparable.
Grupos m>8: `OUTSIDE_DECLARED_CARDINALITY`, conservados en denominadores;
no partirlos para obtener un ajuste. Para m=3..8 informar tamaño y pureza
de pertenencia sólo después del fit, sin orientar el witness por truth:
`purity=max_s count(source_id=s)/m`, categoría pure si todos comparten ID
y mixed en otro caso. No emplear un umbral de pureza ajustable.
Las fuentes verdaderas se ajustan mediante la misma firma observable: sus
miembros constituyen una referencia privilegiada explícita, no un decoder.

Para separar compatibilidad local de conjunta, conservar también los
residuales de todos los triples internos ya cacheados. No comparar sus
valores como si fueran la misma distancia: el residual endpoint del
descriptor y el RMS conjunto perfilado tienen estimandos diferentes.
El diagnóstico muestra juntos RMS conjunto fino, mediana y máximo de los
residuales endpoint internos, siempre por tamaño y categoría pure/mixed.
No restar ni dividir residual local y global; no declarar certificada una
brecha local-global sólo por diferencia numérica.

## Implementación, verificaciones y límites

Código nuevo y outputs nuevos; las 25 fuentes del freeze anterior permanecen
intactas. Verificar manifests, hashes, identidad ordenada, IDs y particiones
completas antes de analizarlas; no basta confiar en el resumen del chat.
Guardar por escena, lector y grupo inputs, derivadas/métricas, witnesses y
estados; conservar configs, fuentes, runtime, tiempos y memoria. Las medias
son primero por escena con denominadores elegibles/total y distribución de
tamaños. Roll-up fijado: para cada escena×lector×tamaño m×categoría pure/mixed,
promediar por igual los grupos elegibles de ese estrato; reportar también
número de grupos, miembros y estados de todos los grupos, incluidos <3 y >8.
No mezclar tamaños para afirmar separación. Conservar los 13 lectores por
separado. Para resumen de brazo, promediar después las tres seeds dentro
de escena sólo donde las tres sean elegibles, reportando la cobertura
completa/total; por último promediar por igual escenas dentro de cada split.
La heurística y la referencia privilegiada de grupos verdaderos son estratos
propios, nunca seeds adicionales. Aplicar el mismo orden seed→escena a los
estimandos de presión, conservando null y denominadores por métrica.
No tratar grupos o triples como réplicas independientes. No agregar
intervalos inferenciales a esta muestra diagnóstica determinística.

Tests CPU: derivada analítica contra diferencias finitas de perturbaciones
simétricas; clases ausentes y denominador cero; fitter sin acceso a truth;
familia conocida con beta sobre grilla; permutación y cambio de offset;
grupos pequeños y cardinalidad fuera de prior; refinamiento anidado no
empeora residual más allá de tolerancia numérica; identidad de artefactos.
Primaria y replay deben coincidir salvo tiempos/RSS/path de salida.

Presupuesto: un proceso CPU experimental, 120s para fixtures/preflight y
300s para cada corrida real, RSS máximo1GiB. Preflight obligatorio sólo
mecánico, sin corpus/truth: N=32, tamaños3..8 y trece particiones distintas
con ambos niveles de grilla, además del cálculo de presión. Medir throughput
del peor tamaño de fit y RSS. Proyectar por corrida hasta
`96*(13+1)*floor(32/3)=13440` fits sin deduplicación, incluyendo la referencia
privilegiada, más96×12 cálculos de presión y lectura/guardado medidos sobre
fixtures. La proyección no es cota garantizada; sólo ejecutar si proyecta
≤300s por corrida y RSS<1GiB, manteniendo timeout y guarda real de memoria.
Sin Torch ni GPU. Si el costo
no cabe, preservar `INCOMPLETE` y revisar costo antes de ampliar recursos;
no recortar lectores, escenas o grillas a posteriori. Esta evaluación pequeña
de arrays y asignaciones no justifica ocupar la 3090 ni Mendieta.

## Cierre y bifurcaciones

Termina con implementación, primaria/replay, informe con todos los casos,
auditoría independiente de evidencia y de alineación, documentación y
commit/push. Puede concluir que ninguna de las dos explicaciones queda
sostenida: no exige un mecanismo favorable. Debe elegir un siguiente goal
construible desde lo observado, sin una sucesión ilimitada de diagnósticos.

Si se observa presión adversa localizada, una pérdida revisada necesitará
contraste nuevo y datos prospectivos, no tuning de estos tests. Si el límite
es compartir parámetros, la alternativa de fuente latente o un decoder
conjunto merece un diseño explícito frente al baseline descriptor-guided.
Si sólo aparece una diferencia entre ranking, probabilidades y lectura,
aislarla antes de atribuirla a geometría; no volver por defecto a optimizar
tau. Audio detectado y etapas históricas se preservan como alternativas,
sin confundir cambio de aparato con corrección de este mecanismo.

No promueve arquitectura ni decide GO/NO-GO. La pregunta global al cierre
es si este diagnóstico realmente acerca la representación a una fuente
geométricamente coherente, o sólo mejora una explicación retrospectiva.
