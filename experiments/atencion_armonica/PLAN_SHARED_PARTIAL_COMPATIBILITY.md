# Compatibilidad física entre parciales: contraste de pérdidas

Estado: **plan corregido y reauditorado independientemente, habilitado para implementación y preflight CPU**, no autorización de training todavía. Continuación de §5 del [plan general](PLAN_GEOMETRIC_RESEARCH_ACTION.md); una hipótesis seleccionada para el contraste, no una arquitectura promovida. Fecha: 2026-09-07.

## 1. Elección y pregunta finita

Se elige la hipótesis de **compatibilidad con una familia compartida de parciales**. La alternativa de **consistencia genérica de particiones** se conserva como control, no se borra ni se llama física por ser relacional. La decisión se apoya en el [diagnóstico de amplitudes](RESULTS_ENERGY_PARTITION_AUDIT.md): una partición consistente puede recuperarse sin frecuencias en la muestra histórica; por tanto, consistencia y armonía no son equivalentes.

Pregunta: con la misma red, los mismos descriptores observables y la misma supervisión, ¿una pérdida que desalienta pertenencias triples incompatibles con la familia espectral mejora generalización frente a BCE sola y frente a la misma penalización desacoplada de los triples físicos? La regularización genérica de transitividad y una red token-only con descriptores completan el contraste. No se repite todo el factorial histórico ni se atribuye el resultado a una arquitectura nueva.

La correspondencia que se ensaya es acotada: familia espectral → compatibilidad entre tres observaciones → organización de probabilidades de pertenencia por la pérdida → generalización. No prueba HIT completa, tensión informacional física, separación de audio ni medición real. Los slots de fuente con parámetros compartidos permanecen como implementación alternativa futura de esta misma familia, ya descrita en el frente histórico; no se implementan en paralelo.

## 2. Relación física, observación y límites

Modelo generativo: `f_n = n f0 sqrt(1 + beta n²)`, con índices presentes distintos `n∈{1,...,8}`. La fórmula se conserva del generador histórico; la fuente primaria dirigida de Hodgkinson, Timoney y Lazzarini (DAFx 2010) estudia parciales de cuerdas de acero con parámetros compartidos, no agrupamiento polifónico universal. La lectura y sus límites están archivados en Biblioteca. Fuente primaria:

https://mural.maynoothuniversity.ie/id/eprint/3961/1/A_MODEL_OF_PARTIAL_TRACKS_FOR_TENSION-MODULATED_STEEL_STRING_GUITAR_TONES.pdf

Derivación propia: para índices candidatos, `f_n²/n² = f0² + f0² beta n²`. Dos observaciones fijan dos parámetros; una tercera introduce una restricción. Los índices verdaderos nunca entran en inferencia: se consideran las 56 ternas crecientes posibles. Un residual bajo no identifica necesariamente una única fuente, ni certifica toda una partición: diferentes triples pueden ajustarse con parámetros distintos.

La observación es una lista desordenada de log-frecuencias con error de medición simulado, centradas por mezcla y cuantizadas a float32. Todos los modelos y descriptores reciben exclusivamente esa lista y su máscara de padding. No reciben amplitudes, orden por fuente, `source_id`, índices de parcial, f0/beta verdaderos ni k verdadero. Las amplitudes desaparecen de este contraste, no se declara que un generador nuevo las haya vuelto inocuas. Tampoco se usa restauración de parciales por amplitud.

La escala absoluta se conserva en el sidecar para reconstrucción, no como input: la query de pertenencia de este banco es invariante a escala sintética global con beta fijo. Esa transformación no equivale a variar tensión física: f0 y beta pueden cambiar conjuntamente con la tensión. La invariancia del centrado ya existía; su prueba es un sanity, no evidencia novedosa. Permutar picos debe permutar las predicciones. El problema temporal, las colisiones de audio y la detección real quedan fuera del aparato simulado.

## 3. Datos nuevos y separación de autoridades

Productor nuevo, separado del histórico y NumPy puro. Cada fuente tiene f0 log-uniforme en 100–500 Hz, beta log-uniforme en 1e-4–1e-3 y un número de parciales uniforme entero entre 4 y 8; seleccionar ese número de índices sin reemplazo, independientemente de f0, beta y otras fuentes. Fuentes independientes, sin diccionario de ratios entre fundamentales. No rechazar escenas por dificultad, cercanía de frecuencias, score de descriptor o coincidencia con el target. La ambigüedad global de particiones queda **no adjudicada**: no se enumeran todas las explicaciones compatibles. Conservar todas las escenas; reportar near-collisions observables, entropía binaria media de pares y desacuerdo entre seeds, sin convertir la partición generativa en prueba de identificabilidad observacional.

Medición: `log(f_observada)=log(f_ideal)+Normal(0, sigma_cents*log(2)/1200)`, sigma=2 cents en train/validation/IID. Orden congelado: calcular log-f ideal en float64, sumar sensor en float64, centrar el vector observado en float64, castear a float32, serializar/entregar. Todos los descriptores se derivan exclusivamente de ese float32 convertido a float64 para aritmética. No audio renderizado, picos fusionados ni falsos positivos; dos eventos generativos cercanos permanecen dos observaciones. El target binario por par es pertenencia generativa, no una anotación perceptual. Evaluar todos los pares fuera de diagonal; informar además el slice de separación ≤10 cents calculado desde el mismo canal observado, sin retirarlo de los denominadores principales.

Una escena constituye la unidad independiente. Seeds de datos distintas por split, con `SeedSequence([master_seed, scene_id])`; ningún source pool reutilizado ni familias compartidas entre splits. Dentro de cada escena, permutar picos antes de serializar. Separar JSON de observaciones del sidecar de verdad; el dataset de inferencia no abre sidecars.

| Split | Escenas | Cambio respecto de train | Seed |
|---|---:|---|---:|
| Desarrollo mecánico | 64 | Igual distribución; nunca entra en métricas finales | 2026090710 |
| Train | 8192 | k=2 o 3 equiprobable, parámetros anteriores | 2026090711 |
| Validation | 1024 | Igual distribución | 2026090712 |
| IID test | 1024 | Igual distribución | 2026090713 |
| OOD beta, primario | 1024 | beta log-uniforme 3e-3–1e-2 | 2026090714 |
| OOD polifonía | 1024 | k=4 | 2026090715 |
| OOD ruido | 1024 | sigma=8 cents | 2026090716 |
| Familia deformada sintética | 1024 | `f_n=n f0 sqrt(1+beta n²+gamma n⁴)`, gamma por fuente log-uniforme 5e-6–5e-5, sensor=2 cents | 2026090717 |

El último slice estresa una deformación estructurada de la familia; no se presenta como ley física alternativa ni como dificultad igualada, y no demuestra especificidad causal por sí solo. Sus f0, beta, cardinalidades y sensor mantienen la distribución de train; gamma no se ajusta por desempeño. El término n⁴ rompe la afinidad de f_n²/n² respecto de n² para índices fijos, pero no garantiza identificar esa deformación desde toda observación incompleta con índices desconocidos. Se descarta el borrador anterior de jitter gaussiano extra: en una escena estática sería indistinguible de aumentar el ruido del sensor. Los tests no se abren para seleccionar lambda, lector, features, checkpoints ni duración. La semilla no sustituye un split: comprobar hashes/IDs, ausencia de escenas idénticas y escritura nueva. Sólo train/validation/desarrollo se usan antes del freeze de implementación.

## 4. Descriptor geométrico observable

Reconstruir cálculos únicamente desde log-f centrado float32 entregado a la red, usando float64 para aritmética. Para cada triple de picos distintos, ordenar sus log-f observados `q1≤q2≤q3`. Para cada terna de índices `a<b<c` de 1..8, usar el ratio externo para proponer beta:

`t = exp(2*(q3-q1))*(a/c)²`, `beta_endpoint = (t-1)/(c²-t*a²)`.

El intervalo público del descriptor es `[1e-5, 2e-2]`, común a todos los brazos y que incluye los regímenes de evaluación. Definir `d_ua(beta)=log(u/a)+0.5*(log1p(beta*u²)-log1p(beta*a²))`. Si `q3-q1 <= d_ca(beta_lo)`, elegir beta_lo; si `q3-q1 >= d_ca(beta_hi)`, elegir beta_hi; sólo en el interior evaluar beta_endpoint y restringir el resultado numérico a `[beta_lo,beta_hi]`. No usar beta verdadero para elegir ni calibrar el intervalo.

Después de elegir beta_hat para cada terna, calcular en este orden:

```text
e_b = q2-q1-d_ba(beta_hat)
e_c = q3-q1-d_ca(beta_hat)
r_abc = (1200/log(2))*sqrt((e_b²+e_c²)/2)
R = min_{1<=a<b<c<=8} r_abc
w = R²/(R²+10²)
support = 1-w
```

Los empates entre ternas no eligen una fuente: R es sólo el mínimo escalar; si se registra argmin diagnóstico, usar el primer índice de la enumeración lexicográfica. Inputs, divisiones o outputs no finitos abortan el artefacto con estado incompleto, nunca se filtran escenas en silencio. Es un residual condicional al ajuste del ratio externo, **no** distancia exacta a una variedad ni certificado de imposibilidad con ruido. La escala de 10 cents define la penalización, no un umbral científico de éxito ni una probabilidad calibrada.

Para cada par, promediar support sobre todos los terceros distintos. Éste es un descriptor explícito disponible en **todos** los brazos neuronales. Features comunes: diferencia absoluta log-f, residual a ratio racional simple, residual common-f0 histórico, soporte triple medio; clase de ratio racional histórica. Tokens: log-f centrado y un segundo canal fijo cero. Se preservan las dimensiones del código previo reemplazando el canal de amplitud, sin modificar el generador ni las features históricas.

El residual completo por triple queda disponible a las pérdidas física y sham, no sólo al candidato: ambas reciben el mismo multiconjunto. Los labels son exclusivos de BCE y evaluación. La información adicional de la pérdida es una relación computada desde observables, no supervisión de parámetros latentes. Que el descriptor resulte suficiente es una salida válida, no un motivo para debilitarlo.

## 5. Brazos y objetivos

Se reutilizan configs históricas `B-local` y `A-rich` sin ampliar profundidad ni anchura. Las cuatro variantes de `B-local` tienen arquitectura, features, init, orden de datos y número de pasos idénticos por seed. Es un contraste de objetivos sobre estados de pares, no de triangle update. Se usa un runner nuevo que no inicializa CUDA cuando el dispositivo es CPU.

Sea `p_ij=sigmoid(logit_ij)`. BCE se promedia primero sobre pares no diagonales de cada escena y después sobre escenas. Las penalizaciones también se promedian por escena; nunca por cantidad global de triples del batch, para no sobreponderar polifonía.

| Brazo | Red | Pérdida | Función |
|---|---|---|---|
| Pares-descriptores | B-local | BCE | Baseline principal idéntico |
| Pares-compatibilidad | B-local | BCE + 0.1 L_fis | Hipótesis seleccionada |
| Pares-sham | B-local | BCE + 0.1 L_sham | Misma penalización, alineación física rota |
| Pares-transitividad | B-local | BCE + 0.1 L_trans | Consistencia genérica preservada |
| Tokens-descriptores | A-rich | BCE | Baseline descriptor-guided sin estados persistentes de par |

`L_fis = mean_{i<j<k} w_ijk * p_ij*p_ik*p_jk`.

`L_sham` usa los mismos pesos w permutados entre triples de la escena. Ordenar picos canónicamente por el log-f float32 observado y enumerar triples lexicográficamente; sea T su número. Usar `rng=default_rng(SeedSequence([2026090730, split_master_seed, scene_id]))`, `shift=rng.integers(1,T)` y `w_sham[t]=w[(t+shift)%T]`. Es una rotación no nula sin puntos fijos, independiente del target; conservar shift y mapa al orden entregado. Esos IDs son metadatos públicos para el control, no inputs de la red ni parámetros verdaderos. Si hay empate exacto en log-f o T<2, conservar escena, BCE y step, fijar sólo L_sham=0 y registrar control no evaluable; no desempatar con source_id. Verificar multiconjunto, equivariancia y proporción de valores efectivamente cambiados: una permutación no identidad no garantiza cambio de valores constantes. Este sham conserva marginal y costo de la penalización, pero puede perjudicar artificialmente: una ventaja sólo frente a sham no alcanza; el contraste frente a BCE es co-primario.

`L_trans` promedia sobre cada triple las tres cantidades `relu(p_ij*p_jk-p_ik)²`, `relu(p_ij*p_ik-p_jk)²`, `relu(p_ik*p_jk-p_ij)²`. Es un surrogate de consistencia, no una identidad obligatoria para marginales calibradas de un posterior sobre particiones. Lambda 0.1 es fija para esta receta; no hay selección por test ni sweep de coeficientes. Igualar lambda no iguala gradientes: reportar componentes y normas de gradiente en desarrollo/train, sin atribuir diferencias sólo al contenido geométrico del control genérico.

Todas las penalizaciones admiten soluciones degeneradas (por ejemplo todos separados); BCE común es la fuerza discriminante. Conservar métricas de prevalencia predicha, recall de pares positivos y predicción constante. No afirmar que la loss prueba consistencia global de una familia compartida: los triples podrían sostener betas distintas y las probabilidades de par no definen automáticamente una distribución sobre particiones.

Referencia no neuronal: matriz de soporte triple medio del §4, sin fitting de red, seguida por el mismo lector. Es una heurística analítica pertinente, no un solver físico óptimo ni estado del arte completo. Los modelos con descriptores son el control aprendido fuerte. Reportar si la heurística ya resuelve el banco; no descartar escenas para impedirlo.

## 6. Entrenamiento, lectura y evaluación

Tres seeds de entrenamiento: 2026090721, 2026090722, 2026090723. Cada brazo usa 50 épocas, batch 128, AdamW lr=3e-4, weight_decay=1e-4, warmup 5% de pasos y cosine hasta cero. Sin early stopping ni búsqueda de arquitectura; evaluar `last_epoch`. Seeds y batches emparejados por réplica; seeds de datos no se confunden con réplicas de entrenamiento. Los resultados se condicionan a un único conjunto de train, no a variación entre corpus independientes.

Co-primarios: diferencias de Brier por escena en OOD beta, Pares-compatibilidad frente a Pares-descriptores y frente a Pares-sham, sobre todos los pares fuera de diagonal. Fijar `Delta=Brier(compatibilidad)-Brier(control)`: negativo favorece compatibilidad. Reportar cada training seed, media/rango entre seeds y deltas emparejados por escena. Para cada comparación y seed, bootstrap pareado de escenas (2000 remuestras de 1024 escenas con reemplazo), media e intervalo percentil 2.5–97.5%. Usar `default_rng(SeedSequence([2026090731, control_index, seed]))`, índices de control 0=BCE, 1=sham, 2=transitividad. Para resumen conjunto, promediar primero los tres deltas por escena y repetir bootstrap con seed=0 en esa derivación. El intervalo conjunto se condiciona a estas tres training seeds y al corpus; no cuantifica variabilidad entre nuevos corpus de train. No tratar pares ni seeds como escenas independientes. Los intervalos son descriptivos, no ajustados por multiplicidad y no funcionan como GO/NO-GO.

Física-vs-transitividad es un **contraste obligatorio de atribución**, con el mismo reporte de deltas/intervalos en OOD beta; no se omite si desfavorece al candidato. Lectura conjunta: física-vs-BCE añade un tensor triple completo durante entrenamiento y una forma de objetivo; física-vs-sham aísla la alineación de los pesos en esa misma penalización; física-vs-transitividad compara dos restricciones, pero cambia forma/escala/gradientes. Si transitividad iguala o supera a física, un positivo frente a BCE/sham no sostiene superioridad específica de la familia frente a esa consistencia genérica. Las normas de gradiente acotan, no eliminan, ese límite de atribución.

Token-only, heurística y demás slices son secundarios. También conservar BCE, AP/AUC donde ambas clases existan, recall positivo y prevalencia; explicar unidades no evaluables. La predicción de una única partición por el reader no adjudica ambigüedad global.

Lector común: aglomerativo average-linkage sobre `1-p` simetrizado, diagonal cero, corte sin k verdadero. Elegir por brazo y seed sólo en validation el umbral entre {0.05,0.10,...,0.95} que maximiza ARI medio; empate al menor umbral. La **regla** y presupuesto son comunes, no necesariamente el valor. La referencia analítica usa la misma selección. Métricas secundarias: ARI, partición exacta, k inferido y error de k por escena. Conservar logits para repetir análisis sin forward. No elegir entre lectores en test.

Sanities: invariancia sintética de escala, equivariancia a permutaciones y relabeling de truth, independencia de amplitudes/sidecars, finitud y padding, auditoría de descriptor contra ternas conocidas y perturbadas, equivalencia de reducción batch/escena, sham marginal idéntica y mapeo equivarante. La baja incompatibilidad medida por el mismo residual que se penalizó es un diagnóstico interno, no prueba independiente de geometría aprendida. La evaluación externa a la loss son pertenencias y particiones OOD, también bajo ruido y familia deformada sintética.

## 7. Presupuesto y cierre de esta etapa

Antes de training: auditoría independiente de este plan, implementación con fixtures, auditoría técnica y preflight de 64 escenas de desarrollo. Añadir dos fixtures mecánicos separados, no draws de test: k=3 con ocho parciales por fuente (N=24) y k=4 con ocho parciales por fuente (N=32), f0=(110,173,281,419) Hz truncado al k requerido, beta=5e-4 y sigma=2 cents, seed 2026090732. Para deformación, verificar sólo finitud/no identidad con gamma=1e-5 sobre ese fixture; no seleccionar magnitud por rendimiento.

Conservar rango, cuantiles, varianza y finitud de R/w/support sobre desarrollo; estratificar triples de una fuente vs fuentes distintas sólo con truth de desarrollo, después de construir las features. Registrar correlación física-sham y proporción de pesos cambiados. En el mismo batch/init conservar BCE, penalizaciones y normas de gradiente de cada componente y del componente ponderado por lambda. No finitos, pesos geométricos constantes en todo desarrollo o gradiente exactamente cero de física/sham en todas las escenas evaluables obligan a revisar antes de entrenar; dominancia numérica (`norm(lambda*grad_penalty)>norm(grad_BCE)`) activa revisión explícita, no cambio automático de lambda. Transitividad puede ser inactiva cerca de p=0.5: su cero inicial se registra sin fingir un fallo, y un fixture de probabilidades (0.95,0.95,0.05) debe verificar penalización/gradiente no nulos. Son guardas mecánicas de informatividad, no umbrales científicos. Una modificación de fórmula/lambda requiere nuevo freeze y auditoría.

CPU sólo para pruebas y preparación proporcionadas. Limitar el preflight de generación/descriptores, incluidos fixtures máximos, a 120 s, 1 GiB RSS y un proceso; registrar costo por escena, máscaras/reader y estimación de campaña antes de generar todo. No convertir la estimación en una corrida CPU larga. La prueba neuronal CPU es sólo de gradientes/forma sobre un batch pequeño, máximo 30 s; no entrenar la campaña por CPU.

Campaña prevista: 15 trainings, máximo 24 horas de GPU acumuladas, objetivo inicial de un dispositivo y <8 GiB VRAM por job; estimaciones a verificar con un smoke GPU autorizado de ≤10 minutos, sin abrir test. Debe medir máximo backward de train (N=24, batch 128, cada pérdida) y máximo forward/evaluación (N=32, batch 128), usando repeticiones de fixtures; registrar tiempo/VRAM en ambas arquitecturas y proyectar los quince trainings más evaluación contra el presupuesto. Si no cabe, detener con estado incompleto y justificar revisión antes de continuar; no reducir batch/épocas post hoc y llamar completa a la receta. Informar objetivo/duración/VRAM por chat y Telegram antes de GPU local y esperar disponibilidad. Mendieta puede recibir jobs pertinentes autónomamente tras leer sus instrucciones y validar SBATCH; la cola larga no se reemplaza por CPU desproporcionada. No dar por disponibles dos A30 sin verificar el scheduler.

Conservar observaciones/sidecars, scripts y hashes, manifest de splits, config, seeds, versiones, estado de optimizador/scheduler/RNG, checkpoints de épocas 10/25/50 y siempre `last_epoch`, curvas de cada loss y componentes, logits float32 por escena, predicciones del lector, métricas crudas y runtime. Outputs nuevos separados de históricos; scratch bajo el proyecto. Test sólo tras congelar y hashear runner/features/losses/lectores y cerrar selección de validation. Ningún rerun modifica un artefacto cerrado.

El cierre requiere todos los brazos previstos y slices, auditoría independiente de resultados e interpretación y auditoría de alineación. Un efecto nulo o perjudicial bajo esta receta no refuta toda compatibilidad física ni toda lambda; un efecto favorable no demuestra una geometría universal. El sucesor se elegirá por lo observado: fortalecer la operación física si aporta, simplificar a descriptores si bastan, examinar el aparato si el ruido domina o revisar la hipótesis si la ley impuesta perjudica. Este documento no redefine el cierre del goal como mera preparación.
