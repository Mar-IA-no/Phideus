# Investigación-acción: geometría, arquitectura y aprendizaje

Estado: plan de ejecución por etapas, autorizado en su orientación el 2026-09-07; auditoría y reauditoría independientes completadas, con los hallazgos resueltos para implementar §4 por CPU. Sólo ese diagnóstico está especificado para implementación. No habilita training neural hasta completar y auditar el protocolo de §5. La GPU local requiere confirmación de disponibilidad; el despacho remoto pertinente está autorizado, sujeto a protocolo y recursos verificados.

## 1. Objetivo y cierre

Formular, implementar y contrastar una hipótesis de geometría armónica computable: identificar relaciones y transformaciones observables de una familia de fuentes, traducirlas conjuntamente a representación, operación y función de pérdida, y medir su aporte frente a explicaciones y métodos alternativos. El banco inicial es Atención Armónica, pero su arquitectura histórica no tiene privilegio de selección. Audio detectado es una posibilidad de observación, no la definición del objetivo.

El goal completo termina con un contraste experimental ejecutado y auditable, artefactos reutilizables, alcance explícito de la inferencia y formulación del siguiente goal a partir de los resultados. Diseñar el protocolo o preparar una corrida GPU no completa este objetivo. Un impedimento de observabilidad o recursos se conserva como impedimento; no se redefine el cierre para convertirlo en éxito. La promoción arquitectónica y el GO/NO-GO pertenecen al usuario.

Cada etapa tiene una pregunta y una salida finitas. Se admite trabajo de varios días, no una sucesión ilimitada de prerequisitos. No se exige un efecto positivo. El presupuesto de cada campaña se fija antes de su ejecución y no se amplía para obtener el resultado deseado.

## 2. Estado real y precedentes

El código actual ya centra `log(freq)` por mezcla (`src/atencion_armonica/peak_tokens.py`, `compute_tokens`). Las features de par también dependen de razones/diferencias logarítmicas. La invariancia a escala global de frecuencia en esa representación no es una operación nueva.

`Pairformer` mantiene estados de par, comunicación token/par y mezcla local o triangular. La pérdida vigente es BCE binaria en pares válidos (`experiments/atencion_armonica/1_train_grouping.py`, `masked_bce`), reducida por número de pares del batch. Ese entrenamiento no certifica por sí solo que la información explotada sea específicamente armónica. El código histórico de semillas llama a `torch.cuda.manual_seed_all` sin distinguir dispositivo; no se usará como runner CPU ni sin la disponibilidad GPU confirmada que exige la política vigente.

El generador vigente normaliza a energía unidad los parciales supervivientes **de cada fuente por separado** (`harmonic_synth.py`, `_source_partials`). Sus amplitudes observadas podrían permitir inferir particiones y cardinalidad sin frecuencias. Es una hipótesis de atajo global que debe comprobarse antes de diseñar una nueva loss de consistencia. Las auditorías locales per-par del pool no resuelven automáticamente esa pregunta global. No se afirma aún que los modelos históricos hayan explotado el atajo.

La campaña proporcional proporciona controles de autoridad, observabilidad y atribución; no se importa su stack de políticas como requisito del nuevo núcleo. El paquete físico set-valued anterior queda pausado e incompleto, sin borrar sus resultados ni certificar sus findings pendientes.

## 3. Método y auditoría de rumbo

La wiki orienta la recuperación; las decisiones se contrastan con fuentes primarias y artefactos actuales. Cada lectura distingue afirmaciones, modelo relacional subyacente, supuestos/exclusiones e interpretación propia. Se examinan también HIT y este plan. Ante un vacío sustantivo se realiza búsqueda dirigida, incluida evidencia adversa; no se abre otra campaña bibliográfica general por inercia.

En cada cierre de etapa material y antes de elegir un sucesor se pregunta: ¿qué geometría del fenómeno se está contrastando?, ¿qué explicación alternativa distingue este trabajo?, ¿el experimento conserva esa pregunta?, ¿el siguiente paso aporta evidencia o sólo perfecciona un auxiliar? La auditoría puede exigir cambiar de dirección. Las correcciones menores se resuelven directamente; los cambios de validez/arquitectura/protocolo se someten a una instancia independiente.

No confundir representación tensorial con probabilidad, consistencia con energía física, simetría con ley constitutiva, ni menor loss con geometría identificada. Una propiedad por construcción es un sanity; su utilidad necesita un contraste externo. Una regularización geométrica no sustituye las distinciones que requiere la tarea ni garantiza evitar colapso.

## 4. Diagnóstico inicial CPU: partición por energía sin frecuencias

### Pregunta

¿La normalización por fuente del pool histórico permite recuperar grupos usando sólo amplitudes, sin frecuencias, índices armónicos, IDs de fuente ni k verdadero?

### Población y acceso

Usar el pool histórico abierto `data/atencion_armonica/final_pool/mixtures.jsonl` y su `pool_meta.json`; no generar ni abrir una evaluación prospectiva. Seleccionar determinísticamente los primeros cuatro IDs de cada una de las seis celdas declaradas en metadata: 24 mezclas. La elección se fija antes de observar resultados del diagnóstico; no representa una muestra aleatoria del corpus. Conservar registros seleccionados y hashes. Recorrer el JSONL en streaming, conservando sólo las líneas seleccionadas; verificar IDs, unicidad, celda y conteo. Hashear el archivo completo durante esa lectura y conservar hashes de metadata y código. No usar `grouping_dataset.load_pool`, que materializa el corpus e importa PyTorch.

El solver recibe exclusivamente un vector desordenado de amplitudes positivas. Antes de cada predicción, permutar picos con `numpy.random.default_rng(numpy.random.SeedSequence([seed, mixture_id]))`; seeds congeladas `2026090701` y `2026090702`, reutilizadas en ambas vistas y precisiones. Invertir la permutación sólo para evaluar. No recibe el orden original, polyphony, regime, source_id, harmonic, freqs ni máscara de colisión. Los metadatos se usan fuera del solver para selección y evaluación, no para inferencia. El prior público declarado es que cada fuente contiene entre 4 y 8 parciales y tiene energía 1 bajo este generador.

### Algoritmo y autoridad

Buscar subconjuntos de energía aproximadamente 1 mediante meet-in-the-middle: dividir N<=24 en mitades, enumerar como máximo 2^12 subconjuntos por mitad y unir sumas mediante búsqueda ordenada. Buscar particiones completas disjuntas en los subconjuntos candidatos. Deducir k únicamente de la energía total observada y devolver, según corresponda, solución única, múltiples soluciones, ninguna o límite computacional. No elegir una solución por comparación con la verdad; toda ambigüedad se reporta.

La energía es `e_i=a_i²`, `E(S)=sum(e_i, i∈S)`, con suma final estable `math.fsum`; no es suma de amplitudes. Sólo son candidatos los subconjuntos de cardinalidad 4..8 que satisfacen `abs(E(S)-1)<=tol`. El rango MITM puede ampliarse conservadoramente por redondeo; la suma final decide admisión. Inferir `K_E={k entero: ceil(N/8)<=k<=floor(N/4), abs(E_total-k)<=k*tol}`. Si está vacío, devolver `PRIOR_VIOLATION`; si hubiera varios, devolver `AMBIGUOUS_K` sin elegir por truth; buscar sólo con un k admisible. La longitud N es observable, no k verdadero.

Cada partición es una colección no ordenada de bloques: índices ordenados dentro de cada bloque y bloques ordenados sólo para serializar. Ramificar por el menor índice no cubierto evita contar relabelings como soluciones distintas. Dos soluciones difieren sólo si inducen matrices de pertenencia diferentes. Deduplicar candidatos; `candidate_count` cuenta subconjuntos únicos admitidos y declara si es exhaustivo. Cada visita a un estado exact-cover cuenta un nodo. `UNIQUE` y `NO_PARTITION` requieren agotamiento; si la enumeración o búsqueda queda truncada, devolver `LIMIT_CANDIDATES` o `LIMIT_NODES`. Dos particiones válidas no equivalentes bastan para `MULTIPLE`, nunca para describir el conjunto total de soluciones.

Dos entradas predeclaradas: amplitudes float64 del pool con tolerancia absoluta de energía 1e-10; y amplitudes reconstruidas desde el canal log-amp float32 que reciben las redes, con tolerancia 1e-6. La segunda es el contraste primario de accesibilidad desde el input neuronal. Las tolerancias son numéricas de búsqueda, no márgenes de éxito científico. No ajustarlas tras ver los resultados. Comprobar que los subconjuntos y la partición elegida satisfacen sus tolerancias; no basta que la energía total sea cercana a un entero.

Reconstrucción exacta de ese canal: `log_amp32=float32(log(max(a_float64,1e-12)))`, `a_accessible=exp(float64(log_amp32))`. El solver recibe únicamente `a_accessible` en el contraste primario, no también la amplitud original. Comparar la fórmula contra `compute_tokens` (módulo NumPy puro); no usar el dataset ni el trainer PyTorch.

Limitar cada instancia a 20.000 candidatos, 50.000 nodos de búsqueda y dos soluciones distintas (suficiente para declarar ambigüedad). Si se alcanza un límite antes de establecer unicidad, devolver límite, no solución única. Evaluar también una permutación adicional como control de equivariancia de la salida.

### Intervención diagnóstica

Crear para las mismas mezclas una vista con ganancias positivas distintas por fuente, factores fijos 0.70, 1.10 y 1.40 para IDs 0, 1 y 2, sin renormalizar. Construirla con etiquetas sólo en el productor de la intervención; el solver sigue recibiendo únicamente amplitudes. Frecuencias y pertenencia no cambian. Esta vista comprueba dependencia del prior de energía, no es un benchmark matched de igual dificultad ni una prueba de especificidad armónica. Conservar los factores y el cambio de energía total. No refinar factores tras observar resultados.

Los totales ideales intervenidos son 0.49, 1.70 y 3.66 según el número de fuentes: puede haber rechazo en el chequeo escalar del prior, antes de exact-cover. Ese rechazo no aísla una contribución de la búsqueda de particiones.

### Evaluación y artefactos

Por mezcla/vista/precisión: IDs seleccionados, permutaciones, amplitudes entregadas, estados del solver, soluciones/candidatos o sus representaciones compactas, cardinalidad inferida, unicidad, cobertura de resolución y coincidencia exacta de la matriz de pertenencia con truth. Separar poly1 trivial de poly2/poly3. No promediar sólo sobre casos resueltos sin reportar el denominador. Una solución única errónea cuenta como error. La evaluación de partición completa en parciales ideales no se presenta como evaluación perceptual de colisiones.

Pruebas unitarias: solución conocida con entradas permutadas; amplitudes no válidas; ausencia de partición; múltiples particiones; límites que no certifican unicidad; recuperación desde float32; evaluación invariante al relabeling; intervención de ganancias. Primaria y replay deterministas, excluyendo tiempos. No cargar checkpoints ni importar PyTorch: NumPy/stdlib bastan.

En `UNIQUE`, registrar `unique_exact_match`; en `MULTIPLE`, coincidencia por witness y `truth_in_returned_witnesses`, sin elegir un witness con truth ni interpretar su ausencia como exclusión del conjunto completo. Reportar cobertura única y cobertura decidida, además de estados y denominadores de todas las instancias. El control de permutación compara estado y k; en `UNIQUE` también la partición canónica tras invertir índices; en `MULTIPLE` exige dos witnesses válidos distintos en cada corrida, no el mismo par truncado. `LIMIT_*` queda no evaluable para equivariancia. Tiempos, orden de candidatos y nodos no son objetos geométricos. El contrato de firma del solver se prueba sin campos privilegiados.

Presupuesto de campaña: timeout externo 180 s total para primaria+replay secuenciales, un proceso CPU experimental a la vez, estimación de RSS <256 MiB para N<=24 y outputs <10 MiB por corrida. Registrar tiempo/RSS/tamaño observado; si se excede un límite, interrumpir o reportar ejecución incompleta sin escalar muestra. Scratch en `.agent-work/phideus-geometric-rebase-20260907/`; código versionado, resultados canónicos nuevos en `data/atencion_armonica/geometry_energy_audit_v1/` y `data/atencion_armonica/geometry_energy_audit_v1_replay/`, sin sobrescribir históricos. Comparar resultados y hashes científicos; excluir sólo runtime/RSS/ruta de salida. Informes interpretativos/crudos en Biblioteca.

### Consecuencias predeclaradas

Si hay recuperación por amplitudes, se habrá demostrado un canal no frecuencial para esos casos. No demuestra uso por las redes ni invalida sus métricas históricas. Antes de atribuir una futura ventaja a armonía, habrá que eliminar/controlar ese canal y comprobar solvabilidad. Si no recupera, el resultado no prueba ausencia de otros atajos: se examina si el límite es numérico/computacional o de identificación. No abrir automáticamente una búsqueda de algoritmos cada vez más complejos.

## 5. Protocolo neuronal posterior: decisión obligatoria, no implementación libre

Tras §4, elegir **una** hipótesis entre no más de dos candidatas documentadas. Comparar restricciones de pertenencia global con compatibilidad de una familia física de parciales; no confundirlas. La invariancia de escala ya implementada y el cierre algebraico gratis quedan como controles, no candidatas novedosas. Antes de elegir, recuperar desde la wiki las fuentes relevantes y realizar relecturas dirigidas cuando los supuestos importen.

La salida de diseño debe especificar objeto, observación, transformaciones, incógnitas identificables, forward, target, pérdida y solución degenerada a evitar. Justificar qué se impone por arquitectura y qué por loss. Fijar un contraste primario de arquitectura con objetivo común o de objetivo con arquitectura común; si varían ambos, declarar el sistema combinado y no atribuir por separado. Incluir baseline clásico y descriptor-guided cuando respondan a la misma query, no por obligación de fusionar tareas incompatibles.

Congelar antes de ejecutar: productor de datos y correspondencia de picos, splits por fuente/escena independiente, seeds, presupuesto de entrenamiento, regla de lectura común sin k verdadero, métrica primaria y slices OOD, incertidumbre, controles de capacidad/cómputo, tratamiento de ambiguos y fallos, checkpoints y estados crudos. No convertir la selección de hiperparámetros en búsqueda sobre test. La auditoría independiente de este protocolo precede la implementación neural significativa.

Un salto a audio renderizado/detectado debe distinguir fuentes generativas de etiquetas observables de picos, y separar cambio de aparato de cambio de ley. Si §4 muestra un atajo, la observación nueva no se usa como supuesto remedio sin comprobarlo. El histórico ideal queda referencia diagnóstica; no se repite toda Fase 0 para presentar novedad.

## 6. Ejecución, cierre y sucesor

Implementar y verificar la candidata seleccionada según el protocolo auditado. CPU sólo si el trabajo es proporcionado. Cuando GPU sea necesaria o materialmente más eficiente, avisar por chat y Telegram con objetivo, duración y VRAM estimados, conservando la evidencia administrativa exigida para la escalación; esperar confirmación explícita de disponibilidad antes de usar la GPU local. El despacho remoto a Mendieta está autorizado para experimentos pertinentes que toleren cola larga: recuperar las instrucciones operativas, verificar límites y justificar el job antes de enviarlo. La posibilidad informada de hasta dos A30 no equivale a disponibilidad comprobada. No ejecutar el training histórico CPU que inicializa CUDA.

Cerrar con resultados de todas las condiciones previstas, interpretación independiente y auditoría de rumbo; propagar sólo el estado sustentado a wiki/docs y hacer commit/push de código y documentación. Los crudos y artefactos históricos se preservan. El sucesor depende de los resultados: desarrollar una operación con aporte atribuible, simplificar si basta un mecanismo menor, investigar observabilidad si ése es el cuello, o reformular la hipótesis si la evidencia la contradice. No abrir automáticamente router, otra policy ni otro corpus general.

El registro anterior ya no ocupa la interfaz de goals. El paquete técnico set-valued permanece pausado e incompleto: ese cambio operativo no subsana sus findings ni autoriza fingir un cierre técnico.
