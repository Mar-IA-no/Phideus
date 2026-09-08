# Lector aprendido de particiones: protocolo prospectivo

Versión prospectiva formulada el 2026-09-08. Este documento no autoriza por
sí solo datos ni entrenamiento: exige las auditorías y recibos de ejecución
definidos abajo. Desarrolla la
[hoja inicial](PLAN_LEARNED_PARTITION_READER.md); no cambia el
[contraste cerrado](RESULTS_SOURCE_STRUCTURED_READER.md).

## Pregunta y frontera de atribución

¿Puede aprenderse a elegir mejores particiones dentro del pool existente,
y aporta el costo conjunto de fuente información adicional frente a los
controles de pares/estructura, compatibilidad local y costo desacoplado?
La unidad geométrica es una partición del conjunto observado, no una lista
de etiquetas con nombres privilegiados. Refinar y fusionar grupos son
operaciones diferentes; la supervisión conserva ambos errores.

Se aprende sólo un lector pequeño sobre tres redes descriptor-guided
congeladas. No se actualizan sus espacios latentes. Comparar los cuatro
lectores aprendidos a loss común aísla acceso incremental a un canal bajo
esta receta; comparar contra lectores fijos cambia cabeza y loss a la vez.
No se interpreta ese último contraste como atribución separada a uno de ambos.
El gap al oracle motiva el ensayo, no garantiza aprendibilidad. El gold es
la partición plantada, no la única interpretación física identificable.

## Observación, candidatos y costos conservados

Se conserva la ley frequency-only de `shared_partial_data.py`: fuentes con
f0 log-uniforme100–500Hz,4–8 índices elegidos sin reemplazo entre1–8;
frecuencias f0·n·sqrt(1+βn²+γn⁴), ruido gaussiano2cents en logfrecuencia,
centrado por escena, float32 y permutación. IID tiene2/3 fuentes equiprobables,
β log-uniforme1e-4–1e-3,γ=0. OOD beta cambia β a3e-3–1e-2; OOD polifonía
tiene4 fuentes; deformada añade γ log-uniforme5e-6–5e-5 y mantiene el β IID.
El orden de draws replica `_draw_scene` del lector estructurado con nuevos
roles, semillas y conteos; se implementa sin modificar su límite histórico256.

Entradas observables: q32 centrado y features de `feature_record`, sin
amplitud, tiempo, fase, fuentes plantadas o nombre del régimen. Los checkpoints
son `pairs_descriptors`, semillas2026090721/22/23 y estados last_epoch ya
pinneados en el freeze histórico. El forward mantiene el mismo runtime,
float32, ausencia de TF32, simetrización y orden de logits.

`structured_source_reader.score_scene` define el pool: unión de todos los
cortes average-linkage neuronal y analítico, firmas canónicas por rango q32,
deduplicación exacta y prior de tamaño máximo8. Los candidatos no reciben
etiquetas. Se conservan grillas1025/257, costos nominales r²/(r²+4), costo
local medio de triples y witness aproximado de grupo. Un grupo de tamaño<3
tiene costo cero por ausencia de restricción, no por coherencia demostrada.
El sham usa exactamente la rotación por tamaño anterior con semilla raíz
2026090785 más split_seed,scene_id y tamaño; conserva multiset de grupos
únicos, no necesariamente su distribución ponderada por ocurrencia en pools.
Un cache de fits se comparte entre los tres checkpoints de cada escena.

## Representación exacta

Para escena con N eventos y grupo G de m miembros, construir ocho entradas:

1. m/8.
2. m/N.
3. m(m−1)/(N(N−1)).
4–7. Media, desviación poblacional, mínimo y máximo de los logits z_ij,
   i<j dentro de G, calculados en float64. Los cuatro valen0 si m=1.
8. Indicador m<3.

Tres brazos añaden una novena entrada: costo de grupo `local_compatibility`,
`shared_source` o `decoupled_source`, respectivamente. Pares/estructura usa
sólo las ocho entradas comunes. Ningún brazo recibe ambos costos físicos.

La partición P recibe seis variables globales:
N/32, k(P)/N, E_pair(P)=−sum_{G∈P}sum_{i<j∈G}z_ij/N,
sum_{G∈P}(m/N)², masa en singleton y masa en grupos<3.
Los IDs de escena/checkpoint/candidato y el nombre del régimen son metadatos
de integridad, no inputs de la red. La pertenencia candidata no es truth.

Normalización por checkpoint, train exclusivo y común a los cuatro brazos:
los cuatro estadísticos de logits usan media y desviación poblacionales
de grupos únicos, peso1/(4096·n_grupos_de_escena); E_pair usa candidatos,
peso1/(4096·n_candidatos_de_escena). Varianza como media ponderada de
desviaciones cuadradas en float64. Escala cero exacta se reemplaza por1
y se registra su máscara. No normalizar las demás entradas; no clipping.
Conservar raw float64, normalizador y tensor final float32. Los singleton
se normalizan como los demás después de asignarles los cuatro ceros crudos.

## Arquitectura e inicialización

MLP de grupo: Linear9→32,ReLU,Linear32→16,ReLU para los tres brazos físicos;
Linear8→33,ReLU,Linear33→16,ReLU para Pares/estructura. Bias en cada Linear.
Por partición, sumar embeddings de grupos con pesos m/N; concatenar las
seis variables globales y aplicar Linear22→32,ReLU,Linear32→2,Softplus
(β=1,threshold=20). Sin BatchNorm, dropout, atención ni residual externo.
Los dos outputs no negativos predicen costos, no probabilidades.

Capacidad declarada:1650 parámetros físicos y1643 del baseline, todos
estructuralmente conectados y trainables; no garantiza actividad empírica.
La diferencia de7 parámetros(≈0,42%) permite evitar una columna fantasma;
no demuestra igualdad exacta de capacidad. Informar conteo, FLOPs y latencia.
Los tres brazos físicos tienen shape e inicialización idénticos por semilla.

Inicialización explícita en CPU float32: para cada reader_seed y block_id,
derivar torch_seed=int(SeedSequence([reader_seed,block_id]).generate_state(
1,dtype=uint64)[0]) y usar torch.Generator(device='cpu').manual_seed(torch_seed).
IDs10/20/30/40 para las cuatro Linear;11/21 para extensiones del baseline.
No sumar semilla e ID. Guardar tabla de18 semillas derivadas, exigir que no
colisionen y consumir pesos en orden row-major antes del bias por bloque.
Pesos/bias uniformes en ±1/sqrt(fan_in), con fan_in
común9,32,22,32. Los bloques físicos9→32 y32→16 se generan una vez;
baseline copia primera matriz[:,0:8],bias y segunda[:,0:32],bias.
La fila33 de su primera capa y columna33 de su segunda se generan con
IDs11/21 y las mismas escalas comunes. Cabeza final idéntica en los
cuatro brazos. Guardar estados iniciales y verificar subbloques exactos.
Las tres redes congeladas comparten inicialización de lector por seed;
no introducir la identidad del checkpoint como feature.

Verificar gradientes mecánicos en todos los bloques antes de datos. Durante
train registrar por celda/época varianza de inputs, fracción de activación
positiva por unidad ReLU y normas L2 de gradiente/update por bloque, incluyendo
columna física y fila/columna extra del baseline. Inputs/activaciones se
promedian sobre filas reales, sin padding; reportar denominadores. Normas
de gradiente y update se promedian sobre los128 updates de cada época.
Estos diagnósticos no habilitan reseed, poda ni tuning por actividad observada.

Tensores compactos: grupos únicos[U,8/9], globales[C,6], incidencia[C,U]
con valor m/N si el grupo pertenece al candidato y0 si no; targets[C,2].
Batch32 escenas con padding a U_max y C_max del batch. Máscaras de grupos
y candidatos explícitas, padding en0; invalidar cualquier fila sin grupos
o sin candidatos. Incidencia anula embeddings de grupos padded, máscaras
excluyen candidatos padded de loss, selección y predicciones guardadas.

## Target, loss y decisión

Con n_ab=|G_a∩Y_b|, n_a=|G_a|, n_b=|Y_b|:

- separación H(P|Y)=−sum_{n_ab>0}(n_ab/N)·log(n_ab/n_b);
- fusión H(Y|P)=−sum_{n_ab>0}(n_ab/N)·log(n_ab/n_a).

Usar log natural,0log0=0 y float64; guardar entropías crudas y targets
t=(H(P|Y),H(Y|P))/logN en float32 para entrenamiento. Sólo supervisión lee Y.
La suma cruda es VI; dividir por logN repondera tamaños y no hereda todas
las propiedades composicionales de VI entre universos distintos.
La fuente motivadora es Meilă2005, no una prueba de superioridad para Phideus:
https://icml.cc/Conferences/2005/proceedings/papers/073_ComparingClustering_Meila.pdf

Loss de batch: media de escenas de la media uniforme de candidatos de
la media de los dos errores cuadrados. No ponderar escenas por número de
candidatos, grupos o pares. No clipping de outputs/targets ni penalty de k.
Seleccionar mínimo de la suma de outputs por candidato; empate float32
exacto resuelto por menor firma canónica. Preservar ambos outputs, empate
y gap al siguiente nivel. No softmax o calibración de probabilidad física.

Fixtures obligatorios: target0 para P=Y, separación positiva/fusión0 para
refinamiento estricto y caso inverso para coarsening; VI y permutación de
labels; loss con pools de tamaños distintos; máscaras; replay de decisión.
Registrar varianza target/predicción, reparto split/merge y distribución de k
para detectar colapso. Un predictor constante tiene costo igual para todos
los candidatos: documentar su elección por tie, no atribuirle información.
No hay propiedad anti-fragmentación garantizada por estas ecuaciones.

## Diseño, datos y receta cerrada

Cruce completo4 brazos×3 checkpoints×3 inicializaciones=36 trainings.
Los brazos e inicializaciones comparten pool por escena/checkpoint. Nueve
predicciones de una escena no son nueve réplicas estadísticas poblacionales.

| Rol | Escenas | Semilla de datos |
|---|---:|---:|
| Train IID |4096|2026090880|
| Calibración IID |512|2026090881|
| Test IID |512|2026090882|
| Test beta |512|2026090883|
| Test polifonía |512|2026090884|
| Test deformada |512|2026090885|

Inicializaciones2026090891/92/93. Bootstrap2026090894. Generación por
SeedSequence([split_seed,scene_id]) e IDs0..count−1, sin RNG global.
No generar ningún test antes del freeze auditado. Train/calibración no
contienen regímenes OOD ni datos de campañas previas.

Roster histórico obligatorio: los ocho splits de `historical_observations()`
del gate anterior y los cinco roles del lector estructurado cerrado, todos
pinneados transitivamente. Añadir el [roster mecánico previo](learned_partition_prior_fixtures.json),
que fija siete archivos y cuatro fingerprints únicos; validar path/SHA,
keys y tipo de extracción (objeto observado o vector q32) y conjunto exacto.
Verificar ambos paths/SHA del alias, no sólo el hash declarado del origen.
El preflight64 es alias de development,
no64 muestras nuevas; fixtures repetidas conservan aliases sin aumentar n.
Añadir observaciones de fixtures usadas durante implementación/perfil a su
propio roster identificado antes de autorizar draws; no contar copias como
muestras. Guardar SHA de cada roster y fingerprints de q32 ordenado. Detectar
duplicados dentro de rol, entre roles nuevos y contra todo ese corpus conocido;
conservar draw fallido y detenerse, sin reemplazo. La ausencia de duplicados
no demuestra independencia semántica ni cubre datasets externos no inventariados.

AdamW lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4,amsgrad=False,
foreach=False,fused=False. Float32 sin AMP/TF32, torch deterministic algorithms,
un thread CPU,50 épocas,batch32 escenas, sin scheduler ni early stopping.
Shuffle por NumPy PCG64 SeedSequence([reader_seed,epoch]), epoch0..49;
exacto y común entre brazos y checkpoints,128 updates/epoch,6400 en total.
Guardar secuencias de batches, train loss por época, RNG/optimizer/model,
inicial,last_epoch y snapshots cada5 épocas. Entrenar siempre las36 celdas.

Calibrar en épocas{5,10,...,50}. Elegir un único epoch por brazo maximizando
ARI medio: nueve combinaciones dentro de escena, luego512 escenas. Empate
por menor epoch. No elegir seed/checkpoint ganador ni cambiar receta desde
calibración. Selección común de epoch, no selección heterogénea de9 modelos.
Guardar todas las predicciones de calibración por candidato/época/celda.

## Evaluación y límites

Primario: ΔARI Compartida menos cada uno de los tres controles aprendidos
en polifonía. Promediar nueve outputs de métricas dentro de escena, después
512 escenas; no promediar logits/costos para crear un ensemble distinto.
Bootstrap pareado2000 remuestras de512 escenas con reemplazo, índices comunes
a todos los contrastes y slices; intervalos percentiles98,333333% nominales
por Bonferroni para tres contrastes. Guardar matriz por checkpoint/lector
y signo por celda. La incertidumbre principal es condicional a las redes y
seeds entrenadas, no intervalo poblacional de entrenamiento ni equivalencia.

Secundarios descriptivos95%: todos los slices y métricas ARI, exactitud de
partición, desacuerdo de pares, k, k−k_true, |k−k_true|, masa sub-3,
VI cruda/normalizada y componentes split/merge. Comparar también Pares fijo,
Compartida fija γ=1 e histórico con thresholds0.55/0.65/0.6 por checkpoint,
sin reselección. Mostrar oracle máximoARI y mínimoVI del pool por separado;
no tratarlos como un único lector ni como inferencia deployable. Brier de
las redes congeladas es común y no mejora porque cambie la partición.

Soporte sham sobre inputs efectivos float32: por candidato comparar tuplas
(ocho features comunes normalizadas,costo,peso m/N) en sus grupos miembros,
primero alineadas por firma canónica y luego como multiset ordenado con
multiplicidad, sin firma como input. Guardar ambas comparaciones e incidencia;
permutar grupos indistinguibles para la cabeza no cuenta como información nueva.
`INPUT_CHANGED` exige multiset distinto; el caso contrario es `INPUT_UNCHANGED`.
Registrar fracción de candidatos/grupos afectados y strata por tamaño. La
energía fija sigue como diagnóstico histórico, no como criterio de este soporte.

Pasar además la misma cabeza Compartida seleccionada por su canal shared y
por el sham original, sin cambiar pesos, normalizador o globals. Guardar delta
de ambos outputs y de su suma por candidato, máximo/mínimo, constancia exacta
de la diferencia de sumas dentro del pool y decisión cambiada. Es dependencia
empírica, no prueba de semántica; informar magnitudes, no sólo flags de redondeo.
Denominadores por test:1536 escena×checkpoint para inputs y4608 casos para
dependencia entrenada; cobertura por escena total(3/3 o9/9), parcial o ausente.
No excluir ningún caso sin soporte ni llamarlo réplica independiente.

Diagnóstico post-hoc adicional para cada brazo físico: reemplazar
su canal por0 y por una rotación adicional de1 dentro de cada tamaño del
pool (identidad si sólo hay un grupo); mantener normalizador/cabeza intactos.
Guardar cambio de costo, fracción de decisiones cambiadas y métricas, sin
usarlo para selección ni afirmar que sensibilidad prueba semántica física.
Documentar delta k y fragmentación junto a cualquier ganancia ARI.

## Integridad, secuencia y recursos

Implementar módulos nuevos `learned_partition_*` y un CLI nuevo; reutilizar
funciones puras verificadas sin editar fuentes ni manifests congelados.
Plan, configuración cerrada, productor, dependencias científicas transitivas,
checkpoints/runtime, auditorías y perfiles forman el binding común por SHA.
Pruebas de rechazo deben cubrir archivo ausente/extra, cambio de bytes,
rol/count/orden equivocado, selección ajena, acceso prematuro a tests,
sidecar expuesto en inferencia, duplicado y marcador de fallo.

Secuencia obligatoria:

1. Auditoría independiente de este protocolo antes de implementación.
2. Implementación y fixtures mecánicos sin semillas de campaña; auditoría
   independiente de código y pruebas. Receipt integrado vincula SHAs reales.
3. Tres perfiles mecánicos separados y sellados, sin draws de campaña:
   geometría/pools/serialización CPU sin Torch(120s/<1GiB RSS), training
   Torch CPU(120s/<4GiB RSS) y training Torch3090 más forwards congelados
   (120s/<4GiB RSS/<2GiB VRAM). En ambos perfiles de training usar batch32,
   C=64,U=94,incidencia densa y máscaras; C/U son cotas de padding, no una
   promesa de64 particiones distintas en fixtures reales. El tensor mecánico
   sobredimensionado y las fixtures observables conservan identidad separada.
   Medir forward/backward/AdamW, checkpoint IO y evaluación para ambas formas
   de head, además de producción de pools. Estimar36×6400 updates, diez
   evaluaciones por celda, selección, tests y replay incluyendo validación de
   artefactos. Elegir dispositivo por menor proyección de tiempo total de
   training más validación; empate a favor de CPU. Congelar un único device
   y runtime para las36 celdas antes de datos, sin cambiar batch/modelo.
   Los forwards descriptor-guided permanecen en3090 como etapa distinta.
4. Autorizar train/calibración sólo si auditorías y perfiles pasan, conservando
   proyecciones y presupuesto. Shards fijos512 escenas (ocho train,uno por
   otro rol); el split agregado requiere el roster exacto y sin huecos.
   Preparación/score CPU≤1200s/<2GiB por shard, forward≤600s/<2GiB VRAM y
   <4GiB RSS por shard. Un límite no habilita cambiar cantidades/seeds.
5. Entrenamiento por celda≤600s/<4GiB RSS/<2GiB VRAM; tope total36 celdas6h
   es límite de seguridad, no estimación ni reserva. Avisar alcance/duración
   estimada/VRAM antes de CUDA, verificar concesión vigente y ausencia de
   procesos de cómputo ajenos; lock local evita solapamiento propio, no otorga
   autoridad sobre otros proyectos. No usar CPU prolongada para eludir GPU.
6. Freeze de selección vincula todas las36 celdas completas, init/snapshots/
   last_epoch, train/cal datos/features/logits/pools/targets/normalizadores,
   predicciones de10 épocas y reselección exacta. Auditoría independiente del
   freeze antes de `TEST_READY`; no basta un flag. Cada consumidor valida
   chain completa y fuente actual antes/después de ejecutar.
7. Tests en orden IID,beta,polifonía,deformada, con referencias obligatorias
   a todos los roles previos. Forward y scoring observables no parsean truth;
   trainer/targets y evaluación son puertos separados. Inferencia del lector
   debe terminar y sellar outputs antes de calcular métricas con sidecars.
8. Replay CPU desde predicciones guardadas, sin reforward/retraining, reconstruye
   elección/métricas/bootstrap byte exactos (excluye recursos/binding).
   Auditorías independientes de evidencia y horizonte, documentación/wiki,
   commit/push y siguiente goal desde resultados, incluso null o adverso.

Cada etapa exige request y referencias SHA exactos, inventario cerrado,
rechazo de overwrite, verificación de estado y conservación de fallos.
Supervisar tiempo/RSS y memoria GPU desde un proceso padre para no depender
sólo de checks posteriores; detener únicamente su hijo por handle propio.
No reejecutar una etapa completa sólo porque expiró el monitor.

Recuperación de training: cada intento vive en directorio nuevo e inmutable;
una continuación referencia por SHA el request, receipt terminal del supervisor
y último snapshot completo de su padre. El padre puede estar INCOMPLETE,
pero su proceso debe estar confirmado terminal antes de reanudar. El lector
especializado de resume acepta sólo snapshots autosellados verificables de
ese intento fallido, nunca convierte su bundle incompleto en COMPLETE.
Cada snapshot se escribe en temporario propio y se publica atómicamente
con manifest/hash sólo después de un update terminado; init cuenta como
snapshot0. Fija modelo/optimizer,next_epoch/next_batch,RNG,acumuladores de
diagnósticos, hash de batches,init,datos/features/pools/targets/normalizador,
código/runtime/device y todos los ancestros. Validar antes y después del resume.

Snapshots obligatorios cada5 épocas y al cierre; parada cooperativa puede
guardar otro después de un update completo. Interrupción dura vuelve al
último snapshot válido, sin rescatar un update parcial. La recuperación
debe reproducir en fixture la trayectoria y outputs del training continuo;
regenerar sólo el sufijo perdido, no resetear una celda por su resultado.
Conservar predicciones de calibración y snapshots de prefijos válidos; el
inventario final exige una sola cadena COMPLETE por celda, con las diez
épocas de evaluación. Intentos fallidos y tiempo consumido permanecen visibles:
los límites600s/celda y6h/campaña acumulan trabajo de todos los intentos,
no se reinician para eludir presupuesto. Reusar las demás celdas completas.

Conservar observaciones/sidecars, features completas, logits, pools, costos,
witnesses, tensores raw/normalizados, incidencia/máscaras, targets separados,
todos los checkpoints descritos, predicciones y costos por candidato, lecturas
por escena, índices bootstrap, configs, versiones, fallos y recursos. Estado
canónico bajo `data/atencion_armonica/learned_partition_reader_v1/`, no Git;
código y documentos científicos versionados. Estimar disco total antes de
autorizar datos; mantener margen libre superior a2×proyección y registrar
medias/máximos por fixture. Si presupuesto no alcanza, documentar impedimento
y reauditar una enmienda antes de continuar, sin redefinir éxito como preflight.

## Cierre y horizonte

El resultado exige36 trainings y cuatro tests frescos completos, replay y
auditorías; no exige que gane Compartida. Si sólo mejora la decisión aprendida,
atribuirlo a ese sistema y conservar el baseline más simple. Si no mejora,
acotar esta representación y receta, no declarar insuficiencia de frecuencias.
El diseño preserva renombrado de grupos y permutación de eventos vía
agregaciones canónicas; q32 empatados y ties de linkage se reportan como
excepciones de orden. El centrado representa invariancia de escala común
antes de medición cuantizada; verificarla con fixtures y tolerancia numérica,
sin prometer igualdad bit a bit al multiplicar frecuencias arbitrarias.
Tiempo/fase/medición/CQT y fuentes latentes siguen alternativas, set-valued
pausado. Promoción arquitectónica y GO/NO-GO corresponden al usuario.
