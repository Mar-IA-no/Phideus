# Protocolo: evidencia generativa en un lector aprendido

2026-09-09. Candidato para auditoría independiente; no autoriza todavía
materializar tests ni describe una campaña ejecutada. Desarrolla el
[diseño inicial](PLAN_GENERATIVE_EVIDENCE_READER.md), SHA256
`25fd6bc61608aebe8195838bced48031a4c25b749dd599bb2df4dfe5c1a24fd7`.

## Pregunta, intervención y autoridad

Se compara una cabeza común con evidencia generativa alineada, ausente o
desacoplada. El factor es el acceso y correspondencia de un canal computado
por candidato; no se aísla una nueva operación neuronal ni una geometría
latente del backbone. Ley, soporte y ruido coinciden con el generador conocido.
No se valida HIT, identidad física en audio observado ni identificabilidad.

Primario: ARI en familia deformada, Generativa−Desacoplada (alineación bajo
los estratos declarados) y Generativa−Local (utilidad incremental). Un canal
ausente no tiene actividad/capacidad efectiva idéntica al alineado. Sham
preserva el marginal condicional especificado, no garantiza igualdad de
activaciones ni elimina todas las correlaciones entre evidencia y target.

## Datos y orden de acceso

Reutilizar las 4096 escenas train (seed2026090880) y 512 de calibración
(2026090881) del lector aprendido cerrado: q32, features observables, logits
de tres checkpoints, pools y sidecars preservados. Son desarrollo abierto.
Fijar rutas/SHA exactos en un manifiesto de reutilización antes de preparar
canales. No copiar checkpoints o rehacer forwards para obtener esos estados.

Checkpoints históricos: seeds2026090721/22/23, mismos pesos y recipe de
inferencia del experimento anterior. Entrenar tres cabezas por checkpoint:
Local (`local`), Generativa (`generative`) y Desacoplada (`decoupled`), con
inicializaciones2026090991/92/93: 27 entrenamientos.

| Test nuevo | Escenas e IDs | Semilla |
|---|---|---:|
| IID | 512, 0–511 | 2026090982 |
| Mayor beta | 512, 0–511 | 2026090983 |
| Mayor polifonía | 512, 0–511 | 2026090984 |
| Familia deformada | 512, 0–511 | 2026090985 |

Generar con la misma ley y orden de draws de
`learned_partition_data._draw_scene(split, scene_id, seed)`, sin mutar sus
constantes ni su código. La nueva frontera valida las semillas anteriores,
las cantidades y el freeze, no usa los wrappers ligados a los roles viejos.
La supervisión debe reconstruir ecuación, ruido, centrado, permutación y q32
sin una segunda llamada al generador. No añadir, sustituir o descartar draws.

Antes del primer draw de test se sellan protocolo/código/runtime, normalizadores,
27 entrenamientos completos, selección de épocas y roster de exclusión.
Deduplicar fingerprints del q32 ordenado contra train/cal, corpus histórico
enlazado por el release anterior, sus cuatro tests abiertos, rivales96 (alias,
no96 nuevas escenas), fixtures y los nuevos tests ya producidos. Resolver
alias sin contar copias como nuevas muestras. Conservar el draw ofensivo y
detenerse si hay coincidencia; ausencia de duplicados no prueba independencia
semántica. Ningún test nuevo se usa para seleccionar receta o arquitectura.

El productor escribe observación y sidecar en puertos separados. Preparación
observable e inferencia no reciben labels, fuentes, índices verdaderos, k real,
parámetros ni régimen verdadero. Metadata de split sólo identifica archivos y
RNG del sham, no entra a los tensores ni escoge familia. Sellar candidatos,
fits, inputs y todas las predicciones del test antes de abrir su supervisión.
Un verificador puede comprobar hash del sidecar sin analizar sus contenidos.

## Candidatos, priors y ajuste reutilizado

Aplicar `structured_source_reader.build_pool` a cada checkpoint con los
mismos q32 y soporte analítico observable. Para desarrollo reutilizar pools
verificados; para tests nuevos preservar los tres pools y logits. Usar
`observable_source_rivals.candidate_inventory`: unión, vecinos de un paso
desde candidatos admisibles y hasta64 vecinos nuevos por hash predeclarado.
La misma lista canónica ordenada alimenta los tres brazos y checkpoints.

La máscara común admite sólo2–4 grupos, cada uno4–8 miembros. Preservar también
el inventario excluido y sus motivos. No generar vecinos desde candidatos
inválidos, no completar usando verdad y no agregar fallback silencioso.
Si la lista admisible está vacía: `NO_OBSERVABLE_CANDIDATE`, outputs vacíos,
escena conservada en el denominador total. Train omite esa escena de la loss,
con roster observable explícito común a todas las celdas; no la reemplaza.

Reutilizar exactamente `GroupFitter(Grid(257,65,4))` y `fit_candidates` del
[protocolo de rivales](PROTOCOL_OBSERVABLE_SOURCE_RIVALS.md), SHA256
`905d67e42de3b39a5898ee7127073ceeaf87ccef7046020d11dbf18b5a051f3b`.
Núcleo SHA256 `e57b93f3894e14f1923e1111231041497b0d634678311ac1fb520bdc359b25f8`.
Conservar minimización por asignación, shortlist4, proyección conjunta de
fundamentales, carry del witness grueso, LB/UB y toleranciaJ1e-7. No reducir
grilla por resultados ni usar labels para escoger rama.

Orden fijo de ramas: `base-low`, `base-high`, `deformed-low`. k2/3 permite
las tres; k4 sólo base-low. Ley, rangos beta/gamma, f0[100,500], ruido2cents,
índices1–8 y cardinalidades son priors del sampler, no invariantes universales.
J es un residual cuadrático escalado sobre el cociente centrado; sus cotas
son de grilla, no continuo, posterior ni likelihood integrada de q32.
El ajuste se hace una vez por escena, no por checkpoint ni cabeza.

## Tensores comunes y canal de candidato

Ordenar eventos por q32 con sort estable; candidatos y grupos por firma.
Para logits de cada checkpoint conservar el mapeo canónico↔observado. La
equivariance a permutación se prueba en q32 sin empates; reportar empates y
desempates estables sin inventar identidad de eventos coincidentes.

Cada grupo recibe nueve features como Local anterior: m/8, m/N,
m(m−1)/(N(N−1)), media/std/min/max de logits intra-grupo, indicador m<3 y
compatibilidad local. Esta última es la media, sobre triples del grupo,
de r²/(r²+4), usando residuals observables preservados en cents. Remapear
triples a rango canónico antes de agregar. No recalcular el fitter antiguo
de fuentes compartidas para obtener un factor que ningún brazo necesita.

Seis globales anteriores: N/32, k/N, energía de pares
`−sum_intra logits/N`, suma de cuadrados de masas de grupo, masa singleton
y masa en grupos<3. Estas dos últimas y m<3 son cero por soporte, no evidencia
de que el modelo aprendió a evitar fragmentación.

Añadir once metadatos comunes por candidato: cinco proporciones de número
de grupos de tamaño4/5/6/7/8 (`count_m/k`), tres indicadores de rama disponible
y tres dimensiones nominales/N (2k−1 para base, 3k−1 para deformada; cero si
rama no disponible). Son información de soporte/flexibilidad, no outcomes.

Canal generativo: seis coordenadas, `(log1p(LB_b/N), log1p(UB_b/N))` para cada
rama en el orden fijo. Valores de ramas no disponibles se representan por
cero tras normalizar y están identificados por metadatos comunes. No entregar
elección de rama mínima, residual plantado, etiquetas, frecuencia absoluta ni
parámetros privilegiados. Guardar valores crudos64 y tensor entregado32.

Normalizadores únicos de train, sin usar calibración/test: cuatro momentos
de logits de grupo y energía global por checkpoint, con igual masa por escena
y luego por grupo/candidato; fórmula de media y varianza poblacional del
lector anterior, recalculada sobre el nuevo universo. Canal generativo común
a checkpoints: por coordenada, media de medias por escena entre candidatos
con rama disponible, y varianza con la misma ponderación. Escenas sin soporte
para esa coordenada no aportan; registrar conteos. Escala=sqrt(var), o1 si
varianza cero. Si una coordenada carece de soporte train, detenerse antes de
entrenar; no aprender su normalizador en test. No clipping ni winsorization.
Metadatos y compatibilidad local no se estandarizan.

Local entrega canal de seis ceros; Generativa el canal normalizado correcto.
Desacoplada recibe los vectores completos normalizados permutados juntos,
no coordenadas permutadas independientemente.

## Sham estratificado

Estrato dentro de una escena: tupla ordenada de tamaños de grupo. Fija N,k,
histograma, ramas disponibles y dimensiones bajo este soporte; por eso no
se necesita estratificar por la rama que gane el fit (un outcome del canal).
Dentro del estrato ordenar candidatos por firma. Con tamañoL>1, obtener
shift uniforme entero[1,L−1] con NumPy PCG64/SeedSequence
`[2026090995, split_seed, scene_id, *sizes]`; asignar donante `(i+shift)%L`.
Para L=1 conservar vector y marcar `NO_PERMUTATION`. No buscar shifts mejores
ni repetir la aleatorización por época/checkpoint/seed.

Guardar tamaños, IDs/donantes/shift y fracción de vectores float32 realmente
distintos. Si ninguna entrada cambia, marcar `INPUT_UNCHANGED`, no presentar
el sham como intervención efectiva. La lista/máscara/incidencia y todos los
inputs comunes permanecen idénticos. El sham conserva exactamente el multiset
conjunto de seis coordenadas dentro de cada estrato. Invariancia relacional
no equivale a equivalencia entre targets; no se permutan labels con el canal.

## Cabeza, loss y receta

Una topología para los tres brazos: grupo9→32→16 con ReLU; agregar embeddings
por masa m/N. Concatenar16 agregados+17 globales/metadatos+6canal=39 y aplicar
39→32 ReLU→2 softplus(beta1,threshold20). Bias en todas las lineales;2194
parámetros. No BatchNorm, dropout, atención nueva ni parámetros por brazo.
Float32, sin AMP/TF32; algoritmos Torch deterministas y un hilo CPU.

Inicialización por bloque10/20/30/40: semilla uint64 obtenida de
SeedSequence([reader_seed,block_id]); torch.Generator CPU local, uniformes
en ±1/sqrt(fan_in), primero weight row-major y luego bias. Misma inicialización
byte por byte para todos los brazos/checkpoints a reader_seed común; no
consumir RNG global en inicializaciones desechadas. Fixtures prueban shapes,
conteo, estados y predicciones idénticas si se entregan inputs idénticos.

Target supervisor: (H(P|Y),H(Y|P))/logN, calculado64 y entregado32; guardar
entropías crudas. Reusar `partition_errors` y `partition_cost_loss` sin
modificarlos. Loss media de escenas de media de candidatos de media de dos
errores cuadrados. No ponderar por número de candidatos, ni penalty adicional.
Decisión por suma32 de outputs, empate exacto por menor firma canónica.

AdamW lr1e-3, betas(.9,.999), eps1e-8, weight_decay1e-4, amsgrad/foreach/fused
False.50épocas completas; batch32 escenas, último batch parcial sin duplicar
ni rellenar escenas. Roster elegible train ordenado, permutado por
PCG64/SeedSequence([reader_seed,epoch]) con epoch0..49; mismo orden en27celdas
salvo las tres inicializaciones declaradas. Cada escena elegible aparece una
vez por época. Padding de grupos/candidatos es cero y queda fuera de loss,
agregación, selección y almacenamiento de outputs válidos.

Guardar inicial, last_epoch siempre, optimizer/RNG/posición de batch y
snapshots cada5épocas. Conservar loss, varianza inputs/targets/predicciones,
activaciones por bloque, norma de gradiente/update y columna generativa por
época, sin reseed/tuning por actividad. Reanudar sólo desde frontera de update
completo con binding idéntico y trayectoria fixture equivalente a continuo.

Calibrar en épocas5,10,…,50. Elegir una época por brazo que maximice ARI medio
en calibración: promedio9celdas por escena, luego escenas con salida común;
empate por época menor. No seleccionar checkpoint/seed ganador. Guardar todas
las predicciones por candidato/época/celda y publicar elección antes de tests.
Si calibración carece de escenas con salida, selección no definida y detener.

## Evaluación y referencias

Para cada test, preservar los27 outputs originales por candidato. Para las
nueve cabezas Generativa, conservar además intervenciones de inferencia
canal cero y canal desacoplado (sin reentrenar), con inputs/outputs y soporte.
No usar estas intervenciones para seleccionar modelo. Informar cambios de
costos, cambios de elección y desplazamientos constantes de la suma.

Evaluar trece métricas del lector anterior: ARI, exactitud, desacuerdo de pares,
k, error firmado/absoluto de k, masa de grupos<3, VI/VI normalizada y las
dos entropías crudas/normalizadas. La masa<3 es cero por prior para candidatos
admisibles; Histórico no comparte esa restricción. Guardar métricas de todos
los candidatos, no sólo el elegido, máximos de ARI/mínimos de VI privilegiados,
presencia plantada en pool/vecinos/ausente y cobertura de salida.

Referencias de sistema: mínimo UB de Base y Extendida con empate por firma,
sin penalización de complejidad añadida ni tuning; y lector Histórico del
backbone con thresholds preservados .55/.65/.60 para checkpoints21/22/23.
Base/Extendida se calculan una vez, Histórico por checkpoint; no inventar9
réplicas independientes. Informar Histórico en todas las escenas y también
sobre soporte común para contextualizar comparaciones, sin hacerlas causales.

Resumir por escenario. Para brazos, promediar métricas de las9celdas dentro
de escena; nunca promediar logits para crear un ensemble. Primario condicionado
a salida común en deformada. Preservar todas512escenas y sus estados. Con
M escenas elegibles, bootstrap2000×M índices PCG64/SeedSequence
([2026090994,split_seed]) con reemplazo, compartidos por contrastes/métricas.
Para cada control, delta_i es la métrica media de Generativa menos la del
control en esa misma escena. Estimador puntual: media aritmética de delta_i
en las M escenas; cada réplica es la media de los M deltas seleccionados
por sus índices, con multiplicidad. Para medias de un brazo se aplica la
misma operación a su métrica por escena. No se bootstrappean celdas como
unidades independientes ni se cambia el soporte entre brazos.
Intervalos percentiles97.5% nominales para los2contrastes ARI primarios
(Bonferroni2), endpoints[1.25,98.75];95% descriptivos para el resto,
endpoints[2.5,97.5]. Usar NumPy percentile con method='linear' sobre las2000
réplicas. M=0: estadístico/intervalo null,
no0; M=1 intervalo degenerado se informa como tal. Incertidumbre condicional
a checkpoints/seeds entrenadas, no intervalo poblacional de entrenamiento.

Informar por celda y soportes del sham; separar el conjunto sin cambio
efectivo del que sí cambia sin redefinir el primario. Describir tamaño de
efecto y límites, no declarar ganador por umbral inventado. No agregar cuatro
escenarios como victoria global ni usar test para escoger otra familia/loss.

## Recursos, almacenamiento y recuperación

Raíz nueva `data/atencion_armonica/generative_evidence_reader_v1/`; temporales
propios bajo `.agent-work/phideus-generative-evidence-20260909/`. Referenciar
inputs históricos inmutables, no duplicarlos. CPU ≤6GiB RSS, GPU ≤6GiB memoria
reservada por proceso; un operador pesado por vez. Límite de nuevos artefactos
60GiB, espacio libre mínimo80GiB antes de iniciar la materialización.

Cada escena conserva inventario, factores y fits completos en JSON gzip nivel3
con mtime0, hash del archivo y de bytes descomprimidos canónicos; lectura
streaming por escena, no todo el árbol en RAM. No reducir asignaciones ni
witnesses: la compresión preserva el objeto completo y `replay_fits` reconstruye
sin otro barrido. Entrenamiento lee sólo arrays compactos de inputs/targets,
no los árboles de factores. Configs/rosters/normalizadores quedan separados.

Antes de materializar: perfil ≤120s por backend, batch32 y envolvente de
82candidatos/328grupos (cota conservadora:18cortes admisibles+64vecinos,
4grupos/candidato), además de un batch de train observado. Comparar CPU/GPU
para fitting y cabeza separadamente, incluyendo preparación/transferencias,
forward/backward/optimizer y guardado/validación; congelar backend por etapa,
sin elección automática de CUDA. Mantener GPU si reduce costo total medido,
CPU si empata. Registrar discrepancias numéricas y decisiones; no exigir
trayectoria CPU/GPU idéntica ni alterar receta por velocidad.

Presupuesto operativo acumulado de intentos: preparación/descriptores/fits
6h,27trainings+calibración12h (por celda1800s incluyendo su calibración),
inferencia/evaluación/replay4h. El máximo por celda no es una reserva de1800s
para cada una: el tope acumulado puede actuar primero. Autorizar la corrida
sólo si el perfil proyecta completar el roster entero, calibración, I/O y
overhead dentro de12h; conservar estimación y método. Exceder cualquiera de
los límites individuales o acumulados
deja estado recuperable, no cambia muestras/epochs ni habilita substituciones.
El reloj de auditorías o lecturas bibliográficas no se finge cómputo de campaña.

Un manifiesto pequeño liga protocolo, código, runtime, checkpoints, datasets,
normalizadores, roster y stage. Estados nuevos se publican atómicamente sin
reescribir receipts visibles; recuperación registra padre y acumulación de
tiempo. Verificar hashes en fronteras y por archivo al consumirlo, no volver
a hashear todo el histórico por cada batch. No usar monkeypatch de constantes
ni encadenar los viejos gates de36celdas/64candidatos para simular el diseño.
Compartir funciones matemáticas y extender sólo las fronteras incompatibles.

RTX3090 local habilitada sin permisos por corrida; comprobar disponibilidad
y ownership antes de CUDA. No interrumpir procesos ajenos. SIGINT/SIGTERM
dejan checkpoint en frontera segura y salida recuperable; atender la petición
del usuario de liberar GPU. No Mendieta por rutina ni corrida CPU muy larga
para sustituir GPU eficiente. Telegram sólo si hay bloqueo indispensable.

## Verificación y cierre

Fixtures previos a campaña: permutación/gauge/soporte y empates; canal conjunto
y metadatos; sham condicional, estrato singleton y vectores idénticos; ninguna
supervisión en inputs; normalización train-only con máscara de ramas; identidad
topología/init; pérdida con batch parcial y padding; escena sin candidatos;
recuperación de training; compresión/replay exactos; prueba de rechazo de
test antes del freeze y de manifests/rosters incompatibles. Los fixtures
geométricos se registran como material abierto en el roster de exclusión.

Auditar protocolo antes de implementar, implementación/preflight antes de
campaña y selección antes de test. Tras los4tests: replay CPU desde outputs
y factores conservados, auditoría independiente técnica y del horizonte,
balance local/global, docs/wiki, commit/push. No cerrar en preflight ni con
entrenamientos preparados. Un resultado nulo/mixto cierra esta única campaña
sin tuning adaptativo de otros readers; la evidencia decide la siguiente
pregunta. Promoción arquitectónica y GO/NO-GO pertenecen al usuario.
