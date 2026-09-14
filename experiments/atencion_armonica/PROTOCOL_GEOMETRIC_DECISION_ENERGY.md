# Energía geométrica para la decisión — contrato experimental

2026-09-14. Versión de diseño para auditoría independiente previa a implementar.
Complementa [el plan](PLAN_GEOMETRIC_DECISION_ENERGY.md). No hay campaña nueva
ejecutada. La admisión requiere resolver los findings materiales, probar el
mecanismo y sellar código, datos reutilizados, perfiles y configuración.

## Pregunta y autoridad

¿Una ruta explícita de ajuste geométrico en la energía final conserva mejor
el mínimo que entregar exactamente esa información como entrada? ¿Depende
su efecto del objetivo de entrenamiento? La geometría es el ajuste conjunto
por ramas de ley del fitter observable existente, no una geometría nueva
aprendida por el backbone ni evidencia física independiente del generador.

El diagnóstico anterior es evidencia de diseño. Sus cuatro tests están
abiertos y no pueden convertirse en nuevos tests, ni elegir hiperparámetros.
La intervención incluye prior inicial y trayectoria de optimización; no
identifica por separado esos mecanismos. La referencia inicial permite
distinguir herencia de la regla clásica y mejora producida por entrenamiento.

## Universo, datos y fuentes reutilizadas

Se conservan las leyes, observación float32, proposer, unión de pools y
vecinos, filtros de soporte y ajuste discreto de
`PROTOCOL_GENERATIVE_EVIDENCE_READER.md`. C es el conjunto canónico completo
de particiones admisibles: hasta 82 candidatos y 328 grupos; no se inserta
la partición verdadera. Se conservan escenas sin candidatos como tales.

TRAIN: 4096 escenas, seed 2026090880. Calibración: 512, seed 2026090881.
Reutilizar los agregados raw y delivered de `generative_evidence_reader_v1`,
autenticados desde `open_prepared.json`, normalizadores e índices por shard.
No invocar materialización, refitting o backbone forward para esos splits.
La raíz completa y cada puerto consumido quedan fijados por SHA256 y tamaño
en el manifiesto nuevo. No modificar ningún módulo o artefacto congelado.

Backbones congelados: 2026090721, 2026090722, 2026090723. Inicializaciones
nuevas del lector: 2026091491, 2026091492, 2026091493. Cada brazo tiene las
nueve celdas del producto cartesiano; ninguna es elegida como ganadora.

Tests nuevos, 512 escenas por escenario, en este orden:

| Escenario | Seed |
|---|---:|
| iid | 2026091582 |
| ood_beta | 2026091483 |
| ood_polyphony | 2026091484 |
| deformed_family | 2026091485 |

Mantener exactamente el orden de sorteos y la ley del productor vigente de
512 escenas; no modificar el productor histórico limitado a 256 ni sus
constantes globales. El wrapper nuevo valida identidad, cardinalidad y seed.
Congelar exclusiones de TRAIN, calibración, todos los tests históricos y
fixtures conocidos, resolviendo aliases por huella de observación canónica
q32. Una colisión se preserva y detiene el draw, sin reemplazo ni seed nueva
silenciosa. Todas las predicciones observables se sellan antes de abrir
sidecars de test para métricas; el productor preserva sidecars pero no los
entrega al fitter, proposer, normalizador o red.

Revisión anterior a entrenamiento/freeze: una comprobación de diseño generó
por error iid/0 con la seed inicialmente propuesta 2026091482. Esa seed queda
retirada de la campaña; 2026091582 es su reemplazo explícito, sin selección
por resultados de modelos. La escena abierta queda como fixture excluido.
Su reconstrucción determinista se archiva con linaje y sin atribuirle un hash
original que no se conservó. No hubo otras escenas nuevas ni entrenamiento
en ese incidente; no se presenta la comprobación como evidencia prospectiva.

## Escalar geométrico y acceso igualado

En el raw observable, cada rama disponible b entrega
`a_b(P)=log1p(UB_b(P)/N)` float64. Definir `a(P)=min_b a_b(P)` directamente
sobre esos canales conservados; no reconstruir UB mediante expm1. Las ramas
ausentes se excluyen, nunca compiten con cero. Al menos una está disponible
por candidato; una violación aborta. Este mínimo incorpora priors y soporte
de las ramas: no es una likelihood marginal ni un score libre de cardinalidad.

La escala única es `s=sqrt(mean_escena(mean_candidato(a**2)))`, calculada en
float64 sobre todas las escenas TRAIN no vacías, en orden de scene_id y
candidato. Si s es exactamente cero, usar uno y registrar la degeneración.
No usar targets ni calibración. Verificar igualdad de a entre backbones y
contar cada escena una vez, no tres. Definir `z32=float32(a/s)`; esos bytes
son toda la precisión disponible tanto al input como a la ruta explícita.

El donor de cada candidato es el desplazamiento estratificado vigente de
`generative_evidence.decouple`: mismo scene_id, split_seed y tupla ordenada
de tamaños; seed base 2026090995. No cambia entre celdas. `d32=z32[donor]`.
Conservar donors, estratos singleton, fracción z modificada y vector de
cotas modificado. El control no elimina toda señal ni modifica el target.

Todos los brazos no Local reciben los mismos ocho canales float32:
seis cotas alineadas normalizadas con los normalizadores TRAIN existentes,
z32 y d32. La ruta desacoplada altera sólo el bypass, no esos ocho inputs.
Es un control distinto del viejo lector con seis features permutadas.
Local recibe ocho ceros; comparte los restantes descriptores, disponibilidad
y dimensión de ramas. No tiene igual información ni igual capacidad activa.

## Ocho brazos y cabeza común

| Ruta | Identificadores de los dos brazos | Bypass b32 |
|---|---|---|
| Inyección | injection_mse / injection_decision | 0 |
| Geométrica | geometric_mse / geometric_decision | z32 |
| Desacoplada | decoupled_mse / decoupled_decision | d32 |
| Local | local_mse / local_decision | 0 |

Todas las cabezas: grupo 9→32→16 con ReLU, agregación por incidencia m/N,
concatenación de 16 coordenadas agregadas + 17 globales + 8 de evidencia,
lineal 41→32 ReLU →2 lineal firmada. Total: 2258 parámetros. Sin softplus,
clipping, temperatura, normalización por batch ni parámetros de bypass.
Ponderaciones e inputs float32. Inicializar los primeros tres bloques con
la misma construcción CPU SeedSequence([reader_seed,10/20/30]) y uniformes
±1/sqrt(fan_in) del lector anterior; última lineal, peso y bias, en cero.
Misma inicialización exacta en los ocho brazos y tres backbones por seed.

La red produce `r32(P,1:2)`. Construir
`h64_j=float64(r32_j)+float64(b32)/2`, y `E64=h64_1+h64_2` en float64.
Elegir argmin E64; empates EXACTOS se resuelven por firma canónica mínima.
No usar tolerancia para elegir. Padding nunca entra en mínimos o promedios.

Con corrección cero, E es exactamente el bypass entregado. Guardar estado
inicial y sus decisiones. Inyección/Local parten constantes; Geométrica
parte de z32, Desacoplada de d32. Esto es una propiedad impuesta, no mérito
del entrenamiento. Conservar referencias no entrenadas de mínimo z32,
mínimo d32, mínimo UB raw Base y Extendida. Medir discrepancias raw/z32:
la transformación/entrega puede crear empates. No afirmar identidad con
la regla raw ni causalidad respecto de los lectores históricos softplus32.

## Loss, targets y aritmética

Los componentes supervisados entregados son u32, dos entropías condicionales
normalizadas en [0,1], preservadas en el corpus. Definir el target de decisión
`tD=sum_j float64(u32_j)`, no la suma redondeada float32. El target matemático
`tM=sum_j u64_j` se reconstruye por el operador VI vigente sólo al evaluar;
mantener separadas ambas aritméticas y sus conjuntos óptimos exactos.

MSE por escena: media sobre candidatos y dos componentes de `(h64-u32)**2`
con u32 convertido a float64. Loss del batch: media de escenas elegibles.

Decisión: `O=argmin_C tD`, `delta=tD-min_C tD`, y
`L=max_P(delta(P)+mean_Q_en_O(E(Q))-E(P))`. Loss del batch: media de escenas.
Evaluar todos los candidatos, sin negativos muestreados, margen adicional,
temperatura o etiqueta coóptima privilegiada. `torch.amax` distribuye el
subgradiente por los máximos empatados; O usa igualdad exacta de tD. Los
gradientes no pasan por targets, máscaras ni el conjunto O.

En aritmética real, para cualquier mínimo p de E, mean_O E≥E(p), luego
L≥delta(p), L≥0 y L=0 implica regret cero en ese universo. Probarlo con un
oracle NumPy independiente y cotas de tolerancia numérica predeclaradas.
No transferir garantías de optimización convexa a la red. Si C es singleton,
L=0. Si todos son coóptimos, L puede penalizar variaciones de E que no
afectan el regret: restricción adicional explícita. C vacío no tiene loss.
Las dos salidas bajo Decisión no son entropías identificadas/calibradas;
solamente su suma participa en el objetivo, con libertad de redistribución.

Misma receta no iguala escalas ni trayectorias de optimización entre losses.
No inferir capacidad óptima de cada familia a partir de este único schedule.
El antecedente de margen estructurado y la adaptación coóptima propia se
distinguen en el plan; no llamar energía física a E.

## Entrenamiento, selección y preservación

72 entrenamientos completos: 8 brazos × 3 backbones × 3 seeds. Cada uno,
50 épocas; batch 32 escenas, último parcial preservado, cada escena elegible
una vez por época. Orden PCG64/SeedSequence([reader_seed,epoch]), epoch 0–49,
misma lista ordenada de scene_ids en todos los brazos y backbones. AdamW
lr 0.001, betas (0.9,0.999), eps 1e-8, weight_decay 1e-4; amsgrad, foreach y
fused false. Sin scheduler, clipping, AMP, TF32 ni tuning por brazo.
Torch determinista, una hebra CPU; backend uniforme de cabeza fijado tras
perfil, nunca cambiado a mitad de una celda. No exigir trayectorias CPU/GPU
idénticas, sí recuperación exacta en el backend/runtime congelado.

Guardar inicial, last_epoch, estados de épocas 5,10,…,50, optimizador,
RNG/schedule/posición y checkpoints de recuperación en fronteras completas.
Guardar calibración completa h64/E64 en inicial y cada 5 épocas, sin elegir
por MSE ni por loss surrogate. Para cada brazo elegir UN epoch entre 5,…,50
por mínimo regret tD medio: primero nueve celdas por escena, luego escenas
elegibles. Empate exacto: epoch menor. No seleccionar epoch 0, semillas o
backbones; su referencia se informa por separado. Mismo selector en todos.

Conservar por época loss, MSE, regret, normas de gradiente/pesos/updates por
bloque, unidades ReLU inactivas y duración. Un NaN, falta de candidato,
preempción o agotamiento de presupuesto no se transforma en entrenamiento
completo ni se resuelve cambiando la receta. Reanudar desde snapshot íntegro
con costo acumulado, conservando fallo y trabajo descartado. Evaluar initial
y el epoch seleccionado en los cuatro tests, todas las celdas; preservar
componentes y energía de cada candidato, no sólo decisiones.

## Estimandos y evaluación

Primario: regret VI matemático tM en deformed_family, condicionado a C no
vacío y con cobertura reportada sobre las 512 escenas. Unidad de inferencia:
escena; nueve celdas son resultados condicionados, no nueve réplicas de datos.
Promediar nueve celdas dentro de escena y después escenas. No recortar por
brazo o por dificultad. Cuatro contrastes primarios, menor regret es mejor:

1. Geométrica–Inyección bajo MSE.
2. Geométrica–Inyección bajo Decisión.
3. Interacción: contraste 2 menos contraste 1.
4. Geométrica–Desacoplada bajo Decisión.

Bootstrap pareado por escena, 10000 remuestras PCG64 seed 2026091494;
conservar índices y distribuciones. IC percentil 98.75% para cada contraste
(familia de cuatro, ajuste Bonferroni nominal 95%). No equivalen a garantías
exactas de cobertura ni a umbral GO/NO-GO. Con cero escenas, NA, nunca cero.

Secundarios: ARI, regret tD, acierto en conjunto óptimo, tau-b, MSE componentes,
gap respecto a inicial y a clásicos, efectos de loss por ruta, correspondencia
bajo MSE y demás escenarios. Reportar cada celda, descomposición entre seeds
y backbones descriptiva, soporte, presencia de verdad en pool/vecinos/ausente,
empates, cambios tD/tM, estratos por tamaños/disponibilidad y fracción sham
efectivamente cambiada. No convertir estratos o intervenciones en los scores
en causalidad de una geometría latente. No comparar MSE de salidas Decisión
como si la loss identificara componentes. Secundarios sin promoción implícita.

## Pruebas geométricas y de mecanismo

Antes de campaña: fixtures NumPy/Torch con cero, uno y múltiples coóptimos,
padding adversarial, C vacío/singleton, todos coóptimos, escalas degeneradas,
ramas ausentes, donantes singleton, cambios de orden y precisión. Verificar
cota L≥regret con tolerancia absoluta 1e-12 para arrays float64 acotados
|E|≤10 y targets [0,2]; no usar esa tolerancia para fabricar empates.

En todos los readouts nuevos comprobar identidad initial/bypass, scores
conservados y replay de decisiones. Para las primeras cuatro escenas
elegibles de cada test (por ID, no resultados), ejecutar dos diagnósticos
observables en initial y selected de las nueve celdas por brazo:

- Transporte de representación: invertir el orden de candidatos y grupos,
  permutar incidencia, ocho canales y bypass consistentemente. Deshacer
  índices para comparar outputs. Candidato debe ser equivarante; cambios
  por reducción float32 de grupos se reportan con max error, elecciones y
  margen de decisión, usando atol=1e-6, rtol=1e-5 sólo como check numérico.
  El selector sigue firmas, no el primer índice del array transportado.
- Estabilidad de cuantización y recentrado: desde q32 entregado, calcular
  `qshift32=float32(float64(q32)+log(2))`, luego
  `v64=float64(qshift32)` y `qprobe32=float32(v64-mean64(v64))`, en ese orden.
  Sólo qprobe32 entra a features, forward congelado, pool, fitting y cabezas,
  con idénticas guardas; no transportar el pool fingiendo otra pipeline.
  Conservar además `qcenter32=float32(float64(q32)-mean64(float64(q32)))`
  como diagnóstico de coordenadas, sin una tercera pipeline. Registrar
  diferencias relativas antes/después de cada conversión, outputs originales
  y transformados, cambios de soporte, candidatos, canales, energía y
  partición. Las diferencias de outputs son efecto CONJUNTO del roundtrip,
  no una atribución separada de centrado y cuantización.

El productor ya elimina la escala global antes de entregar q32. Por ello el
segundo diagnóstico no genera otra observación física ni prueba sustantiva
de invariancia aprendida a la escala: interroga estabilidad numérica de una
representación de razones. La pérdida de correspondencia candidato–score
se interroga por el bypass desacoplado, no por este roundtrip. Conservar
linaje observable de transformación; el sidecar original no reconstruye
qprobe32 y no se valida como si lo hiciera. Reutilizar labels por identidad
de evento sólo tras sellar predicciones. Registrar colisiones q32; si
aparecen empates, no inventar correspondencia unívoca de rangos. No modificar
protocolo por los resultados de estos diagnósticos.

## Recursos, sellos y cierre

Una GPU local RTX3090, sin remoto rutinario. Informar antes de ocuparla;
verificar ownership sin pedir otra autorización por corrida. Perfilar
cabeza y fitter por separado en fixtures declarados y primer batch TRAIN,
hasta 120 s por combinación CPU/GPU, incluyendo collation/transfers,
backward, validación y amortización de serialización. Máximo perfil total
600 s, RSS 8 GiB, VRAM 8 GiB; una hebra CPU para cabeza, sin paralelismo de
fitting no medido. Elegir menor costo proyectado medido por etapa; repetir
sólo si el perfil fue inválido técnicamente, con evidencia y costo retenido.

Topes operativos acumulados: perfil 600 s; adaptación OPEN 1800 s;
entrenamiento 14400 s; preparación/forward/fits de tests y probes 21600 s;
evaluación/replay 7200 s; reserva de auditoría ejecutable 3600 s. Total
49200 s de operadores (13 h 40 min), no duración conversacional del goal.
RSS 8 GiB, VRAM 8 GiB, máximo nuevos artefactos 100 GiB y espacio libre
residual mínimo 30 GiB. Son límites de recursos, no umbrales científicos.
Admitir etapas cuando proyección de perfil más 25% de reserva cabe en su
tope. Si no cabe, documentar revisión de recursos antes de iniciarlas;
no reducir muestras, épocas, controles o auditoría por silencio.

Manifestar protocolo, roster, código importado, runtime, fuentes, escala,
normalizadores, backend y presupuesto antes de entrenar. Sellar selección,
exclusiones y roster de outputs antes del primer test. Outputs raw permiten
replay sin entrenar, forward o fitting. Módulos nuevos y temporales locales
al proyecto; los artefactos anteriores permanecen inmutables.

Auditoría independiente del contrato, del mecanismo implementado antes de
campaña, y técnica/de alineación al cierre. Cierre exige los 72 entrenamientos,
2048 tests, probes declarados, replay, informe, documentación/wiki y commit/
push. Una dificultad operativa mantiene incompleto el goal. El resultado
puede ser negativo o mixto; promoción y GO/NO-GO son decisiones de Mariano.
El siguiente goal se justifica por lo observado, no por preservar esta cabeza.
