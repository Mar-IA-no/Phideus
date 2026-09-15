# Protocolo — operador geométrico bajo medición

Estado: diseño corregido tras revisión independiente; no freeze ni campaña iniciada.

## 1. Fenómeno, roster y puertos

Misma ley de fuentes que `_draw_scene` en `learned_partition_data.py`: IID,
beta alta, polifonía4 y familia deformada. No cambiar beta/gamma/índices/f0.
Cada escenario usa IDs0..127 en test; desarrollo0..7 y calibración0..15.
Semillas por orden iid/ood_beta/ood_polyphony/deformed_family:

| Rol | Semillas | Total de escenas |
|---|---|---:|
| Desarrollo |2026091501,2026091502,2026091503,2026091504|32|
| Calibración detector |2026091511,2026091512,2026091513,2026091514|64|
| Test |2026091531,2026091532,2026091533,2026091534|512|

Antes del freeze, verificar semillas y fijar el inventario autenticado de
fingerprints anteriores como exclusión. Las observaciones prospectivas no
existen aún: cotejar sus fingerprints al generarlas, antes de inferencia, contra
ese inventario y las nuevas ya publicadas. Una colisión detiene la etapa y se
documenta; no reemplazar escenas ni ajustar exclusiones después de observarlas.
Runtime/config, source hashes y exclusiones se fijan antes de generar tests.

Los logaritmos ideales más la perturbación2cents del sampler definen las
frecuencias efectivamente emitidas. En este banco esa perturbación es detuning
de emisión; no se vuelve a añadir como ruido de extracción. La lista canónica
es exactamente la observaciónq32 original, con correspondencia por evento.
Se conserva ideal/detuning/centro sólo en el puerto de emisión/evaluación.

Render no recibe etiquetas de pertenencia: frecuencia absoluta por evento,
amplitud y fase bastan. D_i independientes uniformes en[-12,0]dB,
a_i=10^(D_i/20); fases uniformes[0,2pi). RNG separado:
PCG64(SeedSequence([2026091540,split_seed,scene_id])). Orden: amplitudes,
fases, vector gaussiano de24000muestras. Nada se normaliza por fuente.

## 2. Audio y detector

Mezcla mono aditiva de sinusoides a24000Hz. Evaluar sinusoides/suma enfloat64,
guardar señal limpia y tres mediciones float32; sin clipping ni PCM entero.
Amplitudes escaladas por ganancia común sqrt(.02/sum(a²)). Waveform:
x[n]=sum_i a_i sin(2pi f_i n/fs+phi_i), n=0..N−1, fs=24000Hz.
Toda frecuencia emitida debe ser positiva/finita. Se muestrea la sinusoidal
sin recortar frecuencias a Nyquist; emisiones≥12000Hz se marcan ALIASED_EMISSION,
sin redraw ni eliminación. Su alias no recibe la identidad de la frecuencia
original: el matching conserva la frecuencia emitida absoluta y contabiliza
faltantes/espurios. Emisiones f<50 o f>10000Hz reciben además el flag
OUTSIDE_DETECTION_BAND, independiente del alias. Incluir fixtures de ambos límites.

| Entrada | Duración | SNR | Intervención |
|---|---:|---:|---|
| Canónica | — | — | q32 original con identidad por evento |
| Audio nominal |1s|40dB| extracción desde waveform |
| Audio corto |.125s|40dB| duración frente a nominal |
| Audio ruidoso |1s|20dB| ruido frente a nominal |

Para cada duración, usar prefijo de señal limpia y del mismo vector de ruido;
escalar ruido a RMS(clean)/RMS(noise)·10^(−SNR/20). Guardar escala y SNR
realizados; corto cambia duración y su normalización necesaria para igual SNR,
no una réplica independiente. Audio no pasa amplitud ni fase al lector.
RMS no finito es fallo operativo. Si RMS(clean)=0, conservar señal nula,
ruido escalado0, SNR=null y ZERO_CLEAN_ENERGY; no redraw ni exclusión.
Si RMS(clean)>0 y RMS(noise)=0, fallar explícitamente sin redibujar ruido.
La SNR realizada se calcula sobre clean/ruido float64 antes del cast de salida;
guardar también RMS de la diferencia entre waveform float32 y clean float64.

Detector: ventana periódica w[n]=.5−.5cos(2pi n/N), n=0..N−1;
FFT real con tamaño4·2^ceil(log2N). Magnitudfloat64 convertida a dB relativos
al máximo mediante20log10(max(magnitud/max_global,10^(−300/20))),
piso−300dB; silencio explícito produce lista vacía. No eliminar DC
antes de calcular el máximo global. Buscar máximos con SciPy1.17 `find_peaks`,
sin `distance` ni selección top-k; `height` y `prominence` según calibración.
Refinar pico k con tres valores dB alpha,beta,gamma:
p=.5(alpha−gamma)/(alpha−2beta+gamma), frecuencia=(k+p)·fs/nfft.
Denominador0/meseta o p fuera de[−.5,.5]: conservar frecuencia del bin y flag,
no inventar refinamiento. Filtrar frecuencia final a[50,10000]Hz. Preservar
índices/propiedades descartados, bins/refinamientos y motivo de cada descarte.

Calibrar una pareja global de height∈{−20,−30,−40}dB y prominence∈{3,6,12}dB.
Nueve candidatos, mismos64casos y tres condiciones; minimizar un único escalar:
media uniforme de los192costes escena-condición de§4 (igual peso de los cuatro
escenarios y las tres condiciones). Empates exactos: tupla numérica ascendente,
height −40<−30<−20 y luego prominence3<6<12. Guardar los nueve agregados y los
1728costes individuales antes de congelar una pareja. Ningún modelo/fitter/score relacional
participa en la selección. Tests usan la pareja seleccionada sin retuning.

Referencias de algoritmo, no garantías de separación de fuentes:
https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.find_peaks.html
https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html
https://www.dsprelated.com/freebooks/sasp/Bias_Parabolic_Peak_Interpolation.html

## 3. Inferencia congelada y estados fuera de dominio

Eventos detectados ordenados por frecuencia, ties por índice espectral. Centrar
logfrecuencias enfloat64 y convertir afloat32; preservar IDs observados/rangos.
Todos los lectores reciben sólo esa lista y features frequency-only heredadas.
Port de inferencia sin acceso a fuentes verdaderas ni matching de evaluación.
Identidad externa de cada unidad: (role,scenario,condition,scene_id,split_seed),
con referencias separadas a emisión, waveform, detección e identidad del kernel.
Orden canónico: roles development/calibration/test; escenarios según§1;
condiciones canonical/nominal/short/noisy; scene_id ascendente dentro de escenario.
Celdas por ruta Inyección/Geométrica/Desacoplada/Local, backbone ascendente y
reader seed ascendente. El kernel conserva su esquema mínimo. Cómputo deduplicado
por hash exige receipt many-to-one explícito: nunca colapsar filas pareadas.

Reusar los tres backbones2026090721/22/23 y kernels de `scene_from_sources`,
`inputs_from_fits`, `candidate_inventory` y fitter Grid(257,65,stride4), con
normalizadoresTRAIN y escala geométrica anteriores. No modificar SIGMA2cents
del fitter aunque el ruido efectivo observado cambie; esa discrepancia se mide,
no se arregla antes de preguntar por transferencia del operador fijo.

Cuatro rutas Decisión seleccionadas: Inyección, Geométrica, Desacoplada, Local;
tres semillas de lector2026091491/92/93 y tres backbones:36estados, sin winner
seed ni nuevas inicializaciones. Epochs anteriores:30/45/40/45 respectivamente.
Scores clásicos Base/Extendida y bypass z conservan su precisión original.
Seleccionar estas rutas por la pregunta de medición, no afirmar un nuevo
contraste causal de loss; MSE queda histórico, no eliminado del archivo.

Por observación: unir los pools de los tres backbones y vecinos con misma
regla/prior; ajustar una vez por candidato compartido. La diferencia entre
condiciones puede cambiar ese universo. Registrar pools/candidatos descartados.
Es un proponente conjunto: C mide cobertura de ese universo compartido, no de
un backbone aislado. La referencia se llama «Extendida sobre universo compartido»:
su score es clásico, no necesariamente el origen de sus propuestas. Conservar
origin/checkpoints/parents y pertenencia a pool analítico, nativo de cada backbone
y vecinos, sólo como diagnóstico, sin selección por esas etiquetas.

Contar picos retenidos después del filtro y guardar lista absoluta/descartes.
Si Nfuera8..32, registrar OUTSIDE_OPERATOR_DOMAIN ANTES de q32/features/backbone/
pool/fitter; no recorte a32, relleno a8 ni catch genérico de excepciones del kernel.
Dentro del dominio, universo soportado vacío: NO_OBSERVABLE_CANDIDATE del kernel,
mapeado explícitamente a NO_CANDIDATE. Ambos tienen F=C=0, U evaluable cuando
corresponda, son abstención sin partición y
permanecen en extremo a extremo. No relajar grupos4..8 o introducir etiqueta
de fuente para completar grupos. Error numérico, hash o recurso es fallo
operativo que exige recuperación, no abstención científica.

## 4. Correspondencia y métricas

Matching es sólo evaluación posterior al sello. Distancia en cents absoluta
entre frecuencia emitida y detectada; no centrar separadamente para esta medida.
Grafo bipartito con aristas d≤20cents. Match conservador únicamente cuando
ambos extremos tienen grado1. Emitido grado0: missing; detectado grado0:
spurious/unassigned; componente aislada1–1: unique_tolerance_match.
Todos los vértices de las demás componentes no triviales son ambiguous,
incluidas hojas de grado1 conectadas a un extremo de grado mayor. Preservar
componentes y conteos por ambos lados, no sólo aristas ni grados máximos.
No convertir una fusión potencial en etiqueta única por asignación más cercana.
Grafo indica compatibilidad por tolerancia, no certifica fusión física.

Canónica usa identidad de emisión conocida, no matching aproximado. Duplicados
exactos permanecen eventos distintos. Sensibilidad predeclarada10/40cents sólo
reanaliza predicciones, no ajusta detector ni reemplaza el resultado20cents.

Detección: conservar conteos, errores en matches unívocos y coste tipoGOSPA
p=1,alpha=2,c=20cents. Minimizar suma d en asignaciones parciales uno-a-uno
más10·(n_emit+n_detect−2n_match), con no asignación permitida. Conservar
contribuciones localización/faltantes/espurios; ties d=20 se dejan sin asignar.
Esta asignación no adjudica la identidad para el score relacional. Con eventos
duplicados es una extensión a listas indexadas; no atribuirle sin prueba el
teorema de métrica sobre conjuntos de la fuente:
https://arxiv.org/html/1601.05585v7

Score relacional conservador extremo a extremo:

- T: número de pares emitidos distintos pertenecientes a la misma fuente.
- P: número de pares declarados juntos por la partición detectada, incluyendo
  pares con eventos ambiguos/no asignados. Abstención tieneP=0.
- TP: pares deP cuyos dos eventos poseen matches unívocos con misma fuente.
- F=2TP/(T+P). T>0 en este sampler; en fixturesT=P=0 definirF=1.

No llamarF accuracy, métrica formal ni recuperación verdadera de fusiones.
Es evaluación conservadora de relaciones atribuibles; las relaciones ambiguas
no obtienen crédito y se registra su masa, no se eliminan del denominadorT.

R: pares verdaderos cuyos dos eventos tienen matches unívocos. El máximo sin
restricciones de arquitectura esU=2R/(T+R): agrupar matches por fuente y dejar
otros eventos singleton lo alcanza bajo este score. C=maxF entre candidatos
observados, o0sin candidato/dominio. F≤C≤U. Descomposición exacta:
1−F=(1−U)+(U−C)+(C−F). Nombrar pérdida de correspondencia bajo tolerancia,
cobertura del proponente conjunto y elección, respectivamente. El primer término
no aísla error físico del detector; el segundo incluye los priors de grupos4..8;
el tercero es regret en F, no en VI. Es contabilidad, no identificación causal.

GuardarF de cada candidato y sus optima, sin alimentar esas cantidades al
lector. Secundarios: ARI y VI sobre soporte unívocamente correspondiente,
restringiendo particiones y retirando grupos vacíos sin re-fit. Reportar tamaño
del soporte y null si<2; no usar métricas sobre dos conjuntos de nodos distintos.
El comparador canónico también se restringe al mismo conjunto emitido para
contrastes de soporte común. VI/regret de este soporte no es el primario anterior.
Usar matches ordenados por índice emitido y labels restringidos recodificados
por primera aparición. ARI: sklearn1.8.0 adjusted_rand_score. VI no normalizada,
en bits: sum_ij (n_ij/n)[log2(a_i/n_ij)+log2(b_j/n_ij)], omitiendo n_ij=0,
float64, con a_i y b_j marginales de la contingencia. No usar el guard heredado
de mínimo3eventos ni las restricciones4..8 para estas métricas de soporte.

## 5. Estimandos y reporte

Unidad: escena, no pico ni celda neuronal. Promediar primero las nueve celdas
por ruta. Cuatro primarios en128escenas deformadas, todas las abstenciones
incluidas, audio nominal frente a canónica:

1. F_Extendida(audio)−F_Extendida(canónica).
2. F_Geométrica(audio)−F_Geométrica(canónica).
3. [F_Geométrica−F_Inyección](audio)−[F_Geométrica−F_Inyección](canónica).
4. F_Geométrica(audio)−F_Extendida(audio).

Bootstrap10000remuestras pareadas de128escenas, PCG64(2026091549), ICpercentil
98.75%, quantile linear. Guardar índices y distribuciones. No umbral científico,
no equivalencia inferida por intervalo que incluye0. Las emisiones fuera de la
banda de detección/por encima de Nyquist permanecen en el denominador emitido,
con sus flags; ni el detector ni el matching corrigen frecuencias usando truth.

Otras condiciones/escenarios/rutas, descomposición, matching10/40 y métricas de
soporte común son descriptivos predeclarados. El score20cents permanece fijo.
Probes mecánicos: orden de eventos, relabeling de fuentes sólo evaluación,
gain común sin clipping y conservación de razones antes del detector. No
prometer invariancia del detector con resolución absoluta fija.

## 6. Congelación, artefactos y replay

Manifest debe ligar código/protocolo/config/versiones,36cabezas,3backbones,
normalizadores/escala, rosters/exclusiones, detector seleccionado y presupuesto.
No abrir tests antes de ese manifest. Emisión y sidecars separados del lector;
sello de todas las predicciones antes de matching/evaluación. Cada escena/
condición tiene estados y recibos, sin sobrescritura de outputs completados.

Preservar waveform limpia y ruidosas, parámetros/ruido de render, espectro y
picos/descartes, observaciónq32, features/logits/pools/candidatos/fits, estados
de lectores referenciados, scores por candidato/celda y métricas por escena.
Replay de detección desde audio; replay de decisiones/métricas desde estados
guardados sin nuevo backbone, fitting o entrenamiento. Auditoría independiente
reconstruye toda aritmética de matching/F/descomposición/primarios y coteja
un corte mecánico de render/detector predeclarado, además de procedencia completa.

El perfil fija costes medidos y manifest ejecutable antes del test bajo el
[plan](PLAN_OPERATOR_UNDER_MEASUREMENT.md). Cierre exige resultados y auditorías,
no se satisface con este diseño o con un detector que pasa fixtures.

## 7. Perfil y admisión de costes

Roster inicial: scene_id0 de cada uno de los cuatro escenarios development,
con sus cuatro condiciones en el orden anterior:16observaciones, nunca test.
Antes de calibrar, usar height−30/prominence6 exclusivamente para el perfil.
Registrar N y número de candidatos/grupos por observación, también abstenciones;
no presentar este corte pequeño como cobertura empírica de todos los N8..32.
Agregar fixtures mecánicos N8/N32 y bancos máximos de82candidatos/328grupos
para dimensionar trabajo y memoria, sin truth ni estimandos del test.

Medir sensor, matching/evaluación, IO, forward de3backbones, fitter y36lectores
por separado. Proyectar2048observaciones y las64×3×9evaluaciones de calibración;
render se cuenta por escena, no por umbral. Para etapas variables usar el mayor
coste unitario medido (por evento para forward, grupo para fitter, candidato×lector
para readout) multiplicado por los máximos32/328/82×36 y el roster completo.
Usar coste máximo por waveform en sensor/IO; incluir warmup y costes fijos una
vez por proceso. Publicar fórmula, medidas y proyección tanto sin margen como×1.5.
Estos máximos conservadores pueden exigir revisar presupuesto antes del freeze;
no autorizan reducir tests ni eliminar casos tras observar respuestas.

Techos:600s perfilCPU,600s perfilCUDA,28800s trabajo acumulado (incluye perfiles,
calibración, campaña, recuperación, replay y auditoría),21600s etapas conGPU,
RAM8GiB, VRAM6GiB, nuevos artefactos16GiB. Reservar≥25% del total para replay/
auditoría y comprobar que proyección×1.5 de todo el trabajo pendiente cabe en
el saldo y reserva. La duraciónGPU es ocupación de etapa, no tiempo de kernels.
Si no cabe, revisar explícitamente presupuesto/diseño y publicar antes de freeze.
Replay completo desde artefactos; auditor independiente reconstruye todos los
cálculos de evaluación y coteja render/detector para IDs0y127 de cada escenario,
en las tres condiciones de audio, corte fijado antes del test.
