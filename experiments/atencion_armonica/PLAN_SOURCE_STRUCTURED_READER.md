# Lector estructurado con coherencia de fuente

Estado: diseño reauditorado; habilita implementación, fixtures y preflight.
No ejecutado. No modifica los protocolos ni artefactos anteriores.

## Pregunta y posición arquitectónica

¿La coherencia conjunta de un grupo candidato aporta a producir particiones
mejores, más allá de la evidencia neuronal de pares, la cardinalidad y la
compatibilidad local? El [diagnóstico anterior](RESULTS_SHARED_SOURCE_COHERENCE.md)
construyó el fit compartido, pero también encontró grupos mixtos con residual
pequeño. No justifica un certificado físico ni una regla dura de pertenencia.

La candidata es un sistema compuesto: red de pares congelada → candidatos
de partición compartidos → energía de lectura con factores de grupo →
partición y witnesses descriptivos. La geometría interviene en una operación
de inferencia; no se presenta como nueva geometría neuronal aprendida.
El coeficiente del factor se elige sobre calibración fresca, manteniendo
fijos pesos neuronales, pérdida BCE histórica, búsqueda y evaluación.

El baseline descriptor-guided es adversario central. Se reutilizan los tres
checkpoints `pairs_descriptors` de la campaña cerrada, sin nuevos trainings
ni usar los modelos regularizados. La alternativa de fuentes latentes
aprendidas queda candidata no implementada: este contraste probará primero
si el factor geométrico compra utilidad dentro de una interfaz explícita.
Stage B y audio detectado permanecen alternativas, no pasos obligatorios.

## Fuentes y límites del modelo relacional

La wiki y el programa geométrico sostienen la separación entre observación,
pertenencia, parámetros testigo y decisión. La familia espectral y el
productor frequency-only son los ya estudiados; no se necesita otra ola
bibliográfica para este contraste. Una consulta externa sólo procede por
una dependencia concreta de implementación que no resuelvan las fuentes
locales. No se reclama novedad histórica del método.

El modelo relacional supone que una fuente puede organizar un subconjunto
de eventos con índices distintos y parámetros compartidos. Deja fuera fase,
tiempo, amplitudes, colisiones de detector y fuentes físicas no descritas por
esa familia. Es una hipótesis operativa estrecha, no la ontología de HIT.
El test de familia deformada interroga su fragilidad, pero no reemplaza una
medición real ni adjudica identificabilidad de la partición plantada.

## Datos frescos y acceso

Nuevo productor/versionado en archivos distintos, con la misma ley de
`shared_partial_data.py` y namespace de semillas nuevo. No mutar SPLITS ni
fuentes congeladas anteriores. Fijar antes de generar:

| Rol | Ley | Escenas | Split seed |
|---|---|---:|---:|
| Calibración | IID | 256 | 2026090780 |
| Test IID | IID | 256 | 2026090781 |
| Test beta | OOD beta | 256 | 2026090782 |
| Test polifonía | OOD polifonía | 256 | 2026090783 |
| Test deformada | Familia deformada | 256 | 2026090784 |

IDs 0..255 de cada rol. Mismos draws y observaciones para los tres
checkpoints y todos los lectores. Preservar observaciones, sidecars,
features, RNG/semillas/manifests e identidad ordenada. Prohibir duplicados
observables entre roles y frente al corpus anterior, con un fallo explícito
en vez de reemplazar escenas. La calibración es IID: no suministrar el
nombre del slice ni parámetros verdaderos a la inferencia.

Cada forward recibe sólo q32 y los descriptores históricos; guardar logits
float32 completos, checkpoints/hash y configuración. Etiquetas, índices,
beta/f0 plantados y k verdadero sólo entran en evaluación/calibración de
lectura, nunca en construcción del pool, fit ni energía de test.
No generar tests antes de cerrar implementación, calibración y freeze de
lectores; ninguna corrección posterior puede volverlos a llamar frescos.

## Pool común y búsqueda acotada

Para cada escena×checkpoint construir dos árboles average-linkage:
distancia `1-sigmoid(z)` y distancia `1-pair_support` analítico. Usar el
mismo algoritmo que el lector histórico, pero conservar la partición
después de cada merge, incluida la partición inicial singleton. Es la
unión de hasta 2N particiones, deduplicada por sus conjuntos de miembros.
Todo brazo recibe exactamente el mismo pool, incluyendo evidencia física
local del árbol analítico. El contraste adjudica el factor conjunto
incremental, no todo acceso a información armónica.

Un filtro común conserva sólo particiones cuyo grupo mayor tenga hasta
ocho miembros. Es el prior de cardinalidad del banco, no un hallazgo del
fit. Registrar cuántos candidatos retira; singleton garantiza pool no vacío.
El lector histórico sin ese filtro queda como referencia separada: una
mejora contra él puede provenir del pool o del prior y no acredita geometría.
No usar k verdadero, verdad de grupos ni criterio de error para construir
o podar candidatos. No añadir búsqueda ad hoc después de ver resultados.

Desempates: canonicalizar eventos por q observado, usar orden estable y
firmas de conjuntos para reproducibilidad; registrar empates de q/merge.
La equivariancia se verifica fuera de empates exactos y los casos
degenerados se conservan con autoridad ambigua, sin afirmar unicidad.

## Energías y controles

Para partición pi, y_pi es su matriz binaria de copertenencia. La energía
de pares es `E_pair=-sum_(i<j)y_pi,ij*z_ij/N`. Es la suma de BCE menos
`sum softplus(z)`, constante entre candidatos, normalizada por evento y
no por C(N,2). Conserva exactamente el argmin de BCE para Pares, pero usa
el mismo denominador N que los factores de grupo: agregar otros grupos
no aumenta por construcción la fuerza relativa de gamma sobre una decisión
local fija. No garantiza invariancia estadística frente a cualquier cambio
de N o tamaño de grupo. Guardar suma BCE, constante, suma dependiente de
partición y costo por evento. Usa logits conservados, sin clipping.
Es un costo compuesto de lectura que puede ser negativo, no un posterior
conjunto normalizado ni una log-verosimilitud física identificada.

Para grupo G de tamaño m=3..8, obtener el RMS fino r_G y witness con el
fitter auditado, grilla1025/coarse257, beta1e-5..0,02 e índices1..8.
Definir `h_global(G)=r_G²/(r_G²+4)` con r en cents. El 4 proviene de
(2 cents)², ruido nominal de la ley, y no se recalibra por test.
Para m<3, asignar contribución cero como **ausencia de restricción**, no
como consistencia física; reportar explícitamente su masa y cardinalidad.
La energía geométrica es `E_global=sum_G m*h_global(G)/N`.

Lectores sobre el mismo pool:

1. **Pares**: minimiza E_pair; no coeficiente adicional.
2. **Fuente compartida**: minimiza E_pair + gamma*E_global.
3. **Fuente desacoplada**: misma fórmula con costos globales permutados
   entre grupos candidatos únicos del mismo tamaño dentro de la escena.
4. **Compatibilidad local**: reemplaza h_global por la media de
   `r_endpoint²/(r_endpoint²+4)` sobre todos los triples internos cacheados;
   misma transformación nominal de 2 cents y ponderación m/N. No reutiliza
   los pesos históricos cuya escala era 10 cents. No modifica ese cache.

El control desacoplado usa un corrimiento cíclico no identidad de la lista
lexicográfica de grupos de cada tamaño, elegido determinísticamente con
`rng=default_rng(SeedSequence([2026090785,split_seed,scene_id,m]))`:
para L grupos, `shift=int(rng.integers(1,L))` y costo desacoplado de grupo j
igual al costo global del grupo `(j+shift)%L`. Para L=1 no hay draw.
Conserva exactamente el
multiset de costos por tamaño y no requiere otro fit. Si sólo existe un
grupo de un tamaño, conserva su costo y registra falta de soporte para
desacoplar ese estrato; si todos los costos relevantes quedan idénticos,
la escena es NO_SHAM_CONTRAST, no evidencia de igualdad de mecanismos.
Se ejecuta el control completo y se reporta soporte por escena/estrato:
no se promedian únicamente escenas donde el sham funcionó.

Guardar por tamaño L, shift, fracción de costos cambiados, media/máximo
de |h_sham-h_global| y conteos de aparición en el pool; por escena×seed,
el vector de energías de candidatos, fracción cambiada y media/máximo de
|E_sham-E_global|. Preservar costos únicos no preserva su multiset ponderado
por ocurrencias; no afirmar ese matching adicional. El contraste de ranking
es no evaluable si E_sham-E_global es constante sobre todo el pool
(comparación float64 exacta): `NOT_EVALUABLE_SHAM_SUPPORT`, incluso si
hubo desplazamiento de costos individuales. Reportar cobertura /768
escena×seed por slice y /256 escenas con soporte en las tres seeds.
Si no hay soporte completo, la atribución semántica global queda
`PARTIAL_SHAM_SUPPORT` (o `NOT_EVALUABLE_SHAM_SUPPORT` si cobertura cero);
mantener el efecto sobre todas las escenas, sin convertirlo en atribución
semántica para el corpus completo. Incluso con soporte completo, la
comparación condiciona a una rotación fijada, no a un promedio de shams.

Compatibilidad local y RMS conjunto no son la misma distancia ni comparten
interpretación probabilística. Sus transformaciones acotadas permiten
contrastar operaciones concretas con la misma escala nominal, no igualan
su distribución ni aíslan únicamente compartir parámetros: también cambia
la definición del residual. No sustraer distancias para certificar una
brecha local-global. Todos los candidatos retienen sus costos y estados,
incluidas las distribuciones por tamaño antes de elegir gamma.

Argmin final: energía float64 exacta y luego firma canónica de partición
en orden lexicográfico. La firma ordena los miembros por rango de q y los
grupos lexicográficamente; no usa etiquetas. Registrar mínimo, cantidad de
co-mínimos exactos, siguiente energía distinta/gap (null si no existe) y
autoridad `TIED_CANONICAL_CHOICE` o `UNIQUE_WITHIN_POOL`. Gamma0 usa la
misma función que Pares, sin una ruta de desempate distinta. Los empates de
q conservan el orden estable observado y su autoridad ambigua ya declarada.

Para los tres lectores con factor, gamma pertenece a
`{0,0.01,0.03,0.1,0.3,1}`. Elegir un gamma por lector, común a los tres
checkpoints, maximizando ARI medio en las 256 escenas de calibración:
primero promedio de seeds dentro de escena y después de escenas. Desempate
por menor gamma. No optimizar la forma del costo, resolución, pool, prior
ni escala. Gamma=0 es una salida válida que reproduce Pares y niega una
contribución elegida del factor. Selección y roster quedan hasheados antes
de abrir tests. Preservar además el lector histórico con sus umbrales
0,55/0,65/0,60 como referencia, sin reseleccionarlos.

## Evaluación y atribución

Primario: delta de ARI en OOD polifonía, Fuente compartida menos cada uno
de los tres controles Pares/desacoplada/local. La motivación es comprobar
una operación de grupo cuando aumenta la cantidad de fuentes, no cambiar
retrospectivamente el primario del training anterior (que fue OOD beta).

Todos los tests conservan ARI, partición exacta, error de copertenencia
por escena, k estimado/error absoluto y masa de miembros en grupos m<3.
Informar costo geométrico y RMS por tamaño, sin tomar cada grupo como réplica.
Por candidato guardar también número y masa de grupos m<3 y masa total con
costo geométrico cero, separando ausencia de restricción de residual cero.
El mínimo tiene distinta flexibilidad por tamaño: C(8,m) asignaciones,
56 para m3 y una para m8. No interpretar menores costos entre tamaños
como mayor evidencia física ni introducir una corrección aprendida del test.
Conservar la mejor ARI disponible dentro del pool como referencia
privilegiada de cobertura de búsqueda, no como reader deployable.
Los logits y su Brier permanecen idénticos entre lectores: mejorar una
partición no mejora retrospectivamente las probabilidades neuronales.

Promediar las tres seeds dentro de cada escena; reportar también cada seed.
Bootstrap pareado por escena, 2.000 remuestras, semilla2026090786, índices
preservados y compartidos entre controles dentro de slice. Para los tres
contrastes primarios usar intervalos percentiles98,333333% como ajuste
Bonferroni nominal de familia al5%; no es garantía exacta de cobertura
del bootstrap. Secundarios IID/beta/deformada e histórico: intervalos95%
descriptivos, sin selección ni sustitución del primario.
Reportar efectos e incertidumbre, no inventar una regla de promoción.

Lectura conjunta obligatoria por contraste/slice: delta ARI, error de
copertenencia, k estimado, error absoluto de k, masa sub-3 y oracle del pool.
Registrar `GAIN_WITH_MORE_GROUPS` si delta ARI>0 y delta k>0; registrar
`FRAGMENTATION_ATTRIBUTION_UNRESOLVED` si delta ARI>0 y aumenta la masa
sub-3 o el error absoluto de k. Son signos descriptivos de la media, no
umbrales de significación ni demostración de la causa. La mejora de ARI
no autoriza entonces un claim de coherencia separado de fragmentación.
Ausencia de esas banderas tampoco identifica el mecanismo; mantener
costos y resultados por tamaño, cobertura y controles como evidencia.

La evidencia condiciona a un corpus de training y tres checkpoints
históricos, con nuevas escenas de la misma familia de productor. No es
réplica de training ni validación de audio real. Una diferencia respecto
de Pares mezcla acción del factor y elección de gamma dentro de una receta
fijada; sham/local ayudan a localizarla, no conceden autoridad física.

## Implementación, recursos y auditorías

Archivos nuevos, outputs nuevos y fuentes anteriores intactas. Tests:
pool completo/deduplicado/cobertura; igualdad de acceso entre lectores;
no truth en inferencia; identidad de logits; permutación/offset fuera de
empates; costos/multisets sham; singleton y cardinalidad inválida; energía
comparada con enumeración pequeña; gamma0; selección validation-only;
gate de test; replay e integridad de artefactos.
Agregar fixtures de réplica/disyunción de grupos con cambio local fijo:
tras multiplicar por N, delta E_pair y delta E_global no cambian al agregar
eventos ajenos no fusionados en ese cambio. Verificar transformación local
y global idéntica para un mismo residual nominal, gamma0 byte-exacto,
ties/reordenamiento, sham con cambio de costos pero energía constante,
y todos los estados de fragmentación sin usar truth en la inferencia.

Preflight mecánico N32 sin corpus: pool máximo, costos por tamaño,
serialización y profiler forward de checkpoints sin etiquetas. Primero
estimar costo CPU de1.280 escenas×3checkpoints, reutilizando fits por
grupo/escena y el árbol analítico. Máximo120s para preflight CPU y1GiB RSS;
fases reales hasta1.200sCPU/<2GiB cada una sólo si la proyección lo permite.
Proyectar fits por grupos únicos evaluables a través de los tres pools
neuronales y el árbol analítico por escena, no por cantidad de particiones.
Guardar máximos/distribuciones de candidatos, grupos por tamaño, hits/misses,
celdas de grilla, bytes serializados y tiempos separados de features,
pool/fit, selección, lectura y bootstrap. Incluir los archivos a preservar
y una proyección conservadora sin hits cuando el fixture no justifique más.
No comenzar una corrida CPU larga si una implementación GPU justificada
es materialmente más eficiente: documentar perfil y revisar recursos antes.
No recortar muestra, controles o grilla después de ver costo o resultados.

Los nuevos forwards de tres modelos congelados justifican una ventana3090
breve; estimación inicial≤10min y≤2GiBVRAM basada en el forward anterior,
que se verifica mediante perfil antes del despacho. Informar alcance y
confirmar disponibilidad vigente; Mendieta sólo si el despacho tolera cola
y aporta eficiencia real. La cesión anterior no sustituye verificar un
conflicto de uso actual. No hay entrenamiento neuronal en este contraste.

Auditar implementación antes de datos/selección. El freeze
`FROZEN_BEFORE_TEST` vincula hashes del plan y todas las fuentes auditadas,
productor/ley/roster de escenas, manifests de observación/features de
calibración, checkpoints/logits, pools y selección; liga también algoritmo
y orden del pool, prior, energías/normalización, grids, sham, roster/elección
gamma, desempates, métricas/bootstrap y presupuestos. Un consumidor recibe
el SHA exacto esperado, verifica transitivamente sus dependencias y rechaza
ausencia, modificación o cualquier marcador FAILURE/INCOMPLETE. Generador,
forward y análisis de test comparten esa guarda fail-closed. Los nuevos
manifests de test encadenan ese freeze; análisis verifica también sus
observaciones/features/logits ordenados y completos. No admitir un mero
flag allow_test que eluda el binding. Toda corrección posterior se versiona
y explicita el acceso ya ocurrido, sin recuperar artificialmente frescura.

Congelar antes de test,
ejecutar primaria y replay de lectura sin repetir forwards, y conservar
logits, todos los pools/costos, particiones, witnesses, selección, métricas,
bootstrap, configs, hashes, recursos y fallos. Auditoría independiente final
de evidencia y alineación; documentación/wiki y commit/push periódicos.

## Condición de cierre y bifurcaciones

El goal termina con el sistema implementado, comparación fresca completa,
replay y auditorías, o impedimento material explícito que impida ejecutarlo;
no se redefine como otro diagnóstico preparatorio. Un resultado negativo o
gamma0 también cierra la pregunta. GO/NO-GO pertenece al usuario.

Si el factor conjunto mejora frente a Pares, local y desacoplado sin ocultar
fragmentación o colapso de cobertura, estudiar una fuente latente aprendida
o ampliación de búsqueda con atribución separada. Si sólo mejora Pares,
no atribuir el efecto al significado geométrico. Si el pool carece de
particiones útiles, cambiar búsqueda/representación antes de otra pérdida.
Si los grupos mixtos siguen siendo plausibles o el prior de familia falla,
investigar información temporal/fase/medición o una salida ambigua en vez de
endurecer el residual. El resultado determina el sucesor; no queda elegido
de antemano ni se abre una cadena ilimitada de diagnósticos auxiliares.
