# Protocolo: operación geométrica, objetivo y decisión

2026-09-14. Diseño previo a implementación y barrido; pendiente de auditoría
independiente. Desarrolla el [plan](PLAN_OPERATOR_OBJECTIVE_ALIGNMENT.md), no
reescribe el contraste generativo. Autoridad retrospectiva y diagnóstica.

## Pregunta, unidad y frontera

¿El mínimo de ajuste conjunto ordena las particiones como el objetivo de
entropías y qué diferencias aparecen al pasar por regresión y decisión?
Familia deformada es el escenario principal del diagnóstico. IID, mayor beta
y polifonía son comparadores secundarios, no un score global del programa.
No se busca significación nueva ni se define un umbral GO/NO-GO.

Se usan exactamente cuatro rosters de 512 escenas y las 27 celdas originales
Local/Generativa/Desacoplada × checkpoints 2026090721/22/23 × lectores
2026090991/92/93. Los 18 outputs de intervenciones con cabeza fija permanecen
preservados pero no integran este diagnóstico: el brazo Desacoplada ya aporta
el control aprendido requerido. No se genera otro sham.

No training, forward, fit, muestreo, cambio de pool, selección de celdas,
ajuste de hiperparámetros ni completamiento con etiquetas. Los tests ya están
abiertos: ni congelar este protocolo ni hacer replay recupera prospectividad.
El turno anterior cerró el contraste y abrió este goal; se clasifica como
progreso, no espera ni bloqueo. El rebase actual inspeccionó código y una
escena ya abierta para conocer formatos, sin calcular un nuevo contraste.

## Fuentes y vínculo de identidad

Raíz fuente: `data/atencion_armonica/generative_evidence_reader_v1/`.
El manifiesto nuevo fijará:

- cierre `fresh/test_completion.json`, SHA256
  `160473a7c88ab9e9a296a9e37a3145546a6ccc7019dbec473394ccd966cae3f9`;
- las trece etapas tipadas y sus referencias de salida, con igualdad entre
  evaluación y replay de cada test, y el freeze común;
- índice de evaluación, predicciones y sello de cada test, sus 512 registros
  de escena y los 27 readouts originales con identidad de celda sin duplicados;
- por escena: draw/observación/sidecar, recibo de fit y su archivo comprimido,
  inventario y particiones soportadas, cotas por rama, targets y métricas;
- outputs float32 y offsets de las 27 celdas, ligados a los mismos candidatos;
- bytes y SHA256 de código, protocolo, versión Python/NumPy y presupuesto.

El índice de predicciones se resuelve desde `prediction_seal.prediction_index`
(actualmente `fresh/<split>/predictions.json`). Se autentica el roster completo
de 45, su binding y todos sus recibos antes de filtrar las 27 identidades con
`intervention=original`; no basta que existan 27 archivos con nombres esperados.

Cada lectura verifica path relativo bajo el repositorio, sin `..`, enlaces
simbólicos o escapes; hash y tamaño cuando lo incluya el recibo. Cada referencia
relativa usa la base declarada por su formato, no una búsqueda por filename.
Una identidad incoherente detiene el diagnóstico; no se corrige la fuente.
Los factores gzip se descomprimen una vez por escena con límite del recibo,
se comprueba su hash decodificado y se extraen sólo inventario y cotas. No se
reconstruye el fitter ni se repite su búsqueda. La extracción compacta guarda
los valores y su linaje para que el replay no deba releer árboles de factores.

El candidato `i` es la partición en orden de firma canónica creciente. Se
exigen firmas únicas, cobertura disjunta de eventos y límites heredados
(2–4 grupos de tamaño 4–8; máximo 82 candidatos). No se eliminan candidatos
por resultado. La presencia plantada pool/neighbor/absent debe coincidir con
la evaluación cerrada, al igual que las elecciones Base/Extendida y aprendidas.
El inventario puede contener candidatos excluidos: sólo sus entradas con
`status=SUPPORTED`, ordenadas por firma, se alinean con fits, targets y outputs.

## Tres cantidades no intercambiables

Para cada candidato P y partición plantada Y de N eventos:

Y debe usar el orden canónico de frecuencia: reconstruir con el puerto puro
`reconstruct_truth(observation, sidecar, split)['labels']` o una implementación
equivalente verificada contra ese contrato. Exigir `argsort(q32,kind='stable')`,
reconstrucción de ley/ruido y correspondencia de los IDs; no usar `source_ids`
en orden observado directamente. Las firmas candidatas indexan ese orden
canónico, incluidas frecuencias repetidas.

1. Componentes matemáticos `u64=(H(P|Y), H(Y|P))/log(N)`, usando log natural,
   contingencia en float64, ceros exactos canónicos y ninguna truncación.
   Se cotejan con las entropías crudas archivadas y las métricas normalizadas
   (`rtol=0, atol=1e-12`, control de implementación, no equivalencia científica).
2. Targets entregados `u32=float32(u64)`: igualdad exacta con
   `normalized_targets` archivados. `t64=sum(u64,dtype=float64)` es el target
   matemático principal. `t32=sum(u32,dtype=float32)` conserva el objetivo
   numérico de referencia; su eventual cambio de mínimo se informa aparte.
3. Predicciones `h32` archivadas. La elección real usa
   `s32=sum(h32,dtype=float32)`. Nunca se sustituye por suma float64 ni promedio
   de logits. Para la MSE diagnóstica se convierten operandos a float64 antes
   de restar: por candidato y componente j conservar
   `b_j=float64(h32_j)-float64(u32_j)` y `e_j=b_j**2`. La MSE y el sesgo por
   componente promedian candidatos→escenas; el escalar loss-like promedia
   ambos componentes→candidatos→escenas. Reproduce la ponderación de la loss,
   no promete identidad bit a bit con reducciones Torch float32 del training.

El objetivo se verifica contra `learned_partition_core.partition_errors`,
`generative_evidence_supervision.candidate_supervision` y
`learned_partition_model.partition_cost_loss`. No hay ambigüedad bibliográfica
bloqueante: las fuentes locales fijan la operación concreta que se estudia.
La época fue seleccionada por ARI de calibración, no por MSE. Este diagnóstico
conserva el estado seleccionado: no reelige época ni atribuye a optimización
una discrepancia que podría involucrar también selección o representación.

## Scores, órdenes y oracles

Todos los scores siguientes se minimizan:

- Extendida UB: mínimo UB entre ramas disponibles base-low/base-high/deformed-low.
- Base UB: mínimo UB disponible entre base-low/base-high.
- Extendida LB y Base LB: análogos con LB, sólo secundarios. Una LB baja no
  es una solución realizable ni un sustituto del UB.
- Cada una de las 27 sumas aprendidas `s32`.

La disponibilidad depende del candidato, no del régimen verdadero. Una
partición con k=4 grupos admite sólo base-low según el operador heredado. Se guarda
la rama minimizadora de cada score, con empate por nombre lexicográfico;
la elección de candidato desempata por firma canónica. Se verifican las
elecciones UB contra las elecciones clásicas preservadas sin refit.

El score escalar mínimo UB no representa toda la información recibida por la
cabeza. Conservar también las seis cotas por rama, `log1p(LB/UB/N)`, máscara
de disponibilidad, normalizador TRAIN ligado al freeze y canal float32
efectivamente entregado. Verificar `((log1p(cota/N)-media)/escala).astype(float32)`
y ceros de coordenadas no disponibles contra los inputs autenticados de los
tres checkpoints. Local conserva ceros y Desacoplada se verifica mediante
sus donantes preservados, sin sortear otra permutación. No recalibrar momentos.
El contraste J↔target examina la regla clásica, no prueba suficiencia ni
insuficiencia de ese vector completo o de la interfaz de la cabeza.

Oracle de target64 = argmin `t64`; oracle target32 = argmin `t32`;
oracle ARI = argmax ARI por candidato. Cada uno conserva índice canónico,
conjunto completo de coóptimos exactos y gap al siguiente nivel distinto.
No se colapsan oracles por mera igualdad aproximada. Se registra adicionalmente
cuántos candidatos están a distancia ≤1e-12 del mínimo target64 y si la
elección target32 cae en ese vecindario; es sensibilidad numérica predeclarada,
no una segunda regla de selección ni un umbral práctico.

## Métricas por escena y celda

Para cada score y cada universo considerado:

- Kendall tau-b contra `t64`. Para pares de candidatos i<j, contar concordantes
  C, discordantes D, ties sólo del score Tx y ties sólo del target Ty;
  los ties en ambos no entran al denominador.
  `tau=(C-D)/sqrt((C+D+Tx)*(C+D+Ty))`.
- Índice y firma elegidos, orden total score→firma, bloques de empates,
  número de co-mínimos y gap al siguiente score distinto.
- Regret target `t64[elegido]-min(t64)` y regret numérico análogo sobre t32.
- Acuerdo con el índice canónico del oracle target64 y pertenencia a su
  conjunto de coóptimos; son métricas distintas.
- ARI elegido y gap respecto de máximo ARI, error absoluto de k y exactitud
  de partición de la elección, usando métricas candidatas preservadas y
  cotejadas contra las etiquetas.
- Errores cuadrados y sesgos por candidato/componente en las celdas aprendidas,
  y error de suma por candidato `e_sum=(float64(s32)-float64(t32))**2`.
  Primero sumar en float32, después convertir y restar; no recomponer h ni u
  en float64. MSE de suma promedia e_sum por candidatos y luego escenas.

Se registra además target64 vs target32: coincidencia canónica, coóptimos y
regret cruzado; target64 vs oracle ARI: coincidencia canónica, intersección de
conjuntos óptimos, ARI perdido por mínimo target y target perdido por máximo
ARI. Un cambio de firma entre oracles coóptimos no prueba conflicto de objetivos.

Con cero candidatos: `NO_CANDIDATE`, órdenes vacíos y métricas indefinidas
`null`; la escena permanece en 512. Con un candidato las elecciones/regrets
son válidas, tau es `INSUFFICIENT_PAIRS`. Si su denominador es cero, tau es
`CONSTANT_SCORE`, `CONSTANT_TARGET` o `BOTH_CONSTANT`, no cero ni fallo.
No hay p-values ni intervalos confirmatorios nuevos. Se guardan C/D/Tx/Ty
para revisar qué parte de la alineación depende de empates.

## Estratos y agregación

Además del universo completo se usan tres particiones observables del roster:

1. k;
2. tupla ordenada de tamaños + máscara de ramas disponibles;
3. la segunda clave + rama minimizadora de Extendida UB.

En cada estrato se recalculan tau, elecciones, regret y coóptimos sólo entre
sus candidatos. La tercera estratificación condiciona también sobre el score
observado: es diagnóstico descriptivo, no identificación causal de un prior.
Se guardan todos los estratos, incluso singleton y correlaciones indefinidas.

El primario usa el universo completo de cada escena, sin reponderar por
estratos. Cada esquema estratificado es un resumen secundario separado, con
media uniforme de sus estratos definidos dentro de escena. No se mezcla con
el primario. Después se promedian celdas válidas del brazo dentro de escena
(nueve cuando todas están definidas), y finalmente se calculan media y
cuantiles 10/50/90 de escenas. Se publican denominadores de estratos, celdas
y escenas para cada métrica y conteos de constantes, no una cobertura universal.

Las diferencias se calculan primero por celda emparejada: para cabeza vs
Extendida, su mismo roster; para brazos, el mismo checkpoint y semilla. Se
intersectan estratos definidos en ambos comparadores antes de calcular cada
diferencia y luego se promedian las diferencias de celdas válidas dentro de
escena. Una escena sin pares definidos aporta null. Como sensibilidad
separada se exige soporte común a las nueve celdas (y, en comparación entre
brazos, a las 18); se informa cuánto excluye y cómo cambia el resultado.
No se resta una media condicionada a un soporte de otra con soporte distinto.
No se pondera por cantidad de candidatos, no se convierten celdas en unidades
independientes ni se suman los cuatro escenarios.

Se conserva además la distribución por celda y el desglose de escenas según
plantada pool/neighbor/absent. Esos últimos son slices privilegiados de
evaluación, nunca selectores de rama o features. Los gaps de las distintas
reglas no son componentes causales aditivos.

## Implementación, recursos y evidencia de cierre

Módulos nuevos separados: núcleo NumPy puro; adaptador de lectura autenticada
y extracción compacta; ejecutor con presupuesto y replay de sólo lectura.
No modificar fuentes congeladas. Outputs bajo
`data/atencion_armonica/operator_objective_alignment_v1/`; pruebas, perfiles y
staging propios bajo `.agent-work/phideus-operator-objective-20260914/`.

Antes de barrer: auditoría del protocolo, tests del núcleo (ties, constantes,
oracles disjuntos/coóptimos, rankings, estratos y agregación pareada), auditoría
de implementación y perfil CPU de la escena 0 de cada escenario. No elegir
el caso por velocidad. Antes se autentican los 2048 recibos sin decode y se
inventarían tamaños comprimidos/decodificados por escena y máximo. El rebase
observó 893130717/9280977105 bytes totales y máximo decodificado 12477653;
son metadatos, no tiempos medidos ni autenticación previa de los blobs.
Perfil con cap de 180s acumulados, un hilo y 6GiB RSS;
guardar tiempos de extracción y diagnóstico por separado. Si no alcanza,
detener y revisar costo antes de ampliar. No se genera otra escena.

Luego fijar un recibo de presupuesto conservador basado en ese perfil:
un operador CPU/hilo, 6GiB RSS, máximo 30 minutos acumulados de operadores
del diagnóstico, incluidos perfiles, extracción, barrido, replay y una reserva
de auditoría final; 4GiB de nuevos outputs/staging/fixtures y 80GiB libres.
La proyección debe incluir bytes reales declarados, 2048 escenas, 27 celdas,
estratos, outputs y replay. Usar como proyección conservadora de extracción
el máximo tiempo/byte decodificado de los cuatro perfiles por el total;
para diagnóstico, el máximo tiempo por par de candidatos y celda por la
carga máxima predeclarada de 82 candidatos, 27 celdas y cuatro esquemas.
Reservar al menos 600s del total para comprobación independiente de extracción
y resultados, y margen ×2 sobre la estimación del trabajo restante. Registrar
fórmula, valores y residual disponible antes de iniciar. Si la proyección no
cabe, detener y revisar implementación/costo, sin reducir el roster. Un límite
real excedido deja estado terminal `BUDGET_EXHAUSTED`, no éxito parcial ni
reintento automático; publicar evidencia y rediseñar explícitamente. Todos
los intentos consumen el mismo presupuesto; no se amplía ni reinicia en silencio.
Un perfil sin pares suficientes para estimar el costo no autoriza el barrido:
requiere ampliar explícitamente el perfil antes de fijar la proyección, sin
elegir escenas por resultado. El cap es tiempo de operadores instrumentados,
no tiempo de redacción o revisión humana; las pruebas de desarrollo se
registran aparte del roster diagnóstico.

Guardar manifiesto, referencias fuente, extracción compacta por escena,
arrays de outputs aprendidos, resultados por escena/celda/estrato, tablas y
resumen, consumo y estado terminal. Replay recalcula desde compactos
autenticados y exige igualdad de bytes científicos, sin reparar faltantes;
auditoría final vuelve a fuentes originales para verificar la extracción.
Los registros operativos de tiempo quedan fuera de la igualdad científica.
Publicación atómica sin overwrite: manifiesto `PREPARED`, luego `COMPLETE`
sólo si existen los 2048 resultados autenticados y `REPLAYED` sólo tras
comparación completa. `FAILED`, `PAUSED` y `BUDGET_EXHAUSTED` no confieren
autoridad a prefijos. Una continuación puede reutilizar únicamente prefijos
completos, autenticados y ligados al mismo manifiesto, conservando intentos
y presupuesto; no adopta directorios huérfanos ni sobrescribe discrepancias.

El cierre exige roster completo, replay, auditorías independientes técnica
y geométrica, informe con límites, docs/wiki y commit/push. El resultado debe
justificar un experimento prospectivo discriminante o identificar exactamente
qué permanece confundido; no justificar otra cadena de correlaciones o heads.
Promoción y GO/NO-GO permanecen reservados al usuario.
