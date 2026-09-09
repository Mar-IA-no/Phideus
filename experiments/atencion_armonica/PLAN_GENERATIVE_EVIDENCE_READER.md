# Evidencia generativa en un lector aprendido

2026-09-08. Diseño inicial del siguiente goal finito. No es todavía un
protocolo congelado ni un resultado experimental. Requiere revisión
independiente antes de implementar el contraste.

## Pregunta y fundamento

¿La evidencia de ajuste conjunto a una ley de fuentes mejora la elección
aprendida de particiones fuera de distribución, cuando candidatos, priors,
cabeza y pérdida se mantienen comunes?

El [diagnóstico de rivales](RESULTS_OBSERVABLE_SOURCE_RIVALS.md) completó
96 escenas retrospectivas. Separó cobertura, ajuste e identidad: ampliar la
familia deformada reduce mucho el residual, pero apenas cambia las particiones
exactas. El ajuste explícito merece conservarse como operación experimental;
no demuestra que baste para identificar fuentes ni que una red lo aproveche.

Comparar directamente `argmin J` con los lectores anteriores no responde
la pregunta: J ajusta parámetros y ramas con distinta flexibilidad, mientras
las cabezas aprendidas predicen errores de separación y fusión. Además,
reciben distintos resúmenes de la observación. Esa comparación queda como
referencia de sistemas, no como estimando causal de arquitectura.

## Hipótesis geométrica y alcance

Las log-frecuencias centradas representan razones módulo escala global. La
compatibilidad de una partición exige índices, parámetros de fuentes y una
escala global conjuntamente realizables, no sólo relaciones locales.
La hipótesis es que un resumen de esa operación aporta información útil más
allá de los descriptores locales, y que una loss supervisada de partición
puede aprender cuándo utilizarla sin confundir menor residual con identidad.

El operador comparte ley, rangos y ruido con el generador conocido. Índices
1–8, tamaños 4–8, rango de fundamentales y restricciones de rama según k
son priors del sampler, no invariantes físicos universales. El experimento
no valida HIT ni audio medido y no descubre una geometría natural por ajustar
su propio mundo sintético. Tampoco modifica la geometría del backbone congelado.

## Único contraste neuronal

Tres brazos de una misma cabeza invariante de partición:

1. **Local**: descriptores comunes y compatibilidad local; bloque adicional
   reservado pero sin evidencia generativa.
2. **Generativa**: los mismos inputs más evidencia del ajuste de esa partición.
3. **Desacoplada**: mismo canal, con correspondencia entre evidencia y partición
   alterada determinísticamente dentro de estratos observables de soporte y
   flexibilidad: ramas disponibles, k, tamaños y dimensión, o equivalencia
   justificada en el protocolo. No basta conservar el marginal incondicional.
   Medir soporte efectivo y conservar explícitamente los estratos sin una
   permutación efectiva; no llamar intervención a una entrada intacta.

Los tres comparten arquitectura, dimensiones, inicialización pareada,
normalización train-only, batches, optimizador, calendario, targets y loss.
Un bloque sin señal no garantiza capacidad efectiva idéntica: informar su
actividad y conservar Desacoplada como control de información no alineada.
La intervención es acceso a evidencia computada bajo la ley, no una operación
neuronal nueva aislada ni igualdad informacional entre brazos.

La topología, shapes y estados iniciales deben ser idénticos entre brazos y
comprobarse por fixtures. La cabeza vigente cambia dimensiones según brazo:
se requiere una extensión explícita, no reutilizar esa asimetría. El nuevo
canal es de candidato y conserva el ajuste conjunto de la partición; no se
reduce al escalar por grupo que acepta la implementación anterior.

El operador generativo y los candidatos son comunes, computados una vez por
escena y reutilizados por los brazos. Pool unión de los tres checkpoints y
vecindad observable acotada se preservan con la misma máscara de soporte;
ningún brazo recibe la partición plantada, su k ni su régimen para completarlo.
Las escenas sin candidatos conservan su estado y denominador en evaluación.
Los resúmenes de complejidad, k, tamaños, soporte y disponibilidad de ramas
deben ser comunes: no se atribuye a residual una ventaja de máscara o prior.

Mantener la pérdida anterior: MSE de dos entropías condicionales normalizadas
por log N, con media uniforme de candidatos dentro de escena y luego escenas.
La decisión minimiza la suma de outputs; ties canónicos. No cambiar a la vez
supervisión por J, regularización física, backbone, proposer ni modalidad.

## Rebase y protocolo antes de datos nuevos

Reutilizar el núcleo generativo auditado y los estados originales sin
reescribir fuentes congeladas. Inspeccionar la cabeza, targets, features y
runner vigentes antes de definir sólo la extensión necesaria. La pérdida ya
está implementada en `src/atencion_armonica/learned_partition_model.py`;
el protocolo anterior conserva su definición y sus límites.

El protocolo ejecutable debe fijar antes de entrenar: vector generativo,
normalizadores, máscara, sham y soporte, fórmula de la cabeza y dimensiones,
roster, semillas, cantidades, receta, selección, métricas, incertidumbre,
referencias clásicas y presupuesto. Diseñar el vector a partir de train o
fixtures declarados; no afinarlo sobre los cuatro tests ya abiertos.

Reutilizar train y calibración preservados cuando sus puertos permitan el
contraste; declararlos como material abierto, no como nueva evidencia.
Cruzar los tres brazos con los tres checkpoints y tres inicializaciones:
27 entrenamientos, no seleccionar la semilla ganadora. Confirmación sobre
cuatro tests frescos IID, beta, polifonía y deformación, con muestras y receta
congeladas antes de generarlos; separar explícitamente análisis retrospectivo
de las 96 escenas anteriores. No generar datos nuevos durante este diseño.

Escenario primario: familia deformada. Lo motiva el desacople entre mejora de
fit e identidad observado antes de este diseño; en polifonía, además, faltan
6/24 plantadas del universo, frente a 3/24 bajo deformación. Esa observación
retrospectiva orienta el contraste, no pronostica su efecto ni su poder formal.
Polifonía, beta e IID quedan como transporte y límites, no un ganador agregado.

Generativa−Desacoplada estima el aporte de la alineación del canal bajo los
estratos declarados; Generativa−Local mide utilidad incremental y no aísla por
sí sola esa alineación, porque cambia actividad y capacidad efectiva. Fijar
incertidumbre pareada por escena y corrección para estos dos contrastes antes
de test. El ARI se informa sobre el soporte de salida común a los brazos,
junto a cobertura sobre todas las escenas y estados sin salida, sin convertir
ausencia en un ARI ficticio. Informar ambos errores de entropía, VI, cantidad
de fuentes, fragmentación, oracle del universo y los demás escenarios.
Las nueve celdas checkpoint × semilla no son nueve escenas independientes.
Guardar costos por candidato permite replay y análisis sin otro forward.

## Recursos y preservación

La campaña anterior tardó 44.760 s de búsqueda para 96 escenas según su
ledger; eso no pronostica linealmente una campaña de entrenamiento. Conservó
412.206.798 bytes con evaluación: escalar JSON por asignación puede costar
decenas de GiB. Perfilar cómputo, RAM, VRAM e I/O antes de materializar el
train completo. Usar almacenamiento compacto de los mismos factores o una
representación suficiente explícitamente auditada para replay; no borrar
evidencia antigua ni ocultar un cambio de autoridad como optimización.

La RTX 3090 local está autorizada sin permisos por corrida. Verificar estado
y ownership; usar CPU si es proporcionada, GPU si aporta eficiencia real,
con procesos recuperables que puedan detenerse cuando el usuario lo indique.
No enviar a Mendieta por rutina. Fijar límites operativos medidos en el
protocolo, no umbrales científicos de éxito inventados.

## Cierre y bifurcación

El goal exige implementación, 27 entrenamientos y cuatro tests completos,
replays, auditoría técnica y de alineación, documentación y publicación de
código y resultados. Un preflight o un diseño listo no completa ese objetivo;
un impedimento real se conserva como tal, sin declarar éxito.

Si la evidencia alineada aporta de manera consistente frente a ambos
controles, el siguiente goal podrá estudiar su integración o amortización
como operación aprendida. Si sólo mejora el ajuste pero no la decisión,
revisar la relación entre evidencia y pérdida antes de escalar. Si domina
la falta de candidatos, volver al proposer o fuentes latentes; si Local
basta, conservarlo. Una modalidad adicional requiere evidencia propia de
qué distinción observable falta. Estas son bifurcaciones, no una secuencia
obligatoria. La promoción arquitectónica y GO/NO-GO pertenecen al usuario.
