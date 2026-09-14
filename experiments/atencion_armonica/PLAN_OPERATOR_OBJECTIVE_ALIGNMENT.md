# Alineación entre operación geométrica y objetivo de aprendizaje

2026-09-14. Diseño del siguiente goal finito, condicionado al cierre auditado
del [contraste generativo](RESULTS_GENERATIVE_EVIDENCE_READER.md). No cambia
el experimento anterior ni autoriza adaptar un lector a sus tests.

## Pregunta y motivo

¿El criterio de ajuste conjunto ordena las particiones de manera compatible
con el objetivo que aprende la cabeza, y dónde deja de conservarse esa
correspondencia?

En el primario deformado, Generativa mejora frente a Desacoplada pero no
establece una ventaja clara de ARI frente a Local. Extendida, como sistema,
alcanza mayor ARI que las cabezas; su comparación no localiza la causa. Antes
de amortizar el fitter o cambiar la loss, hace falta distinguir falta de
candidatos, orden geométrico, objetivo de partición y error de predicción.
El máximo ARI no es necesariamente el mínimo de VI normalizada, el target
sumado que gobierna la decisión del lector.

## Acción acotada

Implementar un diagnóstico CPU reproducible sobre los cuatro tests ahora
abiertos, usando sólo sus factores, cotas, candidatos, targets y outputs
preservados. Congelar su manifiesto y fórmulas antes de ejecutarlo. Es
investigación retrospectiva de mecanismo, no confirmación independiente.
No entrenar, ejecutar forwards, repetir fits, generar escenas, completar el
pool con etiquetas ni elegir hiperparámetros a partir del diagnóstico.

Comparación principal de mecanismo en familia deformada: orden de la mínima
cota superior entre las ramas permitidas de Extendida frente a la VI
normalizada verdadera por candidato. Calcular Kendall tau-b intraescena,
regret de target de su elección y acuerdo de elección con el oracle de target.
Contrastar con el orden de la suma de outputs de cada celda Generativa y
conservar Local y Desacoplada como referencias aprendidas. No promediar logits
ni tratar candidatos o celdas como escenas independientes.

El protocolo ejecutable debe precisar antes de correr:

- Identidad de inputs: cierre y hashes del contraste generativo, rosters de
  512 escenas por test y 27 celdas originales, mismos candidatos y máscaras.
- Target: VI normalizada igual a la suma de las dos entropías normalizadas
  verdaderas; comparación en float64 y verificación de su correspondencia
  con los targets float32 entregados durante aprendizaje. No redefinir el
  target por ARI ni confundir error de regresión y error de decisión.
  Separar tres mediaciones: score geométrico frente a target; cada componente
  predicho frente a su entropía verdadera (donde actúa la MSE); y suma predicha
  frente a orden, elección y regret. Una correlación de sumas no diagnostica
  por sí sola qué componente de la regresión falla.
- Oracle de target: mínimo target dentro del universo observable con empate
  por firma canónica; oracle ARI separado. Regret de target y gap de ARI usan
  sus propios oracles. Ninguno representa aprendibilidad demostrada.
- Unidad y soporte: escena; sin candidatos o correlación indefinida se
  conserva estado explícito, no cero. Definir ties, orden canónico, tamaño
  mínimo y tratamiento de vectores constantes. Medias de celdas primero
  dentro de escena; conteos y distribución por escenario, sin ganador global.
- Comparadores secundarios: Base, cotas inferiores y los otros tres
  escenarios; error de k, pertenencia plantada pool/vecinos/ausente y estratos
  observables de k/tamaños/ramas. No usar régimen verdadero para escoger rama.
  Calcular también alineación y regret dentro de esos estratos antes de
  agregar por escena, para distinguir relación de ajuste de prior de complejidad.
- Intervención de referencia: reutilizar el desacople ya fijado, si se
  necesitan rankings de canal perturbado; no sortear un sham más favorable.
- Salidas: tabla por escena y celda, metadatos de soporte, órdenes/elecciones,
  regrets, correlaciones y síntesis de desacuerdos entre oracles. Conservar
  índices que permitan reanálisis sin volver a descomprimir todos los factores.
- Recursos: un operador CPU, un hilo, lectura por escena, máximo 6 GiB RSS.
  Perfilar una unidad abierta antes del barrido y fijar presupuesto acumulado
  proporcionado; no iniciar un sustituto CPU largo de una etapa GPU.

Los contrastes oracle→Extendida, Extendida→Generativa y Generativa→Local
localizan diferencias observadas. No se presentan como una descomposición
causal aditiva: varían objetivo, representación e intervención. Las
correlaciones tampoco adjudican por sí solas qué módulo causó un error.

## Investigación y posibles salidas

Releer wiki, protocolo, definición de loss y fuentes del operador. Sólo
recuperar bibliografía externa si una ambigüedad concreta de la relación
entre energía, ranking y objetivo impide definir el contraste. Archivar esa
dependencia y fuente en Biblioteca; no abrir otra ola general.

Si el criterio geométrico ordena bien el target pero la cabeza no, el relevo
puede diseñar una energía estructurada u operación aprendida que preserve
ese orden; todavía deberá distinguir representación de entrenamiento/loss.
Si no ordena bien el target, revisar la correspondencia entre relación física
y objetivo antes de escalar. Si falta la solución del universo, revisar la
propuesta observable. Una dependencia de los priors puede justificar un
stress de ley o medición, no una afirmación de geometría universal.

Estas bifurcaciones no seleccionan una arquitectura de antemano ni autorizan
tuning post-test. Un análisis que no separe las explicaciones debe concluir
indeterminación y proponer el experimento que falta, sin fabricar una causa.
El diagnóstico debe desembocar en una clase de experimento discriminante o
una indeterminación concreta, no en otra cadena abierta de análisis post-hoc.

## Cierre

Protocolo y núcleo auditados, ejecución del roster completo, replay exacto,
informe con las cuatro familias de diferencias y sus límites, auditorías
técnica y de alineación, documentación y commit/push. El balance debe indicar
qué correspondencia geométrica se conserva o se pierde y cuál es el siguiente
experimento discriminante. No cierra con un plan o preflight; tampoco exige
que una arquitectura gane. Promoción y GO/NO-GO siguen siendo del usuario.
