# Marco Epistemologico de Phideus

> Documento de formulacion general del programa. No reemplaza roadmaps ni reportes tecnicos: fija, en lenguaje mas estable, que tipo de conocimiento intenta producir Phideus, mediante que clase de metodo, y con que limites de validez.

---

## 1. Tesis central

Phideus parte de una hipotesis epistemologica fuerte pero formulada de manera deliberadamente sobria:

> Los modelos aprendidos sobre mediciones reales de un mismo fenomeno, observadas desde modalidades distintas, pueden funcionar como instrumentos experimentales para detectar que estructuras relacionales son privilegiadas para alinear, comprimir y transferir informacion entre esas modalidades.

La tesis no dice que los embeddings sean espejos ontologicos del mundo. Dice algo mas preciso y mas defendible:

- que los embeddings pueden funcionar como **espacios de prueba**;
- que intervenciones controladas sobre esos espacios pueden revelar **invariantes estructurales**;
- y que, si esas invariantes reaparecen de forma robusta a traves de dominios, sensores y arquitecturas, entonces constituyen evidencia indirecta sobre la organizacion informacional del fenomeno medido.

---

## 2. Que estudia Phideus

Phideus opera en tres niveles a la vez.

### 2.1 Nivel de ingenieria

Se estudian:

- arquitecturas de encoders;
- losses contrastivas;
- projection heads;
- mecanismos de inyeccion;
- dinamicas de entrenamiento;
- retrieval y representaciones.

Este nivel es real y no debe negarse. Parte del programa es, efectivamente, investigacion en ML.

### 2.2 Nivel metodologico

Se propone un metodo de investigacion:

1. tomar datos fisicamente anclados en dos o mas modalidades;
2. construir representaciones aprendidas;
3. intervenir esas representaciones con descriptores estructurales;
4. medir cambios causales en alineacion, geometria, invariancia y robustez.

En este nivel, la red no se usa solo para "resolver una tarea", sino como **instrumento experimental**.

### 2.3 Nivel sustantivo

Se prueba una hipotesis mas profunda:

> que ciertas estructuras relacionales, especialmente las razones, son variables privilegiadas para la organizacion de la informacion en fenomenos fisicos.

Este es el nivel donde aparece la Harmonic Information Theory como horizonte del programa.

---

## 3. El puente atomos-bits-modelos

La validez del metodo depende de una cadena de anclaje:

1. hay un fenomeno fisico real;
2. ese fenomeno se mide con uno o mas sensores;
3. la digitalizacion preserva la estructura relevante;
4. el modelo aprende sobre esas mediciones;
5. intervenimos el modelo y observamos que cambia;
6. inferimos si cierta estructura relacional parece invariante al cambio de modalidad.

Phideus no hace inferencia directa "desde el embedding al mundo". Hace inferencia sobre el mundo a traves de una cadena controlada de mediaciones.

La legitimidad de esa inferencia depende de:

- que el dato este fisicamente anclado;
- que la estructura investigada sobreviva a la digitalizacion;
- que los controles reduzcan la probabilidad de artefactos del instrumento;
- y que haya convergencia de evidencia.

---

## 4. Que tipo de realismo asume el programa

La posicion mas adecuada para describir a Phideus no es el realismo ingenuo ni el instrumentalismo puro.

La formulacion mas ajustada es una forma de **realismo estructural modesto**:

> cuando una estructura relacional mejora de manera causal y robusta la alineacion entre modalidades distintas de un mismo fenomeno, eso constituye evidencia indirecta de que dicha estructura captura un aspecto real e invariante del fenomeno medido.

Esto implica:

- no asumir que una capa o una neurona "significan" directamente algo del mundo;
- priorizar relaciones, invariantes y reorganizaciones geometricas;
- distinguir siempre entre observacion, hipotesis e inferencia.

---

## 5. Que no afirma Phideus

Phideus no afirma:

- que una red neuronal revele directamente la ontologia del mundo;
- que toda mejora interna del modelo sea extrapolable al fenomeno fisico;
- que una coordenada del embedding tenga una semantica estable y universal;
- que exito en una tarea equivalga por si mismo a explicacion del fenomeno.

Estas negaciones son importantes. Definen el limite metodologico del programa y evitan una sobrelectura metafisica de resultados instrumentales.

---

## 6. Que si afirma Phideus

Phideus si afirma que:

1. las redes pueden funcionar como **instrumentos de deteccion de estructura**;
2. la validez del instrumento crece con controles causales, comparabilidad estricta y convergencia de evidencia;
3. si una estructura reaparece a traves de escalones y dominios, deja de ser plausible leerla como simple conveniencia del modelo;
4. el objeto de estudio no es la red en si, sino las invariantes relacionales que se vuelven observables a traves de ella.

---

## 7. Que hace fuerte una inferencia en este marco

No toda mejora metricica tiene el mismo peso epistemologico. En Phideus, una inferencia gana fuerza cuando se apoya en varios de estos criterios:

### 7.1 Causalidad o comparabilidad estricta

- ablaciones parameter-matched;
- controles `zero`, `random`, `shuffled`;
- misma receta, misma arquitectura, mismo schedule.

### 7.2 Robustez estadistica

- multiples seeds;
- intervalos de confianza;
- estabilidad entre reruns.

### 7.3 Robustez entre arquitecturas

- el efecto no depende de un solo encoder o mecanismo de inyeccion.

### 7.4 Robustez entre dominios

- la estructura reaparece en musica, voz y, eventualmente, fisiologia.

### 7.5 Capacidad predictiva

- la teoria no solo explica retrospectivamente;
- tambien anticipa donde deberia aparecer o desaparecer el efecto.

---

## 8. El concepto clave: invariantes, no coordenadas

La nocion central del programa no debe ser "el significado de una neurona" sino **la persistencia de una relacion**.

Lo que importa es:

- que vecindades se preservan;
- que distancias se reorganizan;
- que transformaciones dejan la estructura casi intacta;
- que variables sobreviven al cambio de sensor;
- y que descriptores inducen mejoras replicables.

Phideus, en su mejor formulacion, no es una teoria de activaciones internas, sino una teoria experimental de **invariantes relacionales**.

---

## 9. Formula sintetica

Si hubiera que condensar el marco epistemologico del proyecto en una sola proposicion, podria ser esta:

> Phideus usa modelos aprendidos sobre datos fisicamente anclados como instrumentos de intervencion y lectura para explorar que estructuras informacionales son invariantes, causalmente eficaces y transferibles entre modalidades distintas de un mismo fenomeno.

Y una consecuencia fuerte de esa proposicion es:

> si las razones siguen apareciendo como variables privilegiadas a traves de escalones, sensores y dominios, entonces ya no sera razonable interpretarlas como una simple conveniencia del modelo; habra que tratarlas como una pista seria sobre la organizacion de ciertos fenomenos naturales.

---

## 10. Programa de validacion

Para que esta epistemologia se fortalezca, el proyecto necesita avanzar en cuatro direcciones:

1. **Mas dominios**  
   Expandir la observacion desde Audio↔MIDI hacia Speech↔EGG y luego ECG↔PPG.

2. **Mas instrumentos**  
   Probar diferentes arquitecturas y regimens para reducir la probabilidad de artefactos del instrumento.

3. **Mas teoria**  
   Formalizar mejor por que las razones deberian reducir entropia o facilitar alineacion.

4. **Mas prediccion**  
   Pasar de explicaciones retrospectivas a predicciones nuevas sobre donde el efecto deberia o no deberia aparecer.

---

## 11. Conclusión

Phideus no propone una ontologia de embeddings. Propone algo mas sobrio y, por eso mismo, mas fuerte:

> una ciencia experimental de invariantes relacionales mediada por modelos aprendidos.

Si este programa sigue acumulando evidencia a traves de escalones y dominios, su valor no sera solo haber mejorado metricas de retrieval o haber construido mejores pipelines de ML. Su valor sera haber mostrado que ciertos modelos aprendidos, usados con disciplina experimental, pueden servir como instrumentos legitimos para investigar estructura informacional en fenomenos fisicos reales.
