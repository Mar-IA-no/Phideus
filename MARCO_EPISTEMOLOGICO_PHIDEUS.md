# Marco Epistemologico de Phideus

> Documento de posicion general del programa.
> No reemplaza roadmaps, reportes tecnicos ni notas de trabajo:
> fija, en un lenguaje estable, que clase de conocimiento intenta producir Phideus,
> mediante que metodo, con que aspiraciones, y con que limites de validez.

---

## 1. Proposito

Phideus explora la **Harmonic Information Theory**: la hipotesis de que las razones, proporciones y estructuras armonicas naturales constituyen un lenguaje informacional privilegiado para organizar, comprimir y alinear informacion entre modalidades distintas de un mismo fenomeno fisico.

El programa investiga esta hipotesis experimentalmente, usando modelos aprendidos sobre mediciones reales como instrumentos de prueba. La apuesta central:

> si las relaciones armonicas naturales — razones lineales de frecuencia, serie armonica fisica, regularidades del oscilador — mejoran de forma causal, robusta y transferible la alineacion cross-modal entre sensores distintos, eso constituye evidencia de que la armonia natural captura algo real sobre la organizacion informacional de ciertos fenomenos.

La armonia que interesa a Phideus no es la armonia perceptual (escalas logaritmicas, semitonos, temperamento igual), sino la **armonia natural**: la estructura fisica del fenomeno vibratorio antes de toda mediacion perceptual o cultural. Esta distincion es constitutiva del programa y atraviesa todas sus decisiones experimentales.

---

## 2. La pregunta de fondo

La pregunta del proyecto no es que encoder rinde mas, que loss converge mas rapido, ni que inyeccion sube mejor el Recall@10. Esas preguntas existen, pero son subordinadas.

La pregunta de fondo es:

> las razones y proporciones armonicas naturales constituyen variables privilegiadas para la alineacion cross-modal, por encima de controles espectrales genericos y de codificaciones perceptuales del mismo fenomeno?

Esta pregunta tiene tres niveles de exigencia:

1. **Mecanico**: los descriptores basados en armonia natural pueden inyectarse en un modelo y mejorar la alineacion? (Escalon 1 — establecido)
2. **Especifico**: esa mejora es atribuible a la armonia natural y no a informacion auxiliar generica? (Escalon 2 — en curso)
3. **Universal**: el efecto reaparece en dominios y fenomenos fisicos distintos? (Triplescaloneta — horizonte)

La conexion con la Harmonic Information Theory es directa y disciplinada por un metodo experimental que distingue con cuidado entre lo que los datos muestran y lo que la teoria aspira a decir.

---

## 3. Lo que Phideus afirma

### 3.1 Tesis instrumental

La red no es el objeto de estudio. Es un instrumento.

Asi como un telescopio no es el planeta y un espectrometro no es el atomo, un encoder no es el fenomeno fisico que genero los datos. Pero un instrumento bien calibrado, operado con controles y replicacion, produce conocimiento legitimo.

### 3.2 Tesis estructural

Lo que interesa no son activaciones aisladas ni coordenadas internas, sino relaciones armonicas: razones de frecuencia, proporciones entre armonicos, regularidades del oscilador, y la geometria que estas inducen en el espacio de representaciones. Phideus es una teoria experimental de invariantes armonicas naturales.

### 3.3 Tesis de evidencia indirecta

Cuando una estructura armonica natural mejora de manera causal y repetible la alineacion entre sensores distintos del mismo fenomeno — y lo hace por encima de controles espectrales genericos y de codificaciones perceptuales —, eso constituye evidencia indirecta de que la armonia natural captura un aspecto real de la organizacion informacional del fenomeno.

No es una prueba metafisica. Es evidencia indirecta, pero cientificamente seria.

### 3.4 Lo que estas tesis excluyen

El programa no sostiene que una red sea un espejo ontologico del mundo, que una coordenada de embedding tenga significado fisico literal, ni que una mejora en una sola arquitectura cierre una hipotesis de frontera. Estas exclusiones no son concesiones defensivas: son parte constitutiva del metodo. La fuerza de una inferencia en Phideus proviene de la convergencia de evidencia, no de la magnitud de un unico resultado.

---

## 4. El puente atomos-bits-atomos

Una de las ideas centrales del programa:

```
Mundo de atomos  ->  digitalizacion  ->  mundo de bits  ->  modelo  ->  embedding
     ^                                                                    |
     |--------------------------------------------------------------------|
                          inferencia estructural
```

La legitimidad epistemologica del programa depende de que este puente no sea arbitrario.

### 4.1 Primer tramo: atomos -> bits

En Phideus no trabajamos sobre simbolos puros ni sobre ficciones numericas desligadas del mundo. Trabajamos sobre mediciones digitales de fenomenos fisicos reales:

- un piano real ejecutado, grabado y registrado como MIDI;
- una voz real emitida por un oscilador glotal, captada por microfono y electroglotografo;
- mas adelante, un fenomeno fisiologico captado por ECG y PPG.

En todos estos casos hay una cadena de transduccion: el fenomeno existe, un sensor lo captura, la senal se digitaliza, el modelo recibe esa medicion. La validez del programa depende de que esa cadena preserve la estructura relevante.

### 4.2 Segundo tramo: bits -> modelo

El modelo no ve el mundo directo. Ve una medicion digitalizada. Pero esa medicion no es arbitraria. Si los descriptores se construyen sobre regularidades preservadas por la cadena de medicion, el modelo funciona como una superficie de prueba: lo perturbamos con un descriptor, observamos si reorganiza la geometria, medimos si mejora retrieval, verificamos si preserva invariantes, y registramos que estructuras responden de forma robusta.

### 4.3 Tercer tramo: modelo -> atomos

Este es el tramo que requiere mayor disciplina. La vuelta de bits a atomos se justifica cuando se cumplen condiciones precisas:

1. el fenomeno estaba fisicamente anclado desde el inicio;
2. la cadena de digitalizacion preserva la estructura relevante;
3. la intervencion sobre el modelo es controlada;
4. el efecto observado es robusto;
5. el efecto reaparece en mas de un regimen;
6. idealmente, la teoria genera predicciones nuevas.

Cuando estas condiciones se cumplen, el pasaje de vuelta constituye inferencia estructural razonable.

---

## 5. La analogia instrumental

### 5.1 La red como telescopio

Cuando Galileo observo puntos luminosos cerca de Jupiter, la pregunta fue: lunas reales o aberraciones del instrumento? La respuesta no vino de intuicion sobre la verdad del telescopio. Vino de replicacion, observacion repetida, convergencia con otros instrumentos, y capacidad predictiva.

Con Phideus ocurre algo semejante. Una sola mejora en una sola arquitectura no establece nada grande. Pero cuando la misma familia de relaciones reaparece con varios arms, varias seeds, varios tests, varias modalidades y varios escalones, la hipotesis de artefacto instrumental se debilita progresivamente.

### 5.2 La red como espectrometro

Un espectrometro recibe una senal del mundo, la descompone segun una estructura, devuelve una representacion interpretable, y a partir de ella inferimos propiedades del fenomeno fuente.

Phideus opera de manera analoga: la senal fisica entra al sistema, el modelo la transforma, el embedding y las metricas son el espectro que observamos, y los descriptores funcionan como filtros que cambian lo que el instrumento deja ver. Cuando un descriptor reorganiza el espacio y mejora la alineacion, la lectura no es que "la naturaleza vive en el embedding", sino que, al pasar la senal por este instrumento, cierta estructura deja una huella observable y manipulable.

### 5.3 La red como laboratorio

La red no es solo un instrumento pasivo de observacion: es tambien un laboratorio de intervencion. Podemos introducir una estructura, quitarla, degradarla, compararla contra controles, ver donde falla y donde se sostiene. Eso la convierte en un espacio donde hipotesis estructurales se ponen a prueba en condiciones controladas.

---

## 6. Posicion filosofica: realismo estructural modesto

La posicion epistemologica de Phideus se describe como:

> realismo estructural modesto, operacional y experimental.

**Realismo** porque el proyecto apuesta a que algo del mundo se conserva y deja huellas estables en la representacion.

**Estructural** porque lo que importa no son entidades internas literales, sino relaciones armonicas naturales: razones de frecuencia, proporciones entre armonicos, regularidades del oscilador, y la geometria que estas inducen.

**Modesto** porque la afirmacion es restringida: si la armonia natural mejora de manera causal, robusta y transferible la alineacion entre modalidades de un mismo fenomeno, eso es evidencia indirecta de que captura un aspecto real del fenomeno. No es una prueba ontologica absoluta.

**Experimental** porque la validez no viene de meditacion filosofica, sino de intervencion, control, comparabilidad, replicacion, convergencia y prediccion.

---

## 7. Lo que distingue a Phideus de ingenieria de ML

Phideus contiene ingenieria de ML real: encoders, losses, schedulers, projection heads, mecanismos de inyeccion, evaluacion de retrieval, tooling de training y de HPC. Eso forma parte esencial del trabajo.

Pero la diferencia con un proyecto puramente de ML es la pregunta que organiza todo lo demas.

En un proyecto de ML, la pregunta tipica es: como subo la metrica, que arquitectura gana, que tweak funciona mejor.

En Phideus, la pregunta es: la armonia natural del fenomeno fisico funciona como organizador privilegiado de la informacion cross-modal? Esa ventaja es especifica de relaciones armonicas naturales o aparece con cualquier descriptor auxiliar? Reaparece en otro dominio?

La red es el medio. La hipotesis sustantiva es sobre la armonia natural. El nivel ingenieril esta subordinado a esa hipotesis.

---

## 8. De Escalon 1 a Escalon 2: refinamiento de la hipotesis

### 8.1 Lo que establecio Escalon 1

Escalon 1 (Audio ↔ MIDI, MAESTRO) establecio fundamentos solidos:

- la inyeccion de descriptores mejora retrieval de manera causal (d4a4=84.1% ±2.3pp, +9.4pp sobre baseline sin descriptor, Test 02 causal);
- esa mejora es robusta a traves de seeds (5 seeds, CI=[82.6%, 88.4%]);
- reorganiza la geometria del espacio de embeddings (+82% CKA vs baseline);
- y no aparece como mayor decodificabilidad local, sino como reorganizacion global.

En terminos del programa: Escalon 1 valido la mecanica de intervencion descriptor-guided y demostro que ciertas estructuras auxiliares pueden reorganizar de manera causal un espacio cross-modal.

### 8.2 Lo que Escalon 2 precisa

Escalon 1 uso `A4` (forma espectral) y `D4` (intervalos MIDI en log2/semitonos). Estos descriptores demostraron la mecanica, pero no testean directamente la tesis sobre armonia natural: `A4` es espectral-generico y `D4` opera sobre representaciones ya mediadas perceptualmente.

Para que la Harmonic Information Theory reclame algo mas alla de "ciertos descriptores ayudan", la hipotesis necesita enfrentarse con descriptores derivados de la fisica del fenomeno, no de codificaciones culturales o logaritmicas del mismo.

Esta es la exigencia que Escalon 2 (Speech ↔ EGG) incorpora de entrada: los descriptores primarios se derivan de invariantes fisicos del oscilador glotal. Los descriptores perceptuales existen como controles comparativos, no como defaults.

---

## 9. Armonia natural vs armonia perceptual

La distincion entre armonia natural y armonia perceptual, introducida en la seccion 1, tiene consecuencias operativas concretas que conviene desplegar.

### 9.1 Armonia perceptual

La armonia mediada por el oido humano, escalas logaritmicas, semitonos, temperamento igual, codificaciones culturales. Es una descripcion valida de como ciertos organismos procesan la estructura armonica, pero no es el objeto de investigacion de Phideus.

### 9.2 Armonia natural

La armonia como estructura fisica del fenomeno vibratorio: razones lineales, multiplos enteros, serie armonica, regularidades del oscilador, proporciones preservadas por el sistema fisico antes de toda mediacion perceptual. Es el objeto central de investigacion de Phideus.

### 9.3 La consecuencia metodologica

A partir de Escalon 2:

- `log2` no es un default inocente; es una hipotesis perceptual que debe hacerse explicita;
- un descriptor perceptual puede seguir existiendo como control comparativo;
- y los descriptores derivados de la fisica del fenomeno ocupan el lugar principal.

---

## 10. Taxonomia de descriptores

Para mantener claridad sobre que testea cada experimento, el programa distingue cuatro familias.

### Familia A: dinamica temporal del oscilador

Mide cambios locales de F0, cambios de periodo, regularidad del ciclo, fortaleza de voicing. Testea si la evolucion temporal local del oscilador contiene invariantes utiles para la alineacion.

Ejemplos: `F0[t] / F0[t-1]`, `period[t] / period[t-1]`, jitter local.

### Familia B: estructura armonica natural intra-frame

Mide relaciones entre armonicos, concentracion armonica, proximidad a series de multiplos enteros. Testea si la estructura armonica fisica del fenomeno genera una huella privilegiada para la alineacion.

Ejemplos: `H2/H1`, `H3/H1`, `H4/H1`, concentracion de energia armonica.

Esta es la familia mas directamente ligada a la tesis fuerte de Phideus.

### Familia C: controles no-ratio (espectrales genericos)

Mide energia por bandas, envelopes, forma espectral, dinamica espectral. Testea si cualquier estructura auxiliar ayuda, o si la ventaja es especifica de familias relacionales.

### Familia D: descriptores perceptuales o logaritmicos

Versiones logaritmicas o perceptualmente mediadas de estructuras que tambien pueden definirse en coordenadas fisicas. Testea si la ganancia depende de la forma natural de la variable, o si cualquier remapeo monotono basta.

### Por que importa esta taxonomia

Porque si Familia A y Familia B mejoran retrieval, no dicen lo mismo. Una dice que la dinamica del oscilador es un invariante util. La otra dice que la serie armonica fisica deja una huella estructural privilegiada. Ambas son valiosas, pero testean hipotesis distintas.

Y si Familia C tambien mejora, la interpretacion cambia radicalmente: el efecto no seria especifico de la armonia, sino generico de informacion auxiliar. De ahi la necesidad de controles no-ratio en todo experimento.

---

## 11. Harmonic Information Theory: las dos formulaciones

La Harmonic Information Theory es el programa teorico que Phideus investiga experimentalmente. No es una teoria de musica, de tonalidad ni de percepcion humana. Es una hipotesis sobre la organizacion informacional de fenomenos fisicos vibratorios:

> la armonia natural — razones y proporciones fisicas entre frecuencias, armonicos y ciclos del oscilador — funciona como organizador privilegiado de la informacion porque preserva estructura a traves de transformaciones, sensores y modalidades.

Bajo esta lectura, la armonia no es una propiedad del oido ni una codificacion cultural. Es una forma de baja entropia estructural que ciertos fenomenos vibratorios exhiben y que ciertos sistemas de representacion pueden explotar.

### 11.1 La formulacion operativa

> las razones armonicas naturales son variables privilegiadas para representar y alinear mediciones heterogeneas de ciertos procesos fisicos.

Escalon 1 establecio la mecanica. Escalon 2 enfrenta esta formulacion con descriptores derivados de la fisica del oscilador, contra controles espectrales y perceptuales.

### 11.2 La formulacion fuerte

> las razones armonicas naturales expresan una estructura informacional profunda de ciertos fenomenos, y esa estructura puede hacerse visible experimentalmente a traves de modelos aprendidos.

La validacion de esta formulacion requiere convergencia entre dominios (la Triplescaloneta: Audio↔MIDI, Speech↔EGG, ECG↔PPG).

---

## 12. Que hace fuerte una inferencia en Phideus

No toda mejora metrica vale epistemologicamente lo mismo. Una inferencia gana peso cuando hay convergencia entre varios tipos de soporte.

### 12.1 Comparabilidad estricta

Misma arquitectura, misma receta, mismo schedule, mismo protocolo de evaluacion. Sin eso, la inferencia causal se debilita.

### 12.2 Controles de contenido

`zero`, `random`, `shuffled`, parameter-matched. La pregunta no es "subio?" sino "subio por el contenido del descriptor?"

### 12.3 Robustez estadistica

Multiples seeds, intervalos de confianza, estabilidad del efecto.

### 12.4 Robustez entre arquitecturas

Si el efecto existe solo en una arquitectura, es demasiado facil leerlo como artefacto instrumental.

### 12.5 Robustez entre dominios

La razon de ser de la Triplescaloneta. Si una estructura aparece en Audio↔MIDI, reaparece en Speech↔EGG, y luego en ECG↔PPG, la lectura "es un truco del dataset" se vuelve progresivamente insostenible.

### 12.6 Capacidad predictiva

La teoria gana fuerza real cuando no solo explica lo ya visto, sino que anticipa donde deberia aparecer o desaparecer el efecto. Ese es el momento donde un programa exploratorio empieza a convertirse en teoria.

---

## 13. Que investiga Phideus: invariantes, no coordenadas

Lo que importa no es que una neurona "represente la quinta justa", que una capa "sea la armonia natural", ni que una direccion del embedding "sea la esencia del fenomeno".

Lo que importa es:

- que relaciones se preservan;
- que distancias cambian;
- que invariantes resisten al cambio de sensor;
- que estructura sobrevive a la ablacion;
- y que reorganizacion reaparece entre dominios.

> Phideus no investiga el significado ontologico de coordenadas internas; investiga la persistencia experimental de relaciones armonicas naturales a traves de sensores, dominios y arquitecturas.

---

## 14. Arquitectura subordinada a epistemologia

El marco epistemologico afecta las decisiones arquitectonicas.

**Primera fase: simplicidad y trazabilidad.** Encoders sencillos, simetricos y controlables. Esto no es una limitacion: es un requisito epistemologico. Permite aislar efectos, mantener interpretabilidad, y evitar que un foundation model opaque la hipotesis sustantiva.

**Segunda fase: adecuacion controlada.** Una vez establecido el baseline, se introduce asimetria o complejidad como evolucion metodologica, no como salto oportunista hacia mas poder.

**Tercera fase: benchmarks fuertes.** Foundation models o encoders potentes entran como stress test del programa. No deben definir desde el inicio la forma de la pregunta.

> La arquitectura debe servir a la epistemologia, no sustituirla.

---

## 15. La cadena de inferencia

Para mantener disciplina interpretativa, el programa aplica una cadena explicita:

1. **Observacion**: el dato bruto o resultado medido.
2. **Hipotesis**: una explicacion posible de ese dato.
3. **Inferencia**: la conclusion que los datos efectivamente autorizan.

### Ejemplo: inferencia parcial

Observacion: un descriptor mejora retrieval en un dominio.
Hipotesis: la estructura del descriptor captura una invariante del fenomeno.
Inferencia autorizada: hay evidencia a favor, pero la convergencia entre dominios todavia no se ha establecido. Hace falta replicacion.

### Ejemplo: inferencia convergente

Observacion: una familia de descriptores mejora retrieval de forma causal, robusta, transferible y comparativamente especifica frente a controles no-ratio.
Hipotesis: la estructura relacional que esos descriptores codifican es una invariante del fenomeno.
Inferencia autorizada: existe evidencia indirecta razonable a favor de que esa estructura captura un aspecto real del fenomeno medido.

La diferencia entre ambos casos es la convergencia de evidencia, no la magnitud de un resultado individual.

---

## 16. Lo que seria un exito real del programa

Un resultado valioso seria subir retrieval, construir mejores descriptores, o encontrar una arquitectura mas efectiva. Eso ya seria una contribucion real.

Pero el exito propio del programa seria mostrar algo mas:

1. una familia de relaciones fisicamente naturales mejora de forma causal la alineacion entre modalidades;
2. lo hace de manera no reducible a controles genericos no-ratio;
3. reaparece en dominios distintos;
4. y obliga a leer las razones no como una conveniencia del modelo, sino como una estructura informacional realmente privilegiada.

Ese seria el punto donde la Harmonic Information Theory deja de funcionar como intuicion inspiradora y se convierte en programa empirico serio.

---

## 17. Maximas epistemologicas

### Maxima 1

El objeto de estudio no es la red neuronal, sino las invariantes estructurales que la red ayuda a revelar.

### Maxima 2

Ninguna mejora aislada en una arquitectura unica constituye descubrimiento sobre el mundo. La fuerza proviene de la convergencia.

### Maxima 3

Toda inferencia fuerte necesita convergencia de evidencia: controles, comparabilidad, robustez y transferencia de dominio.

### Maxima 4

Los descriptores primarios se derivan de invariantes fisicos del fenomeno medido. Los descriptores perceptuales funcionan como controles.

### Maxima 5

No confundir dinamica temporal del oscilador, estructura armonica intra-frame, y controles espectrales no-ratio. Cada familia testea hipotesis distintas.

### Maxima 6

La arquitectura permanece subordinada a la claridad epistemologica del experimento.

### Maxima 7

El programa prefiere inferencias estructurales sobre coordenadas, relaciones sobre activaciones, y convergencia experimental sobre intuiciones no testeadas.

---

## 18. Formula de cierre

> Phideus propone que los modelos aprendidos sobre mediciones reales pueden funcionar como instrumentos epistemologicos para explorar estructuras informacionales del mundo fisico, no porque sus embeddings sean espejos ontologicos de la realidad, sino porque, al estar entrenados sobre datos fisicamente anclados y ser intervenidos de manera controlada, permiten detectar que relaciones son invariantes, causalmente eficaces y transferibles entre modalidades distintas de un mismo fenomeno.

Y en la etapa actual:

> la pregunta ya no es si un descriptor ayuda, sino si las estructuras fisicamente naturales del fenomeno — las razones y regularidades armonicas naturales — constituyen variables privilegiadas para la alineacion cross-modal, por encima de controles perceptuales y de descriptores genericos no-ratio.

Phideus es un programa experimental de realismo estructural modesto que usa modelos aprendidos como instrumentos para investigar si ciertas estructuras relacionales — especialmente las asociadas a la armonia natural — son invariantes informacionales del mundo fisico.
