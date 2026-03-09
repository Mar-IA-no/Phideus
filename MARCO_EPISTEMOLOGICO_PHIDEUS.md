# Marco Epistemologico de Phideus

> Documento de posicion general del programa.
> No reemplaza roadmaps, reportes tecnicos ni notas de trabajo:
> fija, en un lenguaje mas estable, que clase de conocimiento intenta producir Phideus,
> mediante que metodo, con que aspiraciones, y con que limites de validez.

---

## 1. Proposito

Phideus no quiere limitarse a construir pipelines que mejoren una metrica.
Tampoco quiere saltar de una mejora en embeddings a una metafisica improvisada del mundo.

El programa propone algo mas preciso:

> usar sistemas aprendidos sobre mediciones reales de fenomenos fisicos como instrumentos experimentales para investigar que estructuras informacionales son privilegiadas para la alineacion cross-modal.

Dicho de otra manera:

- no se toma a la red como el fenomeno;
- no se toma al embedding como una ontologia;
- no se toma a una mejora de retrieval como prueba directa de una ley de la naturaleza;
- pero tampoco se reduce todo a "ingenieria de machine learning".

La apuesta de Phideus es intermedia y, por eso mismo, mas interesante:

> si una estructura relacional reaparece de manera causal, robusta y transferible a traves de sensores, dominios y arquitecturas distintas, entonces esa estructura deja de parecer un truco del modelo y empieza a parecer una pista real sobre la organizacion de ciertos fenomenos naturales.

Ese es el espacio epistemologico de Phideus.

---

## 2. La pregunta de fondo

La pregunta profunda del proyecto no es:

- que encoder rinde mas,
- que loss converge mas rapido,
- que inyeccion sube mejor el Recall@10,
- o que arquitectura da el score mas alto.

Esas preguntas existen, pero son subordinadas.

La pregunta de fondo es otra:

> existen estructuras relacionales privilegiadas para organizar informacion entre modalidades distintas de un mismo fenomeno?

Y, dentro de esa pregunta general, Phideus hace una apuesta mas especifica:

> las razones, proporciones y relaciones armonicas pueden constituir un lenguaje estructural especialmente apto para reducir entropia, preservar invariantes y facilitar alineacion cross-modal.

En su formulacion mas ambiciosa, esto se conecta con la Harmonic Information Theory.

Pero esa conexion necesita ser cuidadosamente disciplinada.

Porque hay una diferencia enorme entre decir:

> "esta arquitectura mejora cuando le inyecto un descriptor"

y decir:

> "descubrimos una regularidad estructural del mundo".

El metodo de Phideus vive precisamente en ese pasaje, y por eso necesita un marco epistemologico explicito.

---

## 3. Que NO dice Phideus

Antes de decir que afirma el programa, conviene dejar muy claro lo que no afirma.

Phideus no sostiene:

1. que una red neuronal sea un espejo ontologico del mundo;
2. que una coordenada de un embedding tenga significado fisico estable por si misma;
3. que toda mejora interna de un modelo sea extrapolable sin mas al mundo natural;
4. que exito en una tarea equivalga automaticamente a explicacion del fenomeno;
5. que lo que pasa dentro de una capa del modelo describa de manera literal como funciona la naturaleza;
6. que una mejora en una sola arquitectura, un solo dataset o una sola receta cierre una hipotesis de frontera.

Estas negaciones no son decorativas.
Son la barrera que separa a Phideus de una lectura ingenua o mitologica de sus propios resultados.

---

## 4. Que SI dice Phideus

La afirmacion positiva del programa es mas sobria y mas fuerte:

> modelos aprendidos sobre mediciones reales, observadas desde modalidades distintas, pueden funcionar como instrumentos de intervencion y lectura para detectar que estructuras relacionales son privilegiadas para alinear, comprimir y transferir informacion entre esas modalidades.

Eso implica varias tesis concretas.

### 4.1 Tesis instrumental

La red no es el objeto final de estudio.
Es un instrumento.

Asi como un telescopio no es el planeta y un espectrometro no es el atomo, un encoder no es el fenomeno fisico que genero los datos.
Pero eso no le quita valor epistemologico al instrumento.
Al contrario: lo vuelve util.

### 4.2 Tesis estructural

Lo que interesa no son activaciones aisladas, sino relaciones:

- vecindades,
- alineaciones,
- invariantes,
- proporciones,
- reorganizaciones geometricas,
- transferencias entre modalidades,
- robustez a cambios de sensor y de representacion.

Phideus no es, en su mejor version, una teoria de neuronas o capas.
Es una teoria experimental de invariantes relacionales.

### 4.3 Tesis de evidencia indirecta

Cuando una estructura relacional mejora de manera causal y repetible la alineacion entre sensores distintos del mismo fenomeno, eso constituye evidencia indirecta de que dicha estructura captura algo real del fenomeno medido.

No es una prueba metafisica.
Es evidencia indirecta, pero cientificamente seria.

---

## 5. El puente atomos-bits-atomos

Una de las intuiciones centrales del proyecto puede formularse asi:

```
Mundo de atomos  ->  digitalizacion  ->  mundo de bits  ->  modelo  ->  embedding
     ^                                                                    |
     |--------------------------------------------------------------------|
                          inferencia estructural
```

La legitimidad epistemologica del programa depende de que este puente no sea arbitrario.

### 5.1 Primer tramo: atomos -> bits

En los experimentos de Phideus no trabajamos sobre simbolos puros ni sobre ficciones numericas desligadas del mundo.
Trabajamos sobre mediciones digitales de fenomenos fisicos reales.

Ejemplos:

- un piano real fue ejecutado, grabado y registrado tambien como MIDI;
- una voz real fue emitida por un oscilador glotal, captada por un microfono y por un electroglotografo;
- mas adelante, un fenomeno fisiologico puede ser captado por ECG y PPG.

En todos esos casos hay una cadena de transduccion:

- el fenomeno fisico existe,
- un sensor lo captura,
- la senal se digitaliza,
- el modelo recibe esa medicion.

La validez del programa depende de que esa cadena preserve la estructura que interesa.

### 5.2 Segundo tramo: bits -> modelo

El modelo no "ve" el mundo directo.
Ve una medicion digitalizada.

Pero esa medicion no es arbitraria.
Si los descriptores se construyen sobre regularidades realmente preservadas por la cadena de medicion, entonces el modelo puede funcionar como una superficie de prueba:

- lo perturbamos con un descriptor,
- vemos si reorganiza la geometria,
- vemos si mejora retrieval,
- vemos si preserva o no ciertas invariantes,
- y observamos que estructuras responden de forma robusta.

### 5.3 Tercer tramo: modelo -> atomos

Este es el tramo delicado.
Es donde la inferencia puede volverse legitima o caer en fantasias.

La vuelta de bits a atomos no se justifica porque "la red lo dijo".
Se justifica si se cumplen varias condiciones:

1. el fenomeno estaba fisicamente anclado desde el inicio;
2. la cadena de digitalizacion preserva la estructura relevante;
3. la intervencion sobre el modelo es controlada;
4. el efecto observado es robusto;
5. el efecto reaparece en mas de un regimen;
6. idealmente, la teoria genera predicciones nuevas.

Si esas condiciones se cumplen, el pasaje de vuelta deja de ser una simple fantasia interpretativa y se convierte en inferencia estructural razonable.

---

## 6. La mejor analogia: telescopio, espectrometro y laboratorio

En las discusiones del proyecto aparecieron varias metaforas que vale la pena retener.
No son adornos literarios.
Ayudan a fijar bien la posicion epistemologica.

### 6.1 La red no es un espejo; se parece mas a un telescopio

Cuando Galileo vio puntos luminosos cerca de Jupiter, la pregunta inmediata fue:

> son lunas reales o aberraciones del telescopio?

La respuesta no vino de una intuicion mistica sobre la verdad del instrumento.
Vino de:

- replicacion,
- observacion repetida,
- convergencia con otros instrumentos,
- y capacidad predictiva.

Con Phideus pasa algo semejante.
Una sola mejora en una sola arquitectura no prueba nada grande.
Pero si la misma familia de relaciones reaparece con:

- varios arms,
- varias seeds,
- varios tests,
- varias modalidades,
- y mas adelante varios escalones,

entonces la hipotesis de artefacto del instrumento se debilita.

La red, en esta metafora, no "revela directamente" la naturaleza.
Pero permite verla bajo ciertas condiciones.

### 6.2 La analogia mas fina: la red se parece mas a un espectrometro

Quizas la mejor analogia no sea el telescopio, sino el espectrometro.

Un espectrometro:

1. recibe una senal del mundo;
2. la descompone segun una estructura;
3. devuelve una representacion interpretable;
4. y a partir de esa representacion inferimos propiedades del fenomeno fuente.

Phideus hace algo análogo:

- la senal fisica entra al sistema,
- el modelo la transforma,
- el embedding y las metricas son el "espectro" que observamos,
- y los descriptores funcionan como filtros o perturbaciones que cambian lo que el instrumento deja ver.

Cuando un descriptor reorganiza el espacio y mejora la alineacion, no estamos diciendo que "la naturaleza vive en el embedding".
Estamos diciendo que, al pasar la senal por este instrumento, cierta estructura deja una huella observable y manipulable.

### 6.3 La red como laboratorio activo

Todavia hay una tercera metafora, mas fuerte:

la red no es solo un instrumento pasivo de observacion;
es tambien un laboratorio de intervencion.

Podemos:

- introducir una estructura,
- quitarla,
- degradarla,
- compararla contra controles,
- ver donde falla,
- ver donde se sostiene.

Eso hace que la red no sea solo un visor.
Sea un laboratorio donde ciertas hipotesis estructurales pueden ponerse a prueba en condiciones controladas.

---

## 7. La posicion filosofica adecuada: realismo estructural modesto

Si hubiera que ponerle un nombre filosofico a la posicion de Phideus, el mas apropiado seria algo cercano a:

> realismo estructural modesto, operacional y experimental.

### 7.1 Por que "realismo"

Porque el proyecto no dice que todo sea construccion del modelo.
Apuesta a que algo del mundo se conserva y puede dejar huellas estables en la representacion.

### 7.2 Por que "estructural"

Porque lo que importa no son entidades internas literales, sino relaciones:

- razones,
- proporciones,
- armonicos,
- regularidades del oscilador,
- vecindades,
- geometria,
- invariantes.

### 7.3 Por que "modesto"

Porque no se afirma que el embedding sea la realidad.
Se afirma algo mas restringido:

> si cierta estructura relacional mejora de manera causal, robusta y transferible la alineacion entre modalidades de un mismo fenomeno, entonces eso es evidencia indirecta de que esa estructura captura un aspecto real del fenomeno.

### 7.4 Por que "experimental"

Porque la validez no viene de una meditacion filosofica abstracta, sino de:

- intervencion,
- control,
- comparabilidad,
- replicacion,
- convergencia entre tests,
- y eventualmente prediccion.

---

## 8. La diferencia entre ingenieria de ML y programa epistemologico

Es importante ser brutalmente honestos aca.
Phideus contiene una parte real de ingenieria de ML.

Trabaja con:

- encoders,
- losses,
- schedulers,
- projection heads,
- mecanismos de inyeccion,
- evaluacion de retrieval,
- tooling de training y de HPC.

Eso no hay que negarlo.

Pero tampoco alcanza con decir "es solo ML".

La diferencia crucial es esta:

### 8.1 En un proyecto puramente de ML

La pregunta tipica es:

- como subo la metrica?
- que arquitectura gana?
- que tweak funciona mejor?

### 8.2 En Phideus

La pregunta relevante es:

- que clase de estructura hace posible la alineacion cross-modal?
- que cambia cuando intervengo con una familia de descriptores?
- esa mejora depende del contenido del descriptor o de un artefacto de capacidad?
- reaparece el efecto en otro dominio?
- la estructura parece capturar algo del fenomeno o solo de la arquitectura?

En otras palabras:

> la red es medio; la hipotesis sustantiva es estructural.

Eso no elimina el nivel ingenieril.
Lo subordina.

---

## 9. La leccion de Escalon 1: validacion de la mecanica, no cierre de la ontologia

Escalon 1 fue crucial, pero su significado epistemologico debe formularse con precision.

### 9.1 Lo que Escalon 1 SI mostro

Escalon 1 mostro que:

- la inyeccion de descriptores puede mejorar retrieval causalmente;
- esa mejora puede ser robusta;
- puede reorganizar la geometria del espacio de embeddings;
- y no necesariamente aparece como mayor decodificabilidad local simple.

Dicho mas brevemente:

> Escalon 1 valido la mecanica de intervencion descriptor-guided y la idea de que ciertas estructuras auxiliares pueden reorganizar de manera causal un espacio cross-modal.

Eso ya es muchisimo.

### 9.2 Lo que Escalon 1 NO mostro limpiamente

Escalon 1 no mostro todavia, de manera limpia, la tesis fuerte sobre armonia natural.

Porque:

- `A4` es un descriptor espectral, no de razones naturales;
- `D4` trabaja sobre una representacion MIDI cuantizada y musicalmente mediada;
- y buena parte del trabajo de Escalon 1 puede leerse todavia como validacion de una mecanica de inyeccion, no como prueba definitiva de una ontologia armonica.

Esta distincion es importante.

No debilita Escalon 1.
Lo ubica en su lugar correcto:

> Escalon 1 demostro que el metodo puede detectar y explotar estructura; no demostro aun, en su forma mas fuerte, que esa estructura sea especificamente la armonia natural fisica que interesa a la Harmonic Information Theory.

---

## 10. La rectificacion epistemologica de Escalon 2

Escalon 2 obliga a una explicitacion nueva.

Si Phideus quiere sostener que su tesis fuerte concierne a la armonia natural y no simplemente a descriptores utiles, entonces a partir de Escalon 2 debe quedar fijada una nueva directiva:

> los descriptores primarios deben derivarse preferentemente de invariantes fisicos del fenomeno medido; los descriptores perceptuales o logaritmicos deben pasar a ser brazos comparativos explicitos, no el default silencioso.

Esta directiva tiene consecuencias profundas.

### 10.1 Ya no alcanza con preguntar "un descriptor ayuda?"

Esa pregunta era aceptable como primera fase.
Ahora hace falta preguntar:

- ayuda un descriptor fisicamente natural?
- ayuda mas que un control no-ratio?
- ayuda mas que una codificacion perceptual/logaritmica?
- ayuda por dinamica temporal?
- ayuda por estructura armónica intra-frame?
- o ayuda solo por agregar informacion generica?

### 10.2 La nueva exigencia

Escalon 2 debe separar explicitamente al menos tres hipotesis:

1. **Dinamica temporal del oscilador**
   Relaciones entre ciclos o entre F0s sucesivos.

2. **Estructura armonica natural intra-frame**
   Serie armonica, relaciones entre armonicos, concentracion armonica.

3. **Controles no-ratio**
   Formas espectrales, envelopes, energia por bandas, descriptores genericos.

Sin esa separacion, el programa corre el riesgo de seguir encontrando cosas interesantes sin saber con suficiente precision que esta encontrando.

---

## 11. Armonia natural vs armonia perceptual

Esta es, probablemente, la distincion epistemologica nueva mas importante del proyecto.

### 11.1 Armonia perceptual

Es la armonia mediada por:

- oido humano,
- escalas logaritmicas,
- semitonos,
- temperamento igual,
- codificaciones culturales o musicales ya discretizadas.

No es falsa ni inutil.
Pero no es la tesis fuerte de Phideus.

### 11.2 Armonia natural

Es la armonia pensada como estructura fisica del fenomeno vibratorio:

- razones lineales,
- multiples enteros,
- serie armonica,
- regularidades del oscilador,
- proporciones preservadas por el sistema fisico antes de la mediacion perceptual humana.

Si la Harmonic Information Theory quiere reclamar algo fuerte, tiene que ubicarse ahi.

### 11.3 La consecuencia metodologica

Por eso, a partir de Escalon 2:

- `log2` no puede seguir siendo asumido como default inocente;
- un descriptor perceptual puede seguir existiendo, pero como control;
- y los descriptores fisicamente naturales deben pasar al primer plano.

---

## 12. Cuidado: no toda relacion es lo mismo

La nueva directiva tambien obliga a no usar la palabra "ratio" de forma vaga.

Hay, al menos, dos familias que hay que distinguir:

### 12.1 Relaciones temporales del oscilador

Ejemplos:

- `F0[t] / F0[t-1]`
- `period[t] / period[t-1]`
- regularidad del periodo
- jitter local

Esto mide dinamica temporal del oscilador.

Es importante.
Es natural.
Puede ser central en Speech↔EGG.

Pero no es lo mismo que medir la serie armonica.

### 12.2 Estructura armonica intra-frame

Ejemplos:

- `H2/H1`
- `H3/H1`
- `H4/H1`
- concentracion de energia armonica
- desvio respecto a multiplos enteros de `F0`

Esto si apunta directamente a la estructura armonica natural del fenomeno.

### 12.3 Por que importa distinguirlas

Porque si ambas mejoran retrieval, no significan lo mismo.

Una podria decir:

> la dinamica del oscilador es un invariante util.

La otra podria decir:

> la serie armonica fisica deja una huella estructural privilegiada.

Ambas son valiosas.
Pero testean hipotesis distintas.

---

## 13. Taxonomia epistemica de descriptores

Para evitar confusiones futuras, conviene fijar una taxonomia general.

### Familia A: descriptores de dinamica temporal del oscilador

Miden:

- cambios locales de F0,
- cambios de periodo,
- regularidad del ciclo,
- fortaleza de voicing.

Prueban:

- si la evolucion temporal local del oscilador contiene invariantes utiles para la alineacion.

No prueban por si solos:

- la tesis fuerte de armonia natural intra-frame.

### Familia B: descriptores de estructura armonica natural intra-frame

Miden:

- relaciones entre armonicos,
- concentracion armonica,
- proximidad a series de multiplos enteros.

Prueban:

- si la estructura armonica fisica del fenomeno genera una huella privilegiada para la alineacion.

Son la familia mas directamente ligada a la tesis fuerte de Phideus.

### Familia C: descriptores no-ratio o controles espectrales

Miden:

- energia por bandas,
- envelopes,
- forma espectral,
- dinamica espectral.

Prueban:

- si cualquier estructura auxiliar ayuda,
- o si la ventaja parece especifica de familias mas relacionales.

### Familia D: descriptores perceptuales o logaritmicos

Miden:

- versiones logaritmicas o perceptualmente mediadas de estructuras que tambien pueden definirse en coordenadas fisicas.

Prueban:

- si la ganancia depende de la forma natural de la variable,
- o si cualquier remapeo relacional monotono basta.

Estas variantes no deben desaparecer.
Pero ya no deben ocupar el lugar epistemologicamente principal.

---

## 14. La relacion con la Harmonic Information Theory

Ahora ya puede formularse con mas claridad la conexion con HIT.

La Harmonic Information Theory, tal como la entiende Phideus, no deberia reducirse a una teoria de musica, de tonalidad ni de percepcion humana.

Deberia entenderse como una hipotesis mas general:

> ciertas razones y proporciones fisicas funcionan como organizadores privilegiados de la informacion porque preservan estructura a traves de transformaciones, sensores y modalidades.

Bajo esta lectura:

- la armonia no es solo una propiedad del oido;
- no es solo una codificacion cultural de intervalos;
- no es solo una conveniencia descriptiva;
- puede ser una forma de baja entropia estructural que ciertos fenomenos vibratorios exhiben y que ciertos sistemas de representacion pueden explotar.

### 14.1 La version prudente

La version prudente de HIT en Phideus seria:

> las razones pueden ser variables privilegiadas para representar y alinear mediciones heterogeneas de ciertos procesos fisicos.

### 14.2 La version fuerte

La version fuerte seria:

> las razones armonicas expresan una estructura informacional profunda de ciertos fenomenos naturales, y esa estructura puede hacerse visible experimentalmente a traves de modelos aprendidos.

Hoy, el programa puede defender seriamente la primera formulacion.
La segunda todavia funciona como horizonte de investigacion.

---

## 15. Que hace fuerte una inferencia en Phideus

No toda mejora metrica vale epistemologicamente lo mismo.

Una inferencia gana peso en este marco cuando hay convergencia entre varios tipos de soporte.

### 15.1 Comparabilidad estricta

- misma arquitectura;
- misma receta;
- mismo schedule;
- mismo protocolo de evaluacion.

Sin eso, la inferencia causal se debilita.

### 15.2 Controles de contenido

- `zero`
- `random`
- `shuffled`
- parameter-matched

La pregunta correcta no es "subio?" sino "subio por el contenido del descriptor?"

### 15.3 Robustez estadistica

- multiples seeds;
- intervalos de confianza;
- estabilidad del efecto.

### 15.4 Robustez entre arquitecturas

Si el efecto vive solo en una arquitectura, es demasiado facil interpretarlo como artefacto instrumental.

### 15.5 Robustez entre dominios

Aca esta la razon de ser de la Triplescaloneta.

Si una estructura:

- aparece en Audio↔MIDI,
- reaparece en Speech↔EGG,
- y luego en ECG↔PPG,

la lectura "es un truco del dataset o del dominio" se vuelve cada vez menos plausible.

### 15.6 Capacidad predictiva

La teoria gana fuerza de verdad cuando:

- no solo explica lo ya visto,
- sino que anticipa donde deberia aparecer o desaparecer el efecto.

Ese es el momento donde un programa exploratorio empieza a convertirse en teoria seria.

---

## 16. La validez no vive en las coordenadas; vive en las invariantes

Una fuente clasica de error en programas como este es querer leer demasiado literalmente el espacio interno del modelo.

Phideus deberia evitar convertir eso en dogma.

Lo que importa no es:

- "esta neurona representa la quinta justa",
- "esta capa es la armonia natural",
- "esta direccion del embedding es la esencia del fenomeno".

Lo que importa es:

- que relaciones se preservan;
- que distancias cambian;
- que invariantes resisten al cambio de sensor;
- que estructura sobrevive a la ablacion;
- y que reorganizacion reaparece entre dominios.

En una formula:

> Phideus no investiga el significado ontologico de coordenadas internas; investiga la persistencia experimental de relaciones estructurales.

---

## 17. Consecuencia metodologica para la arquitectura

El marco epistemologico no solo afecta la interpretacion.
Afecta tambien las arquitecturas.

### 17.1 Primera fase: simplicidad y trazabilidad

Es correcto comenzar con encoders sencillos, simetricos y controlables.

Eso no es una concesion menor.
Es un requisito epistemologico:

- permite aislar efectos,
- mantiene interpretabilidad,
- y evita que un foundation model opaque la hipotesis sustantiva.

### 17.2 Segunda fase: adecuacion fisica controlada

Una vez establecido el baseline, puede volverse razonable introducir asimetria controlada en front-ends sensoriales.

Pero esa asimetria tiene que leerse como evolucion metodologica, no como salto oportunista a "mas poder".

### 17.3 Tercera fase: benchmarks fuertes

Foundation models o encoders mas potentes pueden entrar despues, como benchmark o stress test del programa.

No deben definir desde el inicio la forma de la pregunta.

En sintesis:

> la arquitectura debe servir a la epistemologia, no sustituirla.

---

## 18. La cadena de prudencia interpretativa

Para sostener la disciplina de frontera del proyecto, conviene fijar una cadena simple:

1. **Observacion**
   El dato bruto o el resultado medido.

2. **Hipotesis**
   Una explicacion posible de ese dato.

3. **Inferencia**
   La conclusion que efectivamente estamos autorizados a sostener.

Esta cadena tiene que aplicarse tambien en el plano filosofico.

Por ejemplo:

### Caso debil

Observacion:
- un descriptor mejora retrieval.

Hipotesis:
- la estructura del descriptor captura una invariante del fenomeno.

Inferencia legitima:
- todavia no sabemos; hace falta control adicional.

### Caso fuerte

Observacion:
- una familia de descriptores mejora retrieval de forma causal, robusta, transferible y comparativamente especifica.

Hipotesis:
- la estructura relacional que esos descriptores codifican es una invariante del fenomeno.

Inferencia legitima:
- existe evidencia indirecta razonable a favor de que esa estructura captura un aspecto real del fenomeno medido.

Asi es como Phideus debe protegerse de la sobrelectura.

---

## 19. Lo que seria un exito real del programa

Un exito menor seria:

- subir retrieval,
- construir mejores descriptores,
- o encontrar una arquitectura mas efectiva.

Eso ya seria un resultado real.

Pero no seria todavia el exito filosoficamente interesante.

El exito fuerte del programa seria mostrar algo como esto:

1. una familia de relaciones fisicamente naturales mejora de forma causal la alineacion entre modalidades;
2. lo hace de manera no reducible a controles genericos no-ratio;
3. reaparece en dominios distintos;
4. y obliga a leer las razones no como una conveniencia del modelo, sino como una estructura informacional realmente privilegiada.

Ese seria el punto donde la Harmonic Information Theory dejaria de funcionar solo como intuicion inspiradora y empezaria a convertirse en programa empirico serio.

---

## 20. Maximas epistemologicas del proyecto

Para dejar el marco en forma operativa, conviene fijar de manera explicita las maximas que ya no deberian quedar implicitas.

### Maxima 1

El objeto de estudio no es la red neuronal, sino las invariantes estructurales que la red puede ayudar a revelar.

### Maxima 2

Ninguna mejora aislada en una arquitectura unica debe leerse como descubrimiento del mundo.

### Maxima 3

Toda inferencia fuerte necesita convergencia de evidencia: controles, comparabilidad, robustez y, preferentemente, transferencia de dominio.

### Maxima 4

Los descriptores primarios deben derivarse preferentemente de invariantes fisicos del fenomeno medido.

### Maxima 5

Los descriptores perceptuales o logaritmicos pueden seguir existiendo, pero desde Escalon 2 en adelante deben leerse como controles comparativos, no como defaults silenciosos.

### Maxima 6

No se debe confundir:

- dinamica temporal del oscilador,
- estructura armonica intra-frame,
- y controles espectrales no-ratio.

Cada familia prueba hipotesis distintas.

### Maxima 7

La arquitectura debe permanecer subordinada a la claridad epistemologica del experimento.

### Maxima 8

El programa debe preferir inferencias estructurales sobre coordenadas, relaciones sobre activaciones, y convergencia experimental sobre intuiciones fuertes no testeadas.

---

## 21. Formula final

Si hubiera que condensar todo este marco en una sola formulacion, seria esta:

> Phideus propone que los modelos aprendidos sobre mediciones reales pueden funcionar como instrumentos epistemologicos para explorar estructuras informacionales del mundo fisico, no porque sus embeddings sean espejos ontologicos de la realidad, sino porque, al estar entrenados sobre datos fisicamente anclados y ser intervenidos de manera controlada, permiten detectar que relaciones son invariantes, causalmente eficaces y transferibles entre modalidades distintas de un mismo fenomeno.

Y, en la etapa actual del programa, esa formula necesita agregarse con una directiva mas precisa:

> a partir de Escalon 2, la pregunta ya no puede ser solamente si "un descriptor ayuda", sino si las estructuras fisicamente naturales del fenomeno - y en particular las razones y regularidades armonicas naturales - constituyen variables privilegiadas para la alineacion cross-modal, por encima de controles perceptuales y de descriptores genericos no-ratio.

---

## 22. Cierre

Phideus no deberia describirse como una ontologia de embeddings.
Tampoco como una simple bateria de trucos de machine learning.

Su mejor descripcion hoy es esta:

> un programa experimental de realismo estructural modesto que usa modelos aprendidos como telescopios, espectrometros y laboratorios para investigar si ciertas estructuras relacionales - especialmente las asociadas a la armonia natural - son invariantes informacionales del mundo fisico.

Si el programa fracasa, al menos habra mostrado con precision donde no estaba la estructura buscada.
Si tiene exito, su valor no sera solo haber mejorado metricas de retrieval.

Su valor sera haber mostrado que ciertos modelos, usados con disciplina metodologica, pueden servir como instrumentos legitimos para estudiar como la informacion se organiza en fenomenos fisicos reales.
