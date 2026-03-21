# Informe Historico de Representaciones de Ratios en Phideus

**Subtitulo**: de los primeros histogramas al giro epistemologico de la armonia natural  
**Fecha**: 2026-03-20
**Version**: 2.1

---

## Addendum operativo del corte

Este informe ya no puede cerrarse con la vieja formula "Escalon 1 funciono y ahora habria que generalizar". Ese lenguaje quedo atras.

Al corte actual:

- Escalon 1 ya esta cerrado como programa de validacion fuerte de la mecanica descriptor-guided sobre Audio<->MIDI.
- Gate 5B ya dejo evidencia causal, geometrica y multiseed suficientemente robusta como para tratar su lectura central como estable.
- Gate 6, Gate 7.1 y Gate 8 ya empezaron a mostrar donde se agota, donde no traduce y donde podria seguir habiendo cuello de botella en downstream, encoder y proyecciones.
- Gate 9 ya quedo formalizado como reapertura retrospectiva de la deuda natural-harmonic dentro de musica, y la revision `A10` abre una familia continua ontology-free para no confundir armonia natural con ontologia JI preimpuesta.
- Escalon 2 ya no es posibilidad teorica:
  - `S2-P0` y `S2-P1` ya cerraron,
  - `S2-P2-control` (`D0`) ya cerro con `S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`,
  - `S2-P2-main` por concatenación también ya cerró (`V4-lin=67.8%`, `H-series=59.8%`, `A4-16k=77.8%=D0`),
  - `S2-P2.5` ya dejó de ser solo fase activa: sus `6/6` celdas ya fueron leídas con bootstrap pareado y patrón preregistrado,
  - esa lectura no dio lift defendible sobre `D0`, sí dejó un caso claramente peor (`V4-lin + attn_bias`) y mantuvo visible la interacción descriptor × mecanismo,
  - y `S2-P2.5b` ya cerró `3/3` brazos `pca`, dejando formalmente cerrado ese primer null mecanístico.
- Escalon 2 ya no solo explicita una taxonomía de familias; también explicita una disciplina de lectura:
  - existe un preregistro interpretativo propio (`PREDICCIONES_EPISTEMOLOGICAS_P25.md`),
  - la comparación fuerte entre arms pasa a leerse con bootstrap pareado sobre `Delta`,
  - el salto concat→attention deja de ser solo intuición metodológica para convertirse en un cambio falsificable de régimen experimental,
  - y la fase `pca` ya cerrada deja explícito que un null de `concat/attn_bias/xattn/pca` no equivale todavía a clausura fuerte de teoría, aunque sí estrecha mucho el margen de ambigüedad y obliga a abrir `S2-P3`.
- Gate 10 ya aparece como consecuencia metodológica natural de ese corte: si `A7r/A9r/A10a-e` comprimieron sus resultados bajo `reverse cross-attention`, la siguiente pregunta histórica ya no es “qué descriptor agregar” sino “qué parte del resultado dependía del mecanismo”.
- Y ese barrido ya dejó un primer indicio empírico, aunque todavía parcial: con `8/9` arms en `e10`, `FiLM/pca` aparece por encima de `concat` y `attn_bias` en los descriptores visibles. La lectura fuerte sigue pendiente de `30ep`, pero la pregunta histórica ya dejó de ser hipotética.
- Y aparece además **Escalón 3**: el banco sintético nuevo donde la relación armónica pasa a ser visible en figuras de Lissajous y puede estudiarse con ground truth determinista. La expansión fisiológica `ECG↔PPG` pasa a ocupar el **Escalón 4**.
- Y ahora existe también una capa editorial larga que reordena retrospectivamente esta historia: el libro HIT en `manifiesto_HIT_Beancon_Phideus/` ya absorbió el nuevo problema `storage / retrieval`, fijó el `activation problem` como bisagra teórica y dejó más nítida la diferencia entre validación de mecánica y prueba de armonía natural.

La consecuencia de ese corte es importante:

> a partir de ahora, la historia de las representaciones de ratios en Phideus debe contarse distinguiendo con total claridad entre lo que fue validacion de mecanica y lo que pasa a ser prueba de armonia natural.

Y ahora hay una segunda consecuencia, todavía más fina:

> a partir de este punto, tampoco alcanza con distinguir descriptor de control; hay que distinguir descriptor, mecanismo de inyección y tipo de relación que se le pide organizar dentro del modelo.

Y ahora hay una tercera consecuencia, todavía más exigente:

> a partir de este punto, tampoco alcanza con registrar que una familia existe; hay que registrar bajo qué preregistro esa familia puede contar como evidencia a favor, en contra o como ambigüedad todavía abierta.

Eso explica por qué Gate 9 / `A10` entra ahora en esta historia sin desordenarla. No aparece para reescribir el cierre de Escalón 1 ni para competir con Speech↔EGG por el foco del programa. Aparece para volver más fina una deuda histórica: distinguir mejor entre descriptores que ya validaron mecánica y descriptores que todavía intentan tocar la tesis fuerte de armonía natural dentro del dominio musical.

Ese es el motivo de esta version 2.0 del informe.

---

## 0. Resumen ejecutivo

La historia de Phideus no es la historia lineal de un unico descriptor que se va perfeccionando. Es la historia de una pregunta que va cambiando de precision.

Al principio, la pregunta era casi brutalmente simple:

> como represento una relacion de frecuencias si lo unico que tengo es una onda?

Despues, la pregunta se volvio mas fina:

> como hago para que esa representacion no sea solo una estadistica bonita, sino una estructura util para un sistema aprendido?

Mas tarde, la pregunta cambio otra vez:

> como inyecto esa estructura dentro de una arquitectura cross-modal para que altere de forma causal la geometria del embedding?

Y hoy, con Escalon 2 abierto, la pregunta vuelve a cambiar:

> que parte de todo lo anterior realmente toca la tesis fuerte de Phideus sobre armonia natural y que parte fue, mas bien, una validacion de la mecanica general de intervencion descriptor-guided?

La tesis historica de este informe es la siguiente:

1. Las primeras generaciones del proyecto resolvieron el problema de la **representacion densa** de ratios.
2. El revisionismo UOEMD resolvio negativamente el problema de la **sparsificacion excesiva** y del matching exacto como camino principal.
3. Escalon 1 resolvio afirmativamente el problema de la **inyeccion causal de descriptores** y de la **reorganizacion geometrica**.
4. Escalon 2 abre, por primera vez de manera no ambigua, el problema de la **armonia natural** como directiva epistemologica primaria.

Esta ultima distincion es nueva y tiene que reordenar la lectura de todo lo anterior.

---

## 1. El hilo fundacional: Phideus nunca fue solo una idea de ML

Desde sus versiones mas tempranas, Phideus no se penso como un proyecto cuyo objetivo principal fuera optimizar una metrica cualquiera. Siempre hubo una intuicion mas fuerte debajo de la implementacion:

> las relaciones, las proporciones y las razones de frecuencia podrian ser una forma privilegiada de informacion.

Eso se conectaba con varias fuentes:

- intuiciones sobre proporciones naturales;
- la idea de que los intervalos son adimensionales y transferibles;
- observaciones sobre organizacion sonora natural;
- y, mas adelante, la aspiracion de una Harmonic Information Theory.

Pero esa intuicion convivio desde el principio con un problema dificil:

> una intuicion filosofica no sirve experimentalmente hasta que encuentra una forma de representacion operativa.

Ese fue el primer gran drama del proyecto.

---

## 2. Primera epoca: de la inspiracion musical a la disciplina frecuencial

### 2.1 El impulso inicial

La primera imagen de Phideus tenia algo de manifiesto: una inteligencia que pudiera "entender el mundo a traves de las proporciones". El nombre mismo, ligado a Phidias, ya anunciaba un interes por la organizacion relacional y no solo por la senal bruta.

Pero al llevar esa intuicion a codigo aparecio inmediatamente una tension:

- el oido humano y la teoria musical occidental ofrecen una grilla lista para pensar alturas;
- la tesis del proyecto, en cambio, queria salir de la musicalidad antropocentrica y volver a relaciones mas primitivas y mas fisicas.

La primera epoca del proyecto puede entenderse como la lucha por salir de esa grilla sin perder representabilidad.

### 2.2 CQT: una herramienta util y al mismo tiempo sospechosa

La `Constant-Q Transform` ofrecio un primer camino evidente:

- ya organiza la energia de manera logaritmica;
- ya se parece a como pensamos alturas e intervalos;
- ya permite comparar proporciones de manera relativamente natural.

Por eso fue una primera estacion razonable.

Pero tenia un problema de fondo:

> la propia herramienta arrastraba un sesgo musical/perceptual que chocaba con la ambicion de una lectura mas natural y mas general de las proporciones.

No fue un fracaso en el sentido banal de "sirve o no sirve". Fue un descubrimiento metodologico:

> una representacion puede ser tecnicamente util y epistemologicamente sospechosa al mismo tiempo.

Ese patron va a reaparecer mas adelante en el programa, y por eso importa tanto nombrarlo aca.

### 2.3 STFT: el regreso a una base menos comprometida

El paso hacia STFT fue una especie de gesto de austeridad. Si la grilla musical parecia imponer demasiado, hacia falta un espacio mas lineal, mas bruto y menos ya interpretado.

STFT permitio eso:

- ejes lineales de frecuencia;
- control multi-resolucion;
- construccion posterior de ratios sin pasar necesariamente por una teoria musical ya dada.

El proyecto todavia no tenia resuelto el problema de la representacion final, pero habia dado un paso clave:

> separo el analisis fisico de la grilla perceptual prefabricada.

Ese movimiento, visto retrospectivamente, anticipa la rectificacion epistemologica mucho mas madura que recien ahora se vuelve explicita en Escalon 2.

---

## 3. Segunda epoca: la escuela de histogramas densos

Una vez resuelto que la representacion no iba a descansar ingenuamente sobre una escala musical heredada, aparecio la siguiente pregunta:

> como guardo muchos ratios sin destruir su distribucion?

La respuesta fue la escuela de histogramas.

### 3.1 H0: contar sin adornar

El primer gesto fue casi elemental:

- detectar componentes;
- construir ratios entre pares;
- acumularlos en bins.

H0 tenia una gran virtud: era denso. No intentaba ya elegir unos pocos eventos "inteligentes". Guardaba una forma estadistica global.

Su problema no era la falta de inteligencia sino la falta de contexto:

- no sabia cuando pasaban las cosas;
- no distinguia estabilidad, energia ni dispersion;
- no tenia temporalidad.

### 3.2 H1: el ratio gana espesor semantico

Con H1 el proyecto deja de decir solo "cuantas veces aparece este ratio" y empieza a decir tambien:

- con que energia;
- con que estabilidad;
- con que dispersion local.

Ese agregado es mucho mas importante de lo que parece. A partir de ahi, la representacion deja de ser solo conteo y empieza a parecerse a un descriptor de estado.

### 3.3 H2: el momento en que entra el tiempo

El salto de H2 fue uno de los primeros cambios de paradigma reales del programa:

- ya no un histograma por archivo;
- sino una secuencia de histogramas en el tiempo.

Esto importa muchisimo historicamente, porque en retrospectiva prepara dos cosas:

1. el interes por descriptores locales y no solo globales;
2. la intuicion de que una estructura relacional puede importar no solo por su distribucion global, sino por su evolucion.

Si mas tarde `D4`, `A4`, `V4-lin` o `H-series` tienen sentido, es porque el proyecto ya habia aprendido que el tiempo no es un detalle secundario.

### 3.4 Lo que la escuela H resolvio

La escuela H no resolvio el proyecto entero, pero resolvio una cuestion esencial:

> que una representacion densa de relaciones puede ser mas valiosa que una seleccion demasiado agresiva de eventos.

Esa leccion va a ser decisiva cuando aparezca la tentacion de los tokens sparse.

---

## 4. Tercera epoca: revisionismo, tokens sparse y la tentacion Shazam

El revisionismo UOEMD y sus parientes fueron una etapa de alta productividad negativa. Es decir: una etapa que cerro caminos con elegancia y por eso hizo avanzar al programa.

### 4.1 La seduccion de los tokens

La logica sparse parecia perfecta:

- en vez de cargar una distribucion entera, quedarse con pocos eventos relevantes;
- en vez de una masa densa, construir constellations;
- en vez de histogramas, usar hashes o tokens con estructura.

Habia un atractivo muy fuerte en esa via:

- mas interpretabilidad local;
- menos costo potencial;
- parentesco con Shazam y con esquemas de fingerprinting ya famosos.

### 4.2 Constellations: el problema de adelgazar demasiado

Las familias `S0` y `S1` mostraron rapidamente un problema profundo:

> al sparsear demasiado, se corre el riesgo de destruir la distribucion que hacia informativo al sistema.

El resultado historico no fue solo un "no anduvo". Fue un diagnostico:

- la informacion relacional no necesariamente sobrevive al paso a unos pocos tokens;
- la discriminatividad puede vivir en la densidad global o en una continuidad que el sparseado rompe.

### 4.3 Hashes y voting exacto: otra via que no era la via principal

Las familias `K0`, `K1`, `K2` y sus sucesoras exploraron la logica del matching exacto:

- discretizacion;
- overlap;
- voting;
- hash buckets;
- IDF;
- anclaje de eventos.

Esto produjo resultados interesantes, incluso prometedores a pequena escala. Pero la conclusion metodologica fue dura:

> matching exacto y alineacion cross-modal por embeddings no son el mismo problema.

Shazam funciona porque su tarea es otra. Sirve para recuperar una huella exacta. Phideus, en cambio, busca una alineacion mas abstracta, mas geometrica y mas robusta a diferencias de sensor, representacion y estilo.

Ese descubrimiento despejo el terreno para BIAS_CONTROL.

---

## 5. Escalon 1: MAESTRO y el paso a la fase realmente experimental

Escalon 1 no fue solo un cambio de dataset. Fue el momento en que el programa dejo de preguntarse principalmente como representar ratios y paso a preguntarse:

> como hacer que una arquitectura cross-modal los use de verdad?

### 5.1 Por que MAESTRO importo tanto

MAESTRO resolvio varios problemas a la vez:

- escala;
- alineacion casi perfecta;
- par audio/MIDI claro;
- dominio suficientemente rico para exigir mucho al modelo.

Eso permitio salir del mundo de las pruebas demasiado chicas y pasar a una fase donde la comparabilidad experimental importaba tanto como la intuicion teorica.

### 5.2 El gran cambio: de representacion a intervencion

Con BIAS_CONTROL, el descriptor deja de ser solo un objeto para analizar y pasa a ser una intervencion sobre el modelo.

Ya no se trata solo de preguntar:

- que informacion hay en el descriptor?

Ahora se pregunta:

- que pasa con la geometria del embedding si esta informacion entra en tal o cual punto del sistema?

Ese cambio fue decisivo. Es ahi donde Phideus se convierte de verdad en un programa experimental sobre invariantes y no solo en una familia de extractores.

### 5.3 D4: el lado MIDI aprende relaciones locales

`D4` hizo visible algo importante:

- el descriptor podia dejar de vivir como branch auxiliar;
- podia actuar directamente sobre la entrada del encoder;
- y esa modalidad de inyeccion no era un detalle de implementacion, sino una hipotesis causal.

`D4` mostraba relaciones locales de pitch del lado MIDI. Esto lo hacia muy relevante para el programa, pero hoy hay que contarlo con mas precision:

- si, era un descriptor relacional;
- no, no era aun una prueba limpia de armonia natural;
- seguia atado a un dominio discreto y cuantizado.

### 5.4 A4: el gran descriptor audio de Escalon 1

`A4` termino siendo el descriptor mas causalmente fuerte del cierre. Esto ya no esta en discusion.

Pero `A4` necesita una lectura muy disciplinada:

- mide dinamica espectral local;
- trabaja sobre energia por bandas y sus cambios;
- es continuo y audio-side;
- pero no es un descriptor de armonia natural en sentido fuerte.

Esta es probablemente la rectificacion mas importante en la lectura de Escalon 1. Durante mucho tiempo, el proyecto tendio a hablar de `A4` como si fuera la realizacion fuerte de la intuicion fundacional. Hoy conviene decir algo mas preciso:

> `A4` fue el descriptor operativo correcto para hacer visible la mecanica descriptor-guided en Escalon 1, aunque no haya sido aun el descriptor filosoficamente mas fiel a la tesis fuerte de Phideus.

### 5.5 A4r, D4r y el descubrimiento del mecanismo

Las variantes reverse y otras similares hicieron visible una segunda capa del problema:

> no alcanza con preguntar que descriptor entra; tambien importa radicalmente por donde entra.

Esto abrio una linea enteramente nueva de investigacion:

- reverse cross-attention;
- conditioned projections;
- third towers;
- FiLM;
- MoE.

Todas esas familias ya no deben leerse como "mas descriptores". Son experimentos sobre la ecologia de acoplamiento entre descriptor y arquitectura.

### 5.6 El cierre Gate 5B

Gate 5B es, historicamente, el punto donde Escalon 1 deja de ser promesa.

Que deja cerrado:

- scoreboard robusto;
- multi-seed;
- ablaciones causales;
- pruebas de geometria;
- pruebas de robustez y de estres;
- evidencia de que la ventaja descriptor-guided existe y no es un espejismo de una sola corrida.

La lectura correcta del cierre es esta:

1. los descriptores pueden mejorar causalmente retrieval;
2. la mejora es principalmente geometrica;
3. el espacio latente se reorganiza;
4. esa mejora no se traduce automaticamente a mejor decodificabilidad frame-wise.

### 5.7 Lo que Escalon 1 no debe sobreafirmar

Escalon 1 no necesita ser devaluado, pero si disciplinado.

No debe afirmarse, ya sin mas:

- que `D4` y `A4` ya demostraron la teoria fuerte de la armonia natural;
- que un descriptor audio continuo equivale automaticamente a una representacion fisicamente privilegiada;
- que la superioridad de `A4/A4r` clausura la pregunta ontologica del proyecto.

Escalon 1 resolvio una pregunta enorme:

> los descriptores pueden reorganizar causalmente el espacio cross-modal.

Pero dejo abierta otra:

> cuales de esos descriptores expresan de verdad una estructura natural privilegiada y cuales son "solo" herramientas eficaces?

Esa deuda es precisamente la que abre Escalon 2.

---

## 6. Gate 6, Gate 7 y Gate 8: por que importan para la historia de las representaciones

Estos gates no suelen contarse dentro del "historial de descriptores", pero deberian aparecer aca porque cambiaron la lectura de lo que el descriptor estaba haciendo.

### 6.1 Gate 6: la traduccion a downstream no es automatica

Gate 6 pregunta algo elemental y brutal:

> si la geometria del embedding mejora, eso mejora tambien una tarea downstream como AMT?

La respuesta, por ahora, no fue simple. El salto descriptor-guided en retrieval no se traslado de forma obvia a decodificacion directa.

Leccion para este informe:
- un descriptor puede reorganizar muy bien el espacio y, aun asi, no aparecer como mejora simple en downstream;
- eso vuelve todavia mas importante separar geometria, decodificabilidad y causalidad.

### 6.2 Gate 7.1a: encoder mas fuerte no equivale a mejor lectura

`D0_mert330m_frozen ~ D0_lite` fue una leccion incomoda pero fundamental.

En numeros, la senal fue casi provocativamente sobria:

- `D0_mert330m_frozen = 75.0%`
- `D0_lite = 75.2% +/- 2.3pp`

No resolvio todo, pero si destruyo una simplificacion facil:

> que bastaria con un encoder mas fuerte para hacer visible la estructura correcta.

Eso fortalecio dos intuiciones historicas:

1. la geometria y la co-adaptacion importan tanto como la capacidad bruta;
2. el proyecto no debe rendirse rapido ante la tentacion de "foundation model como solucion".

### 6.3 Gate 8: las proyecciones tambien son parte de la historia descriptorial

Gate 8 mostro otra cosa:

> no alcanza con tener la informacion; tambien importa si las proyecciones la preservan o la destruyen.

El punto historiografico importante es que Gate 8 no crea nuevos descriptores. Relee los viejos descriptores desde otra pregunta:

- donde se pierde la informacion que el descriptor ayudaba a introducir?

Y ya dejo un primer contraste concreto:

- `a4r-ctrl = 79.2%`
- `a4r-pcm = 80.0%`

La mejora es pequena. Precisamente por eso es valiosa historicamente: obliga a leer Gate 8 como auditoria de preservacion y proyeccion, no como irrupcion de un descriptor completamente nuevo.

Eso enriquece mucho la historia general. Ya no hay una sola narrativa "de descriptor"; hay una narrativa de descriptor + ruta + proyeccion + geometria.

---

## 7. Escalon 2: la primera rectificacion epistemologica explicita

Si Escalon 1 fue la validacion de la mecanica descriptor-guided, Escalon 2 es la etapa en que el programa se mira a si mismo y se exige mas.

### 7.1 Por que Speech<->EGG cambia tanto la escena

Speech<->EGG es un dominio especialmente importante porque elimina parte de la ambiguedad que siempre podia quedar en Audio<->MIDI.

En Audio<->MIDI siempre estaba la objecion:

- una mitad del problema es simbolica;
- la musica humana trae ya una organizacion cultural muy fuerte;
- el descriptor podria estar aprovechando una estructura que no prueba nada sobre la naturaleza fuera de ese marco.

Speech<->EGG cambia eso:

- hay dos sensores del mismo fenomeno fisico;
- uno mide la fuente glotal mas directamente;
- el otro mide la señal ya filtrada por el tracto vocal;
- el problema es mas crudo, mas fisico y menos ya domesticado por una notacion simbolica.

### 7.2 S2-P0 y S2-P1: la disciplina antes del entusiasmo

Una de las mejores cosas de Escalon 2 es que no salto directo a un modelo.

Se hizo primero:
- manifest;
- split por speaker;
- auditoria de alineacion;
- pool canónico;
- baseline lineal.

Y el resultado de `CCA S=64.4%` en `noise0` fue decisivo, porque mostro que no se trataba de una fantasia documental. Habia señal cross-modal real antes de cualquier arquitectura sofisticada.

### 7.3 S2-P2-control: el baseline neural ya existe

Con `S2-P2-control` completo (`S=77.8% @ ep25`), Escalon 2 deja de ser un dominio preparado y pasa a ser un frente experimental real.

Esto cambia por completo el estatuto de los descriptores de Escalon 2:

- ya no son ideas para "cuando haya un modelo";
- son candidatos para intervenir sobre un baseline neural ya existente.

### 7.4 La rectificacion

Y aca aparece el punto mas importante de todo este informe:

> Escalon 2 obliga a reconocer que no toda representacion relacional del proyecto estaba igualmente alineada con la tesis fuerte de armonia natural.

La directiva nueva, ya explicitada en el marco epistemologico del programa, es esta:

> los descriptores primarios deben derivarse preferentemente de invariantes fisicos del fenomeno medido; los descriptores perceptuales o logaritmicos pasan a ser controles comparativos, no default.

Esta frase reordena retrospectivamente todo el historial.

---

## 8. Que significa ahora "armonia natural"

Este es el punto donde el informe historico tiene que volverse un poco mas filosofico, porque si no la rectificacion queda mal explicada.

### 8.1 Armonia perceptual

La armonia perceptual es la armonia tal como la experiencia humana y la teoria musical la organizaron:

- logaritmos;
- octavas como equivalencias;
- rejillas de alturas;
- semitonos;
- temperamento;
- distancias percibidas.

Nada de eso es ilegitimo. Pero tampoco debe confundirse con la tesis fuerte del proyecto.

### 8.2 Armonia natural

La armonia natural, en el sentido en que Phideus la quiere investigar ahora, apunta a otra cosa:

- relaciones lineales;
- periodos y razones fisicas;
- estructura de la serie armonica;
- multiplicidad `f, 2f, 3f, 4f...`;
- invariantes que no dependen de la grilla perceptual humana para existir.

Esto no significa que lo perceptual sea falso. Significa que el proyecto quiere dejar de tomarlo como punto de partida obligatorio.

### 8.3 Dos familias que no deben mezclarse

Esta distincion se vuelve crucial en Escalon 2:

1. **Dinamica temporal del oscilador**  
   Ejemplo: `V4-lin`, `V4-log`.  
   Miden como cambia el `F0` o el periodo entre frames.

2. **Estructura armonica intra-frame**  
   Ejemplo: `H-series`.  
   Mide relaciones entre armonicos dentro del frame mismo.

Ambas familias son relacionales. Pero no prueban la misma hipotesis. Confundirlas bajo el mismo nombre de "ratios" seria conceptualmente pobre.

---

## 9. La nueva taxonomia de Escalon 2

Despues de la rectificacion, el frente Speech<->EGG ya no puede organizarse solo en "descriptor si / no". Ahora necesita familias claramente diferenciadas.

### 9.1 V4-lin

Rol:
- descriptor temporal natural del oscilador.

Que aporta:
- ratios lineales de `F0`;
- continuidad del cambio prosodico;
- regularidad fisica del ciclo.

Que lo vuelve importante:
- ya no usa por default una escala perceptual.

### 9.2 V4-log

Rol:
- control comparativo perceptual.

Que aporta:
- misma intuicion temporal, pero en coordenadas logaritmicas.

Por que importa:
- permite preguntar si la ventaja, en caso de aparecer, depende realmente de la coordenada fisica o solo de tener una representacion relacional razonable.

### 9.3 H-series

Rol:
- descriptor mas directamente alineado con la tesis fuerte de armonia natural.

Que aporta:
- razones entre armonicos;
- concentracion armonica;
- desviacion respecto a estructura armonica esperada.

Por que importa:
- si este descriptor ayuda, el argumento a favor de una estructura fisica privilegiada se vuelve mucho mas fuerte que en Escalon 1.

### 9.4 A4-16k

Rol:
- control de dinamica espectral.

Por que entra:
- para evitar que cualquier mejora se lea de inmediato como triunfo de la armonia natural.

Que aclara:
- si el control espectral gana y los descriptores naturales no, la lectura cambia;
- si los descriptores naturales superan al control, la lectura cambia de otra manera.

### 9.5 V4-lin+H

Rol:
- prueba de complementariedad natural.

Que pregunta:
- si dinamica temporal del oscilador y estructura armonica intra-frame son capas complementarias del mismo fenomeno.

---

## 10. Relectura retrospectiva de todo el proyecto

Con esta directiva nueva, la historia completa se ve distinta.

### 10.1 Lo temprano no fue un error; fue preparacion

Las primeras representaciones:
- no estaban "equivocadas" por no ser la armonia natural perfecta;
- estaban resolviendo el problema previo: como codificar relaciones de manera estable y densa.

Sin esa etapa:
- no habria habido escuela de histogramas;
- no habria habido intuicion temporal;
- no habria habido criterio para rechazar sparse tokens;
- no habria habido lenguaje experimental suficiente para Escalon 1.

### 10.2 Escalon 1 no fue un desvio; fue una validacion de primer orden

Escalon 1 tampoco queda disminuido por la rectificacion.

Al contrario, queda mejor ubicado:

- fue la prueba de que la intervencion descriptorial puede funcionar;
- fue la prueba de que la geometria importa;
- fue la prueba de que la red puede responder causalmente a una estructura agregada.

Eso es exactamente lo que hacia falta antes de pedirle al programa una tesis ontologica mas dura.

### 10.3 Escalon 2 es la primera vez que la teoria fuerte se deja evaluar de verdad

Esa es la frase historica mas importante del momento:

> Escalon 2 no es solo la continuacion del programa; es la primera rectificacion epistemologica explicita del programa.

Porque por primera vez:
- el dominio obliga menos a esconderse detras de la simbolizacion musical;
- los descriptores se reordenan en funcion de natural vs perceptual;
- la pregunta ya no es solo "mejora retrieval?", sino "que clase de estructura mejora retrieval y que se sigue de eso?"

---

## 11. Relacion con la Harmonic Information Theory

Si este informe no conectara con la Harmonic Information Theory, quedaria mutilado. Pero esa conexion necesita disciplina.

### 11.1 La lectura prudente

La lectura prudente es esta:

> la historia de Phideus aporta evidencia de que ciertas estructuras relacionales pueden ser especialmente utiles para organizar, comprimir y alinear informacion entre modalidades distintas de un mismo fenomeno.

Esto ya es mucho.

### 11.2 La lectura fuerte

La lectura fuerte seria:

> las razones y la estructura armonica natural constituyen un lenguaje privilegiado de la informacion en la naturaleza.

Hoy, el programa no debe afirmar esto como conclusion cerrada.

Pero si puede decir algo mas interesante:

> esta empezando a construir un metodo experimental para poner a prueba esa posibilidad.

Esa formulacion le hace justicia al proyecto sin convertirlo en metafisica apresurada.

---

## 12. Cronologia reordenada del proyecto

| Etapa | Pregunta principal | Tipo de respuesta que produce | Resultado historico |
|---|---|---|---|
| CQT / STFT inicial | como representar ratios sin quedar prisioneros de la grilla musical? | infraestructura y representacion | paso de sesgo musical a base frecuencial mas austera |
| Escuela H | como preservar distribucion y enriquecerla? | histogramas densos y temporales | consolidacion de representaciones ricas |
| Revisionismo sparse | alcanza con tokens y matching exacto? | falsacion de hipotesis rivales | NO-GO para sparse como camino principal |
| Escalon 1 | puede un descriptor reorganizar causalmente un embedding cross-modal? | evidencia causal y geometrica | SI, con cierre robusto Gate 5B |
| Gate 6/7/8 | donde se agota, no traduce o se pierde la senal descriptorial? | auditoria de mecanismos y limites | frente aun abierto pero ya informativo |
| Escalon 2 | cuales descriptores testean de verdad la tesis natural? | rectificacion epistemologica y nueva taxonomia | frente abierto con baseline neural ya cerrado |

---

## 13. Lo que este informe deja fijado

Despues de esta reescritura, quedan congeladas varias afirmaciones que deberian ordenar la narrativa futura del proyecto.

1. Escalon 1 no se reinterpreta como "prueba ya lograda de armonia natural".
2. Escalon 1 si queda fijado como validacion fuerte de la mecanica descriptor-guided.
3. `A4` queda clasificado como descriptor audio-espectral local eficaz, no como descriptor natural fuerte.
4. `D4` queda clasificado como descriptor relacional local en dominio MIDI, importante pero aun ligado a cuantizacion musical.
5. Gate 8 se clasifica como investigacion sobre mecanismos de preservacion, no sobre nuevos descriptores.
6. Escalon 2 queda fijado como primera arena donde la directiva de armonia natural pasa a ser metodologicamente obligatoria.
7. `V4-lin`, `H-series` y `A4-16k` ya no son nombres intercambiables: cada uno representa una hipotesis distinta.

---

## 14. Cierre

La historia de las representaciones de ratios en Phideus no es la historia de una idea fija que siempre supo lo que buscaba. Es la historia de un programa que fue aprendiendo a formular mejor su propia pregunta.

Primero aprendio a representar.
Despues aprendio a no perder informacion al sparsear.
Despues aprendio a intervenir arquitecturas y cambiar la geometria de sus embeddings.
Y recien ahora esta aprendiendo a distinguir con la dureza necesaria entre:

- descriptor util,
- descriptor relacional,
- descriptor natural,
- y control comparativo.

Eso no debilita al proyecto. Lo madura.

Si Escalon 1 fue la demostracion de que Phideus podia producir ciencia experimental seria dentro del mundo de modelos aprendidos, Escalon 2 es la etapa en que esa ciencia empieza a pedirle a sus propios descriptores una fidelidad mayor a la intuicion fundacional.

Esa es, hoy, la verdadera historia de las representaciones de ratios en Phideus.
