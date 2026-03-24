# Catalogo Narrativo de Descriptores de Ratios en Phideus

Fecha de actualizacion: 2026-03-24
Documentos de apoyo:
- `MARCO_EPISTEMOLOGICO_PHIDEUS.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md`

---

## Addendum vivo del corte

Este catalogo cambia de estatuto en este corte. Hasta ahora habia tendido a contar de corrido una sola historia: la de "los descriptores de ratios" de Phideus. Esa continuidad era util para no perder la genealogia, pero dejaba mezcladas tres cosas distintas:

1. la historia larga de representaciones de ratios, histogramas y tokens;
2. la historia operativa de Escalon 1, donde se valido la mecanica de inyeccion de descriptores y reorganizacion geometrica;
3. la historia nueva de Escalon 2, donde por primera vez la pregunta por la armonia natural deja de ser solo inspiracion teorica y pasa a convertirse en directiva metodologica explicita.

La correccion importante de este corte es la siguiente:

> Escalon 1 y Escalon 2 ya no deben leerse como si estuvieran probando exactamente lo mismo.

Escalon 1 mostro que intervenir un sistema cross-modal con descriptores puede cambiar causalmente la geometria del espacio latente y mejorar retrieval. Ese hallazgo es grande, robusto y ya no depende de retorica. Pero no prueba todavia, de manera limpia, la tesis fuerte de Phideus sobre armonia natural.

Escalon 2 nace precisamente para volver explicita esa deuda. Desde ahora, los descriptores del programa se organizan en dos planos:

- descriptores que validaron la mecanica de intervencion, alineacion y reorganizacion en Escalon 1;
- descriptores que intentan testear de manera mas directa la primacia de invariantes fisicamente naturales en Escalon 2.

Este documento ya no oculta esa diferencia. La hace visible, la explica y la deja congelada en lenguaje narrativo.

Al corte operativo actual, la rectificación ya no vive solo como plan:

- `S2-P2-control` (`D0`) ya cerró con `S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`;
- `S2-P2-main` por concatenación ya cerró también, con una lectura negativa útil: `V4-lin=67.8%`, `H-series=59.8%`, `A4-16k=77.8%=D0`;
- `S2-P2.5` ya dejó de ser solo fase activa: sus `6/6` celdas ya fueron interpretadas bajo preregistro y no produjeron lift defendible sobre `D0`, aunque sí dejaron un caso claramente peor (`V4-lin + attn_bias`) y una interacción descriptor × mecanismo todavía informativa;
- y `S2-P2.5b` ya cerró `3/3` brazos `pca`, precisamente para separar mejor un null de mecanismo de un null de descriptor y permitir ahora el salto a `S2-P3`.

Pero el corte no cambió solo el tipo de descriptor o el mecanismo de entrada. También cambió la disciplina con la que el frente se permite interpretar sus propios resultados. Escalón 2 ya tiene un preregistro interpretativo propio: la lectura de `S2-P2.5` quedó fijada por una matriz de predicciones, una regla operativa basada en bootstrap pareado sobre `Delta` y guardrails para no confundir un null de mecanismo con un null de descriptor. Y ahora suma una vuelta más: el cierre de `pca` deja asentado que un empate entre `concat/attn_bias/xattn/pca` y `D0` ya puede contarse como primer null mecanístico cerrado. Eso endurece el catálogo mismo: ya no alcanza con decir qué familia existe; hay que decir bajo qué regla esa familia gana, empata o queda en ambigüedad, y qué comparación de encoder viene después (`S2-P3`).

La consecuencia narrativa de este corte es todavía más importante que los números:

> Escalón 2 no solo reordenó qué descriptores valen la pena. Reordenó también qué mecanismo de inyección es filosóficamente compatible con la hipótesis de Phideus, y bajo qué preregistro esa compatibilidad puede leerse como evidencia en vez de intuición post-hoc.

Y este corte agrega una segunda corrección importante: la reapertura `Gate 9 / A10` ya no puede contarse como si fuera simplemente “más Escalón 1”. `A7r/A9r` reabren retrospectivamente la deuda natural-harmonic dentro de música, mientras `A10a-e` intentan separar mejor entre ontologías dirigidas, controles genéricos y variantes continuas ontology-free. Esa rama no desplaza a Escalón 2; lo que hace es impedir que el catálogo vuelva a colapsar bajo una sola etiqueta cosas epistemológicamente distintas. Y ahora suma una tercera corrección: `Gate 10` ya no existe como promesa de separación descriptor × mecanismo, sino como evidencia cerrada de esa separación. El barrido completo dejó `concat > FiLM/pca >> attn_bias`, con spreads intra-mecanismo mucho más chicos que los inter-mecanismo. La lectura útil ya no es solo que "hay interacción"; es que, en esta rama retrospectiva, el mecanismo pesa más que el descriptor y aun el mejor `concat` no desplaza a los mejores brazos canónicos del programa.

Desde ahora, en el catálogo ya no alcanza con preguntar “qué descriptor es este”. También hay que preguntar:

- qué tipo de relación codifica;
- qué mecanismo de entrada le corresponde;
- y si el fracaso o éxito del arm dice algo sobre el descriptor, sobre la arquitectura o sobre ambos.

El catálogo ya no necesita cargar solo con esa tarea. La formulación larga de ese reordenamiento vive ahora también en el libro HIT (`manifiesto_HIT_Beancon_Phideus/Harmonic_Information_Theory_Foundations.md`), donde la dualidad `storage / retrieval`, el `activation problem` y la secuencia `Phideus -> Beacon` quedaron integrados en una arquitectura teórica más amplia. Este documento conserva entonces otra función: mantener visible la genealogía descriptorial y el criterio de lectura que permite no colapsar descriptor, mecanismo y arm bajo una sola etiqueta.

Y este corte suma un complemento importante: Escalón 3 ya no es solo una promesa lateral del programa. `E3-P0` ya dejó materializado un banco Lissajous canónico, `P2` ya fijó un baseline dual, `P4` ya mostró el límite de la lectura post-hoc sobre latente plano y `P5/P6` ya dejaron una primera lectura geométrica completa. Eso no agrega una nueva familia descriptorial al catálogo en el sentido clásico, pero sí agrega una nueva arena de lectura: un lugar donde el ratio ya no entra solo como descriptor o como control, sino como estructura visible y generable sobre la cual pueden compararse storage plano, lectura por probes y geometrías no planas. La síntesis útil del corte es sobria: `P2-flat` sigue siendo el baseline general, `P5-cqtshift` emerge como mejor brazo geométrico/OOD y el toro puro no se vuelve automáticamente la respuesta.

---

## 1. Como leer este catalogo

Antes de enumerar familias y nombres, hace falta fijar una disciplina minima de lectura. En Phideus no todo lo que recibe un nombre corto es la misma clase de cosa. Si no se separan estas capas, la historia se vuelve opaca.

### 1.1 Descriptor, mecanismo y arm no son sinonimos

Un **descriptor** es la informacion adicional que queremos introducir en el sistema. Responde a la pregunta:

> que estructura le estamos pidiendo al modelo que vea mejor?

Ejemplos:
- `D4`: intervalos locales del lado MIDI.
- `A4`: dinamica espectral local del lado audio.
- `V4-lin`: ratios lineales de `F0` entre frames sucesivos.
- `H-series`: razones de amplitud armonica intra-frame.

Un **mecanismo** es la forma en que ese descriptor entra al modelo. Responde a otra pregunta:

> por donde y con que arquitectura inyectamos esa informacion?

Ejemplos:
- input augmentation;
- reverse cross-attention;
- FiLM;
- conditioned projections;
- third tower;
- MoE.

Un **arm** es la combinacion concreta de descriptor, mecanismo y receta de entrenamiento. Responde a la pregunta:

> que experimento corrimos exactamente?

Ejemplos:
- `a4r`: descriptor `A4` con mecanismo reverse cross-attention en audio;
- `d4a4`: `D4` + `A4` con receta dual same-modality;
- `a4r-pcm`: no es un descriptor nuevo, sino `A4r` mas una modificacion de proyeccion MIDI via FiLM.

Esta separacion importa porque una parte de la confusion historica del programa vino de hablar como si `A4r` fuera un descriptor nuevo, cuando en realidad es el mismo contenido descriptorial `A4` con otra ruta de entrada. El rendimiento cambia, pero cambia por mecanismo, no por ontologia del descriptor.

### 1.2 Dos preguntas diferentes

Desde este corte, el catalogo contesta dos preguntas distintas:

1. **Que descriptores existieron o fueron propuestos en la historia del proyecto?**
2. **Cuales de ellos realmente testean la tesis fuerte de armonia natural y cuales no?**

Sin esa segunda pregunta, el inventario tecnico puede dar una falsa impresion de continuidad. Y justamente lo que ahora interesa es romper esa falsa continuidad.

---

## 2. La gran separacion: Escalon 1 no es Escalon 2

### 2.1 Que fue Escalon 1

Escalon 1 se trabajo sobre Audio<->MIDI en MAESTRO. El objetivo real de esa etapa fue responder algo mas acotado pero mas concreto:

> se puede intervenir un modelo cross-modal con descriptores de estructura relacional y obtener una mejora causal, robusta y repetible en retrieval?

La respuesta, al cierre de Gate 5B, fue si.

Eso se vio en varias capas:
- `d4a4` como mejor brazo dual robusto;
- `A4/A4r` como señal causal dominante del paquete final;
- mejoras sostenidas sobre `D0`;
- reorganizacion geometrica confirmada por RSA/CKA;
- ablaciones causales donde la senal descriptor-guided se cae cuando el descriptor real se reemplaza por `zero`, `random` o `shuffled`.

Pero esa historia no debe sobreleerse. Escalon 1 mostro:

- que la inyeccion de descriptores funciona;
- que la geometria importa;
- que el descriptor puede cambiar causalmente la organizacion del espacio latente;
- que no toda mejora en retrieval aparece como mejor decodificabilidad local.

Lo que Escalon 1 no mostro limpiamente es otra cosa:

> que la armonia natural, entendida como estructura fisica lineal privilegiada, ya haya sido probada como tesis fuerte.

### 2.2 Que pasa a ser Escalon 2

Escalon 2 ya no puede vivir de esa ambiguedad. Speech<->EGG no es un apendice exotico de Escalon 1: es la primera arena donde el programa queda obligado a explicitar si realmente quiere investigar armonia natural o si solo estaba haciendo una ingenieria sofisticada de descriptores.

Por eso, desde este corte:

- los descriptores de Escalon 1 quedan leidos como validacion de mecanica y geometria;
- los descriptores de Escalon 2 quedan organizados como intento de testear invariantes fisicamente naturales del oscilador y de la serie armonica.

No es una correccion cosmetica. Es una rectificacion de programa.

---

## 3. Genealogia larga antes de los escalones

Antes de que existieran Escalon 1 y Escalon 2, el proyecto ya habia producido varias familias de representacion. Conviene mantenerlas vivas porque explican de donde sale la intuicion actual, pero ya no deben leerse como si todas fueran equivalentes.

### 3.1 Familia H: histogramas densos

#### H0 - Histograma STFT global

Es el punto de partida sobrio. La idea era simple: si la señal contiene multiples componentes frecuenciales, entonces la informacion no esta solo en los picos aislados sino en las relaciones entre esos picos. H0 captura la distribucion global de esos ratios.

Que aporta:
- densidad;
- estabilidad;
- una forma de "firma" global.

Que no aporta:
- secuencia;
- contexto local;
- temporalidad fina.

#### H1 - Histograma enriquecido

Acá el proyecto deja de contar solo "cuanto aparece cada ratio" y empieza a preguntar "con que energia" y "con que dispersion". H1 es el primer momento en que el ratio deja de ser solo conteo y gana semantica por canal.

Leccion historica:
- la representacion empieza a volverse mas rica;
- pero sigue siendo mayormente global.

#### H2 - Histograma temporal

Este es uno de los grandes cambios silenciosos del proyecto. El ratio deja de ser estadistica por archivo y pasa a ser secuencia de histogramas en el tiempo. La pregunta ya no es solo "que proporciones caracterizan una pieza", sino "como evolucionan".

Importancia:
- abre la puerta a pensar en dinamica;
- vuelve mas natural el salto posterior a descriptores locales e inyecciones frame-wise;
- prepara el terreno para comprender por que Escalon 2 no puede conformarse con descriptores solo globales.

#### H3/H4/H5/H6/H7/H8

Estas variantes son la transicion entre la escuela de histogramas y el periodo BIAS_CONTROL.

- `H3`: histogramas pairwise de pitch.
- `H4`: enriquecimiento por velocity/duration.
- `H5/H6`: continuidad de H3/H4 ya bajo el regimen mas robusto de Gate 4.2.
- `H7`: empuje hacia ritmo y temporalidad.
- `H8`: gran giro hacia input augmentation local (`D4`), donde el descriptor ya no vive como branch auxiliar sino como perturbacion directa de la entrada.

Leccion de esta familia:

> el proyecto fue abandonando lentamente la idea de "resumen global suficiente" y acercandose a la idea de "intervencion local sobre la representacion".

### 3.2 Familia S: constellations y tokens sparse

#### S0 / S1

La inspiracion Shazam propuso algo seductor: en vez de histogramas densos, representar la estructura con pocos tokens muy informativos. Esto parecia elegante y eficiente.

Pero el revisionismo fue duro con esta escuela. La conclusion no fue que los ratios no importen, sino otra:

> al sparsear demasiado se corre el riesgo de destruir precisamente la distribucion que hace informativa a la representacion.

Esa derrota metodologica fue importante. Ayudo a entender que Phideus no iba a avanzar solo por "eventos inteligentes", sino por representaciones que preserven mejor densidad, continuidad y geometria.

### 3.3 Familia K: hash y fingerprinting exacto

#### K0 / K1 / K2

Esta familia es la mas cercana a la logica de matching exacto:
- hashes,
- voting,
- stoplists,
- overlap de eventos,
- precision de token.

Fue util para eliminar hipotesis alternativas y para mostrar que no todo lenguaje relacional sirve para alineacion aprendida. Tambien sirvio para entender la diferencia entre:

- fingerprinting exacto,
- y alineacion cross-modal por embeddings.

Esta distincion historica es crucial. Phideus no termino siendo un proyecto de hash mejorado; termino siendo un proyecto de representaciones intervenibles.

---

## 4. Escalon 1: descriptores operativos sobre Audio<->MIDI

Ahora empieza la separacion importante. A partir de aqui, cada entrada no solo dira "que era", sino tambien "que parte de la tesis fuerte prueba y que parte no".

### 4.1 D0 - Control VICReg puro

`D0` no es un descriptor, pero merece entrar en el catalogo por una razon metodologica decisiva: sin `D0` no existe inferencia causal limpia.

Que representa:
- el modelo cross-modal sin informacion adicional inyectada.

Que permite:
- medir cuanto de la alineacion viene del regimen base;
- distinguir mejora por descriptor de mejora por entrenamiento o arquitectura.

Leccion:
- en Phideus, el control no es relleno; es una pieza epistemica.

### 4.2 D4 - Intervalos locales del lado MIDI

`D4` fue uno de los grandes hallazgos instrumentales de Gate 4.2.

Que mide:
- relaciones locales entre notas consecutivas del lado MIDI;
- semitone deltas;
- log-ratios de pitch;
- vecindad intervalica inmediata.

Que tiene de ratio:
- si, hay estructura relacional real.

Que no tiene:
- no es armonia natural en sentido fuerte;
- sigue anclado a un dominio discreto, cuantizado y en buena medida temperado.

Lectura correcta:
- `D4` no prueba la primacia de la serie armonica natural;
- `D4` si prueba que intervenir el encoder con relaciones locales cambia de forma productiva la alineacion.

Ese matiz es fundamental. `D4` sigue siendo importantisimo, pero ya no debe sobrepresentarse.

### 4.3 A4 - Dinamica espectral local del lado audio

`A4` es el descriptor audio mas importante de Escalon 1. Y precisamente por eso hace falta decir con toda claridad que es y que no es.

Que mide:
- energia por bandas;
- cambio temporal de esa energia;
- dinamica espectral local;
- forma de la envolvente espectral en movimiento.

Que no mide:
- no mide ratios lineales de frecuencia entre ciclos;
- no mide la serie armonica como tal;
- no mide "armonias naturales" en sentido fuerte.

Entonces, que lugar ocupa?

`A4` es un descriptor audio continuo, local y altamente operativo. Fue el brazo audio que mostro mayor eficacia causal y que sostuvo parte central del cierre Gate 5B. Pero su fuerza no debe leerse como prueba de la tesis natural; debe leerse como prueba de algo mas acotado y muy importante:

> el lado audio se beneficia fuertemente de una descripcion local adicional de la dinamica espectral, incluso cuando esa descripcion no es todavia el descriptor filosoficamente soñado por Phideus.

### 4.4 A4r y D4r - mismo descriptor, otra ruta

`A4r` y variantes `r` no son familias ontologicas nuevas de descriptor. Son el mismo contenido descriptorial pasando por otra ruta de inyeccion.

Esto conviene subrayarlo porque durante meses el programa tendio a hablar de `A4`, `A4r`, `D4`, `D4r` como si cada una fuera una cosa sustantivamente nueva. No lo son en el mismo sentido.

La diferencia real es:
- el descriptor sigue siendo `A4` o `D4`;
- cambia el mecanismo de acople con la arquitectura.

Leccion:
- el proyecto no aprendio solo "que descriptor sirve";
- tambien aprendio que el **como entra** el descriptor puede cambiar mucho el efecto final.

### 4.5 D4+A4 - la sinergia fuerte de Escalon 1

`D4+A4` fue la combinacion que mejor sintetizo la intuicion de Escalon 1:

- lado MIDI con estructura intervalica local;
- lado audio con dinamica espectral local;
- ambos acoplados en un regimen contrastivo comparable.

Ese fue el brazo que termino sosteniendo el record robusto del frente (`d4a4`, multi-seed `84.1% +/- 2.3pp`, con cierres largos aun mas altos segun regimen).

Pero conviene decirlo sin mistica:

`D4+A4` no es la prueba de que el mundo "esta hecho de ratios naturales".  
`D4+A4` si es la prueba de que una intervencion descriptor-guided bien elegida puede reorganizar causalmente la geometria y dar una ventaja cross-modal grande.

### 4.6 D4-A4r y duales relacionados

Los duales con reverse cross-attention mostraron que el mapa no era trivial:
- a veces el descriptor correcto con el mecanismo equivocado pierde fuerza;
- a veces un mecanismo mas fuerte rescata un descriptor que por otra via rendia menos;
- a veces la sinergia no es aditiva.

Leccion:
- el descriptor no vive aislado;
- vive dentro de una ecologia de acoplamiento con la arquitectura.

### 4.7 A7 - attractor racional

`A7` es especialmente importante en este catalogo, no por haber sido el mejor, sino por lo que representa historicamente.

`A7` fue el primer intento mas frontal de acercarse a la hipotesis fundacional de Phideus:
- cercania a razones simples;
- atractores racionales;
- estructura mas explicitamente ligada a armonia natural.

Y que paso?

No se convirtio en descriptor canonico ganador.

Eso no lo vuelve irrelevante. Al contrario, deja una leccion muy fina:

> el programa todavia no habia encontrado la forma correcta de traducir la intuicion de armonia natural a un descriptor que realmente funcione dentro del regimen entrenable de Escalon 1.

Esto es importante porque impide una lectura comoda y retrospectiva del proyecto. La tesis fuerte no "ya estaba demostrada y no nos habiamos dado cuenta". No. El propio historial muestra que la forma operacional de esa tesis seguia verde.

### 4.8 A8 y A9 - exploraciones que no se volvieron canon

Estas variantes entran en el catalogo por honestidad historica. No todo descriptor candidato se transforma en columna central del programa.

`A8` y `A9` mostraron que:
- la proliferacion de ramas descriptoriales no garantiza valor;
- el hecho de ser continuo o audio-side no basta para volverse relevante;
- el programa necesita controles negativos y resultados mediocres para no fabricar relatos retrospectivos demasiado redondos.

### 4.9 Gate 8 no agrega un descriptor nuevo

Gate 8 es crucial, pero por otra razon.

Conditioned projections no introducen un descriptor ontologicamente nuevo. Lo que hacen es preguntarse:

> si la informacion descriptor-guided ya esta, donde se pierde? en el encoder? en la proyeccion? en el cuello de botella de la geometria final?

Por eso `a4r-ctrl`, `a4r-pcm`, `a4r-pcd-zero`, `a4r-pcd`, `a4r-pca` no deben entrar a este catalogo como nuevos descriptores. Deben leerse como:

- mismos contenidos descriptoriales de Escalon 1;
- nuevos mecanismos de modulacion y preservacion.

La distincion no es pedante. Es la que permite no inflar artificialmente el inventario del programa.

Ademas, los primeros numeros del frente refuerzan esa lectura:

- `a4r-ctrl = 79.2%`
- `a4r-pcm = 80.0%`

La diferencia existe, pero es pequena. Eso significa que Gate 8, al menos en este corte, no esta descubriendo un descriptor nuevo milagroso. Esta auditando si el cuello de botella podia seguir estando en la proyeccion y cuanto de la informacion descriptorial sobrevivio hasta ese punto.

### 4.10 Que queda validado por Escalon 1

Lectura consolidada del escalon:

1. la inyeccion de descriptores puede ayudar causalmente;
2. el efecto descriptor-guided puede ser grande y estable;
3. la mejora opera como ventaja geometrica, no como decodificabilidad local simple;
4. `A4/A4r` fueron los brazos causales mas importantes del cierre;
5. `D4` aporta estructura relacional valiosa, pero no monopoliza la ganancia;
6. el programa aprendio tanto sobre descriptores como sobre mecanismos de inyeccion.

Y, al mismo tiempo:

1. Escalon 1 no debe presentarse ya como prueba limpia de armonia natural;
2. `A4` no debe confundirse con un descriptor ratio-natural;
3. `D4` no debe confundirse con una prueba fuerte de la serie armonica fisica.

Eso deja a Escalon 1 en el lugar correcto: un exito grande, pero un exito de primera capa epistemica.

---

## 5. Escalon 2: los descriptores dejan de ser solo instrumentales

Escalon 2 inaugura una etapa mas exigente. No basta con que el descriptor ayude. Ahora importa por que ayuda y que clase de estructura representa.

### 5.1 El cambio de dominio cambia la exigencia

Speech<->EGG es un frente distinto por una razon profunda:

- en Audio<->MIDI siempre podia quedar la objecion de que una mitad del problema ya venia muy estructurada simbolicamente;
- en Speech<->EGG tenemos dos sensores distintos del mismo fenomeno vocal;
- eso vuelve mas dificil esconderse detras de la conveniencia musical o de la codificacion humana.

Entonces, si un descriptor funciona aca, su lectura epistemica cambia.

### 5.2 D0 - baseline neural del escalon

De nuevo, `D0` no es descriptor, pero entra como referencia imprescindible.

Al corte:
- `S2-P0` y `S2-P1` ya dejaron manifiesto, split, audit y baseline lineal fuerte;
- `S2-P2-control` (`D0`) ya cerro como baseline neural con `S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`.

Eso significa que el frente ya tiene piso neural serio. Los descriptores de Escalon 2 no se montan sobre posibilidad abstracta; se montan sobre un baseline real.

### 5.3 V4-lin - dinamica temporal natural del oscilador

`V4-lin` es uno de los descriptores nuevos mas importantes del programa.

Que intenta medir:
- relacion lineal entre `F0[t]` y `F0[t-1]`;
- relacion lineal entre `F0[t+1]` y `F0[t]`;
- fuerza de voicing;
- regularidad del periodo.

Que lo vuelve epistemicamente nuevo:
- no pasa por la rejilla de semitonos;
- no toma la escala perceptual humana como default;
- trata al cambio del oscilador en coordenadas lineales, fisicas.

Que prueba:
- si la dinamica temporal del oscilador glotal contiene una estructura relacional util para alinear speech y EGG.

Que no prueba por si solo:
- no es aun la serie armonica intra-frame;
- no equivale automaticamente a "armonias naturales" en el sentido mas fuerte del programa.

Por eso `V4-lin` es natural, pero en la familia correcta: dinamica temporal del oscilador, no armonica intra-frame.

### 5.4 V4-log - control perceptual explicito

`V4-log` tiene un rol muy sano epistemicamente.

No entra como descriptor principal. Entra para responder otra pregunta:

> si el descriptor temporal funciona, importa realmente la coordenada lineal natural o bastaba cualquier representacion relacional razonable?

Ese rol comparativo es valioso porque impide una conclusion facilista. Si `V4-lin` y `V4-log` empatan, la lectura cambia. Si `V4-lin` gana claramente, la lectura tambien cambia. En ambos casos el experimento enseña algo.

### 5.5 H-series - la apuesta fuerte

`H-series` es probablemente el descriptor mas directamente alineado con la tesis fuerte de Phideus dentro del escalon actual.

Que intenta medir:
- razones de amplitud entre armonicos;
- concentracion armonica;
- desvio respecto al patron esperado de la serie;
- estructura intra-frame del contenido armonico.

Por que es importante:
- ya no pregunta solo por cambios temporales del oscilador;
- pregunta por la organizacion armonica del frame mismo;
- se acerca mucho mas a la intuicion de que la serie armonica fisica es un principio privilegiado de organizacion informacional.

Tambien trae una cautela:
- `H-series` en speech y en EGG no tienen por que significar exactamente lo mismo;
- eso no invalida el descriptor;
- al contrario, vuelve al experimento mas interesante, porque obliga a preguntar si esas diferencias son ruido o complementariedad.

### 5.6 A4-16k - control dinamico espectral

`A4-16k` no es el heredero natural de `A4` en el sentido fuerte. Es, mas precisamente, un control de dinamica espectral local para Speech<->EGG.

Conviene explicitar esto con cuidado:

- si `A4-16k` gana, no se sigue que la teoria fuerte de armonia natural este mal;
- si `H-series` o `V4-lin` ganan sobre `A4-16k`, no se sigue que hayan derrotado a "todo descriptor no-ratio imaginable";
- lo que se obtiene es una comparacion limpia contra un control espectral dinamico concreto.

Esa modestia interpretativa es parte de la disciplina nueva del programa.

### 5.7 V4-lin+H - combinacion natural fuerte

Esta combinacion existe para una pregunta muy especifica:

> si la dinamica temporal del oscilador y la estructura armonica intra-frame son dos caras complementarias del mismo fenomeno, se potencian cuando entran juntas?

Es una pregunta mas fuerte que "la suma mejora". Pregunta si el programa empieza a encontrar una familia natural mas rica que cualquier descriptor aislado.

### 5.8 Lo que Escalon 2 intenta corregir

Escalon 2 no nace solo para "hacer otra tarea". Nace para corregir una ambiguedad del programa:

- en Escalon 1, descriptor util y descriptor filosoficamente central no coincidieron del todo;
- en Escalon 2, el objetivo es acercar mejor ambas cosas.

Esa es la rectificacion.

---

## 6. Taxonomia congelada del catalogo

Desde este corte, el inventario de Phideus queda mejor leido si se organiza en cinco familias.

### Familia I - representaciones historicas de ratios

Incluye:
- `H0` a `H8`
- `S0` a `S1`
- `K0` a `K2`

Funcion historica:
- explorar formas de codificar relaciones frecuenciales antes del regimen actual de escalones.

### Familia II - descriptores operativos de Escalon 1

Incluye:
- `D4`
- `A4`
- `A7`
- duales como `D4+A4`

Funcion epistemica:
- validar que la inyeccion descriptorial puede reorganizar la geometria y ayudar causalmente.

### Familia III - mecanismos de inyeccion y preservacion

Incluye:
- reverse cross-attention (`A4r`, `D4r`);
- conditioned projections (Gate 8);
- third tower;
- FiLM;
- MoE.

Funcion epistemica:
- preguntar por donde se pierde o se capitaliza la informacion descriptorial.

### Familia IV - descriptores naturales de Escalon 2

Incluye:
- `V4-lin`
- `H-series`
- `V4-lin+H`

Funcion epistemica:
- testear la tesis fuerte de invariantes fisicos privilegiados.

### Familia V - controles comparativos de Escalon 2

Incluye:
- `V4-log`
- `A4-16k`
- `D0`

Funcion epistemica:
- impedir que cualquier mejora se lea automaticamente como confirmacion de armonia natural.

---

## 7. Estado vivo por descriptor

| Familia | Descriptor | Dominio | Que mide principalmente | Estatuto al corte |
|---|---|---|---|---|
| Historica | H0/H1/H2 | audio general | distribuciones globales y temporales de ratios | genealogia cerrada |
| Historica | H3/H4/H5/H6/H7/H8 | musical / MIDI | transicion hacia intervalos locales e input augmentation | genealogia cerrada |
| Historica | S0/S1 | sparse tokens | constellations y eventos anclados | infraestructura e historia, no linea central |
| Historica | K0/K1/K2 | hash / matching | fingerprinting exacto y voting | cerrada como familia auxiliar |
| Escalon 1 | D0 | Audio<->MIDI | baseline sin descriptor | control canonico |
| Escalon 1 | D4 | MIDI | relaciones locales de pitch | descriptor operativo validado |
| Escalon 1 | A4 | audio | dinamica espectral local | descriptor operativo validado |
| Escalon 1 | A4r | audio | mismo A4, otra ruta de inyeccion | mecanismo fuerte, no descriptor nuevo |
| Escalon 1 | D4+A4 | dual | sinergia MIDI+audio | brazo dual canonico del cierre |
| Escalon 1 | A7 | audio | attractores racionales | intento natural temprano, no canonico |
| Gate 9 | A7r / A9r | Audio<->MIDI | attractores racionales bajo reverse cross-attention | piloto retrospectivo preregistrado |
| Gate 8 | a4r-pcm / etc. | Audio<->MIDI | modulacion de proyecciones con descriptores ya existentes | pregunta de mecanismo, no descriptor nuevo |
| Escalon 2 | D0 | Speech<->EGG | baseline neural sin descriptor | completo, referencia |
| Escalon 2 | V4-lin | Speech<->EGG | dinamica temporal lineal de F0 | descriptor primario en rectificacion |
| Escalon 2 | H-series | Speech<->EGG | razones armonicas intra-frame | descriptor primario en rectificacion |
| Escalon 2 | A4-16k | Speech<->EGG | control de dinamica espectral local | control primario |
| Escalon 2 | V4-log | Speech<->EGG | control perceptual/logaritmico | descriptor secundario comparativo |
| Escalon 2 | V4-lin+H | Speech<->EGG | combinacion natural | descriptor secundario / complementariedad |
| Gate 9 / Escalon 2 adj. | A10a/A10b/A10c/A10d/A10e | audio / vocal | recurrencia temporal JI, control generico y variantes continuas ontology-free | `a10a-e` ya con datos en música (`69.2-71.8%`), con `a10er` best `71.8% @ e27` y final `70.2% @ e30` |

---

## 8. Lo que este catalogo ya no permite decir

Despues de esta rectificacion, hay varias frases comodas que ya no deberian aparecer en el programa sin una aclaracion inmediata.

No deberia decirse, sin mas:

- "`A4` es un descriptor de armonia natural".
- "Escalon 1 ya probo la tesis fuerte de Harmonic Information Theory".
- "`A4r` es un descriptor nuevo".
- "Cualquier mejora descriptor-guided confirma la primacia de los ratios naturales".
- "`V4` ya prueba la serie armonica".

En su lugar, la lectura correcta debe ser mas disciplinada:

- `A4` es un descriptor audio-espectral local muy eficaz, pero no prueba por si solo la tesis fuerte;
- Escalon 1 valido la mecanica de intervencion, no cerro la ontologia;
- `A4r` cambia la ruta, no la naturaleza del descriptor;
- las comparaciones de Escalon 2 son las que empiezan a discriminar entre natural, perceptual y control no-ratio.

---

## 9. Cierre

La historia de Phideus se vuelve mas fuerte, no mas debil, cuando se la cuenta con esta precision.

Escalon 1 no necesita ser inflado para seguir siendo impresionante:
- valido causalmente la utilidad de intervenir embeddings con descriptores;
- mostro reorganizacion geometrica robusta;
- produjo un cierre experimental fuerte.

Escalon 2 no necesita prometer victorias prematuras:
- necesita, mas bien, volverse el lugar donde la teoria fuerte se juega de verdad.

Este catalogo, en su nueva forma, deja fijada una diferencia crucial:

> no todo descriptor relacional es un descriptor de armonia natural; y no toda mejora descriptor-guided responde ya a la pregunta profunda del proyecto.

La tarea de Escalon 2 es precisamente empezar a responder esa pregunta sin ambiguedad.
