# Atención armónica: aprender la estructura en lugar de medirla

## La pregunta que abre el frente

Phideus viene sosteniendo una hipótesis precisa: las relaciones de frecuencia —los ratios entre los tonos que componen un sonido— forman un lenguaje informacional, y una red neuronal puede aprovecharlo. Hasta ahora ese lenguaje entró a los modelos como un descriptor calculado a mano y *inyectado* en un encoder genérico: el modelo recibía, además del audio, una lista de números que resumían la estructura armónica, y el experimento medía si esa lista ayudaba. El frente de atención armónica cambia el lugar donde vive el conocimiento. En vez de calcular los ratios afuera y pasárselos al modelo, busca una arquitectura cuya atención opere directamente sobre la geometría armónica del sonido, de modo que la red infiera la estructura en lugar de recibirla resumida.

El cambio no es de grado sino de naturaleza. Un descriptor inyectado es una respuesta entregada; una geometría interna es un espacio donde la respuesta se construye. La pregunta del frente, entonces, es si una red puede razonar con ratios —usar la consistencia global de las relaciones de frecuencia para resolver lo que cada relación aislada deja ambiguo— o si el aporte de los ratios se agota en pasarlos como features.

## La lección de AlphaFold, bien entendida

El disparador de este frente fue AlphaFold, el sistema que predice la forma tridimensional de una proteína a partir de su secuencia. Su potencia suele atribuirse a que usa transformers. Esa lectura se queda corta y conviene desarmarla, porque lo que importa para Phideus es otra cosa.

Una proteína es una cadena de aminoácidos que se pliega en el espacio. Predecir el plegamiento es predecir, para cada par de aminoácidos, qué tan cerca quedan en la estructura final. AlphaFold no trata a los aminoácidos como una secuencia plana a la que aplica atención: construye una **representación de pares**, una tabla donde cada casilla (i, j) acumula lo que el modelo cree sobre la relación entre el aminoácido i y el j. Esa tabla es el objeto central del cómputo, y la operación que la hace funcionar es la **actualización triangular**: para revisar lo que sabe del par (i, j), el modelo mira todos los terceros aminoácidos k y combina lo que sabe de (i, k) con lo que sabe de (k, j). La intuición es geométrica y estricta: si A está cerca de B y B está cerca de C, eso restringe cuán lejos puede estar A de C. La desigualdad triangular del espacio físico se vuelve una operación aprendible sobre tripletes.

Ahí está la enseñanza transferible. La inteligencia de AlphaFold no es la atención en abstracto, sino que la atención vive **dentro de la geometría natural del problema**: razona en el espacio de relaciones donde las reglas del plegamiento tienen forma, y deja que la consistencia global entre pares —la restricción triangular— corrija las estimaciones locales ruidosas. Una distancia entre dos residuos, tomada aislada, es una conjetura; el conjunto de todas las distancias tiene que ser realizable como una única figura en tres dimensiones, y esa exigencia de coherencia es la que convierte muchas conjeturas dudosas en una estructura.

## Dónde la analogía se rompe, y dónde se sostiene

Trasladar esto a la música tiene una trampa que es necesario nombrar, porque cuesta cara si se pasa por alto. La tentación inmediata es armar una actualización triangular sobre las frecuencias en escala logarítmica: si la distancia de A a B es `log(fB) − log(fA)`, y la de B a C es `log(fC) − log(fB)`, entonces la de A a C debería ser la suma. Pero esa "restricción" es una identidad algebraica: vale para tres números cualesquiera, siempre, exactamente. No hay nada que enforcar ni nada que aprender. En el espacio tridimensional la desigualdad triangular recorta el conjunto de configuraciones posibles; en la recta de las log-frecuencias la suma de diferencias no recorta nada. La pieza que parecía el corazón de AlphaFold, traída literalmente, queda hueca.

La salida es reubicar la no-trivialidad donde sí habita en el dominio armónico. Un sonido con altura definida está hecho de **parciales**: la frecuencia fundamental y una serie de componentes en frecuencias aproximadamente múltiplas de ella. Cuando suenan varias notas a la vez —una mezcla polifónica—, el oído recibe un solo conjunto de parciales entreverados y tiene que repartirlos: decidir qué parcial pertenece a qué fuente. Ese reparto es un problema de inferencia global. La relación que importa no es la distancia continua entre dos parciales, sino la pertenencia: *¿estos dos parciales nacen de la misma fundamental?* Y la pertenencia a una misma fuente es una relación de equivalencia, lo que la vuelve no trivial: si el parcial i pertenece a la misma fuente que j, y j a la misma que k, entonces i y k tienen que pertenecer a la misma fuente. Esa transitividad es la análoga genuina de la restricción triangular. Aparece donde la evidencia de a pares es ambigua y solo la coherencia del conjunto resuelve.

El caso que vuelve tangible la ambigüedad es una mezcla de dos notas cuyas fundamentales están en relación simple. Tomemos fundamentales en 100 y 150 hertz:

```
fuente A (f0 = 100):   100   200   300   400   500   600 ...
fuente B (f0 = 150):       150   300   450   600 ...
                                  ↑           ↑
                          300 = 3×100 = 2×150   600 = 6×100 = 4×150
```

El parcial en 300 hertz es, a la vez, el tercer armónico de A y el segundo de B. Mirado solo, no se puede asignar: pertenece coherentemente a cualquiera de las dos series. Lo que desambigua no es ese parcial sino el resto de la evidencia —que 200 y 400 anclan a A, que 450 ancla a B— y la exigencia de que la asignación final sea una partición consistente. Resolver esto pide integrar la estructura completa, no leer un par por vez. Ese es exactamente el régimen donde una representación de pares con propagación transitiva puede aportar sobre un modelo que clasifica cada par por separado.

## El problema concreto: repartir parciales por fuente

El experimento decisivo del frente toma esa intuición y la vuelve medible en su forma más limpia. La tarea es el agrupamiento armónico: dado el conjunto de parciales de una mezcla polifónica, predecir la matriz de "misma fuente" —para cada par de parciales, si nacen o no de la misma fundamental. Esa matriz es el objeto central, el equivalente del mapa de contactos de AlphaFold.

La materia prima es audio sintético, generado de modo que se conozca con exactitud qué parcial vino de qué fuente. Esa exactitud es la condición que hace posible el experimento: AlphaFold pudo aprender porque tuvo decenas de miles de estructuras resueltas como verdad de referencia, y acá la verdad de referencia se fabrica al sintetizar la señal. Cada mezcla combina una, dos o tres fuentes; cada fuente aporta sus parciales con una fundamental propia; el agrupamiento verdadero queda registrado pieza por pieza. La supervisión es perfecta porque la construimos.

## Las arquitecturas que se comparan

La pregunta del frente solo tiene respuesta si se aísla con cuidado qué se está midiendo. Por eso el experimento no enfrenta un modelo contra otro cualquiera, sino una escalera de arquitecturas donde cada peldaño agrega una sola cosa.

El primer peldaño es un transformer que mira los parciales como un conjunto de elementos y atiende entre ellos, con un sesgo de atención derivado de la distancia en log-frecuencia. Lee cada par desde las representaciones de los dos parciales. No recibe ninguna pista armónica explícita: tiene que descubrir la estructura desde la frecuencia y la amplitud crudas.

```
A-naive
  parciales [logf, logamp] ─► embedding ─► self-attention ×L  (sesgo = Δlogf)
                                                  │
                                                  ▼
                              readout(token_i, token_j) ─► ¿misma fuente?
```

El segundo peldaño recibe, además, las relaciones armónicas calculadas a mano para cada par: qué tan cerca está su cociente de un ratio simple, qué fundamental común implícita comparten. Es el mismo transformer, pero con la pista entregada. Funciona como el mejor competidor posible *a nivel de par*: si la respuesta ya está contenida en esas pistas, este modelo la alcanza.

```
A-rich
  parciales ─► embedding ─► self-attention ×L
                                  │
        pair-features(i,j) ───────┤   [Δlogf, cercanía a ratio simple,
                                  ▼    fundamental común implícita, ...]
                          readout(token_i, token_j, pair-features) ─► ¿misma fuente?
```

El tercer peldaño es la arquitectura que el frente pone a prueba. Mantiene una representación de pares que se actualiza a lo largo de las capas, deja que la atención entre parciales se sesgue por esa representación, y aplica la actualización triangular que propaga la transitividad: para revisar el par (i, j), combina lo que sabe de (i, k) y (k, j) sobre todos los terceros k. La lectura final sale de la representación de pares, no de los tokens sueltos.

```
B (Harmonic Pairformer)
  parciales ─► tokens ──────────────────────────────┐
  pair-features ─► z[i,j]  (representación de pares)  │
        repetir L veces:                             │
          atención entre tokens  sesgada por z  ◄────┘
          z ← z + comunicación(token_i, token_j)
          z ← z + TRIÁNGULO:  Σ_k  a(z[i,k]) ⊙ b(z[k,j])   ← propaga transitividad
                                  │
                                  ▼
                          readout(z[i,j]) ─► ¿misma fuente?
```

La comparación que importa es B contra el segundo peldaño, con las pistas armónicas igualadas: si B gana, gana la *maquinaria* —la representación de pares y la propagación transitiva— y no el hecho de tener mejores features. Para separar el aporte del triángulo del aporte de simplemente mantener una representación de pares, el experimento incluye una variante de B con la misma cantidad de parámetros pero sin la suma sobre terceros: si B le gana también a esa variante, lo que aporta es específicamente la transitividad. Un control adicional, con las pistas armónicas barajadas, vigila que cualquier ventaja no venga del mero tamaño del modelo.

## La síntesis no alcanza con generarla: hay que probar que deja lugar

El punto más delicado del frente no es la arquitectura sino la validez del experimento, y conviene exponerlo porque enseña algo que excede a este caso. Un experimento que compara B contra el competidor con features igualadas solo dice algo si las features no resuelven ya la tarea por sí solas. Si las pistas armónicas entregadas alcanzan para separar las fuentes, el competidor llega al techo y B no tiene dónde mejorar: una eventual igualdad entre ambos no probaría que la transitividad no sirve, sino que la tarea no dejaba lugar para que sirviera.

Esa trampa apareció dos veces, y las dos veces una auditoría previa la atrapó antes de gastar cómputo. En el primer diseño, con parciales en múltiplos enteros exactos, dos parciales de la misma fuente quedaban en un cociente de enteros pequeños —un ratio simple— y una sola feature calculada los separaba casi perfectamente: la tarea era trivial a nivel de par. En el segundo diseño, con los parciales ya desafinados para romper ese atajo, sobrevivía otro: la envolvente de amplitud, que decaía de forma fija con el número de armónico, filtraba indirectamente cuál era ese número, y volvía a hacer separable cada par. Ninguno de los dos atajos era visible a simple vista; ambos habrían producido un resultado falsamente nulo, leído como "la transitividad no aporta" cuando en realidad el dato no la necesitaba.

De ahí queda una práctica que el frente sostiene como norma: ningún conjunto de datos pasa a entrenamiento sin demostrar dos cosas a la vez. Que ninguna combinación cerrada de las pistas disponibles resuelva la tarea por sí sola —hay margen genuino para que la maquinaria global aporte—, y que la tarea siga siendo resoluble cuando se la mira en conjunto —no es imposible, solo localmente ambigua. El experimento decisivo vale en la medida en que existe esa franja entre lo trivial y lo imposible, y la franja se verifica antes, no después.

## Qué decidió Fase 0

El frente no prometía que una red razonara con armonía; puso esa afirmación en riesgo de ser refutada barata. La `Fase 0` ya produjo una primera respuesta matizada. El salto más fuerte no vino de una regla local de ratios ni de una feature externa, sino de sostener una representación explícita de pares: `B-minus ≫ A-rich` mostró que el objeto útil del cómputo no es solo el pico, sino la relación entre picos.

El `triangle`, en cambio, no quedó validado como ganador universal. En `IID` y `OOD-regime`, una mezcla local param-matched puede igualarlo o superarlo levemente. Pero en `OOD-poly`, donde la polifonía del test aumenta y la ambigüedad global se vuelve más dura, `B` supera a `B-local` en `AUC/AP` y también supera claramente a `B-shuffle`. Esa combinación sostiene una lectura acotada: la estructura triangular no es simple capacidad adicional, sino un sesgo relacional que ayuda a generalizar cuando cambia la cantidad de fuentes.

El caveat final también es parte del resultado. La red puede rankear mejor los pares y, sin embargo, fallar como sistema de agrupamiento si el umbral `τ` elegido en validación no transfiere. Por eso la salida de `Fase 0` no es una promoción sin reservas, sino un `GO` acotado: la arquitectura tiene señal, pero hay que resolver la decisión de clustering.

## Geometría relacional de la armonía

La intuición geométrica del frente no debe confundirse con una geometría euclídea de frecuencias. En `log f`, las diferencias entre tres picos cumplen identidades algebraicas triviales; esa no es una restricción aprendible. La geometría que sí aparece es relacional:

```
picos espectrales      → nodos
same-source[i,j]       → aristas aprendidas
fuentes armónicas      → clases de equivalencia
```

Cada fuente puede pensarse como una familia generativa discreta, por ejemplo:

```
f_n = n · f0 · sqrt(1 + beta · n²)
```

Los picos de una misma fuente no son solo cercanos; son coherentes con un mismo generador latente. La matriz de pares `z[i,j]` es el espacio donde la red representa esa compatibilidad. La actualización triangular propaga evidencia indirecta: si `i` parece ir con `k`, y `k` con `j`, entonces `i` y `j` reciben información nueva. No se impone transitividad como regla lógica dura; se aprende cuándo esa evidencia global corrige la ambigüedad local.

Esa es la diferencia con un descriptor clásico. Un descriptor calcula afuera una señal armónica y se la entrega al modelo. El Harmonic Pairformer intenta que la relación armónica sea el medio interno del cómputo.

## Qué sigue

El siguiente paso no es todavía saltar a audio real. Primero viene `Fase 0.5`: auditar calibración. Esa fase reusa el setup de `Fase 0`, pero guarda matrices de validación/test y checkpoints para poder estudiar cómo convertir logits de pares en clusters sin usar información privilegiada de test. La pregunta es si la ventaja representacional de `B` en `OOD-poly` se convierte en una ventaja de agrupamiento bajo una regla deployable.

Recién después conviene pasar a `Fase 1a`: renderizar las mezclas sintéticas, detectar picos con CQT y usar esos picos detectados como tokens. Ahí entran picos faltantes, picos corridos, picos fusionados y espurios, pero todavía con ground truth controlado. Audio real/stems queda como una fase posterior.
