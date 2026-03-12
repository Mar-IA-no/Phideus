S2-P1: Explicación Detallada

  Qué es S2-P1 y por qué importa

  S2-P1 es el primer test de señal del Escalón 2. La pregunta es: "¿Speech y EGG comparten información cross-modal suficiente para retrieval, incluso con
  métodos simples?"

  Contexto: Speech (micrófono) y EGG (electroglotógrafo) capturan el mismo fenómeno físico — las cuerdas vocales vibrando — pero con sensores distintos. El
  micrófono captura el sonido radiado (incluye resonancias del tracto vocal, formantes), mientras el EGG mide directamente el contacto de los pliegues
  vocales vía impedancia eléctrica en el cuello. Son como dos fotografías del mismo objeto desde ángulos distintos.

  Qué hace el script

  1. Extracción de features (20 dims por segmento)

  Para cada ventana de 2 segundos (32000 muestras a 16kHz), tanto de speech como de EGG:

  8 band energies (media del log-magnitud por banda):
  Se calcula la STFT (n_fft=1024, hop=256) y se divide el espectro en 8 bandas de frecuencia:
  - Banda 0: ~47-94 Hz (F0 fundamental masculina)
  - Banda 1: ~94-188 Hz (F0 femenina, primer armónico masculino)
  - Banda 2: ~188-375 Hz (armónicos bajos)
  - Banda 3: ~375-750 Hz (formante F1)
  - Banda 4: ~750-1500 Hz (F1 alto, F2 bajo)
  - Banda 5: ~1500-3000 Hz (F2, F3)
  - Banda 6: ~3000-6000 Hz (formantes altos)
  - Banda 7: ~6000-8000 Hz (fricativas, ruido)

  Para cada banda se calcula mean(log(1+|STFT|)). Esto da la distribución promedio de energía espectral.

  8 band energy stds: La variabilidad temporal de cada banda dentro de la ventana. Captura si la energía es estable o fluctúa (indica transiciones
  fonéticas).

  F0 (3 dims): Media, desviación estándar y rango del pitch fundamental, estimado por autocorrelación frame a frame (busca el pico de autocorrelación entre
  50-500 Hz). Solo frames voiced (pico > 0.3).

  Voicing fraction (1 dim): Fracción de frames con energía por encima del 1% del máximo. Indica cuánto de la ventana es voz vs silencio.

  2. ¿Por qué estas features funcionan cross-modalmente?

  El F0 es idéntico en ambas señales porque ambas miden la misma fuente vibratoria. Si las cuerdas vocales vibran a 120 Hz, tanto el micrófono como el EGG
  registran 120 Hz.

  Las band energies son más sutiles: el EGG captura directamente el patrón de contacto glotal (rico en armónicos), mientras el speech captura eso filtrado
  por el tracto vocal. Pero la distribución relativa de energía entre bandas está correlacionada porque la fuente es la misma.

  La voicing fraction también coincide: cuando hay voz, ambos sensores la detectan; cuando hay silencio, ambos callan.

  3. Pool canónico con hard negatives

  Para cada query (un segmento de test), se construye un pool de 128 candidatos con estructura:

  Positivo (1): La misma ventana temporal del mismo clip, pero de la otra modalidad. Es decir, si el query es speech[1.0:3.0] del clip X, el positivo es
  egg[1.0:3.0] del clip X.

  L1 — Same clip, different window (hasta 16): El negativo más duro. Es del mismo clip (mismo speaker, mismo momento de grabación) pero de una ventana
  temporal diferente no solapada (separación ≥ 2s). Testea: ¿el modelo simplemente identifica el clip por su "firma" acústica global, o realmente alinea el
  contenido temporal? Si L1 engaña al modelo, la señal es por identidad de clip, no por alineación fina.

  L2 — Same speaker, different utterance (hasta 16): Mismo hablante, pero diciendo otra cosa. Testea: ¿el modelo reconoce al speaker en vez de alinear
  contenido? Las características vocales (F0 medio, timbre) son similares entre utterances del mismo speaker.

  L3 — Different speaker, same sentence_id (hasta 16): Otro hablante diciendo lo mismo. Testea: ¿el modelo reconoce el contenido verbal? En la práctica esto
   quedó sparse (avg 2.0 por query) porque pocos test speakers comparten sentence_ids.

  L4 — Random (resto hasta completar 127): Speaker y utterance diferentes. El caso más fácil de distinguir.

  La idea es que si el retrieval solo funciona contra L4 (random) pero falla contra L1 (mismo clip), la señal es trivial. Un resultado fuerte requiere que
  el positivo supere incluso a los negativos duros.

  4. Métodos de retrieval

  Raw cosine: Simplemente calcula la similitud coseno entre el vector de 20 dims del speech query y el vector de 20 dims de cada EGG candidato. Sin ningún
  aprendizaje.

  CCA (Canonical Correlation Analysis): Encuentra transformaciones lineales de ambas modalidades que maximizan la correlación entre ellas. Es el método
  lineal óptimo para alinear dos espacios de features. Se entrena en los pares de train (speech_i, egg_i) y se aplica a test.

  - Se aprenden 10 componentes canónicos
  - Cada componente define una dirección en speech-space y una en egg-space que están máximamente correlacionadas
  - En test, se proyectan los features a este espacio compartido de 10 dims y se usa cosine similarity

  Ridge regression: Predice EGG features desde Speech features (y viceversa) con regularización L2. R² mide cuánta varianza de una modalidad es predecible
  desde la otra. Es puramente diagnóstico — no se usa para retrieval.

  5. CI Bootstrap Agrupado

  Con solo 5 test speakers, los queries no son independientes: los segmentos del mismo speaker están correlacionados (comparten F0, timbre, etc.). Un
  bootstrap naïve por query sobreestimaría la precisión del CI.

  El bootstrap agrupado resamplea speakers completos (no queries individuales): en cada iteración bootstrap, se muestrean 5 speakers con reemplazo del pool
  de 5, y se computan las métricas con todos los queries de esos speakers. Esto produce CIs más honestos que reflejan la variabilidad real.

  Interpretación de los resultados

  Ridge R² = 0.851 (Speech→EGG)

  Esto es extraordinariamente alto. Significa que el 85% de la varianza de los features del EGG es predecible linealmente desde los features del speech.
  Esto confirma que la información espectral está fuertemente compartida. La dirección inversa (EGG→Speech) es menor (0.694) porque el speech tiene
  información adicional (formantes del tracto vocal) que el EGG no captura.

  CCA correlations: 0.975, 0.940, 0.920...

  Las primeras 3 componentes canónicas tienen correlación >0.92. Esto significa que existen al menos 3 ejes lineales en los que speech y EGG son casi
  idénticos. Probablemente corresponden a:
  1. F0 / energía global (~0.975)
  2. Distribución espectral baja (~0.940)
  3. Voicing pattern / dinámica temporal (~0.920)

  Las correlaciones caen gradualmente: 0.836, 0.698, 0.654, 0.572, 0.487, 0.382, 0.311 — incluso la componente 10 tiene correlación 0.31, significativa.

  CCA Retrieval S = 64.4% [57.8%, 70.2%]

  R@10 con pool=128 y random=7.8%. Obtener 64.4% es 8.2x sobre azar. El CI inferior (57.8%) está muy por encima del azar (7.8%). Esto es con features de 20
  dimensiones y un método lineal — señal masiva.

  Para comparar: en Escalón 1-C (Audio↔MIDI), el D0 neural con ~74M parámetros y VICReg obtuvo S=75.2%. Aquí, con 20 dims y CCA, ya estamos en 64.4%. La
  señal cross-modal Speech↔EGG es mucho más fuerte que Audio↔MIDI.

  Raw cosine S = 46.8% [38.0%, 54.5%]

  Incluso sin CCA, la similitud coseno directa en features crudos da 46.8% — 6x sobre azar. Esto indica que los features de speech y EGG viven en espacios
  naturalmente similares (mismas bandas de frecuencia → mismos valores numéricos cuando la fuente es la misma).

  Strata analysis

  - L1 avg 6.1 por query: los clips son ~5s, con ventanas de 2s y hop 0.5s dan ~7 ventanas, pero la restricción de no-overlap (≥2s apart) reduce los
  candidatos. Pocos, pero los que hay son los más duros.
  - L2 avg 16.0: el target completo. Cada test speaker tiene ~60 clips × ~7 ventanas, suficientes.
  - L3 avg 2.0: sparse, como predije del análisis de overlap de sentence_ids.
  - L4 avg 102.9: rellena el pool hasta 127.

  El rank analysis muestra median_rank_s2e=9.0 y median_rank_e2s=12.0. En un pool de 128, un median rank de 9-12 es muy bueno — la mitad de las queries
  encuentran el positivo en el top 9-12.

  Implicaciones para el proyecto

  1. H3b viable: La hipótesis de que la representación relacional puede transferirse a sensores vocales distintos tiene terreno fértil. La señal base es
  fuerte.
  2. El neural tiene margen para ganar: CCA obtiene 64.4% con 20 dims lineales. Un encoder neural debería poder capturar relaciones no-lineales y features
  más ricos, potencialmente superando esto significativamente.
  3. Los descriptores tienen sentido aquí: Si los descriptores de ratios de F0 (V4) capturan información que los features simples no, deberían mejorar sobre
   esta baseline CCA. Si NO mejoran, hay dos lecturas: o los features simples ya capturan todo, o los descriptores no aportan señal en este dominio.
  4. La asimetría S2E > E2S es esperable: Speech tiene más información (formantes, fricativas) que EGG (solo contacto glotal). Predecir EGG desde speech es
  más fácil que al revés.









  


  Los tres métodos de S2-P1 explicados                                                                                                                      
                                                                                                                                                            
  El setup común                                                                                                                                          

  Cada segmento de 2s genera un vector de 20 features tanto para Speech como para EGG:
  - 8 band energies: log-magnitud promedio en 8 bandas de octava (47-125, 125-250, ..., 6000-8000 Hz), calculadas con STFT a 16kHz
  - 8 band stds: desviación estándar temporal de la energía en cada banda (cuánto varía la energía dentro de esos 2s)
  - 3 F0 stats: F0 mediana, std de F0, y fracción de frames voiced (estimados por autocorrelación)
  - 1 voicing fraction: proporción del segmento que tiene actividad vocal

  Entonces: un segmento de speech → vector [20], un segmento de EGG → vector [20]. Mismas 20 dimensiones, misma definición, pero calculadas sobre señales
  físicamente distintas.

  ---
  Método 1: Raw Cosine Retrieval (S = 46.8%)

  Idea: ¿los vectores crudos de Speech y EGG del mismo momento temporal ya se parecen sin ningún procesamiento?

  Cómo funciona:
  1. Tomo un segmento speech de test, calculo su vector [20]
  2. Armo un pool de 128 segmentos EGG (el positivo verdadero + 127 negativos estructurados por dificultad)
  3. Calculo similitud coseno entre el vector speech query y cada vector EGG del pool
  4. Veo si el positivo verdadero cae dentro del Top-10

  Qué mide: correlación directa entre las distribuciones espectrales de Speech y EGG, sin aprender nada. Es el piso más bajo posible — si esto ya funciona,
  hay señal cruda.

  Resultado: S=46.8% (vs 7.8% random). Hay señal cruda fuerte — tiene sentido porque ambas señales provienen de las mismas cuerdas vocales, así que
  comparten F0 y estructura armónica.

  ---
  Método 2: CCA Retrieval (S = 64.4%)

  Idea: ¿existe una proyección lineal que alinee mejor los espacios Speech y EGG?

  CCA (Canonical Correlation Analysis) busca 10 pares de direcciones (una en el espacio Speech, una en el espacio EGG) tal que la correlación entre las
  proyecciones sea máxima. Es como encontrar las "dimensiones compartidas" óptimas entre ambos espacios.

  Cómo funciona:
  1. Train: Con los 19,910 pares (speech, egg) del train set, CCA aprende 10 componentes — dos matrices W_speech [20×10] y W_egg [20×10]
  2. Test: Proyecto cada vector speech y egg al espacio CCA de 10 dims: z_speech = X @ W_speech, z_egg = X @ W_egg
  3. Retrieval: Igual que raw cosine, pero en el espacio CCA de 10 dims

  Qué mide: la máxima información lineal compartida entre Speech y EGG. Las correlaciones canónicas (0.975, 0.940, 0.920, ...) te dicen cuánta varianza
  compartida hay en cada componente. Top-3 > 0.92 significa que hay al menos 3 dimensiones fuertemente acopladas.

  Resultado: S=64.4% — +17.6pp sobre raw. La proyección lineal captura estructura que el coseno crudo no ve (por ejemplo, rescala bandas de frecuencia que
  tienen rangos de magnitud muy diferentes entre Speech y EGG).

  CI grouped: [57.8%, 70.2%]. El bootstrap agrupa por speaker (resampling de 5 test speakers) para no subestimar la incertidumbre — con solo 5 speakers, un
  speaker atípico puede mover mucho la métrica.

  ---
  Método 3: Ridge Regression (R² = 0.851 Speech→EGG)

  Idea: ¿puedo predecir los features de EGG a partir de los de Speech (y viceversa)?

  Cómo funciona:
  1. Entreno una regresión Ridge (lineal con regularización L2) que toma el vector [20] de speech y predice el vector [20] de EGG
  2. Mido R² en el test set: ¿qué proporción de la varianza de EGG se explica linealmente desde Speech?
  3. Hago lo mismo en dirección inversa (EGG→Speech)

  Qué mide: causalidad predictiva lineal. R²=0.851 Speech→EGG significa que el 85% de la varianza de los features de EGG se puede predecir linealmente desde
   los features de Speech. Esto es muy alto.

  Asimetría: Speech→EGG (R²=0.851) >> EGG→Speech (R²=0.694). Tiene sentido físico: Speech contiene más información (armónicos + resonancias del tracto vocal
   + ruido de articulación) que EGG (que es básicamente la fuente glótica pura). Predecir la fuente desde la señal completa es más fácil que predecir la
  señal completa desde solo la fuente.

  ---
  Resumen visual

  Raw cosine:  Speech[20] ·cos· EGG[20]           → S=46.8%  (señal cruda)
  CCA:         Speech[20] →W→ z[10] ·cos· z[10] ←W← EGG[20]  → S=64.4%  (proyección óptima)
  Ridge:       Speech[20] →β→ ÊGG[20]  (R²=0.851)             (predicción directa)

  Los tres juntos dicen: la relación Speech↔EGG es masivamente lineal. El desafío para P2 (neural) es superar S=64.4% — si lo logra, el modelo neural está
  capturando estructura no-lineal que CCA no puede ver.