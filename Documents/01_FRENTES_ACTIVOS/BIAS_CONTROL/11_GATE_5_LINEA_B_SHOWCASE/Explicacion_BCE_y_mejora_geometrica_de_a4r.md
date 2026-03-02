● BCE — Binary Cross-Entropy                                                                                                                                
                                                                                                                                                            
  En nuestro contexto, el decoder produce un piano roll: una matriz [188 frames, 88 teclas] donde cada celda es la probabilidad de que esa nota esté sonando
   en ese frame. El ground truth es binario: 1 si la nota suena, 0 si no.
                                                                                                                                                            
  BCE mide qué tan bien calibradas están esas probabilidades:

  BCE = -( y·log(p) + (1-y)·log(1-p) )

  - Si la nota suena (y=1) y el modelo predice p=0.95 → BCE ≈ 0.05 (bueno)
  - Si la nota suena (y=1) y el modelo predice p=0.50 → BCE ≈ 0.69 (malo)
  - Si la nota NO suena (y=0) y el modelo predice p=0.30 → BCE ≈ 0.36 (mediocre)

  BCE bajo = el modelo asigna probabilidades cercanas a la verdad. BCE alto = el modelo está confundido o mal calibrado.

  En nuestros resultados:
  - BCE 0.83 (D0 pool-188): El modelo no está seguro de casi nada. Sobre un piano roll donde ~5% de las celdas tienen nota, predice probabilidades tibias en
   vez de decisiones claras.
  - BCE subiendo durante training (0.76 → 0.84 para D0): El modelo mejora F1 (se vuelve ligeramente más selectivo) pero su calibración general empeora — un
  signo clásico de overfit. Aprende patrones específicos del train set que no generalizan.

  La diferencia entre F1 y BCE: F1 mide la calidad de las decisiones binarias (después de aplicar un threshold de 0.1). BCE mide la calidad de las
  probabilidades antes del threshold. Podés mejorar F1 mientras empeorás BCE — y eso es exactamente lo que pasa acá.

  ---
  Qué significa "ventaja puramente geométrica"

  Esto requiere desempacar con cuidado porque es la conclusión central de Gate 5B.

  El espacio de embeddings

  Después del encoder + pooling + proyección, cada segmento de audio se convierte en un vector z_audio ∈ ℝ²⁵⁶ y cada segmento MIDI en z_midi ∈ ℝ²⁵⁶.
  Retrieval funciona por distancia: dado un query de audio, buscamos el MIDI más cercano en distancia L2.

  Dos formas en que un descriptor podría mejorar el retrieval

  Hipótesis A — Enriquecimiento informacional:
  El descriptor hace que las representaciones internas del encoder codifiquen MÁS información musical. Las features de a4r "saben más" sobre la música que
  las de D0. Si congelás el encoder y ponés un decoder encima, a4r debería poder reconstruir más fielmente la partitura.

  Hipótesis B — Reorganización geométrica:
  El descriptor no agrega información nueva a las features internas. La información musical presente es la misma. Pero cambia CÓMO esa información se
  distribuye en el espacio de embeddings — la geometría de las distancias entre vectores.

  Qué dice la evidencia

  ┌───────┬────────────────────────────────────────────────┬─────────────────────────┬─────────────┐
  │ Test  │                    Qué mide                    │        Resultado        │  Favorece   │
  ├───────┼────────────────────────────────────────────────┼─────────────────────────┼─────────────┤
  │ 13G-B │ ¿Features internas más ricas?                  │ F1~10% para TODOS       │ Hipótesis B │
  ├───────┼────────────────────────────────────────────────┼─────────────────────────┼─────────────┤
  │ 03    │ ¿Embeddings más linealmente decodificables?    │ D0 GANA en probe lineal │ Hipótesis B │
  ├───────┼────────────────────────────────────────────────┼─────────────────────────┼─────────────┤
  │ 06    │ ¿Geometría cross-encoder más alineada?         │ Descriptores +82% CKA   │ Hipótesis B │
  ├───────┼────────────────────────────────────────────────┼─────────────────────────┼─────────────┤
  │ 08    │ ¿Relación lineal entre descriptor y embedding? │ Correlaciones           │ r           │
  ├───────┼────────────────────────────────────────────────┼─────────────────────────┼─────────────┤
  │ 02    │ ¿La info del descriptor importa?               │ +9.4pp causal           │ (Ambas)     │
  └───────┴────────────────────────────────────────────────┴─────────────────────────┴─────────────┘

  Todo apunta a B.

  Qué implica concretamente

  Imaginá dos espacios de embeddings, ambos con la misma "cantidad de información" sobre la música:

  D0 (sin descriptor):                a4r (con descriptor):

     A₁  M₃                              A₁ M₁
      ·    ·                               · ·
           M₁                          A₂ M₂
     A₂ ·  ·                            · ·
      ·        M₂
                ·                       A₃ M₃
     A₃                                 · ·
      ·

  Audio y MIDI del mismo segmento      Audio y MIDI del mismo segmento
  están "cerca" pero no consistentemente  están CONSISTENTEMENTE cerca

  En D0, audio₁ y midi₁ pueden estar cerca, pero la distancia relativa entre pares no es consistente. A veces midi₃ está más cerca de audio₁ que midi₁. El
  modelo "sabe" la información pero no la organiza de forma que la distancia L2 sea un buen proxy de similitud.

  En a4r, el descriptor actúa como una fuerza organizadora que alinea las distancias: segmentos musicalmente similares → vectores geométricamente cercanos,
  de manera más consistente.

  La analogía

  Pensá en una biblioteca. Dos bibliotecas tienen exactamente los mismos libros (misma información). Pero una tiene los libros organizados por color de
  tapa, y la otra por tema. Si buscás "libros sobre física", la segunda biblioteca te da mejores resultados — no porque tenga más libros, sino porque la
  organización (geometría) hace que la búsqueda por proximidad funcione mejor.

  El descriptor de A4 (dinámica espectral temporal) actúa como un principio organizador: fuerza al encoder a disponer sus embeddings de forma que segmentos
  con dinámica espectral similar queden cerca. Como la dinámica espectral es un buen proxy de contenido musical, esa organización mejora el retrieval.

  Qué implica para el proyecto

  Buenas noticias:
  - La ventaja es real, causal y cuantificada (+9.4pp)
  - El mecanismo está identificado: reorganización geométrica vía A4
  - Es publicable y científicamente sólido

  Limitaciones:
  - No podés clamar que "los descriptores hacen que el encoder entienda mejor la música". Entiende lo mismo — organiza diferente.
  - Un decoder downstream (transcripción, generación) probablemente NO se beneficiaría del descriptor, porque la info en las features pre-pooling es la
  misma.
  - La ventaja está atada a tareas basadas en distancia (retrieval, nearest-neighbor). Para tareas que requieren decodificar contenido (transcripción,
  clasificación lineal), D0 puede ser igual o mejor.

  Para el paper:
  El claim correcto es: "Spectral temporal descriptors reorganize the embedding geometry such that L2 distances better reflect musical similarity, yielding
  +9.4pp retrieval improvement without enriching the encoder's internal representations."

  Eso es un hallazgo limpio, honesto, y bastante interesante — porque dice algo sobre cómo VICReg organiza representaciones y cómo señales auxiliares pueden
   guiar esa organización sin necesariamente agregar información nueva al proceso.

