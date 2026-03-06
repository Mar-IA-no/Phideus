  Gate 7: qué era, qué pasó, y qué significan los resultados                                                                                                
                                                                                                                                                            
  1. La pregunta                                                                                                                                            
                                                            
  Venías de Gate 5B con un hallazgo muy específico: los descriptores (A4, D4) mejoran el retrieval cross-modal porque reorganizan la geometría del espacio
  de embeddings (+82% CKA), pero Gate 13G-B mostró que esa ventaja no se traduce a decodificabilidad local — el F1 de piano roll es idéntico para todos los
  brazos (~0.10). Y Gate 6 Exp C confirmó que el plateau de AMT (~F1=0.157) es consistente con eso.

  Quedaba una ambigüedad sin resolver:

  ¿El límite es el encoder (MERTLite era demasiado pequeño/limitado), o el objetivo de entrenamiento, o A4 es genuinamente complementario incluso para
  encoders más fuertes?

  Gate 7 ataca el lado audio de esa pregunta con el test más barato y directo posible: ¿cuánta información del descriptor A4 ya está implícitamente en el
  encoder, accesible de forma lineal?

  Si MERT-330M (330M params, entrenado con 160k horas de audio) ya codifica linealmente lo que A4 captura → el encoder era una limitación relevante en
  nuestro setup. Si no → A4 retiene valor como descriptor complementario incluso para encoders más fuertes.

  ---
  2. El diseño del probe

  El setup es el más limpio posible:

  - Modelo de probe: Ridge regression de solución cerrada (W = (X'X + αI)^{-1} X'Y). Sin backpropagation, sin varianza de optimización, resultado
  completamente determinístico dado el split.
  - Features X: representación pooled de cada encoder [N, D] — un vector por segmento de 4 segundos.
  - Target Y: el descriptor A4 — 8 valores escalares, uno por banda octava (47Hz → 12kHz).
  - Split: 80/20 por pieza (todos los segmentos de la misma pieza van al mismo fold → no hay data leakage). 5 splits con semillas distintas → CIs sobre
  varianza de split.
  - Normalización: z-score fit en train, aplicado a test. Sin leakage.
  - Nulls:
    - Shuffled between: entrena con targets mezclados aleatoriamente → debería dar R²≈0. Es el gate de sanidad.
    - Dummy: predice la media de train para todos los samples de test → R²≤0.

  ---
  3. Los tres encoders

  ┌──────────────┬────────┬─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │   Encoder    │ Params │                                             Origen                                              │
  ├──────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ MERTLite-D0  │ ~60M   │ Nuestro — 4 CNN + 4 Transformer, entrenado con VICReg sobre MAESTRO. Tiene régimen cross-modal. │
  ├──────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ MERT-v1-95M  │ ~95M   │ HuggingFace — foundation model audio puro, sin régimen cross-modal.                             │
  ├──────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ MERT-v1-330M │ ~330M  │ HuggingFace — foundation model audio puro, test principal.                                      │
  └──────────────┴────────┴─────────────────────────────────────────────────────────────────────────────────────────────────┘

  La comparación no es simétrica: MERTLite fue entrenado con VICReg (régimen cross-modal), los HF no. La diferencia de R² mezcla tamaño, datos de
  pretraining y objetivo. No es solo "más grande = mejor".

  ---
  4. Los bugs que aparecieron (y cómo se resolvieron)

  Acá es donde la sesión fue larga. Cuatro bugs en secuencia:

  Bug 1: audio_encoder no encontrado

  El checkpoint D0 carga como Gate42Model, que envuelve un CrossModalModel en .base_model. El código buscaba model.audio_encoder directamente →
  AttributeError. Fix: buscar primero ahí, luego en .base_model.audio_encoder.

  Bug 2: savez_compressed colgado

  El primer intento cacheaba tanto pooled como frames. frames para MERTLite son [685, 2400, 1024] = ~6.7GB. npz.savez_compressed intentaba comprimir eso en
  RAM y se colgaba indefinidamente. Fix: cachear solo pooled [N, H]. Los frames se recomputan on-demand si se usa --frame-level.

  Bug 3: Target A4 degenerado (el más sutil)

  Este fue el problema conceptual más importante.

  El descriptor A4 que usamos en entrenamiento (compute_audio_descriptor_a4()) está diseñado como descriptor diferencial temporal: para cada frame, calcula
  la diferencia de log-magnitud respecto al frame anterior, y luego lo normaliza a zero-mean por banda dentro de cada segmento. El resultado es un
  descriptor de cambios de timbre.

  El problema: si promediás ese descriptor sobre todos los frames de un segmento para obtener una representación segment-level, el promedio es exactamente
  ≈0 por construcción (la normalización garantiza eso). Todos los segmentos tienen el mismo target ≈ 0.

  Esto es catastrófico para un probe:
  - Dummy R² = 1.0 (predecir cero siempre es perfecto)
  - Ridge R² = números absurdos negativos de millones

  Fix: En lugar de usar el descriptor A4 interno, usar directamente la media de log-magnitud STFT por banda octava — la envolvente espectral temporal
  promediada. Esto captura el mismo concepto (qué tan prominentes son las distintas regiones del espectro en ese segmento) pero varía entre segmentos y es
  un target no-degenerado.

  # NO: compute_audio_descriptor_a4() → zero-mean por construcción
  # SÍ: media de log1p(|STFT|) agrupada por bandas A4
  log_mag_mean = log1p(|STFT|).mean(dim=tiempo)   # [B, 1025]
  a4_seg = [log_mag_mean[:, lo:hi].mean() for lo, hi in A4_BAND_EDGES]  # [B, 8]

  Bug 4: Null shuffled incorrectamente implementado

  El null shuffled original mezcló tanto los targets de train como los de test antes de normalizar. Esto crea una inconsistencia: la normalización se fit
  sobre los targets mezclados (distribución diferente), y se evalúa sobre targets también mezclados pero de otra distribución → R² = -98736.

  Fix: Shufflear solo Y_train (para romper la correspondencia feature↔target durante el entrenamiento), y evaluar sobre Y_test real. Así el null testea
  exactamente lo que debe: "¿qué pasa si el modelo aprende una correspondencia al azar?"

  ---
  5. Los resultados

  ┌───────────────┬───────────┬───────┬──────┐
  │    Encoder    │ R²_global │ ±std  │ dim  │
  ├───────────────┼───────────┼───────┼──────┤
  │ MERT-v1-330M  │ 0.850     │ 0.126 │ 1024 │
  ├───────────────┼───────────┼───────┼──────┤
  │ MERTLite-D0   │ 0.734     │ 0.229 │ 1024 │
  ├───────────────┼───────────┼───────┼──────┤
  │ MERT-v1-95M   │ 0.659     │ 0.178 │ 768  │
  ├───────────────┼───────────┼───────┼──────┤
  │ Null shuffled │ -1.568    │ —     │ —    │
  ├───────────────┼───────────┼───────┼──────┤
  │ Null dummy    │ -0.038    │ —     │ —    │
  └───────────────┴───────────┴───────┴──────┘

  Señales claras:

  1. Los nulls funcionan correctamente. Shuffled en -1.568 (muy negativo, como corresponde cuando el modelo aprendió basura y predice sobre targets reales).
   Dummy en -0.038 (≈0, como esperado).
  2. La señal es real y fuerte. Los tres encoders están muy por encima de los nulls. A4 (envolvente espectral) es linealmente accesible.
  3. MERT-330M lidera con R²=0.850. Tiene +11.6pp sobre MERTLite (0.734) y +19.1pp sobre MERT-95M (0.659).
  4. MERTLite supera a MERT-95M (0.734 vs 0.659), que es interesante porque MERT-95M tiene más parámetros pero fue entrenado sin régimen cross-modal.
  5. El patrón por bandas: MERT-330M es más consistente en todo el espectro. MERTLite es fuerte en las bandas medias (bands 3-5: 0.93-0.95) pero débil en
  extremos. MERT-95M es el más inconsistente.

  ---
  6. Qué significa para el proyecto

  Reducción de ambigüedad: La envolvente espectral que A4 captura ya estaba linealmente codificada en todos los encoders, y más aún en MERT-330M. Esto
  sugiere que el encoder era una limitación relevante en nuestro setup — si hubiéramos entrenado con MERT-330M como backbone, el decoder habría tenido
  acceso más fácil a esa información.

  Lo que NO dice:
  - No prueba que "solo el encoder era el cuello de botella" — el objetivo de entrenamiento y la arquitectura cross-modal también pueden ser límites.
  - No resuelve la ambigüedad cross-modal completa — para eso habría que hacer Exp 7.1 (mini Test02 con MERT-330M como backbone), que quedó diferida.
  - R²=0.85 para la envolvente espectral no significa que MERT-330M sea mejor para retrieval cross-modal — esa es una tarea completamente distinta que Gate
  7 no toca.

  La lectura canónica: A4 es información que encoders de audio grandes ya internalizan linealmente. En el contexto de Gate 5B (ventaja geométrica, no de
  feature richness), Gate 7 agrega: esa ventaja geométrica no viene de que A4 tenga información que los encoders no tienen — viene de la manera en que A4
  organiza el espacio de embeddings al forzar alineamiento cross-modal.