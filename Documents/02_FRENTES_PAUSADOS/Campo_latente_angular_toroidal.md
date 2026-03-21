● Pregunta muy buena, pero tengo que contradecirte en un punto antes de llegar a lo interesante.

  La tensión central

  El propio Ch10 dice que φ es para retrieval, no para storage. VICReg opera durante el training — eso es la fase de storage: construir una estructura cross-modal estable.
  Reorganizar el espacio de entrenamiento con φ contradice la lógica del capítulo que acabamos de incorporar. Los ratios enteros (recurrencia, mode-locking, estabilidad) son los
  que deben gobernar el storage. φ entra cuando querés leer lo almacenado sin destruirlo.

  Dicho eso — la intuición de que el espacio plano de VICReg es un cuello de botella es legítima. Y hay una formulación que sí es consistente con Ch10.

  Lo que sí se puede hacer (y es coherente)

  Separar las dos fases: training en toro, retrieval con φ-probes.

  1. Training en T^N (espacio toroidal): En vez de z ∈ ℝ^256 con distancia euclídea, z ∈ T^N con distancia angular. Cada dimensión es una fase θ_i ∈ [0, 2π). VICReg se reformula:
    - Invariance: distancia geodésica en el toro (no coseno plano)
    - Variance: dispersión angular (no varianza euclídea)
    - Covariance: independencia circular entre dimensiones

  Esto ya no es VICReg flat — es un espacio con topología intrínseca que favorece relaciones proporcionales (las fases se relacionan por ratios).
  2. Retrieval con φ-shifted probes: En vez de buscar el vecino más cercano por coseno, recorrer el toro con un shift de φ que muestrea toda la estructura almacenada sin
  mode-locking con ningún componente. Exactamente la lógica del capítulo: el probe no necesita saber qué hay almacenado, φ garantiza cobertura uniforme.

  Pero — dónde testearlo

  No en Escalón 2. Tenemos un null de 15 condiciones. Cambiar la geometría no va a arreglar un par de modalidades que no comparten estructura descriptorial.

  Los candidatos son:

  - Escalón 3 (Lissajous): El testbed natural. Ground truth determinista, φ tiene significado físico directo (ratio de frecuencias XY), y podemos generar escenas con ratios enteros
   vs φ-ratios. Es literalmente el experimento que el material de Nicolás proponía.
  - Escalón 1 retrospectivo: Ya tenemos 84.1% en flat VICReg. Si un toro + φ-probes mejora eso, sería evidencia fuerte. Pero es arriesgado: si no mejora, no sabemos si es porque
  flat ya era suficiente o porque la implementación toroidal tiene bugs.

  El experimento mínimo (factible ahora)

  Antes de tocar VICReg, hay un experimento más chico que testea la idea sin reescribir el framework:

  1. Tomar los embeddings ya entrenados de Escalón 1 (flat, 256d)
  2. Proyectarlos a un toro (normalizar a ángulos)
  3. Comparar retrieval con coseno estándar vs φ-shifted traversal
  4. Si el φ-traversal recupera más estructura que coseno plano sobre los mismos embeddings, entonces vale la pena reformular el training