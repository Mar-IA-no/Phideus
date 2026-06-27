• N-strict y N-adapt son dos políticas distintas de normalización del descriptor para el hablante de test.

  N-strict

  - Objetivo: medir generalización honesta a un hablante nuevo.
  - Train: cada hablante de train se normaliza con sus propias estadísticas.
  - Test: el hablante de test no se normaliza con stats suyas.
  - En la implementación actual, test usa solo la referencia derivada del train, no adaptación del propio speaker.
  - Interpretación: “¿el sistema funciona sobre un hablante no visto, sin calibración previa?”

  N-adapt

  - Objetivo: medir el caso donde sí permitís una pequeña calibración del hablante nuevo.
  - Del speaker de test se separan 25 utterances sin usar para evaluar.
  - Con esas 25 utts se calculan sus estadísticas de normalización.
  - El resto de las utts del mismo speaker se evalúan usando esa normalización.
  - Interpretación: “¿el sistema mejora si antes ve una pequeña muestra no etiquetada del usuario?”

  Dicho más simple:

  - N-strict = cero adaptación del speaker de test.
  - N-adapt = adaptación ligera del speaker de test con 25 utts.

  Por eso N-strict es la condición más fuerte metodológicamente, y N-adapt es más parecida a un escenario de despliegue con calibración previa.