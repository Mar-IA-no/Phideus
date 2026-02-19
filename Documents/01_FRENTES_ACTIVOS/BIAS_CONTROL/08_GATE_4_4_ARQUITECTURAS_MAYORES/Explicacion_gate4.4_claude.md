  ---                                                                                                                                      
  Gate 4.4 — Arquitecturas Mayores: La Explicación                         
                                                                                                                                           
  De dónde venimos                                                         

  Gate 4.3 respondió la pregunta "¿qué descriptor y qué mecanismo de inyección maximiza la señal de ratios?". Probamos 13 brazos y
  aprendimos tres cosas fundamentales:

  1. d4a4 (concatenar A4 al audio + D4 al MIDI, same-modality) = campeón absoluto: S=83.6% a 30 epochs from scratch
  2. Reverse cross-attention (Q=descriptor, K/V=features) supera a forward cross-attention por 4-6pp — los ratios funcionan mejor como
  principio organizador que como fuente de consulta
  3. Cross-modal injection (inyectar descriptores de una modalidad en la otra) destruye señal: -7.8pp

  Pero todo lo que probamos en Gate 4.3 fueron variaciones sobre el mismo tema: tomar los descriptores de ratios y pegarlos al encoder
  existente, ya sea concatenándolos, haciendo cross-attention, o invirtiendo la dirección. El encoder en sí nunca cambió.

  Qué pregunta Gate 4.4

  ¿Qué pasa si cambiamos la arquitectura del modelo, no solo cómo le pegamos los descriptores?

  Tres familias de cambio, cada una con una hipótesis distinta:

  ---
  Familia 1: Third Tower (t3-tri, t3-anc, t3-wt)

  Idea: Darle a los ratios su propia torre — un encoder independiente que vive al lado de la torre de audio y la torre de MIDI. En vez de
  "pegar" los ratios a un encoder existente, los ratios se procesan por su cuenta y convergen con las otras dos modalidades en el espacio
  latente.

  Torre Audio          Torre Ratios          Torre MIDI
  Waveform             A4+D4 concat          MIDI Events
      |                     |                    |
  MERTEncoderLite      Transformer 2L        Transformer 4L
  d=1024               d=256                 d=512
      |                     |                    |
  Projection           Projection            Projection
      |                     |                    |
  audio_emb [256]      ratio_emb [256]       midi_emb [256]
      \                     |                    /
       =========== VICReg en el latente =========


  La torre de ratios toma ambos descriptores (A4 del audio: 8 dimensiones de deltas log-frecuencia, y D4 del MIDI: 4 dimensiones de
  intervalos locales), los concatena en [B, 188, 12], y los procesa con un Transformer liviano (2 capas, d=256, ~2.3M params).

  Los tres brazos prueban hipótesis distintas sobre la loss:

  Brazo: t3-tri
  Loss: (VICReg_am + VICReg_ar + VICReg_mr) / 3
  Pregunta: ¿Un puente triangular explícito mejora sobre d4a4?
  ────────────────────────────────────────
  Brazo: t3-anc
  Loss: (VICReg_ar + VICReg_mr) / 2
  Pregunta: La más audaz: ¿pueden los ratios solos bridgear audio↔MIDI, sin loss directa entre ellos?
  ────────────────────────────────────────
  Brazo: t3-wt
  Loss: VICReg_am + 0.3 × (VICReg_ar + VICReg_mr)/2
  Pregunta: Ratios como regularizador suave sobre d4a4

  t3-anc es el brazo más importante científicamente. No tiene loss audio↔MIDI — le pide al modelo que alinee audio con ratios y MIDI con
  ratios, y que la alineación audio↔MIDI emerja del hecho de que ambos pasan por el mismo puente. Si funciona, es evidencia directa de que
  los ratios son un lenguaje compartido entre modalidades. Esto es el corazón de la tesis de Phideus.

  Detalle: t3-tri y t3-wt usan d4a4-concat en los encoders base (el mecanismo ganador de Gate 4.3), mientras que t3-anc usa encoders
  vanilla sin ninguna inyección — para aislar la hipótesis limpiamente.

  ---
  Familia 2: FiLM (film-a4, film-d4, film-dual)

  Idea: En vez de concatenar descriptores a la entrada del encoder, usarlos para modular dinámicamente lo que cada capa del Transformer
  computa. Feature-wise Linear Modulation: el descriptor genera un par (gamma, beta) por capa, y después de cada capa del Transformer se
  aplica:

  output = (1 + gamma) × layer_output + beta


  Es como decirle al encoder "prestá atención a estas cosas" en cada capa, en vez de darle información extra en la entrada y esperar que la
   aproveche.

  Descriptor (ej: A4 [B, 8])
      |
     MLP → (gamma, beta) × 4 capas
      |
      v
  Layer 1 → FiLM → Layer 2 → FiLM → Layer 3 → FiLM → Layer 4 → FiLM → pool → proj


  El truco: la última capa del generador FiLM se inicializa en cero, así que al inicio del training gamma=0 y beta=0, y FiLM es una
  identidad — no desestabiliza nada. El modelo aprende gradualmente cuánto modular.

  Los tres brazos:

  Brazo: film-a4
  Qué modula: A4 modula las 4 capas del Transformer de audio
  Pregunta: ¿Los ratios de frecuencia mejoran la representación de audio como modulación?
  ────────────────────────────────────────
  Brazo: film-d4
  Qué modula: D4 modula las 4 capas del Transformer de MIDI
  Pregunta: ¿Los intervalos locales mejoran MIDI como modulación?
  ────────────────────────────────────────
  Brazo: film-dual
  Qué modula: Ambos
  Pregunta: ¿La modulación dual es mejor que single?

  Punto clave: el lado que NO tiene FiLM usa encoding vanilla (sin d4a4 concat). Así la comparación es limpia: la única variable es la
  modulación FiLM.

  ---
  Familia 3: MoE (moe-a4, moe-dual)

  Idea: Después de cada capa del Transformer, agregar un "adaptador" con múltiples expertos (2 FFNs pequeños). Un router — condicionado por
   el descriptor — decide cuánto peso darle a cada experto, token por token. El modelo decide dinámicamente cuándo y cómo usar la
  información de ratios a nivel de frame.

  Layer output [B, T, D]
      |
      +--→ Router([features; descriptor]) → weights [B, T, 2]
      |
      +--→ Expert 0 (FFN bottleneck D→D/4→D)
      +--→ Expert 1 (FFN bottleneck D→D/4→D)
      |
      v
  weighted_sum → residual add → next layer


  El router toma cada token concatenado con un resumen del descriptor, y produce weights de softmax sobre los 2 expertos. Los expertos son
  FFNs tipo adapter (bottleneck D/4) inicializados cerca de cero para estabilidad.

  Dos métricas de salud del MoE:
  - Load balance (loss auxiliar, 0.01×): que ambos expertos reciban trabajo similar globalmente → estabilidad
  - Segment preference variance: que distintas muestras del batch prefieran distintos expertos → especialización real, no redundancia

  ┌──────────┬──────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
  │  Brazo   │                              Dónde                               │                        Pregunta                        │
  ├──────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ moe-a4   │ MoE adapters en las 4 capas de audio, router condicionado por A4 │ ¿El routing experto mejora la representación de audio? │
  ├──────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ moe-dual │ MoE en audio (A4) + MoE en MIDI (D4)                             │ ¿Dual MoE es mejor?                                    │
  └──────────┴──────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘

  ---
  Protocolo experimental

  Fase 1 — Screening 5 epochs (en UNC Mendieta, 8 GPUs paralelo):
  - Los 8 brazos, partiendo de foundation_locked_e25.pt, freeze-policy run-d
  - ~3h total gracias al paralelismo de UNC
  - Threshold: S > 72% a epoch 5 para avanzar (vs d4a4 69.8% a 5ep)
  - Excepción: t3-anc tiene umbral especial de S > 45% — si los ratios solos bridgean algo, es hallazgo científico independiente del
  performance absoluto

  Fase 2 — Runs largos 30 epochs (solo ganadores):
  - From scratch, 30 epochs completos
  - Comparación contra d4a4-scratch (S=83.6%, el récord)

  GO/NO-GO final:

  ┌────────────────────────────────┬────────────────────────────────────────────────────────┐
  │            Criterio            │                         Umbral                         │
  ├────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ S(mejor brazo) > 85.6%         │ GO — nueva arquitectura supera récord                  │
  ├────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ S(mejor brazo) 83-85.6%        │ PARTIAL GO — comparable, interés científico            │
  ├────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Todos < 83%                    │ NO-GO — d4a4 concat sigue campeón                      │
  ├────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ t3-anc > 50%                   │ HALLAZGO — ratios solos bridgean (independiente de GO) │
  ├────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ MoE: balance + especialización │ Evidencia de routing útil                              │
  └────────────────────────────────┴────────────────────────────────────────────────────────┘

  ---
  En resumen

  Gate 4.4 es la pregunta: "¿los ratios pueden hacer más que pegarse a la entrada del encoder?"

  - Third Tower: ¿como modalidad propia?
  - FiLM: ¿como modulación interna?
  - MoE: ¿como criterio de routing experto?

  Y dentro de eso, t3-anc es la apuesta grande: si funciona, demuestra que los ratios son el puente entre audio y MIDI, no solo una feature
   auxiliar que mejora cada encoder por separado. Eso es exactamente la tesis de Phideus.