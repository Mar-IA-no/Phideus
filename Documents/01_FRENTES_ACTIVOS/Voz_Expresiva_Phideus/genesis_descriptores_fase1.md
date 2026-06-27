De dónde salieron los descriptores de Fase 0B

  Los cuatro vectores que 0B usa para sus clasificadores clásicos vienen del extractor de Fase 0A (experiments/voz_expresiva/0A_extract.py → src/voz_expresiva/compound_descriptor.py). Cada familia tiene
  origen distinto y rol distinto en el experimento.

  Familia A — Phideus-ratio                       48d (pooled)
  ────────────────────────────────────────────────────────────
    V4-lin        4 dims frame-level @ 100 Hz
    H-series      8 dims frame-level @ 100 Hz
                                          ─────────────────
                                          12 dims frame-level
                                          × 4 stats (mean, std, max, min) pool
                                          = 48 dims utterance-level

    origen del código:
      src/bias_control/vocal_descriptors.py
        compute_v4_linear()   ← intervalos armónicos lineales sobre F0
        compute_h_series()    ← presencia relativa de 8 armónicos sobre fundamental

    origen del experimento:
      descriptor canónico de Phideus, validado en Escalón 1
      sobre MAESTRO (música, audio↔MIDI).
      Cruzó al frente Voz Expresiva sin cambios — misma firma.

    rol en 0B:
      el descriptor HIPÓTESIS. Si Phideus aporta, A es lo que aporta.

  Familia B — Voice quality clásica                9d
  ────────────────────────────────────────────────────────────
    7 medidas directas (utterance-level):
      HNR              harmonics-to-noise ratio
      CPP              cepstral peak prominence
      jitter           variación cíclo-a-cíclo de F0
      shimmer          variación cíclo-a-cíclo de amplitud
      F2/F1            ratio formantes 2/1
      F3/F1            ratio formantes 3/1
      alpha-ratio      energía aguda/grave en dB
    2 proxies (utterance-level):
      H1-H2 proxy      tilt espectral en las bajas
      H1-A3 proxy      tilt espectral hacia el primer formante alto

    origen del código:
      src/voz_expresiva/voice_quality.py
        compute_voice_quality()

    origen del experimento:
      bibliografía clásica de fonación expresiva
      (Awan, Heman-Ackah, Kreiman, Hanson, Stevens).
      Construido específicamente para el frente — no preexistía.

    rol en 0B:
      descriptor COMPETIDOR / COMPLEMENTARIO. Mide características
      de fonación reconocidas en la literatura. Permite contrastes
      A vs B, A+D vs B+D, etc.

  Familia C — Control no-ratio                    32d (pooled)
  ────────────────────────────────────────────────────────────
    A4-16k        8 dims frame-level @ 100 Hz
                  × 4 stats pool
                  = 32 dims utterance-level

    origen del código:
      src/bias_control/vocal_descriptors.py
        compute_a4_16k()  ← bandas espectrales octavadas 0-8 kHz
                            (energía por banda, no ratios armónicos)

    origen del experimento:
      descriptor CONTROL del Escalón 2 (Bias Control). Mide energía
      espectral por bandas — captura el "qué frecuencias hay" SIN
      capturar relaciones armónicas. Funciona como contrafactual:
      si el descriptor no-armónico iguala al armónico, la ventaja
      de Phideus es ilusoria (cualquier descriptor espectral serviría).

    rol en 0B:
      descriptor NULL. La especificidad ratio se mide como
      A > C en métricas comparables.

  Familia D — eGeMAPSv02 baseline                 88d
  ────────────────────────────────────────────────────────────
    eGeMAPSv02 functionals: 88 dims utterance-level
      F0 stats, energy stats, MFCC stats, jitter/shimmer stats,
      spectral slope, harmonic stats, voicing, loudness,
      formant stats, ... (set extendido GeMAPS)

    origen del código:
      src/voz_expresiva/voice_quality.py
        compute_egemaps_functionals()
      (wrapper sobre opensmile-python con feature_set='eGeMAPSv02')

    origen del experimento:
      Eyben et al. 2016, set canónico de la comunidad SER
      (Speech Emotion Recognition). Es el baseline de referencia
      contra el que se mide cualquier descriptor nuevo en la literatura.

    rol en 0B:
      BASELINE DE COMPARACIÓN. Permite preguntarse si Phideus aporta
      sobre el estado del arte de descriptores clásicos: A+D vs D solo,
      C+D vs D solo. Si A+D > D y C+D ≈ D, Phideus tiene firma.

  Cadena de extracción end-to-end

  audio.wav (ESD)
      │
      ↓
  ESDLoader (src/voz_expresiva/esd_loader.py)
      │
      │   carga 16 kHz mono
      │   metadata: speaker_id, emotion, sentence_id, language
      ↓
  compute_all_descriptors() (src/voz_expresiva/compound_descriptor.py)
      │
      ├──→ extract_f0_speech() [librosa.pyin, fmin=75 Hz, fmax=500 Hz]
      │         │
      │         ↓
      │     F0 + voiced mask @ 100 Hz
      │
      ├──→ compute_v4_linear(f0)         → [1, T, 4]   ┐
      ├──→ compute_h_series(wav, f0)     → [1, T, 8]   │
      │                                                │
      │   pool_frame_level (mean/std/max/min)          │ Familia A
      │                                                │
      │     concat → 12d × 4 = 48 dims    ──────────→ ┘
      │
      ├──→ compute_voice_quality(wav, f0) → 9d         ──→ Familia B
      │
      ├──→ compute_a4_16k(wav)           → [1, T, 8]   ┐
      │                                                │ Familia C
      │   pool_frame_level                             │
      │     → 8d × 4 = 32 dims            ──────────→ ┘
      │
      └──→ compute_egemaps_functionals(wav) → 88d      ──→ Familia D


  Output (por utt):
      family_A_pooled    (48,)
      family_B           (9,)
      family_C_pooled    (32,)
      family_D_egemaps   (88,)

  Importante para la lectura del experimento

  Familia A es Phideus genuino — viene del mismo extractor que validamos en Escalón 1 sobre música (src/bias_control/vocal_descriptors.py). La transición de música a habla no rediseñó el descriptor; solo
  cambió el extractor de F0 (de CQT-pitch para piano a librosa.pyin para habla). El núcleo compute_v4_linear y compute_h_series es el mismo código que generó los embeddings de MAESTRO. Esta continuidad es
  deliberada: la pregunta del frente es si la firma armónica que funcionó en música transfiere a habla, no si una variante adaptada al habla funciona.

  Familia C cumple un rol que en muchos experimentos del campo se omite y termina sobrevalorando los hallazgos: el control negativo bien construido. Si la energía por bandas espectrales (sin estructura
  armónica explícita) clasifica emoción igual que Phideus, la especificidad armónica no existe. En 0B vimos que A > C y A+D > C+D, lo cual es exactamente la firma que sostenía la hipótesis.

  Familia D, eGeMAPSv02, es lo que cualquier reviewer de SER va a pedir como baseline. Tenerla en el experimento permite que el claim "Phideus aporta" no quede atado a un baseline débil — D es el set de
  referencia del campo. Que A+D > D bajo N-adapt (clf-dependiente, pequeño pero consistente) es la lectura que da viabilidad al frente.