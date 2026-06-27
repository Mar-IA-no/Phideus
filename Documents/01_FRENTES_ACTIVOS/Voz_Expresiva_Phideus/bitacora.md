● Qué salió bien en Voz Expresiva Phideus

  Fase 0A — viabilidad del descriptor

  Pregunta:  ¿Familia A (V4-lin + H-series) discrimina emociones más que un control no-armónico?
  Método:    eta² robusto sobre 17 500 utts EN, ANOVA por dimensión, control = A4-16k bandas espectrales.

  Salió bien:
    H-series      eta² = 0.385   ← señal fuerte
    control C     eta² = 0.076   ← señal débil
    ratio         5×            ← descriptor armónico domina sobre control

  Conclusión:  el descriptor captura varianza específica del fenómeno emocional,
               no es ruido común a cualquier descriptor espectral.

  Fase 0B — descriptor solo con clasificadores clásicos

  Pregunta:  ¿el descriptor solo (sin SSL) clasifica emoción con LogReg / SVM bajo LOSO?
  Método:    Familia A pooled 48d sobre LOSO 10-fold, dos normas (N-strict / N-adapt 3 repeats).

  Salió bien:
    N-adapt:
      A-only > C-only            ← especificidad ratio robusta
      A+D > C+D                  ← Familia A aporta sobre eGeMAPSv02
      C+D < D-only               ← control no agrega
      A+D > D-only (clf-dep)     ← mejora pequeña pero consistente

  Conclusión:  con calibración per-speaker, el descriptor tiene firma propia
               y no es redundante con eGeMAPSv02 (88 functionals estándar).

  No salió:
    N-strict     ≈ chance         ← descriptor solo no escapa speaker-independent estricto
                                    → motivó Fase 1 con SSL como techo más alto

  Fase 1 — descriptor inyectado en WavLM frozen

  Pregunta:  ¿WavLM levanta el techo de 0B, y Phideus aporta sobre WavLM solo?
  Método:    LOSO 10-fold × 4 configs × 2 norm × 3 seeds = 240 runs.

  Salió bien:

    Baseline WavLM-only escapa chance fuerte:
      UAR N-strict = 0.698  (chance = 0.20)
      → SSL resuelve buena parte del problema

    Concat pasa formalmente el contraste primario en N-strict:
      Δ = +0.039  CI95 [+0.019, +0.060]  P(Δ>0) = 1.00
      → primera evidencia honesta de transferencia Phideus a habla expresiva con SSL
      → encaja con el primer escenario prefigurado del plan

    N-adapt: los tres mecanismos pasan robustos:
      concat +4.4pp   CI95 [+0.022, +0.063]   P=1.00
      film   +4.1pp   CI95 [+0.022, +0.061]   P=1.00
      xattn  +4.4pp   CI95 [+0.028, +0.063]   P=1.00
      → la calibración per-speaker estabiliza la señal uniformemente

    Disociación CKA reveladora (no buscada, emergió de los datos):
      concat / xattn   CKA ~0.23   ← reorganizan geometría
      film             CKA ~0.85   ← modula sin reorganizar
      → FiLM logra el efecto funcional manteniendo la representación
         geométricamente cercana al baseline
      → hallazgo interpretativo de primera línea para el libro

  Lo metodológico que salió bien (transversal)

  Plan mode iterado ~8 rondas con Codex antes de escribir código.
    → cero ajustes de protocolo durante la corrida.
    → 12 decisiones congeladas auditables se respetaron sin desviación.

  Pre-cache strategy bien dimensionada.
    → estimación original 2-3 días GPU; corrida real 6.9 h.
    → 240 runs sin un solo fallo, sin un solo NaN.

  Trazabilidad de calibración (calib_manifest.json + SHA256).
    → bit-exact reproducible quién fue calibración y quién fue evaluación.

  Spike Fase 1.0 pre-implementación.
    → detectó que mecanismos E2 no eran drop-in para WavLM antes de
       comprometer el código a esa asunción.
    → reimplementación honesta vs import directo, mecanismos paritarios
       frame-level para comparación limpia.

  Disciplina contra selección post hoc:
    → 1 calib repeat N-adapt congelado de entrada (no ampliado al ver resultados).
    → contrastes por mecanismo (no "mejor mecanismo vs baseline").
    → bootstrap sobre per-speaker values, no sobre runs.
    → estatuto de N-adapt declarado como secundaria menos estable que la 0B.

  Arco completo en una línea

  0A: el descriptor tiene firma propia.
  0B: solo no alcanza en estricto, pero es específico bajo calibración.
  1:  inyectado en WavLM frozen, aporta evidencia formal en estricto (concat)
      y uniforme en adaptativo (los tres), con disociación geométrica entre
      mecanismos que abre la lectura interpretativa.