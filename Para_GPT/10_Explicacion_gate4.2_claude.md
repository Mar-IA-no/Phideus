  La pregunta central                    
                                                                                                                                           
  ¿Los ratio descriptors aportan información que el modelo end-to-end no captura por sí solo?
                                                                                                                                           
  Esta es la pregunta más importante del proyecto Phideus. La tesis del proyecto es que los ratios de frecuencia constituyen un lenguaje   
  informacional universal. Gate 4.2 es el experimento diseñado para testear eso empíricamente.

  ---
  El contexto: por qué hace falta este experimento

  El foundation model (Run D-02, S=61.8%) aprendió a alinear audio y MIDI usando VICReg loss — una loss contrastiva que dice "acercá los
  embeddings de audio y MIDI del mismo segmento, alejá los de segmentos distintos". Eso funciona, pero no sabemos qué aprendió. Hay dos
  posibilidades:

  1. El modelo descubrió los ratios por sí solo — los encoders internamente ya extraen relaciones de frecuencia y las usan para alinear las
   modalidades. En este caso, darle ratios explícitamente no ayudaría (información redundante).
  2. El modelo usa otra información — timbres, envolventes de energía, patrones rítmicos macroscópicos, u otras features que correlacionan
  con la identidad del segmento pero que NO son ratios. En este caso, darle ratios explícitos debería mejorar el rendimiento.

  Si cae en (1): la tesis de Phideus sobre ratios como lenguaje cross-modal es compatible pero no necesariamente causal. El modelo los
  captura implícitamente.

  Si cae en (2) y los ratios mejoran: evidencia fuerte de que los ratios aportan señal que el modelo no descubre solo. La tesis se
  fortalece.

  Si cae en (2) y los ratios NO mejoran: los ratios no son útiles para cross-modal alignment. La tesis se debilita.

  ---
  El diseño experimental: control + 4 variantes

  Gate 4.2 compara 5 descriptores (D0-D4) contra exactamente el mismo baseline (foundation locked). Todos parten del mismo checkpoint
  inmutable, usan la misma freeze policy, los mismos hiperparámetros, el mismo seed. La ÚNICA variable es cómo se inyectan (o no) los
  ratios.

  D0 — Control (sin ratios)

  Continúa el training del foundation con VICReg puro, exactamente como lo hizo Bloque A. No recibe información de ratios. Cualquier mejora
   de D0 viene exclusivamente del entrenamiento continuado.

  D0 es la línea base: si un descriptor Dx mejora, tiene que mejorar más que D0, no más que el foundation estático.

  D1 — Pitch Ratio Histogram (auxiliary loss)

  Toma los pitches MIDI del segmento, calcula todos los ratios pairwise f_i/f_j entre notas, y construye un histograma de 128 bins en el
  rango [0.5, 2.0] con binning Gaussiano suave.

  Este histograma se pasa por un RatioEncoder (MLP) que produce un embedding de 256d. Luego se añade una loss auxiliar:

  L_total = VICReg(audio, midi) + 0.1 × [VICReg(audio, ratio) + VICReg(midi, ratio)]

  La idea: forzar al modelo a que sus embeddings de audio y MIDI se alineen también con el embedding del histograma de ratios. Si los
  ratios contienen información cross-modal relevante, este tercer "polo" de atracción debería guiar los encoders hacia mejores
  representaciones.

  Lo que testea: ¿Un resumen estadístico global de los ratios de pitch (histograma) aporta señal cross-modal adicional, inyectado como
  regularización vía loss auxiliar?

  D2 — Enriched Multi-Channel (Stage 2, condicional)

  Igual que D1 pero con 3 canales en el histograma:
  1. Ponderado por velocity (las notas fuertes pesan más)
  2. Ponderado por duración (las notas largas pesan más)
  3. Sin ponderar (como D1)

  Resultado: [B, 384] en vez de [B, 128]. Hipótesis: la expresividad musical (dinámicas, articulación) contiene información de ratios que
  un histograma plano pierde.

  Solo corre si D1 muestra señal en Stage 1.

  D3 — Temporal-Rhythmic Ratios (Stage 2, condicional)

  Explora una dimensión completamente diferente: ratios temporales, no de pitch.

  Calcula:
  - IOI ratios (Inter-Onset Interval): ratio entre duraciones sucesivas entre ataques. Captura patrones rítmicos — una blanca seguida de
  una negra tiene ratio 2:1.
  - Duration ratios: ratio entre duraciones de notas consecutivas.
  - Pitch intervals: histograma de intervalos en semitonos.

  Hipótesis: quizás no son los ratios de pitch los que importan, sino los ratios temporales/rítmicos. Si D3 funciona y D1 no, eso redirige
  toda la investigación hacia ritmo como lenguaje informacional.

  Solo corre si Stage 1 muestra alguna señal.

  D4 — Input-Augmented Local Intervals (mecanismo diferente)

  D4 testea algo fundamentalmente distinto a D1-D3. En vez de usar una loss auxiliar (agregar un tercer polo de atracción), D4 inyecta la
  información de ratios directamente en el input del MIDI encoder.

  Para cada nota del segmento, calcula 4 features locales:
  1. semitone_prev: intervalo en semitonos con la nota anterior
  2. semitone_next: intervalo con la siguiente
  3. log_ratio_prev: log2(freq_actual/freq_anterior)
  4. log_ratio_next: log2(freq_siguiente/freq_actual)

  Estos 4 valores se concatenan al embedding de cada nota (512d → 516d) y se proyectan de vuelta a 512d con un Linear(516, 512) antes de
  pasar por el positional encoding y el transformer.

  La loss es VICReg puro — sin componente auxiliar. La hipótesis es que si el transformer MIDI recibe información explícita de intervalos
  locales, puede procesarla para producir mejores embeddings.

  Lo que testea: ¿Es el mecanismo de inyección (auxiliary loss vs input augmentation) el que falla, o es la información de ratios la que no
   sirve? Si D1 falla pero D4 funciona → el problema era el mecanismo. Si ambos fallan → la información de ratios no aporta.

  ---
  La matriz de interpretación: 5 escenarios posibles

  Escenario: A
  D1 (aux loss): Mejora
  D4 (input aug): Mejora
  Qué significa: Ratios aportan señal robusta. El modelo no los captura solo. Evidencia fuerte pro-ratios, independiente del mecanismo.
  ────────────────────────────────────────
  Escenario: B
  D1 (aux loss): Mejora
  D4 (input aug): No mejora
  Qué significa: Auxiliary loss funciona. Los ratios como histograma global son informativos.
  ────────────────────────────────────────
  Escenario: C
  D1 (aux loss): No mejora
  D4 (input aug): Mejora
  Qué significa: El mecanismo era el problema, no los ratios. Input augmentation es el camino correcto.
  ────────────────────────────────────────
  Escenario: D
  D1 (aux loss): No mejora
  D4 (input aug): No mejora
  Qué significa: DROP_RATIO. El modelo ya captura los ratios implícitamente, o los ratios no son relevantes para cross-modal alignment.
  ────────────────────────────────────────
  Escenario: E
  D1 (aux loss): (D3 mejora, D1 no)
  D4 (input aug): —
  Qué significa: Ratios temporales > pitch. Redirige investigación hacia ritmo.

  Escenario D es el más probable (a priori y según los datos preliminares que estamos viendo). Pero es igualmente informativo: si un modelo
   end-to-end con 60M parámetros ya captura los ratios sin ayuda explícita, eso informa que los ratios están presentes en la señal (H1
  validada) y son aprendibles (H2 validada), solo que no necesitan ser explicitados.

  ---
  El protocolo estadístico: por qué es riguroso

  Problema: Las métricas tienen ruido inherente. El pool de evaluación (256 candidatos, 500 queries) introduce varianza estocástica. En las
   multi-seed de e25 vimos std ~1.1pp. Un delta de +0.5pp podría ser ruido.

  Solución en 2 niveles:

  1. Filtro blando: S_Dx - S_D0 >= +0.5pp AND hard_neg no degrada más de 1pp. Esto pasa candidatos a re-evaluación.
  2. Confirmación multi-seed: Re-evaluar con 4 seeds [42, 123, 456, 789]. Si el delta sobrevive → señal real. Si no → ruido.
  3. Confirmación completa (si sobrevive): 5 epochs desde foundation (run limpio), umbral más alto: S_Dx - S_D0 >= +1.5pp en al menos 2/3
  de las últimas epochs.
  4. Robustez (si pasa confirmación): Repetir con freeze policy alternativa. Si funciona en run-d pero no en run-b → es artefacto de
  configuración, no señal real.

  ---
  La historia: por qué Gate 4.2 se necesita AHORA

  Gate 4 (el primer intento) ya intentó esto con descriptors de ratios — y todos fallaron. El diagnóstico post-mortem (DEC-005) descubrió
  que el fallo fue porque el audio encoder estaba congelado. Los ratios generaban gradients que no podían propagarse a través del audio
  encoder frozen → 0% drift en los pesos → cero aprendizaje.

  Bloque A desbloqueó el audio encoder progresivamente (Run A → B → C → D), logrando que S suba de 34.4% a 61.8%. Con un foundation sano
  donde el audio encoder SÍ aprende, Gate 4.2 re-testea los ratios en condiciones correctas.

  Es el retest justo de una pregunta que se contestó mal la primera vez por un defecto de configuración, no por la ciencia.

  ---
  En resumen: qué está pasando ahora mismo

  El screening corre D0 (control), D1 (pitch histogram + aux loss), D4 (local intervals + input augmentation) durante 3 epochs cada uno.
  Los datos preliminares muestran:

  - D0 recupera progresivamente (57.0% → 59.8% → 60.4%)
  - D1 va por debajo de D0 en las 2 epochs disponibles (-0.2pp, -0.8pp)
  - D4 aún no empezó

  Si D1 y D4 no superan a D0 → Escenario D (DROP_RATIO). Significado: el modelo end-to-end ya captura lo que los ratios ofrecen, o los
  ratios simplemente no son la señal cross-modal relevante en este banco de pruebas (Audio↔MIDI piano).

  Cualquiera sea el resultado, es informativo para la tesis de Phideus. Eso es lo que hace a Gate 4.2 el corazón científico del proyecto.





    Comparación Gate 4 vs Gate 4.2 — Descriptores de Ratios
                                                                                                                                           
  1. Diferencias ESTRUCTURALES (las que importan)

  ┌───────────────────────┬───────────────────────────────┬───────────────────────────────────────────────────┐
  │        Aspecto        │            Gate 4             │                     Gate 4.2                      │
  ├───────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Foundation            │ Gate 2 ep45 (S=34.4%)         │ Foundation locked e25 (S=61.8%)                   │
  ├───────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Audio encoder         │ 100% FROZEN (línea 910-912)   │ Layers 2-3 trainable (run-d)                      │
  ├───────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Param groups          │ 3: midi, proj, ratio          │ 6: audio_low, audio_hi, midi, proj, ratio, warmup │
  ├───────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Audio LR              │ 0 (frozen)                    │ 5e-6 (layers 0-1), 1e-5 (layers 2-3)              │
  ├───────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Trainable params      │ ~14.5M (solo MIDI+proj+ratio) │ ~64.9M (todo menos CNN+PosEmb)                    │
  ├───────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Preflight/Sentinel    │ NO                            │ SI (validate_training_setup + DriftSentinel)      │
  ├───────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Checkpoint validation │ strict=False                  │ strict=True en _base.pt                           │
  └───────────────────────┴───────────────────────────────┴───────────────────────────────────────────────────┘

  Esta es LA diferencia que motivó Gate 4.2. DEC-005 diagnosticó que Gate 4 falló porque el audio encoder estaba 100% congelado. Los
  gradients del ratio branch necesitan fluir hasta el audio encoder para modificar las representaciones audio — si el audio encoder no se
  mueve, la información de ratios no tiene dónde actuar.

  2. Descriptores: qué cambió y qué no

  D1 (Gate 4 "baseline" → Gate 4.2 "D1")

  ┌────────────────┬─────────────────────────────────────────┬────────────────────────────────────┬────────────┐
  │    Aspecto     │                 Gate 4                  │            Gate 4.2 D1             │  Cambió?   │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ Función        │ compute_batch_ratio_histograms()        │ La misma función exacta            │ NO         │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ n_bins         │ 256                                     │ 128                                │ SI         │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ Tipo de ratios │ Pairwise f_i/f_j (todos los pares)      │ Pairwise f_i/f_j (todos los pares) │ NO         │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ Rango          │ [0.5, 2.0]                              │ [0.5, 2.0]                         │ NO         │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ max_notes      │ 128                                     │ 128                                │ NO         │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ Binning        │ Soft Gaussiano                          │ Soft Gaussiano                     │ NO         │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ RatioEncoder   │ (256, 1, hidden=128, out=64)            │ (128, 1, hidden=128, out=64)       │ Dims input │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ Projection     │ Linear(64,256) → ReLU → Linear(256,256) │ Linear(64,256) → LayerNorm(256)    │ SI         │
  ├────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────┤
  │ ratio_weight   │ 0.1                                     │ 0.1                                │ NO         │
  └────────────────┴─────────────────────────────────────────┴────────────────────────────────────┴────────────┘

  Resumen D1: Mismo algoritmo, misma función. Cambió n_bins (256→128), y la projection head usa LayerNorm en vez de ReLU+Linear. El
  descriptor en sí es idéntico en concepto.

  D2 (Gate 4 "enriched" → Gate 4.2 "D2")

  ┌────────────────────┬───────────────────────────────────────────┬──────────────────────────────────┬──────────────────┐
  │      Aspecto       │                  Gate 4                   │           Gate 4.2 D2            │     Cambió?      │
  ├────────────────────┼───────────────────────────────────────────┼──────────────────────────────────┼──────────────────┤
  │ Función            │ compute_batch_ratio_histograms_enriched() │ La misma función exacta          │ NO               │
  ├────────────────────┼───────────────────────────────────────────┼──────────────────────────────────┼──────────────────┤
  │ n_bins             │ 256                                       │ 128                              │ SI               │
  ├────────────────────┼───────────────────────────────────────────┼──────────────────────────────────┼──────────────────┤
  │ Canales            │ 3 (vel, dur, unweighted)                  │ 3 (vel, dur, unweighted)         │ NO               │
  ├────────────────────┼───────────────────────────────────────────┼──────────────────────────────────┼──────────────────┤
  │ Output shape       │ [B, 256, 3] → flatten [B, 768]            │ [B, 128, 3] → flatten [B, 384]   │ SI (x dimensión) │
  ├────────────────────┼───────────────────────────────────────────┼──────────────────────────────────┼──────────────────┤
  │ RatioEncoder       │ (256, 3, hidden=128, out=64)              │ (128, 3, hidden=256, out=128)    │ SI               │
  ├────────────────────┼───────────────────────────────────────────┼──────────────────────────────────┼──────────────────┤
  │ Projection         │ Linear(64,256) → ReLU → Linear(256,256)   │ Linear(128,256) → LayerNorm(256) │ SI               │
  ├────────────────────┼───────────────────────────────────────────┼──────────────────────────────────┼──────────────────┤
  │ Duration weighting │ Usa midi_duration (bucket 0-31)           │ Usa midi_duration (bucket 0-31)  │ NO               │
  └────────────────────┴───────────────────────────────────────────┴──────────────────────────────────┴──────────────────┘

  Resumen D2: Mismo algoritmo. n_bins reducido (256→128), encoder más grande (hidden=256, out=128 vs 128/64). La ponderación por duración
  usa buckets discretos (NO los floats midi_duration_sec que D3 introduce).

  D3 — NUEVO en Gate 4.2

  No existe en Gate 4. Completamente nuevo: ratios temporales (IOI, duración) + intervalos de pitch. Usa midi_onset y midi_duration_sec que
   son campos nuevos en el dataset.

  D4 — NUEVO en Gate 4.2

  No existe en Gate 4. Mecanismo de integración completamente diferente (input augmentation vs auxiliary loss).

  3. Diferencias en la LOSS

  ┌──────────────────────────────────────┬────────────────────────────────────────┐
  │                Gate 4                │            Gate 4.2 (D1-D3)            │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ total = main + λ*(ar_loss + mr_loss) │ total = main + λ*(ar_loss + mr_loss)/2 │
  └──────────────────────────────────────┴────────────────────────────────────────┘

  Gate 4.2 divide la aux_loss por 2 (promedio de los dos VICReg auxiliares). Gate 4 las suma sin dividir. Con λ=0.1, esto significa:
  - Gate 4: total = vicreg + 0.1*ar + 0.1*mr → el ratio branch pesa 0.2x del main
  - Gate 4.2: total = vicreg + 0.1*(ar+mr)/2 → el ratio branch pesa 0.1x del main

  Gate 4.2 es más conservador con la aux loss.

  4. Diferencias en la Projection Head

  ┌─────────────────────────────────────────┬─────────────────────────────────┐
  │                 Gate 4                  │            Gate 4.2             │
  ├─────────────────────────────────────────┼─────────────────────────────────┤
  │ Linear(64,256) → ReLU → Linear(256,256) │ Linear(64,256) → LayerNorm(256) │
  ├─────────────────────────────────────────┼─────────────────────────────────┤
  │ 2 capas, activación no-lineal           │ 1 capa + normalización          │
  └─────────────────────────────────────────┴─────────────────────────────────┘

  Gate 4 usa una projection head no-lineal de 2 capas (~131K params). Gate 4.2 usa una capa lineal con LayerNorm (~65K params). La
  simplificación es deliberada — menos params nuevos, menos riesgo de que el ratio branch domine.

  5. Tabla resumen completa

  ┌─────────────┬────────────────┬────────────────┬────────────┬────────────────┬────────────────┬───────────────────┬────────────────┐
  │             │    Gate 4      │    Gate 4      │ Gate 4.2   │  Gate 4.2 D1   │  Gate 4.2 D2   │    Gate 4.2 D3    │  Gate 4.2 D4   │
  │             │    baseline    │    enriched    │     D0     │                │                │                   │                │
  ├─────────────┼────────────────┼────────────────┼────────────┼────────────────┼────────────────┼───────────────────┼────────────────┤
  │ Existe en   │ G4             │ G4             │ G4.2       │ G4.2           │ G4.2           │ G4.2              │ G4.2           │
  ├─────────────┼────────────────┼────────────────┼────────────┼────────────────┼────────────────┼───────────────────┼────────────────┤
  │ Qué mide    │ Pitch pairwise │ Pitch +        │ Control    │ Pitch pairwise │ Pitch +        │ IOI + dur + pitch │ Local          │
  │             │                │ vel/dur        │            │                │ vel/dur        │                   │ intervals      │
  ├─────────────┼────────────────┼────────────────┼────────────┼────────────────┼────────────────┼───────────────────┼────────────────┤
  │ n_bins      │ 256            │ 256            │ -          │ 128            │ 128            │ 153 (64+64+25)    │ 4/nota         │
  ├─────────────┼────────────────┼────────────────┼────────────┼────────────────┼────────────────┼───────────────────┼────────────────┤
  │ Scope       │ Global         │ Global         │ -          │ Global         │ Global         │ Consecutivo       │ Local (O(N))   │
  │             │ (O(N^2))       │ (O(N^2))       │            │ (O(N^2))       │ (O(N^2))       │ (O(N))            │                │
  ├─────────────┼────────────────┼────────────────┼────────────┼────────────────┼────────────────┼───────────────────┼────────────────┤
  │ Mecanismo   │ Aux VICReg     │ Aux VICReg     │ VICReg     │ Aux VICReg     │ Aux VICReg     │ Aux VICReg        │ Input augment  │
  │             │                │                │ puro       │                │                │                   │                │
  ├─────────────┼────────────────┼────────────────┼────────────┼────────────────┼────────────────┼───────────────────┼────────────────┤
  │ λ effective │ 0.2            │ 0.2            │ 0          │ 0.1            │ 0.1            │ 0.1               │ 0              │
  ├─────────────┼────────────────┼────────────────┼────────────┼────────────────┼────────────────┼───────────────────┼────────────────┤
  │ Audio       │ 100%           │ 100%           │ Layers 0-1 │ Layers 0-1     │ Layers 0-1     │ Layers 0-1        │ Layers 0-1     │
  │ frozen      │                │                │            │                │                │                   │                │
  ├─────────────┼────────────────┼────────────────┼────────────┼────────────────┼────────────────┼───────────────────┼────────────────┤
  │ Projection  │ 2-layer MLP    │ 2-layer MLP    │ -          │ Linear+LN      │ Linear+LN      │ Linear+LN         │ Linear+LN      │
  └─────────────┴────────────────┴────────────────┴────────────┴────────────────┴────────────────┴───────────────────┴────────────────┘

  6. Hallazgo clave de esta comparación

  D1 de Gate 4.2 usa el mismo algoritmo exacto de descriptor que Gate 4 baseline. La función compute_batch_ratio_histograms() es la misma.
  Los cambios son:
  - n_bins: 256 → 128 (menor resolución)
  - λ effective: 0.2 → 0.1 (más conservador)
  - Projection: 2-layer MLP → Linear+LN (más simple)
  - Audio encoder: 100% frozen → layers 2-3 trainable (EL cambio principal)

  Si D1 no mejora sobre D0 en Gate 4.2 (que es lo que sugieren los datos preliminares), ya no se puede atribuir al audio encoder frozen.
  Significaría que el descriptor de pitch ratio histogram genuinamente no aporta señal adicional, independientemente de si el audio encoder
   puede aprender o no.

  La reducción de bins (256→128) es menor y no debería causar pérdida de información significativa — la octava [0.5, 2.0] se discretiza en
  intervalos de ~12 cents vs ~6 cents, ambos muy por debajo de la resolución útil para música temperada (100 cents/semitono).