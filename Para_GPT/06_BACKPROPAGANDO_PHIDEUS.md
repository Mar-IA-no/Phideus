# BACKPROPAGANDO_PHIDEUS

Fecha: 2026-02-09
Estado: borrador corto de redefinicion conceptual

## Objetivo de este documento (importante)

Este archivo es un espacio de trabajo para anotar ideas, redefiniciones e hipotesis que surgen de la experimentacion, pero que aun no estan implementadas ni cerradas.

Regla de uso:
- Editar este documento solo para agregar o refinar ideas en discusion.
- No usar este documento para registrar estado operativo oficial, roadmap formal o decisiones cerradas.

## Contexto

Phideus apunta a un lenguaje modalmente agnostico, centrado en la firma de ratios de los datos.
En BIAS_CONTROL, la evidencia reciente sugiere:

- El camino "quitar modalidad" de forma adversarial (DANN) no dio mejoras robustas.
- El camino "agregar estructura" (ratios como vista auxiliar) parece mas prometedor.
- La decision real debe hacerse con evaluacion estructurada (hard negatives), no solo con metricas globales de entrenamiento.

## Redefinicion operativa del objetivo

En esta etapa, el objetivo no se formula como "agnosticismo total universal", sino como:

1. Invariancia fuerte a modalidad y transformaciones musicales plausibles.
2. Recuperacion temporal robusta frente a hard negatives (misma pieza, distinto tiempo).
3. Espacio compartido de relaciones (ratios) utilizable entre modalidades.

Si esas tres condiciones se cumplen de forma consistente, Phideus ya funciona, en la practica, como lenguaje modalmente agnostico.

## Justificacion

- El agnosticismo absoluto es una meta filosofica; para investigacion aplicada conviene una definicion falsable y medible.
- El enfoque hibrido (embedding aprendido + estructura de ratios) combina performance con interpretabilidad.
- Este marco permite experimentar "arriesgado" sin perder rigor: cada avance se valida contra criterios concretos.

## Criterios minimos de validacion

- Mejora o no degradacion en structured pool frente al baseline vigente.
- Mejora en discriminacion temporal (same-time vs diff-time, misma pieza).
- Evidencia en analisis representacional (Gate 6) de que los ratios realmente estan codificados en el embedding.

## Nota de direccion

Siguiente foco recomendado:

1. Cerrar Gate 4 con comparacion causal (ratio vs control).
2. Ejecutar Gate 6 para explicar que aprende realmente la representacion.
3. Recién ahi consolidar auditoria final global.








Quiero que me expliques muy bien ahora qué informacion de ratios toma D4, como la organiza, cómo la        
  inyecta al encoder de midi... y quiero me respondas lo siguiente: ¿por que no hacemos lo mismo que en D4 pero tanto en midi como en     
  audio?                                                                                                                                   
                                                                                                                                           
● D4: Cómo funciona en detalle                                                                                                             
                                      
  Qué información de ratios toma                                                                                                           
   
  D4 usa compute_local_interval_features(midi_pitch, midi_mask) que calcula 4 features por cada nota MIDI:                                 
                                      
  Para cada nota i en la secuencia:

  1. semitone_prev = (pitch[i] - pitch[i-1]) / 24     ← intervalo hacia atrás
  2. semitone_next = (pitch[i+1] - pitch[i]) / 24     ← intervalo hacia adelante
  3. log_ratio_prev = clamp((pitch[i] - pitch[i-1]) / 12, [-2,2]) / 2
  4. log_ratio_next = clamp((pitch[i+1] - pitch[i]) / 12, [-2,2]) / 2

  Son esencialmente la misma información (intervalo melódico local) expresada en dos escalas:
  - Semitonos/24: escala fina, rango amplio (~±2 octavas = ±1.0)
  - Semitonos/12 clamped: escala de octava, saturada a ±2 octavas

  Notas en padding o en bordes de secuencia → cero. El resultado es [B, N, 4].

  Cómo la organiza

  Es un tensor denso, nota por nota, alineado temporalmente con los event embeddings del MIDI. No hay histograma, no hay binning, no hay
  resumen global. Cada nota "sabe" qué intervalo tiene con su vecina anterior y siguiente. Es información local y posicional.

  Cómo la inyecta

  MIDI pitch/vel/dur → Event Embedding → [B, N, 512]
                                              ↓
  MIDI pitch/mask → Local Intervals → [B, N, 4]  ←── las 4 features
                                              ↓
                                concat → [B, N, 516]
                                              ↓
                       interval_projection: Linear(516→512) + LayerNorm
                                              ↓
                                         [B, N, 512]
                                              ↓
                                Positional Encoding
                                              ↓
                             Transformer (4 capas, d=512)
                                              ↓
                                Pooling → [B, 512]
                                              ↓
                          MIDI Projection → [B, 256]

  La clave: la inyección ocurre antes del Transformer. Cada token que entra al Transformer ya "sabe" su contexto melódico local. El
  Transformer entonces puede usar esa información para construir representaciones que capturen patrones de intervalos a escala global.

  Es elegante porque no agrega una loss auxiliar (como D1-D3) — simplemente enriquece la entrada. El VICReg normal hace el resto.

  ---
  ¿Por qué no hacemos lo mismo en audio?

  Pregunta excelente. La respuesta corta: porque en audio no tenemos notas discretas.

  En MIDI, pitch[i] es un número exacto (60 = C4, 64 = E4). El intervalo entre nota i y nota i+1 es una resta trivial. En audio, lo que
  entra al encoder es una forma de onda cruda — no hay "nota i" ni "nota i+1".

  Pero — esto no significa que sea imposible. Significa que hay que resolver un paso intermedio: extraer algo análogo a "notas" o
  "frecuencias locales" del audio. Las opciones:

  Opción A: F0 tracking → intervalos

  Usar un estimador de pitch (CREPE, pYIN) para extraer la frecuencia fundamental frame a frame, y después calcular intervalos entre frames
   consecutivos. Problema: MAESTRO es piano polifónico, F0 tracking en polifonía es ruidoso y ambiguo.

  Opción B: CQT/Chroma → intervalos entre frames

  Calcular un CQT (que ya está en escala log-frecuencia = escala de ratios) y computar diferencias entre frames consecutivos. Esto daría
  "intervalos espectrales locales" — análogo borroso de los intervalos MIDI.

  Opción C: Inyectar en los features del CNN

  Después de las 4 capas CNN de MERTEncoderLite, tenemos [B, 1024, T']. Podríamos computar diferencias entre frames consecutivos de esos
  features y concatenarlos — sería "intervalos en el espacio de features aprendido".

  Lo que me parece más interesante

  La opción B es la más coherente con la filosofía Phideus. Un CQT con resolución de semitono te da 12 bins por octava — esencialmente un
  "piano roll borroso" del audio. De ahí podés calcular:

  Para cada frame temporal t del CQT:
  1. chroma_change_prev = chroma[t] - chroma[t-1]     ← cambio cromático hacia atrás
  2. chroma_change_next = chroma[t+1] - chroma[t]     ← cambio cromático hacia adelante

  Y esto se inyectaría en el audio encoder de la misma manera:

  Audio waveform → CNN features → [B, T', 1024]
                                        ↓
  Audio waveform → CQT → interval features → [B, T', K]  ← NUEVO
                                        ↓
                          concat → [B, T', 1024+K]
                                        ↓
                    audio_interval_projection: Linear(1024+K → 1024)
                                        ↓
                          Transformer (4 capas)
                                        ↓
                           ...resto igual...

  Por qué tiene sentido científicamente

  Si la hipótesis Phideus es correcta (los ratios son un lenguaje informacional universal), entonces dar a ambos encoders un vocabulario
  explícito de intervalos/ratios debería facilitar el alineamiento cross-modal. Estás diciendo: "mirá, encoder de audio, prestá atención a
  los cambios de frecuencia entre frames; encoder de MIDI, prestá atención a los cambios de pitch entre notas". Ambos "hablan intervalos".

  Por qué no se hizo así originalmente

  Creo que la razón es práctica: en MIDI la información es exacta y gratis (una resta), en audio requiere un paso de extracción (CQT, F0,
  etc.) que agrega complejidad y ruido. D4 se diseñó como el camino más simple para testear la idea de input augmentation.

  ¿Querés que lo exploremos?

  Sería un D4-dual o D11: inyección simétrica de intervalos locales en ambos encoders. El CQT se puede computar con torchaudio sin
  dependencias nuevas. Es una hipótesis que vale la pena testear — y encaja perfectamente con la directiva fundamental del proyecto.

✻ Brewed for 1m 8s              

❯ en estos descriptores que estamos manejando ahora, ¿seguimos hablando de ratios en términos de phi, quintas, terceras, etc etc?          
  ¿seguimos respetando una idea central de phideus que es abandonar musica temperada?                                                  
                                                                                                                                           
● No. Y es una observación crucial.                                                                                                        
                                                                                                                                           
  Miremos lo que D4 realmente computa:                                                                                                     
   
  semitone_prev = (pitch[i] - pitch[i-1]) / 24                                                                                             
  semitone_next = (pitch[i+1] - pitch[i]) / 24
  log_ratio_prev = (pitch[i] - pitch[i-1]) / 12   # clamped, normalizado
  log_ratio_next = (pitch[i+1] - pitch[i]) / 12   # clamped, normalizado

  Eso son intervalos en semitonos. Resta de números MIDI dividida por constantes. Es aritmética temperada pura — 12-TET. No hay ratios de
  frecuencia, no hay phi, no hay quintas justas, no hay nada continuo.

  Y no es solo D4. Repasemos todos:

  ┌────────────┬────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │ Descriptor │            ¿Qué computa realmente?             │                              ¿Son ratios?                              │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D1         │ Histograma de pitch_j - pitch_i (semitonos)    │ No. Intervalos temperados                                              │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D2         │ Igual + velocity + duration                    │ No                                                                     │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D3         │ IOI ratios + duration ratios + pitch intervals │ IOI/dur sí son ratios temporales. Pitch no                             │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D4         │ Diferencias de semitonos consecutivos          │ No                                                                     │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D5         │ (delta_t, delta_pitch) en semitonos            │ delta_t sí. Pitch no                                                   │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D6         │ log2(freq_j/freq_i) mod 1.0                    │ Más cerca, pero freq = 440·2^((p-69)/12), así que es 12-TET disfrazado │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D7         │ Hashes sobre pitch_class + intervalos          │ No. Pitch class = temperamento                                         │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D8         │ n-gramas de pitch ratios en Hz                 │ Misma trampa que D6                                                    │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D9         │ Grafo con peso 1/(1+                           │ pitch_i - pitch_j                                                      │
  ├────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ D10        │ Histogramas de ratios por ventana temporal     │ Misma trampa que D6/D8                                                 │
  └────────────┴────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

  La trampa es siempre la misma: MIDI pitch ES 12-TET. Cualquier operación sobre números MIDI, incluyendo convertirlos a Hz via 440 *
  2^((p-69)/12), produce ratios temperados. Un "ratio" entre MIDI 60 y 67 siempre da 2^(7/12) = 1.4983..., nunca 3/2 = 1.5 (la quinta
  justa).

  Nos alejamos de Phideus sin darnos cuenta. Estamos testeando "¿los intervalos temperados ayudan al alineamiento audio↔MIDI?", no "¿los
  ratios de frecuencia son un lenguaje informacional universal?"

  ---
  Dónde sí hay ratios reales

  En el audio. Cuando un piano toca la nota MIDI 60 (C4), la forma de onda contiene:

  Fundamental:    261.6 Hz
  2do armónico:   523.2 Hz   → ratio 2:1 (octava pura)
  3er armónico:   784.8 Hz   → ratio 3:2 (quinta justa) desde el 2do
  5to armónico:  1308.0 Hz   → ratio 5:4 (tercera mayor justa) desde el 4to
  7mo armónico:  1831.2 Hz   → ratio 7:4 (séptima armónica)

  La serie de armónicos produce ratios puros, no temperados. Y el encoder de audio (MERTEncoderLite) ve esta señal cruda. Los CNNs y
  Transformers procesan frecuencias reales, no números MIDI. El audio contiene la información que Phideus busca — y nosotros no la estamos
  extrayendo explícitamente.

  La paradoja actual

  MIDI:  Tiene notas discretas → fácil computar intervalos → pero son temperados
  Audio: Tiene ratios puros en los armónicos → difícil extraer → pero son los reales

  Estamos inyectando información del lado fácil-pero-incorrecto (MIDI temperado) y dejando sin explotar el lado difícil-pero-correcto
  (audio con armónicos puros).

  ---
  Qué significaría volver a los ratios reales

  Para que los descriptores sean fieles a Phideus, necesitaríamos trabajar con ratios de frecuencia continuos, no cuantizados a 12-TET.
  Algunas ideas:

  1. Del lado audio: Extraer picos espectrales reales (via CQT, FFT, o peak picking) y computar ratios entre armónicos. Esos ratios serían
  2.0, 1.5, 1.333..., 1.25, etc. — la serie armónica real.
  2. Del lado MIDI: Admitir que MIDI es temperado y tratarlo como una aproximación discreta del espacio continuo de ratios. En vez de
  trabajar con semitonos (enteros), trabajar con los ratios de frecuencia que esos semitonos aproximan: 7 semitonos ≈ 3/2, 4 semitonos ≈
  5/4, etc.
  3. Cross-modal: La pregunta clave de Phideus sería: ¿el encoder de audio aprende representaciones que capturan los ratios puros de los
  armónicos, y el encoder de MIDI aprende representaciones que capturan los ratios temperados que los aproximan, y VICReg puede alinear
  ambos?

  ¿Esto resuena con la visión original? ¿Querés que replanteemos los descriptores desde esta perspectiva antes de seguir con Gate 4.3?


  Entonces, mientras trabajemos con inyectar informacion de ratios desde el lado del     
   midi, es imposible salirnos de la musica temperada, cierto? Esto sería así si tuvieramos midis de audios que nos sean piezas           
  musicales?                                                                                                                               
                                                                                                                                           
● Correcto. MIDI pitch es 12-TET por definición. Pitch 60 = C4 = 261.63 Hz. Pitch 67 = G4 = 392.00 Hz. El ratio es siempre 2^(7/12) =      
  1.4983..., nunca 3/2 = 1.5. No importa qué computes a partir de esos números — restas, divisiones, histogramas, log2 — siempre estás en
  la grilla temperada.                                                                                                                     
   
  Y esto no cambia aunque el MIDI no sea de música. Si alguien graba ruido de calle y lo transcribe a MIDI, el transcriptor cuantiza cada  
  frecuencia al semitono más cercano. MIDI es una grilla de 128 alturas fijas en 12-TET. Es su naturaleza.

  Dicho de otra forma: MIDI no puede representar ratios puros. Una quinta justa (3:2 = 1.5) cae "entre" el semitono 6.98 y 7.02 — MIDI lo
  redondea a 7. Una tercera mayor justa (5:4 = 1.25) cae en el semitono 3.86 — MIDI lo redondea a 4. El ratio phi (1.618...) cae en 8.35
  semitonos — MIDI lo redondea a 8. La información que Phideus quiere estudiar se pierde en la cuantización.

  Para los escalones 2 y 3 esto no aplica. En Speech↔EGG (Escalón 2), la F0 de la voz es continua — no está cuantizada a semitonos. En
  ECG↔PPG (Escalón 3), las frecuencias cardíacas son continuas. En esos dominios sí podríamos trabajar con ratios reales.

  Pero en Escalón 1, la única fuente de ratios no-temperados es el audio. La forma de onda del piano contiene la serie armónica con ratios
  puros — eso no pasa por la grilla MIDI.

✻ Brewed for 32s                  

❯ Entonces tenemos dos lineas de investigacion dentro del escalon 1... por una lado todo lo que hagamos del lado midi será dentro de       
  la musica temperada... pero según los resultados de D4, A2M mejora igual que M2A, cierto? es decir que la información de ratios de      
  midi que inyectamos ayuda en ambas direcciones?                                                                                          
                                                                                                                                           
● Miremos los números con precisión. D4 vs D0 en epoch 3:                                                                                  
                                      
           A2M     M2A     diff A2M   diff M2A                                                                                             
  D0 e3:  62.0%   60.4%      —          —
  D4 e3:  64.6%   64.2%   +2.6pp     +3.8pp                                                                                                
                                      
  Sí, ambas direcciones mejoran. Pero mirá algo interesante: M2A mejora más que A2M (+3.8pp vs +2.6pp).

  Pensemos qué significa cada dirección:

  - A2M (Audio→MIDI): dado un audio, encontrar su MIDI. El query es audio, busca entre embeddings MIDI.
  - M2A (MIDI→Audio): dado un MIDI, encontrar su audio. El query es MIDI, busca entre embeddings de audio.

  D4 inyecta intervalos solo en el encoder MIDI. El encoder de audio no se toca. Entonces:

  - M2A mejora (+3.8pp): lógico — el MIDI encoder produce embeddings más ricos, mejores queries MIDI → encuentra mejor su audio.
  - A2M mejora (+2.6pp): menos obvio — el audio encoder no recibió información nueva. ¿Por qué mejora?

  La respuesta es VICReg. El entrenamiento contrastivo empuja a ambos encoders a alinearse. Cuando el MIDI encoder produce representaciones
   más informativas (porque tiene intervalos), VICReg fuerza al audio encoder a reorganizar su espacio de embeddings para matchear esas
  representaciones más ricas. El audio encoder aprende indirectamente a capturar estructura de intervalos porque su target (el embedding
  MIDI) ahora la contiene.

  Es como si le dijeras al encoder de audio: "tu embedding tiene que parecerse a ESTE embedding MIDI que ahora sabe de intervalos" — y el
  encoder de audio se adapta.

  Esto tiene una implicación importante: la información temperada del MIDI sí es útil, y se propaga al lado del audio via la loss
  contrastiva. No son ratios puros, pero la estructura de intervalos temperados es una aproximación suficientemente buena como para mejorar
   el alineamiento en ambas direcciones.

  Y abre la pregunta que naturalmente sigue: si inyectar intervalos temperados en un solo lado produce +3.8pp, ¿qué pasaría si inyectamos
  ratios reales (de la serie armónica) en el lado del audio también?
