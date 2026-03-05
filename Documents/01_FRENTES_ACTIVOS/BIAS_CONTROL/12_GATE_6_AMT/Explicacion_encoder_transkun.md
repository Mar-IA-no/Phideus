● Ahora tengo el código real. La arquitectura es más interesante de lo que esperaba.                                                                        
                                                                                                                                                            
  ---                                                                                                                                                       
  El input: no es un espectrograma normal                                                                                                                   
                                                                                                                                                            
  nExtraWins = 5  →  6 canales temporales del mel spectrogram                                                                                               
  n_mels = 229, f_min=30, f_max=8000                                                                                                                      
  hop=1024 @ 44100Hz → ~23ms/frame

  Input shape: [N, T, 229, 6] — el eje "canal" son 6 ventanas temporales del mismo frame (contexto multi-escala). Nuestro encoder recibe waveform raw.

  ---
  El truco central: las 90 pitch queries

  self.targetMIDIPitch = [-64, -67] + list(range(21, 108+1))
  # = 2 pedales + 88 notas = 90 entradas

  En el Backbone.forward():

  # hTarget = embeddings posicionales indexados por MIDI pitch number
  hTarget = posEmbedBuilderAttnTE(coord_T, outputIndices)  # [N, T/8, 90, D]

  # Se concatenan con las features del espectrograma EN la dimensión frecuencia
  hAll = torch.cat([h, hTarget], dim=-2)  # [N, T/8, F'+90, D]

  # Todo pasa por los 6 bloques axiales juntos
  for l in encoderLayers:
      hAll = l(hAll)

  # Se separan de vuelta
  h, hTarget = hAll.split([h_F, 90], dim=-2)

  Las 90 pitch queries no son embeddings discretos de eventos — son embeddings posicionales aprendibles, uno por nota MIDI, que se concatenan con el
  espectrograma en la dimensión de frecuencia y pasan por los transformers junto con él.

  ---
  Los bloques axiales (BasicBlock)

  Config activa: enabledAttn = ["F", "T"]

  Cada bloque hace dos pasadas separadas:

  Input: [N, T, F+90, D]   ← frecuencias + 90 pitch queries mezcladas

  1) F-attention:
     Aplana → [N*T, F+90, D]   (cada frame como secuencia)
     MHA sobre la dimensión F+90
     → cada frame: las pitch queries leen qué frecuencias están activas
     → cada nota MIDI atiende a las frecuencias que le corresponden

     Transpose → [N, F+90, T, D]

  2) T-attention:
     Aplana → [N*(F+90), T, D]  (cada "canal" como secuencia temporal)
     MHA sobre la dimensión T
     → cada pitch query ve su historia en el tiempo
     → mantiene coherencia temporal de cada nota

     Transpose → [N, T, F+90, D]  (vuelve al shape original)

  Después de 6 bloques, las pitch queries acumularon información de: qué frecuencias están activas en cada frame (F-att) y cómo evoluciona cada pitch en el
  tiempo (T-att).

  ---
  El upsampling y scoring

  El backbone comprimió 8× en tiempo. Después:

  # Upsample back: ConvTranspose1d con stride=8
  hTarget = upConv1dSkip(hTarget)  # [N*90, D', T_original]
  # → ctx shape: [N, 90, T, 160]

  El scorer ScaledInnerProductIntervalScorer calcula para cada nota el score de cada intervalo [t_onset, t_offset]:

  S[t_onset, t_offset, pitch] = dot(q[t_onset], k[t_offset]) * (t_offset - t_onset)

  Favorece intervalos más largos (nota sustanida) y maximiza la coherencia inicio-final de la nota.

  El Semi-CRF recorre esta matriz de scores para encontrar el conjunto óptimo de notas no superpuestas por pitch.

  ---
  Tabla comparativa final

  ┌─────────────────────────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────┐
  │                         │         Nuestro encoder          │                          Transkun                           │
  ├─────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Input                   │ Waveform raw 1D                  │ Mel multi-contexto [T, F, 6]                                │
  ├─────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Estructura              │ CNN → seq 1D → transformer       │ Conv2D → grid 2D [T, F] → axial att                         │
  ├─────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Representación de notas │ No existe — pooling a 256d       │ 90 pitch queries × T frames = representación explícita      │
  ├─────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Cómo "sabe" de notas    │ No sabe — nadie le enseñó        │ F-attention: cada query aprende qué frecuencias son su nota │
  ├─────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Qué optimiza cada frame │ "sé parte del segmento correcto" │ "predice si esta nota está activa aquí"                     │
  ├─────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Output                  │ 256d embedding de pieza          │ 90 × T × 160d → Semi-CRF → nota events                      │
  ├─────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Velocidad               │ 128-dim prediction separada      │ Predictor sobre features de onset+offset                    │
  └─────────────────────────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────┘

  ---
  Por qué es relevante para Exp C

  El decoder de Exp C hace cross-attention hacia nuestras features pre-pooling [188, 1024]. Esas features son el análogo de hAll antes de las pitch queries
  — pero nunca tuvieron pitch queries mezcladas en ellas, nunca pasaron por F-attention que localice frecuencias por nota, y nunca fueron entrenadas con
  supervision de piano roll.

  El decoder tiene que aprender desde cero la función de las 90 pitch queries de Transkun, partiendo de features que nunca fueron organizadas para eso. El
  F1=0.157 es el límite de lo que puede recuperar de esa señal.