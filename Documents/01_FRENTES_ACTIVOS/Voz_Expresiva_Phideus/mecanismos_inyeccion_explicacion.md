● Arquitectura, capa por capa

  INPUT (precomputado, no entrena):
    features    [B, T, 1024]   ← WavLM-large frozen, last hidden state @ 50 Hz
    descriptor  [B, T, 12]     ← Familia A frame-level, alineada a 50 Hz
    mask        [B, T]         ← bool, True en posiciones válidas, False padding

    donde
      B  = batch size = 64
      T  = T_max del batch (≤ 303), variable
      1024 = dim WavLM-large
      12 = 4 (V4-lin) + 8 (H-series)


  INJECTION MODULE (entrenable, una sola variante por run):

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  config "none" (baseline):                                              │
  │    injection = identity                                                 │
  │    features unchanged                                                   │
  │    params entrenables: 0                                                │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  config "concat":                                                       │
  │                                                                         │
  │    merged = cat([features, descriptor], dim=-1)     [B, T, 1036]        │
  │    out    = Linear(1036, 1024)(merged)              [B, T, 1024]        │
  │                                                                         │
  │    init: W = [I_1024 | 0_{1024×12}],  b = 0                             │
  │      → out_init == features (la rama del descriptor está bloqueada)     │
  │                                                                         │
  │    params entrenables: 1036 × 1024 + 1024 = 1.062 M                     │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  config "film" (Feature-wise Linear Modulation, frame-level):           │
  │                                                                         │
  │    gen = Sequential(                                                    │
  │      Linear(12, 64),                                                    │
  │      ReLU,                                                              │
  │      Linear(64, 2048)            ← gamma + beta concatenados            │
  │    )                                                                    │
  │                                                                         │
  │    film  = gen(descriptor)                          [B, T, 2048]        │
  │    γ, β  = film.chunk(2, dim=-1)                    cada uno [B, T, 1024]│
  │    out   = (1 + γ) * features + β                   [B, T, 1024]        │
  │                                                                         │
  │    init: última Linear con W=0, b=0  →  γ_init=0, β_init=0              │
  │      → out_init == features (modulación nula)                           │
  │                                                                         │
  │    params entrenables: 12×64 + 64 + 64×2048 + 2048 = 0.134 M            │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  config "xattn" (cross-attention con residual escalado):                │
  │                                                                         │
  │    desc_q   = Linear(12, 1024)(descriptor)          [B, T, 1024]        │
  │    attn, _  = MultiheadAttention(                                       │
  │                 query = desc_q,                                         │
  │                 key   = features,                                       │
  │                 value = features,                                       │
  │                 embed_dim=1024, heads=4, dropout=0.1                    │
  │               )                                     [B, T, 1024]        │
  │    out      = features + scale * LayerNorm(attn)    [B, T, 1024]        │
  │                                                                         │
  │    scale    = nn.Parameter(0.01)   ← entrenable, init casi nulo         │
  │                                                                         │
  │    init: scale=0.01  →  contribución del residual ≈ 0 al inicio         │
  │                                                                         │
  │    params entrenables: ~4.2 M                                           │
  │      desc_proj    : 12 × 1024 + 1024            = 13 K                  │
  │      attn (Q,K,V,O): 4 × (1024² + 1024)         = 4.2 M                 │
  │      LayerNorm    : 2 × 1024                    = 2 K                   │
  │      scale        : 1                                                   │
  └─────────────────────────────────────────────────────────────────────────┘

  POOL + HEAD (común a las 4 configs):

    embedding = sum(out * mask) / sum(mask)              [B, 1024]
                ← mean pool sobre T, enmascarando padding

    logits    = Linear(1024, 5)(embedding)               [B, 5]
                ← cabeza compartida, 5125 params entrenables

  LOSS:

    loss = CrossEntropy(logits, labels, weight=class_weight_balanced)

  Cuántos parámetros entrenan en cada config

  config    injection      head        total entrenable
  none      0              5.125 K     5.1 K
  concat    1.062 M        5.125 K     1.067 M
  film      0.134 M        5.125 K     0.139 M
  xattn     4.205 M        5.125 K     4.210 M

  WavLM (316 M) está congelado, no entra al cómputo del gradiente.

  Por qué los tres mecanismos arrancan en el baseline

  La invariancia compartida es: con la inicialización elegida, las cuatro configs producen al inicio el mismo embedding. La diferencia entre arms aparece solo si el optimizador encuentra que abrir la rama
  del descriptor reduce la loss.

  concat → init [I|0]·[f;d] = f                ⇒ out_init = features
  film   → (1+0)·f + 0      = f                ⇒ out_init = features
  xattn  → f + 0.01·LN(attn) ≈ f               ⇒ out_init ≈ features
  none   → f                                   ⇒ out_init = features

  Esta propiedad es la condición para que la comparación entre arms sea limpia: cualquier divergencia del baseline durante el training viene del descriptor entrando en uso, no de un sesgo arquitectural
  inicial.

  Cómo entra el descriptor frame-level en cada uno

  concat:   descriptor se concatena a features y se aprende qué combinar
            → mecanismo aditivo en el espacio de features

  film:     descriptor genera una transformación afín por frame
            que multiplica y desplaza features
            → mecanismo multiplicativo + aditivo, modulación dependiente del frame

  xattn:    descriptor define qué tokens de features mirar
            a través de attention scores
            → mecanismo selectivo, el descriptor pondera contextos

  Los tres operan a la misma resolución (50 Hz, T frames), antes del mean pool. Esto es lo que garantiza que la unidad de comparación entre arms sea homogénea: la única diferencia es cómo el descriptor se
  acopla a las features, no a qué nivel temporal lo hace.

  Forward completo del baseline vs concat

  BASELINE (none):

    features[B,T,1024] ──→ mean_pool(mask) ──→ Linear(1024,5) ──→ logits[B,5]
                              ↓
                         [B,1024]

  CONCAT:

    features[B,T,1024] ──┐
                         ├─ cat ──→ [B,T,1036] ──→ Linear(1036,1024) ──→ [B,T,1024]
    descriptor[B,T,12] ──┘                                                   │
                                                                             ↓
                                                                      mean_pool(mask)
                                                                             │
                                                                             ↓
                                                                     Linear(1024,5)
                                                                             │
                                                                             ↓
                                                                         logits[B,5]

  Optimizer + schedule

  optimizer:   AdamW, lr=1e-3, weight_decay=1e-4
  scheduler:   linear warmup 1 epoch + cosine decay a 0
  loss:        CrossEntropy con class_weight balanced (compensa imbalance entre emociones)
  epochs:      30 máximo
  early stop:  paciencia 5 en val_UAR
  batch:       64 utterances

  Lo que 1_train.py persiste por run

  uar_results.json (concat de todas las runs):
    {fold, test_speaker, val_speaker, norm, config, seed,
     uar, f1_macro, train_loss_final, val_uar_best, early_stop_epoch}

  embeddings/{fold}_{norm}_{config}_{seed}.npz:
    emb_test [N_test, 1024]    ← post-pool pre-head, para CKA
    labels   [N_test]
    row_idx  [N_test]

  predictions/{fold}_{norm}_{config}_{seed}.npz:
    logits   [N_test, 5]
    preds    [N_test]
    labels   [N_test]

  Los embeddings son la pieza que 1_report.py usa para calcular CKA por mecanismo contra el baseline, comparando dos matrices [N_test_utts, 1024] del mismo hablante con misma seed y norm pero distinta
  config.

  Forward de cada mecanismo de inyección

  FiLM

                         ┌─────── descriptor[B,T,12]
                         │              │
                         │              ↓
                         │      Linear(12, 64)
                         │              │
                         │              ↓
                         │           ReLU
                         │              │
                         │              ↓
                         │      Linear(64, 2048)        ← W=0, b=0 init
                         │              │
                         │              ↓
                         │       chunk(2, dim=-1)
                         │           ↙        ↘
                         │      γ[B,T,1024]  β[B,T,1024]
                         │           │           │
    features[B,T,1024] ──┤           ↓           ↓
                         └────→ (1 + γ) * features + β  ──→ [B,T,1024]
                                                                │
                                                                ↓
                                                         mean_pool(mask)
                                                                │
                                                                ↓
                                                        Linear(1024, 5)
                                                                │
                                                                ↓
                                                            logits[B,5]

  init:  W_final = 0, b_final = 0
         ⇒ γ_init = 0, β_init = 0
         ⇒ out_init = (1+0) * features + 0 = features

  núcleo: modulación afín per-frame
          descriptor genera escala (1+γ) y desplazamiento β
          para cada una de las 1024 dimensiones de features, frame por frame

  Cross-attention (xattn)

                                  ┌────── descriptor[B,T,12]
                                  │             │
                                  │             ↓
                                  │     Linear(12, 1024)
                                  │             │
                                  │             ↓
                                  │     desc_q[B,T,1024]
                                  │             │
                                  │             ↓ Q
    features[B,T,1024] ──┬────────┼─→ K   MultiheadAttention(
                         │        │         embed=1024, heads=4,
                         └────────┼─→ V     dropout=0.1
                                  │       )
                                  │             │
                                  │             ↓
                                  │       attn[B,T,1024]
                                  │             │
                                  │             ↓
                                  │       LayerNorm
                                  │             │
                                  │             ↓
                                  │       scale * (·)              ← scale = nn.Parameter(0.01)
                                  │             │
    features[B,T,1024] ───────────┴────→ + ←────┘
                                         │
                                         ↓
                                   [B,T,1024]
                                         │
                                         ↓
                                  mean_pool(mask)
                                         │
                                         ↓
                                 Linear(1024, 5)
                                         │
                                         ↓
                                     logits[B,5]

  init:  scale = 0.01  (parámetro entrenable)
         ⇒ out_init = features + 0.01 * LN(attn_init) ≈ features

  núcleo: attention con descriptor como query
          cada frame del descriptor decide qué frames de features atender
          el resultado se suma como residual con escala aprendible

  Q, K, V dentro de MultiheadAttention:
    Q = W_Q · desc_q       [B,T,1024]
    K = W_K · features     [B,T,1024]
    V = W_V · features     [B,T,1024]
    attn_scores = softmax(Q·Kᵀ / √d_head)   [B, heads, T, T]
    attn_out    = W_O · (attn_scores · V)   [B,T,1024]

    4 heads ⇒ d_head = 256

  Comparación lado a lado de los puntos de acoplamiento

  config    acoplamiento descriptor↔features    operación efectiva en init
  ─────────────────────────────────────────────────────────────────────────
  none      ninguno                              out = features
  concat    cat → linear                         out = I · features + 0 · desc
  film      desc → MLP → (γ,β) → modular         out = (1+0) · features + 0
  xattn     desc → Q, features → K,V             out = features + 0.01 · LN(0...)

  donde out_init ≡ features en los cuatro casos

  Los tres mecanismos son maneras distintas de poner el descriptor en contacto con features manteniendo el mismo punto de salida arquitectural [B,T,1024] antes del pool. Concat lo fusiona linealmente,
  FiLM lo usa para reparametrizar features, xattn lo usa como índice para seleccionar contexto.


  Qué es WavLM-large por dentro

  WavLM-large es un modelo autosupervisado de Microsoft Research del paper WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing (Chen et al., 2021). Comparte familia con
  wav2vec 2.0 y HuBERT: un extractor convolucional que baja la resolución de la forma de onda, seguido de un transformer profundo que contextualiza. Lo distintivo de WavLM frente a sus parientes es el
  gated relative position bias en la atención y la estrategia de utterance mixing durante preentrenamiento, que lo hace más robusto a habla con ruido y multi-hablante.

  Preentrenado sobre noventa y cuatro mil horas de habla sin etiquetas, mezclando LibriLight, GigaSpeech y VoxPopuli. La variante large tiene trescientos dieciséis millones de parámetros.

  Arquitectura interna

  INPUT: waveform [B, samples]   ← raw audio, 16 kHz, mono, sin normalizar
                  │
                  ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │  CNN Feature Extractor (7 capas convolucionales 1D)              │
  │                                                                  │
  │    Conv1d(1→512, k=10, stride=5)  + GroupNorm + GELU             │
  │    Conv1d(512→512, k=3, stride=2) + GELU      ┐                  │
  │    Conv1d(512→512, k=3, stride=2) + GELU      │                  │
  │    Conv1d(512→512, k=3, stride=2) + GELU      │ ratio total      │
  │    Conv1d(512→512, k=3, stride=2) + GELU      │ 5·2·2·2·2·2·2    │
  │    Conv1d(512→512, k=2, stride=2) + GELU      │ = 320            │
  │    Conv1d(512→512, k=2, stride=2) + GELU      ┘                  │
  │                                                                  │
  │  output: [B, T_frames, 512]                                      │
  │           T_frames = samples / 320  ≈ 50 frames por segundo      │
  │                                                                  │
  │  rol: reducción dimensional y descarte de información            │
  │       fina por debajo de 20 ms                                   │
  └──────────────────────────────────────────────────────────────────┘
                  │
                  ↓
          Linear(512 → 1024)        ← proyección de entrada al transformer
          + Positional Conv
                  │
                  ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │  Transformer (24 capas idénticas)                                │
  │                                                                  │
  │    cada capa:                                                    │
  │      MultiHeadAttention(embed=1024, heads=16)                    │
  │        + gated relative position bias (distintivo de WavLM)      │
  │      LayerNorm                                                   │
  │      FeedForward(1024 → 4096 → 1024) + GELU                      │
  │      LayerNorm                                                   │
  │      residuales en torno a cada subbloque                        │
  │                                                                  │
  │  output: 24 hidden_states intermedios                            │
  │          + last_hidden_state = output de la capa 24              │
  │                                                                  │
  │  rol: contextualización temporal larga (cada frame final         │
  │       atiende a todos los frames del audio)                      │
  └──────────────────────────────────────────────────────────────────┘
                  │
                  ↓
          last_hidden_state [B, T_frames, 1024]   ← lo que cosechamos


  cabezas de preentrenamiento (descartadas en uso downstream):
          - cabeza de masked prediction tipo HuBERT
          - cabeza de utterance discrimination

  Cómo lo usamos en este experimento

  La interacción con WavLM ocurre una sola vez en toda Fase 1, durante el precache. Después no se lo vuelve a tocar.

  Llamada en el precache (1_precache_wavlm.py)

  para cada uno de los 17 500 audios:
      waveform = load_wav(path, sr=16000, mono=True)   ← [samples]
      waveform = waveform.unsqueeze(0)                 ← [1, samples] batch=1

      encoder = WavLMEncoder(
          model_name="microsoft/wavlm-large",
          freeze=True,            ← param.requires_grad = False ∀ param
          device="cuda",
      )

      with torch.no_grad():
          hidden = encoder(
              waveform,
              return_sequence=True,   ← devuelve [B, T_frames, 1024]
          )                            ← NO mean pool, no CLS, no nada
                                       ← queremos toda la secuencia temporal

      # hidden corresponde a last_hidden_state (layer = -1 por default)
      # T_frames = samples / 320

      persist hidden[0] en wavlm_features.npy[row_idx, :T_frames, :]
      persist T_frames en wavlm_lengths.npy[row_idx]

  Tres detalles operativos importantes:

  encoder.train()  →  override custom: si freeze=True, siempre eval()
                      el modelo NUNCA sale de eval mode
                      no hay dropout, no hay batch norm en train mode

  freeze=True      →  param.requires_grad = False para todos los 316M params
                      el autograd no construye el grafo para ellos
                      no hay backward al pasar por WavLM

  torch.no_grad()  →  garantía adicional dentro del forward
                      no se almacenan activations intermedias

  Llamada en el training (1_train.py)

  NO HAY LLAMADA A WAVLM.

  el dataset lee directo desde memmap:
      features  = np.memmap("wavlm_features.npy")[row_idx, :T, :]
      descriptor = np.memmap("family_A.npy")[row_idx, :T, :]

  el modelo entrenado es WavLMInjectionClassifier, pero WavLM NO está adentro:

      class WavLMInjectionClassifier(nn.Module):
          def __init__(...):
              self.injection = ConcatInjection|FiLM|XAttn|None
              self.classifier = Linear(1024, 5)
              # NO hay self.wavlm

          def forward(self, features, descriptor, mask):
              # features YA viene como [B, T, 1024] precomputada
              if self.injection is not None:
                  features = self.injection(features, descriptor)
              embedding = masked_mean_pool(features, mask)
              return self.classifier(embedding)

  Diagrama completo end-to-end

  ═══════════════ FASE OFFLINE (una sola vez, 5.7 min en GPU) ═══════════════

                                   ┌──── WavLM-large 316M, FROZEN
                                   │     CNN extractor + 24 transformer layers
  audio.wav  ─→  load_wav 16 kHz  ─┤
                                   │     forward con torch.no_grad()
                                   │     return_sequence=True
                                   └──── last_hidden_state
                                                │
                                                ↓
                                features [T, 1024] @ 50 Hz
                                                │
                                                ↓
                         np.memmap("wavlm_features.npy")[row_idx]


  ═══════════════ FASE OFFLINE (una sola vez, 33 min en 14 CPUs) ═══════════

  audio.wav  ─→  extract Familia A frame-level (V4-lin + H-series)
                  @ 100 Hz → mean pool 2 → @ 50 Hz
                                                │
                                                ↓
                         np.memmap("family_A.npy")[row_idx]


  ═══════════════ FASE ONLINE (240 runs, ~8 h GPU) ════════════════════════

  Dataset.__getitem__(idx):
      features   = memmap_wavlm[row_idx, :T]      ← [T, 1024]  ya precomputado
      descriptor = memmap_descA[row_idx, :T]      ← [T, 12]    ya precomputado
                                                │
                                                ↓  collate_padded → batch
                                                ↓
                            features[B,T,1024]   descriptor[B,T,12]   mask[B,T]
                                                │
                                                ↓
                              ┌─────────────────────────────────────┐
                              │  WavLMInjectionClassifier           │
                              │                                     │
                              │   inyección  ← entrena              │
                              │   ↓                                 │
                              │   mean pool over T (mask aware)     │
                              │   ↓                                 │
                              │   Linear(1024, 5)  ← entrena        │
                              │                                     │
                              │   ← WavLM NO está acá adentro       │
                              └─────────────────────────────────────┘
                                                │
                                                ↓
                                            logits [B, 5]
                                                │
                                                ↓
                                       CrossEntropy(logits, labels)
                                                │
                                                ↓
                              backward pasa SOLO por inyección + head

  Por qué esta arquitectura es legítima

  La estrategia de precachear y entrenar solo encima es estándar para SSL (self-supervised learning) en habla. Se la conoce como linear probing cuando solo se entrena una capa lineal arriba; en este caso
  es probing extendido porque agregamos un módulo de inyección con descriptor externo entre WavLM y la cabeza lineal. Lo que no es: fine-tuning. Si modificáramos los pesos de WavLM tendríamos que entrenar
  end-to-end sobre la señal cruda en cada run, y el costo subiría varios órdenes de magnitud.

  La equivalencia matemática entre "WavLM frozen + precache" y "WavLM frozen + forward por batch" se sostiene exactamente porque:

  WavLM(audio_i)  =  cte ∀ run     (pesos no cambian)
                    ⇒ basta computarlo una vez

  el grafo de cómputo del gradiente arranca recién en la inyección:
      ∂loss/∂θ_injection  =  ∂loss/∂features_out · ∂features_out/∂θ_injection
      ∂loss/∂θ_wavlm      =  no existe (requires_grad=False)

  WavLM cumple el rol de un extractor de features sofisticado y fijo, no el de un modelo entrenable. Equivalente conceptual: como si en computer vision usáramos un ResNet preentrenado en ImageNet
  congelado para producir features, y entrenáramos solo el clasificador downstream sobre esas features. La diferencia es que acá el features extractor tiene 316M parámetros y vio noventa y cuatro mil
  horas de habla sin etiquetas.

  Qué información sobrevive en features[T, 1024]

  información preservada (lo que CNN+transformer codifican):
      - estructura fonética (qué fonema en cada frame)
      - prosodia (entonación, ritmo, énfasis)
      - timbre del hablante (identidad vocal)
      - estado paralingüístico (afecto, esfuerzo)
      - contexto temporal hasta varios segundos

  información perdida o atenuada:
      - amplitud absoluta (WavLM aplica normalización interna)
      - detalles espectrales por debajo de la resolución de 20 ms
      - estructura armónica fina (no es objetivo de SSL)
                ↑
                └── acá es donde Familia A puede aportar:
                    WavLM no fue entrenado para preservar ratios armónicos
                    explícitos; el descriptor Phideus los reinyecta como
                    canal lateral.

  La hipótesis Phideus de Fase 1 es justamente que ese aporte armónico tiene algo que WavLM no resolvió por sí solo. Si la inyección no mueve UAR ni CKA respecto al baseline, la lectura es que WavLM ya
  captura suficiente de lo armónico para esta tarea. Si la mueve, hay evidencia de que el descriptor aporta información no redundante.