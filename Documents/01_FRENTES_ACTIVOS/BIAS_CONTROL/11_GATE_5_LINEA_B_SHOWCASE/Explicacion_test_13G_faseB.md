● Test 13G Phase B — Post-Hoc Pre-Pooling Decoder                                                                                                           
                                                            
  Pregunta científica                                                                                                                                       
                                                            
  "Dadas las representaciones pre-pooling del encoder de audio, ¿qué tan decodificable es el piano roll?"                                                   
                                                            
  Phase A mostró que decodificar desde z[256] (post-pooling) da F1 idéntico y pobre (~0.114) sin importar λ. El mean-pooling destruye la info temporal
  (compresión 750:1 para D0). Phase B esquiva ese cuello de botella enganchándose antes del pooling.

  Arquitectura: PostHocPRDecoder (2.44M params)

  encoder_feats [B, N, 1024]     N=2400 (D0, d4a4) ó 188 (a4r)
    │
    ├── k_proj: Linear(1024, 256) ─→ K [B, N, 256]
    ├── v_proj: Linear(1024, 256) ─→ V [B, N, 256]
    │
    └── frame_queries [188, 256] (aprendidos) + PE sinusoidal
         │
         ├── 1× CrossAttention(Q=queries, K, V) + residual + LN  → [B, 188, 256]
         ├── 2× SelfAttention (norm_first, GELU, d_ff=1024)      → [B, 188, 256]
         └── Linear(256, 88)                                      → logits [B, 188, 88]

  El encoder está completamente congelado. Solo el decoder recibe gradientes. Los features se capturan con un forward hook en audio_encoder.transformer —
  sin modificar el encoder ni su forward pass.

  Cómo funciona

  Extracción de features (hook): Un context manager (EncoderFeatureExtractor) registra un hook en el módulo transformer del audio encoder. Cada forward pass
   captura la salida pre-pooling:
  - D0/d4a4: [B, 2400, 1024] — 2400 tokens temporales de la CNN+Transformer
  - a4r: [B, 188, 1024] — 188 tokens post reverse cross-attention

  Los features se detachan pero quedan en GPU (sin .cpu()), para que el decoder los procese sin copia.

  Targets precomputados: build_pr_targets() (de Phase A) convierte MIDI → piano roll [188, 88]. Como es Python puro (lento), se precomputa una vez y se
  guarda en NPZ:
  - Train: 8000 segmentos subsampled → `pr_targets_train_8k.npz` (cache NPZ comprimido liviano)
  - Val: validación completa (~12.8k segmentos) → `pr_targets_val.npz` (cache NPZ comprimido liviano)

  Alineación shuffle-safe: PRProbeDataset envuelve el dataset base y adjunta el target precomputado a cada item. Con shuffle=True, cada item lleva su propio
   target — no hay confusión de índices. Verificado: targets precomputados coinciden con build_pr_targets() on-the-fly.

  Training config

  ┌─────────────────┬───────────────────────────┬──────────────────────┐
  │    Parámetro    │           Valor           │        Origen        │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ lr              │ 1e-4                      │ —                    │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ batch_size      │ 16                        │ —                    │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ epochs          │ 40 max                    │ —                    │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ eval_every      │ 5 epochs                  │ —                    │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ patience        │ 4 eval rounds (no epochs) │ earliest stop: ep 25 │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ pos_weight      │ 50.0                      │ Test 11              │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ threshold       │ 0.1                       │ Test 11              │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ onset_tolerance │ ±2 frames                 │ Test 11              │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ grad_clip       │ 1.0                       │ —                    │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ train segments  │ 8000 (subsample)          │ —                    │
  ├─────────────────┼───────────────────────────┼──────────────────────┤
  │ val quick       │ 2000 (de ~12.8k)          │ —                    │
  └─────────────────┴───────────────────────────┴──────────────────────┘

  Patience semántica: Se evalúa cada 5 epochs. Si frame_f1 no mejora por 4 evaluaciones consecutivas → early stop. Eso da un mínimo de 25 epochs antes de
  parar.

  4 métricas

  1. frame_f1 — threshold 0.1, TP/FP/FN global (métrica primaria para best checkpoint)
  2. onset_f1 — ±2 frames, pitch-specific, greedy matching 1-a-1 (reusa _compute_onset_f1 de Test 11)
  3. bce — BCEWithLogitsLoss con pos_weight=50 (computado en chunks para no reventar CPU)
  4. cosine — similitud coseno entre sigmoid(pred) y target

  Los 3 brazos (+1 control)

  ┌────────────────┬──────────────────────────────────┬──────────────┬──────────────────────────────────┐
  │     Brazo      │            Checkpoint            │   N tokens   │             Qué mide             │
  ├────────────────┼──────────────────────────────────┼──────────────┼──────────────────────────────────┤
  │ D0             │ models/gate5b/D0/best_model.pt   │ 2400         │ Baseline sin descriptores        │
  ├────────────────┼──────────────────────────────────┼──────────────┼──────────────────────────────────┤
  │ d4a4           │ models/gate5b/d4a4/best_model.pt │ 2400         │ Concat descriptores              │
  ├────────────────┼──────────────────────────────────┼──────────────┼──────────────────────────────────┤
  │ a4r            │ models/gate5b/a4r/best_model.pt  │ 188          │ Reverse cross-attention          │
  ├────────────────┼──────────────────────────────────┼──────────────┼──────────────────────────────────┤
  │ D0 pool-to-188 │ mismo D0                         │ 188 (pooled) │ Control de longitud de secuencia │
  └────────────────┴──────────────────────────────────┴──────────────┴──────────────────────────────────┘

  El control D0-pool-188 es clave: si a4r gana sobre D0 pero D0-pooled también mejora, la ganancia sería por compresión de secuencia, no por el mecanismo de
   ratios.

  Constraint interpretativa (del plan)

  Si a4r gana, el claim es "la representación pre-pooling de a4r es más decodificable musicalmente." NO atribuible a "ratios" solamente — a4r cambia
  mecanismo (reverse cross-att) Y régimen de compresión (2400→188).

  Pipeline de ejecución

  # Paso 0: Precomputar targets (una vez, ~15 min)
  python experiments/bias_control/gate5b/test13g_posthoc_decoder.py \
      --phase precompute

  # Paso 1: Entrenar cada brazo (~2h por brazo GPU)
  python .../test13g_posthoc_decoder.py --descriptor d0   --phase train
  python .../test13g_posthoc_decoder.py --descriptor a4r   --phase train
  python .../test13g_posthoc_decoder.py --descriptor d4a4  --phase train

  # Opcional: control de longitud
  python .../test13g_posthoc_decoder.py --descriptor d0 --pool-to 188 --phase train

  # Generación cualitativa (MIDI/WAV/PNG)
  python .../test13g_posthoc_decoder.py --descriptor a4r --phase generate

  Output

  data/gate5b_results/
  ├── pr_targets_train_8k.npz              # compartido (NPZ comprimido)
  ├── pr_targets_val.npz                   # compartido (NPZ comprimido)
  ├── pr_train_8k_indices.npy              # reproducibilidad
  ├── {D0,d4a4,a4r}/test13g_posthoc/
  │   ├── config.json
  │   ├── training_history.json
  │   ├── test13g_posthoc_results.json     # incluye full-val final
  │   ├── checkpoints/
  │   │   ├── checkpoint_epoch{1..40}.pt   # cada epoch
  │   │   └── best_f1.pt                  # mejor frame_f1
  │   ├── eval_per_epoch/
  │   │   └── eval_epoch{5,10,...}.json
  │   └── generation_samples/              # MIDI, WAV, PNG comparativos

  Al terminar el training, automáticamente corre full-val (12887 segmentos) con el mejor checkpoint y guarda los resultados finales.

  Tiempo estimado

  ┌──────────────────────────────────┬───────────┬──────────┐
  │               Fase               │ Por brazo │ 3 brazos │
  ├──────────────────────────────────┼───────────┼──────────┤
  │ Precompute (una vez)             │ —         │ ~15 min  │
  ├──────────────────────────────────┼───────────┼──────────┤
  │ Training (40ep × 500bat)         │ ~70 min   │ ~3.5h    │
  ├──────────────────────────────────┼───────────┼──────────┤
  │ Quick eval (8× durante training) │ ~16 min   │ ~48 min  │
  ├──────────────────────────────────┼───────────┼──────────┤
  │ Full val final                   │ ~20 min   │ ~1h      │
  ├──────────────────────────────────┼───────────┼──────────┤
  │ Total por brazo                  │ ~2h       │ —        │
  ├──────────────────────────────────┼───────────┼──────────┤
  │ 3 brazos + D0-pool + overhead    │ —         │ ~10-14h  │
  └──────────────────────────────────┴───────────┴──────────┘
