╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan de Implementación: Gate 4.2 — Exploración Ratio-Céntrica (v2.1, post-audit Codex ×2)                                               

 Contexto

 Por qué este plan: Phideus = exploración de ratios como lenguaje informacional. Cross-modality es el banco de pruebas, NO el objetivo.
 Gate 4.2 es el corazón científico del proyecto: medir si descriptores de ratios aportan señal causal sobre un foundation sano.

 Historia relevante:
 - Gate 4/4.1 intentaron ratio descriptors como auxiliary VICReg loss → TODOS fallaron
 - DEC-005 diagnosticó: el fallo fue por audio encoder congelado (0% drift), NO por descriptores malos
 - Bloque A Run B desbloqueó el audio encoder (layers 2-3) → S pasó de 34.4% a 43.2% en 3 epochs
 - Ahora tenemos un foundation sano para re-testear los ratios con rigor

 Pregunta central: ¿Los ratio descriptors aportan información que el modelo end-to-end no captura por sí solo?

 Auditorías Codex: v1→v2 (4 bloqueantes), v2→v2.1 (5 bloqueantes más) — ver sección "Fixes aplicados" al final.

 Addendum de secuenciacion (2026-02-12)
 - Gate 4.2 mantiene dos carriles:
   1) codigo (dataset + descriptors + training script) en paralelo,
   2) screening cientifico solo despues de foundation lock definitivo.
 - Run D (full-unfreeze) fue condicional en DEC-007 y ahora esta en curso; no bloquea implementacion de codigo.
 - Gate2R-lite se agenda como higiene metodologica post Gate 4.2; no bloquea la pregunta causal D0 vs Dx.

 ---
 Pre-requisito: Fase 0 — Foundation Lock

 Antes de Gate 4.2, cerrar Bloque A:
 1. Run C ya esta cerrado (5 epochs, mejor ep5: `S=49.4%`, `hard_neg=88.4%`).
 2. Run D esta en curso (full-unfreeze con split-LR) para cierre de foundation lock.
 3. Al cerrar Run D, tabla comparativa C/D (con referencia historica A/B/C) -> seleccionar ganador final por:
   - Primario: S = min(A2M@10, M2A@10)
   - Desempate 1: hard_neg
   - Desempate 2: menor asimetría |A2M - M2A|
 4. Checkpoint ganador final = foundation para Gate 4.2 screening.
 5. Freeze policy ganadora = policy primaria para Gate 4.2.

 Cuadros de arquitectura/configuracion por run (preflight real):
 - Fuente: `data/bias_control_medium/training_outputs/bloqueA_runA_log.txt`
 - Fuente: `data/bias_control_medium/training_outputs/bloqueA_runB_log.txt`
 - Fuente: `data/bias_control_medium/training_outputs/bloqueA_runC_log.txt`
 - Fuente: `data/bias_control_medium/training_outputs/bloqueA_runD/training.log`

 Run A (adapter bottleneck)
 | Module Group | Trainable | Frozen | Status |
 |---|---:|---:|---|
 | Audio Adapters | 528,640 | 0 | TRAIN |
 | Audio CNN | 0 | 3,158,528 | FROZEN |
 | Audio PosEmb | 0 | 6,144,000 | FROZEN |
 | Audio Projection | 920,832 | 0 | TRAIN |
 | Audio Transformer | 0 | 50,384,896 | FROZEN |
 | MIDI Embedding | 316,928 | 0 | TRAIN |
 | MIDI OutputNorm | 1,024 | 0 | TRAIN |
 | MIDI Projection | 658,688 | 0 | TRAIN |
 | MIDI Transformer | 12,609,536 | 0 | TRAIN |
 | **TOTAL** | **15,035,648** | **59,687,424** | |
 - LR: `adapters=5e-4`, `midi_encoder=5e-5`, `projections=1e-4`

 Run B (partial unfreeze)
 | Module Group | Trainable | Frozen | Status |
 |---|---:|---:|---|
 | Audio CNN | 0 | 3,158,528 | FROZEN |
 | Audio PosEmb | 0 | 6,144,000 | FROZEN |
 | Audio Projection | 920,832 | 0 | TRAIN |
 | Audio Transformer | 25,192,448 | 25,192,448 | MIXED |
 | MIDI Embedding | 316,928 | 0 | TRAIN |
 | MIDI OutputNorm | 1,024 | 0 | TRAIN |
 | MIDI Projection | 658,688 | 0 | TRAIN |
 | MIDI Transformer | 12,609,536 | 0 | TRAIN |
 | **TOTAL** | **39,699,456** | **34,494,976** | |
 - LR: `audio_layers_2_3=1e-5`, `midi_encoder=5e-5`, `projections=1e-4`

 Run C (hybrid)
 | Module Group | Trainable | Frozen | Status |
 |---|---:|---:|---|
 | Audio Adapters | 264,320 | 0 | TRAIN |
 | Audio CNN | 0 | 3,158,528 | FROZEN |
 | Audio PosEmb | 0 | 6,144,000 | FROZEN |
 | Audio Projection | 920,832 | 0 | TRAIN |
 | Audio Transformer | 25,192,448 | 25,192,448 | MIXED |
 | MIDI Embedding | 316,928 | 0 | TRAIN |
 | MIDI OutputNorm | 1,024 | 0 | TRAIN |
 | MIDI Projection | 658,688 | 0 | TRAIN |
 | MIDI Transformer | 12,609,536 | 0 | TRAIN |
 | **TOTAL** | **39,963,776** | **34,494,976** | |
 - LR: `adapters=5e-4`, `audio_layers_2_3=1e-5`, `midi_encoder=5e-5`, `projections=1e-4`

 Run D (full unfreeze, split-LR)
 | Module Group | Trainable | Frozen | Status |
 |---|---:|---:|---|
 | Audio CNN | 0 | 3,158,528 | FROZEN |
 | Audio PosEmb | 0 | 6,144,000 | FROZEN |
 | Audio Projection | 920,832 | 0 | TRAIN |
 | Audio Transformer | 50,384,896 | 0 | TRAIN |
 | MIDI Embedding | 316,928 | 0 | TRAIN |
 | MIDI OutputNorm | 1,024 | 0 | TRAIN |
 | MIDI Projection | 658,688 | 0 | TRAIN |
 | MIDI Transformer | 12,609,536 | 0 | TRAIN |
 | **TOTAL** | **64,891,904** | **9,302,528** | |
 - LR: `audio_layers_0_1=5e-6`, `audio_layers_2_3=1e-5`, `midi_encoder=5e-5`, `projections=1e-4`

 Foundation loader explícito (FIX v2.1 #3)
 Ganador: Run B
 Checkpoint: bloqueA_runB/checkpoint_epoch{N}_base.pt
 Loader: load_base_model() → CrossModalModel directo
 Model type para Gate 4.2: CrossModalModel
 ────────────────────────────────────────
 Ganador: Run C
 Checkpoint: bloqueA_runC/checkpoint_epoch{N}.pt
 Loader: Reconstruir BloqueAModel desde arch_config, luego bloqueA_model.base_model como foundation CrossModalModel. Los adapters 0-1 de
   Run C se DESCARTAN — Gate 4.2 parte solo del CrossModalModel base.
 Model type para Gate 4.2: CrossModalModel
 Justificación de descartar adapters Run C: Gate 4.2 testea ratio descriptors, no adapters. Usar adapters contaminaría la variable. El
 foundation siempre es un CrossModalModel puro con los pesos del backbone entrenado.

 Gate de extracción (FIX v2.1 #3b): Al extraer base_model de Run C (descartando adapters 0-1), el rendimiento puede caer porque los
 adapters contribuían a la calidad. Antes de usar este base_model como foundation:
 1. Evaluar base_model extraído con evaluate_structured_pool.py
 2. Si S_base < S_runC_full - 1.5pp: la extracción pierde demasiado → usar Run B como foundation (ya es CrossModalModel puro, sin pérdida
 por extracción)
 3. Si S_base >= S_runC_full - 1.5pp: la extracción es tolerable → usar Run C base como foundation

 Esto garantiza que el foundation para Gate 4.2 sea el mejor CrossModalModel puro disponible.

 ---
 Fase 1: Infraestructura (archivos a modificar/crear)

 1.1 Ordenar notas por onset + exponer midi_onset (OBLIGATORIO para Stage 1)

 Archivo: src/bias_control/datasets/maestro_segments.py

 FIX v2.1 #2: D4 depende del orden temporal de las notas. El dataset actual itera por instrumento (línea 264), no por onset global. Aunque
  MAESTRO es piano (1 instrumento), no es garantía.

 Cambios en _load_midi_segment() (línea ~260):
 1. Recolectar (onset, pitch, velocity, duration_sec, duration_bucket) tuples de TODOS los instrumentos
 2. Ordenar por (onset, pitch) antes de construir tensores
 3. Agregar midi_onset como tensor float (onset relativo al inicio del segmento)
 4. Agregar midi_duration_sec como tensor float (duración en segundos, NO bucketed — necesario para D3)

 Cambios en collate_segments() (línea ~316):
 1. Padear midi_onset (padding = -1.0) y midi_duration_sec (padding = 0.0)
 2. Retornar ambos en batch dict

 Sobre el sort-by-onset: El MIDIEncoder usa positional encoding sinusoidal, por lo que el orden de entrada SÍ afecta la semántica. Ordenar
  por onset es el comportamiento correcto: queremos que la posición encode el orden temporal de las notas. Esto es un cambio semántico
 menor pero correcto: antes las notas podían estar en orden arbitrario per-instrumento, ahora están en orden temporal global. Dado que
 MAESTRO es piano (1 instrumento, notes ya suelen estar en orden de onset por pretty_midi), el impacto práctico es mínimo.

 Backward-compatible: Los campos adicionales midi_onset y midi_duration_sec son ignorados por scripts existentes (bloqueA_training solo
 usa pitch/velocity/duration/mask).

 1.2 Crear módulo de descriptores de ratios

 Archivo nuevo: src/bias_control/ratio_descriptors.py (~250 líneas)

 Contendrá funciones de cómputo de cada descriptor. Para D1 y D2 reutiliza directamente las funciones existentes de
 gate4_ratio_auxiliary.py:
 - compute_batch_ratio_histograms() — líneas 187-212 (base para D1)
 - compute_batch_ratio_histograms_enriched() — líneas 215-330 (base para D2)
 - RatioEncoder — líneas 77-111

 Funciones nuevas:
 - compute_descriptor_d3() — IOI ratios + duration ratios + pitch intervals
 - compute_local_interval_features() — features per-note para D4

 1.3 Crear script de training Gate 4.2

 Archivo nuevo: experiments/bias_control/gate42_training.py (~600 líneas)

 Importa de infraestructura existente:
 # De bloqueA_training.py:
 from experiments.bias_control.bloqueA_training import (
     run_structured_eval,     # Evaluación canónica (línea 512)
     quick_val_eval,          # Eval rápido
 )
 # NOTA: NO importar train_loop — Gate 4.2 tiene su propio loop
 # porque la loss computation difiere (auxiliary ratio branch)

 # De evaluate_structured_pool.py:
 from experiments.bias_control.evaluate_structured_pool import (
     extract_all_embeddings,  # línea 133 — espera forward() → (audio, midi)
     build_segment_index,
     evaluate_with_precomputed_embeddings,
     analyze_hard_negatives_fast,
     PoolConfig,
 )

 1.4 Reutilizar código existente de Gate 4

 Archivo existente: experiments/bias_control/gate4_ratio_auxiliary.py

 Importar directamente:
 - RatioEncoder (MLP encoder para histogramas) — lines 77-111
 - compute_batch_ratio_histograms() — lines 187-212 (D1)
 - compute_batch_ratio_histograms_enriched() — lines 215-330 (D2)

 NO reutilizar:
 - MultiViewModel — reemplazada por Gate42Model (interfaz 2-output)
 - Training loop de Gate 4 — obsoleto

 ---
 Fase 2: Descriptores — 5 Variantes Concretas

 D0: Control (sin ratios)

 Objetivo: Medir cuánto mejora el modelo con entrenamiento continuado SIN ratio descriptors.
 ┌──────────────┬───────────────────────────────────────────┐
 │    Campo     │                   Valor                   │
 ├──────────────┼───────────────────────────────────────────┤
 │ Input        │ Ninguno adicional                         │
 ├──────────────┼───────────────────────────────────────────┤
 │ Output       │ N/A                                       │
 ├──────────────┼───────────────────────────────────────────┤
 │ Encoder      │ N/A                                       │
 ├──────────────┼───────────────────────────────────────────┤
 │ Loss         │ VICReg(audio, midi) — idéntico a Bloque A │
 ├──────────────┼───────────────────────────────────────────┤
 │ Params extra │ 0                                         │
 ├──────────────┼───────────────────────────────────────────┤
 │ Stage        │ 1                                         │
 └──────────────┴───────────────────────────────────────────┘
 D1: Pitch Ratio Histogram (baseline)

 Objetivo: Retestear el descriptor más simple de Gate 4 sobre foundation sano.
 ┌─────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │      Campo      │                                                    Valor                                                     │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Input           │ midi_pitch [B, N], midi_mask [B, N]                                                                          │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Cómputo         │ MIDI pitch → freq (Hz). Ratios pairwise f_i/f_j. Soft binning Gaussiano en [0.5, 2.0], 128 bins. Normalizar. │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Output          │ [B, 128]                                                                                                     │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Encoder         │ RatioEncoder(128, 1, hidden=128, out=64) → ProjectionHead(64→256)                                            │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Loss            │ VICReg(audio, midi) + λ * [VICReg(audio, ratio) + VICReg(midi, ratio)]                                       │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ λ               │ 0.1                                                                                                          │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Params extra    │ ~50K                                                                                                         │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Stage           │ 1                                                                                                            │
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Requiere onsets │ No                                                                                                           │
 └─────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 Código base: gate4_ratio_auxiliary.py:compute_batch_ratio_histograms() con n_bins=128.

 D2: Enriched Multi-Channel (velocity + duration weighted)

 Objetivo: Enriquecer D1 con expresividad musical.
 ┌─────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
 │      Campo      │                                         Valor                                          │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
 │ Input           │ midi_pitch, midi_velocity, midi_duration, midi_mask                                    │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
 │ Cómputo         │ 3 canales: (1) velocity-weighted, (2) duration-weighted, (3) unweighted. 128 bins × 3. │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
 │ Output          │ [B, 384] (flatten de [B, 128, 3])                                                      │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
 │ Encoder         │ RatioEncoder(128, 3, hidden=256, out=128) → ProjectionHead(128→256)                    │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
 │ Loss            │ Auxiliary VICReg, λ=0.1                                                                │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
 │ Params extra    │ ~130K                                                                                  │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
 │ Stage           │ 2 (solo si D1 muestra señal)                                                           │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
 │ Requiere onsets │ No                                                                                     │
 └─────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘
 D3: Temporal-Rhythmic Ratios

 Objetivo: Explorar dimensión temporal (ritmo) — NO testeada anteriormente.
 Campo: Input
 Valor: midi_pitch, midi_onset (NUEVO), midi_duration_sec (NUEVO), midi_mask
 ────────────────────────────────────────
 Campo: Cómputo
 Valor: (a) IOI ratios: IOI_n = onset_{n+1} - onset_n (notas consecutivas, pre-ordenadas por onset con desempate por pitch). Ratio: IOI_n
 /
    IOI_{n+1}. Clamp IOI mínimo a 10ms para evitar div/0. Soft binning [0.25, 4.0], 64 bins. (b) Duration ratios: dur_n / dur_{n+1}
   (usando midi_duration_sec float en segundos, NO los buckets discretos de midi_duration). Clamp mínimo 50ms. 64 bins [0.25, 4.0].
   (c) Pitch interval histogram: `
 ────────────────────────────────────────
 Campo: Output
 Valor: [B, 153]
 ────────────────────────────────────────
 Campo: Encoder
 Valor: RatioEncoder(153, 1, hidden=128, out=64) → ProjectionHead(64→256)
 ────────────────────────────────────────
 Campo: Loss
 Valor: Auxiliary VICReg, λ=0.1
 ────────────────────────────────────────
 Campo: Params extra
 Valor: ~55K
 ────────────────────────────────────────
 Campo: Stage
 Valor: 2 (solo si Stage 1 muestra señal)
 ────────────────────────────────────────
 Campo: Requiere onsets
 Valor: Sí (Fase 1.3)
 D4: Input-Augmented Local Intervals

 Objetivo: Testear mecanismo de integración DIFERENTE (input features vs auxiliary loss).
 Campo: Input
 Valor: midi_pitch [B, N], midi_mask [B, N]
 ────────────────────────────────────────
 Campo: Cómputo por nota
 Valor: (1) semitone_prev = pitch[i] - pitch[i-1] (0 para primera), (2) semitone_next = pitch[i+1] - pitch[i] (0 para última), (3)
   log_ratio_prev = log2(freq[i]/freq[i-1]) clamped [-2,2], (4) log_ratio_next (ídem). Normalizar features a [-1, 1].
 ────────────────────────────────────────
 Campo: Output
 Valor: [B, N, 4] features adicionales por nota
 ────────────────────────────────────────
 Campo: Integración
 Valor: AugmentedMIDIEncoder wrapper (ver arquitectura abajo) — intercepta DESPUÉS de event_embedding, ANTES de pos_encoding:
   [B,N,512]+[B,N,4] → Linear(516,512) → continua pipeline normal del MIDIEncoder.
 ────────────────────────────────────────
 Campo: Loss
 Valor: VICReg(audio, midi) — SIN auxiliary loss
 ────────────────────────────────────────
 Campo: Params extra
 Valor: ~265K (Linear 516→512 con bias)
 ────────────────────────────────────────
 Campo: Stage
 Valor: 1
 ────────────────────────────────────────
 Campo: Requiere onsets
 Valor: No (notas ya en orden por extracción)
 ---
 Arquitectura del Modelo (FIX bloqueantes #2 y #4)

 Principio: forward() SIEMPRE retorna 2 tensores

 extract_all_embeddings() (evaluate_structured_pool.py:133) llama model(audio, pitch, vel, dur, mask) y espera exactamente (audio_emb,
 midi_emb). TODAS las variantes Gate 4.2 respetan esto.

 Gate42Model (para D0, D1, D2, D3)

 class Gate42Model(nn.Module):
     """
     Wrapper para descriptores con auxiliary loss.
     forward() → 2 outputs (eval-compatible).
     compute_total_loss() → maneja ratio branch internamente.
     """
     def __init__(self, base_model, descriptor_fn, ratio_encoder,
                  ratio_projection, ratio_weight=0.1):
         self.base_model = base_model
         self.descriptor_fn = descriptor_fn  # None para D0
         self.ratio_encoder = ratio_encoder
         self.ratio_projection = ratio_projection
         self.ratio_weight = ratio_weight

     def forward(self, audio, midi_pitch, midi_velocity, midi_duration,
                 midi_mask=None):
         """SIEMPRE 2 outputs — compatible con extract_all_embeddings."""
         audio_emb = self.base_model.encode_audio(audio)
         midi_emb = self.base_model.encode_midi(
             pitch=midi_pitch, velocity=midi_velocity,
             duration=midi_duration, padding_mask=midi_mask)
         return audio_emb, midi_emb

     def compute_total_loss(self, audio, midi_pitch, midi_velocity,
                            midi_duration, midi_mask=None, midi_onset=None):
         """Training loss con ratio branch auxiliar."""
         audio_emb, midi_emb = self.forward(
             audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
         main_loss, metrics = self.base_model.compute_vicreg_loss(
             audio_emb, midi_emb)

         if self.descriptor_fn is None:  # D0 control
             return main_loss, metrics

         # Compute ratio descriptor + embed
         ratio_hist = self.descriptor_fn(
             midi_pitch, midi_velocity, midi_duration,
             midi_mask, midi_onset)
         ratio_emb = self.ratio_projection(self.ratio_encoder(ratio_hist))

         # Auxiliary VICReg losses (SIN detach — gradients fluyen a backbone)
         aux_a, _ = self.base_model.compute_vicreg_loss(audio_emb, ratio_emb)
         aux_m, _ = self.base_model.compute_vicreg_loss(midi_emb, ratio_emb)

         total = main_loss + self.ratio_weight * (aux_a + aux_m)
         metrics['ratio_aux_loss'] = (aux_a + aux_m).item()
         return total, metrics

 Gate42InputAugModel (para D4)

 class Gate42InputAugModel(nn.Module):
     """
     Wrapper para D4: MIDI encoder con input augmentation.
     forward() → 2 outputs (eval-compatible).
     """
     def __init__(self, base_model, interval_dim=4):
         self.base_model = base_model
         self.midi_encoder = base_model.midi_encoder  # referencia

         # Projection post-augmentation
         embed_dim = self.midi_encoder.embed_dim  # 512
         self.interval_projection = nn.Sequential(
             nn.Linear(embed_dim + interval_dim, embed_dim),
             nn.LayerNorm(embed_dim),
         )

     def forward(self, audio, midi_pitch, midi_velocity, midi_duration,
                 midi_mask=None):
         """2 outputs — compatible con extract_all_embeddings."""
         # Audio: normal
         audio_emb = self.base_model.encode_audio(audio)

         # MIDI: augmented forward
         midi_emb = self._encode_midi_augmented(
             midi_pitch, midi_velocity, midi_duration, midi_mask)
         midi_emb = self.base_model.midi_projection(midi_emb)

         return audio_emb, midi_emb

     def _encode_midi_augmented(self, pitch, velocity, duration, mask):
         """Forward MIDI con interval features intercalados."""
         enc = self.midi_encoder
         B, T = pitch.shape

         # Step 1: Event embedding (normal)
         x = enc.event_embedding(pitch, velocity, duration)  # [B, T, 512]

         # Step 2: Compute + inject interval features
         interval_feats = compute_local_interval_features(
             pitch, mask)  # [B, T, 4]
         x = torch.cat([x, interval_feats], dim=-1)  # [B, T, 516]
         x = self.interval_projection(x)  # [B, T, 512]

         # Step 3: CLS token (si aplica)
         padding_mask = mask
         if enc.aggregation == "cls":
             cls_tokens = enc.cls_token.expand(B, -1, -1)
             x = torch.cat([cls_tokens, x], dim=1)
             if padding_mask is not None:
                 cls_mask = torch.zeros(
                     B, 1, dtype=torch.bool, device=padding_mask.device)
                 padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

         # Step 4: Positional encoding
         x = enc.pos_encoding(x)

         # Step 5: Transformer
         if padding_mask is not None:
             x = enc.transformer(x, src_key_padding_mask=padding_mask)
         else:
             x = enc.transformer(x)

         # Step 6: Output norm
         x = enc.output_norm(x)

         # Step 7: Pooling
         if enc.aggregation == "mean":
             if padding_mask is not None:
                 m = ~padding_mask.unsqueeze(-1)
                 x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
             else:
                 x = x.mean(dim=1)
         elif enc.aggregation == "cls":
             x = x[:, 0, :]
         elif enc.aggregation == "attention":
             weights = enc.attention_pool(x)
             if padding_mask is not None:
                 weights = weights.masked_fill(
                     padding_mask.unsqueeze(-1), float("-inf"))
             weights = torch.softmax(weights, dim=1)
             x = (x * weights).sum(dim=1)

         return x  # [B, 512] — pre-projection

     def compute_total_loss(self, audio, midi_pitch, midi_velocity,
                            midi_duration, midi_mask=None, midi_onset=None):
         """D4 usa VICReg normal (sin auxiliary)."""
         audio_emb, midi_emb = self.forward(
             audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
         return self.base_model.compute_vicreg_loss(audio_emb, midi_emb)

 Por qué este diseño resuelve los bloqueantes:
 1. forward() SIEMPRE → (audio_emb, midi_emb) — compatible con extract_all_embeddings()
 2. D4 replica el pipeline completo del MIDIEncoder (event_emb → cls → pos → transformer → norm → pool) sin tocar midi_encoder.py
 3. compute_total_loss() encapsula toda la lógica de auxiliary loss internamente

 Checkpoint format (compatible con contrato Bloque A)

 D0, D1, D2, D3 (auxiliary loss):
 # Checkpoint principal: full state para resume
 torch.save({
     'model_state_dict': gate42_model.base_model.state_dict(),
     'ratio_state_dict': {
         'ratio_encoder': ratio_encoder.state_dict(),
         'ratio_projection': ratio_projection.state_dict(),
     },  # None para D0
     'arch_config': {
         'mode': 'gate42', 'descriptor': 'd1',
         'ratio_weight': 0.1,
         'foundation_checkpoint': '...', 'freeze_policy': 'run-b',
         'eval_compatible': True,
     },
     'optimizer_state_dict': ..., 'scheduler_state_dict': ..., 'epoch': epoch,
 }, checkpoint_path)

 # _base.pt separado: CrossModalModel puro → evaluate_structured_pool.py compatible
 torch.save({'model_state_dict': gate42_model.base_model.state_dict()},
            base_path)

 D4 (input augmentation) — FIX v2.1 #1:
 # Checkpoint principal: full state (incluyendo interval_projection)
 torch.save({
     'model_state_dict': gate42_model.base_model.state_dict(),
     'd4_state_dict': gate42_model.interval_projection.state_dict(),
     'arch_config': {
         'mode': 'gate42', 'descriptor': 'd4',
         'foundation_checkpoint': '...', 'freeze_policy': 'run-b',
         'eval_compatible': False,  # ← D4 NO es eval-compatible sin wrapper
     },
     'optimizer_state_dict': ..., 'scheduler_state_dict': ..., 'epoch': epoch,
 }, checkpoint_path)

 # _archive_base_not_for_eval.pt (nombre explícito para prevenir uso accidental)
 # Contiene solo CrossModalModel SIN augmentation — métricas serán peores que realidad
 torch.save({'model_state_dict': gate42_model.base_model.state_dict(),
             'eval_compatible': False},
            archive_path)

 Evaluación D4: gate42_training.py --mode evaluate --checkpoint D4_full.pt reconstruye Gate42InputAugModel desde arch_config, carga
 d4_state_dict, y evalúa con el forward augmentado.

 ---
 Preflight y Anti-Variable-Fantasma (FIX v2.1 #5)

 Todos los runs Gate 4.2 (screening, confirm, robustness) DEBEN ejecutar:

 1. validate_training_setup(model, optimizer, mode, ...) — preflight de 6 puntos (src/bias_control/training/preflight.py:50). Requiere
 contrato de frozen/trainable prefixes adaptado a Gate 4.2.
 2. DriftSentinel(model, trainable_prefixes) — snapshot de params al inicio, check después de epoch 1 para detectar frozen-by-accident.

 Contratos Gate 4.2 (FIX v2.1 #4 — prefijos explícitos por wrapper):

 Gate 4.2 usa CrossModalModel directo (sin BloqueAModel wrapper), por lo que todos los prefijos son SIN base_model.:

 # Gate42*Model wrappers almacenan CrossModalModel como self.base_model,
 # por lo que named_parameters() devuelve prefijos con 'base_model.'.
 # Los contratos usan estos prefijos REALES.

 # Foundation policy = run-b (más probable ganador)
 GATE42_FROZEN_BASE = [
     'base_model.audio_encoder.feature_extractor.',
     'base_model.audio_encoder.pos_embedding',
     'base_model.audio_encoder.transformer.layers.0.',
     'base_model.audio_encoder.transformer.layers.1.',
 ]
 GATE42_TRAINABLE_BASE = [
     'base_model.audio_encoder.transformer.layers.2.',
     'base_model.audio_encoder.transformer.layers.3.',
     'base_model.midi_encoder.',
     'base_model.audio_projection.',
     'base_model.midi_projection.',
 ]

 GATE42_CONTRACTS = {
     'd0': {
         'frozen_prefixes': GATE42_FROZEN_BASE,
         'trainable_prefixes': GATE42_TRAINABLE_BASE,
     },
     'd1': {
         'frozen_prefixes': GATE42_FROZEN_BASE,
         'trainable_prefixes': GATE42_TRAINABLE_BASE + [
             'ratio_encoder.', 'ratio_projection.',
         ],
     },
     'd2': {  # Igual que d1 (encoder más grande, mismos prefijos)
         'frozen_prefixes': GATE42_FROZEN_BASE,
         'trainable_prefixes': GATE42_TRAINABLE_BASE + [
             'ratio_encoder.', 'ratio_projection.',
         ],
     },
     'd3': {  # Igual que d1
         'frozen_prefixes': GATE42_FROZEN_BASE,
         'trainable_prefixes': GATE42_TRAINABLE_BASE + [
             'ratio_encoder.', 'ratio_projection.',
         ],
     },
     'd4': {
         'frozen_prefixes': GATE42_FROZEN_BASE,
         'trainable_prefixes': GATE42_TRAINABLE_BASE + [
             'interval_projection.',
         ],
     },
 }

 GATE42_PARAM_RANGES = {
     'd0': (39_000_000, 40_500_000),    # = run-b baseline
     'd1': (39_045_000, 40_555_000),    # + ~50K ratio params
     'd2': (39_120_000, 40_640_000),    # + ~130K ratio params
     'd3': (39_050_000, 40_560_000),    # + ~55K ratio params
     'd4': (39_260_000, 40_770_000),    # + ~265K interval_projection
 }

 ---
 Fase 3: Screening en 2 Stages

 Stage 1: D0, D1, D4 (onset ordering ya aplicado en Fase 1.1)

 Protocolo:
 1. Cargar foundation checkpoint (ver Foundation loader Fase 0)
 2. Aplicar MISMA freeze policy que ganador Bloque A
 3. Preflight + DriftSentinel obligatorios
 4. Mismo seed (42), max_batches (1000), batch_size (16)
 5. 3 epochs por descriptor, evaluación canónica por epoch

 Orden: D0 → D1 → D4

 Por qué este subset:
 - D0 = control (obligatorio)
 - D1 = test directo de pitch ratios con auxiliary loss (retestea Gate 4)
 - D4 = test de mecanismo alternativo (input augmentation)
 - Mínimo viable: 3 descriptores × 3 epochs con eval canónica por epoch

 Stage 2: D2, D3 (solo si Stage 1 muestra señal)

 Condición de activación: Al menos D1 o D4 supera criterio de promoción.

 Pre-requisito: D3 usa midi_onset (ya expuesto en Fase 1.1). Implementar compute_descriptor_d3() en ratio_descriptors.py.

 Protocolo: Mismo que Stage 1 (preflight + sentinel + eval canónica).

 Criterio de promoción (ambos stages)

 - Promueve si: S_Dx - S_D0 >= +0.5pp Y hard_neg_Dx >= hard_neg_D0 - 1pp
 - Si NINGUNO promueve en Stage 1 → DROP_RATIO (no ejecutar Stage 2)
 - Si promueven > 2 → tomar top-2 por ΔS

 CLI

 python experiments/bias_control/gate42_training.py \
     --descriptor {d0,d1,d2,d3,d4} \
     --checkpoint PATH_FOUNDATION \
     --output data/bias_control_medium/training_outputs/gate42/screening/{descriptor}/ \
     --maestro-dir data/maestro_v3/maestro-v3.0.0 \
     --epochs 3 \
     --ratio-weight 0.1 \
     --batch-size 16 --num-workers 8 \
     --max-val-batches 846 --embed-batch-size 16

 Tiempos estimados (corregidos, incluyendo eval ~10min/epoch)
 ┌───────────────┬──────────┬────────┬─────────────┬──────────┐
 │  Descriptor   │ Training │  Eval  │ Total/epoch │ 3 epochs │
 ├───────────────┼──────────┼────────┼─────────────┼──────────┤
 │ D0            │ ~18min   │ ~12min │ ~30min      │ ~1.5h    │
 ├───────────────┼──────────┼────────┼─────────────┼──────────┤
 │ D1            │ ~20min   │ ~12min │ ~32min      │ ~1.6h    │
 ├───────────────┼──────────┼────────┼─────────────┼──────────┤
 │ D4            │ ~20min   │ ~12min │ ~32min      │ ~1.6h    │
 ├───────────────┼──────────┼────────┼─────────────┼──────────┤
 │ Stage 1 total │          │        │             │ ~4.7h    │
 ├───────────────┼──────────┼────────┼─────────────┼──────────┤
 │ D2            │ ~22min   │ ~12min │ ~34min      │ ~1.7h    │
 ├───────────────┼──────────┼────────┼─────────────┼──────────┤
 │ D3            │ ~22min   │ ~12min │ ~34min      │ ~1.7h    │
 ├───────────────┼──────────┼────────┼─────────────┼──────────┤
 │ Stage 2 total │          │        │             │ ~3.4h    │
 └───────────────┴──────────┴────────┴─────────────┴──────────┘
 ---
 Fase 4: Confirmación (5 epochs)

 Protocolo

 1. Tomar top-2 descriptores + D0 control
 2. Correr 5 epochs completos (run limpio desde foundation, NO reanudar screening)
 3. Mismo protocolo canónico

 Criterio GO_RATIO

 - S_Dx - S_D0 >= +1.5pp en al menos 2 de las últimas 3 epochs
 - hard_neg_Dx >= hard_neg_D0 - 1pp
 - Sin degradación de asimetría: |A2M - M2A|_Dx < |A2M - M2A|_D0 + 3pp

 ---
 Fase 5: Robustez (opcional)

 Cuándo

 Solo si algún descriptor pasa confirmación.

 Protocolo

 - Correr mejor descriptor en policy secundaria (= mejor alternativa entre Run B/C)
 - 5 epochs, mismo protocolo

 Criterio

 - S_Dx - S_D0 >= +1.0pp (umbral relajado)
 - Pasa: descriptor robusto a policy. Falla: frágil (artefacto de policy).

 ---
 Fase 6: Decisión Formal
 ┌──────────────────┬──────────────────────────────────────────────────────┐
 │    Veredicto     │                       Criterio                       │
 ├──────────────────┼──────────────────────────────────────────────────────┤
 │ GO_RATIO         │ ≥1 descriptor pasa confirmación + robustez           │
 ├──────────────────┼──────────────────────────────────────────────────────┤
 │ GO_RATIO_PARCIAL │ Pasa confirmación pero falla robustez                │
 ├──────────────────┼──────────────────────────────────────────────────────┤
 │ INCONCLUSO       │ Señal en screening pero no sostenida en confirmación │
 ├──────────────────┼──────────────────────────────────────────────────────┤
 │ DROP_RATIO       │ Ningún descriptor promueve en screening              │
 └──────────────────┴──────────────────────────────────────────────────────┘
 Anti-goalpost

 - Máximo 5 descriptores en screening (2 stages)
 - Máximo 2 en confirmación + D0
 - Máximo 1 en robustez
 - Total máximo: 5×3 + 3×5 + 1×5 = 35 epochs ≈ 20h GPU

 ---
 Archivos a Crear/Modificar
 Acción: Modificar
 Archivo: src/bias_control/datasets/maestro_segments.py
 Cambio: Sort notes por (onset,pitch) + exponer midi_onset (~15 lín)
 Fase: 1.1
 ────────────────────────────────────────
 Acción: Crear
 Archivo: src/bias_control/ratio_descriptors.py
 Cambio: D1-D4 computation + local intervals (~250 lín)
 Fase: 1.2
 ────────────────────────────────────────
 Acción: Crear
 Archivo: experiments/bias_control/gate42_training.py
 Cambio: Gate42Model, Gate42InputAugModel, train loop, CLI, preflight (~700 lín)
 Fase: 1.3
 ────────────────────────────────────────
 Acción: NO modificar
 Archivo: bloqueA_training.py
 Cambio: Import run_structured_eval/quick_val_eval
 Fase: —
 ────────────────────────────────────────
 Acción: NO modificar
 Archivo: cross_modal_model.py, mert_encoder.py, midi_encoder.py
 Cambio: Intactos
 Fase: —
 ────────────────────────────────────────
 Acción: Reutilizar
 Archivo: gate4_ratio_auxiliary.py
 Cambio: Import RatioEncoder, compute_batch_ratio_histograms[_enriched]
 Fase: 1.2
 ────────────────────────────────────────
 Acción: Reutilizar
 Archivo: src/bias_control/training/preflight.py
 Cambio: Import validate_training_setup, DriftSentinel
 Fase: 1.3
 ---
 Verificación

 1. Smoke test D1: 1 epoch, batch_size=4 → preflight pasa, loss computa, checkpoint guarda, eval pasa
 2. Smoke test D4: 1 epoch, batch_size=4 → augmented forward shapes correctas, DriftSentinel detecta drift en interval_projection
 3. Eval D0/D1/D2/D3: evaluate_structured_pool.py --model *_base.pt funciona sin wrapper
 4. Eval D4: gate42_training.py --mode evaluate --checkpoint D4_full.pt reconstruye wrapper y evalúa correctamente.
 *_archive_base_not_for_eval.pt tiene eval_compatible: False en metadata.
 5. D0 consistency: D0 epoch 1 S ≈ foundation S (no degradación al recargar)
 6. Ratio signal: ratio_emb std_z > 0 (no colapso del ratio encoder)
 7. Note ordering: Verificar que midi_pitch en batch está en orden de onset (print primeras 10 notas de un segment)

 ---
 Outputs Esperados

 data/bias_control_medium/training_outputs/gate42/
 ├── screening/
 │   ├── d0/  (eval_per_epoch/eval_epoch{1-3}.json, checkpoint_epoch{1-3}.pt)
 │   ├── d1/
 │   ├── d4/
 │   ├── d2/  (solo Stage 2)
 │   └── d3/  (solo Stage 2)
 ├── confirm/       (solo si screening promueve)
 │   ├── d0/  (5 epochs)
 │   ├── {best_1}/
 │   └── {best_2}/
 ├── robustness/    (solo si confirmación pasa)
 │   └── {winner}/
 └── GATE42_DECISION.md

 ---
 Secuencia de Ejecución

 PRE-REQUISITO:
 ├── Run C cerrado (mejor ep5: S=49.4)            [completado]
 ├── Run D full-unfreeze (split-LR)               [en curso]
 ├── Tabla comparativa C/D + lock final           [~30 min tras Run D]
 └── Foundation lock definitivo

 IMPLEMENTACIÓN STAGE 1 (~3h):
 ├── 1. maestro_segments.py (sort + midi_onset)    ~15 min
 ├── 2. ratio_descriptors.py (D1, D4 functions)    ~1h
 ├── 3. gate42_training.py (modelos + loop + eval) ~1.5h
 └── 4. Smoke tests D0/D1/D4                      ~10 min

 SCREENING STAGE 1 (~4.7h GPU):
 ├── 5. D0 control (3 epochs)                     ~1.5h
 ├── 6. D1 pitch ratio (3 epochs)                 ~1.6h
 └── 7. D4 input-augmented (3 epochs)             ~1.6h

 ANÁLISIS STAGE 1 (~15 min):
 └── 8. Tabla comparativa + decisión Stage 2

 IMPLEMENTACIÓN STAGE 2 (condicional, ~30 min):
 └── 9. Implementar D2, D3 en ratio_descriptors    ~30 min

 SCREENING STAGE 2 (condicional, ~3.4h GPU):
 ├── 10. D2 enriched (3 epochs)                   ~1.7h
 └── 11. D3 temporal (3 epochs)                   ~1.7h

 CONFIRMACIÓN (condicional, ~7.5h GPU):
 ├── 12. D0 control full (5 epochs)               ~2.5h
 ├── 13. Top-1 (5 epochs)                         ~2.5h
 └── 14. Top-2 (5 epochs)                         ~2.5h

 ROBUSTEZ (condicional, ~2.5h GPU):
 └── 15. Winner en policy secundaria (5 epochs)   ~2.5h

 DECISIÓN:
 └── 16. GATE42_DECISION.md + update roadmap      ~30 min

 ---
 Interpretación de Resultados
 ┌───────────┬───────────┬────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
 │ Escenario │ D1 (aux)  │ D4 (input) │                                      Interpretación                                      │
 ├───────────┼───────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
 │ A         │ Mejora    │ Mejora     │ Ratios aportan señal robusta, modelo no los captura solo. Fuerte evidencia pro-ratios.   │
 ├───────────┼───────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
 │ B         │ Mejora    │ No mejora  │ Auxiliary loss funciona, ratios como global histogram son informativos.                  │
 ├───────────┼───────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
 │ C         │ No mejora │ Mejora     │ Mecanismo auxiliary loss es el problema, no los ratios. Input augmentation es el camino. │
 ├───────────┼───────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
 │ D         │ No mejora │ No mejora  │ Modelo end-to-end YA captura información de ratios implícitamente. DROP_RATIO.           │
 ├───────────┼───────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
 │ E         │ —         │ —          │ D3 mejora, D1 no → ratios temporales > pitch. Redirige exploración a ritmo.              │
 └───────────┴───────────┴────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘
 Todos los escenarios son informativos para la tesis de Phideus.

 ---
 Fixes Aplicados

 Auditoría Codex v1→v2
 #: 1
 Bloqueante: train_loop_base no existe
 Severidad: Crítico
 Fix: No importar train_loop. Gate 4.2 tiene su propio loop porque compute_total_loss() difiere de VICReg puro. Importar solo
   run_structured_eval y quick_val_eval.
 ────────────────────────────────────────
 #: 2
 Bloqueante: extract_all_embeddings espera 2 outputs
 Severidad: Crítico
 Fix: Gate42Model.forward() SIEMPRE retorna (audio_emb, midi_emb). Ratio branch encapsulada en compute_total_loss().
 ────────────────────────────────────────
 #: 3
 Bloqueante: Notes no ordenadas por onset
 Severidad: Alto
 Fix: Sort notes por (onset, pitch) en _load_midi_segment() antes de construir tensores.
 ────────────────────────────────────────
 #: 4
 Bloqueante: D4 requiere wrapper MIDI completo
 Severidad: Alto
 Fix: Gate42InputAugModel._encode_midi_augmented() replica pipeline completo.
 ────────────────────────────────────────
 #: 5
 Bloqueante: Tiempo subestimado
 Severidad: Medio
 Fix: Presupuesto: ~30-34 min/epoch. Stage 1 ≈ 4.7h.
 ────────────────────────────────────────
 #: +
 Bloqueante: Screening en 2 stages
 Severidad: Mejora
 Fix: Stage 1 (D0/D1/D4) → Stage 2 (D2/D3) solo si hay señal.
 Auditoría Codex v2→v2.1 (primera ronda)
 #: 1
 Bloqueante: D4 no evaluable con base.pt
 Severidad: Alto
 Fix: D4 checkpoint: eval_compatible: False. Eval solo via --mode evaluate que reconstruye wrapper.
 ────────────────────────────────────────
 #: 2
 Bloqueante: D4 requiere orden temporal en Stage 1
 Severidad: Alto
 Fix: Sort-by-(onset,pitch) + midi_onset movido a Fase 1.1 obligatoria.
 ────────────────────────────────────────
 #: 3
 Bloqueante: Foundation Run C no resuelto
 Severidad: Alto
 Fix: Loader explícito + gate de extracción (si S cae >1.5pp, usar Run B).
 ────────────────────────────────────────
 #: 4
 Bloqueante: Doble definición detach/no-detach
 Severidad: Medio
 Fix: Eliminada versión con detach.
 ────────────────────────────────────────
 #: 5
 Bloqueante: Falta preflight/sentinel
 Severidad: Medio
 Fix: Sección Preflight con contratos y DriftSentinel obligatorio.
 Auditoría Codex v2.1 (segunda ronda — ajustes finales)
 #: 1
 Ajuste: D3 cortado + fuente de duración
 Fix: D3 spec completa. Agregado midi_duration_sec (float) al dataset. D3 usa float, no buckets.
 ────────────────────────────────────────
 #: 2
 Ajuste: Foundation Run C gate
 Fix: Gate de extracción: evaluar base_model extraído; si S cae >1.5pp vs full → usar Run B.
 ────────────────────────────────────────
 #: 3
 Ajuste: "Transformer order-agnostic" incorrecto
 Fix: Corregido: PE sinusoidal hace que el orden SÍ importe. Sort-by-onset es cambio semántico correcto y documentado.
 ────────────────────────────────────────
 #: 4
 Ajuste: Preflight prefijos por wrapper
 Fix: Contratos con prefijos explícitos por descriptor, incluyendo nota sobre base_model. prefix en wrappers Gate42.
