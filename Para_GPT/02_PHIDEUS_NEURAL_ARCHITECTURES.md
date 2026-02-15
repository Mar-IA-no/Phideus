# PHIDEUS — Neural Architectures & Hyperparameters

**Fecha**: 2026-02-15
**Referencia tecnica para onboarding de agentes AI y colaboradores**

---

## 1. Modelo Principal: CrossModalModel (BIAS_CONTROL)

Archivo: `src/bias_control/architectures/cross_modal_model.py`

### 1.1 Audio Encoder — MERTEncoderLite

Archivo: `src/bias_control/encoders/mert_encoder.py`

Encoder de audio inspirado en MERT (Music Understanding Model), implementacion "lite" local (no usa el modelo 330M de HuggingFace para training — solo para referencia de pesos iniciales).

```
Input: Waveform [B, 96000] (4s @ 24kHz)
  |
  v
Feature Extractor (4x Conv1d):
  Conv1d(1, 256, k=10, s=5) + GroupNorm(32) + GELU    -> [B, 256, 19200]
  Conv1d(256, 512, k=3, s=2) + GroupNorm(32) + GELU   -> [B, 512, 9600]
  Conv1d(512, 512, k=3, s=2) + GroupNorm(32) + GELU   -> [B, 512, 4800]
  Conv1d(512, 1024, k=3, s=2) + GroupNorm(32) + GELU  -> [B, 1024, 2400]
  |
  transpose -> [B, 2400, 1024]   (T'=2400 frames, downsample 96000/2400 = 40x)
  |
  + Positional Embedding (learnable) [1, max_pos_len, 1024]
  |
  v
Transformer Encoder (4 layers):
  nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=4096, dropout=0.1)
  |
  v
Mean Pooling -> [B, 1024]
```

**Parametros**: ~59.7M (CNN: ~3.2M, PosEmb: ~6.1M, Transformer: ~50.4M)
**Sample rate**: 24000 Hz

### 1.2 MIDI Encoder — MIDIEncoder

Archivo: `src/bias_control/encoders/midi_encoder.py`

```
Input: pitch [B, N], velocity [B, N], duration [B, N]  (N = num events, max ~2048)
  |
  v
Event Embedding:
  pitch_embed: Embedding(128, 256)      # D/2
  velocity_embed: Embedding(128, 128)   # D/4
  duration_embed: Embedding(32, 128)    # D/4
  |
  concat -> [B, N, 512]
  |
  Linear(512, 512) + LayerNorm(512)
  |
  v
Sinusoidal Positional Encoding:
  pe[pos, 2i]   = sin(pos / 10000^(2i/d))
  pe[pos, 2i+1] = cos(pos / 10000^(2i/d))
  + Dropout(0.1)
  |
  v
Transformer Encoder (4 layers):
  nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
  |
  v
Output Norm: LayerNorm(512)
  |
  v
Mean Pooling (or CLS) -> [B, 512]
```

**Parametros**: ~13.9M (Embedding: ~0.3M, Transformer: ~12.6M, Norm: ~1K)

### 1.3 Projection Heads

```
Audio Projection:  Linear(1024, 512) + BatchNorm1d(512) + ReLU + Linear(512, 256)
MIDI Projection:   Linear(512, 512) + BatchNorm1d(512) + ReLU + Linear(512, 256)
```

**Output**: Embeddings [B, 256] para ambos lados.
**Parametros**: Audio ~0.66M, MIDI ~0.66M. Total projections: ~1.3M.

### 1.4 VICReg Loss

```python
vicreg_invariance_weight = 10.0    # MSE between paired embeddings
vicreg_variance_weight   = 10.0    # Hinge loss on std (target > 1)
vicreg_covariance_weight = 1.0     # Off-diagonal covariance penalty
```

No requiere negative sampling. Los tres terminos previenen colapso de representacion.

### 1.5 Total de Parametros

| Componente | Parametros |
|-----------|-----------|
| Audio encoder (MERTEncoderLite) | ~59.7M |
| MIDI encoder | ~13.9M |
| Audio projection | ~0.66M |
| MIDI projection | ~0.66M |
| **Total base** | **~75M** |

---

## 2. Freeze Policies (Bloque A)

Cada "Run" desbloquea mas parametros del audio encoder:

| Policy | Audio CNN | Audio PosEmb | Audio Transformer | Trainable total |
|--------|-----------|-------------|-------------------|-----------------|
| **run-a** | FROZEN | FROZEN | FROZEN + Adapters | ~15M (adapters + MIDI + proj) |
| **run-b** | FROZEN | FROZEN | Layers 2-3 unfrozen | ~39M |
| **run-c** | FROZEN | FROZEN | Adapters 0-1 + Unfreeze 2-3 | ~40M |
| **run-d** | FROZEN | FROZEN | ALL unfrozen | ~66M (full unfreeze) |

**Nota**: En run-d, las CNN features y positional embeddings quedan congeladas. Solo se descongelan los 4 Transformer layers del audio encoder + MIDI completo + projections.

### Learning Rates por Grupo (run-d)

```python
lr_audio_unfreeze = 1e-5       # Audio transformer layers
lr_audio_low      = 5e-6       # Audio transformer (lower layers)
lr_midi           = 5e-5       # MIDI encoder
lr_proj           = 1e-4       # Projection heads
lr_ratio          = 5e-4       # Descriptor projections (nuevos params)
```

### Scheduler

Cosine annealing con warmup:
- warmup_steps = 200
- Decay to 10% of initial LR
- Calculado sobre total_steps = epochs * max_batches_per_epoch

---

## 3. Gate 4.3 — Modelos con Descriptores

Archivo principal: `experiments/bias_control/gate42_training.py`

### 3.1 D4 — MIDI Local Intervals (Concat)

**Clase**: `Gate42InputAugModel`
**Descriptor**: `compute_local_interval_features(midi_pitch, midi_mask)` -> [B, N, 4]

4 features por nota MIDI:
```
semitone_prev  = (pitch[i] - pitch[i-1]) / 24     # intervalo hacia atras
semitone_next  = (pitch[i+1] - pitch[i]) / 24     # intervalo hacia adelante
log_ratio_prev = clamp((pitch[i]-pitch[i-1])/12, [-2,2]) / 2
log_ratio_next = clamp((pitch[i+1]-pitch[i])/12, [-2,2]) / 2
```

**Pipeline de inyeccion**:
```
Event Embedding [B, N, 512]
    + Local Intervals [B, N, 4]   (torch.no_grad, detached)
    -> concat [B, N, 516]
    -> interval_projection: Linear(516, 512) + LayerNorm(512)
    -> [B, N, 512]
    -> Positional Encoding
    -> Transformer (4 layers)
    -> Pool -> MIDI Projection -> [B, 256]
```

**Params nuevos**: ~267K (Linear 516->512 + LN)

### 3.2 A4 — Audio Log-Freq Deltas (Concat)

**Clase**: `Gate42AudioAugModel`
**Descriptor**: `compute_audio_descriptor_a4(audio, target_length=2400)` -> [B, 2400, 8]

Archivo: `src/bias_control/audio_descriptors.py`

```
Audio [B, 96000]
  -> torch.stft(n_fft=2048, hop=512, hann_window) -> magnitude [B, 1025, 188]
  -> log1p(magnitude)
  -> Group into 8 log-freq bands:
     [4-8], [8-16], [16-32], [32-64], [64-128], [128-256], [256-512], [512-1025] bins
     (~47Hz to ~12kHz)
  -> [B, 8, 188]
  -> Temporal delta: diff(dim=-1), pad first frame with zeros
  -> Normalize per band per sample (zero mean, unit std)
  -> F.interpolate(size=2400, mode='linear') -> [B, 8, 2400]
  -> transpose -> [B, 2400, 8]
```

**Pipeline de inyeccion**:
```
CNN Features [B, 2400, 1024]
    + Audio Descriptor [B, 2400, 8]   (torch.no_grad, detached)
    -> concat [B, 2400, 1032]
    -> audio_descriptor_projection: Linear(1032, 1024) + LayerNorm(1024)
    -> [B, 2400, 1024]
    -> + Positional Embedding
    -> Transformer (4 layers)
    -> Pool -> Audio Projection -> [B, 256]
```

**Params nuevos**: ~1.06M (Linear 1032->1024 + LN)

### 3.3 A7 — Rational Attractor (Concat)

**Clase**: `Gate42AudioAugModel` (misma clase, audio_descriptor_type='a7')
**Descriptor**: `compute_audio_descriptor_a7(audio, target_length=2400)` -> [B, 2400, 12]

```
Audio [B, 96000]
  -> STFT -> magnitude [B, 1025, 188]
  -> topk(magnitude, k=8, dim=1) -> 8 peaks por frame
  -> Sort peaks by frequency (not magnitude)
  -> Filter peaks < 50Hz
  -> Pairwise log2 ratios: r_ij = log2(f_j / f_i) mod 1.0   (octave-folded)
  -> Soft Gaussian assignment to 12 JI attractors (sigma=0.02):
     activation = exp(-0.5 * ((r - attractor) / sigma)^2) * sqrt(mag_i * mag_j)
  -> Sum over C(8,2)=28 pairs -> [B, 12, 188]
  -> Normalize per frame (sum=1)
  -> F.interpolate -> [B, 12, 2400]
  -> transpose -> [B, 2400, 12]
```

**12 atractores Just Intonation** (log2, octave-folded):
```
 0: 1:1   = 0.000  (unisono/octava)      6: 7:5  = 0.485  (tritono)
 1: 16:15 = 0.093  (2da menor)           7: 3:2  = 0.585  (5ta justa)
 2: 9:8   = 0.170  (2da mayor)           8: 8:5  = 0.678  (6ta menor)
 3: 6:5   = 0.263  (3ra menor)           9: 5:3  = 0.737  (6ta mayor)
 4: 5:4   = 0.322  (3ra mayor)          10: 7:4  = 0.807  (7ma armonica)
 5: 4:3   = 0.415  (4ta justa)          11: 15:8 = 0.907  (7ma mayor)
```

**Params nuevos**: ~1.06M (Linear 1036->1024 + LN)

### 3.4 A4x / A7x — Audio Cross-Attention

**Clase**: `Gate42AudioCrossAttModel`

```
CNN Features [B, 2400, 1024]
    + Positional Embedding (ANTES de cross-attention)
    |
    v  Query: features [B, 2400, 1024]

Audio Descriptor [B, 188, 8/12]     (resolucion nativa STFT, SIN interpolar)
    -> descriptor_kv_proj: Linear(8/12, 1024) -> [B, 188, 1024]
    |
    v  Key/Value: [B, 188, 1024]

nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True, dropout=0.1)
    query  = features [B, 2400, 1024]
    key    = descriptor_proj [B, 188, 1024]
    value  = descriptor_proj [B, 188, 1024]
    need_weights = False
    |
    v  attn_output [B, 2400, 1024]

features = LayerNorm(features + attn_output)   (residual + norm)
    |
    v
Transformer (4 layers, pos_emb ya aplicado, NO se repite)
    -> Pool -> Audio Projection -> [B, 256]
```

**Attn matrix**: [B*8, 2400, 188] = ~57.7M elements. ~231MB fp32.
K/V a resolucion nativa STFT (188 frames), 12.8x menos memoria que si se interpolara a 2400.

**Params nuevos**: ~4.2M (kv_proj ~9-13K + MHA in/out_proj ~4.19M + LN ~2K)

### 3.5 D4x — MIDI Cross-Attention

**Clase**: `Gate42MidiCrossAttModel`

```
Event Embedding [B, N, 512]
    + Positional Encoding (ANTES de cross-attention)
    + CLS token (if aggregation="cls") prepended to Q
    |
    v  Query: embeddings [B, N(+1), 512]

Local Intervals [B, N, 4]
    -> interval_kv_proj: Linear(4, 512) -> [B, N, 512]
    |
    v  Key/Value: [B, N, 512]

nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.1)
    |
    v  attn_output [B, N(+1), 512]

embeddings = LayerNorm(embeddings + attn_output)
    |
    v
Transformer (4 layers) -> Pool -> MIDI Projection -> [B, 256]
```

Q y K/V a misma resolucion (N tokens). Sin mismatch temporal.
**Params nuevos**: ~1.05M (d=512 vs d=1024 en audio)

### 3.6 d4a4 — Dual Same-Modality Concat

**Clase**: `Gate42DualAugModel`

Combina D4 (MIDI side) + A4 (Audio side). Cada descriptor va a su propio encoder.

```
Audio side: CNN -> concat(features, A4_desc) -> audio_descriptor_projection -> Transformer -> proj
MIDI side:  Embedding -> concat(tokens, D4_intervals) -> interval_projection -> Transformer -> proj
```

**Params nuevos**: ~1.3M (audio_descriptor_projection + interval_projection)
**Resultado**: S=69.8% — GANADOR de Gate 4.3, superaditivo.

### 3.7 d4a4cm — Dual Cross-Modal

**Clase**: `Gate42DualCrossModalModel`

Inyecta descriptores CRUZADOS:
- MIDI intervals (D4) -> Audio encoder (via F.interpolate N tokens -> T'=2400 frames)
- Audio descriptors (A4) -> MIDI encoder (via F.interpolate T_stft=188 -> N tokens)

```
Audio side: CNN -> concat(features, D4_MIDI_intervals_interpolated) -> cross_modal_audio_proj -> Transformer
MIDI side:  Embedding -> concat(tokens, A4_audio_desc_interpolated) -> cross_modal_midi_proj -> Transformer
```

**Params nuevos**: ~1.3M
**Resultado**: S=52.4% — PEOR que baseline. Cross-modal injection es destructiva.

---

## 4. Modelos Historicos

### 4.1 DANN (Gate 3)

Gradient Reversal Layer entre embeddings y domain classifier:
```
[Audio emb, MIDI emb] -> concat [2B, 256] -> L2-normalize
    -> GRL (identity forward, -lambda*grad backward)
    -> Domain Classifier: 256->64->64->2 (ReLU + Dropout)
    -> CE Loss (domain)
```

Lambda schedule: linear 0->1 over training.
**Resultado**: No mejora sobre Gate 2. CERRADO.

### 4.2 RosetaVAE (UOEMD)

Archivo: `src/RNA/roseta_vae.py`

Dual-domain VAE para Audio-Vibracion con:
- z_shared [B, T, 32]: latente compartido entre dominios
- z_private [B, T, 16]: latente privado por dominio
- Reparameterization trick
- Decoders por dominio
- Losses: Recon + KL_shared + KL_private + InfoNCE(z_shared) + Diff(shared, private)

**Resultado**: NO-GO (gap aligned-shuffled = 0.007).

### 4.3 ConstellationVAE (UOEMD)

Archivo: `src/RNA/constellation_vae.py`

4 configuraciones: encoder (MLP vs Transformer) x decoder (Histogram vs Token)
- Token input: [B, T, 48, 5] = {log_ratio, delta_t, weight, anchor_band, target_band}
- MLP encoder: Token MLP(5->128) + AttnPool(48->1) + BiLSTM
- Transformer encoder: Token Embed + CLS + SelfAttn + Temporal TF

**Resultado**: NO-GO (Top-1 = 0.78%, = random).

### 4.4 JEPA-Lite (UOEMD)

Archivo: `src/RNA/jepa_lite.py`

Sin decoder — usa predictores bidireccionales:
- z_audio [B, T, 32], z_vib [B, T, 32]
- Stop gradient en targets
- Predictor a->v: MLP(32->64->64->32)
- Predictor v->a: MLP(32->64->64->32)
- InfoNCE contrastive loss

**Resultado**: Top-1 = 1.56% (marginalmente mejor que random).

### 4.5 HRM — Hierarchical Reasoning Model

Archivo: `src/hrm/`

- L-Module (rapido): encoder MLP + GRU(2 layers, d=256) + Spectral Attention (8 heads)
- H-Module (lento): L-Aggregator(10x256->128) + Harmonic Attention(4 heads) + Memory LSTM(128)
- ACT: Q-Network para adaptive halting (min=2, max=20 steps)
- Loop: L produce -> H agrega -> ACT decide -> repeat o halt

---

## 5. Training Configuration (Gate 4.3)

### Datos

```
Dataset:        MAESTRO v3.0.0 (1276 piezas de piano)
segment_len:    4.0s
hop:            1.0s
sample_rate:    24000 Hz
Audio shape:    [B, 96000]
MIDI max events: ~2048 per segment
Train/Val split: standard MAESTRO split
```

### Training Loop

```
batch_size:           16
max_batches_per_epoch: 1000
max_val_batches:      846
num_workers:          8
pin_memory:           True
grad_clip:            clip_grad_norm_(model.parameters(), max_norm=1.0)
seed:                 42
device:               cuda (RTX 3090)
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
```

### Optimizer

```
AdamW, per-param-group LRs (see section 2)
Weight decay: 0.01 (default AdamW)
```

### Scheduler

```
CosineAnnealingLR con warmup lineal
warmup_steps: 200
eta_min: 0.1 * initial_lr (por grupo)
```

### Checkpointing

```
Cada epoch guarda: checkpoint_epoch{N}.pt
  - model_state_dict
  - optimizer_state_dict
  - scheduler_state_dict
  - epoch number
  - training_history
Resume: --resume path/to/checkpoint.pt
```

### Epoch Timing

```
~30 min/epoch con --skip-structured-eval
~36 min/epoch con structured eval (embedding extraction)
```

### Evaluation

```
Script:     experiments/bias_control/evaluate_structured_pool.py
Flag:       --model path/to/best_model.pt (NOT --checkpoint)
Pool:       256 candidates (1 pos + 64 hard + 32 semi-hard + 159 random)
Queries:    500
Seed:       42
Metricas:   S, A2M_R@10, M2A_R@10, hard_neg, MRR, R@1, R@5, R@20

JSON keys:  d['gate_metrics']['S'], d['gate_metrics']['a2m_r10'],
            d['gate_metrics']['hard_neg']
```

---

## 6. Tabla de Parametros por Modelo

| Modelo | Base params | Nuevos params | Total trainable (run-d) | S |
|--------|------------|---------------|------------------------|---|
| D0 (baseline) | 75M | 0 | ~66M | 60.2% |
| D4 (MIDI concat) | 75M | ~267K | ~66.3M | 63.6% |
| A4 (audio concat) | 75M | ~1.06M | ~67M | 63.6% |
| A7 (audio concat) | 75M | ~1.06M | ~67M | 58.8% |
| A4x (audio cross-att) | 75M | ~4.2M | ~70M | 62.6% |
| A7x (audio cross-att) | 75M | ~4.2M | ~70M | 62.2% |
| D4x (MIDI cross-att) | 75M | ~1.05M | ~67M | 60.0% |
| d4a4 (dual concat) | 75M | ~1.3M | ~67.3M | **69.8%** |
| d4a4cm (dual cross-modal) | 75M | ~1.3M | ~67.3M | 52.4% |

---

## 7. Resolucion Temporal de Cada Componente

| Componente | Resolucion | Frames (4s) | Fuente |
|-----------|-----------|-------------|--------|
| Audio waveform | 24kHz | 96,000 | Input |
| STFT (n_fft=2048, hop=512) | ~21ms/frame | 188 | Audio descriptors |
| CNN output (post 4x downsample) | ~1.67ms/frame | 2,400 | Audio encoder |
| Audio Transformer | Same as CNN | 2,400 | Audio encoder |
| MIDI events | Variable | N (~50-200) | MIDI encoder |

**Cross-attention resolutions**:
- Audio (A4x/A7x): Q=2400 (CNN), K/V=188 (STFT). Mismatch resuelto por MHA.
- MIDI (D4x): Q=N, K/V=N. Misma resolucion.

---

## 8. Preflight Contract

Cada run verifica antes de empezar:
1. **Param ranges**: Total trainable params dentro de rango esperado
2. **Trainable prefixes**: Solo los modules correctos tienen requires_grad=True
3. **Frozen verification**: Modules que deben estar frozen lo estan
4. **Drift sentinel**: Monitorea cambio relativo de norma de pesos durante training (<1% para pre-trained, >1% para random init es esperado)

Ejemplo preflight para d4a4 (run-d):
```
Trainable prefixes:
  - base_model.audio_encoder.transformer.
  - base_model.midi_encoder.
  - base_model.audio_projection.
  - base_model.midi_projection.
  - audio_descriptor_projection.
  - interval_projection.

Frozen prefixes:
  - base_model.audio_encoder.feature_extractor.
  - base_model.audio_encoder.pos_embedding
```

---

## 9. Visualizaciones Interactivas

8 arquitecturas visualizadas en WebGL2 en [altermundi.github.io/Phideus/](https://altermundi.github.io/Phideus/):

| Viz | Ruta | Arquitectura | Color |
|-----|------|-------------|-------|
| MERT + MIDI Transformer | /phideus | Foundation Run D | #3366cc |
| Hybrid Adapter Fine-Tuning | /bloquea | Bloque A Run C | #cc6633 |
| Cross-Attention Injection | /crossatt | Gate 4.3 | #cc3366 |
| Domain Adversarial Network | /dann | Gate 3 DANN | #996633 |
| Hierarchical Reasoning | /hrm | HRM | #339966 |
| ConstellationVAE | /constellation | C1-C4 | #cc9933 |
| JEPA-Lite | /jepa | No-decoder predictive | #6633cc |
| RosetaVAE | /roseta | Dual-domain VAE | #9933cc |

Cada visualizacion tiene 4-6 fases de walkthrough interactivo explicando la arquitectura paso a paso.
