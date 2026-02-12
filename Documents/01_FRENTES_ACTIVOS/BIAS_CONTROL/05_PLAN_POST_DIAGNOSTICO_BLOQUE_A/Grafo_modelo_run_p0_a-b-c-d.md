  Arquitectura de Run D — Resumen                                                                                                       
                                                                                                                                           
  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐                                                               
  │       AUDIO TOWER               │    │       MIDI TOWER                │
  │                                 │    │                                 │                                                               
  │  Waveform [B, T] @ 24kHz       │    │  Events [B, N] (pitch/vel/dur) │                                                                 
  │           │                     │    │           │                     │                                                               
  │  ┌────────▼────────┐           │    │  ┌────────▼────────┐           │                                                                 
  │  │ CNN Feature Ext. │ FROZEN    │    │  │ Event Embedding │ TRAINABLE │                                                                
  │  │ 4x Conv1d       │ (gris)    │    │  │ pitch(D/2) +    │           │                                                                 
  │  │ 1→512→512→      │           │    │  │ vel(D/4) +      │           │                                                                 
  │  │  512→1024        │           │    │  │ dur(D/4) → D    │           │                                                                
  │  └────────┬────────┘           │    │  │ + Linear + LN   │           │                                                                 
  │           │                     │    │  └────────┬────────┘           │
  │  ┌────────▼────────┐           │    │           │                     │
  │  │ Pos Embedding    │ FROZEN    │    │  ┌────────▼────────┐           │
  │  │ learnable params │ (gris)    │    │  │ Sinusoidal PosE │           │
  │  └────────┬────────┘           │    │  └────────┬────────┘           │
  │           │                     │    │           │                     │
  │  ┌────────▼────────┐           │    │  ┌────────▼────────┐           │
  │  │ Transformer L0   │ TRAINABLE │    │  │ Transformer L0   │ TRAINABLE │
  │  │ d=1024, 8 heads  │ lr_low    │    │  │ d=512, 8 heads   │           │
  │  ├─────────────────┤           │    │  ├─────────────────┤           │
  │  │ Transformer L1   │ TRAINABLE │    │  │ Transformer L1   │           │
  │  │                  │ lr_low    │    │  │                  │           │
  │  ├─────────────────┤           │    │  ├─────────────────┤           │
  │  │ Transformer L2   │ TRAINABLE │    │  │ Transformer L2   │           │
  │  │                  │ lr_high   │    │  │                  │           │
  │  ├─────────────────┤           │    │  ├─────────────────┤           │
  │  │ Transformer L3   │ TRAINABLE │    │  │ Transformer L3   │           │
  │  │                  │ lr_high   │    │  │ + Output LN      │           │
  │  └────────┬────────┘           │    │  └────────┬────────┘           │
  │           │                     │    │           │                     │
  │     Mean Pooling → [B, 1024]   │    │     Mean Pooling → [B, 512]   │
  └───────────┬─────────────────────┘    └───────────┬─────────────────────┘
              │                                       │
     ┌────────▼────────┐                    ┌────────▼────────┐
     │ Audio Projection │ TRAINABLE          │ MIDI Projection  │ TRAINABLE
     │ MLP 3-layer     │                    │ MLP 3-layer     │
     │ 1024→512→512→256│                    │ 512→512→512→256 │
     │ BN + ReLU       │                    │ BN + ReLU       │
     └────────┬────────┘                    └────────┬────────┘
              │                                       │
              └──────────┐         ┌──────────────────┘
                         ▼         ▼
                ┌────────────────────┐
                │  Shared Space 256D │
                │                    │
                │  VICReg Loss:      │
                │  inv=10, var=10,   │
                │  cov=1             │
                └────────────────────┘
