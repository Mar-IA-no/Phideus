# Plan de Ejecución: Fase 3A - Ratio Constellations

**Fecha**: 2026-01-31
**Objetivo**: Implementar representación sparse de tokens (constellations) para superar la limitación del histograma denso
**Criterio de éxito**: Gap aligned-shuffled > 0.15

---

## Resumen Ejecutivo

### Por qué Ratio Constellations

La Fase 2 demostró que el VAE colapsa la información discriminativa del histograma:
- **Pre-red gap**: 0.691 (172× mejor que v1)
- **Post-red gap**: 0.007 (solo 3.5× mejor)
- **Problema**: El histograma pierde "quién se relaciona con quién"

**Solución**: Representación sparse inspirada en [Shazam audio fingerprinting](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf) - tokens que preservan relaciones estructurales entre picos.

### Investigación SOTA

| Referencia | Concepto Clave | Aplicación a Phideus |
|------------|----------------|---------------------|
| [Shazam Algorithm](https://www.cameronmacleod.com/blog/how-does-shazam-work) | Anchor-target pairs con (f1, f2, Δt) | Base para tokens de ratio |
| [PeakNetFP (ISMIR 2025)](https://arxiv.org/abs/2506.21086) | Peaks sparse + PointNet++ + contrastive | Arquitectura de encoder |
| [Audio-JEPA](https://arxiv.org/abs/2507.02915) | Predicción latente sin decoder | Alternativa a VAE (Fase 3B) |
| [SparseVLM (ICML 2025)](https://icml.cc/virtual/2025/poster/46297) | Token sparsification eficiente | Técnicas de pruning |

---

## Arquitectura Propuesta

### Formato de Token (Constellation)

```python
# En lugar de histograma [T, 256, 3], generamos tokens sparse
token = {
    'log_ratio': np.log2(target.freq / anchor.freq),  # [-2.58, 2.58] para ratios [1, 6]
    'delta_t': target.time - anchor.time,              # Offset temporal (frames)
    'weight': np.sqrt(anchor.amp * target.amp),        # Peso combinado
    'anchor_band': get_band_id(anchor.freq),           # Banda frecuencial (0-7)
    'target_band': get_band_id(target.freq),           # Banda del target
}

# Output shape: [T, max_tokens, 5] donde max_tokens = K * M
# Con K=12 anchors, M=4 targets: max_tokens = 48
```

### Pipeline de Extracción

```
Señal Raw
    ↓
STFT + Peak Picking (v2.2 existente)
    ↓
Temporal Stability Filter (v2.2 existente)
    ↓
┌─────────────────────────────────────────┐
│ NUEVO: Constellation Extraction         │
│                                         │
│ Para cada pico anchor (K=12):           │
│   Seleccionar M=4 targets cercanos      │
│   Generar token (log_ratio, Δt, w, b)   │
│                                         │
│ Output: lista de tokens por frame       │
└─────────────────────────────────────────┘
    ↓
Padding a [T, 48, 5]
```

### Arquitectura del Encoder

**Opción A: MLP + Attention Pooling** (Recomendada - Relacional y eficiente)

> **NOTA**: Usamos **attention-weighted pooling** en vez de mean pooling para preservar relaciones entre tokens (crítica GPT5.2Think - Riesgo A).

```python
class MLPConstellationEncoder(nn.Module):
    def __init__(self, token_dim=5, hidden_dim=128, z_dim=32):
        # Token embedding
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Attention pooling (preserva relaciones, no solo promedio)
        self.attention_query = nn.Linear(hidden_dim, 1)
        # Temporal LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True)
        # Heads para z
        self.z_shared_head = nn.Linear(hidden_dim * 2, z_dim * 2)
        self.z_private_head = nn.Linear(hidden_dim * 2, z_dim)

    def forward(self, tokens, mask):
        # tokens: [B, T, max_tokens, 5]
        # mask: [B, T, max_tokens] (1=valid, 0=pad)

        # 1. Embed each token
        x = self.token_mlp(tokens)  # [B, T, max_tokens, hidden]

        # 2. Attention-weighted pooling per timestep
        attn_logits = self.attention_query(x).squeeze(-1)  # [B, T, max_tokens]
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, T, max_tokens]
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=2)  # [B, T, hidden]

        # 3. Temporal modeling
        x, _ = self.lstm(x)  # [B, T, hidden*2]

        # 4. Generate z
        z_shared = self.z_shared_head(x)
        z_private = self.z_private_head(x)
        return z_shared, z_private
```

**Opción B: Transformer con Self-Attention** (Más expresivo, más costoso)
```python
class TransformerConstellationEncoder(nn.Module):
    # Self-attention sobre tokens dentro de cada frame
    # Pooling por CLS token o attention pooling
    # Cross-attention temporal entre frames
```

### Decoder (Ambas Variaciones)

**Decoder A: Reconstrucción a Histograma**
```python
class HistogramDecoder(nn.Module):
    # z_shared + z_private → [T, 256, 3] histograma
    # Permite comparar directamente con v2.2
    # Loss: MSE sobre histograma reconstruido
```

**Decoder B: Reconstrucción a Tokens**
```python
class TokenDecoder(nn.Module):
    # z → [T, max_tokens, 5] tokens predichos
    # Loss: MSE sobre tokens + máscara de padding
    # Más consistente con representación sparse
```

### Matriz de Configuraciones (6 variantes)

> **NOTA**: Agregamos C5/C6 sin decoder (JEPA-lite) para evitar shortcut reconstructivo (crítica GPT5.2Think - Riesgo B).

| Config | Encoder | Decoder | Loss Principal | Descripción |
|--------|---------|---------|----------------|-------------|
| **C1** | MLP+Attention | Histograma | Recon + KL + InfoNCE | Baseline comparable con v2.2 |
| **C2** | MLP+Attention | Tokens | Recon + KL + InfoNCE | Encoder simple, decoder consistente |
| **C3** | Transformer | Histograma | Recon + KL + InfoNCE | Expresivo, comparable con v2.2 |
| **C4** | Transformer | Tokens | Recon + KL + InfoNCE | Full sparse pipeline |
| **C5** | MLP+Attention | **Sin decoder** | **InfoNCE + Predictor** | JEPA-lite: predicción cross-modal en latente |
| **C6** | Transformer | **Sin decoder** | **InfoNCE + Predictor** | JEPA-lite: máxima expresividad sin shortcut |

**Variantes JEPA-lite (C5/C6)**:
```python
class JEPALiteModel(nn.Module):
    def __init__(self, encoder_type='mlp'):
        self.encoder = MLPConstellationEncoder(...) if encoder_type == 'mlp' else TransformerConstellationEncoder(...)
        # Predictor cross-modal (sin decoder)
        self.predictor = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, audio_tokens, vib_tokens, audio_mask, vib_mask):
        z_audio = self.encoder(audio_tokens, audio_mask)
        z_vib = self.encoder(vib_tokens, vib_mask)
        z_vib_pred = self.predictor(z_audio)
        # Loss: cosine similarity(z_vib_pred, z_vib.detach())
        return z_audio, z_vib, z_vib_pred
```

**Estrategia**: Entrenar las 6 variantes y comparar en gap aligned-shuffled **intra-condición**.

---

## Plan de Implementación

### Fase 3A-0: Auditoría de Evaluación (1 día) - PREREQUISITO

> **CRÍTICO**: GPT5.2Think identificó inconsistencia en reportes (10.94% vs 0.78% Top-1). Antes de ejecutar 3A, debemos garantizar consistencia.

**Archivos a auditar:**
- `experiments/evaluate_retrieval.py`
- `experiments/evaluate_cross_reconstruction.py`

**Tareas:**
1. Verificar que embeddings se guardan/cargan con `sample_id` consistente
2. Agregar logging de dimensiones y ordenamiento
3. Ejecutar test de sanidad: mismos embeddings → mismo resultado
4. Documentar cualquier discrepancia encontrada

**Verificación:**
```bash
# Test de consistencia: correr 2 veces, verificar mismo resultado
python experiments/evaluate_retrieval.py \
    --model data/training_outputs/roseta_v22/best_model.pt \
    --seed 42 --output /tmp/eval_run1.json

python experiments/evaluate_retrieval.py \
    --model data/training_outputs/roseta_v22/best_model.pt \
    --seed 42 --output /tmp/eval_run2.json

diff /tmp/eval_run1.json /tmp/eval_run2.json  # Debe ser idéntico
```

---

### Fase 3A-1: Extractor de Constellations (2-3 días)

**Archivos a modificar:**
- `src/analizador/analizador_roseta.py` - Agregar `extract_constellation()`

**Tareas:**
1. Implementar `extract_constellation()`:
   ```python
   def extract_constellation(
       stable_peaks: List[Dict],
       max_targets_per_anchor: int = 4,
       target_zone_frames: int = 5,
       target_zone_hz: float = 500.0,
   ) -> np.ndarray:
       """
       Para cada anchor, selecciona M targets más cercanos.
       Output: [n_tokens, 5] (log_ratio, delta_t, weight, anchor_band, target_band)
       """
   ```

2. Agregar CLI flag `--output-format constellation`

3. Modificar `save_roseta_dataset()` para nuevo formato NPZ

**Verificación:**
```bash
# Test con señal sintética
pytest tests/test_constellation_extractor.py -v
```

### Fase 3A-2: Dataset Loader (1 día)

**Archivos a modificar:**
- `src/datasets/roseta_dataset.py` - Agregar soporte para tokens

**Tareas:**
1. Detectar formato automáticamente (histograma vs constellation)
2. Implementar `collate_constellation_sequences()` con padding y máscaras
3. Retornar `(audio_tokens, vib_tokens, audio_mask, vib_mask, lengths)`

### Fase 3A-3: Modelos ConstellationVAE (3-4 días)

**Archivos a crear:**
- `src/RNA/constellation_vae.py` - Modelo base con encoders/decoders intercambiables

**Tareas:**
1. Implementar `MLPConstellationEncoder` (MLP + pooling + LSTM)
2. Implementar `TransformerConstellationEncoder` (self-attention sobre tokens)
3. Implementar `HistogramDecoder` (z → [T, 256, 3])
4. Implementar `TokenDecoder` (z → [T, max_tokens, 5])
5. Clase wrapper `ConstellationVAE` que acepta encoder/decoder como parámetros
6. Mantener loss functions existentes + nueva loss para tokens
7. Agregar máscara en todas las operaciones

**Arquitectura modular:**
```python
class ConstellationVAE(nn.Module):
    def __init__(self, encoder_type='mlp', decoder_type='histogram'):
        if encoder_type == 'mlp':
            self.encoder = MLPConstellationEncoder(...)
        else:
            self.encoder = TransformerConstellationEncoder(...)

        if decoder_type == 'histogram':
            self.decoder = HistogramDecoder(...)
        else:
            self.decoder = TokenDecoder(...)
```

### Fase 3A-4: Training Loop (1 día)

**Archivos a modificar:**
- `experiments/run_roseta_experiment.py` - Agregar soporte para constellation

**Tareas:**
1. Detectar formato de dataset
2. Instanciar modelo correcto (RosetaVAE vs ConstellationVAE)
3. Pasar máscaras a forward()

### Fase 3A-5: Sweep Arquitectural y Evaluación (3-4 días)

**Tareas:**

**1. Generar dataset constellation:**
```bash
python src/analizador/analizador_roseta.py \
    --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files \
    --output data/datasets/roseta_constellation.npz \
    --output-format constellation \
    --max-targets-per-anchor 4 \
    --workers 12
```

**2. Entrenar las 6 configuraciones:**
```bash
# C1: MLP+Attention encoder + Histogram decoder
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --output data/training_outputs/constellation_C1_mlp_hist \
    --model constellation \
    --encoder-type mlp \
    --decoder-type histogram \
    --epochs 100

# C2: MLP+Attention encoder + Token decoder
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --output data/training_outputs/constellation_C2_mlp_token \
    --model constellation \
    --encoder-type mlp \
    --decoder-type token \
    --epochs 100

# C3: Transformer encoder + Histogram decoder
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --output data/training_outputs/constellation_C3_trans_hist \
    --model constellation \
    --encoder-type transformer \
    --decoder-type histogram \
    --epochs 100

# C4: Transformer encoder + Token decoder
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --output data/training_outputs/constellation_C4_trans_token \
    --model constellation \
    --encoder-type transformer \
    --decoder-type token \
    --epochs 100

# C5: MLP+Attention encoder + JEPA-lite (sin decoder)
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --output data/training_outputs/constellation_C5_mlp_jepa \
    --model jepa-lite \
    --encoder-type mlp \
    --epochs 100

# C6: Transformer encoder + JEPA-lite (sin decoder)
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --output data/training_outputs/constellation_C6_trans_jepa \
    --model jepa-lite \
    --encoder-type transformer \
    --epochs 100
```

**3. Evaluación comparativa (protocolo P0 mejorado):**

> **NOTA**: Usamos **hard negatives intra-condición** como métrica principal (crítica GPT5.2Think - Riesgo C).

```bash
# Evaluar cada configuración con hard negatives
for config in C1_mlp_hist C2_mlp_token C3_trans_hist C4_trans_token C5_mlp_jepa C6_trans_jepa; do
    python experiments/evaluate_cross_reconstruction.py \
        --model data/training_outputs/constellation_${config}/best_model.pt \
        --run-all-controls \
        --use-hard-negatives \
        --output data/evaluations/constellation_${config}/

    python experiments/evaluate_retrieval.py \
        --model data/training_outputs/constellation_${config}/best_model.pt \
        --eval-intra-condition \
        --output data/evaluations/constellation_${config}/
done
```

**Controles hard-negatives:**
- **Shuffled global**: Cualquier par aleatorio (baseline fácil)
- **Shuffled intra-condición**: Par de la misma condición pero diferente archivo (hard negative)
- **Shuffled intra-archivo**: Par del mismo archivo pero diferente timestamp (very hard negative)

**4. Tabla comparativa de resultados:**

| Config | Encoder | Decoder | Gap (global) | Gap (intra-cond) | Retr Top-1 (intra) | Silhouette |
|--------|---------|---------|--------------|------------------|-------------------|------------|
| C1 | MLP+Attn | Histogram | ? | ? | ? | ? |
| C2 | MLP+Attn | Token | ? | ? | ? | ? |
| C3 | Transformer | Histogram | ? | ? | ? | ? |
| C4 | Transformer | Token | ? | ? | ? | ? |
| C5 | MLP+Attn | JEPA-lite | ? | ? | ? | ? |
| C6 | Transformer | JEPA-lite | ? | ? | ? | ? |

**Criterio de selección**: La configuración con mejor **gap aligned-shuffled intra-condición** será la elegida para claim H3. El gap global es secundario.

---

## Archivos Críticos

| Archivo | Acción | Cambios |
|---------|--------|---------|
| `src/analizador/analizador_roseta.py` | Modificar | +`extract_constellation()`, +CLI flag `--output-format` |
| `src/datasets/roseta_dataset.py` | Modificar | +`RosetaConstellationDataset`, +collate con máscaras |
| `src/RNA/constellation_vae.py` | **Crear** | Modelo modular con 2 encoders × 3 decoders |
| `src/RNA/jepa_lite.py` | **Crear** | Modelo JEPA sin decoder (C5/C6) |
| `src/RNA/encoders/mlp_encoder.py` | **Crear** | MLP + **attention pooling** + LSTM encoder |
| `src/RNA/encoders/transformer_encoder.py` | **Crear** | Self-attention encoder |
| `src/RNA/decoders/histogram_decoder.py` | **Crear** | Decoder a histograma [T, 256, 3] |
| `src/RNA/decoders/token_decoder.py` | **Crear** | Decoder a tokens [T, 48, 5] |
| `experiments/run_roseta_experiment.py` | Modificar | +flags `--model`, `--encoder-type`, `--decoder-type` |
| `experiments/evaluate_retrieval.py` | Modificar | +`--eval-intra-condition`, +hard negatives |
| `experiments/evaluate_cross_reconstruction.py` | Modificar | +`--use-hard-negatives` |
| `tests/test_constellation_extractor.py` | **Crear** | Tests para extractor de tokens |
| `tests/test_constellation_vae.py` | **Crear** | Tests para las 6 configuraciones |

---

## Criterios GO/NO-GO

### GO-3A: Constellation Funciona

> **NOTA**: Usamos gap **intra-condición** como criterio crítico (más estricto que global).

| Criterio | Umbral | Peso |
|----------|--------|------|
| **Gap aligned-shuffled (intra-cond)** | **> 0.10** | **CRÍTICO** |
| Gap aligned-shuffled (global) | > 0.15 | Alto |
| Retrieval Top-1 (intra-cond) | > 2× random | Alto |
| Silhouette score | > 0.20 | Medio |
| Token coverage | > 80% frames con ≥10 tokens | Medio |

**Decisión por configuración**:
- Si **alguna de C1-C6** cumple gap intra-cond > 0.10 → **GO** con esa config
- Si **solo C5/C6** (JEPA-lite) cumplen → La arquitectura sin decoder es necesaria
- Si **ninguna** cumple → **NO-GO**, analizar por qué

**Después de GO**:
- Documentar mejor configuración
- Claim: "H3 supported under Protocol P0 + Constellations + [config]"
- Planificar Rosetta v3

---

## Contingencia: Fase 3B (PRISM-JEPA)

Si Fase 3A falla, el problema puede ser el decoder VAE (shortcut de reconstrucción).

**PRISM-JEPA elimina el decoder:**
- Encoder de tokens → z_shared
- Predictor: z_audio → z_vib_pred (sin decoder)
- Loss: cosine similarity en espacio latente
- Inspirado en [VL-JEPA](https://arxiv.org/abs/2512.10942) y [Audio-JEPA](https://arxiv.org/abs/2507.02915)

---

## Verificación End-to-End

```bash
# 1. Verificar extractor genera tokens
python src/analizador/analizador_roseta.py \
    --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files/1_Unloaded_Condition \
    --output /tmp/test_constellation.npz \
    --output-format constellation \
    --workers 1
python -c "import numpy as np; d=np.load('/tmp/test_constellation.npz', allow_pickle=True); print(f'Keys: {list(d.keys())[:10]}')"

# 2. Verificar loader funciona
python -c "
from src.datasets.roseta_dataset import RosetaConstellationDataset
ds = RosetaConstellationDataset('/tmp/test_constellation.npz')
print(f'Samples: {len(ds)}')
sample = ds[0]
print(f'Audio tokens: {sample[\"audio_tokens\"].shape}')
print(f'Audio mask: {sample[\"audio_mask\"].shape}')
"

# 3. Verificar las 4 configuraciones de modelo
python -c "
import torch
from src.RNA.constellation_vae import ConstellationVAE

configs = [
    ('mlp', 'histogram'),
    ('mlp', 'token'),
    ('transformer', 'histogram'),
    ('transformer', 'token'),
]

for enc, dec in configs:
    model = ConstellationVAE(encoder_type=enc, decoder_type=dec)
    x = torch.randn(2, 50, 48, 5)  # [B, T, tokens, features]
    mask = torch.ones(2, 50, 48)
    z_shared, z_private, recon = model(x, mask)
    print(f'{enc}/{dec}: z_shared={z_shared.shape}, recon={recon.shape}')
"

# 4. Verificar training loop
python experiments/run_roseta_experiment.py \
    --data /tmp/test_constellation.npz \
    --output /tmp/test_train \
    --model constellation \
    --encoder-type mlp \
    --decoder-type histogram \
    --epochs 2 \
    --batch-size 2

# 5. Verificar evaluación
python experiments/evaluate_cross_reconstruction.py \
    --model /tmp/test_train/best_model.pt \
    --run-all-controls
```

---

## Documentación a Generar

Tras completar Fase 3A:
- `Documents/Analizador/Fase_3A_results.md` - Resultados y análisis
- `Documents/Roseta/CONSTELLATION_ARCHITECTURE.md` - Especificación técnica
- Actualizar `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

---

## Referencias SOTA

- [Shazam Original Paper (2003)](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)
- [How Shazam Works - Cameron MacLeod](https://www.cameronmacleod.com/blog/how-does-shazam-work)
- [PeakNetFP: Peak-based Neural Audio Fingerprinting (ISMIR 2025)](https://arxiv.org/abs/2506.21086)
- [Neural Audio Fingerprint with Contrastive Learning](https://ieeexplore.ieee.org/document/9414337/)
- [Pretrained Conformers for Audio Fingerprinting (2025)](https://arxiv.org/html/2508.11609)
- [Audio-JEPA (July 2025)](https://arxiv.org/abs/2507.02915)
- [VL-JEPA: Vision-Language JEPA (Dec 2025)](https://arxiv.org/abs/2512.10942)
- [SparseVLM (ICML 2025)](https://icml.cc/virtual/2025/poster/46297)

---

*Plan para Fase 3A del Revisionismo de Extracción de Ratios*
*Proyecto Phideus v5.0 - Enero 2026*
