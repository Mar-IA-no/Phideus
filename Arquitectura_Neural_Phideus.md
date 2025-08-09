# Arquitectura Neural Phideus v4.1

**Variational Autoencoder con Linear Attention para Análisis de Estructuras Harmónicas**

---

## 🎯 Resumen Ejecutivo

Phideus implementa un **VAE (Variational Autoencoder) con Linear Attention** diseñado específicamente para aprender estructuras harmónicas a partir de histogramas de proporciones de frecuencia. La arquitectura actual es **audio-only** con preparaciones para futura expansión multimodal.

### Principio Fundamental
```
WAV Audio → Análisis STFT → Histograma Proporciones (512, 3) → VAE → Espacio Latente 128D → Conocimiento Harmónico
```

---

## 🏗️ Arquitectura Actual (v4.1)

### Input: Histogramas Enriquecidos
```python
Shape: (batch_size, 512, 3)
Channels:
├─ Canal 0: Ratio (proporción fundamental f₂/f₁)
├─ Canal 1: Energía (magnitud espectral log-scale)  
└─ Canal 2: Entropía (distribución información armónica)

Resolución: 6.1 cents/bin (detección microintervalos)
Rango: 1.0 - 6.0 (unison → quinta+octava)
```

### Encoder: CNN 1D Dilatada + Linear Attention

```python
# Arquitectura Encoder Simplificada
class PhideusEncoder(nn.Module):
    # CNN Dilatada - 6 bloques con residual connections
    # Linear Attention estabilizada - sin NaN values
    # Latent projection → mu, logvar (128D cada uno)
    
    Conv1D Blocks: [3→64→128→256→256→256→256]
    Dilations: [1, 2, 4, 8, 16, 32] 
    Receptive Field: ~2000 bins
    Linear Attention: 8 heads, temperature scaling
    Output: mu(128D), logvar(128D)
```

### Decoder: CNN Transpose Simétrica

```python
# Arquitectura Decoder
class PhideusDecoder(nn.Module):
    # Projection: 128D → feature maps
    # Transpose CNN simétrica al encoder
    # Skip connections para preservar detalles
    
    Input: z(128D) 
    ConvTranspose: [256→256→256→256→128→64→3]
    Output: Reconstructed histograms (512, 3)
```

### Linear Attention: Estabilizada

```python
# Características clave Linear Attention
Complexity: O(N) vs O(N²) tradicional
Stability: Pre/post LayerNorm + residual connections
Kernel: ReLU + epsilon (no ELU inestable)
Temperature: Learnable scaling parameter
Context norm: Previene value explosion
```

---

## 🧠 Espacio Latente: 128 Dimensiones

### Representación Aprendida
```python
Latent Vector z ∈ ℝ¹²⁸
├─ Dimensiones 1-40: Estructura harmónica fundamental
├─ Dimensiones 41-80: Características tímbricas/energéticas  
├─ Dimensiones 81-120: Patrones microtonales complejos
└─ Dimensiones 121-128: Variabilidad residual
```

### Capacidades Demostradas
- **Reconstrucción**: 79.7% quality (target: >80%)
- **Interpolación**: Transiciones suaves entre estructuras harmónicas
- **Clustering**: 5 clusters claros (octavas, quintas, terceras)
- **Generación**: Sampling coherente desde prior gaussiano

---

## ⚙️ Optimizaciones RTX 3090

### Training Configuration
```python
Precision: FP16 (mixed precision)
Optimizer: Adam8bit (bitsandbytes) 
Memory: <1GB VRAM (de 24GB disponible)
Batch size: 16 (expandible a 32)
Learning rate: 2e-4 con cosine annealing
Beta VAE: 1.0 (ajustable 4-10)
```

### Performance Metrics
```python
Parameters: 15.3M (encoder: 8.2M, decoder: 7.1M)
Inference: <100ms histogram → embedding
Training: ~0.1 min/epoch (78 samples)
Memory footprint: 847MB PyTorch model
```

---

## 🔬 Validación Actual

### Latent Space Analysis
```python
PCA variance: [3.96%, 3.68%, 3.54%, 3.35%, 3.26%]  # Equilibrado
Clusters: 5 (separación clara por tipo harmónico)
Reconstruction MSE: 0.254 (bajo error)
KL divergence: Balanceada (no posterior collapse)
```

### Tests Implementados
- **Harmonic intervals**: ¿Octavas, quintas separadas?
- **Microtonal detection**: ¿Commas, ratios irracionales?
- **Interpolation quality**: ¿Transiciones musicalmente sensatas?
- **Semantic structure**: ¿z contiene armonía vs patrones?

---

## 🌐 Preparación Multimodal (Futuro)

### Arquitectura Preparatoria (No Activa)
```python
# Modificaciones ready para Fase 2.0+
latent_dim: 128 → 160  # 128 core + 32 preparación
partitioned_latent: z_shared(128) + z_audio(32) + z_img(32)
domain_tokens: Embedding preparado para diferentes sensores
replay_buffer: Sistema incremental learning ready
```

### Criterios de Activación Multimodal
```python
multimodal_ready = (
    dataset_size >= 500 AND
    reconstruction_quality > 0.85 AND  
    latent_semantic_structure_validated AND
    interpolation_musically_coherent AND
    training_pipeline_optimized
)
```

---

## 📊 Pipeline de Datos

### Flujo Actual
```
1. WAV Files → Generador sintético
2. STFT Analysis → Analizador v4.1 Enriched  
3. Histogramas (512,3) → Dataset JSON
4. VAE Training → Modelo .pth
5. Validation → PCA, clustering, interpolación
```

### Comandos Principales
```bash
# Generar WAVs sintéticos
python src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py

# Crear histogramas enriquecidos
python src/analizador/analizador_4.1_Enriched.py --input-dir wavs/ --output dataset.json

# Entrenar VAE
python src/RNA/train_vae_phideus.py

# Validar modelo  
python src/RNA/validate_vae_phideus.py
```

---

## 🎯 Estado y Roadmap

### ✅ Completado (Fase 1)
- VAE con Linear Attention estabilizada (15.3M params)
- Training pipeline GPU-optimized
- Validation system completo
- Base dataset: 78 WAVs, 79.7% reconstruction

### 🚀 Próximo (Fase 1.1 - 2-3 meses)
1. **Dataset Expansion**: 78 → 500+ samples (PRIORIDAD #1)
2. **Architecture Optimization**: Hyperparameters, memory, performance  
3. **Validation Rigurosa**: Semantic structure crítica
4. **Multimodal Preparation**: Código ready (sin implementar)

### 🌐 Futuro Condicional (Fase 2.0+)
- **Multimodal MVP**: Solo si audio base >85% quality
- **Cross-modal Test**: ¿φ, 3:2, 5:4 → patrones espaciales?
- **Full System**: Pipeline incremental v2 completo

---

## 📚 Especificaciones Técnicas

### Frameworks & Tools
- **PyTorch**: 2.0+ con CUDA support
- **Optimizations**: bitsandbytes (Adam8bit), FP16 mixed precision
- **Hardware**: RTX 3090 optimized
- **Storage**: NPZ histogramas + PyTorch checkpoints

### Inspiración Arquitectural  
- **VAE**: Kingma & Welling (2013)
- **Linear Attention**: Katharopoulos et al. (2020)
- **Dilated Conv**: WaveNet (DeepMind)
- **β-VAE**: Higgins et al. (2017)

---

*Phideus Neural Architecture v4.1*  
*Audio-only consolidation phase*  
*Multimodal preparation ready*  
*Next: Dataset 500+ + semantic validation*