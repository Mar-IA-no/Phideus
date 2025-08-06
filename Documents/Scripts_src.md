# Scripts src/ - Documentación Técnica

## Resumen General

El directorio `/src/` contiene la implementación completa del pipeline de análisis armónico de Phideus, **reorganizado por componentes funcionales**:

```
src/
├── analizador/          # 🎵 Análisis de audio → histogramas armónicos
├── auditor/             # 🔍 Validación y verificación de resultados  
├── generador/           # 🎹 Generación sintética de ratios y WAVs
├── RNA/                 # 🧠 Redes neuronales y modelos de ML
└── temp/                # 🧪 Scripts temporales y testing
```

---

## 1. Análisis y Generación de Audio

### `analizador/analizador_v4.0.py`
**Propósito**: Analizador base de ratios armónicos con STFT multi-resolución.

**Funcionalidades principales**:
- **Multi-resolution STFT**: Análisis con múltiples resoluciones temporales
- **Peak detection**: Detección de picos armónicos con umbrales adaptativos
- **Ratio extraction**: Cálculo de ratios de frecuencia entre picos detectados
- **Histogram generation**: Histogramas logarítmicos de ratios (256 bins por defecto)

**Parámetros clave**:
```python
DEFAULT_N_RATIO_BINS = 256
THRESHOLD_DB = -40
MIN_PEAK_DISTANCE = 5
```

**Output**: Histogramas de ratios normalizados para análisis posterior.

---

### `analizador/analizador_4.1_Enriched.py` 
**Propósito**: **PRINCIPAL** - Versión mejorada del analizador con histogramas enriquecidos de 3 canales.

**Mejoras sobre v4.0**:
- **512 bins**: Resolución aumentada para mayor precisión (6.1 cents/bin)
- **3-channel histograms**: 
  - Canal 0: Proporción normalizada (PDF)
  - Canal 1: Energía ponderada (segundo momento en escala log₂)
  - Canal 2: Entropía local (Shannon entropy)
- **Corrección de energía**: Escala log₂ consistente para todos los canales

**Arquitectura de salida**:
```python
ratio_hist_enriched: ndarray = (512, 3)
# Dimensión 0: bins de ratios logarítmicos
# Dimensión 1: [proporción, energía, entropía]
```

**Casos de uso**:
- Entrada principal para VAE Phideus
- Análisis de alta resolución de patrones armónicos
- Preparación de datasets para machine learning

---

### `auditor/auditor_v4.0.py`  
**Propósito**: Sistema de auditoría y validación de análisis armónicos.

**Funciones principales**:
- **Validation metrics**: Comparación entre análisis esperado vs obtenido
- **Ratio verification**: Verificación de ratios detectados contra ground truth
- **Statistical analysis**: Métricas de precisión, recall y F1-score
- **Report generation**: Informes detallados de calidad del análisis

**Métricas implementadas**:
- Precision: `TP / (TP + FP)`
- Sensitivity (Recall): `TP / (TP + FN)`  
- F1-Score: `2 * (precision * recall) / (precision + recall)`

**Output**: Reportes JSON con métricas detalladas de calidad.

---

### `generador/generador_wavs_ratios_complejos_v3.0_Ninja.py`
**Propósito**: **PRINCIPAL** - Generador avanzado de señales de audio con ratios armónicos específicos.

**Capacidades**:
- **Multi-harmonic synthesis**: Generación de hasta 20 armónicos simultáneos
- **Complex ratio patterns**: Ratios no-enteros y microtonales
- **Amplitude envelopes**: ADSR y envolventes personalizadas
- **Noise injection**: Ruido controlado para realismo
- **Batch generation**: Generación masiva de archivos de entrenamiento

**Configuración típica**:
```python
ratios = [1.0, 1.25, 1.5, 1.618, 2.0, 2.5]  # Golden ratio + armónicos
duration = 5.0  # segundos
sample_rate = 44100
harmonics = 12  # número de armónicos
```

**Output**: Archivos WAV con metadatos JSON de ratios esperados.

---

## 2. RNA - Red Neuronal Artificial

### `RNA/vae_phideus_v1.py`
**Propósito**: **PRINCIPAL** - Implementación completa del Variational Autoencoder con Linear Attention estabilizada.

**Arquitectura principal**:

```python
class PhideusVAE(nn.Module):
    - Input: (batch, 3, 512) - Histogramas enriquecidos
    - Encoder: CNN 1D dilatada con 6 bloques
    - Latent: 128D (μ, σ) con reparametrization trick
    - Decoder: CNN Transpose simétrica
    - Output: (batch, 3, 512) - Reconstrucción
```

**Componentes clave**:

**DilatedCNNBlock**:
- Conv1D con dilataciones exponenciales [1, 2, 4, 8, 16, 32]
- BatchNorm + GELU activation + Dropout(0.1)
- Conexiones residuales para preservar información
- MaxPooling cada 2 bloques para reducción dimensional

**LinearAttention (ESTABILIZADA)**:
- Pre/post LayerNorm + Xavier initialization + temperature scaling
- Context normalization para prevenir value explosion  
- ReLU + epsilon kernel (más estable que ELU+1)
- 4 heads, manejo eficiente de secuencias largas (512 bins)

**Latent Space**:
- 128 dimensiones (compresión 12:1 desde 1536D)
- Distribución objetivo: Gaussiana estándar N(0,I)
- Reparametrization trick: `z = μ + ε*σ`

**Loss Function**:
```python
def vae_loss(recon, target, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, target)  # MSE
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss  # β-VAE
```

---

### `RNA/train_vae_phideus.py`
**Propósito**: Pipeline completo de entrenamiento del VAE con optimizaciones para RTX 3090.

**Optimizaciones implementadas**:

**Mixed Precision (FP16)**:
```python
from torch.cuda.amp import GradScaler, autocast
with autocast():
    output = model(batch)
    loss = vae_loss(output['reconstruction'], batch, ...)
```

**Adam8bit Optimizer**:
- 75% menos VRAM para optimizer states
- Cuantización dinámica 32-bit ↔ 8-bit
- Performance equivalente a Adam estándar

**Gradient Accumulation**:
- Simula batches más grandes con VRAM limitada
- `accumulation_steps=4` → batch efectivo 4x mayor
- Reduce varianza de gradientes

**β-VAE Scheduling**:
- `constant`: β=1.0 (VAE estándar)
- `linear`: Ramp up en primera mitad del entrenamiento
- `cyclical`: Alternancia periódica para evitar colapso

**Learning Rate Scheduling**:
- Cosine annealing decay
- LR inicial: 1e-3, LR mínimo: 1e-6
- Warm-up implícito en β scheduling

**Checkpointing**:
- Guardado automático cada 5 epochs
- Mejor modelo según validation loss
- Mantiene solo últimos 5 checkpoints
- Estado completo: modelo, optimizer, scheduler, scaler

---

### `RNA/validate_vae_phideus.py`
**Propósito**: Sistema completo de validación y análisis del VAE entrenado.

**Métricas de Reconstrucción**:
```python
mse_per_sample = np.mean((originals - reconstructions) ** 2, axis=(1, 2))
correlation = np.corrcoef(orig_flat, recon_flat)[0, 1]
reconstruction_quality = 1.0 / (1.0 + np.mean(mse_per_sample))  # [0,1]
```

**Análisis del Espacio Latente**:

**PCA Analysis**:
- Explained variance ratio de primeras 10 componentes
- Dimensionalidad efectiva (95% de varianza)
- Detección de distribución gaussiana

**t-SNE Visualization**:
- Proyección 2D para datasets <500 muestras
- Clustering visual de muestras similares
- Verificación de continuidad del espacio latente

**K-Means Clustering**:
- Agrupación automática de embeddings
- Número de clusters: `min(5, len(samples) // 3)`
- Análisis de separabilidad de diferentes tipos de audio

**Interpolación Latente**:
```python
def interpolate_latent(z1, z2, n_steps=8):
    alphas = np.linspace(0, 1, n_steps)
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        reconstruction = model.decode(z_interp)
```

**Visualizaciones generadas**:
1. `latent_space_analysis.png`: 6 subplots de análisis completo
2. `reconstructions.png`: Comparación original vs reconstrucción
3. `latent_interpolation.png`: Interpolación suave entre muestras
4. `validation_report.json`: Métricas cuantitativas completas

---

## 3. Utilidades de Entrenamiento

### `RNA/train_ratio_model.py`
**Propósito**: Script de entrenamiento para modelos de clasificación/regresión de ratios (legacy).

**Características**:
- Entrenamiento de modelos clásicos (SVM, Random Forest, etc.)
- Cross-validation y hyperparameter tuning
- Métricas de clasificación para ratios discretos
- Export de modelos entrenados para inferencia

**Estado**: Mantenido para compatibilidad con versiones anteriores.

---

## Pipeline de Uso Recomendado

### 1. Generación de Datos
```bash
# Generar dataset de entrenamiento
python src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py --output train/VAE/ --count 100

# Procesar con analizador enriquecido
python src/analizador/analizador_4.1_Enriched.py --input train/VAE/ --output train_data.json
```

### 2. Entrenamiento VAE
```bash
# Entrenar VAE con configuración óptima
python src/RNA/train_vae_phideus.py \
    --data train_data.json \
    --epochs 100 \
    --batch-size 32 \
    --beta-schedule linear \
    --use-attention \
    --accumulation-steps 4
```

### 3. Validación y Análisis
```bash
# Validar modelo entrenado
python src/RNA/validate_vae_phideus.py \
    --checkpoint RNA/vae_checkpoints/best_model.pth \
    --data validation_data.json \
    --save-dir RNA/vae_validation/
```

### 4. Auditoría de Calidad
```bash
# Verificar calidad del pipeline completo
python src/auditor/auditor_v4.0.py --test-data validation_data.json
```

---

## Consideraciones Técnicas

### Dependencias Críticas
```python
torch >= 1.12.0        # PyTorch con CUDA support
bitsandbytes >= 0.35.0  # Adam8bit optimizer
scikit-learn >= 1.0.0  # PCA, t-SNE, clustering
seaborn >= 0.11.0      # Visualizaciones
librosa >= 0.9.0       # Procesamiento audio
```

### Hardware Recomendado
- **GPU**: RTX 3090 (24GB) o superior
- **RAM**: 32GB+ para datasets grandes
- **Storage**: SSD para I/O rápido de histogramas

### Limitaciones Conocidas
1. **VAE compression**: Pérdida inevitable en compresión 12:1
2. **Peak detection**: Sensible a ruido en señales de baja calidad  
3. **Memory usage**: Crece linealmente con batch size
4. **Training time**: ~40h para 100 epochs en RTX 3090

---

*Documentación técnica - Scripts src/*  
*Actualizada: 2025-08-06*