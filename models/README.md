# Phideus Models Directory - Dual Architecture

This directory contains trained models organized by architecture line.

## 🗂️ **Nueva Estructura Dual**

```
models/
├── datasets/                    # Shared datasets
│   └── train_vae_enriched_512.json  # Main training dataset (78 samples)
├── vae/                        # VAE Current Line models
│   ├── baseline/               # VAE without Linear Attention
│   ├── attention/              # VAE with Linear Attention (current)
│   └── contrastive/            # VAE with contrastive learning (future)
└── hrm/                        # HRM Research Line models  
    ├── core/                   # Basic HRM implementation
    ├── act/                    # HRM with Adaptive Computation Time
    └── harmonic/               # Phideus-optimized HRM
```

---

## 🧠 **Modelos Disponibles**

### **VAE Baseline (Sin Attention)**
- **Ubicación**: `vae_baseline/checkpoints/`
- **Arquitectura**: CNN 1D dilatada + 128D latent space
- **Performance**: 79.7% reconstruction quality
- **Parámetros**: 15.08M
- **Estado**: ✅ Estable, production-ready

**Archivos principales**:
- `best_model.pth` - Mejor modelo según validation loss
- `config.json` - Configuración de entrenamiento
- `training_curves.png` - Evolución de loss

### **VAE Attention (Con Linear Attention)**
- **Ubicación**: `vae_attention/checkpoints/`  
- **Arquitectura**: CNN 1D dilatada + Linear Attention + 128D latent
- **Performance**: 10x mejor loss (343→36 vs baseline)
- **Parámetros**: 15.3M (+264k vs baseline)
- **Estado**: ✅ Estabilizada después de gradient explosion fix

**Mejoras implementadas**:
- Pre/post LayerNorm para gradient flow controlado
- Xavier initialization de proyecciones  
- ReLU + epsilon kernel (más estable que ELU+1)
- Context normalization previene value explosion

---

## 📊 **Datasets**

### **train_vae_enriched_512.json** (8.5MB)
- **Ubicación**: `datasets/`
- **Contenido**: 78 WAVs reales → histogramas enriquecidos
- **Format**: JSON con shape (512, 3) por sample
- **Canales**: [Proporción, Energía, Entropía]
- **Fuente**: Audio urbano y musical de FreeSound

**Estructura del dataset**:
```json
{
  "audio_file.wav": {
    "ratio_hist_enriched": [[...512 bins...], [...], [...]],  // (512, 3)
    "metadata": {
      "duration": 5.0,
      "sample_rate": 44100,
      "peaks_detected": 12
    }
  }
}
```

---

## 📈 **Análisis y Validación**

### **Métricas Disponibles**

#### VAE Baseline:
- **MSE mean**: 0.254426
- **Reconstruction quality**: 79.7%
- **Latent space PCA**: [3.96%, 3.68%, 3.54%...]
- **Clusters detectados**: 5 grupos

#### VAE Attention:
- **Total loss**: 36.93 (vs 343.46 baseline)
- **Reconstruction loss**: 0.0628 (vs 0.0722 baseline) 
- **Memory usage**: 429MB peak (RTX 3090 compatible)
- **Training stability**: Sin NaN values

### **Visualizaciones**

Cada modelo incluye en su directorio `validation/`:
- `latent_space_analysis.png` - t-SNE, PCA, clustering
- `reconstructions.png` - Original vs reconstrucción
- `latent_interpolation.png` - Interpolaciones suaves
- `validation_report.json` - Métricas cuantitativas

---

## 🔧 **Uso de los Modelos**

### **Cargar Modelo Entrenado**

```python
import torch
from src.RNA.vae_phideus_v1 import PhideusVAE

# Cargar VAE con attention
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PhideusVAE(use_attention=True).to(device)
checkpoint = torch.load('models/vae_attention/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    output = model(histograms_batch)
    reconstructed = output['reconstruction']
    latent_codes = output['mu']  # Para clustering/similarity
```

### **Cargar Dataset**

```python
import json
from src.RNA.vae_phideus_v1 import PhideusDataset

# Cargar dataset procesado
dataset = PhideusDataset('models/datasets/train_vae_enriched_512.json')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
```

---

## 💾 **Consideraciones de Storage**

### **Tamaños de Archivos** (Ignorados por .gitignore)

- **VAE Baseline checkpoints**: ~493MB total
- **VAE Attention checkpoints**: ~531MB total  
- **Validation plots**: ~2MB por modelo
- **Dataset JSON**: 8.5MB

**Total**: ~1GB+ de modelos entrenados (solo en local)

### **Backup y Versionado**

Los modelos están **excluidos de GitHub** por su tamaño. Para backup:

1. **Local backup**: Copiar directorio `models/` completo
2. **External storage**: Subir a Google Drive/AWS S3 para compartir
3. **Model registry**: Considerar MLflow/Weights&Biases para versioning

---

## 🚀 **Próximos Modelos**

### **Roadmap de Modelos Planeados**

- **Fase 1.2**: `vae_contrastive/` - MoCo-v3 integration
- **Fase 2.1**: `vae_hybrid/` - VAE + Mamba/Perceiver  
- **Fase 3**: `production/` - Modelos optimizados para deployment

### **Datasets Futuros**

- **Dataset 500+**: Expansión a 500+ samples (Fase 1.1)
- **Validation set**: Dataset separado para testing
- **Synthetic augmented**: Generación automática de variaciones

---

*Organización implementada: 2025-08-06*  
*Modelos ready para production con Linear Attention estabilizada*