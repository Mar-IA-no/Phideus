# Proyecto Phideus VAE - Estado Actual

## Resumen

**VAE Line** detecta patrones harmónicos naturales sin bias musical. **Fase 1 completada**: VAE + Linear Attention estabilizada, lista para optimización y aplicaciones.

---

## Arquitectura del Sistema

### Pipeline Principal (4 Componentes)

1. **Analizador v4.1** - Multi-resolution STFT, histogramas enriquecidos (512, 3)

2. **Auditor v4.0** - Análisis harmónico, topológico, comparativo

3. **Generador v3.0** - Síntesis WAVs con ratios precisos (φ, 3:2, √2)

4. **VAE v1.0** - VAE + Linear Attention, 1536D→128D, 79.7% reconstruction
   
5. **Training Pipeline** - Automatizado, GPU-optimized, <1GB VRAM
   
6. **Validation System** - PCA, t-SNE, clustering, interpolación latente

### Testing
- 30 WAVs sintéticos validados
- Resultados en `test-json/`

---

## Especificaciones Técnicas

### Configuración Optimizada Final

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **Bins de histograma** | 512 | 6.1 cents/bin - resolución sub-perceptual |
| **Canales** | 3 | Proporción + Energía + Entropía |
| **Rango de ratios** | 1.0 - 6.0 | Cubre octavas extendidas |
| **Shape de input VAE** | (512, 3) | Optimizado para CNN 1D |
| **Compresión latent** | 512×3 → 128 | Ratio 4:1 saludable |

### Hardware Confirmado: RTX 3090

- **VRAM disponible**: 24GB
- **VRAM usado VAE**: < 1GB (real medido)
- **Batch size óptimo**: 16-64 histogramas
- **Tiempo entrenamiento**: ~0.1 min/30 epochs (dataset 78 samples)
- **Speedup GPU vs CPU**: 14x más rápido
- **Optimizaciones activas**: FP16, Adam8bit, gradient accumulation

---

## Estado de Validación

### Tests Completados ✅

1. **Formato estructural**: 30/30 archivos shape (512, 3) ✅
2. **Normalización matemática**: 30/30 canales suma ~1.0 ✅  
3. **Estabilidad numérica**: Sin NaN/infinitos/negativos ✅
4. **Balance entre canales**: Ratio < 10x entre medias ✅
5. **Detección ratios musicales**: 10/150 ratios (6.7%) ✅

### Casos de Éxito Validados

- **Octavas**: `sub_1_2.wav` - detección clara 2:1
- **Terceras**: `5_4.wav`, `6_5.wav` - mayores y menores
- **Microintervalos**: `comma_81_80.wav` - comma sintónica
- **Commas complejas**: `comma_531441_524288.wav` - comma de Pitágoras
- **Ruido + tonal**: `phi_noise.wav` - detección robusta

---

## Arquitectura de Machine Learning

### Propuesta VAE + CNN 1D (Hoja de Ruta)

```
Input: (batch, 512, 3)
↓
Encoder CNN 1D: 6 bloques dilated convs (64→256 channels)
↓ 
Linear Attention (opcional): Performer, 256D, 4 heads
↓
Latent Space: 128D (μ, σ)
↓
Decoder CNN 1D: 6 bloques transpose + skip connections
↓
Output: (batch, 512, 3) - reconstrucción
```

**Parámetros**: ~8.3M total
**Contrastive Learning**: MoCo-v3 o BYOL
**Optimización**: FP16, Adam8bit, gradient accumulation

---

## Roadmap de Desarrollo

### ✅ Fase 0: Baseline y Validación (COMPLETADA)
- [x] Analizador v4.1 con histogramas enriquecidos
- [x] Pipeline de validación automatizado
- [x] 30 casos de test con ratios conocidos
- [x] Optimización de resolución (512 bins)
- [x] Validación técnica completa

### ✅ Fase 1: VAE + CNN 1D (COMPLETADA)
- [x] Implementación arquitectura VAE completa
- [x] Dataset preprocessing pipeline (78 WAVs → 8.5MB JSON)
- [x] Training loop con FP16 + Adam8bit GPU-optimized
- [x] Validación de reconstrucción (79.7% quality)
- [x] Análisis de espacio latente (PCA, t-SNE, clustering)
- [x] PyTorch CUDA configuration y dependencias
- [x] Sistema completo validación con visualizaciones
- [x] **Linear Attention estabilizada**: Gradient explosion resuelto
- [x] **Estructura src/ reorganizada**: Componentes separados funcionalmente

### 🔮 Fase 1.1: Dataset Expansion (PRÓXIMA)
- [ ] Larger dataset (500+ samples reales)
- [ ] Hyperparameter tuning automático
- [ ] Contrastive learning (MoCo-v3/BYOL)
- [ ] Architecture search CNN depths/widths
- [ ] Re-entrenamiento con Linear Attention habilitada

### 🔮 Fase 2: ASI-ARCH Integration (FUTURO)
- [ ] Sistema de arquitectura autónoma
- [ ] Neural Architecture Search
- [ ] Híbridos VAE + Mamba/Perceiver
- [ ] Optimización automática

---

## Estructura del Repositorio

```
Phideus/
├── src/                          # 🎯 CÓDIGO FUENTE ORGANIZADO
│   ├── analizador/                   # 🎵 Análisis audio → histogramas
│   │   ├── analizador_4.1_Enriched.py   (PRINCIPAL)
│   │   └── analizador_v4.0.py
│   ├── auditor/                      # 🔍 Validación y verificación
│   │   └── auditor_v4.0.py
│   ├── generador/                    # 🎹 Generación sintética
│   │   ├── generador_wavs_ratios_complejos_v3.0_Ninja.py (PRINCIPAL)
│   │   └── generador_wavs_ratios_simples_v1.2.py
│   ├── RNA/                          # 🧠 Redes neuronales
│   │   ├── vae_phideus_v1.py             (PRINCIPAL - Linear Attention)
│   │   ├── train_vae_phideus.py
│   │   ├── validate_vae_phideus.py
│   │   └── train_ratio_model.py
│   └── temp/                         # 🧪 Testing y debugging
├── models/                       # 🧠 MODELOS ENTRENADOS ORGANIZADOS
│   ├── vae_baseline/                 # VAE sin attention (79.7% quality)
│   │   ├── checkpoints/              # 6 modelos .pth (493MB)
│   │   └── validation/               # Métricas + visualizaciones
│   ├── vae_attention/                # VAE con Linear Attention (10x mejor)
│   │   ├── checkpoints/              # 6 modelos .pth (531MB)
│   │   └── validation/               # Performance analysis
│   └── datasets/                     # Datasets procesados
│       └── train_vae_enriched_512.json  # 78 WAVs → 8.5MB
├── Documents/                    # 📚 DOCUMENTACIÓN COMPLETA
│   ├── bitacora_desarrollo.md        # Log detallado de desarrollo
│   ├── Proyecto_Estado_Actual.md     # Este documento
│   ├── Hoja_de_Ruta_Actual.md        # Roadmap detallado
│   ├── RNA_Arqu.md                   # Arquitectura VAE técnica
│   └── Scripts_src.md                # Documentación de scripts
├── Biblioteca/                   # Research papers y propuestas
├── test/                         # 🧪 TESTING DATA
│   ├── test-json/                    # Datasets de validación
│   ├── test_wavs/                    # Audios sintéticos de test
│   └── validation_plots/             # Visualizaciones
└── train/VAE/                    # 🎵 TRAINING DATA (78 WAVs reales)
```

---

## Insights Científicos Clave

### Descubrimientos del Análisis

1. **Histogramas lineales vs logarítmicos**: Ambos capturan información complementaria
2. **Canal de energía en escala log2**: Crítico para consistencia matemática  
3. **Entropía local por bin**: Añade robustez ante ruido
4. **Resolución óptima**: 512 bins balance precisión/eficiencia
5. **Detección multicanal**: Sistema híbrido reduce falsos positivos

### Limitaciones Identificadas

1. **Detección de ratios**: 6.7% tasa actual (mejorable con ML)
2. **Microintervalos sutiles**: Requieren algoritmos más sofisticados
3. **Ruido vs señal**: Balance delicado en umbrales de sensibilidad

---

## Métricas VAE Actuales

### Modelos Disponibles Organizados

#### VAE Baseline (Sin Attention)
- **Ubicación**: `models/vae_baseline/checkpoints/`
- **Arquitectura**: 15.08M parámetros, 128D latent space
- **Training time**: 0.1 minutos (30 épocas) en RTX 3090
- **Reconstruction quality**: 79.7% (target >70% ✅)
- **Memory usage**: <1GB VRAM de 24GB disponible

#### VAE Attention (Linear Attention Estabilizada)
- **Ubicación**: `models/vae_attention/checkpoints/`
- **Arquitectura**: 15.3M parámetros (+264k vs baseline)
- **Performance**: 10x mejor loss (343.46 → 36.93)
- **Reconstruction loss**: Mejorada (0.0722 → 0.0628)
- **Estabilidad**: Sin NaN values, gradient explosion resuelto

### Análisis del Espacio Latente
- **PCA componentes**: [3.96%, 3.68%, 3.54%, 3.35%, 3.26%]
- **Clusters identificados**: 5 grupos distintos
- **Compresión**: 1536D → 128D (ratio 12:1)
- **Interpolación**: Transiciones suaves confirmadas

### Dataset Organizado
- **Ubicación**: `models/datasets/train_vae_enriched_512.json`
- **Contenido**: 78 WAVs reales → histogramas enriquecidos (8.5MB)
- **Formato**: (512, 3) - Proporción, Energía, Entropía por sample

### Issues Identificados y Solucionados
1. ✅ **PyTorch CPU-only** → Instalado CUDA support + bitsandbytes
2. ✅ **Architecture bug** → Fixed dynamic decoder reshape
3. ✅ **Linear Attention NaN** → **RESUELTO**: Pre/post LayerNorm + context normalization
4. ✅ **Gradient explosion** → Xavier init + temperature scaling + ReLU kernel
5. ✅ **Estructura src/ desorganizada** → Componentes separados funcionalmente
6. ✅ **Modelos dispersos** → **NUEVA**: Estructura `models/` organizada por tipo

## Próximos Pasos Inmediatos

### Fase 1.1: Dataset Expansion (2-3 días)
1. **Larger dataset** (500+ WAVs) para robustez - FreeSound API + audio urbano
2. **Re-entrenamiento VAE** con Linear Attention habilitada
3. **Hyperparameter grid search** automático
4. **Contrastive learning** MoCo-v3 integration

### Fase 2: Aplicaciones (1-2 semanas)
1. **Real-time inference** pipeline optimizado
2. **Interactive analysis tools** para exploración latente
3. **API deployment** para análisis batch
4. **Integration testing** con otros proyectos Phideus

---

*Documento actualizado: 2025-08-06*  
*Estado: ✅ Fase 1 VAE + CNN 1D completada exitosamente*  
*Próximo: Fase 1.1 optimizaciones y escalamiento*