# Proyecto Phideus - Estado Actual

## Resumen Ejecutivo

**Phideus v4.0** es un sistema de análisis harmónico avanzado que detecta patrones naturales de frecuencias en audio sin bias musical. El proyecto está en **Fase 0 completada** con pipeline de análisis validado y listo para proceder a **Fase 1: Entrenamiento VAE + CNN 1D**.

---

## Arquitectura del Sistema

### Pipeline Principal (4 Componentes)

1. **🎵 Analizador v4.1 Enriched** (`src/analizador_4.1_Enriched.py`)
   - Multi-resolution STFT analysis (8192, 4096, 2048, 1024)
   - Detección adaptativa de picos espectrales
   - Generación de histogramas enriquecidos de 3 canales
   - Output: JSON con shape (512, 3) - Proporción, Energía, Entropía

2. **🔍 Auditor v4.0** (`src/auditor_v4.0.py`)
   - Análisis harmónico, topológico y comparativo
   - Procesamiento de datasets JSON del analizador
   - Output: Reportes en consola y Markdown

3. **🚀 Generador Ninja v3.0** (`src/generador_wavs_ratios_complejos_v3.0_Ninja.py`)
   - Síntesis de WAVs con ratios harmónicos precisos
   - Serie armónica, subarmónicos, microintervalos, commas
   - Ratios irracionales (phi, sqrt2, sqrt3) y ruido rosa

4. **🧠 Train Ratio Model** (`src/train_ratio_model.py`)
   - CNN para predicción de histogramas de ratios
   - CQT + Deep Learning pipeline

### Validación y Testing

- **Validador Híbrido** (`src/temp/test_enriched_validation.py`)
- **Dataset de Test** (30 WAVs sintéticos en `test_wavs/`)
- **Resultados JSON** (`test-json/`)

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

### Hardware Target: RTX 3090

- **VRAM disponible**: 24GB
- **Memoria estimada VAE**: ~7.5GB
- **Batch size recomendado**: 256 histogramas
- **Tiempo entrenamiento**: 36-40h para 100 epochs

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

### 🚧 Fase 1: VAE + CNN 1D (PRÓXIMA)
- [ ] Implementación arquitectura VAE
- [ ] Dataset preprocessing pipeline
- [ ] Training loop con FP16 + MoCo-v3
- [ ] Validación de reconstrucción
- [ ] Análisis de espacio latente

### 🔮 Fase 2: ASI-ARCH Integration (FUTURO)
- [ ] Sistema de arquitectura autónoma
- [ ] Neural Architecture Search
- [ ] Híbridos VAE + Mamba/Perceiver
- [ ] Optimización automática

---

## Estructura del Repositorio

```
Phideus/
├── src/                          # Scripts principales del pipeline
│   ├── analizador_4.1_Enriched.py    # Análisis principal
│   ├── auditor_v4.0.py               # Auditoría de datasets
│   ├── generador_..._Ninja.py        # Generación de test WAVs
│   ├── train_ratio_model.py          # Entrenamiento CNN
│   └── temp/                         # Scripts temporales/testing
├── Documents/                    # Documentación del proyecto
│   ├── bitacora_desarrollo.md        # Log detallado de desarrollo
│   └── Proyecto_Estado_Actual.md     # Este documento
├── Biblioteca/                   # Research papers y propuestas
├── test-json/                    # Datasets de validación
├── test_wavs/                    # Audios sintéticos de test
└── validation_plots/             # Visualizaciones de validación
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

## Próximos Pasos Inmediatos

1. **Implementar arquitectura VAE** según especificaciones de la hoja de ruta
2. **Preparar dataset grande** para entrenamiento (objetivo: 10M histogramas)  
3. **Configurar pipeline de entrenamiento** con checkpointing y validación
4. **Monitorear métricas** de reconstrucción y clustering latente
5. **Documentar resultados** y actualizar este documento

---

*Documento actualizado: 2025-08-06*  
*Estado: Pipeline Fase 0 completado, listo para Fase 1*