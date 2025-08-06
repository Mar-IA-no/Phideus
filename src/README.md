# Phideus src/ - Estructura Organizada

## 📁 **Organización por Componentes**

```
src/
├── analizador/          # 🎵 Análisis de audio → histogramas armónicos
├── auditor/             # 🔍 Validación y verificación de resultados
├── generador/           # 🎹 Generación sintética de ratios y WAVs
├── RNA/                 # 🧠 Redes neuronales y modelos de ML
└── temp/                # 🧪 Scripts temporales y testing
```

### **analizador/** - Audio Analysis Pipeline
- `analizador_4.1_Enriched.py` - **PRINCIPAL**: Análisis enriquecido 512 bins + 3 canales
- `analizador_v4.0.py` - Versión anterior (512 bins, 1 canal)

### **auditor/** - Validation & Quality Control  
- `auditor_v4.0.py` - Validación híbrida WAVs reales vs sintéticos

### **generador/** - Synthetic Audio Generation
- `generador_wavs_ratios_complejos_v3.0_Ninja.py` - **PRINCIPAL**: Generación ratios complejos
- `generador_wavs_ratios_simples_v1.2.py` - Generación ratios simples (temp)

### **RNA/** - Neural Networks & ML Models
- `vae_phideus_v1.py` - **PRINCIPAL**: VAE + CNN 1D + Linear Attention
- `train_vae_phideus.py` - Training pipeline VAE
- `validate_vae_phideus.py` - Validación y visualización VAE
- `train_ratio_model.py` - Entrenamiento modelos ratio prediction
- `vae_checkpoints/` - Checkpoints del VAE entrenado
- `vae_validation/` - Resultados de validación y visualizaciones

### **temp/** - Development & Testing
Scripts temporales de desarrollo, debugging y experimentación.

## 🔗 **Pipeline de Uso**

1. **Generador** → Crear WAVs sintéticos de ratios específicos
2. **Analizador** → Procesar WAVs → histogramas enriquecidos  
3. **Auditor** → Validar calidad del análisis
4. **RNA** → Entrenar/usar modelos neurales con datos procesados

## 📊 **Archivos Principales por Fase**

- **Fase 0** (Baseline): `analizador_4.1_Enriched.py` + `auditor_v4.0.py`
- **Fase 1** (VAE): `RNA/vae_phideus_v1.py` + training/validation scripts
- **Dataset Generation**: `generador_wavs_ratios_complejos_v3.0_Ninja.py`