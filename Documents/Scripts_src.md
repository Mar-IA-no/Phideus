# Scripts src/ - Phideus v4.1 Components

**Actualizado**: 2025-08-22  
**Estado**: Dual Architecture Complete - VAE + HRM Full Implementation

---

## 📁 Estructura src/ Completa

```
src/
├── analizador/           # 🎵 Audio Analysis Pipeline
├── auditor/              # 🔍 Dataset Validation & Analysis
├── generador/            # 🎹 Synthetic Audio Generation
├── RNA/                  # 🧠 VAE Neural Architecture (Production)
├── hrm/                  # 🧠 HRM Neural Architecture (Research Complete)
└── temp/                 # 🧪 Development & Testing Scripts
```

---

## 🎵 Analizador - Audio Analysis Pipeline

### `analizador_4.1_Enriched.py` ⭐ **PRINCIPAL**

**Función**: Convierte audio WAV → histogramas enriquecidos (512, 3)

**Características**:
- **Multi-resolution STFT**: Análisis spectral optimizado
- **Histogramas enriquecidos**: 3 canales (proporción, energía, entropía)
- **Resolución**: 512 bins, 6.1 cents/bin (sub-perceptual)
- **Range**: Ratios 1.0 a 6.0 (cubre all intervalos musicales)

**Usage**:
```bash
python src/analizador/analizador_4.1_Enriched.py \
    --input-dir wavs_sinteticos_v3.0 \
    --output dataset_enriched.json \
    --bins 512
```

**Output Format**:
```json
{
  "filename": "audio.wav",
  "ratio_hist_lin": [512 values],      // Linear scale (physical)
  "ratio_hist_log": [512 values],      // Log scale (perceptual)  
  "ratio_hist_entropy": [512 values],  // Entropy channel
  "metadata": {...}
}
```

### `analizador_v4.0.py`

**Función**: Versión básica 2-channel analysis

**Status**: Deprecated, usar analizador_4.1_Enriched.py

---

## 🔍 Auditor - Dataset Validation

### `auditor_v4.0.py` ⭐ **PRINCIPAL**

**Función**: Análisis y validación de datasets JSON

**Modos de análisis**:

**1. Modo Harmónico** (perceptual):
```bash
python src/auditor/auditor_v4.0.py dataset.json \
    --analisis armonico \
    --markdown > results_harmonic.md
```
- Usa `ratio_hist_log` (escala logarítmica)
- Semantic ratio matching con tolerancia cents
- Musical interval labeling

**2. Modo Topológico** (físico):
```bash
python src/auditor/auditor_v4.0.py dataset.json \
    --analisis topologico \
    --markdown > results_topological.md
```
- Usa `ratio_hist_lin` (escala linear)
- Physical metrics: entropy, spectral flatness, Gini coefficient
- Topological analysis sin bias musical

**3. Modo Comparativo**:
```bash
python src/auditor/auditor_v4.0.py dataset.json \
    --analisis comparativo
```
- Side-by-side analysis harmónico vs topológico
- Cross-validation entre approaches

**Key Parameters**:
- `-t TOL`: Tolerance in cents para harmonic mode (default: 40.0)
- `-T UMBRAL`: Threshold para topological mode (default: 1.0)
- `--markdown`: Output como Markdown tables

---

## 🎹 Generador - Synthetic Audio Generation

### `generador_wavs_ratios_complejos_v3.0_Ninja.py` ⭐ **PRINCIPAL**

**Función**: Generación WAVs sintéticos con relaciones harmónicas precisas

**Características**:
- **Harmonic Ratios**: 2:1, 3:2, 5:4, etc.
- **Irrational Ratios**: √2, √3, φ (golden ratio)
- **Microintervals**: Commas, syntonic comma, etc.
- **Ninja Circuits**: Combinaciones complejas ratios

**Usage**:
```bash
python src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py
```

**Output**:
- Directory: `wavs_sinteticos_v3.0/`
- Formats: Uncompressed WAV, monophonic
- Precision: Exact mathematical ratios

**Configuration**:
```python
# Editable parameters
SAMPLE_RATE = 44100
DURATION = 2.0  # seconds
AMPLITUDE = 0.7
BASE_FREQ = 220  # A3
```

### `generador_wavs_ratios_simples_v1.2.py`

**Función**: Generación básica ratios simples

**Status**: Available para testing, usar Ninja version para production

---

## 🧠 RNA - VAE Neural Architecture (Production)

### `vae_phideus_v1.py` ⭐ **ARQUITECTURA PRINCIPAL**

**Función**: VAE + Linear Attention architecture

**Especificaciones**:
- **Parameters**: 15.3M
- **Input**: (batch, 3, 512) histogramas enriquecidos
- **Latent**: 128D
- **Output**: (batch, 3, 512) reconstruction

**Components**:
```python
class PhideusVAE(nn.Module):
    - Encoder: CNN 1D + Linear Attention
    - Latent: μ, σ con reparametrization trick
    - Decoder: CNN Transpose con skip connections

class LinearAttention(nn.Module):
    - Stabilized attention mechanism
    - Pre/post LayerNorm + context normalization
```

### `train_vae_phideus.py` ⭐ **TRAINING PIPELINE**

**Función**: Complete training pipeline GPU-optimized

**Features**:
- **Mixed Precision**: FP16 para RTX 3090
- **Adam8bit**: Memory-efficient optimizer
- **β-VAE**: Configurable β scheduling
- **Checkpointing**: Auto-save best models

**Usage**:
```bash
python src/RNA/train_vae_phideus.py \
    --data-path dataset.json \
    --epochs 30 \
    --batch-size 16 \
    --learning-rate 1e-4
```

**Results**: Trained model en `models/vae_attention/`

### `validate_vae_phideus.py` ⭐ **VALIDATION SYSTEM**

**Función**: Comprehensive VAE validation analysis

**Analysis**:
- **PCA**: Principal component analysis latent space
- **t-SNE**: 2D visualization clustering
- **Interpolation**: Smooth transitions between samples
- **Reconstruction**: Quality metrics y visualizations

**Usage**:
```bash
python src/RNA/validate_vae_phideus.py \
    --model-path models/vae_attention/best_model.pth \
    --data-path dataset.json
```

**Output**: Validation plots y metrics en `vae_validation/`

### `linear_attention_fixed.py`

**Función**: Stabilized Linear Attention implementation

**Features**:
- **Gradient Stability**: No NaN values durante training
- **Memory Efficient**: Optimized para sequences 512-length
- **Multiple Variants**: Standard, enhanced, residual options

---

## 🧠 HRM - Hierarchical Reasoning Model (Research Complete) ✅

### Core Models (`src/hrm/models/`)

#### `h_module.py` ⭐ **HIGH-LEVEL REASONING**

**Función**: Slow timescale reasoning module

**Architecture**:
```python
class HModule(nn.Module):
    - L-output aggregator con attention weights
    - LSTM memory (128D hidden)
    - Context generator para L-Module
    - Harmonic pattern recognition
```

**Usage**:
```python
h_module = create_h_module({
    'l_output_dim': 256,
    'h_hidden_dim': 128,
    'memory_depth': 10
})
```

#### `l_module.py` ⭐ **LOW-LEVEL COMPUTATION**

**Función**: Fast timescale spectral processing

**Architecture**:
```python
class LModule(nn.Module):
    - Multi-layer GRU (256D hidden) 
    - Spectral attention mechanism
    - High temporal resolution processing
    - Compatible con histogramas (512, 3)
```

**Usage**:
```python
l_module = create_l_module({
    'input_dim': 512 * 3,
    'hidden_dim': 256,
    'h_context_dim': 128
})
```

#### `hierarchical_convergence.py` ⭐ **CORE INNOVATION**

**Función**: O(1) memory hierarchical convergence mechanism

**Innovation**:
- **O(1) Memory**: Constant complexity vs O(T) standard RNNs
- **Dual Timescales**: N high-level cycles × T low-level steps
- **Periodic Resets**: Maintains constant memory usage
- **Deep Supervision**: Multiple forward passes con gradient detachment

**Usage**:
```python
convergence = create_hierarchical_convergence({
    'input_dim': (512, 3),
    'N': 4,  # High-level cycles
    'T': 8   # Low-level steps per cycle
})
```

#### `adaptive_computation_time.py` ⭐ **Q-LEARNING ACT**

**Función**: Adaptive computation time con Q-learning

**Features**:
- **Q-Network**: Decision making halt vs continue
- **Experience Replay**: Buffer para learning stability
- **ε-greedy**: Exploration strategy
- **Harmonic Complexity**: Adapted para frequency analysis

**Usage**:
```python
act = create_act_module({
    'l_output_dim': 256,
    'max_steps': 10,
    'q_hidden_dim': 128
})
```

### Training Infrastructure (`src/hrm/training/`)

#### `train_hrm_hierarchical.py` ⭐ **COMPLETE PIPELINE**

**Función**: Full HRM training pipeline (571 lines)

**Architecture Integration**:
```python
class PhideusHRM(nn.Module):
    - Integrates all HRM components
    - H-Module + L-Module + Hierarchical Convergence + ACT
    - Compatible con histogramas (512, 3)
    - ~25M parameters total

class HRMTrainer:
    - Deep supervision training
    - O(1) memory optimization 
    - Mixed precision FP16
    - Multi-loss function optimization
```

**Loss Functions**:
```python
total_loss = (
    α₁ * reconstruction_loss +    # Primary objective
    α₂ * convergence_loss +       # Hierarchical consistency
    α₃ * act_loss +               # Adaptive computation
    α₄ * deep_supervision_loss    # Multiple layer supervision
)
```

### Validation System (`src/hrm/validation/`)

#### `validate_hrm_vs_vae.py` ⭐ **COMPREHENSIVE COMPARISON**

**Función**: Complete HRM vs VAE comparison system

**Analysis**:
- **Harmonic Accuracy**: Semantic ratio detection (15-cent tolerance)
- **Latent Consistency**: Cross-architecture space analysis
- **Spectral Fidelity**: Reconstruction quality comparison
- **Computational Efficiency**: Memory usage y inference time
- **Statistical Significance**: Performance improvement validation

**Usage**:
```bash
python src/hrm/validation/validate_hrm_vs_vae.py \
    --hrm-model models/hrm/best_hrm_model.pth \
    --vae-model models/vae_attention/best_model.pth \
    --test-data dataset.json
```

**Output**:
- **Quantitative Report**: Detailed metrics comparison
- **Qualitative Analysis**: Advantages/disadvantages each architecture
- **Statistical Tests**: Significance validation
- **Recommendations**: Data-driven architecture selection

### Production Scripts (`src/hrm/scripts/`)

#### `train_hrm_real.py` ⭐ **PRODUCTION TRAINING**

**Función**: Real dataset training script production-ready

**Features**:
- **Complete CLI**: Argument parsing con all options
- **Checkpoint Management**: Auto-save best/latest/interrupted models
- **Progress Tracking**: Training curves y real-time monitoring
- **Logging System**: File + console logging comprehensive
- **Error Recovery**: Graceful interruption handling

**Usage**:
```bash
python src/hrm/scripts/train_hrm_real.py \
    --data-path dataset.json \
    --output-dir ./hrm_output \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --l-hidden-dim 256 \
    --h-hidden-dim 128 \
    --N 4 \
    --T 8
```

**Advanced Options**:
```bash
# Custom architecture
--convergence-type adaptive \
--act-type enhanced \
--validation-split 0.2

# Resume training
--resume
```

### Examples & Documentation (`src/hrm/examples/`)

#### `demo_hrm_inference.py` ⭐ **STANDALONE DEMO**

**Función**: Complete HRM inference demonstration

**Features**:
- **Component Testing**: Individual module validation
- **Full Model Demo**: End-to-end inference
- **Sample Data Generation**: Synthetic harmonic data
- **Visualization**: Results plotting y analysis
- **Educational**: Step-by-step component explanation

**Usage**:
```bash
python src/hrm/examples/demo_hrm_inference.py
```

**Output**:
- `hrm_demo_results.png`: Visualization
- `hrm_demo_results.json`: Numerical results
- Console: Component-by-component analysis

---

## 🧪 Temp - Development & Testing

### Scripts de Testing
- `test_enriched_validation.py`: Histogram validation
- `compare_bins.py`: 256 vs 512 bins analysis
- `debug_linear_attention.py`: Attention mechanism testing
- `debug_vae_loss.py`: Loss function analysis

### Usage
```bash
# Validation testing
python src/temp/test_enriched_validation.py

# Comparative analysis
python src/temp/compare_bins.py
```

---

## 🚀 Production Workflows

### Complete VAE Pipeline
```bash
# 1. Generate synthetic data
python src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py

# 2. Analyze to histograms
python src/analizador/analizador_4.1_Enriched.py \
    --input-dir wavs_sinteticos_v3.0 \
    --output dataset.json

# 3. Audit dataset
python src/auditor/auditor_v4.0.py dataset.json \
    --analisis comparativo --markdown > audit_report.md

# 4. Train VAE
python src/RNA/train_vae_phideus.py \
    --data-path dataset.json --epochs 30

# 5. Validate results
python src/RNA/validate_vae_phideus.py \
    --model-path models/vae_attention/best_model.pth \
    --data-path dataset.json
```

### Complete HRM Pipeline ✅
```bash
# 1-3. Same preprocessing as VAE

# 4. Train HRM
python src/hrm/scripts/train_hrm_real.py \
    --data-path dataset.json \
    --epochs 100 \
    --output-dir ./hrm_output

# 5. Compare HRM vs VAE
python src/hrm/validation/validate_hrm_vs_vae.py \
    --hrm-model hrm_output/models/best_hrm_model.pth \
    --vae-model models/vae_attention/best_model.pth \
    --test-data dataset.json

# 6. Demo inference
python src/hrm/examples/demo_hrm_inference.py
```

---

## 📊 Script Maturity Status

### Production Ready ✅
- `analizador_4.1_Enriched.py` - Audio analysis pipeline
- `auditor_v4.0.py` - Dataset validation
- `generador_wavs_ratios_complejos_v3.0_Ninja.py` - WAV generation
- `vae_phideus_v1.py` - VAE architecture
- `train_vae_phideus.py` - VAE training
- `validate_vae_phideus.py` - VAE validation

### Research Ready ✅ (HRM Complete Implementation)
- `src/hrm/models/*` - All HRM core components
- `train_hrm_hierarchical.py` - Complete training pipeline
- `validate_hrm_vs_vae.py` - Comprehensive comparison
- `train_hrm_real.py` - Production training script
- `demo_hrm_inference.py` - Standalone demo

### Development/Testing
- `src/temp/*` - Development scripts
- Legacy scripts (deprecated)

---

## 🎯 Next Scripts to Implement

**All major scripts implemented** ✅

**Potential Enhancements**:
1. **Multi-GPU Training**: Distributed training scripts
2. **Real-time Inference**: Streaming audio analysis
3. **Web Interface**: Dashboard para model comparison
4. **Automated Benchmarking**: Continuous performance testing

---

## 🎵 *"Every script harmonizes towards understanding."*

**Estado**: ✅ **ALL CORE SCRIPTS IMPLEMENTED - DUAL ARCHITECTURE COMPLETE**