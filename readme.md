# Phideus - Nature's Harmonic Structure Analysis Toolkit (v4.1)

**Dual Architecture Research Project** analyzing natural harmonic relationships in audio.

**Two parallel development lines**:
- **VAE Line**: Proven VAE + Linear Attention (production-ready)
- **HRM Line**: Hierarchical Reasoning Model (experimental)

Enables A/B testing while mitigating research risk.

---

## 🌱 Core Concept

Soundscapes contain meaningful frequency relationships (3:2, 5:4, √2, φ). Goal: detect and learn these patterns with neural networks trained on pure physical representations, avoiding tempered musical bias.

**🚀 NEW: Temporal Dimension Analysis** - Comprehensive technical manual completed for implementing temporal processing capabilities, enabling detection of harmonic evolution patterns over time.

---

## 🏗️ Dual Architecture Overview

### 🎵 VAE Line
- **Architecture**: VAE + Linear Attention + CNN
- **Performance**: 79.7% reconstruction
- **Parameters**: 15.3M, <1GB VRAM
- **Status**: Production-ready
- **🚀 Temporal Option**: Attention-Based Temporal VAE (recommended)

### 🧠 HRM Line  
- **Architecture**: Hierarchical Reasoning Model
- **Innovation**: Two-module recurrent architecture
- **Target**: >20% harmonic detection
- **Memory**: O(1) constant
- **Status**: Experimental
- **🚀 Temporal Option**: HRM Temporal (research phase)

## 🧰 Main Components

### Shared Components

#### 1. WAV Generator
`generador_wavs_ratios_complejos_v3.0_Ninja.py` - Generates WAVs with precise harmonic relationships (φ, 3:2, √2, microintervals).

#### 2. Analyzer  
`analizador_4.1_Enriched.py` - Multi-resolution STFT generating enriched histograms (512, 3).

- `ratio_hist_log`: log₂ domain (perceptual)
- `ratio_hist_lin`: linear domain (physical)

---

#### 3. Auditor
`auditor_v4.0.py` - Three analysis modes:
- **Harmonic**: log histogram, musical intervals
- **Topological**: linear histogram, physical metrics
- **Comparative**: side-by-side results

---

#### 4. Neural Architecture
- **VAE Training**: `train_vae_phideus.py` (15.3M parameters)
- **Architecture**: `vae_phideus_v1.py` (CNN encoder/decoder)
- **Validation**: `validate_vae_phideus.py` (latent space analysis)
- **🚀 Temporal Manual**: `Documents/Manual_Tecnico_Dimension_Temporal_Phideus.md`

Learns harmonic structures in 128D latent space from enriched histograms (512, 3).

**Temporal Capabilities**: Technical manual documents two advanced architectures for adding temporal dimension, enabling analysis of harmonic evolution patterns, call-response sequences, and temporal modulations.

---

## ⚙️ Quick usage

1️⃣ **Generate synthetic WAVs**  
```bash
python generar_wavs_ratios_complejos_v3.0.py
```

2️⃣ **Analyze frequency relationships**  
```bash
python analizador_v4.0.py --input-dir wavs_sinteticos_v3.0 --output ratios_dataset.json
```

3️⃣ **Audit (harmonic mode)**  
```bash
python auditor_v4.0.py ratios_dataset.json --analisis armonico --markdown > results_harmonic.md
```

4️⃣ **Audit (topological mode)**  
```bash
python auditor_v4.0.py ratios_dataset.json --analisis topologico --markdown > results_topological.md
```

5️⃣ **Audit (comparative mode)**  
```bash
python auditor_v4.0.py ratios_dataset.json --analisis comparativo
```

6️⃣ **Train neural model**
```bash
python src/RNA/train_vae_phideus.py
```

7️⃣ **Validate results**
```bash
python src/RNA/validate_vae_phideus.py
```

---

## 📦 Requirements

Python 3.8+  
Install dependencies:

```bash
pip install numpy scipy librosa soundfile tabulate tqdm matplotlib torch bitsandbytes
```

**GPU Support (recommended)**:
- PyTorch with CUDA support for RTX 3090 optimization
- FP16 precision and Adam8bit optimizer support
- <1GB VRAM usage for training

---

## 🎯 Goals

**Current Focus**:
- Detect non-tempered harmonic structures in natural recordings
- Learn topological representations from physical ratios
- Develop AI that recognizes resonances beyond musical conventions

**Future Vision**:
- Extend harmonic understanding across sensory modalities
- Test universal harmony hypothesis across domains
- Contribute to acoustic ecology and bioacoustics research

---

## 📚 Bibliography

- Krause, B. *Acoustic Niche Hypothesis*
- *Harmonic Information Theory* concepts
- Ecoacoustics and bioacoustics studies

---

## 🌍 Impact

- Preserve and analyze at-risk soundscapes
- Create ecologically informed AI
- Bridge science, philosophy, and sound art
- Inspire new ways of understanding natural resonances


---

🎶 *“The forest already sings. Our task is to understand its tuning.”*


[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Mar-IA-no/Phideus)

