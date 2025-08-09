# Nature’s Harmonic Structure – Analysis Toolkit (v4.0)

This repository contains Python tools to **analyze, audit, and explore natural harmonic relationships** in soundscapes and synthetic 
signals. Version 4.0 integrates perceptual (harmonic) analysis, physical/topological analysis, and a unified comparative mode.

---

## 🌱 Core concept

The project starts from the hypothesis that soundscapes (natural or synthetic) are structured by meaningful frequency relationships — 
both rational and irrational (e.g., 3:2, 5:4, √2, φ). The goal is to detect, quantify, and eventually **learn** these patterns with 
neural networks trained on pure physical representations, without imposing tempered musical logic.

---

## 🧰 Main components

### 1. 🎹 **Synthetic WAV generator**

`generar_wavs_ratios_complejos_v3.0.py`  
Generates `.wav` files with precise combinations of pure sine waves and pink noise. Includes a "ninja circuit" of harmonic, 
irrational, and micro-interval ratios, ready for testing and training.

---

### 2. 📈 **Multi-scale analyzer (v4.0)**

`analizador_v4.0.py`  
Analyzes WAV files using multi-resolution STFT, detects peaks, and computes **all valid frequency relationships**.  
Now generates two complementary histograms:

- `ratio_hist_log`: log₂ domain (perceptual/cents scale).
- `ratio_hist_lin`: pure linear domain (physical relationships).

---

### 3. 🔎 **Hybrid auditor (v4.0)**

`auditor_v4.0.py`  
Allows three modes of analysis:

- **Harmonic:** uses the logarithmic histogram, labels with musical intervals, and shows readable tables.
- **Topological:** uses the linear histogram, computes physical metrics (entropy, flatness, Gini coefficient, centroid, 
autocorrelation).
- **Comparative:** combines both modes and displays side-by-side results.

---

### 4. 🧠 **VAE con Linear Attention** (Arquitectura Principal)

**Sistema Neural Único** (`src/RNA/`)  
- **Entrenamiento**: `train_vae_phideus.py` - VAE con Linear Attention estabilizada (15.3M parámetros)
- **Arquitectura**: `vae_phideus_v1.py` - Implementación completa con encoder/decoder CNN dilatado
- **Validación**: `validate_vae_phideus.py` - Análisis completo del espacio latente y visualizaciones

**Enfoque Audio-Only Consolidado**: Aprende estructuras harmónicas en espacio latente 128D desde histogramas enriquecidos (512, 3), evitando sesgo musical cultural a través de relaciones físicas puras.

**Estado**: Fase 1 completada - Linear Attention estabilizada, preparación multimodal diferida hasta base audio sólida (500+ samples, >85% reconstruction quality).

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

6️⃣ **Train VAE model** (Arquitectura principal)
```bash
python src/RNA/train_vae_phideus.py
```

7️⃣ **Validate VAE results** (Análisis espacio latente)
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

## 🎯 Scientific and philosophical goals

**Current Focus (Audio-Only Consolidation)**:
- Detect **non-tempered harmonic structures** in natural recordings through VAE latent space
- Learn topological representations from physical ratios, beyond cultural musical conventions  
- Develop AI "listener" that recognizes resonances and proportions before notes or chords
- **Phase 1.1 Priority**: Expand dataset (78→500+ samples), optimize architecture, validate semantic structure

**Future Vision (Multimodal - Post Audio Mastery)**:
- Extend harmonic understanding across sensory modalities (vision, physiological signals)
- Test "universal harmony" hypothesis: do φ, 3:2, 5:4 ratios correspond across audio↔spatial domains?
- Contribute to **acoustic ecology, bioacoustics, cross-modal perception research**

---

## 📚 Bibliography and context

Based on:

- Krause, B. *Acoustic Niche Hypothesis*
- Concept of *Harmonic Information Theory*
- Studies in ecoacoustics, bioacoustics, and deep learning models.
- Documents: `Harmonic Information Theory.docx`, `Bibliografía_Relacionada.md`, and conceptual proposals (EOI).

---

## 🌍 Potential impact

- Preserve and analyze at-risk soundscapes.
- Create ecologically informed AI.
- Contribute to interdisciplinary studies bridging science, philosophy, and sound art.
- Inspire new ways of listening and understanding natural resonances.


---

🎶 *“The forest already sings. Our task is to understand its tuning.”*


[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Mar-IA-no/Phideus)

