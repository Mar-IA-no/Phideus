# Nature’s Harmonic Structure – Analysis Toolkit

This repository hosts a set of Python tools designed to extract, audit, visualize, and learn from **natural harmonic relationships** 
found in soundscapes. It provides a full workflow from synthetic data generation to neural network training, enabling exploration of 
**natural harmonic ratios** beyond conventional music theory.

## 🌱 Conceptual Framework

This toolkit is part of a broader research initiative grounded in the **Acoustic Niche Hypothesis** (Krause, 1993) and the emerging 
field of **Harmonic Information Theory**. We posit that natural soundscapes are structured by meaningful frequency ratios—both 
rational and irrational (e.g., 3:2, 5:4, √2, φ)—which may encode communicative, aesthetic, or ecological information. Our goal is to 
detect, quantify, and learn from these patterns.

---

## 🧰 Project Components

### 1. 🔊 **Synthetic WAV Generator**
`generador_wavs_ratios_complejos_v2.0_Ninja.py`  
Generates a stress-test dataset ("circuito ninja") of synthetic `.wav` files containing sine waves in harmonic or irrational 
frequency ratios. Includes pink noise injection and extreme test cases.

### 2. 📈 **Analyzer (STFT-based)**
`analizador_v3.3.py`  
Processes WAV files with multi-resolution STFT to extract dominant peaks and compute all valid frequency ratios. Outputs normalized 
histograms (`ratio_hist`) in log₂ space, saved in a `.json` dataset.

### 3. 🧮 **Auditor**
`auditor_v3.1.py`  
Reads the JSON output and evaluates key metrics (entropy, flatness, dominant ratio). Can output a summary table in Markdown or CSV 
format and print human-readable insights per file.

### 4. 🖼️ **Histogram Visualizer**
`plot_ratio_histograms_v1.1.py`  
Generates PNG bar plots for each entry in the dataset, showing the harmonic profile of each sound.

### 5. 🧠 **Neural Network Trainer**
`train_ratio_model.py`  
Trains a simple CNN using the spectrograms of the WAVs and their corresponding harmonic histograms (`ratio_hist`) as targets. The 
goal is to learn the "harmonic profile" directly from audio.

---

## ⚙️ Quickstart

1. **Generate synthetic WAVs**
```bash
python generador_wavs_ratios_complejos_v2.0_Ninja.py
```

2. **Analyze harmonic content**
```bash
python analizador_v3.3.py --input-dir synthetic_wavs --output ratios_dataset.json
```

3. **Audit analysis**
```bash
python auditor_v3.1.py ratios_dataset.json --markdown > results.md
```

4. **Visualize histograms**
```bash
python plot_ratio_histograms_v1.1.py
```

5. **Train the neural network**
```bash
python train_ratio_model.py
```

---

## 📦 Requirements

Python 3.8+  
Install dependencies with:

```bash
pip install numpy scipy librosa soundfile tabulate tqdm matplotlib torch
```

Or create a `requirements.txt` with:

```
numpy
scipy
librosa>=0.10
soundfile
tabulate
tqdm
matplotlib
torch
```

---

## 🧠 Scientific and Humanistic Goals

- Detect and interpret **non-tempered** harmonic structures in ecological audio.
- Develop **AI models** that learn patterns of harmonic resonance beyond human cultural norms.
- Contribute to fields like **acoustic ecology**, **multilingual AI**, and **sound-based cognition**.
- Support the vision of AI as a listener: capable of **resonant, contextual understanding**.

---

## 📚 References and Background

This toolkit is informed by foundational works including:

- Bernie Krause’s *Acoustic Niche Hypothesis*
- The concept of *Harmonic Information Theory* (see `Harmonic Information Theory.docx`)
- Research in **bioacoustics, ecoacoustics**, and **deep learning for sound analysis**
- Theoretical framing in the EOI: *Nature's Harmonic Structure – An Interdisciplinary Exploration...*

See the file `Bibliografía_Relacionada.md` for detailed citations.

---

## 🌍 Broader Impact

This work aligns with efforts to:

- Preserve and analyze endangered soundscapes
- Train **ecologically informed AI**
- Explore **resonance as a principle of meaning across biology, cognition, and language**
- Contribute tools and datasets to interdisciplinary fields combining **AI + Humanities**

---

## 🗺️ Future Directions

- Expand from synthetic to **field recordings**
- Test models on real biophonies
- Explore generative harmonic systems (autoencoders, VAE)
- Apply to other domains (neural waves, EMG, motion patterns)

---

🎶 *“The forest is already singing. Our task is to understand its tuning.”*

