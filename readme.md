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

### 4. 🧠 **Neural network training**

`train_ratio_model.py`  
Trains a neural network to recognize **harmonic profiles** directly from linear histograms, aiming for the model to learn from 
physical proportions without cultural bias.

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

6️⃣ **Train neural network (optional)**  
```bash
python train_ratio_model.py
```

---

## 📦 Requirements

Python 3.8+  
Install dependencies:

```bash
pip install numpy scipy librosa soundfile tabulate tqdm matplotlib torch
```

---

## 🎯 Scientific and philosophical goals

- Detect **non-tempered harmonic structures** in natural recordings.
- Learn topological representations that emerge from physical ratios, beyond human hearing conventions.
- Explore AI as a “listener” that recognizes resonances and proportions before notes or chords.
- Contribute to research in **acoustic ecology, bioacoustics, and sound cognition**.

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

