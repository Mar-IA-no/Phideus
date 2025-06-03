# Nature's Harmonic Structure — Extraction and Analysis Toolkit

This project provides a set of tools to **generate**, **analyze**, and **visualize** frequency ratio structures (harmonic 
relationships) found in natural soundscapes, and to **train a neural network** capable of learning emergent harmonic patterns beyond 
traditional human musical systems.

## 🧩 What does this project do?

- **Generates synthetic WAV files** with tones in specific harmonic relationships.
- **Processes real or synthetic recordings** to detect frequency peaks and calculate frequency ratios.
- **Builds normalized histograms** of detected harmonic relationships.
- **Visualizes the histograms** for quick visual analysis.
- **Audits the analysis results**, searching for key proportions (like 3:2, 5:4, φ).
- **Trains a neural network** to recognize harmonic patterns within sound recordings.

The goal is to investigate **natural harmonic structures** without relying on predefined musical assumptions.

## 📦 Requirements

- Python 3.8+
- Additional libraries:

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

> To install everything at once:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

| File                                   | Description                                                                 |
|----------------------------------------|-----------------------------------------------------------------------------|
| `generar_wavs_con_ration_v1.2.py`      | Generates simple 2-tone synthetic WAVs with harmonic ratios.               |
| `generar_wavs_con_ration_v2.0_Ninja.py`| Extended generator: builds a full "ninja" stress-test circuit with rational and irrational 
ratios. |
| `wav-ratios-json-STFT-v3.3.py`         | Analyzer: extracts frequency peaks and calculates ratios from `.wav` files.|
| `plot_ratio_histograms_v1.1.py`        | Visualizer: creates PNG histograms of harmonic distributions.              |
| `check_ratios_json_v1.5.py`            | Auditor: generates detailed tables of detected ratios.                     |
| `check_ratios_json_v2.0_Ninja.py`      | Extended auditor: produces a summary and detailed table for the "ninja" test set. |
| `train_ratio_model.py`                 | Trains a simple neural network using the harmonic histogram vectors.       |

## 🚀 Quick Start Guide

**Generate synthetic WAVs:**
```bash
python generar_wavs_con_ration_v2.0_Ninja.py
```

**Analyze WAV files:**
```bash
python wav-ratios-json-STFT-v3.3.py --input-dir ./wavs --output ratios_dataset.json
```

**Audit the analysis:**
```bash
python check_ratios_json_v2.0_Ninja.py ratios_dataset.json --markdown
```

**Visualize histograms:**
```bash
python plot_ratio_histograms_v1.1.py
```

**Train the neural network:**
```bash
python train_ratio_model.py
```

## 📚 Key Concepts

This project is based on:

- **Acoustic Niche Hypothesis** (Bernie Krause).
- **Harmonic Information Theory**: considering intervals as universal information carriers.
- Investigation of **natural harmonic ratios** (both rational and irrational), beyond the limits of Western equal temperament tuning.

