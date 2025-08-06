# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Phideus is a Nature's Harmonic Structure Analysis Toolkit (v4.0) for analyzing, auditing, and exploring natural harmonic relationships in soundscapes and synthetic signals. The project hypothesizes that soundscapes contain meaningful frequency relationships (both rational and irrational) and aims to detect, quantify, and learn these patterns through neural networks trained on pure physical representations.

## Core Architecture

### Main Components

The system follows a pipeline architecture with four main stages:

1. **WAV Generation** (`generador_wavs_ratios_complejos_v3.0_Ninja.py`)
   - Generates synthetic WAV files with precise harmonic relationships
   - Creates "ninja circuit" combinations of harmonic, irrational, and micro-interval ratios
   - Outputs to `wavs_sinteticos_v3.0/` directory

2. **Analysis** (`analizador_v4.0.py`)
   - Multi-resolution STFT analysis of WAV files
   - Generates dual histograms: logarithmic (perceptual) and linear (physical)
   - Outputs structured JSON with frequency ratio data

3. **Auditing** (`auditor_v4.0.py`)
   - Three analysis modes: harmonic, topological, comparative
   - Harmonic mode: uses log histogram, musical interval labeling
   - Topological mode: uses linear histogram, physical metrics (entropy, flatness, Gini coefficient)
   - Comparative mode: side-by-side analysis

4. **Neural Training** (`train_ratio_model.py`)
   - CNN training for harmonic profile recognition
   - Uses linear histograms to learn from physical proportions

5. **Visualization** (`plot_ratio_histograms_v1.0.py`)
   - Generates PNG histogram plots from JSON datasets
   - Creates visual representations of frequency ratio distributions

### Data Flow

```
WAV Files → Analyzer → JSON Dataset → Auditor → Analysis Reports
                                   ↘ Neural Trainer → Trained Model
                                   ↘ Visualization → PNG Histograms
```

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requeriments.txt
# or manually:
pip install numpy scipy librosa>=0.10 soundfile tabulate tqdm matplotlib torch
```

### Primary Workflow Commands

1. **Generate synthetic WAVs**
```bash
python src/generador_wavs_ratios_complejos_v3.0_Ninja.py
```

2. **Analyze frequency relationships**
```bash
python src/analizador_v4.0.py --input-dir wavs_sinteticos_v3.0 --output ratios_dataset.json
```

3. **Audit with different modes**
```bash
# Harmonic perceptual analysis
python src/auditor_v4.0.py ratios_dataset.json --analisis armonico --markdown > results_harmonic.md

# Topological physical analysis  
python src/auditor_v4.0.py ratios_dataset.json --analisis topologico --markdown > results_topological.md

# Comparative analysis (both modes)
python src/auditor_v4.0.py ratios_dataset.json --analisis comparativo
```

4. **Train neural network**
```bash
python src/train_ratio_model.py
```

5. **Generate histogram visualizations**
```bash
python src/plot_ratio_histograms_v1.0.py
```

### Command-line Options

#### Analyzer (`analizador_v4.0.py`)
- `--input-dir`: Directory containing WAV files
- `--output`: Output JSON file path
- `--bins`: Number of histogram bins (default: 512)
- `--thr`: Peak threshold factor (default: 1.25)

#### Auditor (`auditor_v4.0.py`)
- `--analisis`: Analysis mode (armonico|topologico|comparativo)
- `--markdown`: Format output as Markdown
- `-t TOL`: Tolerance in cents for harmonic mode (default: 40.0)
- `-T UMBRAL`: Threshold for topological mode (default: 1.0)

## Key Technical Details

### File Requirements
- WAV files must be monophonic and uncompressed for accurate analysis
- All Python scripts are executable with `#!/usr/bin/env python3` shebang

### Output Formats
- Analyzer generates JSON with both `ratio_hist_log` (perceptual, log₂ scale) and `ratio_hist_lin` (physical, linear scale)
- Auditor supports both console and Markdown table output
- Default ratio range: 1.0 to 6.0 with 512 bins

### Semantic Ratios
The system recognizes standard musical intervals plus irrational ratios:
- Musical: unison (1:1), perfect fifth (3:2), octave (2:1), etc.
- Irrational: √2, √3, φ (golden ratio)
- Custom tolerance: 15-40 cents for matching

### Neural Network Architecture
- CNN designed for frequency ratio histogram analysis
- Uses linear histograms to avoid cultural musical bias
- Configurable via constants in `train_ratio_model.py:24-30`

## File Structure Notes

- All main scripts are in `src/` directory
- Generated WAVs typically go to `wavs_sinteticos_v3.0/`
- JSON datasets contain detailed metadata per audio file
- No external configuration files - parameters are CLI-based or hardcoded constants