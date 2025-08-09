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

4. **Neural Training** (`src/RNA/train_vae_phideus.py`)
   - VAE with Linear Attention training
   - Uses linear histograms for physical harmonic representation learning
   - 15.3M parameters, 128D latent space, GPU-optimized

5. **Validation & Analysis** (`src/RNA/validate_vae_phideus.py`)
   - Comprehensive VAE validation system
   - PCA, t-SNE, clustering analysis, latent space interpolation
   - Visual reconstruction quality assessment

### Data Flow

```
WAV Files → Analyzer → JSON Dataset → Auditor → Analysis Reports
                                   ↘ VAE Training → VAE Model + Latent Space
                                   ↘ VAE Validation → Reconstruction Analysis
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

4. **Train VAE model**
```bash
python src/RNA/train_vae_phideus.py
```

5. **Validate VAE results**
```bash
python src/RNA/validate_vae_phideus.py
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

### VAE Architecture
- VAE with Linear Attention designed for harmonic structure analysis
- Uses linear histograms to avoid cultural musical bias  
- 15.3M parameters, 128D latent space, CNN encoder/decoder with dilated convolutions
- GPU-optimized with FP16 precision and Adam8bit optimizer
- Stable training without NaN values in Linear Attention mechanism

## File Structure Notes

- **Core scripts**: `src/analizador/`, `src/auditor/`, `src/generador/`, `src/RNA/`
- **VAE components**: All neural architecture in `src/RNA/` subdirectory
- **Models**: Trained models and validation in `models/` directory structure
- **Generated data**: JSON datasets contain enriched histograms (512, 3) format
- **Test data**: `test/` directory with validation WAVs and analysis plots

### File Organization Rules

**🚫 NEVER commit multimedia files to GitHub**:
- **Audio files (WAVs, MP3)** → Local only, never commit
- **Large JSONs (>1MB)** → Local only, never commit  
- **Training datasets** → `train/` directory (gitignored)
- **Testing data** → `test/` directory (gitignored)
- **Exception**: Small config/example files < 1MB

**📁 Dataset organization**:
- **Training WAVs** → `train/[model]/[subset]/` (e.g., `train/VAE/real_audio/`)
- **Test WAVs** → `test/test_wavs/`
- **Test JSONs** → `test/test-json/`

**🗂️ Script placement**:
- **Production scripts** → `src/` (organized by function)
- **Temporary/testing scripts** → `src/temp/`
- **Documentation** → `Documents/`

## Documentation Update Commands

### "actualizar documentos" Command
When the user requests "actualizar documentos" (update documents), Claude should **automatically update these 5 documents**:

1. **`Documents/bitacora_desarrollo.md`** - Add new entry with date and recent changes
2. **`Documents/Proyecto_Estado_Actual.md`** - Update completed phases, current metrics, next steps
3. **`Documents/RNA_Arqu.md`** - Verify architectural specifications match current code
4. **`Documents/Scripts_src.md`** - Synchronize with actual scripts in src/ directory
5. **`Documents/Hoja_de_Ruta_Actual.md`** - Update roadmap, completed milestones, objectives
6. **`README.md`** - Update with current project status and features

**Process**: Review each document for consistency with codebase, update sections that have changed, and report what documents were updated and main changes made.

## Development Workflow

### For Technical Changes:
1. **Implement** change in corresponding file
2. **Test** functionality if applicable  
3. **Update bitácora** with technical details
4. **Update project overview** if significant change
5. **Organize** new files in correct structure
6. **Use TodoWrite** systematically for multi-step tasks

### For New Features:
1. **Create todos** for planning
2. **Document** in bitácora before starting
3. **Implement** in organized manner
4. **Validate** with tests if applicable
5. **Update** final documentation
6. **Mark todos complete** in real-time

### For Research/Analysis:
1. **Document** findings in bitácora
2. **Move** analysis scripts to `src/temp/`
3. **Preserve** important results in `test-json/`
4. **Integrate** conclusions into project overview