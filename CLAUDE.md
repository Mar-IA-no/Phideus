# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Phideus is a Nature's Harmonic Structure Analysis Toolkit (v4.1) for analyzing natural harmonic relationships in soundscapes through neural networks. Based on revolutionary three-architecture comparison findings, the project uses Enhanced HRM (Hierarchical Reasoning Model) which dramatically outperforms VAE approaches by 153,500%.

## Core Architecture

### Neural Models (Priority Order)

1. **Enhanced HRM** - WINNER ⭐⭐⭐⭐⭐
   - `src/hrm/training/train_hrm_hierarchical.py`
   - Dual-timescale hierarchical processing (L-Module + H-Module)
   - Multi-head attention for temporal relationships
   - 6M parameters, validation loss: 2.74
   - **153,500% better than VAE variants**

2. **Enhanced VAE** - Limited improvement
   - `src/RNA/train_vae_phideus.py`
   - VAE with Linear Attention
   - 1.64M parameters, validation loss: 4212.58
   - Only 0.000% better than Baseline VAE

3. **Baseline VAE** - Reference only
   - `experiments/train_vae_base.py`
   - Standard CNN without enhancements
   - 1.19M parameters, validation loss: 4212.58

### Pipeline Components

1. **WAV Generation** (`src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py`)
2. **Analysis** (`src/analizador/analizador_v4.0.py`) 
3. **Auditing** (`src/auditor/auditor_v4.0.py`)
4. **Enhanced HRM Training** (`src/hrm/training/train_hrm_hierarchical.py`)
5. **Architecture Comparison** (`experiments/compare_three_architectures.py`)

## Development Commands

### Primary Workflow (Updated)

1. **Train Enhanced HRM (Recommended)**
```bash
python src/hrm/training/train_hrm_hierarchical.py
```

2. **Compare All Three Architectures**
```bash
python experiments/compare_three_architectures.py
```

3. **Generate Training Data**
```bash
python src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py
```

4. **Analyze Audio**
```bash
python src/analizador/analizador_v4.0.py --input-dir wavs_sinteticos_v3.0 --output ratios_dataset.json
```

## Repository Directory Organization Standards

### **MANDATORY Structure (ALWAYS Respect)**

```
/root/Phideus/
├── src/                           # Production source code
│   ├── hrm/                       # Enhanced HRM (PRIMARY MODEL)
│   │   ├── training/              # HRM training scripts
│   │   ├── models/                # HRM model definitions
│   │   ├── validation/            # HRM validation tools
│   │   └── scripts/               # HRM utility scripts
│   ├── RNA/                       # VAE models (secondary)
│   ├── analizador/                # Audio analysis tools
│   ├── auditor/                   # Auditing and reporting
│   ├── generador/                 # WAV generation
│   └── temp/                      # Temporary/experimental scripts
├── experiments/                   # Research experiments
│   ├── temporal/                  # Temporal VAE experiments
│   ├── benchmarks/               # Performance benchmarks
│   └── compare_three_architectures.py
├── data/                         # Training outputs & datasets
│   ├── datasets/                 # JSON datasets
│   └── training_outputs/         # Model outputs organized by type
├── models/                       # Saved models by architecture
├── test/                         # Test data (gitignored)
├── Documents/                    # Project documentation
└── config/                       # Configuration files
```

### **File Organization Rules**

**🚫 NEVER commit these files**:
- Audio files (WAVs, MP3) - too large
- Large datasets (>1MB JSON)
- Virtual environments (`*_env/`)
- Python cache (`__pycache__/`)
- Training datasets in `train/`

**✅ Core principles**:
- **Enhanced HRM priority** - Primary model in `src/hrm/`
- **Experiments separate** - Research code in `experiments/`
- **Clean temp** - Move experimental scripts to `src/temp/`
- **Organized outputs** - Results in `data/training_outputs/[model_type]/`

### **Documentation Update Protocol**

When user requests "actualizar documentos", automatically update:
1. `README.md` - Project overview with latest findings
2. `Documents/bitacora_desarrollo.md` - Development log entry
3. `Documents/Proyecto_Estado_Actual.md` - Current project status
4. All other .md files in `Documents/` folder

### **Development Workflow**

**Priority Order for Neural Models**:
1. **Enhanced HRM** - Use for all new harmonic analysis work
2. Enhanced VAE - Legacy support only
3. Baseline VAE - Reference comparison only

**TodoWrite Usage**:
- ALWAYS use for multi-step tasks
- Mark completed tasks immediately
- Track reorganization and documentation updates