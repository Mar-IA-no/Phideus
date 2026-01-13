# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Phideus is a Nature's Harmonic Structure Analysis Toolkit (v5.0) for analyzing natural harmonic relationships in soundscapes through neural networks.

**PARADIGM SHIFT (January 2026)**: With Analizador 5.0 (linear scale + temporal data), VAE and HRM achieve equivalent performance. The previous HRM superiority was due to data representation issues, not architectural limitations.

## Core Architecture

### Neural Models (Current Status)

Both architectures are now viable with Analizador 5.0 data:

1. **VAE Temporal** - BEST ABSOLUTE PERFORMANCE
   - `experiments/run_experiments_5.0.py` (VAETemporal class)
   - LSTM encoder + decoder for sequences
   - 1.82M parameters, validation loss: **0.4560**
   - Best for: Absolute performance

2. **HRM Temporal** - BEST EFFICIENCY
   - `experiments/run_experiments_5.0.py` (HRMTemporal class)
   - GRU + LSTM + Multi-head attention
   - 2.27M parameters, validation loss: **0.4607**
   - Best for: Efficiency per parameter

3. **Static Variants** - For comparison
   - HRM Static: 0.5906 val_loss
   - VAE Static: 0.5997 val_loss

### Key Finding

| Analyzer | HRM val_loss | VAE val_loss | Winner |
|----------|--------------|--------------|--------|
| 4.1 (log scale) | 2.74 | 4212.58 | HRM (153,500%) |
| **5.0 (linear)** | **0.4607** | **0.4560** | **VAE (-1%)** |

**Data representation matters more than architecture.**

### Pipeline Components

1. **WAV Generation** (`src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py`)
2. **Analysis 5.0** (`src/analizador/analizador_5.0.py`) - **RECOMMENDED**
3. **Analysis 4.1** (`src/analizador/analizador_v4.0.py`) - Legacy
4. **Dataset Loader** (`src/datasets/temporal_dataset_5.py`)
5. **Experiments** (`experiments/run_experiments_5.0.py`)
6. **Auditing** (`src/auditor/auditor_v4.0.py`)

## Development Commands

### Primary Workflow (v5.0)

1. **Generate Temporal Dataset (Analizador 5.0)**
```bash
source venv/bin/activate
python src/analizador/analizador_5.0.py \
    --input-dir train/synthetic_dataset_500 \
    --output data/datasets/temporal_5.0.npz \
    --format npz \
    --workers 14
```

2. **Run 4-Experiment Comparison**
```bash
python experiments/run_experiments_5.0.py \
    --data data/datasets/temporal_5.0_full.npz \
    --output data/training_outputs/experiments_5.0 \
    --epochs 50 \
    --batch-size 32 \
    --max-frames 100
```

3. **Generate Training WAVs (if needed)**
```bash
python src/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py
```

### Legacy Workflow (v4.1)

```bash
python src/analizador/analizador_v4.0.py --input-dir wavs_dir --output dataset.json
python experiments/compare_three_architectures.py
```

## Repository Directory Organization Standards

### **MANDATORY Structure (ALWAYS Respect)**

```
/root/Phideus/
├── src/                           # Production source code
│   ├── analizador/                # Audio analysis tools
│   │   ├── analizador_5.0.py      # NEW: Linear scale + temporal
│   │   └── analizador_v4.0.py     # Legacy: Log scale + static
│   ├── datasets/                  # Dataset loaders
│   │   └── temporal_dataset_5.py  # NPZ/JSON loader
│   ├── hrm/                       # HRM implementations
│   ├── RNA/                       # VAE implementations
│   ├── auditor/                   # Auditing and reporting
│   ├── generador/                 # WAV generation
│   └── temp/                      # Temporary/experimental scripts
├── experiments/                   # Research experiments
│   ├── run_experiments_5.0.py     # NEW: 4-experiment comparison
│   ├── compare_three_architectures.py  # Legacy comparison
│   └── temporal/                  # Temporal experiments
├── data/                         # Training outputs & datasets
│   ├── datasets/                 # Processed datasets
│   │   └── temporal_5.0_full.npz # Binary temporal dataset
│   └── training_outputs/         # Model outputs organized by type
│       └── experiments_5.0/      # Latest results
├── models/                       # Saved models by architecture
├── train/                        # Training WAV files (848)
├── Documents/                    # Project documentation
└── config/                       # Configuration files
```

### **File Organization Rules**

**NEVER commit these files**:
- Audio files (WAVs, MP3) - too large
- Large datasets (>1MB JSON, NPZ files)
- Virtual environments (`venv/`, `*_env/`)
- Python cache (`__pycache__/`)

**Core principles**:
- **Analizador 5.0 priority** - Use for all new work
- **Both architectures valid** - VAE and HRM are equivalent
- **Experiments separate** - Research code in `experiments/`
- **Organized outputs** - Results in `data/training_outputs/`

### **Documentation Update Protocol**

When user requests "actualizar documentos", automatically update:
1. `README.md` - Project overview with latest findings
2. `Documents/bitacora_desarrollo.md` - Development log entry
3. `Documents/Proyecto_Estado_Actual.md` - Current project status
4. `Documents/REPORTE_COMPARATIVO_4.1_vs_5.0.md` - If experiments changed
5. All other relevant .md files in `Documents/` folder

### **Development Workflow**

**Priority Order for Data Analysis**:
1. **Analizador 5.0** - Linear scale + temporal (NPZ format)
2. Analizador 4.1 - Legacy support only (JSON format)

**Priority Order for Neural Models**:
1. **VAE Temporal** - Best absolute performance
2. **HRM Temporal** - Best efficiency per parameter
3. Static variants - For comparison only

**TodoWrite Usage**:
- ALWAYS use for multi-step tasks
- Mark completed tasks immediately
- Track reorganization and documentation updates

## Key Scientific Findings

1. **Data representation > Architecture**: Linear scale + temporal data enables both VAE and HRM to perform well
2. **VAE rehabilitation**: VAE was not inherently bad - it failed due to log2 scale data
3. **Temporal helps both**: ~22-24% improvement from temporal vs static data
4. **No clear winner**: With optimal data, architectures are equivalent
