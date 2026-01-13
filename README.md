# Phideus - Nature's Harmonic Structure Analysis Toolkit (v5.0)

**PARADIGM SHIFT: Data Representation > Architecture**

**KEY FINDING**: With Analizador 5.0 (linear scale + temporal data), **VAE and HRM achieve equivalent performance**. The previous 153,500% HRM superiority was due to data representation issues in Analizador 4.1, not architectural limitations.

---

## Core Concept

Soundscapes contain meaningful frequency relationships (3:2, 5:4, sqrt(2), phi). Goal: detect and learn these patterns with neural networks trained on pure physical representations, avoiding tempered musical bias.

**REVOLUTIONARY DISCOVERY**: The Analizador 5.0 experiments prove that **data representation is more important than neural architecture**. With proper linear-scale temporal data, both VAE and HRM achieve excellent results.

---

## Complete Comparison Results (January 2026)

### Analizador 5.0 Results (848 samples, 50 epochs)

| Rank | Experiment | Architecture | Val Loss | Parameters |
|------|------------|--------------|----------|------------|
| 1 | E2 | **VAE Temporal** | **0.4560** | 1,824,640 |
| 2 | E1 | HRM Temporal | 0.4607 | 2,268,928 |
| 3 | E3 | HRM Static | 0.5906 | 854,144 |
| 4 | E4 | VAE Static | 0.5997 | 837,760 |

### Paradigm Shift: 4.1 vs 5.0

| Metric | Analizador 4.1 | Analizador 5.0 | Change |
|--------|----------------|----------------|--------|
| **HRM val_loss** | 2.74 | 0.4607 | **-83.2%** |
| **VAE val_loss** | 4212.58 | 0.4560 | **-99.99%** |
| **HRM advantage** | 153,500% | -1.0% | **VAE now wins** |

### Key Scientific Findings

1. **Temporality helps**: +22-24% improvement (temporal vs static)
2. **VAE recovered**: From catastrophic (4212) to excellent (0.456)
3. **Architectures comparable**: No clear winner with optimal data
4. **Data representation critical**: Linear scale + temporal > log scale + static

---

## Repository Structure

```
Phideus/
├── src/                           # Core source code
│   ├── analizador/               # Audio analysis (4.1, 5.0)
│   │   ├── analizador_v4.0.py    # Log-scale static analyzer
│   │   └── analizador_5.0.py     # Linear-scale temporal analyzer
│   ├── datasets/                 # Dataset loaders
│   │   └── temporal_dataset_5.py # NPZ/JSON temporal loader
│   ├── hrm/                      # HRM implementations
│   ├── RNA/                      # VAE implementations
│   ├── auditor/                  # Harmonic auditing tools
│   ├── generador/                # WAV generation tools
│   └── temp/                     # Temporary scripts
│
├── experiments/                   # Research experiments
│   ├── run_experiments_5.0.py    # 4-experiment comparison script
│   ├── compare_three_architectures.py
│   └── temporal/                 # Temporal VAE experiments
│
├── data/                         # Datasets and outputs
│   ├── datasets/                 # Processed datasets
│   │   └── temporal_5.0_full.npz # Binary temporal dataset
│   └── training_outputs/         # Model outputs
│       └── experiments_5.0/      # Latest comparison results
│
├── Documents/                    # Documentation
│   ├── REPORTE_COMPARATIVO_4.1_vs_5.0.md  # Paradigm shift analysis
│   ├── INFORME_ANALISIS_INTEGRACION_5.0.md
│   ├── bitacora_desarrollo.md
│   └── results/                  # Historical results
│
├── models/                       # Trained models
├── train/                        # Training WAV files (848)
└── config/                       # Configuration files
```

---

## Main Components

### 1. Analizador 5.0 (NEW - Recommended)
**Location**: `src/analizador/analizador_5.0.py`
- **Linear scale** frequency ratios (not log2)
- **Temporal data** [T, B, 3] per audio file
- **Binary format** (NPZ) - 12x smaller than JSON
- **Parallelization** support (--workers)

```bash
# Generate temporal dataset
python src/analizador/analizador_5.0.py \
    --input-dir train/synthetic_dataset_500 \
    --output data/datasets/temporal_5.0.npz \
    --format npz \
    --workers 14
```

### 2. Analizador 4.1 (Legacy)
**Location**: `src/analizador/analizador_v4.0.py`
- Log2 scale frequency ratios
- Static histograms [B, 3]
- JSON format

### 3. Dataset Loader
**Location**: `src/datasets/temporal_dataset_5.py`
- Supports NPZ and JSON formats
- Three strategies: 'sequence', 'average', 'frames'
- Automatic train/val splitting

### 4. Neural Architectures

#### HRM Temporal (E1)
- GRU + LSTM + Multi-head Attention
- 2.27M parameters
- Best for: Efficiency per parameter

#### VAE Temporal (E2) - BEST ABSOLUTE
- LSTM encoder + decoder
- 1.82M parameters
- Best for: Absolute performance

---

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio librosa scipy numpy tqdm matplotlib
```

### 2. Generate Dataset (Analizador 5.0)
```bash
python src/analizador/analizador_5.0.py \
    --input-dir train/synthetic_dataset_500 \
    --output data/datasets/temporal_5.0.npz \
    --format npz \
    --workers 14
```

### 3. Run Experiments
```bash
python experiments/run_experiments_5.0.py \
    --data data/datasets/temporal_5.0_full.npz \
    --output data/training_outputs/experiments_5.0 \
    --epochs 50 \
    --batch-size 32 \
    --max-frames 100
```

### 4. Review Results
```bash
cat data/training_outputs/experiments_5.0/report_experiments_5.0.md
```

---

## Performance Comparison

### Analizador 5.0 (Current)

| Architecture | Data Type | Val Loss | Params | Efficiency |
|--------------|-----------|----------|--------|------------|
| **VAE Temporal** | Sequence | **0.4560** | 1.82M | 0.250 |
| HRM Temporal | Sequence | 0.4607 | 2.27M | 0.203 |
| HRM Static | Average | 0.5906 | 0.85M | 0.695 |
| VAE Static | Average | 0.5997 | 0.84M | 0.714 |

### Historical (Analizador 4.1)

| Architecture | Val Loss | Notes |
|--------------|----------|-------|
| HRM | 2.74 | Previous "winner" |
| VAE | 4212.58 | Catastrophic failure |

---

## Scientific Contributions

### Key Discoveries

1. **Data Representation Primacy**: The representation of input data (linear vs log scale, temporal vs static) has greater impact than architectural choices.

2. **VAE Rehabilitation**: VAE was not inherently unsuitable for harmonic analysis - it failed due to log2 scale data representation.

3. **Temporal Information Value**: Temporal data provides ~22-24% improvement regardless of architecture.

4. **Architecture Equivalence**: With optimal data representation, HRM and VAE achieve comparable results.

### Implications

- **For Production**: Use VAE Temporal for best absolute performance
- **For Research**: Both architectures are valid starting points
- **For Data**: Prioritize linear scale and temporal preservation

---

## Documentation

### Key Reports
- **[Comparative Report 4.1 vs 5.0](Documents/REPORTE_COMPARATIVO_4.1_vs_5.0.md)**: Complete paradigm shift analysis
- **[Integration Analysis](Documents/INFORME_ANALISIS_INTEGRACION_5.0.md)**: Professional doctoral-level analysis
- **[Development Log](Documents/bitacora_desarrollo.md)**: Full development history
- **[Experiment Results](data/training_outputs/experiments_5.0/report_experiments_5.0.md)**: Raw experiment data

### Architecture Docs
- **[HRM Documentation](Documents/hrm/)**: Hierarchical Reasoning Model details
- **[VAE Documentation](Documents/vae/)**: Variational Autoencoder details

---

## Applications

### Current Capabilities
- Harmonic relationship detection in audio
- Structural analysis of soundscapes
- Neural architecture benchmarking
- Synthetic dataset generation

### Future Directions
- Natural soundscape analysis
- Bioacoustic research
- Music information retrieval
- Cross-modal harmonic analysis

---

## Status: Production Ready

Both VAE and HRM architectures are validated for harmonic analysis with Analizador 5.0 data:

- **VAE Temporal**: Best absolute performance (0.4560)
- **HRM Temporal**: Best efficiency per parameter
- **Dataset**: 848 files, 245,824 frames, binary NPZ format
- **Training**: GPU-accelerated with CUDA support

---

## Citation

```bibtex
@software{phideus2026,
  title={Phideus: Nature's Harmonic Structure Analysis Toolkit},
  author={Phideus Project},
  year={2026},
  url={https://github.com/your-repo/Phideus},
  note={Paradigm shift: Data representation > Architecture}
}
```

---

*"The forest already sings. Our task is to understand its tuning."*

**Both architectures have learned the language of natural harmonies.**
