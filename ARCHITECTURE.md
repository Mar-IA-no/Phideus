# Phideus Dual Architecture Documentation

## Overview

Phideus v4.1 has evolved into a **dual architecture research project** with two parallel development lines:

1. **VAE Current Line**: Consolidation and optimization of existing VAE + Linear Attention approach
2. **HRM Research Line**: Novel Hierarchical Reasoning Model for breakthrough harmonic analysis

This architecture enables rigorous A/B testing while mitigating development risk.

## Architecture Lines

### 🎵 VAE Current Line (Consolidation)

**Focus**: Optimize and scale existing proven approach

**Architecture**: VAE + Linear Attention + CNN
- **Parameters**: 15.3M, **Latent**: 128D
- **Performance**: 79.7% reconstruction
- **Status**: ✅ Production-ready

**Roadmap**: Dataset expansion, contrastive learning, optimization

### 🧠 HRM Research Line (Innovation)

**Focus**: Revolutionary hierarchical reasoning for harmonic search

**Architecture**: Hierarchical Reasoning Model
- **H-Module**: Abstract planning (slow)
- **L-Module**: Fast computation (reset)
- **ACT**: Adaptive Computation Time
- **Memory**: O(1) vs O(T) traditional

**Target**: >20% harmonic detection, ~27M parameters
**Status**: 🚀 Experimental

## Repository Structure

```
Phideus/
├── src/
│   ├── shared/           # Common components
│   │   ├── analizador/   # STFT analysis
│   │   ├── auditor/      # Validation
│   │   └── generador/    # WAV synthesis
│   ├── vae/              # VAE Current Line
│   │   ├── models/       # VAE implementations
│   │   ├── training/     # Training scripts
│   │   └── experiments/  # Contrastive learning
│   └── hrm/              # HRM Research Line
│       ├── models/       # HRM implementations
│       ├── training/     # Hierarchical training
│       └── experiments/  # ACT, deep supervision
├── models/
│   ├── vae/              # VAE trained models
│   └── hrm/              # HRM trained models
├── config/               # Architecture configurations
├── scripts/              # Development tools
├── benchmarks/           # Independent testing
└── Documents/            # Project documentation
```

## Development Workflow

### Environment Switching

```bash
# Switch to VAE current line
source scripts/switch_env.sh vae

# Switch to HRM research line  
source scripts/switch_env.sh hrm

# Comparison mode
source scripts/switch_env.sh compare
```

### Training

```bash
# VAE current line
python src/vae/training/train_vae_current.py

# HRM research line
python src/hrm/training/train_hrm_hierarchical.py
```

### Benchmarking

```bash
# Independent benchmarks
python benchmarks/vae_benchmarks.py
python benchmarks/hrm_benchmarks.py

# A/B comparison
python scripts/compare_models.py
```

## Git Branch Structure

```
main/                    # Stable production
├── develop/             # Integration branch
├── feature/vae-current  # VAE development
└── feature/hrm-research # HRM development
```

## Decision Criteria

The project will evaluate both architectures based on:

1. **Harmonic Detection Performance**: Primary metric
2. **Memory Efficiency**: Secondary consideration  
3. **Training Stability**: Production readiness
4. **Research Impact**: Scientific contribution

Timeline: **3 months** for comprehensive comparison and final architecture selection.

## Success Metrics

### VAE Current Line
- **Target**: >80% reconstruction, >15% harmonic detection
- **Timeline**: 4-6 weeks optimization
- **Risk**: Low (proven approach)

### HRM Research Line  
- **Target**: >20% harmonic search, O(1) memory
- **Timeline**: 6-8 weeks implementation + validation
- **Risk**: High (novel architecture)

## Configuration

Each architecture maintains independent configuration:

- **VAE**: `config/vae_config.yaml`
- **HRM**: `config/hrm_config.yaml`  
- **Base**: `config/base_config.py`

Environment variables control active architecture:
- `PHIDEUS_ARCH`: VAE|HRM
- `PHIDEUS_CONFIG`: Path to config file
- `PHIDEUS_LINE`: current|research

This dual architecture enables both **safe consolidation** and **breakthrough innovation** in parallel.