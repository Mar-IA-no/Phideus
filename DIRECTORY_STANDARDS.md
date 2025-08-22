# Directory Organization Standards - Phideus Project

## **MANDATORY Repository Structure**

This document defines the official directory organization for the Phideus project. **ALWAYS respect this structure**.

```
/root/Phideus/
├── README.md                     # Project overview and main documentation
├── LICENSE.md                    # Project license
├── CLAUDE.md                     # Claude Code guidance
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── DIRECTORY_STANDARDS.md        # This file
│
├── src/                          # Production source code
│   ├── hrm/                      # Enhanced HRM (PRIMARY MODEL ⭐⭐⭐⭐⭐)
│   │   ├── training/             # HRM training scripts
│   │   ├── models/               # HRM model definitions (ACT, L-Module, H-Module)
│   │   ├── validation/           # HRM validation and comparison tools
│   │   ├── scripts/              # HRM utility scripts
│   │   └── examples/             # HRM usage examples
│   ├── RNA/                      # VAE models (secondary/legacy)
│   │   ├── train_vae_phideus.py  # Enhanced VAE training
│   │   ├── validate_vae_phideus.py # VAE validation
│   │   ├── vae_checkpoints/      # VAE model checkpoints
│   │   └── vae_validation/       # VAE validation outputs
│   ├── analizador/               # Audio analysis tools
│   │   └── analizador_v4.0.py    # Main analyzer
│   ├── auditor/                  # Auditing and reporting
│   │   └── auditor_v4.0.py       # Main auditor
│   ├── generador/                # WAV generation
│   │   └── generador_wavs_ratios_complejos_v3.0_Ninja.py
│   └── temp/                     # Temporary/experimental scripts
│
├── experiments/                  # Research experiments and comparisons
│   ├── train_vae_base.py         # Baseline VAE for comparison
│   ├── compare_three_architectures.py # Main comparison script
│   ├── temporal/                 # Temporal VAE experiments (moved from src)
│   └── benchmarks/               # Performance benchmarking
│       ├── vae_benchmarks/       # VAE performance tests
│       └── hrm_benchmarks/       # HRM performance tests
│
├── data/                         # Training data and outputs
│   ├── datasets/                 # JSON datasets (gitignored if >1MB)
│   ├── training_outputs/         # Organized by model type
│   │   ├── hrm/                  # HRM training outputs
│   │   ├── vae/                  # VAE training outputs
│   │   └── comparisons/          # Multi-architecture comparisons
│   └── test_data/                # Test datasets
│
├── models/                       # Saved models organized by architecture
│   ├── hrm/                      # HRM models (production ready)
│   └── vae/                      # VAE models (legacy/comparison)
│       ├── baseline/             # Baseline VAE models
│       └── attention/            # Enhanced VAE models
│
├── test/                         # Test files (gitignored)
│   ├── test_wavs/                # Test audio files
│   ├── test-json/                # Test JSON outputs
│   └── validation_plots/         # Test validation plots
│
├── Documents/                    # Project documentation
│   ├── bitacora_desarrollo.md    # Development log
│   ├── Proyecto_Estado_Actual.md # Current project status
│   ├── RNA_Arqu.md               # Neural architecture documentation
│   └── [other documentation]
│
├── config/                       # Configuration files
├── scripts/                      # Utility scripts (if any)
└── train/                        # Training datasets (gitignored)
    └── [large training files]
```

## **File Organization Principles**

### **🚫 NEVER Commit These Files**
- **Audio files**: WAV, MP3, etc. (too large for Git)
- **Large datasets**: JSON files >1MB
- **Virtual environments**: `*_env/`, `venv/`, etc.
- **Python cache**: `__pycache__/`, `*.pyc`
- **Training data**: Files in `train/` directory
- **Temporary files**: Large outputs, intermediate results

### **✅ Core Organization Rules**

1. **Enhanced HRM Priority**: 
   - Primary model in `src/hrm/`
   - All new harmonic analysis work uses Enhanced HRM
   - VAE models are legacy/comparison only

2. **Clean Separation**:
   - Production code: `src/`
   - Research experiments: `experiments/`
   - Temporary scripts: `src/temp/`
   - Documentation: `Documents/`

3. **Training Outputs Organization**:
   - By model type: `data/training_outputs/[hrm|vae]/`
   - Include plots, models, and metrics
   - Comparison results in `data/training_outputs/comparisons/`

4. **Model Storage**:
   - Production models: `models/[architecture]/`
   - Checkpoints with training outputs
   - Clear naming conventions

## **Maintenance Rules**

### **Regular Cleanup Tasks**
```bash
# Remove Python cache
find /root/Phideus -name "__pycache__" -type d -exec rm -rf {} +

# Remove virtual environments
rm -rf /root/Phideus/*_env/

# Clean temporary files
find /root/Phideus/src/temp -name "*.pyc" -delete
```

### **When Adding New Files**

1. **Scripts**: Determine if production (`src/`) or experimental (`experiments/`)
2. **Models**: Save in appropriate `models/[architecture]/` subdirectory
3. **Results**: Organize in `data/training_outputs/[model_type]/`
4. **Documentation**: Add to `Documents/` with clear naming

### **Repository Reorganization Protocol**

When reorganizing:
1. **Backup important results** before moving files
2. **Update documentation** to reflect changes
3. **Test that scripts still work** after path changes
4. **Update CLAUDE.md** with new structure
5. **Document changes** in development log

## **Scientific Priority Based on Findings**

### **Architecture Performance Ranking** (Latest Results)
1. **Enhanced HRM**: 153,500% better than VAE variants
   - Primary focus for all future development
   - Production-ready harmonic analysis

2. **Enhanced VAE**: 0.000% improvement over Baseline VAE
   - Minimal value, legacy support only
   - Comparison reference

3. **Baseline VAE**: Reference implementation
   - Comparison baseline only
   - No production use

### **Development Focus**
- **Primary**: Enhanced HRM improvements and applications
- **Secondary**: Real-world validation and testing
- **Minimal**: VAE enhancements (proven ineffective)

---

**Last Updated**: 2025-08-22  
**Based on**: Three-architecture comparison findings  
**Status**: ✅ Revolutionary HRM superiority established