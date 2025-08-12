# Getting Started - Phideus Dual Architecture

Quick start guide for developing with Phideus dual architecture system.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install torch torchvision numpy scipy librosa soundfile matplotlib tqdm tabulate pyyaml
```

## Architecture Selection

### Option 1: VAE Current Line (Recommended for Beginners)

```bash
# Switch to VAE environment
source scripts/switch_env.sh vae

# Verify configuration
python config/base_config.py

# Train VAE model
python src/vae/training/train_vae_current.py

# Run benchmarks
python benchmarks/vae_benchmarks.py
```

### Option 2: HRM Research Line (Advanced)

```bash
# Switch to HRM environment
source scripts/switch_env.sh hrm

# Verify configuration  
python config/base_config.py

# Train HRM model (experimental)
python src/hrm/training/train_hrm_hierarchical.py

# Run benchmarks
python benchmarks/hrm_benchmarks.py
```

### Option 3: Comparison Mode

```bash
# Switch to comparison mode
source scripts/switch_env.sh compare

# Run A/B comparison
python scripts/compare_models.py

# Save detailed comparison
python scripts/compare_models.py --output comparison_results.json
```

## Basic Workflow

### 1. Generate Test Data
```bash
# Generate synthetic WAVs with known harmonic ratios
python src/shared/generador/generador_wavs_ratios_complejos_v3.0_Ninja.py
```

### 2. Analyze Audio
```bash
# Convert WAVs to enriched histograms
python src/shared/analizador/analizador_4.1_Enriched.py --input-dir test_wavs --output dataset.json
```

### 3. Train Architecture
```bash
# VAE approach
source scripts/switch_env.sh vae
python src/vae/training/train_vae_current.py

# OR HRM approach  
source scripts/switch_env.sh hrm
python src/hrm/training/train_hrm_hierarchical.py
```

### 4. Evaluate Results
```bash
# Architecture-specific benchmarks
python benchmarks/vae_benchmarks.py
python benchmarks/hrm_benchmarks.py

# Compare both approaches
python scripts/compare_models.py
```

## Project Structure Quick Reference

```
Phideus/
├── src/
│   ├── shared/     # Common analysis tools
│   ├── vae/        # VAE current line
│   └── hrm/        # HRM research line
├── models/         # Trained models
├── config/         # Architecture configs
├── scripts/        # Development tools
└── benchmarks/     # Testing suites
```

## Environment Variables

- `PHIDEUS_ARCH`: Current architecture (VAE|HRM)
- `PHIDEUS_CONFIG`: Path to config file
- `PHIDEUS_LINE`: Development line (current|research)

## Common Tasks

### Switch Architectures
```bash
source scripts/switch_env.sh [vae|hrm|compare]
```

### Check Current Environment
```bash
echo "Architecture: $PHIDEUS_ARCH"
echo "Config: $PHIDEUS_CONFIG"
echo "Line: $PHIDEUS_LINE"
```

### Run Tests
```bash
# Quick validation
python config/base_config.py

# Full benchmarks
python benchmarks/${PHIDEUS_ARCH,,}_benchmarks.py
```

### Model Comparison
```bash
python scripts/compare_models.py --vae-model models/vae/attention/best_model.pth --hrm-model models/hrm/core/hrm_initial.pth
```

## Development Guidelines

1. **Always use environment switcher** before development
2. **Keep architectures isolated** - no cross-imports
3. **Use shared components** for common functionality  
4. **Run benchmarks** before committing changes
5. **Document experiments** in respective directories

## Getting Help

- **Configuration Issues**: Check `config/base_config.py`
- **Architecture Questions**: See `ARCHITECTURE.md`
- **Development**: See `Documents/` folder
- **Bugs**: Create issue with environment details

## Next Steps

1. Choose your development focus (VAE vs HRM)
2. Run initial benchmarks to establish baseline
3. Review architecture-specific documentation
4. Start with small experiments
5. Compare results between architectures