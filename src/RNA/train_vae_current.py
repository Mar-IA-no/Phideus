#!/usr/bin/env python3
"""
Training script for VAE current line - Phideus Dual Architecture
Maintains current VAE + Linear Attention approach with optimizations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.vae.models.vae_phideus_v1 import *
from src.vae.models.train_vae_phideus import *

def main():
    """Main training function for VAE current line"""
    print("🎵 Phideus VAE Current Line - Training Started")
    print("Architecture: VAE + Linear Attention")
    print("Dataset: models/datasets/train_vae_enriched_512.json")
    print("Target: Consolidate current approach with optimizations")
    
    # Use existing training pipeline with minor modifications
    config = {
        'architecture': 'vae_current',
        'model_path': 'models/vae/',
        'dataset_expansion': True,  # Enable 500+ samples
        'contrastive_learning': True,  # Enable MoCo-v3
        'hyperparameter_tuning': True
    }
    
    # Call existing training function
    train_vae_phideus(config)

if __name__ == "__main__":
    main()