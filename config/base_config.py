#!/usr/bin/env python3
"""
Base configuration management for Phideus Dual Architecture
Handles switching between VAE and HRM configurations
"""

import os
import yaml
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Union

class Architecture(Enum):
    VAE = "vae"
    HRM = "hrm"

@dataclass
class PhideusConfig:
    """Base configuration class for Phideus dual architecture"""
    architecture: Architecture
    line: str
    model_path: str
    data_path: str
    
    # Shared parameters
    input_shape: tuple = (512, 3)
    batch_size: int = 32
    
    def __post_init__(self):
        self.config_data = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load architecture-specific configuration"""
        config_file = f"config/{self.architecture.value}_config.yaml"
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def from_env(cls) -> 'PhideusConfig':
        """Create config from environment variables"""
        arch = os.getenv('PHIDEUS_ARCH', 'VAE')
        
        if arch.upper() == 'VAE':
            architecture = Architecture.VAE
            line = "current"
        elif arch.upper() == 'HRM':
            architecture = Architecture.HRM  
            line = "research"
        else:
            raise ValueError(f"Unknown architecture: {arch}")
            
        return cls(
            architecture=architecture,
            line=line,
            model_path=f"models/{architecture.value}/",
            data_path="models/datasets/train_vae_enriched_512.json"
        )
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.config_data.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return self.config_data.get('training', {})
        
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation-specific configuration"""
        return self.config_data.get('validation', {})

def load_phideus_config() -> PhideusConfig:
    """Load configuration based on current environment"""
    return PhideusConfig.from_env()

def print_current_config():
    """Debug function to print current configuration"""
    config = load_phideus_config()
    
    print(f"🔧 Phideus Configuration")
    print(f"Architecture: {config.architecture.value.upper()}")
    print(f"Line: {config.line}")
    print(f"Model Path: {config.model_path}")
    print(f"Input Shape: {config.input_shape}")
    print(f"Batch Size: {config.batch_size}")
    
    if config.architecture == Architecture.VAE:
        print("📊 VAE Specific:")
        model_cfg = config.get_model_config()
        print(f"  Linear Attention: {model_cfg.get('linear_attention', False)}")
        print(f"  Latent Dim: {model_cfg.get('latent_dim', 128)}")
        print(f"  Parameters: {model_cfg.get('parameters', 'Unknown')}")
        
    elif config.architecture == Architecture.HRM:
        print("🧠 HRM Specific:")
        model_cfg = config.get_model_config()
        print(f"  H-Module Dim: {model_cfg.get('h_module_dim', 256)}")
        print(f"  L-Module Dim: {model_cfg.get('l_module_dim', 256)}")
        print(f"  N Cycles: {model_cfg.get('N_cycles', 4)}")
        print(f"  T Steps: {model_cfg.get('T_steps', 8)}")
        print(f"  ACT Enabled: {model_cfg.get('act_enabled', True)}")

if __name__ == "__main__":
    print_current_config()