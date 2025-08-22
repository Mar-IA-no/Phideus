#!/usr/bin/env python3
"""
Setup script para Attention-Based Temporal VAE
Configura el environment, instala dependencias y verifica hardware
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is too old")
        print("   Temporal VAE requires Python >= 3.8")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} OK")
    return True

def check_gpu_availability():
    """Verificar disponibilidad de GPU"""
    print("\n🎮 Checking GPU availability...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            
            # Extraer información básica de GPU
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                    print(f"   GPU: {line.strip()}")
                    
                    if 'RTX 3090' in line:
                        print("   🚀 RTX 3090 detected - Optimal for Temporal VAE!")
                    elif any(gpu in line for gpu in ['RTX 4090', 'RTX 4080', 'A100']):
                        print("   ⚡ High-end GPU - Excellent for Temporal VAE!")
                    elif any(gpu in line for gpu in ['RTX 3080', 'RTX 3070', 'RTX 2080']):
                        print("   ✅ Good GPU - Suitable for Temporal VAE")
                    else:
                        print("   ⚠️  Lower-end GPU - May need reduced sequence length")
                    break
            
            return True
    except FileNotFoundError:
        pass
    
    print("❌ No NVIDIA GPU detected")
    print("   Temporal VAE will run on CPU (very slow)")
    return False

def install_dependencies():
    """Instalar dependencias de PyTorch"""
    print("\n📦 Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements_temporal.txt"
    
    if not requirements_file.exists():
        print("❌ requirements_temporal.txt not found")
        return False
    
    try:
        # Intentar instalar con pip
        print("   Installing PyTorch and dependencies...")
        
        # Para sistemas con GPU CUDA
        cuda_available = check_gpu_availability()
        
        if cuda_available:
            # Instalar PyTorch con CUDA
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True)
        else:
            # Instalar PyTorch CPU-only
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ], check=True)
        
        # Instalar otras dependencias
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_file)
        ], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Try manual installation:")
        print(f"   pip install -r {requirements_file}")
        return False

def verify_installation():
    """Verificar que PyTorch funciona correctamente"""
    print("\n🧪 Verifying PyTorch installation...")
    
    try:
        import torch
        
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Check CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.get_device_name(0)
            
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"   Current device: {current_device}")
            
            # Test basic tensor operations
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x.t())
            
            print("✅ GPU tensor operations working")
            
        else:
            print("⚠️  CUDA not available - using CPU")
        
        # Test imports for temporal VAE components
        test_imports = [
            'numpy', 'librosa', 'matplotlib', 'tqdm', 'scipy'
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                print(f"✅ {module} imported OK")
            except ImportError:
                print(f"❌ {module} not available")
                
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def create_config():
    """Crear configuración optimizada basada en hardware"""
    print("\n⚙️  Creating optimized configuration...")
    
    config = {
        "model": {
            "embed_dim": 128,
            "latent_dim": 128,
            "num_attention_heads": 8,
            "max_sequence_length": 120
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 50,
            "mixed_precision": True,
            "gradient_clipping": 1.0
        },
        "data": {
            "window_size": 1.0,
            "overlap": 0.5,
            "sample_rate": 44100,
            "normalize": True
        },
        "hardware_optimizations": {
            "rtx_3090": {
                "max_sequence_length": 60,
                "batch_size": 2,
                "num_attention_heads": 4,
                "gradient_checkpointing": True
            },
            "rtx_4090": {
                "max_sequence_length": 120,
                "batch_size": 4,
                "num_attention_heads": 8,
                "gradient_checkpointing": False
            },
            "cpu_only": {
                "max_sequence_length": 20,
                "batch_size": 1,
                "num_attention_heads": 2,
                "mixed_precision": False
            }
        }
    }
    
    # Detectar hardware y ajustar config
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            
            if "RTX 3090" in device_name:
                print("   Applying RTX 3090 optimizations")
                config["model"].update(config["hardware_optimizations"]["rtx_3090"])
                config["training"]["batch_size"] = 2
                
            elif any(gpu in device_name for gpu in ["RTX 4090", "RTX 4080"]):
                print("   Applying RTX 4090 optimizations")
                config["model"].update(config["hardware_optimizations"]["rtx_4090"])
                
            else:
                print("   Applying generic GPU optimizations")
                config["model"]["max_sequence_length"] = 60
                config["training"]["batch_size"] = 2
        else:
            print("   Applying CPU optimizations")
            config["model"].update(config["hardware_optimizations"]["cpu_only"])
            config["training"]["batch_size"] = 1
            
    except ImportError:
        print("   Using default configuration")
    
    # Guardar configuración
    config_file = Path(__file__).parent / "config.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_file}")
    
    # Mostrar configuración recomendada
    print("\n📋 Recommended configuration:")
    print(f"   Max sequence length: {config['model']['max_sequence_length']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Attention heads: {config['model']['num_attention_heads']}")
    print(f"   Mixed precision: {config['training']['mixed_precision']}")
    
    return config

def setup_directories():
    """Crear estructura de directorios necesaria"""
    print("\n📁 Setting up directories...")
    
    base_dir = Path(__file__).parent
    
    directories = [
        "checkpoints",
        "logs", 
        "results",
        "data",
        "experiments"
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")

def main():
    """Setup principal"""
    print("🚀 Phideus Attention-Based Temporal VAE Setup")
    print("=" * 60)
    
    success = True
    
    # 1. Verificar Python
    if not check_python_version():
        success = False
    
    # 2. Verificar GPU
    gpu_available = check_gpu_availability()
    
    # 3. Instalar dependencias
    if success:
        if not install_dependencies():
            print("\n⚠️  Dependency installation failed")
            print("   You can try manual installation:")
            print("   pip install torch torchvision torchaudio")
            print("   pip install librosa numpy matplotlib tqdm")
            success = False
    
    # 4. Verificar instalación
    if success:
        if not verify_installation():
            success = False
    
    # 5. Crear configuración
    if success:
        config = create_config()
    
    # 6. Setup directorios
    if success:
        setup_directories()
    
    print("\n" + "=" * 60)
    
    if success:
        print("✅ Setup completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Prepare your WAV files in a directory")
        print("2. Run: python train_temporal_vae.py")
        print("3. Monitor training progress")
        print("4. Use trained model for temporal analysis")
        
        if gpu_available:
            print("\n💡 GPU Tips:")
            print("• Monitor GPU memory usage during training")
            print("• Reduce batch_size if you get OOM errors")
            print("• Use mixed precision for better performance")
        else:
            print("\n⚠️  CPU Mode:")
            print("• Training will be very slow on CPU")
            print("• Consider using cloud GPU instances")
            print("• Reduce sequence length for faster processing")
            
    else:
        print("❌ Setup failed!")
        print("   Please fix the issues above and try again")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)