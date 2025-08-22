#!/usr/bin/env python3
"""
Test rápido del Attention-Based Temporal VAE
Verifica que el pipeline básico funciona sin entrenar
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported")
        
        # Test CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_name}")
            
            # Simple GPU test
            x = torch.randn(10, 10).cuda()
            y = torch.matmul(x, x.t())
            print("✅ GPU computation test passed")
        else:
            print("⚠️  CUDA not available - using CPU")
        
        import numpy as np
        import librosa
        import matplotlib.pyplot as plt
        print("✅ Audio processing libraries imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_temporal_dataset():
    """Test temporal dataset creation"""
    print("\n📊 Testing TemporalDataset...")
    
    try:
        from temporal_dataset import TemporalHistogramDataset
        
        # Test with actual WAV files
        test_wavs_dir = Path("data/test_wavs")
        wav_files = list(test_wavs_dir.glob("*.wav"))
        
        if not wav_files:
            print("⚠️  No test WAV files found")
            return False
        
        print(f"📁 Found {len(wav_files)} test WAVs")
        
        # Create dataset
        dataset = TemporalHistogramDataset(
            audio_files_list=[str(f) for f in wav_files],
            window_size=1.0,
            overlap=0.5,
            max_sequence_length=20  # Short for testing
        )
        
        if len(dataset) == 0:
            print("❌ Dataset is empty")
            return False
            
        print(f"✅ Dataset created with {len(dataset)} sequences")
        
        # Test getting a sample
        histogram_seq, metadata = dataset[0]
        print(f"✅ Sample shape: {histogram_seq.shape}")
        print(f"   Sequence length: {metadata.get('sequence_length')}")
        print(f"   Original duration: {metadata.get('original_duration')}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_model_creation():
    """Test model instantiation"""
    print("\n🧠 Testing model creation...")
    
    try:
        from attention_temporal_vae import RTX3090OptimizedTemporalVAE
        
        model = RTX3090OptimizedTemporalVAE()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n⚡ Testing forward pass...")
    
    try:
        import torch
        from attention_temporal_vae import RTX3090OptimizedTemporalVAE
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RTX3090OptimizedTemporalVAE().to(device)
        model.eval()
        
        # Create dummy input: (batch_size=1, seq_len=10, 512, 3)
        dummy_input = torch.randn(1, 10, 512, 3).to(device)
        
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Device: {device}")
        
        with torch.no_grad():
            reconstructed, mu, logvar, attention_weights = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Reconstructed shape: {reconstructed.shape}")
        print(f"   Latent mu shape: {mu.shape}")
        print(f"   Attention weights shape: {attention_weights.shape}")
        
        # Check for NaN values
        if torch.isnan(reconstructed).any():
            print("⚠️  NaN values detected in reconstruction")
        if torch.isnan(mu).any():
            print("⚠️  NaN values detected in mu")
        if torch.isnan(attention_weights).any():
            print("⚠️  NaN values detected in attention weights")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Quick Test - Attention-Based Temporal VAE")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_temporal_dataset, 
        test_model_creation,
        test_forward_pass
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\n🚀 Next steps:")
        print("  1. python setup_temporal_vae.py  # Complete setup")
        print("  2. python train_temporal_vae.py  # Start training")
        print("  3. python run_temporal_analysis.py <audio.wav> --model <model.pt>")
        return True
    else:
        print("❌ Some tests failed. Check dependencies and implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)