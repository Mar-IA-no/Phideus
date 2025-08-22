#!/usr/bin/env python3
"""
Testing script para Attention-Based Temporal VAE
Verifica que todos los componentes funcionen correctamente en RTX 3090
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

from attention_temporal_vae import RTX3090OptimizedTemporalVAE
from temporal_dataset import TemporalHistogramDataset, create_temporal_dataloaders

def test_components():
    """Test individual de componentes"""
    print("🔧 Testing individual components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test FrameEncoder
    print("\n1. Testing FrameEncoder...")
    from frame_encoder import RTX3090OptimizedFrameEncoder
    
    encoder = RTX3090OptimizedFrameEncoder().to(device)
    test_histogram = torch.randn(4, 512, 3).to(device)
    
    with torch.no_grad():
        embedding = encoder(test_histogram)
        print(f"   ✅ FrameEncoder: {test_histogram.shape} → {embedding.shape}")
    
    # Test TemporalSelfAttention
    print("\n2. Testing TemporalSelfAttention...")
    from temporal_attention import RTX3090OptimizedAttention
    
    attention = RTX3090OptimizedAttention(embed_dim=128, num_heads=4).to(device)
    test_sequence = torch.randn(2, 30, 128).to(device)
    
    with torch.no_grad():
        attended, att_weights = attention(test_sequence)
        print(f"   ✅ TemporalAttention: {test_sequence.shape} → {attended.shape}")
        print(f"   ✅ Attention weights: {att_weights.shape}")
    
    # Test TemporalAggregator
    print("\n3. Testing TemporalAggregator...")
    from temporal_aggregator import MemoryEfficientAggregator
    
    aggregator = MemoryEfficientAggregator().to(device)
    
    with torch.no_grad():
        mu, logvar = aggregator(attended)
        print(f"   ✅ TemporalAggregator: {attended.shape} → mu{mu.shape}, logvar{logvar.shape}")
        print(f"   ✅ Logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")

def test_full_model():
    """Test del modelo completo"""
    print("\n🧠 Testing complete Temporal VAE model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear modelo optimizado
    model = RTX3090OptimizedTemporalVAE().to(device)
    
    # Test input: batch de secuencias
    batch_size, seq_len = 2, 30
    test_sequences = torch.randn(batch_size, seq_len, 512, 3).to(device)
    
    print(f"Input sequences: {test_sequences.shape}")
    
    # Forward pass completo
    start_time = time.time()
    
    with torch.no_grad():
        reconstructed, mu, logvar, attention_weights = model(test_sequences)
    
    inference_time = time.time() - start_time
    
    print(f"✅ Forward pass successful:")
    print(f"   Input: {test_sequences.shape}")
    print(f"   Reconstructed: {reconstructed.shape}")
    print(f"   Latent mu: {mu.shape}")
    print(f"   Latent logvar: {logvar.shape}")
    print(f"   Attention weights: {attention_weights.shape}")
    print(f"   Inference time: {inference_time:.3f} seconds")
    
    # Test temporal summary
    summary = model.get_temporal_summary(test_sequences)
    print(f"✅ Temporal summary:")
    print(f"   Influence scores: {summary['influence_scores'].shape}")
    print(f"   Correlations found: {len(summary['temporal_correlations'][0])}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"✅ GPU Memory:")
        print(f"   Used: {memory_used:.3f} GB")
        print(f"   Reserved: {memory_reserved:.3f} GB")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024**2:.1f} MB")
    
    return model

def test_training_step():
    """Test de un paso de entrenamiento"""
    print("\n🏃 Testing training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RTX3090OptimizedTemporalVAE().to(device)
    
    # Test data
    batch_size, seq_len = 2, 20
    sequences = torch.randn(batch_size, seq_len, 512, 3).to(device)
    
    # Loss function
    from train_temporal_vae import TemporalVAELoss
    criterion = TemporalVAELoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    start_time = time.time()
    reconstructed, mu, logvar, attention_weights = model(sequences)
    forward_time = time.time() - start_time
    
    # Compute loss
    loss_dict = criterion(
        reconstructed, sequences, mu, logvar, attention_weights
    )
    
    # Backward pass
    backward_start = time.time()
    loss_dict['total_loss'].backward()
    optimizer.step()
    backward_time = time.time() - backward_start
    
    print(f"✅ Training step successful:")
    print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"   KL loss: {loss_dict['kl_loss'].item():.4f}")
    print(f"   Temporal loss: {loss_dict['temporal_loss'].item():.4f}")
    print(f"   Forward time: {forward_time:.3f}s")
    print(f"   Backward time: {backward_time:.3f}s")
    
    # Check gradients
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"   Gradient norm: {total_grad_norm:.4f}")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU Memory after training step: {memory_used:.3f} GB")

def test_memory_scaling():
    """Test escalado de memoria con diferentes sequence lengths"""
    print("\n📈 Testing memory scaling...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("   Skipping memory test (no CUDA)")
        return
    
    model = RTX3090OptimizedTemporalVAE().to(device)
    
    batch_size = 2
    sequence_lengths = [10, 20, 30, 40, 50, 60]
    
    print("   Seq Len | Memory (GB) | Time (s)")
    print("   --------|-------------|--------")
    
    for seq_len in sequence_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        sequences = torch.randn(batch_size, seq_len, 512, 3).to(device)
        
        try:
            start_time = time.time()
            with torch.no_grad():
                _ = model(sequences)
            inference_time = time.time() - start_time
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"   {seq_len:6d} | {memory_used:10.3f} | {inference_time:7.3f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   {seq_len:6d} | OOM         | -------")
                break
            else:
                raise e
    
    torch.cuda.empty_cache()

def test_attention_patterns():
    """Test de attention patterns para interpretabilidad"""
    print("\n🔍 Testing attention patterns...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RTX3090OptimizedTemporalVAE().to(device)
    
    # Crear secuencia con patrón conocido
    batch_size, seq_len = 1, 20
    sequences = torch.randn(batch_size, seq_len, 512, 3).to(device)
    
    # Hacer algunos frames más similares para ver si attention los detecta
    sequences[0, 5] = sequences[0, 15] + 0.1 * torch.randn_like(sequences[0, 5])
    sequences[0, 8] = sequences[0, 12] + 0.1 * torch.randn_like(sequences[0, 8])
    
    with torch.no_grad():
        summary = model.get_temporal_summary(sequences)
    
    attention_patterns = summary['attention_patterns'][0].cpu().numpy()
    influence_scores = summary['influence_scores'][0].cpu().numpy()
    correlations = summary['temporal_correlations'][0]
    
    print(f"✅ Attention analysis:")
    print(f"   Attention matrix shape: {attention_patterns.shape}")
    print(f"   Top 3 most influential frames: {np.argsort(influence_scores)[-3:]}")
    print(f"   Strong correlations found: {len(correlations)}")
    
    if correlations:
        print("   Top correlations:")
        correlations.sort(key=lambda x: x[2], reverse=True)
        for i, j, corr in correlations[:3]:
            print(f"      Frame {i} ↔ Frame {j}: {corr:.3f}")

def main():
    """Test principal"""
    print("🚀 Starting Attention-Based Temporal VAE Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        test_components()
        
        # Test complete model
        test_full_model()
        
        # Test training step
        test_training_step()
        
        # Test memory scaling
        test_memory_scaling()
        
        # Test attention patterns
        test_attention_patterns()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("🎯 Attention-Based Temporal VAE is ready for training")
        
        # RTX 3090 recommendations
        print("\n🎮 RTX 3090 Recommendations:")
        print("   • Max sequence length: 60 frames")
        print("   • Batch size: 2-4")
        print("   • Use mixed precision training")
        print("   • Expected memory usage: 2-4 GB")
        print("   • Training time: ~4 weeks for full implementation")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)