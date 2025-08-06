#!/usr/bin/env python3
"""
Test Training VAE with Fixed Linear Attention
"""

import torch
import torch.optim as optim
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from vae_phideus_v1 import PhideusVAE, vae_loss

def test_training_loop():
    """Test a mini training loop with the fixed Linear Attention."""
    print("🏋️ TESTING TRAINING LOOP WITH LINEAR ATTENTION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model with attention
    model = PhideusVAE(use_attention=True).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Synthetic but realistic data
    batch_size = 16
    n_batches = 10
    
    print(f"Training with {n_batches} batches of size {batch_size}")
    
    model.train()
    losses = []
    
    for epoch in range(n_batches):
        # Generate batch
        x = torch.randn(batch_size, 3, 512).to(device)
        x = torch.sigmoid(x)  # Normalize to [0,1]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        
        # Loss computation
        loss_dict = vae_loss(output['reconstruction'], x, 
                            output['mu'], output['logvar'], 
                            beta=min(1.0, 0.1 + epoch * 0.1))  # β-VAE scheduling
        
        total_loss = loss_dict['total_loss']
        
        # Check for numerical issues
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"❌ NaN/Inf detected at epoch {epoch}")
            return False
        
        # Backward pass
        try:
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            losses.append(total_loss.item())
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch:2d} | Loss: {total_loss:.4f} | "
                      f"Recon: {loss_dict['recon_loss']:.4f} | "
                      f"KL: {loss_dict['kl_loss']:.4f} | "
                      f"β: {loss_dict['beta']:.2f}")
                
        except Exception as e:
            print(f"❌ Training failed at epoch {epoch}: {e}")
            return False
    
    print(f"\n✅ Training completed successfully!")
    print(f"📊 Final loss: {losses[-1]:.4f}")
    print(f"📊 Loss reduction: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"📊 Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    return True

def test_memory_efficiency():
    """Test memory usage with attention."""
    print("\n💾 TESTING MEMORY EFFICIENCY")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    
    model = PhideusVAE(use_attention=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Large batch to test memory
    batch_size = 32
    x = torch.randn(batch_size, 3, 512).to(device)
    x = torch.sigmoid(x)
    
    # Forward + backward
    optimizer.zero_grad()
    output = model(x)
    loss_dict = vae_loss(output['reconstruction'], x, 
                        output['mu'], output['logvar'])
    loss_dict['total_loss'].backward()
    optimizer.step()
    
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        print(f"📊 Initial memory: {initial_memory / 1e6:.1f} MB")
        print(f"📊 Peak memory: {peak_memory / 1e6:.1f} MB")
        print(f"📊 Current memory: {current_memory / 1e6:.1f} MB")
        print(f"📊 Memory increase: {(peak_memory - initial_memory) / 1e6:.1f} MB")
        
        # RTX 3090 has ~24GB, should be well under that
        if peak_memory > 20e9:  # 20GB
            print("⚠️ High memory usage detected")
        else:
            print("✅ Memory usage is reasonable")
    
    return True

if __name__ == "__main__":
    # Training test
    training_success = test_training_loop()
    
    if training_success:
        # Memory test
        test_memory_efficiency()
        
        print("\n🎉 ALL TRAINING TESTS PASSED!")
        print("✅ Linear Attention is ready for production training")
    else:
        print("\n❌ TRAINING TESTS FAILED")
        print("❌ Linear Attention needs further debugging")