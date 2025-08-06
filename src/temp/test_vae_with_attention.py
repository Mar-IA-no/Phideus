#!/usr/bin/env python3
"""
Test VAE with Fixed Linear Attention - Validation completa
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from vae_phideus_v1 import PhideusVAE, vae_loss

def test_vae_with_attention():
    """Test completo del VAE con Linear Attention fijo."""
    print("🧪 TESTING VAE WITH FIXED LINEAR ATTENTION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test con attention habilitada
    model = PhideusVAE(use_attention=True).to(device)
    
    batch_size = 8
    x = torch.randn(batch_size, 3, 512).to(device)
    x = torch.sigmoid(x)  # Normalizar a [0,1] como histogramas
    
    print(f"📊 Input shape: {x.shape}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(x)
        
        print(f"✅ Forward pass successful")
        print(f"📊 Reconstruction shape: {output['reconstruction'].shape}")
        print(f"📊 Latent μ shape: {output['mu'].shape}")
        print(f"📊 Latent σ shape: {output['logvar'].shape}")
        
        # Verificar salud de outputs
        recon = output['reconstruction']
        mu = output['mu']
        logvar = output['logvar']
        
        # Check NaN/Inf
        has_nan_recon = torch.isnan(recon).any()
        has_inf_recon = torch.isinf(recon).any()
        has_nan_mu = torch.isnan(mu).any()
        has_inf_mu = torch.isinf(mu).any()
        has_nan_logvar = torch.isnan(logvar).any()
        has_inf_logvar = torch.isinf(logvar).any()
        
        print(f"📊 Reconstruction - NaN: {has_nan_recon}, Inf: {has_inf_recon}")
        print(f"📊 Mu - NaN: {has_nan_mu}, Inf: {has_inf_mu}")
        print(f"📊 LogVar - NaN: {has_nan_logvar}, Inf: {has_inf_logvar}")
        
        if any([has_nan_recon, has_inf_recon, has_nan_mu, has_inf_mu, has_nan_logvar, has_inf_logvar]):
            print("❌ Forward pass has numerical issues")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Loss computation
    try:
        loss_dict = vae_loss(recon, x, mu, logvar, beta=1.0)
        
        total_loss = loss_dict['total_loss']
        recon_loss = loss_dict['recon_loss']
        kl_loss = loss_dict['kl_loss']
        
        print(f"📉 Total loss: {total_loss:.4f}")
        print(f"📉 Recon loss: {recon_loss:.4f}")
        print(f"📉 KL loss: {kl_loss:.4f}")
        
        # Check loss health
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("❌ Loss computation has numerical issues")
            return False
            
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    # Backward pass test
    try:
        x.requires_grad_(True)
        output = model(x)
        loss_dict = vae_loss(output['reconstruction'], x, 
                            output['mu'], output['logvar'], beta=1.0)
        
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        print("✅ Backward pass successful")
        
        # Check gradients health
        grad_issues = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"❌ Gradient issues in {name}")
                    grad_issues += 1
        
        if grad_issues > 0:
            print(f"❌ {grad_issues} parameters have gradient issues")
            return False
        else:
            print("✅ All gradients are healthy")
            
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    print("\n🎉 VAE WITH ATTENTION VALIDATION SUCCESSFUL!")
    return True

def compare_with_without_attention():
    """Compare VAE performance with and without attention."""
    print("\n🔍 COMPARING VAE WITH/WITHOUT ATTENTION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_no_attn = PhideusVAE(use_attention=False).to(device)
    model_with_attn = PhideusVAE(use_attention=True).to(device)
    
    # Same input for fair comparison
    torch.manual_seed(42)
    x = torch.randn(4, 3, 512).to(device)
    x = torch.sigmoid(x)
    
    results = {}
    
    # Test without attention
    with torch.no_grad():
        output_no = model_no_attn(x)
        loss_no = vae_loss(output_no['reconstruction'], x, 
                          output_no['mu'], output_no['logvar'], beta=1.0)
        
        results['no_attention'] = {
            'total_loss': loss_no['total_loss'].item(),
            'recon_loss': loss_no['recon_loss'].item(),
            'kl_loss': loss_no['kl_loss'].item(),
            'params': sum(p.numel() for p in model_no_attn.parameters())
        }
    
    # Test with attention
    with torch.no_grad():
        output_with = model_with_attn(x)
        loss_with = vae_loss(output_with['reconstruction'], x,
                            output_with['mu'], output_with['logvar'], beta=1.0)
        
        results['with_attention'] = {
            'total_loss': loss_with['total_loss'].item(),
            'recon_loss': loss_with['recon_loss'].item(),
            'kl_loss': loss_with['kl_loss'].item(),
            'params': sum(p.numel() for p in model_with_attn.parameters())
        }
    
    print(f"📊 WITHOUT Attention:")
    print(f"   Parameters: {results['no_attention']['params']:,}")
    print(f"   Total Loss: {results['no_attention']['total_loss']:.4f}")
    print(f"   Recon Loss: {results['no_attention']['recon_loss']:.4f}")
    print(f"   KL Loss: {results['no_attention']['kl_loss']:.4f}")
    
    print(f"\n📊 WITH Attention:")
    print(f"   Parameters: {results['with_attention']['params']:,}")
    print(f"   Total Loss: {results['with_attention']['total_loss']:.4f}")
    print(f"   Recon Loss: {results['with_attention']['recon_loss']:.4f}")
    print(f"   KL Loss: {results['with_attention']['kl_loss']:.4f}")
    
    param_increase = results['with_attention']['params'] - results['no_attention']['params']
    print(f"\n📊 Parameter increase: +{param_increase:,} ({param_increase/results['no_attention']['params']*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    # Main validation
    success = test_vae_with_attention()
    
    if success:
        # Comparison test
        compare_with_without_attention()
        
        print("\n✅ ALL TESTS PASSED - LINEAR ATTENTION FIX VALIDATED!")
    else:
        print("\n❌ VALIDATION FAILED - FURTHER DEBUGGING NEEDED")