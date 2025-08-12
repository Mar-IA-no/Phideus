#!/usr/bin/env python3
"""
Debug VAE Loss - Identificar exactamente dónde aparecen NaN en el loss

El Linear Attention funciona bien aislado, pero el VAE falla cuando se calcula la loss.
Vamos a debuggear step-by-step el cálculo de loss para encontrar el origen de los NaN.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add src to path para imports  
sys.path.append(str(Path(__file__).parent.parent))
from vae_phideus_v1 import PhideusVAE, vae_loss

def check_tensor_health(tensor, name="tensor"):
    """Analiza la salud de un tensor."""
    if tensor.numel() == 0:
        return {
            'name': name,
            'shape': tensor.shape,
            'empty': True
        }
    
    stats = {
        'name': name,
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item() if tensor.numel() > 1 else 0,
        'zero_ratio': (tensor == 0).float().mean().item(),
        'empty': False
    }
    return stats

def print_tensor_stats(stats, verbose=False):
    """Imprime estadísticas de tensor de forma legible."""
    if stats.get('empty', False):
        print(f"\n📊 {stats['name']}: {stats['shape']} (EMPTY)")
        return False
        
    print(f"\n📊 {stats['name']}: {stats['shape']}")
    print(f"   NaN: {stats['has_nan']:5} | Inf: {stats['has_inf']:5} | Dtype: {stats['dtype']}")
    print(f"   Range: [{stats['min']:8.4f}, {stats['max']:8.4f}] | Mean: {stats['mean']:8.4f}")
    print(f"   Std: {stats['std']:8.4f} | Zero ratio: {stats['zero_ratio']:6.2%}")
    
    if verbose and stats['has_nan']:
        # Localizar NaN values
        tensor = torch.zeros(1)  # Placeholder - en real habría que pasar el tensor
        if hasattr(torch, '_tensor_debug'):
            nan_locations = torch.isnan(tensor).nonzero()
            print(f"   NaN locations: {nan_locations[:5]}...")  # Mostrar primeros 5
    
    is_problematic = stats['has_nan'] or stats['has_inf']
    if is_problematic:
        print(f"   ⚠️ PROBLEMA DETECTADO: {'NaN' if stats['has_nan'] else ''}{'Inf' if stats['has_inf'] else ''}")
    
    return is_problematic

def debug_vae_loss_step_by_step(recon, target, mu, logvar, beta=1.0):
    """Debug detallado del cálculo de VAE loss."""
    print("\n" + "="*60)
    print("🔍 DEBUG VAE LOSS COMPUTATION")
    print("="*60)
    
    # Check inputs
    has_issues = False
    has_issues |= print_tensor_stats(check_tensor_health(recon, "Reconstruction"))
    has_issues |= print_tensor_stats(check_tensor_health(target, "Target"))
    has_issues |= print_tensor_stats(check_tensor_health(mu, "Mu (latent mean)"))
    has_issues |= print_tensor_stats(check_tensor_health(logvar, "LogVar (latent log-var)"))
    
    if has_issues:
        print("\n🚨 INPUTS AL LOSS YA TIENEN ISSUES!")
        return None
    
    # Step 1: Reconstruction loss (MSE)
    print(f"\n🔬 STEP 1: Reconstruction Loss (MSE)")
    print(f"   Shapes: recon={recon.shape}, target={target.shape}")
    
    try:
        diff = recon - target
        diff_stats = check_tensor_health(diff, "Recon - Target difference")
        print_tensor_stats(diff_stats)
        
        squared_diff = diff ** 2
        squared_stats = check_tensor_health(squared_diff, "Squared differences")
        print_tensor_stats(squared_stats)
        
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        recon_loss_stats = check_tensor_health(recon_loss, "MSE Loss")
        has_issues |= print_tensor_stats(recon_loss_stats)
        
        print(f"   📊 Reconstruction Loss: {recon_loss.item():.6f}")
        
    except Exception as e:
        print(f"   ❌ ERROR en reconstruction loss: {e}")
        return None
    
    # Step 2: KL Divergence
    print(f"\n🔬 STEP 2: KL Divergence")
    print(f"   Shapes: mu={mu.shape}, logvar={logvar.shape}")
    
    try:
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        
        # Check mu^2 
        mu_squared = mu.pow(2)
        mu_sq_stats = check_tensor_health(mu_squared, "Mu squared")
        print_tensor_stats(mu_sq_stats)
        
        # Check exp(logvar)
        exp_logvar = logvar.exp()
        exp_stats = check_tensor_health(exp_logvar, "Exp(logvar)")
        exp_issues = print_tensor_stats(exp_stats)
        
        if exp_issues:
            print("   🔍 ANÁLISIS DETALLADO DE EXP(LOGVAR):")
            print(f"       LogVar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
            print(f"       LogVar mean: {logvar.mean().item():.4f}")
            print(f"       LogVar std: {logvar.std().item():.4f}")
            
            # Valores muy altos de logvar pueden causar exp(logvar) = inf
            large_logvar = (logvar > 10).sum().item()
            if large_logvar > 0:
                print(f"       ⚠️ {large_logvar} valores de logvar > 10 (exp > 22026)")
            
            very_large_logvar = (logvar > 20).sum().item()  
            if very_large_logvar > 0:
                print(f"       ⚠️ {very_large_logvar} valores de logvar > 20 (exp > 485M)")
        
        # KL term: 1 + logvar - mu^2 - exp(logvar)
        kl_term = 1 + logvar - mu_squared - exp_logvar
        kl_term_stats = check_tensor_health(kl_term, "KL term (before sum)")
        kl_term_issues = print_tensor_stats(kl_term_stats)
        
        # Sum and scale
        kl_sum = torch.sum(kl_term)
        kl_sum_stats = check_tensor_health(kl_sum, "KL sum")
        print_tensor_stats(kl_sum_stats)
        
        batch_size = target.size(0)
        kl_loss = -0.5 * kl_sum / batch_size
        kl_loss_stats = check_tensor_health(kl_loss, "KL Loss")
        has_issues |= print_tensor_stats(kl_loss_stats)
        
        print(f"   📊 KL Loss: {kl_loss.item():.6f}")
        print(f"   📊 Batch size: {batch_size}")
        
    except Exception as e:
        print(f"   ❌ ERROR en KL divergence: {e}")
        return None
    
    # Step 3: Total loss
    print(f"\n🔬 STEP 3: Total Loss")
    
    try:
        total_loss = recon_loss + beta * kl_loss
        total_stats = check_tensor_health(total_loss, "Total Loss")
        has_issues |= print_tensor_stats(total_stats)
        
        print(f"   📊 Total Loss: {total_loss.item():.6f}")
        print(f"   📊 Beta: {beta}")
        print(f"   📊 Beta * KL: {(beta * kl_loss).item():.6f}")
        
    except Exception as e:
        print(f"   ❌ ERROR en total loss: {e}")
        return None
    
    if has_issues:
        print(f"\n🚨 LOSS COMPUTATION TIENE ISSUES!")
    else:
        print(f"\n✅ LOSS COMPUTATION OK")
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'has_issues': has_issues
    }

def test_vae_outputs_detailed():
    """Test detallado de outputs del VAE antes del loss."""
    print("\n🧪 TEST DETALLADO: VAE Outputs")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Crear modelos con y sin attention
    model_with_attention = PhideusVAE(use_attention=True).to(device)
    model_without_attention = PhideusVAE(use_attention=False).to(device)
    
    # Input sintético pero realista
    batch_size = 4
    x = torch.randn(batch_size, 3, 512).to(device)
    # Normalizar para que sea más realista (histogramas están en [0,1])
    x = torch.sigmoid(x)
    
    print(f"Input shape: {x.shape}")
    print_tensor_stats(check_tensor_health(x, "Input (normalized)"))
    
    # Test VAE sin attention primero
    print(f"\n🔍 Testing VAE WITHOUT Attention:")
    try:
        with torch.no_grad():
            output_no_attn = model_without_attention(x)
        
        recon = output_no_attn['reconstruction']
        mu = output_no_attn['mu']
        logvar = output_no_attn['logvar']
        
        print_tensor_stats(check_tensor_health(recon, "Reconstruction (no attn)"))
        print_tensor_stats(check_tensor_health(mu, "Mu (no attn)"))
        print_tensor_stats(check_tensor_health(logvar, "LogVar (no attn)"))
        
        # Test loss
        loss_result = debug_vae_loss_step_by_step(recon, x, mu, logvar, beta=1.0)
        
        if loss_result and not loss_result['has_issues']:
            print("✅ VAE sin attention: Loss OK")
        else:
            print("❌ VAE sin attention: Loss tiene issues")
            return False
            
    except Exception as e:
        print(f"❌ Error en VAE sin attention: {e}")
        return False
    
    # Test VAE con attention
    print(f"\n🔍 Testing VAE WITH Attention:")
    try:
        with torch.no_grad():
            output_with_attn = model_with_attention(x)
        
        recon = output_with_attn['reconstruction']
        mu = output_with_attn['mu'] 
        logvar = output_with_attn['logvar']
        
        print_tensor_stats(check_tensor_health(recon, "Reconstruction (with attn)"))
        print_tensor_stats(check_tensor_health(mu, "Mu (with attn)"))
        print_tensor_stats(check_tensor_health(logvar, "LogVar (with attn)"))
        
        # Test loss
        loss_result = debug_vae_loss_step_by_step(recon, x, mu, logvar, beta=1.0)
        
        if loss_result and not loss_result['has_issues']:
            print("✅ VAE con attention: Loss OK")
            return True
        else:
            print("❌ VAE con attention: Loss tiene issues")
            return False
            
    except Exception as e:
        print(f"❌ Error en VAE con attention: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Simula un paso de entrenamiento completo."""
    print("\n🧪 TEST: Training Step Simulation")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PhideusVAE(use_attention=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Datos más realistas
    batch_size = 4
    x = torch.randn(batch_size, 3, 512).to(device)
    x = torch.sigmoid(x)  # Normalizar a [0,1] como histogramas reales
    
    print("Simulando training step...")
    
    try:
        # Forward pass
        model.train()
        x.requires_grad_(False)  # Input no necesita gradientes
        
        output = model(x)
        
        # Check outputs antes del loss
        recon_stats = check_tensor_health(output['reconstruction'], "Training Reconstruction")
        mu_stats = check_tensor_health(output['mu'], "Training Mu")
        logvar_stats = check_tensor_health(output['logvar'], "Training LogVar")
        
        issues_before_loss = any([
            print_tensor_stats(recon_stats),
            print_tensor_stats(mu_stats), 
            print_tensor_stats(logvar_stats)
        ])
        
        if issues_before_loss:
            print("❌ Issues ANTES del loss computation")
            return False
        
        # Loss computation con debugging
        loss_result = debug_vae_loss_step_by_step(
            output['reconstruction'], x, output['mu'], output['logvar'], beta=1.0
        )
        
        if not loss_result or loss_result['has_issues']:
            print("❌ Issues EN el loss computation")
            return False
        
        # Backward pass
        total_loss = loss_result['total_loss']
        optimizer.zero_grad()
        total_loss.backward()
        
        print("✅ Backward pass completado")
        
        # Check gradients
        gradient_issues = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats = check_tensor_health(param.grad, f"Grad {name}")
                if grad_stats['has_nan'] or grad_stats['has_inf']:
                    print(f"❌ Gradient issue in {name}")
                    print_tensor_stats(grad_stats)
                    gradient_issues = True
        
        if gradient_issues:
            print("❌ Issues en gradients")
            return False
        
        print("✅ Training step completado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en training step: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecuta debugging detallado del VAE loss."""
    print("🔍 DEBUG VAE LOSS - IDENTIFICACIÓN DE NaN EN LOSS COMPUTATION")
    print("=" * 80)
    
    # Test 1: Outputs detallados
    success1 = test_vae_outputs_detailed()
    
    # Test 2: Training step
    if success1:
        print(f"\n✅ VAE outputs OK, probando training step...")
        success2 = test_training_step()
        
        if success2:
            print(f"\n🎉 VAE FUNCIONA COMPLETAMENTE - ISSUE DEBE SER EN TRAINING LOOP")
        else:
            print(f"\n❌ ISSUE CONFIRMADO EN TRAINING STEP")
    else:
        print(f"\n❌ ISSUE CONFIRMADO EN VAE OUTPUTS")
    
    print("\n" + "=" * 80)
    print("🔍 DEBUG COMPLETADO")

if __name__ == "__main__":
    main()