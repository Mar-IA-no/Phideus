#!/usr/bin/env python3
"""
Debug Linear Attention - Identificar fuente de NaN values

Analiza step-by-step el forward pass del Linear Attention para identificar
exactamente dónde se originan los valores NaN durante el entrenamiento.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path para imports
sys.path.append(str(Path(__file__).parent.parent))
from vae_phideus_v1 import LinearAttention, PhideusVAE

def check_tensor_health(tensor, name="tensor"):
    """Analiza la salud de un tensor."""
    stats = {
        'name': name,
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'min': tensor.min().item() if tensor.numel() > 0 else 0,
        'max': tensor.max().item() if tensor.numel() > 0 else 0,
        'mean': tensor.mean().item() if tensor.numel() > 0 else 0,
        'std': tensor.std().item() if tensor.numel() > 0 else 0,
        'zero_ratio': (tensor == 0).float().mean().item()
    }
    return stats

def print_tensor_stats(stats):
    """Imprime estadísticas de tensor de forma legible."""
    print(f"\n📊 {stats['name']}: {stats['shape']}")
    print(f"   NaN: {stats['has_nan']:5} | Inf: {stats['has_inf']:5} | Dtype: {stats['dtype']}")
    print(f"   Range: [{stats['min']:8.4f}, {stats['max']:8.4f}] | Mean: {stats['mean']:8.4f}")
    print(f"   Std: {stats['std']:8.4f} | Zero ratio: {stats['zero_ratio']:6.2%}")
    
    if stats['has_nan'] or stats['has_inf']:
        print(f"   ⚠️ PROBLEMA DETECTADO: {'NaN' if stats['has_nan'] else ''}{'Inf' if stats['has_inf'] else ''}")
    
    return stats['has_nan'] or stats['has_inf']

class DebugLinearAttention(nn.Module):
    """Linear Attention con debugging step-by-step."""
    
    def __init__(self, d_model: int, n_heads: int = 4, nb_features: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.nb_features = nb_features
        
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        
        # Inicialización más cuidadosa
        self.register_buffer('projection', torch.randn(nb_features, d_model // n_heads) * 0.1)
        
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        if debug:
            print("\n" + "="*60)
            print("🔍 DEBUG LINEAR ATTENTION FORWARD PASS")
            print("="*60)
        
        b, n, d = x.shape
        h = self.n_heads
        
        if debug:
            has_issues = print_tensor_stats(check_tensor_health(x, "Input x"))
        
        # Step 1: Linear projections
        q = self.to_q(x)
        k = self.to_k(x) 
        v = self.to_v(x)
        
        if debug:
            has_issues |= print_tensor_stats(check_tensor_health(q, "Q projection"))
            has_issues |= print_tensor_stats(check_tensor_health(k, "K projection"))
            has_issues |= print_tensor_stats(check_tensor_health(v, "V projection"))
        
        # Step 2: Reshape for multi-head
        try:
            q = q.reshape(b, n, h, -1)
            k = k.reshape(b, n, h, -1) 
            v = v.reshape(b, n, h, -1)
        except Exception as e:
            if debug:
                print(f"⚠️ ERROR en reshape: {e}")
                print(f"Input shape: {x.shape}, d_model: {d}, n_heads: {h}")
            raise
        
        if debug:
            print_tensor_stats(check_tensor_health(q, "Q reshaped"))
            print_tensor_stats(check_tensor_health(k, "K reshaped"))
            print_tensor_stats(check_tensor_health(v, "V reshaped"))
        
        # Step 3: Kernel feature map (ELU + 1)
        q_pre_kernel = q.clone()
        k_pre_kernel = k.clone()
        
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        if debug:
            has_issues |= print_tensor_stats(check_tensor_health(q, "Q after ELU+1"))
            has_issues |= print_tensor_stats(check_tensor_health(k, "K after ELU+1"))
            
            # Análisis específico del kernel feature map
            print(f"\n🔬 ANÁLISIS KERNEL FEATURE MAP:")
            print(f"Q pre-kernel range: [{q_pre_kernel.min():.4f}, {q_pre_kernel.max():.4f}]")
            print(f"K pre-kernel range: [{k_pre_kernel.min():.4f}, {k_pre_kernel.max():.4f}]")
            print(f"Q post-kernel range: [{q.min():.4f}, {q.max():.4f}]")  # Should be [1, +inf]
            print(f"K post-kernel range: [{k.min():.4f}, {k.max():.4f}]")  # Should be [1, +inf]
        
        # Step 4: Linear attention computation
        # kv = torch.einsum('bnhd,bnhf->bhdf', k, v)
        try:
            kv = torch.einsum('bnhd,bnhf->bhdf', k, v)
        except Exception as e:
            if debug:
                print(f"⚠️ ERROR en KV einsum: {e}")
                print(f"K shape: {k.shape}, V shape: {v.shape}")
            raise
            
        if debug:
            has_issues |= print_tensor_stats(check_tensor_health(kv, "KV matrix"))
            
            # Análisis específico de KV
            kv_norm = torch.norm(kv, dim=-1).mean()
            print(f"🔬 KV matrix norm mean: {kv_norm:.6f}")
        
        # Step 5: Final attention output
        try:
            out = torch.einsum('bnhd,bhdf->bnhf', q, kv)
        except Exception as e:
            if debug:
                print(f"⚠️ ERROR en Q-KV einsum: {e}")
                print(f"Q shape: {q.shape}, KV shape: {kv.shape}")
            raise
            
        if debug:
            has_issues |= print_tensor_stats(check_tensor_health(out, "Attention output"))
        
        # Step 6: Reshape back
        out = out.reshape(b, n, d)
        
        if debug:
            print_tensor_stats(check_tensor_health(out, "Output reshaped"))
        
        # Step 7: Final linear projection
        final_out = self.to_out(out)
        
        if debug:
            has_issues |= print_tensor_stats(check_tensor_health(final_out, "Final output"))
            
            if has_issues:
                print("\n🚨 ISSUES DETECTADOS EN ATTENTION!")
            else:
                print("\n✅ Attention forward pass completado sin issues")
        
        return final_out

def test_attention_isolated():
    """Test Linear Attention de forma aislada."""
    print("\n🧪 TEST 1: Linear Attention Aislada")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Configuración similar a VAE
    d_model = 256
    seq_len = 32  # Empezar pequeño
    batch_size = 4
    
    attention = DebugLinearAttention(d_model=d_model, n_heads=4, nb_features=64).to(device)
    
    # Input sintético
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass con debugging
    try:
        with torch.no_grad():
            output = attention(x, debug=True)
        print("\n✅ Forward pass exitoso sin gradientes")
        
        # Test con gradientes
        x.requires_grad_(True)
        output = attention(x, debug=False)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        print("✅ Backward pass exitoso")
        
        # Verificar gradientes
        grad_stats = check_tensor_health(x.grad, "Input gradients")
        has_grad_issues = print_tensor_stats(grad_stats)
        
        if not has_grad_issues:
            print("✅ Gradientes saludables")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en attention test: {e}")
        return False

def test_attention_scaling():
    """Test con diferentes tamaños de secuencia."""
    print("\n🧪 TEST 2: Scaling de Secuencia")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 256
    batch_size = 2
    
    attention = DebugLinearAttention(d_model=d_model, n_heads=4, nb_features=64).to(device)
    
    sequence_lengths = [16, 32, 64, 128, 256, 512]
    
    for seq_len in sequence_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        try:
            x = torch.randn(batch_size, seq_len, d_model).to(device)
            x.requires_grad_(True)
            
            output = attention(x, debug=(seq_len >= 256))
            loss = output.sum()
            loss.backward()
            
            # Check gradients
            if x.grad is not None:
                grad_stats = check_tensor_health(x.grad, f"Gradients seq_len={seq_len}")
                if grad_stats['has_nan'] or grad_stats['has_inf']:
                    print(f"❌ Gradient issues at seq_len={seq_len}")
                    print_tensor_stats(grad_stats)
                    return seq_len
                else:
                    print(f"✅ seq_len={seq_len} OK")
            
        except Exception as e:
            print(f"❌ Error at seq_len={seq_len}: {e}")
            return seq_len
    
    print("✅ Todos los tamaños de secuencia funcionan")
    return None

def test_in_vae_context():
    """Test Linear Attention dentro del contexto completo del VAE."""
    print("\n🧪 TEST 3: Attention en Contexto VAE")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Crear VAE con attention
    model = PhideusVAE(use_attention=True).to(device)
    
    # Input realista
    batch_size = 4
    x = torch.randn(batch_size, 3, 512).to(device)  # Histogramas reales
    
    print(f"VAE input shape: {x.shape}")
    
    try:
        # Forward pass
        x.requires_grad_(True)
        output = model(x)
        
        print("✅ VAE forward pass with attention successful")
        
        # Test loss computation
        from vae_phideus_v1 import vae_loss
        loss_dict = vae_loss(output['reconstruction'], x, 
                            output['mu'], output['logvar'], beta=1.0)
        
        total_loss = loss_dict['total_loss']
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"❌ Loss es NaN/Inf: {total_loss}")
            return False
        
        print(f"✅ Loss computation OK: {total_loss:.4f}")
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients en attention layer
        for name, param in model.named_parameters():
            if 'attention' in name and param.grad is not None:
                grad_stats = check_tensor_health(param.grad, f"Grad {name}")
                if grad_stats['has_nan'] or grad_stats['has_inf']:
                    print(f"❌ Gradient issue in {name}")
                    print_tensor_stats(grad_stats)
                    return False
        
        print("✅ VAE con attention funciona completamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en VAE test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecuta todos los tests de debugging."""
    print("🔍 DEBUG LINEAR ATTENTION - IDENTIFICACIÓN DE NaN VALUES")
    print("=" * 70)
    
    # Test 1: Attention aislada
    success1 = test_attention_isolated()
    
    # Test 2: Scaling
    if success1:
        failure_length = test_attention_scaling()
        if failure_length:
            print(f"\n⚠️ ATTENTION FALLA EN SECUENCIAS >= {failure_length}")
        else:
            print("\n✅ ATTENTION ESTABLE EN TODOS LOS TAMAÑOS")
    
    # Test 3: Contexto VAE
    if success1:
        success3 = test_in_vae_context()
        
        if success3:
            print("\n🎉 ATTENTION FUNCIONA EN VAE - ISSUE PUEDE SER EN TRAINING")
        else:
            print("\n❌ ISSUE CONFIRMADO EN CONTEXTO VAE")
    
    print("\n" + "=" * 70)
    print("🔍 DEBUG COMPLETADO")
    
    if success1:
        print("💡 SIGUIENTE PASO: Test durante training con datos reales")
    else:
        print("💡 SIGUIENTE PASO: Fix issues básicos en attention mechanism")

if __name__ == "__main__":
    main()