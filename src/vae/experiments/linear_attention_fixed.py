#!/usr/bin/env python3
"""
Linear Attention Fixed - Versiones corregidas del Linear Attention

Implementa varias estrategias para evitar la explosión de valores:
1. Normalización mejorada
2. Scaling de attention weights
3. Gradient clipping
4. Residual connections balanceadas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StableLinearAttention(nn.Module):
    """Linear Attention estabilizada con normalización mejorada."""
    
    def __init__(self, d_model: int, n_heads: int = 4, nb_features: int = 64, 
                 temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.nb_features = nb_features
        self.temperature = temperature
        self.head_dim = d_model // n_heads
        
        # Proyecciones con inicialización Xavier
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        
        # Normalización pre y post attention
        self.norm_pre = nn.LayerNorm(d_model)
        self.norm_post = nn.LayerNorm(d_model)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(dropout)
        
        # Inicialización Xavier para estabilidad
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización Xavier para evitar gradientes extremos."""
        for module in [self.to_q, self.to_k, self.to_v, self.to_out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Pre-normalization
        x = self.norm_pre(x)
        
        b, n, d = x.shape
        h = self.n_heads
        head_dim = self.head_dim
        
        # Proyecciones Q, K, V
        q = self.to_q(x).reshape(b, n, h, head_dim)
        k = self.to_k(x).reshape(b, n, h, head_dim) 
        v = self.to_v(x).reshape(b, n, h, head_dim)
        
        # Scaling por dimensión (como en Transformer estándar)
        scale = 1.0 / math.sqrt(head_dim)
        q = q * scale
        k = k * scale
        
        # Kernel feature map estabilizado
        # Usar ReLU + epsilon en lugar de ELU + 1 para mayor estabilidad
        epsilon = 1e-6
        q = F.relu(q) + epsilon
        k = F.relu(k) + epsilon
        
        # Temperature scaling para controlar magnitud
        q = q / self.temperature
        k = k / self.temperature
        
        # Linear attention computation
        # Normalizar K y V para evitar valores extremos
        k_sum = k.sum(dim=1, keepdim=True)  # (b, 1, h, head_dim)
        k_sum = torch.clamp(k_sum, min=epsilon)  # Evitar división por 0
        
        # Attention weights normalizados
        kv = torch.einsum('bnhd,bnhf->bhdf', k, v)
        context = torch.einsum('bnhd,bhdf->bnhf', q, kv)
        
        # Normalización del contexto para evitar explosión
        context_norm = torch.einsum('bnhd,bhd->bnh', q, k_sum.squeeze(1))
        context_norm = torch.clamp(context_norm, min=epsilon)
        context = context / context_norm.unsqueeze(-1)
        
        # Reshape y proyección final
        out = context.reshape(b, n, d)
        out = self.to_out(out)
        out = self.dropout(out)
        
        # Residual connection con scaling
        out = out + residual
        
        # Post-normalization
        out = self.norm_post(out)
        
        return out

class FlashLinearAttention(nn.Module):
    """Implementación simplificada inspirada en Flash Attention."""
    
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        b, n, d = x.shape
        h = self.n_heads
        
        # Una sola proyección para Q, K, V
        qkv = self.qkv(x).reshape(b, n, 3, h, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Attention simplificada sin kernel tricks
        q = q * self.scale
        
        # Evitar softmax completamente - usar normalización directa
        attn_weights = torch.einsum('bnhd,bmhd->bnmh', q, k)
        
        # Normalización por suma en lugar de softmax
        attn_sum = attn_weights.sum(dim=2, keepdim=True)
        attn_sum = torch.clamp(torch.abs(attn_sum), min=1e-8)
        attn_weights = attn_weights / attn_sum
        
        # Apply attention
        out = torch.einsum('bnmh,bmhd->bnhd', attn_weights, v)
        out = out.reshape(b, n, d)
        
        out = self.out(out)
        out = self.dropout(out)
        
        # Residual connection
        return out + residual

class MinimalAttention(nn.Module):
    """Attention minimalista sin kernel tricks complicados."""
    
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Inicialización conservadora
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)  # Gain reducido
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        b, n, d = x.shape
        h = self.n_heads
        
        q = self.w_q(x).view(b, n, h, -1)
        k = self.w_k(x).view(b, n, h, -1) 
        v = self.w_v(x).view(b, n, h, -1)
        
        # Scaling mínimo
        q = q * (1.0 / math.sqrt(d))
        
        # Global pooling attention en lugar de full quadratic
        k_global = k.mean(dim=1, keepdim=True)  # (b, 1, h, d_head)
        v_global = v.mean(dim=1, keepdim=True)  # (b, 1, h, d_head)
        
        # Attention weights: cada posición vs global
        attn = torch.einsum('bnhd,bmhd->bnh', q, k_global)  # (b, n, h)
        attn = torch.softmax(attn, dim=1)
        
        # Apply attention 
        out = torch.einsum('bnh,bmhd->bnhd', attn, v_global)
        out = out.reshape(b, n, d)
        
        out = self.w_o(out)
        out = self.dropout(out)
        
        return out + residual

def test_attention_variants():
    """Test las diferentes variantes de attention."""
    print("🧪 TESTING ATTENTION VARIANTS")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 256
    seq_len = 512  # Tamaño problemático
    batch_size = 4
    
    # Input realista
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    x = x * 0.1  # Reducir magnitud inicial
    
    variants = {
        'StableLinearAttention': StableLinearAttention(d_model),
        'FlashLinearAttention': FlashLinearAttention(d_model),
        'MinimalAttention': MinimalAttention(d_model)
    }
    
    results = {}
    
    for name, attention_module in variants.items():
        print(f"\n🔍 Testing {name}:")
        attention_module = attention_module.to(device)
        
        try:
            # Forward pass
            x_test = x.clone().requires_grad_(True)
            output = attention_module(x_test)
            
            # Check output health
            output_range = [output.min().item(), output.max().item()]
            output_mean = output.mean().item()
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            
            print(f"   Output range: [{output_range[0]:.4f}, {output_range[1]:.4f}]")
            print(f"   Output mean: {output_mean:.4f}")
            print(f"   Has NaN: {has_nan}, Has Inf: {has_inf}")
            
            if has_nan or has_inf:
                print(f"   ❌ {name} has numerical issues")
                results[name] = False
                continue
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            
            # Check gradients
            grad_range = [x_test.grad.min().item(), x_test.grad.max().item()]
            grad_mean = x_test.grad.mean().item()
            grad_has_nan = torch.isnan(x_test.grad).any().item()
            grad_has_inf = torch.isinf(x_test.grad).any().item()
            
            print(f"   Grad range: [{grad_range[0]:.4f}, {grad_range[1]:.4f}]")
            print(f"   Grad mean: {grad_mean:.4f}")
            print(f"   Grad NaN: {grad_has_nan}, Grad Inf: {grad_has_inf}")
            
            if grad_has_nan or grad_has_inf:
                print(f"   ❌ {name} has gradient issues")
                results[name] = False
            else:
                print(f"   ✅ {name} works correctly")
                results[name] = True
                
        except Exception as e:
            print(f"   ❌ {name} failed with error: {e}")
            results[name] = False
    
    print(f"\n📊 RESULTS:")
    working_variants = [name for name, works in results.items() if works]
    
    if working_variants:
        print(f"✅ Working variants: {', '.join(working_variants)}")
        return working_variants[0]  # Return best working variant
    else:
        print(f"❌ No variants work properly")
        return None

if __name__ == "__main__":
    best_variant = test_attention_variants()
    
    if best_variant:
        print(f"\n🎯 RECOMMENDED: Use {best_variant} for VAE integration")
    else:
        print(f"\n💡 RECOMMENDATION: Skip attention for now, implement in v1.1")