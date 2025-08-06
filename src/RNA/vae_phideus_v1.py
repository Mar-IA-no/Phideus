#!/usr/bin/env python3
"""
VAE Phideus v1.0 - Variational Autoencoder para Histogramas Armónicos

Arquitectura VAE + CNN 1D optimizada para RTX 3090 según hoja de ruta:
- Input: (batch, 512, 3) - Histogramas enriquecidos
- Encoder: CNN 1D dilatada + Linear Attention opcional
- Latent: 128D (μ, σ)  
- Decoder: CNN Transpose + skip connections
- Contrastive: MoCo-v3 o BYOL

Hardware target: RTX 3090 (24GB)
Batch size: 256, FP16, Adam8bit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class DilatedCNNBlock(nn.Module):
    """Bloque CNN 1D con dilatación y conexión residual."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=dilation * (kernel_size // 2), dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Conexión residual si dimensiones coinciden
        self.residual = (in_channels == out_channels)
        if not self.residual:
            self.proj = nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Conexión residual
        if self.residual:
            out = out + identity
        elif hasattr(self, 'proj'):
            out = out + self.proj(identity)
            
        return out


class LinearAttention(nn.Module):
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


class PhideusVAE(nn.Module):
    """VAE principal con arquitectura optimizada para histogramas armónicos."""
    
    def __init__(self, input_bins: int = 512, input_channels: int = 3, 
                 latent_dim: int = 128, use_attention: bool = False):
        super().__init__()
        
        self.input_bins = input_bins
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        
        # Encoder CNN 1D con dilataciones progresivas
        encoder_channels = [input_channels, 64, 128, 256, 256, 256, 256]
        dilations = [1, 2, 4, 8, 16, 32]
        
        encoder_layers = []
        for i in range(len(encoder_channels) - 1):
            dilation = dilations[min(i, len(dilations) - 1)]
            encoder_layers.append(
                DilatedCNNBlock(encoder_channels[i], encoder_channels[i + 1], 
                              dilation=dilation)
            )
            # Downsampling cada 2 bloques
            if i > 0 and i % 2 == 0:
                encoder_layers.append(nn.MaxPool1d(2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Linear Attention opcional
        if use_attention:
            self.attention = LinearAttention(256, n_heads=4, nb_features=64)
        
        # Calcular tamaño después de CNN
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_bins)
            encoded_size = self._get_conv_output_size(dummy)
            self.encoded_shape = self._get_conv_output_shape(dummy)
        
        # Capas latentes (μ y σ)
        self.fc_mu = nn.Linear(encoded_size, latent_dim)
        self.fc_logvar = nn.Linear(encoded_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, encoded_size)
        
        # Decoder CNN Transpose (simétrico al encoder)
        decoder_channels = encoder_channels[::-1]  # Reverse
        decoder_layers = []
        
        for i in range(len(decoder_channels) - 1):
            # Upsampling antes de conv si corresponde
            if i > 0 and i % 2 == 0:
                decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
            dilation = dilations[max(0, len(dilations) - 1 - i)]
            decoder_layers.append(
                DilatedCNNBlock(decoder_channels[i], decoder_channels[i + 1],
                              dilation=dilation)
            )
        
        # Capa final para reconstrucción
        decoder_layers.append(nn.Conv1d(input_channels, input_channels, 1))
        decoder_layers.append(nn.Sigmoid())  # Salida [0,1] para histogramas
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_conv_output_size(self, x: torch.Tensor) -> int:
        """Calcula el tamaño de salida del encoder."""
        x = self.encoder(x)
        if self.use_attention:
            x = x.transpose(1, 2)  # (B, L, C)
            x = self.attention(x)
            x = x.transpose(1, 2)  # (B, C, L)
        return x.numel() // x.size(0)
    
    def _get_conv_output_shape(self, x: torch.Tensor) -> torch.Size:
        """Calcula la forma de salida del encoder."""
        x = self.encoder(x)
        if self.use_attention:
            x = x.transpose(1, 2)  # (B, L, C)
            x = self.attention(x)
            x = x.transpose(1, 2)  # (B, C, L)
        return x.shape[1:]  # Sin batch dimension
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoding a espacio latente."""
        x = self.encoder(x)
        
        if self.use_attention:
            x = x.transpose(1, 2)  # (B, L, C)
            x = self.attention(x)
            x = x.transpose(1, 2)  # (B, C, L)
        
        x = x.flatten(1)  # (B, features)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoding desde espacio latente."""
        x = self.fc_decode(z)
        
        # Reshape usando la forma calculada del encoder
        batch_size = z.size(0)
        x = x.view(batch_size, *self.encoded_shape)
        x = self.decoder(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass completo."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return {
            'reconstruction': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }


def vae_loss(recon: torch.Tensor, target: torch.Tensor, 
             mu: torch.Tensor, logvar: torch.Tensor, 
             beta: float = 1.0) -> Dict[str, torch.Tensor]:
    """Loss VAE: reconstrucción + KL divergence."""
    
    # Reconstruction loss (MSE para histogramas)
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    
    # Total loss con β-VAE
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'beta': beta
    }


class PhideusDataset(Dataset):
    """Dataset para histogramas enriquecidos de Phideus."""
    
    def __init__(self, json_path: Path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.files = list(self.data.keys())
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        file = self.files[idx]
        hist_enriched = self.data[file]['ratio_hist_enriched']
        
        # Convertir (512, 3) → (3, 512) para CNN 1D
        hist = torch.tensor(hist_enriched, dtype=torch.float32).transpose(0, 1)
        
        return hist


def create_model_and_dataset(json_path: Path, device: str = 'cuda', 
                           use_attention: bool = False) -> Tuple[PhideusVAE, PhideusDataset]:
    """Factory para modelo y dataset."""
    
    dataset = PhideusDataset(json_path)
    model = PhideusVAE(input_bins=512, input_channels=3, latent_dim=128, 
                       use_attention=use_attention)
    
    model = model.to(device)
    
    return model, dataset


if __name__ == "__main__":
    # Test básico de la arquitectura
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Testing VAE architecture on {device}")
    
    # Crear modelo
    model = PhideusVAE(use_attention=False).to(device)
    
    # Test con batch sintético
    batch_size = 4
    x = torch.randn(batch_size, 3, 512).to(device)
    
    print(f"📊 Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"🔄 Reconstruction shape: {output['reconstruction'].shape}")
    print(f"🧠 Latent μ shape: {output['mu'].shape}")
    print(f"🧠 Latent σ shape: {output['logvar'].shape}")
    
    # Test loss
    loss_dict = vae_loss(output['reconstruction'], x, 
                        output['mu'], output['logvar'])
    
    print(f"📉 Total loss: {loss_dict['total_loss']:.4f}")
    print(f"📉 Recon loss: {loss_dict['recon_loss']:.4f}")
    print(f"📉 KL loss: {loss_dict['kl_loss']:.4f}")
    
    print("✅ VAE architecture test completed!")