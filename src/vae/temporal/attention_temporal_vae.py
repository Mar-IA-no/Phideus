#!/usr/bin/env python3
"""
Attention-Based Temporal VAE - Implementación completa
Integra FrameEncoder + TemporalSelfAttention + TemporalAggregator + VAE Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Agregar path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal.frame_encoder import FrameEncoder, RTX3090OptimizedFrameEncoder
from temporal.temporal_attention import TemporalSelfAttention, RTX3090OptimizedAttention  
from temporal.temporal_aggregator import TemporalAggregator, MemoryEfficientAggregator

class AttentionBasedTemporalVAE(nn.Module):
    """
    VAE temporal completo con self-attention para análisis harmónico temporal
    
    Pipeline:
    1. Secuencia histogramas → FrameEncoder → embeddings temporales
    2. Embeddings → TemporalSelfAttention → attended sequence  
    3. Attended sequence → TemporalAggregator → mu, logvar
    4. Latent z → VAE Decoder → histograma reconstruido
    """
    def __init__(self,
                 input_channels=3,
                 input_bins=512,
                 embed_dim=128,
                 latent_dim=128,
                 num_attention_heads=8,
                 max_sequence_length=120,
                 decoder_architecture='simple'):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length
        
        # Componentes principales
        self.frame_encoder = FrameEncoder(
            input_channels=input_channels,
            input_bins=input_bins, 
            embed_dim=embed_dim
        )
        
        self.temporal_attention = TemporalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            max_sequence_length=max_sequence_length
        )
        
        self.temporal_aggregator = TemporalAggregator(
            embed_dim=embed_dim,
            latent_dim=latent_dim
        )
        
        # Decoder (reutilizar arquitectura existente o crear simple)
        if decoder_architecture == 'simple':
            self.decoder = SimpleVAEDecoder(latent_dim, input_bins, input_channels)
        else:
            # Aquí se podría cargar el decoder del VAE existente
            self.decoder = self._load_existing_vae_decoder()
    
    def _load_existing_vae_decoder(self):
        """
        Cargar decoder del VAE existente (placeholder)
        En implementación real, cargaría el decoder entrenado
        """
        # Por ahora, usar decoder simple
        return SimpleVAEDecoder(self.latent_dim, 512, 3)
    
    def encode(self, histogram_sequence):
        """
        Encoding temporal completo
        
        Args:
            histogram_sequence: (batch_size, seq_len, 512, 3)
            
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim) 
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = histogram_sequence.shape[:2]
        
        # Truncar secuencia si es muy larga
        if seq_len > self.max_sequence_length:
            histogram_sequence = histogram_sequence[:, :self.max_sequence_length]
            seq_len = self.max_sequence_length
        
        # 1. Frame encoding: procesar cada histograma individualmente
        frame_embeddings = []
        for t in range(seq_len):
            frame_emb = self.frame_encoder(histogram_sequence[:, t])
            frame_embeddings.append(frame_emb)
        
        # Stack embeddings temporales
        temporal_sequence = torch.stack(frame_embeddings, dim=1)
        # Shape: (batch_size, seq_len, embed_dim)
        
        # 2. Temporal attention: capturar dependencias temporales
        attended_sequence, attention_weights = self.temporal_attention(temporal_sequence)
        
        # 3. Aggregation: convertir a representación fija
        mu, logvar = self.temporal_aggregator(attended_sequence)
        
        return mu, logvar, attention_weights
    
    def decode(self, z):
        """
        Decoding VAE estándar
        
        Args:
            z: (batch_size, latent_dim)
            
        Returns:
            reconstructed: (batch_size, 512, 3)
        """
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick del VAE
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, histogram_sequence):
        """
        Forward pass completo del Temporal VAE
        
        Args:
            histogram_sequence: (batch_size, seq_len, 512, 3)
            
        Returns:
            reconstructed: (batch_size, 512, 3) - Histograma reconstruido
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # Encoding temporal
        mu, logvar, attention_weights = self.encode(histogram_sequence)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        reconstructed = self.decode(z)
        
        return reconstructed, mu, logvar, attention_weights
    
    def get_temporal_summary(self, histogram_sequence):
        """
        Extraer resumen temporal interpretable
        
        Returns:
            Dict con información temporal procesada
        """
        with torch.no_grad():
            mu, logvar, attention_weights = self.encode(histogram_sequence)
            
            # Promedio de attention weights across heads
            avg_attention = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)
            
            # Identificar momentos más "influyentes"
            influence_scores = avg_attention.sum(dim=-1)  # (batch, seq_len)
            
            # Detectar pares de momentos con alta correlación
            batch_size, seq_len = avg_attention.shape[:2]
            correlations = []
            
            for b in range(batch_size):
                batch_correlations = []
                for i in range(seq_len):
                    for j in range(i+1, seq_len):
                        corr = avg_attention[b, i, j].item()
                        if corr > 0.5:  # Threshold para correlación significativa
                            batch_correlations.append((i, j, corr))
                correlations.append(batch_correlations)
            
            return {
                'latent_mean': mu,
                'latent_logvar': logvar,
                'attention_patterns': avg_attention,
                'influence_scores': influence_scores,
                'temporal_correlations': correlations
            }

class SimpleVAEDecoder(nn.Module):
    """
    Decoder simple para el Temporal VAE
    En implementación final se usaría el decoder del VAE existente
    """
    def __init__(self, latent_dim, output_bins, output_channels):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_bins = output_bins
        self.output_channels = output_channels
        
        # Expandir latente a feature maps
        self.latent_expansion = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128 * (output_bins // 4)),
            nn.ReLU(inplace=True)
        )
        
        # Transpose convolutions para reconstrucción
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Histogramas están normalizados [0,1]
        )
    
    def forward(self, z):
        """
        Args:
            z: (batch_size, latent_dim)
        Returns:
            reconstructed: (batch_size, 512, 3)
        """
        batch_size = z.shape[0]
        
        # Expandir latente
        x = self.latent_expansion(z)  # (batch_size, 128 * bins//4)
        x = x.view(batch_size, 128, self.output_bins // 4)
        
        # Transpose convolutions
        x = self.decoder_conv(x)  # (batch_size, 3, 512)
        
        # Transpose back
        reconstructed = x.transpose(1, 2)  # (batch_size, 512, 3)
        
        return reconstructed

class RTX3090OptimizedTemporalVAE(AttentionBasedTemporalVAE):
    """
    Versión optimizada específicamente para RTX 3090
    Con memory management y configuración adaptada
    """
    def __init__(self, **kwargs):
        # Configuración optimizada RTX 3090
        kwargs.update({
            'num_attention_heads': 4,  # Reducido de 8
            'max_sequence_length': 60,  # Máximo viable
            'embed_dim': 128
        })
        
        super().__init__(**kwargs)
        
        # Reemplazar con componentes optimizados
        self.frame_encoder = RTX3090OptimizedFrameEncoder()
        self.temporal_attention = RTX3090OptimizedAttention(
            embed_dim=kwargs['embed_dim'],
            num_heads=kwargs['num_attention_heads']
        )
        self.temporal_aggregator = MemoryEfficientAggregator(
            embed_dim=kwargs['embed_dim'],
            latent_dim=kwargs['latent_dim']
        )
        
        # Configuración memory-efficient
        self.gradient_checkpointing = True
        self.mixed_precision = True
    
    def encode(self, histogram_sequence):
        """Encoding optimizado con memory management"""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                super().encode, histogram_sequence, use_reentrant=False
            )
        else:
            return super().encode(histogram_sequence)

if __name__ == "__main__":
    # Test completo del Temporal VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Crear modelo optimizado para RTX 3090
    model = RTX3090OptimizedTemporalVAE().to(device)
    
    # Test input: batch de secuencias temporales
    batch_size, seq_len = 2, 30
    histogram_sequence = torch.randn(batch_size, seq_len, 512, 3).to(device)
    
    print(f"Testing Temporal VAE with input: {histogram_sequence.shape}")
    
    with torch.no_grad():
        # Forward pass completo
        reconstructed, mu, logvar, attention_weights = model(histogram_sequence)
        
        print(f"✅ Forward pass successful:")
        print(f"  Input: {histogram_sequence.shape}")
        print(f"  Reconstructed: {reconstructed.shape}")
        print(f"  Latent mu: {mu.shape}")
        print(f"  Latent logvar: {logvar.shape}")
        print(f"  Attention weights: {attention_weights.shape}")
        
        # Test temporal summary
        summary = model.get_temporal_summary(histogram_sequence)
        print(f"✅ Temporal summary generated:")
        print(f"  Influence scores shape: {summary['influence_scores'].shape}")
        print(f"  Correlations found: {len(summary['temporal_correlations'][0])}")
        
        # Parameters y memoria
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU Memory used: {memory_used:.3f} GB")
            
    print("🚀 Attention-Based Temporal VAE implementation complete!")