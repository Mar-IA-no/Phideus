#!/usr/bin/env python3
"""
Temporal Aggregator para Attention-Based Temporal VAE
Agrega información temporal en representación fija para VAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAggregator(nn.Module):
    """
    Agrega secuencias temporales attended en representación fija para VAE
    Combina múltiples estrategias de agregación para robustez
    """
    def __init__(self, 
                 embed_dim=128, 
                 latent_dim=128,
                 aggregation_strategy='hybrid'):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.aggregation_strategy = aggregation_strategy
        
        # Estrategia 1: Adaptive average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Estrategia 2: Attention pooling con learnable query
        self.attention_pool = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )
        self.pooling_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Estrategia 3: Convolutional temporal reduction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Combinar estrategias
        if aggregation_strategy == 'hybrid':
            fusion_dim = embed_dim * 3  # avg + attention + conv
        else:
            fusion_dim = embed_dim
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Proyección a parámetros VAE (mu y logvar)
        self.mu_projection = nn.Linear(embed_dim, latent_dim)
        self.logvar_projection = nn.Linear(embed_dim, latent_dim)
        
        # Inicialización
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización cuidadosa para estabilidad VAE"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
        # Inicializar logvar projection con valores pequeños para estabilidad
        nn.init.xavier_uniform_(self.logvar_projection.weight, gain=0.1)
        
    def forward(self, attended_sequence):
        """
        Agregación temporal para VAE
        
        Args:
            attended_sequence: (batch_size, seq_len, embed_dim)
            
        Returns:
            mu: (batch_size, latent_dim) - Media del VAE
            logvar: (batch_size, latent_dim) - Log varianza del VAE
        """
        batch_size, seq_len, embed_dim = attended_sequence.shape
        
        if self.aggregation_strategy == 'average':
            aggregated = self._average_pooling(attended_sequence)
        elif self.aggregation_strategy == 'attention':
            aggregated = self._attention_pooling(attended_sequence)
        elif self.aggregation_strategy == 'conv':
            aggregated = self._conv_pooling(attended_sequence)
        elif self.aggregation_strategy == 'hybrid':
            aggregated = self._hybrid_aggregation(attended_sequence)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
        
        # Fusión final
        fused = self.fusion_layer(aggregated)
        
        # Proyectar a parámetros VAE
        mu = self.mu_projection(fused)
        logvar = self.logvar_projection(fused)
        
        # Clamp logvar para estabilidad numérica
        logvar = torch.clamp(logvar, -10, 10)
        
        return mu, logvar
    
    def _average_pooling(self, sequence):
        """Promedio simple sobre dimensión temporal"""
        return sequence.mean(dim=1)  # (batch_size, embed_dim)
    
    def _attention_pooling(self, sequence):
        """Attention pooling con query aprendible"""
        batch_size = sequence.shape[0]
        
        # Expandir query para batch
        query = self.pooling_query.expand(batch_size, -1, -1)
        
        # Attention pooling
        pooled, _ = self.attention_pool(query, sequence, sequence)
        
        return pooled.squeeze(1)  # (batch_size, embed_dim)
    
    def _conv_pooling(self, sequence):
        """Convolutional temporal pooling"""
        # Transpose para conv1d
        x = sequence.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        pooled = self.temporal_conv(x).squeeze(-1)  # (batch_size, embed_dim)
        return pooled
    
    def _hybrid_aggregation(self, sequence):
        """Combinar múltiples estrategias de agregación"""
        avg_pooled = self._average_pooling(sequence)
        att_pooled = self._attention_pooling(sequence)
        conv_pooled = self._conv_pooling(sequence)
        
        # Concatenar estrategias
        combined = torch.cat([avg_pooled, att_pooled, conv_pooled], dim=1)
        
        return combined
    
    def get_aggregation_weights(self, attended_sequence):
        """
        Obtener weights de atención para interpretabilidad
        Solo funciona con attention pooling
        """
        if self.aggregation_strategy not in ['attention', 'hybrid']:
            return None
            
        batch_size = attended_sequence.shape[0]
        query = self.pooling_query.expand(batch_size, -1, -1)
        
        with torch.no_grad():
            _, attention_weights = self.attention_pool(
                query, attended_sequence, attended_sequence
            )
        
        return attention_weights.squeeze(1)  # (batch_size, seq_len)

class MemoryEfficientAggregator(TemporalAggregator):
    """
    Versión optimizada para memoria (RTX 3090)
    """
    def __init__(self, **kwargs):
        # Forzar estrategia más eficiente en memoria
        kwargs['aggregation_strategy'] = 'average'  # Más simple
        super().__init__(**kwargs)
        
        # Reducir complejidad
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, attended_sequence):
        """Forward pass optimizado para memoria"""
        # Usar gradient checkpointing si está en training
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                super().forward, attended_sequence, use_reentrant=False
            )
        else:
            return super().forward(attended_sequence)

if __name__ == "__main__":
    # Test del TemporalAggregator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear modelo
    aggregator = TemporalAggregator().to(device)
    
    # Test input: secuencia attended
    batch_size, seq_len, embed_dim = 2, 30, 128
    attended_sequence = torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    with torch.no_grad():
        mu, logvar = aggregator(attended_sequence)
        
        print(f"✅ TemporalAggregator test:")
        print(f"Input: {attended_sequence.shape}")
        print(f"Mu: {mu.shape}")
        print(f"Logvar: {logvar.shape}")
        print(f"Logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")
        print(f"Parameters: {sum(p.numel() for p in aggregator.parameters()):,}")
        
        # Test reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        print(f"Latent z: {z.shape}, range: [{z.min():.3f}, {z.max():.3f}]")