#!/usr/bin/env python3
"""
Frame Encoder para Attention-Based Temporal VAE
Reutiliza la arquitectura VAE existente para procesar histogramas individuales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameEncoder(nn.Module):
    """
    Procesa histogramas individuales (512, 3) generando embeddings temporales de 128D
    Reutiliza la arquitectura CNN del VAE existente pero optimizada para frames
    """
    def __init__(self, 
                 input_channels=3,
                 input_bins=512,
                 embed_dim=128):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_bins = input_bins
        self.embed_dim = embed_dim
        
        # CNN 1D con dilated convolutions (del VAE existente)
        self.conv_layers = nn.Sequential(
            # Primera capa: características básicas
            nn.Conv1d(input_channels, 64, kernel_size=15, dilation=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Segunda capa: patrones complejos  
            nn.Conv1d(64, 128, kernel_size=15, dilation=2, padding=14),
            nn.BatchNorm1d(128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Tercera capa: representación alta nivel
            nn.Conv1d(128, 256, kernel_size=15, dilation=4, padding=28),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Reducción a embedding dimension
            nn.Conv1d(256, 128, kernel_size=15, dilation=2, padding=14),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling + projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.frame_projection = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización Xavier para estabilidad"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, histogram):
        """
        Forward pass para un histograma individual
        
        Args:
            histogram: (batch_size, 512, 3) - histograma enriquecido
            
        Returns:
            frame_embedding: (batch_size, embed_dim)
        """
        # Transponer para conv1d: (batch, channels, sequence) 
        x = histogram.transpose(1, 2)  # (batch_size, 3, 512)
        
        # Convoluciones dilatadas
        x = self.conv_layers(x)  # (batch_size, 128, 512)
        
        # Global pooling y projection
        x = self.global_pool(x).squeeze(-1)  # (batch_size, 128)
        frame_embedding = self.frame_projection(x)  # (batch_size, embed_dim)
        
        return frame_embedding

if __name__ == "__main__":
    # Test básico
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = FrameEncoder().to(device)
    test_histogram = torch.randn(4, 512, 3).to(device)
    
    with torch.no_grad():
        embedding = encoder(test_histogram)
        print(f"✅ FrameEncoder: {test_histogram.shape} → {embedding.shape}")
        print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")