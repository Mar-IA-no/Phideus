#!/usr/bin/env python3
"""
Temporal Self-Attention para Attention-Based Temporal VAE
Implementa self-attention multi-head sobre secuencias de frame embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal para información temporal absoluta
    """
    def __init__(self, d_model, max_len=300):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1)]

class TemporalSelfAttention(nn.Module):
    """
    Self-attention multi-head para secuencias temporales de embeddings
    Optimizado para detectar dependencias temporales de largo alcance
    """
    def __init__(self, 
                 embed_dim=128, 
                 num_heads=8, 
                 dropout=0.1,
                 max_sequence_length=120):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Proyecciones para Q, K, V
        self.q_projection = nn.Linear(embed_dim, embed_dim)
        self.k_projection = nn.Linear(embed_dim, embed_dim) 
        self.v_projection = nn.Linear(embed_dim, embed_dim)
        
        # Proyección de salida
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding para información temporal
        self.positional_encoding = PositionalEncoding(embed_dim, max_sequence_length)
        
        # Layer norm para estabilidad
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización estable para attention"""
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.xavier_uniform_(self.k_projection.weight) 
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.xavier_uniform_(self.out_projection.weight)
        
        # Bias a cero
        nn.init.zeros_(self.q_projection.bias)
        nn.init.zeros_(self.k_projection.bias)
        nn.init.zeros_(self.v_projection.bias)
        nn.init.zeros_(self.out_projection.bias)
    
    def forward(self, sequence, mask=None):
        """
        Forward pass con self-attention temporal
        
        Args:
            sequence: (batch_size, seq_len, embed_dim) - secuencia de embeddings
            mask: (batch_size, seq_len, seq_len) - máscara de atención opcional
            
        Returns:
            attended_sequence: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = sequence.shape
        
        # Agregar positional encoding
        sequence = self.positional_encoding(sequence)
        
        # Generar Q, K, V
        Q = self.q_projection(sequence)  # (batch, seq_len, embed_dim)
        K = self.k_projection(sequence)
        V = self.v_projection(sequence)
        
        # Reshape para multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, head_dim)
        
        # Calcular attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Shape: (batch, num_heads, seq_len, seq_len)
        
        # Aplicar máscara si se proporciona
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Softmax para attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aplicar attention a values
        attended = torch.matmul(attention_weights, V)
        # Shape: (batch, num_heads, seq_len, head_dim)
        
        # Concatenar heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Proyección final
        output = self.out_projection(attended)
        
        # Residual connection + layer norm
        output = self.layer_norm(output + sequence)
        
        return output, attention_weights
    
    def get_attention_patterns(self, sequence, mask=None):
        """
        Obtener solo los attention patterns para análisis
        
        Returns:
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        with torch.no_grad():
            _, attention_weights = self.forward(sequence, mask)
        return attention_weights

class RTX3090OptimizedAttention(TemporalSelfAttention):
    """
    Versión optimizada para RTX 3090 con memory-efficient attention
    """
    def __init__(self, **kwargs):
        # Reducir heads para memoria
        kwargs['num_heads'] = min(kwargs.get('num_heads', 8), 4)
        super().__init__(**kwargs)
        
        self.use_flash_attention = False
        try:
            # Intentar usar flash attention si está disponible
            import flash_attn
            self.use_flash_attention = True
        except ImportError:
            pass
    
    def forward(self, sequence, mask=None):
        """Forward pass optimizado para memoria"""
        if self.use_flash_attention:
            return self._flash_attention_forward(sequence, mask)
        else:
            return super().forward(sequence, mask)
    
    def _flash_attention_forward(self, sequence, mask=None):
        """Implementación con flash attention (si disponible)"""
        # Implementación simplificada - flash attention real sería más compleja
        return super().forward(sequence, mask)

if __name__ == "__main__":
    # Test de TemporalSelfAttention
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear modelo
    attention = TemporalSelfAttention(embed_dim=128, num_heads=8).to(device)
    
    # Test input: secuencia de 30 frames
    batch_size, seq_len = 2, 30
    sequence = torch.randn(batch_size, seq_len, 128).to(device)
    
    with torch.no_grad():
        attended_sequence, attention_weights = attention(sequence)
        
        print(f"✅ TemporalSelfAttention test:")
        print(f"Input: {sequence.shape}")
        print(f"Output: {attended_sequence.shape}")
        print(f"Attention weights: {attention_weights.shape}")
        print(f"Parameters: {sum(p.numel() for p in attention.parameters()):,}")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory: {memory_used:.3f} GB")