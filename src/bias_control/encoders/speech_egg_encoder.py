"""
Speech/EGG Encoder for Escalón 2 cross-modal learning.

Lightweight CNN+Transformer encoder for 16kHz waveforms (speech or EGG).
Same architecture used for both modalities (symmetric encoders).

Architecture:
  Waveform [B, 32000] (2s @ 16kHz)
    -> Conv1d(1, 256, k=10, s=5)  -> GN -> GELU   -> [B, 256, 6400]
    -> Conv1d(256, 256, k=3, s=2) -> GN -> GELU   -> [B, 256, 3200]
    -> Conv1d(256, 256, k=3, s=2) -> GN -> GELU   -> [B, 256, 1600]
    -> Conv1d(256, 512, k=3, s=2) -> GN -> GELU   -> [B, 512, 800]
    -> Positional Embedding
    -> Transformer(4 layers, 8 heads, d=512)
    -> Mean pooling -> [B, 512]

~15M params per encoder, ~30M total for the pair.
"""

import torch
import torch.nn as nn


class SpeechEGGEncoder(nn.Module):
    """CNN + Transformer encoder for 16kHz waveforms."""

    def __init__(self, output_dim: int = 512, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.output_dim = output_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Conv1d(256, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, output_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Max 1000 positions (2s@16kHz -> CNN ~800 tokens)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, output_dim) * 0.02)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] raw audio at 16kHz

        Returns:
            [B, output_dim] mean-pooled embeddings
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]

        features = self.feature_extractor(waveform)  # [B, D, T']
        features = features.transpose(1, 2)  # [B, T', D]

        T = features.size(1)
        features = features + self.pos_embedding[:, :T, :]

        encoded = self.transformer(features)  # [B, T', D]
        return encoded.mean(dim=1)  # [B, D]

    def get_output_dim(self) -> int:
        return self.output_dim
