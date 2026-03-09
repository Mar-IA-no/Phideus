"""
Augmented Speech/EGG Encoder with descriptor injection for S2-P2-main.

Subclass of SpeechEGGEncoder that injects vocal descriptors between
CNN and Transformer via concatenation + linear projection:

  CNN [B, 512, T'] -> transpose -> [B, T', 512]
  + descriptor [B, T', D] -> concat -> Linear(512+D, 512) -> pos_emb -> Transformer

Near-identity init: W=[I|0], bias=0 -> ep0 output identical to D0.
NO LayerNorm in injection (base encoder has none between CNN and Transformer).
"""

import torch
import torch.nn as nn

from .speech_egg_encoder import SpeechEGGEncoder


class SpeechEGGEncoderAug(SpeechEGGEncoder):
    """SpeechEGGEncoder with descriptor injection via input augmentation.

    When descriptor=None in forward(), behaves identically to base encoder.
    """

    def __init__(self, descriptor_dim: int, output_dim: int = 512,
                 n_layers: int = 4, n_heads: int = 8):
        super().__init__(output_dim=output_dim, n_layers=n_layers, n_heads=n_heads)
        self.descriptor_dim = descriptor_dim

        # Projection: concat(CNN_features, descriptor) -> output_dim
        self.descriptor_proj = nn.Linear(output_dim + descriptor_dim, output_dim)

        # Near-identity init: W = [I | 0], bias = 0
        with torch.no_grad():
            self.descriptor_proj.weight.zero_()
            self.descriptor_proj.weight[:output_dim, :output_dim] = torch.eye(output_dim)
            self.descriptor_proj.bias.zero_()

    def forward(self, waveform: torch.Tensor,
                descriptor: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] raw audio at 16kHz
            descriptor: [B, T_cnn, D] pre-aligned descriptor, or None for D0 mode

        Returns:
            [B, output_dim] mean-pooled embeddings
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]

        features = self.feature_extractor(waveform)  # [B, D, T']
        features = features.transpose(1, 2)  # [B, T', D]

        # Descriptor injection (only if provided)
        if descriptor is not None:
            T = features.size(1)
            # Safety: ensure descriptor matches temporal dim
            if descriptor.size(1) != T:
                descriptor = descriptor.transpose(1, 2)  # [B, D_desc, T_desc]
                descriptor = nn.functional.interpolate(
                    descriptor, size=T, mode='linear', align_corners=False
                )
                descriptor = descriptor.transpose(1, 2)  # [B, T, D_desc]

            features = torch.cat([features, descriptor], dim=-1)  # [B, T', 512+D]
            features = self.descriptor_proj(features)  # [B, T', 512]

        T = features.size(1)
        features = features + self.pos_embedding[:, :T, :]

        encoded = self.transformer(features)  # [B, T', D]
        return encoded.mean(dim=1)  # [B, D]
