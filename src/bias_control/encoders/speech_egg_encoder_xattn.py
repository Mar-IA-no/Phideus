"""
Cross-Attention Speech/EGG Encoder for S2-P2.5.

Subclass of SpeechEGGEncoder that injects H-series descriptor via
residual cross-attention between CNN and Transformer.

Architecture:
  CNN [B, 512, T'] -> transpose -> [B, T', 512]
  Descriptor [B, T', 8] -> desc_proj -> [B, T', 512]
  Cross-attention: Q=desc_proj, K/V=CNN_features (no pos_emb)
  features += xattn_scale * LayerNorm(xattn_out)   [residual]
  + pos_embedding (after xattn)
  Transformer -> mean pool -> [B, 512]

Near-identity: xattn_scale=0.01 -> ~1% perturbation at ep0.

K/V = raw CNN features — EXPLICIT ARCHITECTURAL HYPOTHESIS:
  H-series captures "what harmonic structure is present in this instant" —
  inherently local, position-independent. Two frames with identical H-series
  at different times MUST produce identical cross-attention patterns.
  The cross-attention is permutation-equivariant: reorganizes by content.
  Temporal structure enters after via pos_emb + Transformer self-attention.

4 heads (not 8): d=512 -> d_head=128 (matching a4r from Gate 8).
H-series is only 8D — too few dimensions for 8 heads.

Params: desc_proj(~4.6K) + MHA(~1.05M) + LN(1K) + scale(1) ~ 1.06M per encoder.
"""

import torch
import torch.nn as nn

from .speech_egg_encoder import SpeechEGGEncoder


class SpeechEGGEncoderXAttn(SpeechEGGEncoder):
    """SpeechEGGEncoder with cross-attention from descriptor.

    When descriptor=None, behaves identically to base encoder.
    When provided, descriptor queries CNN features via cross-attention,
    modulating features with a small residual contribution.
    """

    def __init__(self, descriptor_dim: int = 8, n_xattn_heads: int = 4,
                 output_dim: int = 512, n_layers: int = 4, n_heads: int = 8):
        super().__init__(output_dim=output_dim, n_layers=n_layers, n_heads=n_heads)
        self.descriptor_dim = descriptor_dim

        self.desc_proj = nn.Linear(descriptor_dim, output_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=n_xattn_heads,
            batch_first=True,
            dropout=0.1,
        )

        self.xattn_norm = nn.LayerNorm(output_dim)

        # Near-zero init: ~1% perturbation at ep0
        self.xattn_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, waveform: torch.Tensor,
                descriptor: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] raw audio at 16kHz
            descriptor: [B, T_cnn, desc_dim] or None for D0 bypass

        Returns:
            [B, output_dim] mean-pooled embeddings
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]

        features = self.feature_extractor(waveform)  # [B, D, T']
        features = features.transpose(1, 2)          # [B, T', D]

        if descriptor is not None:
            T = features.size(1)
            # Ensure descriptor matches temporal dim
            if descriptor.size(1) != T:
                desc = descriptor.transpose(1, 2)
                desc = nn.functional.interpolate(
                    desc, size=T, mode='linear', align_corners=False
                )
                descriptor = desc.transpose(1, 2)

            desc_q = self.desc_proj(descriptor)  # [B, T, 512]

            # K/V = raw CNN features (no pos_emb — content-only hypothesis)
            xattn_out, _ = self.cross_attention(
                query=desc_q, key=features, value=features,
                need_weights=False,
            )
            features = features + self.xattn_scale * self.xattn_norm(xattn_out)

        T = features.size(1)
        features = features + self.pos_embedding[:, :T, :]

        encoded = self.transformer(features)
        return encoded.mean(dim=1)  # [B, D]
