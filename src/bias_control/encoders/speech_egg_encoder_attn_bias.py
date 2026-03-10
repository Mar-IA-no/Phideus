"""
Attention-Bias Speech/EGG Encoder for S2-P2.5.

Subclass of SpeechEGGEncoder that modulates Transformer self-attention
via a factored bilinear bias computed from V4-lin descriptor.

Architecture:
  CNN [B, 512, T'] -> transpose -> [B, T', 512]
  + pos_embedding
  Descriptor [B, T', 4] -> AttentionBiasComputer -> bias [B*H, T', T']
  Transformer(features, mask=bias)  [bias as additive attn_mask]
  -> mean pool -> [B, 512]

Near-identity init: W=0 -> bias=0 -> exact D0 at ep0.

Gradient bootstrap (1-step delay):
  Step 1: only W moves (grad = bias_scale * phi x psi, nonzero)
  Step 2+: W != 0 -> all params (phi, psi, bias_scale) get gradient

Params: phi_net(~80) + psi_net(~80) + W(2048) + scale(1) ~ 2,200 per encoder.
"""

import torch
import torch.nn as nn

from .speech_egg_encoder import SpeechEGGEncoder


class AttentionBiasComputer(nn.Module):
    """Asymmetric directional compatibility from descriptor.

    Factored bilinear: bias[h,i,j] = scale * phi(d_i)^T W_h psi(d_j)

    phi and psi are different networks -> asymmetric (bias[i,j] != bias[j,i]).
    Captures directionality: "what frame i seeks" vs "what frame j offers".
    V4-lin's ratio_prev (incoming) and ratio_next (outgoing) naturally map to this.
    W_h per-head learns structural interaction patterns.

    W=0 at init -> bias=0 -> exact identity with base model.
    """

    def __init__(self, desc_dim: int = 4, d_bias: int = 16, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.d_bias = d_bias

        self.phi_net = nn.Sequential(
            nn.Linear(desc_dim, d_bias),
            nn.GELU(),
        )
        self.psi_net = nn.Sequential(
            nn.Linear(desc_dim, d_bias),
            nn.GELU(),
        )

        # Per-head bilinear weight, zero-init for identity
        self.W = nn.Parameter(torch.zeros(n_heads, d_bias, d_bias))

        # Learnable scale, small but nonzero
        self.bias_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        """Compute per-head bilinear attention bias.

        Args:
            descriptor: [B, T, desc_dim]

        Returns:
            [B*H, T, T] additive attention bias for nn.TransformerEncoder mask
        """
        phi = self.phi_net(descriptor)    # [B, T, d_bias]
        psi = self.psi_net(descriptor)    # [B, T, d_bias]

        # phi @ W_h -> [B, H, T, d_bias]
        phi_w = torch.einsum('btd,hde->bhte', phi, self.W)

        # bias[h,i,j] = phi_w[h,i,:] . psi[j,:]
        bias = torch.einsum('bhid,bjd->bhij', phi_w, psi)  # [B, H, T, T]

        B, H, T, _ = bias.shape
        return (self.bias_scale * bias).reshape(B * H, T, T)


class SpeechEGGEncoderAttnBias(SpeechEGGEncoder):
    """SpeechEGGEncoder with attention bias from descriptor.

    When descriptor=None in forward(), behaves identically to base encoder.
    When descriptor is provided with W=0 (init), bias=0 -> exact D0 output.
    """

    def __init__(self, descriptor_dim: int = 4, d_bias: int = 16,
                 output_dim: int = 512, n_layers: int = 4, n_heads: int = 8):
        super().__init__(output_dim=output_dim, n_layers=n_layers, n_heads=n_heads)
        self.descriptor_dim = descriptor_dim
        self.bias_computer = AttentionBiasComputer(
            desc_dim=descriptor_dim, d_bias=d_bias, n_heads=n_heads,
        )

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

        T = features.size(1)
        features = features + self.pos_embedding[:, :T, :]

        if descriptor is not None:
            # Ensure descriptor matches temporal dim
            if descriptor.size(1) != T:
                desc = descriptor.transpose(1, 2)
                desc = nn.functional.interpolate(
                    desc, size=T, mode='linear', align_corners=False
                )
                descriptor = desc.transpose(1, 2)

            bias = self.bias_computer(descriptor)  # [B*H, T, T]
        else:
            bias = None

        encoded = self._transformer_forward(self.transformer, features, mask=bias)
        return encoded.mean(dim=1)  # [B, D]

    @staticmethod
    def _transformer_forward(encoder, features, mask=None):
        """Manual transformer forward bypassing PyTorch 2.10 fused kernel.

        The fused fast path in TransformerEncoderLayer produces NaN in eval
        mode when any mask is provided. Iterating layers manually and calling
        MHA directly avoids this bug while producing identical results.
        """
        if mask is None:
            return encoder(features)

        x = features
        for layer in encoder.layers:
            sa_out = layer.self_attn(
                x, x, x, attn_mask=mask, need_weights=False,
            )[0]
            x = layer.norm1(x + layer.dropout1(sa_out))
            ff_out = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x)))
            )
            x = layer.norm2(x + layer.dropout2(ff_out))
        if encoder.norm is not None:
            x = encoder.norm(x)
        return x
