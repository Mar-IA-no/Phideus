#!/usr/bin/env python3
"""
Gate 5B — Conditioned Transformer Decoder for Test 11 (Decoder Suite).

Architecture:
    z[256] --> Linear(256, 16*512) --> reshape --> [16, 512] conditioning tokens (memory)
                                                      |
                                                      v (cross-attention)
    T=188 learnable frame queries [188, 512]  -->  TransformerDecoder  --> [188, 512]
         + sinusoidal positional encoding           (6 layers, h=8)
                                                      |
                                                      v
                                                 Output Head
                                            Linear(512, n_output)

- Mel head: Linear(512, 128), outputs log(1+mel) directly
- PR head: Linear(512, 88), outputs RAW LOGITS (no sigmoid in forward)
- Training: BCEWithLogitsLoss(pos_weight=50) for PR (sigmoid inside loss)
- Eval/generation: apply sigmoid() externally for PR probabilities

~20M params total.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(n_pos: int, d_model: int) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding [n_pos, d_model].

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [n_pos, d_model]


class ConditionedTransformerDecoder(nn.Module):
    """
    Transformer decoder conditioned on a frozen embedding z[256].

    z is projected to n_cond_tokens conditioning tokens that serve as
    cross-attention memory. Learnable frame queries (with sinusoidal PE)
    attend to these tokens via a standard TransformerDecoder, then a
    linear head maps to the target dimension.

    Args:
        z_dim: Embedding dimension (256)
        d_model: Transformer hidden size (512)
        n_heads: Number of attention heads (8)
        n_layers: Number of decoder layers (6)
        d_ff: Feed-forward dimension (2048)
        n_frames: Number of output frames (188)
        n_cond_tokens: Number of conditioning tokens derived from z (16)
        n_output: Output dimension per frame (128 for mel, 88 for PR)
        dropout: Dropout rate (0.1)
    """

    def __init__(
        self,
        z_dim: int = 256,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        n_frames: int = 188,
        n_cond_tokens: int = 16,
        n_output: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.d_model = d_model
        self.n_frames = n_frames
        self.n_cond_tokens = n_cond_tokens
        self.n_output = n_output

        # z -> conditioning memory tokens
        self.z_projection = nn.Linear(z_dim, n_cond_tokens * d_model)

        # Learnable frame queries
        self.frame_queries = nn.Parameter(torch.randn(n_frames, d_model) * 0.02)

        # Sinusoidal positional encoding (registered as buffer, not parameter)
        pe = sinusoidal_positional_encoding(n_frames, d_model)
        self.register_buffer('pos_encoding', pe)

        # Transformer decoder (non-causal: decode all frames at once)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers,
        )

        # Output head (raw values: log-mel or logits)
        self.output_head = nn.Linear(d_model, n_output)

        # Initialize output head near zero for stable start
        nn.init.zeros_(self.output_head.bias)
        nn.init.normal_(self.output_head.weight, std=0.01)

        self._log_param_count()

    def _log_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        self._n_params = total

    @property
    def n_params(self) -> int:
        return self._n_params

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: [B, z_dim] frozen embedding

        Returns:
            [B, n_frames, n_output] — raw values:
                - For mel tasks: log(1+mel) predictions
                - For PR tasks: raw logits (apply sigmoid externally)
        """
        B = z.shape[0]

        # Project z to conditioning tokens: [B, n_cond_tokens * d_model] -> [B, n_cond, d_model]
        memory = self.z_projection(z).view(B, self.n_cond_tokens, self.d_model)

        # Frame queries with positional encoding: [n_frames, d_model] -> [B, n_frames, d_model]
        queries = (self.frame_queries + self.pos_encoding).unsqueeze(0).expand(B, -1, -1)

        # Transformer decoder: queries attend to memory (no causal mask)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)  # [B, n_frames, d_model]

        # Output projection
        output = self.output_head(decoded)  # [B, n_frames, n_output]

        return output


def create_decoder(task_name: str, z_dim: int = 256) -> ConditionedTransformerDecoder:
    """
    Factory: create decoder for a specific task.

    Args:
        task_name: One of 'audio2mel', 'midi2mel', 'midi2pr', 'audio2pr',
                   'random2mel', 'random2pr'
        z_dim: Input embedding dimension

    Returns:
        ConditionedTransformerDecoder with correct output dimension
    """
    if 'mel' in task_name:
        n_output = 128
    elif 'pr' in task_name:
        n_output = 88
    else:
        raise ValueError(f"Cannot determine output dim for task '{task_name}'. "
                         f"Task name must contain 'mel' or 'pr'.")

    model = ConditionedTransformerDecoder(
        z_dim=z_dim,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        n_frames=188,
        n_cond_tokens=16,
        n_output=n_output,
        dropout=0.1,
    )
    return model


if __name__ == '__main__':
    # Quick sanity check
    print("Decoder Model Sanity Check")
    print("=" * 50)

    for task, n_out in [('audio2mel', 128), ('midi2pr', 88)]:
        dec = create_decoder(task)
        z = torch.randn(4, 256)
        out = dec(z)
        print(f"  {task}: z{list(z.shape)} -> {list(out.shape)}, "
              f"params={dec.n_params/1e6:.1f}M")
        assert out.shape == (4, 188, n_out), f"Shape mismatch: {out.shape}"

    print("\nAll checks passed!")
