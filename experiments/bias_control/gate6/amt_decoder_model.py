"""
Gate 6 — AMT Decoder Model (~38M params)

8-layer TransformerDecoder with cross-attention from learnable frame queries
to frozen VICReg encoder features. Significantly more powerful than the
Test 13G-B PostHocPRDecoder (2.4M params).

Architecture:
    VICReg Encoder (frozen) → [B, N, 1024] pre-pooling features
        ↓ Input projection: Linear(1024, 512)
        ↓ Learnable frame queries: [188, 512] + sinusoidal PE
        ↓ 8× TransformerDecoderLayer:
            ├── Cross-attention: queries ← encoder features
            ├── Self-attention: temporal context
            └── FFN: 512 → 2048 → 512
        ↓ Output head: Linear(512, 88) → sigmoid
        ↓
    Piano Roll [B, 188, 88]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class AMTDecoder(nn.Module):
    """Powerful AMT decoder for probing VICReg encoder features.

    Takes pre-pooling features from a frozen VICReg encoder and attempts
    to decode a piano roll via cross-attention.

    Args:
        feat_dim: Encoder feature dimension (1024 for MERT)
        d_model: Transformer dimension
        n_heads: Number of attention heads
        n_layers: Number of TransformerDecoder layers
        d_ff: Feed-forward hidden dimension
        n_frames: Output frames (188 for 4s@24kHz/hop=512)
        n_pitches: Output pitches (88 piano keys)
        dropout: Dropout rate
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        n_frames: int = 188,
        n_pitches: int = 88,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.d_model = d_model
        self.n_frames = n_frames
        self.n_pitches = n_pitches

        # Project encoder features → d_model
        self.input_proj = nn.Linear(feat_dim, d_model)

        # Learnable frame queries
        self.frame_queries = nn.Parameter(torch.randn(n_frames, d_model) * 0.02)
        self.query_pe = SinusoidalPE(d_model, max_len=n_frames)

        # PE for encoder features (key side)
        self.key_pe = SinusoidalPE(d_model, max_len=4096)

        # 8-layer TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, n_pitches)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        encoder_features: torch.Tensor,
        pool_to: int = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            encoder_features: [B, N, feat_dim] pre-pooling features from VICReg encoder
            pool_to: If set, adaptive-pool features to this length first

        Returns:
            logits: [B, n_frames, n_pitches] raw logits (before sigmoid)
        """
        B = encoder_features.shape[0]

        # Optional pooling for high-resolution features (e.g., 2400 → 188)
        if pool_to is not None and encoder_features.shape[1] != pool_to:
            feats = encoder_features.transpose(1, 2)  # [B, D, N]
            feats = F.adaptive_avg_pool1d(feats, pool_to)
            feats = feats.transpose(1, 2)  # [B, pool_to, D]
        else:
            feats = encoder_features

        # Project to d_model
        memory = self.input_proj(feats)  # [B, N', d_model]
        memory = self.key_pe(memory)

        # Expand frame queries for batch
        queries = self.frame_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 188, d_model]
        queries = self.query_pe(queries)

        # Decode via cross-attention
        decoded = self.decoder(queries, memory)  # [B, 188, d_model]

        # Output head
        decoded = self.output_norm(decoded)
        logits = self.output_head(decoded)  # [B, 188, 88]

        return logits

    def predict(
        self,
        encoder_features: torch.Tensor,
        pool_to: int = None,
        threshold: float = 0.3,
    ) -> torch.Tensor:
        """Predict binary piano roll."""
        logits = self.forward(encoder_features, pool_to=pool_to)
        return (torch.sigmoid(logits) >= threshold).float()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_params_M(self) -> float:
        return self.num_params / 1e6


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification.

    Reduces contribution of easy negatives which dominate in sparse piano rolls.

    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        onset_weight: Extra weight for onset frames
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        onset_weight: float = 5.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.onset_weight = onset_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        onset_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: [B, T, 88] raw logits
            targets: [B, T, 88] binary targets
            onset_mask: [B, T, 88] onset frames (optional)

        Returns:
            Scalar loss
        """
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
        )

        # Focal modulation
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * ce_loss

        # Onset weighting
        if onset_mask is not None:
            onset_boost = 1.0 + self.onset_weight * onset_mask
            loss = loss * onset_boost

        return loss.mean()


def detect_onsets(piano_roll: torch.Tensor) -> torch.Tensor:
    """Detect onset frames in a binary piano roll.

    Onsets are positive-going edges: target[t] > 0 and target[t-1] == 0.

    Args:
        piano_roll: [B, T, 88] binary piano roll

    Returns:
        onset_mask: [B, T, 88] binary onset mask
    """
    onset_mask = torch.zeros_like(piano_roll)
    onset_mask[:, 0] = piano_roll[:, 0]
    onset_mask[:, 1:] = torch.clamp(piano_roll[:, 1:] - piano_roll[:, :-1], min=0)
    return onset_mask
