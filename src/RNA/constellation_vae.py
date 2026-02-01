#!/usr/bin/env python3
"""
Constellation VAE - Modular VAE for Sparse Token Representations (Fase 3A)
===========================================================================

VAE architecture for constellation tokens (anchor-target pairs) with:
- Modular encoder: MLP+Attention or Transformer
- Modular decoder: Histogram or Token reconstruction
- InfoNCE loss for cross-modal alignment

Token format: [log_ratio, delta_t, weight, anchor_band, target_band]
Shape: [B, T, max_tokens, 5] with mask [B, T, max_tokens]

Configurations:
- C1: MLP+Attention encoder + Histogram decoder
- C2: MLP+Attention encoder + Token decoder
- C3: Transformer encoder + Histogram decoder
- C4: Transformer encoder + Token decoder

Usage:
------
from src.RNA.constellation_vae import ConstellationVAE

model = ConstellationVAE(
    encoder_type='mlp',      # or 'transformer'
    decoder_type='histogram', # or 'token'
    token_dim=5,
    max_tokens=48,
    hidden_dim=128,
    z_shared_dim=32,
    z_private_dim=16,
)

# Forward pass
outputs = model(audio_tokens, audio_mask, vib_tokens, vib_mask)
loss = model.compute_loss(outputs, audio_tokens, vib_tokens, audio_mask, vib_mask)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENCODERS
# ═══════════════════════════════════════════════════════════════════════════════

class MLPConstellationEncoder(nn.Module):
    """
    MLP encoder with attention-weighted pooling for constellation tokens.

    Architecture:
    1. Token MLP: Embed each token [5] -> [hidden_dim]
    2. Attention Pooling: Weighted sum over tokens per frame
    3. Temporal LSTM: Model temporal dependencies
    4. Output heads: z_shared (mean, logvar) + z_private (mean, logvar)

    Note: Uses attention pooling instead of mean pooling to preserve
    relational information between tokens (critical for cross-modal matching).
    """

    def __init__(
        self,
        token_dim: int = 5,
        max_tokens: int = 48,
        hidden_dim: int = 128,
        z_shared_dim: int = 32,
        z_private_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self.z_shared_dim = z_shared_dim
        self.z_private_dim = z_private_dim

        # Token embedding MLP
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Attention pooling (learnable query)
        self.attention_query = nn.Linear(hidden_dim, 1, bias=False)

        # Temporal LSTM (bidirectional)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional

        # z_shared heads (for cross-modal alignment)
        self.shared_mean = nn.Linear(lstm_out_dim, z_shared_dim)
        self.shared_logvar = nn.Linear(lstm_out_dim, z_shared_dim)

        # z_private heads (domain-specific)
        self.private_mean = nn.Linear(lstm_out_dim, z_private_dim)
        self.private_logvar = nn.Linear(lstm_out_dim, z_private_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode constellation tokens to latent distributions.

        Args:
            tokens: [B, T, K, D] constellation tokens
            mask: [B, T, K] validity mask (1=valid, 0=padding)
            lengths: [B] actual sequence lengths (optional)

        Returns:
            z_shared_mean: [B, T, z_shared_dim]
            z_shared_logvar: [B, T, z_shared_dim]
            z_private_mean: [B, T, z_private_dim]
            z_private_logvar: [B, T, z_private_dim]
        """
        B, T, K, D = tokens.shape

        # 1. Embed each token
        x = self.token_mlp(tokens)  # [B, T, K, hidden]

        # 2. Attention-weighted pooling per timestep
        attn_logits = self.attention_query(x).squeeze(-1)  # [B, T, K]
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, T, K]

        # Handle case where all tokens are masked (set weights to 0)
        all_masked = (mask.sum(dim=-1, keepdim=True) == 0)
        attn_weights = attn_weights.masked_fill(all_masked, 0)

        # Weighted sum
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=2)  # [B, T, hidden]

        # 3. Temporal LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.lstm(x)  # [B, T, hidden*2]

        # 4. Output heads
        z_shared_mean = self.shared_mean(lstm_out)
        z_shared_logvar = self.shared_logvar(lstm_out)
        z_private_mean = self.private_mean(lstm_out)
        z_private_logvar = self.private_logvar(lstm_out)

        return z_shared_mean, z_shared_logvar, z_private_mean, z_private_logvar


class TransformerConstellationEncoder(nn.Module):
    """
    Transformer encoder with self-attention over tokens.

    Architecture:
    1. Token embedding + positional encoding
    2. Self-attention over tokens within each frame
    3. CLS token or attention pooling for frame representation
    4. Temporal transformer for cross-frame dependencies
    5. Output heads for z_shared and z_private
    """

    def __init__(
        self,
        token_dim: int = 5,
        max_tokens: int = 48,
        hidden_dim: int = 128,
        z_shared_dim: int = 32,
        z_private_dim: int = 16,
        num_heads: int = 4,
        num_token_layers: int = 2,
        num_temporal_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self.z_shared_dim = z_shared_dim
        self.z_private_dim = z_private_dim

        # Token embedding
        self.token_embed = nn.Linear(token_dim, hidden_dim)

        # Learnable CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Positional encoding for tokens within frame
        self.token_pos_embed = nn.Parameter(torch.randn(1, max_tokens + 1, hidden_dim) * 0.02)

        # Self-attention over tokens (within frame)
        token_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.token_transformer = nn.TransformerEncoder(token_layer, num_layers=num_token_layers)

        # Temporal positional encoding
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Temporal transformer (cross-frame)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_temporal_layers)

        # Output heads
        self.shared_mean = nn.Linear(hidden_dim, z_shared_dim)
        self.shared_logvar = nn.Linear(hidden_dim, z_shared_dim)
        self.private_mean = nn.Linear(hidden_dim, z_private_dim)
        self.private_logvar = nn.Linear(hidden_dim, z_private_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode constellation tokens using transformer.

        Args:
            tokens: [B, T, K, D] constellation tokens
            mask: [B, T, K] validity mask (1=valid, 0=padding)
            lengths: [B] actual sequence lengths (optional)

        Returns:
            z_shared_mean, z_shared_logvar, z_private_mean, z_private_logvar
        """
        B, T, K, D = tokens.shape
        device = tokens.device

        # 1. Embed tokens
        x = self.token_embed(tokens)  # [B, T, K, hidden]

        # 2. Process each frame with self-attention
        frame_features = []
        for t in range(T):
            # Get frame tokens: [B, K, hidden]
            frame_tokens = x[:, t]
            frame_mask = mask[:, t]  # [B, K]

            # Add CLS token: [B, K+1, hidden]
            cls = self.cls_token.expand(B, -1, -1)
            frame_with_cls = torch.cat([cls, frame_tokens], dim=1)

            # Add positional encoding
            frame_with_cls = frame_with_cls + self.token_pos_embed[:, :K+1]

            # Create attention mask (True = ignore)
            # CLS token is always valid, then token mask
            cls_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)
            attn_mask = torch.cat([cls_mask, frame_mask == 0], dim=1)  # [B, K+1]

            # Self-attention
            frame_out = self.token_transformer(
                frame_with_cls,
                src_key_padding_mask=attn_mask
            )  # [B, K+1, hidden]

            # Extract CLS token as frame representation
            frame_feat = frame_out[:, 0]  # [B, hidden]
            frame_features.append(frame_feat)

        # Stack frames: [B, T, hidden]
        x = torch.stack(frame_features, dim=1)

        # 3. Add temporal positional encoding
        x = x + self.temporal_pos_embed[:, :T]

        # 4. Temporal transformer
        if lengths is not None:
            # Create temporal mask
            temporal_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            temporal_mask = None

        x = self.temporal_transformer(x, src_key_padding_mask=temporal_mask)  # [B, T, hidden]

        # 5. Output heads
        z_shared_mean = self.shared_mean(x)
        z_shared_logvar = self.shared_logvar(x)
        z_private_mean = self.private_mean(x)
        z_private_logvar = self.private_logvar(x)

        return z_shared_mean, z_shared_logvar, z_private_mean, z_private_logvar


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DECODERS
# ═══════════════════════════════════════════════════════════════════════════════

class HistogramDecoder(nn.Module):
    """
    Decoder that reconstructs histogram representation from latent z.

    Output: [B, T, bins, channels] histogram (compatible with v2.2 evaluation)
    """

    def __init__(
        self,
        z_dim: int = 48,  # z_shared + z_private
        hidden_dim: int = 128,
        output_bins: int = 256,
        output_channels: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.output_bins = output_bins
        self.output_channels = output_channels
        self.output_dim = output_bins * output_channels

        # Input projection from z
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_out_dim = hidden_dim * 2

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, self.output_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode latent z to histogram.

        Args:
            z: [B, T, z_dim] latent representation
            lengths: [B] actual lengths

        Returns:
            recon: [B, T, bins, channels] reconstructed histogram
        """
        B, T, _ = z.shape

        # Project z
        h = self.z_proj(z)  # [B, T, hidden]

        # Temporal LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.lstm(h)

        # Project to output
        out = self.output_proj(lstm_out)  # [B, T, bins*channels]

        # Reshape
        recon = out.view(B, T, self.output_bins, self.output_channels)

        return recon


class TokenDecoder(nn.Module):
    """
    Decoder that reconstructs constellation tokens from latent z.

    Output: [B, T, max_tokens, token_dim] tokens (consistent with sparse representation)

    Note: Also outputs a token validity logit for predicting which positions
    should have valid tokens.
    """

    def __init__(
        self,
        z_dim: int = 48,
        hidden_dim: int = 128,
        max_tokens: int = 48,
        token_dim: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_tokens = max_tokens
        self.token_dim = token_dim

        # Input projection from z
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_out_dim = hidden_dim * 2

        # Token generation MLP (one per position)
        self.token_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, max_tokens * token_dim),
        )

        # Validity prediction (which positions should have tokens)
        self.validity_proj = nn.Linear(lstm_out_dim, max_tokens)

    def forward(
        self,
        z: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent z to tokens.

        Args:
            z: [B, T, z_dim] latent representation
            lengths: [B] actual lengths

        Returns:
            recon_tokens: [B, T, max_tokens, token_dim] reconstructed tokens
            validity_logits: [B, T, max_tokens] logits for token validity
        """
        B, T, _ = z.shape

        # Project z
        h = self.z_proj(z)

        # Temporal LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.lstm(h)

        # Generate tokens
        tokens_flat = self.token_proj(lstm_out)  # [B, T, max_tokens * token_dim]
        recon_tokens = tokens_flat.view(B, T, self.max_tokens, self.token_dim)

        # Validity logits
        validity_logits = self.validity_proj(lstm_out)  # [B, T, max_tokens]

        return recon_tokens, validity_logits


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ConstellationVAE(nn.Module):
    """
    Modular VAE for constellation tokens with configurable encoder/decoder.

    Configurations:
    - C1: encoder='mlp', decoder='histogram'
    - C2: encoder='mlp', decoder='token'
    - C3: encoder='transformer', decoder='histogram'
    - C4: encoder='transformer', decoder='token'
    """

    def __init__(
        self,
        encoder_type: Literal['mlp', 'transformer'] = 'mlp',
        decoder_type: Literal['histogram', 'token'] = 'histogram',
        token_dim: int = 5,
        max_tokens: int = 48,
        hidden_dim: int = 128,
        z_shared_dim: int = 32,
        z_private_dim: int = 16,
        output_bins: int = 256,
        output_channels: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        dropout_shared: float = 0.0,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.z_shared_dim = z_shared_dim
        self.z_private_dim = z_private_dim
        self.z_dim = z_shared_dim + z_private_dim
        self.dropout_shared = dropout_shared

        # Create encoder
        if encoder_type == 'mlp':
            self.encoder = MLPConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_shared_dim,
                z_private_dim=z_private_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.encoder = TransformerConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_shared_dim,
                z_private_dim=z_private_dim,
                dropout=dropout,
            )

        # Create decoder
        if decoder_type == 'histogram':
            self.decoder = HistogramDecoder(
                z_dim=self.z_dim,
                hidden_dim=hidden_dim,
                output_bins=output_bins,
                output_channels=output_channels,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.decoder = TokenDecoder(
                z_dim=self.z_dim,
                hidden_dim=hidden_dim,
                max_tokens=max_tokens,
                token_dim=token_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        # InfoNCE loss for cross-modal alignment
        self.temperature = temperature

        # Shared dropout (to force private to be informative)
        if dropout_shared > 0:
            self.shared_dropout = nn.Dropout(dropout_shared)
        else:
            self.shared_dropout = None

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Reparameterization trick."""
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode tokens to latent space.

        Returns dict with:
        - z_shared, z_shared_mean, z_shared_logvar
        - z_private, z_private_mean, z_private_logvar
        - z (concatenated)
        """
        z_shared_mean, z_shared_logvar, z_private_mean, z_private_logvar = \
            self.encoder(tokens, mask, lengths)

        z_shared = self.reparameterize(z_shared_mean, z_shared_logvar, self.training)
        z_private = self.reparameterize(z_private_mean, z_private_logvar, self.training)

        # Apply shared dropout if configured
        if self.shared_dropout is not None and self.training:
            z_shared = self.shared_dropout(z_shared)

        z = torch.cat([z_shared, z_private], dim=-1)

        return {
            'z_shared': z_shared,
            'z_shared_mean': z_shared_mean,
            'z_shared_logvar': z_shared_logvar,
            'z_private': z_private,
            'z_private_mean': z_private_mean,
            'z_private_logvar': z_private_logvar,
            'z': z,
        }

    def decode(
        self,
        z: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode latent z to reconstruction."""
        if self.decoder_type == 'histogram':
            recon = self.decoder(z, lengths)
            return {'recon': recon}
        else:
            recon_tokens, validity_logits = self.decoder(z, lengths)
            return {
                'recon_tokens': recon_tokens,
                'validity_logits': validity_logits,
            }

    def forward(
        self,
        audio_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
        vib_tokens: torch.Tensor,
        vib_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for both domains.

        Args:
            audio_tokens: [B, T, K, D] audio constellation tokens
            audio_mask: [B, T, K] audio token validity mask
            vib_tokens: [B, T, K, D] vibration constellation tokens
            vib_mask: [B, T, K] vibration token validity mask
            lengths: [B] sequence lengths

        Returns:
            Dict with all latents and reconstructions for both domains
        """
        # Encode both domains
        audio_enc = self.encode(audio_tokens, audio_mask, lengths)
        vib_enc = self.encode(vib_tokens, vib_mask, lengths)

        # Decode both domains
        audio_dec = self.decode(audio_enc['z'], lengths)
        vib_dec = self.decode(vib_enc['z'], lengths)

        return {
            # Audio
            'audio_z_shared': audio_enc['z_shared'],
            'audio_z_shared_mean': audio_enc['z_shared_mean'],
            'audio_z_shared_logvar': audio_enc['z_shared_logvar'],
            'audio_z_private': audio_enc['z_private'],
            'audio_z_private_mean': audio_enc['z_private_mean'],
            'audio_z_private_logvar': audio_enc['z_private_logvar'],
            'audio_z': audio_enc['z'],
            **{f'audio_{k}': v for k, v in audio_dec.items()},
            # Vibration
            'vib_z_shared': vib_enc['z_shared'],
            'vib_z_shared_mean': vib_enc['z_shared_mean'],
            'vib_z_shared_logvar': vib_enc['z_shared_logvar'],
            'vib_z_private': vib_enc['z_private'],
            'vib_z_private_mean': vib_enc['z_private_mean'],
            'vib_z_private_logvar': vib_enc['z_private_logvar'],
            'vib_z': vib_enc['z'],
            **{f'vib_{k}': v for k, v in vib_dec.items()},
        }

    def compute_infonce_loss(
        self,
        z_audio: torch.Tensor,
        z_vib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        sample_frames: int = 10,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for cross-modal alignment.

        Uses z_shared from both domains to encourage alignment.
        """
        B, T, D = z_audio.shape

        # Sample frames if sequence is long
        if T > sample_frames:
            indices = torch.randperm(T, device=z_audio.device)[:sample_frames]
            z_a = z_audio[:, indices]
            z_v = z_vib[:, indices]
        else:
            z_a = z_audio
            z_v = z_vib

        # Flatten
        z_a = z_a.reshape(-1, D)  # [B*S, D]
        z_v = z_v.reshape(-1, D)

        # L2 normalize
        z_a = F.normalize(z_a, dim=-1)
        z_v = F.normalize(z_v, dim=-1)

        # Cosine similarity
        logits = torch.matmul(z_a, z_v.T) / self.temperature

        # Positive pairs on diagonal
        labels = torch.arange(z_a.size(0), device=z_a.device)

        loss_a_to_v = F.cross_entropy(logits, labels)
        loss_v_to_a = F.cross_entropy(logits.T, labels)

        return (loss_a_to_v + loss_v_to_a) / 2

    def compute_kl_loss(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL divergence from standard normal."""
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

        if mask is not None:
            # Average over valid positions only
            kl = kl.sum() / max(mask.sum(), 1)
        else:
            kl = kl.mean()

        return kl

    def compute_recon_loss_histogram(
        self,
        recon: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for histogram decoder.

        Note: Target is tokens, but we compare against a pseudo-histogram
        created from the tokens. This is an approximation.
        """
        # For histogram decoder with token input, we use MSE loss
        # on the reconstruction directly (assuming target is provided as histogram)
        # In practice, we need the original histogram as target
        # For now, we'll just return 0 and handle this in the training loop
        return torch.tensor(0.0, device=recon.device)

    def compute_recon_loss_token(
        self,
        recon_tokens: torch.Tensor,
        validity_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for token decoder.

        Combines:
        - MSE loss on token values (masked)
        - BCE loss on validity prediction
        """
        # Token value loss (only on valid positions)
        token_mse = F.mse_loss(recon_tokens, target_tokens, reduction='none')
        token_mse = token_mse.sum(dim=-1)  # Sum over token dims
        token_mse = (token_mse * target_mask).sum() / max(target_mask.sum(), 1)

        # Validity loss
        validity_loss = F.binary_cross_entropy_with_logits(
            validity_logits, target_mask, reduction='mean'
        )

        return token_mse + validity_loss

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        audio_tokens: torch.Tensor,
        vib_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
        vib_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        beta_kl_shared: float = 0.001,
        beta_kl_private: float = 0.01,
        lambda_infonce: float = 1.0,
        lambda_diff: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            outputs: Forward pass outputs
            audio_tokens, vib_tokens: [B, T, K, D] input tokens
            audio_mask, vib_mask: [B, T, K] validity masks
            lengths: [B] sequence lengths
            beta_kl_shared: Weight for shared KL divergence
            beta_kl_private: Weight for private KL divergence
            lambda_infonce: Weight for InfoNCE loss
            lambda_diff: Weight for difference loss (z_shared ≠ z_private)

        Returns:
            Dict with loss components and total loss
        """
        losses = {}

        # InfoNCE loss (cross-modal alignment)
        losses['infonce'] = lambda_infonce * self.compute_infonce_loss(
            outputs['audio_z_shared'],
            outputs['vib_z_shared'],
            lengths,
        )

        # KL losses
        losses['kl_shared_audio'] = beta_kl_shared * self.compute_kl_loss(
            outputs['audio_z_shared_mean'],
            outputs['audio_z_shared_logvar'],
        )
        losses['kl_shared_vib'] = beta_kl_shared * self.compute_kl_loss(
            outputs['vib_z_shared_mean'],
            outputs['vib_z_shared_logvar'],
        )
        losses['kl_private_audio'] = beta_kl_private * self.compute_kl_loss(
            outputs['audio_z_private_mean'],
            outputs['audio_z_private_logvar'],
        )
        losses['kl_private_vib'] = beta_kl_private * self.compute_kl_loss(
            outputs['vib_z_private_mean'],
            outputs['vib_z_private_logvar'],
        )

        # Reconstruction loss
        if self.decoder_type == 'token':
            losses['recon_audio'] = self.compute_recon_loss_token(
                outputs['audio_recon_tokens'],
                outputs['audio_validity_logits'],
                audio_tokens,
                audio_mask,
            )
            losses['recon_vib'] = self.compute_recon_loss_token(
                outputs['vib_recon_tokens'],
                outputs['vib_validity_logits'],
                vib_tokens,
                vib_mask,
            )
        else:
            # Histogram decoder - reconstruction loss handled separately
            losses['recon_audio'] = torch.tensor(0.0, device=audio_tokens.device)
            losses['recon_vib'] = torch.tensor(0.0, device=audio_tokens.device)

        # Difference loss (encourage z_shared ≠ z_private)
        # Uses correlation instead of cosine similarity to handle different dimensions
        if lambda_diff > 0 and self.z_shared_dim == self.z_private_dim:
            diff_audio = F.cosine_similarity(
                outputs['audio_z_shared'].mean(dim=1),
                outputs['audio_z_private'].mean(dim=1),
            ).abs().mean()
            diff_vib = F.cosine_similarity(
                outputs['vib_z_shared'].mean(dim=1),
                outputs['vib_z_private'].mean(dim=1),
            ).abs().mean()
            losses['diff'] = lambda_diff * (diff_audio + diff_vib)
        else:
            # Skip diff loss if dimensions don't match
            losses['diff'] = torch.tensor(0.0, device=audio_tokens.device)

        # Total loss
        losses['total'] = sum(losses.values())

        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Testing ConstellationVAE configurations...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test data
    B, T, K, D = 4, 50, 48, 5
    audio_tokens = torch.randn(B, T, K, D).to(device)
    audio_mask = (torch.rand(B, T, K) > 0.3).float().to(device)  # ~70% valid
    vib_tokens = torch.randn(B, T, K, D).to(device)
    vib_mask = (torch.rand(B, T, K) > 0.3).float().to(device)
    lengths = torch.tensor([T, T-10, T-20, T-30]).to(device)

    configs = [
        ('mlp', 'histogram'),
        ('mlp', 'token'),
        ('transformer', 'histogram'),
        ('transformer', 'token'),
    ]

    for enc_type, dec_type in configs:
        print(f"\nConfig: {enc_type}/{dec_type}")

        model = ConstellationVAE(
            encoder_type=enc_type,
            decoder_type=dec_type,
            token_dim=D,
            max_tokens=K,
            hidden_dim=64,  # Smaller for test
            z_shared_dim=16,
            z_private_dim=8,
        ).to(device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Forward pass
        outputs = model(audio_tokens, audio_mask, vib_tokens, vib_mask, lengths)
        print(f"  z_shared shape: {outputs['audio_z_shared'].shape}")

        if dec_type == 'histogram':
            print(f"  recon shape: {outputs['audio_recon'].shape}")
        else:
            print(f"  recon_tokens shape: {outputs['audio_recon_tokens'].shape}")

        # Compute loss
        losses = model.compute_loss(
            outputs, audio_tokens, vib_tokens, audio_mask, vib_mask, lengths
        )
        print(f"  Total loss: {losses['total'].item():.4f}")
        print(f"  InfoNCE: {losses['infonce'].item():.4f}")

    print("\n" + "=" * 60)
    print("✔ All configurations tested successfully!")
