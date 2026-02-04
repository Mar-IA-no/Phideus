#!/usr/bin/env python3
"""
Barlow Twins: Self-Supervised Learning via Redundancy Reduction
================================================================

Implementation for cross-modal learning (Audio <-> MIDI).

Barlow Twins Loss = sum((diag(C) - 1)^2) + λ * sum(off_diag(C)^2)

Where C is the cross-correlation matrix between z_audio and z_midi.

The loss pushes:
- Diagonal elements (same dimension) toward 1 (invariance)
- Off-diagonal elements (different dimensions) toward 0 (redundancy reduction)

Reference: Zbontar et al. "Barlow Twins: Self-Supervised Learning via
           Redundancy Reduction" ICML 2021

Usage:
------
from src.RNA.barlow_twins import BarlowTwinsLoss, BarlowTwinsCrossModal

model = BarlowTwinsCrossModal(audio_dim=128, midi_dim=88, z_dim=64)
loss = model(audio, midi)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BARLOW TWINS LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss for cross-modal learning.

    Computes cross-correlation matrix between two embedding batches
    and pushes it toward identity.
    """

    def __init__(
        self,
        lambda_off_diag: float = 0.0051,  # 1/z_dim is common default
        eps: float = 1e-5,
    ):
        """
        Args:
            lambda_off_diag: Weight for off-diagonal elements
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.lambda_off_diag = lambda_off_diag
        self.eps = eps

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Barlow Twins loss.

        Args:
            z_a: [N, D] first modality embeddings
            z_b: [N, D] second modality embeddings (paired)

        Returns:
            Dict with loss components and cross-correlation matrix
        """
        # Flatten if temporal
        if z_a.dim() == 3:
            z_a = z_a.reshape(-1, z_a.size(-1))
            z_b = z_b.reshape(-1, z_b.size(-1))

        N, D = z_a.shape

        # Normalize (zero mean, unit std)
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + self.eps)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + self.eps)

        # Cross-correlation matrix [D, D]
        C = (z_a_norm.T @ z_b_norm) / N

        # On-diagonal loss: (diag - 1)^2
        on_diag = (C.diagonal() - 1).pow(2).sum()

        # Off-diagonal loss: C[i,j]^2 for i != j
        off_diag = self._off_diagonal(C).pow(2).sum()

        # Total loss
        total = on_diag + self.lambda_off_diag * off_diag

        return {
            'total': total,
            'on_diag': on_diag,
            'off_diag': off_diag,
            'cross_correlation': C.detach(),
        }

    def _off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Extract off-diagonal elements of a square matrix."""
        n = x.shape[0]
        # Create mask for off-diagonal
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        return x[mask]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENCODER WITH PROJECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class BarlowEncoder(nn.Module):
    """
    Encoder with projector head for Barlow Twins.

    Architecture:
    - Encoder: Maps input to representation
    - Projector: MLP that maps representation to embedding space
    """

    def __init__(
        self,
        input_dim: int,
        repr_dim: int = 256,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        n_encoder_layers: int = 2,
        n_proj_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.repr_dim = repr_dim
        self.proj_dim = proj_dim

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.BatchNorm1d(hidden_dim))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(dropout))

        for _ in range(n_encoder_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))

        encoder_layers.append(nn.Linear(hidden_dim, repr_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Projector
        proj_layers = []
        proj_layers.append(nn.Linear(repr_dim, hidden_dim))
        proj_layers.append(nn.BatchNorm1d(hidden_dim))
        proj_layers.append(nn.ReLU())

        for _ in range(n_proj_layers - 2):
            proj_layers.append(nn.Linear(hidden_dim, hidden_dim))
            proj_layers.append(nn.BatchNorm1d(hidden_dim))
            proj_layers.append(nn.ReLU())

        proj_layers.append(nn.Linear(hidden_dim, proj_dim))
        self.projector = nn.Sequential(*proj_layers)

    def forward(
        self,
        x: torch.Tensor,
        return_repr: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, D] or [B, T, D] input
            return_repr: If True, return representation instead of projection

        Returns:
            z: [B, proj_dim] or [B, repr_dim] embeddings
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            h = self.encoder(x_flat)
            if return_repr:
                return h.reshape(B, T, -1)
            z = self.projector(h)
            return z.reshape(B, T, -1)
        else:
            h = self.encoder(x)
            if return_repr:
                return h
            return self.projector(h)


class BarlowTemporalEncoder(nn.Module):
    """
    Temporal encoder with LSTM for Barlow Twins.
    """

    def __init__(
        self,
        input_dim: int,
        repr_dim: int = 256,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.repr_dim = repr_dim
        self.proj_dim = proj_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Representation layer
        self.repr_proj = nn.Linear(lstm_out_dim, repr_dim)

        # Projector (for Barlow Twins loss)
        self.projector = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_repr: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, D] input sequence
            lengths: [B] sequence lengths
            return_repr: Return representation instead of projection

        Returns:
            z: [B, T, proj_dim] or [B, T, repr_dim]
        """
        B, T, D = x.shape

        # Input projection
        h = self.input_proj(x)

        # LSTM
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

        # Representation
        repr = self.repr_proj(lstm_out)

        if return_repr:
            return repr

        # Project (need to flatten for BatchNorm)
        repr_flat = repr.reshape(B * T, -1)
        z_flat = self.projector(repr_flat)
        return z_flat.reshape(B, T, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class BarlowTwinsCrossModal(nn.Module):
    """
    Complete Barlow Twins model for cross-modal learning.
    """

    def __init__(
        self,
        audio_input_dim: int = 128,
        midi_input_dim: int = 88,
        repr_dim: int = 256,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        temporal: bool = True,
        lambda_off_diag: float = 0.0051,
    ):
        super().__init__()

        self.temporal = temporal
        self.repr_dim = repr_dim
        self.proj_dim = proj_dim

        # Auto-compute lambda from proj_dim
        if lambda_off_diag == 0.0051:  # default
            lambda_off_diag = 1.0 / proj_dim

        # Encoders
        if temporal:
            self.encoder_audio = BarlowTemporalEncoder(
                input_dim=audio_input_dim,
                repr_dim=repr_dim,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
            )
            self.encoder_midi = BarlowTemporalEncoder(
                input_dim=midi_input_dim,
                repr_dim=repr_dim,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.encoder_audio = BarlowEncoder(
                input_dim=audio_input_dim,
                repr_dim=repr_dim,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
            )
            self.encoder_midi = BarlowEncoder(
                input_dim=midi_input_dim,
                repr_dim=repr_dim,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
            )

        # Loss
        self.loss_fn = BarlowTwinsLoss(lambda_off_diag=lambda_off_diag)

    def forward(
        self,
        audio: torch.Tensor,
        midi: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            audio: [B, T, D_audio] or [B, D_audio]
            midi: [B, T, D_midi] or [B, D_midi]
            lengths: [B] sequence lengths

        Returns:
            Dict with embeddings and losses
        """
        # Encode
        if self.temporal and lengths is not None:
            z_audio = self.encoder_audio(audio, lengths)
            z_midi = self.encoder_midi(midi, lengths)
        else:
            z_audio = self.encoder_audio(audio)
            z_midi = self.encoder_midi(midi)

        # Loss
        losses = self.loss_fn(z_audio, z_midi)

        return {
            'z_audio': z_audio,
            'z_midi': z_midi,
            **losses,
        }

    def get_embeddings(
        self,
        audio: torch.Tensor,
        midi: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        use_repr: bool = True,
        pool: str = 'mean',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings for retrieval.

        Args:
            use_repr: Use representation (before projector) instead of projection
            pool: 'mean' or 'last' for temporal pooling

        Returns:
            z_audio, z_midi: [B, D] embeddings
        """
        with torch.no_grad():
            if self.temporal and lengths is not None:
                z_audio = self.encoder_audio(audio, lengths, return_repr=use_repr)
                z_midi = self.encoder_midi(midi, lengths, return_repr=use_repr)
            else:
                z_audio = self.encoder_audio(audio, return_repr=use_repr)
                z_midi = self.encoder_midi(midi, return_repr=use_repr)

            # Pool
            if z_audio.dim() == 3:
                if pool == 'mean':
                    z_audio = z_audio.mean(dim=1)
                    z_midi = z_midi.mean(dim=1)
                else:
                    z_audio = z_audio[:, -1]
                    z_midi = z_midi[:, -1]

        return z_audio, z_midi

    def check_collapse(
        self,
        audio: torch.Tensor,
        midi: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Check if model has collapsed by examining variance.
        """
        z_a, z_m = self.get_embeddings(audio, midi, lengths)

        var_audio = z_a.var(dim=0).mean().item()
        var_midi = z_m.var(dim=0).mean().item()

        return {
            'var_audio': var_audio,
            'var_midi': var_midi,
            'collapsed': var_audio < 0.01 or var_midi < 0.01,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Barlow Twins Test")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test data
    B, T, D_audio, D_midi = 8, 50, 128, 88

    audio = torch.randn(B, T, D_audio).to(device)
    midi = torch.randn(B, T, D_midi).to(device)
    lengths = torch.tensor([T, T-5, T-10, T-15, T-20, T-25, T-30, T-35]).to(device)

    # Model
    model = BarlowTwinsCrossModal(
        audio_input_dim=D_audio,
        midi_input_dim=D_midi,
        repr_dim=128,
        proj_dim=64,
        hidden_dim=128,
        temporal=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Forward
    outputs = model(audio, midi, lengths)

    print(f"\nz_audio shape: {outputs['z_audio'].shape}")
    print(f"z_midi shape: {outputs['z_midi'].shape}")
    print(f"Total loss: {outputs['total'].item():.4f}")
    print(f"  On-diagonal: {outputs['on_diag'].item():.4f}")
    print(f"  Off-diagonal: {outputs['off_diag'].item():.4f}")
    print(f"Cross-correlation shape: {outputs['cross_correlation'].shape}")

    # Get embeddings
    z_a, z_m = model.get_embeddings(audio, midi, lengths, use_repr=True)
    print(f"\nRepresentation embeddings: {z_a.shape}")

    # Check collapse
    collapse = model.check_collapse(audio, midi, lengths)
    print(f"Variance audio: {collapse['var_audio']:.4f}")
    print(f"Variance midi: {collapse['var_midi']:.4f}")
    print(f"Collapsed: {collapse['collapsed']}")

    print("\n" + "=" * 50)
    print("Barlow Twins test passed!")
