#!/usr/bin/env python3
"""
VICReg: Variance-Invariance-Covariance Regularization
======================================================

Implementation of VICReg for cross-modal learning (Audio <-> MIDI).

VICReg Loss = λ_inv * invariance + λ_var * variance + λ_cov * covariance

Where:
- Invariance: MSE(z_audio, z_midi) - push paired embeddings together
- Variance: Hinge loss on std(z) - prevent collapse
- Covariance: Off-diagonal elements of cov matrix -> 0 - decorrelate dims

Reference: Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization"
           ICLR 2022

Usage:
------
from src.RNA.vicreg import VICRegLoss, VICRegEncoder

encoder_audio = VICRegEncoder(input_dim=128, hidden_dim=256, z_dim=64)
encoder_midi = VICRegEncoder(input_dim=88, hidden_dim=256, z_dim=64)
loss_fn = VICRegLoss(lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0)

z_audio = encoder_audio(audio_features)
z_midi = encoder_midi(midi_features)
loss = loss_fn(z_audio, z_midi)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VICReg LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class VICRegLoss(nn.Module):
    """
    VICReg loss for cross-modal learning.

    Components:
    - Invariance: MSE between paired embeddings
    - Variance: Hinge loss on std (prevent collapse)
    - Covariance: Off-diagonal covariance -> 0 (decorrelate)
    """

    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        var_target: float = 1.0,
        eps: float = 1e-4,
    ):
        """
        Args:
            lambda_inv: Weight for invariance loss
            lambda_var: Weight for variance loss
            lambda_cov: Weight for covariance loss
            var_target: Target standard deviation (default 1.0)
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.var_target = var_target
        self.eps = eps

    def invariance_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invariance: MSE between paired embeddings.

        Args:
            z_a, z_b: [N, D] embeddings (should be paired)

        Returns:
            MSE loss
        """
        return F.mse_loss(z_a, z_b)

    def variance_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Variance: Hinge loss on per-dimension std.

        Prevents collapse by requiring std(z[:, d]) >= var_target.

        Args:
            z: [N, D] embeddings

        Returns:
            Variance hinge loss
        """
        # Std per dimension
        std = torch.sqrt(z.var(dim=0) + self.eps)

        # Hinge: max(0, target - std)
        hinge = F.relu(self.var_target - std)

        return hinge.mean()

    def covariance_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Covariance: Push off-diagonal cov to zero.

        Decorrelates the dimensions of the embedding.

        Args:
            z: [N, D] embeddings

        Returns:
            Off-diagonal covariance loss
        """
        N, D = z.shape

        # Center
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Covariance matrix
        cov = (z_centered.T @ z_centered) / (N - 1)

        # Off-diagonal elements
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        off_diag = off_diag / D

        return off_diag

    def forward(
        self,
        z_audio: torch.Tensor,
        z_midi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VICReg loss.

        Args:
            z_audio: [N, D] audio embeddings
            z_midi: [N, D] MIDI embeddings

        Returns:
            Dict with loss components and total
        """
        # Flatten if needed (e.g., [B, T, D] -> [B*T, D])
        if z_audio.dim() == 3:
            z_audio = z_audio.reshape(-1, z_audio.size(-1))
            z_midi = z_midi.reshape(-1, z_midi.size(-1))

        # Invariance
        inv_loss = self.invariance_loss(z_audio, z_midi)

        # Variance (on both modalities)
        var_loss_audio = self.variance_loss(z_audio)
        var_loss_midi = self.variance_loss(z_midi)
        var_loss = (var_loss_audio + var_loss_midi) / 2

        # Covariance (on both modalities)
        cov_loss_audio = self.covariance_loss(z_audio)
        cov_loss_midi = self.covariance_loss(z_midi)
        cov_loss = (cov_loss_audio + cov_loss_midi) / 2

        # Total
        total = (
            self.lambda_inv * inv_loss +
            self.lambda_var * var_loss +
            self.lambda_cov * cov_loss
        )

        return {
            'total': total,
            'invariance': inv_loss,
            'variance': var_loss,
            'covariance': cov_loss,
            'var_audio': var_loss_audio,
            'var_midi': var_loss_midi,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENCODER ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class VICRegEncoder(nn.Module):
    """
    Simple MLP encoder for VICReg.

    Architecture: Input -> [Linear -> BN -> ReLU] x (n_layers-1) -> Linear -> z
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        z_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer (no activation, no BN)
        layers.append(nn.Linear(hidden_dim, z_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: [B, D] or [B, T, D] input features

        Returns:
            z: [B, z_dim] or [B, T, z_dim] embeddings
        """
        if x.dim() == 3:
            # Temporal input: process each timestep
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            z_flat = self.encoder(x_flat)
            return z_flat.reshape(B, T, -1)
        else:
            return self.encoder(x)


class VICRegTemporalEncoder(nn.Module):
    """
    Temporal encoder with LSTM for sequence inputs.

    Processes [B, T, D] sequences and outputs [B, T, z_dim] embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        z_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Output projection
        self.output_proj = nn.Linear(lstm_out_dim, z_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode sequence to latent space.

        Args:
            x: [B, T, D] input sequence
            lengths: [B] sequence lengths

        Returns:
            z: [B, T, z_dim] embeddings
        """
        B, T, D = x.shape

        # Project input
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

        # Project to z
        z = self.output_proj(lstm_out)

        return z


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FULL VICReg MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class VICRegCrossModal(nn.Module):
    """
    Complete VICReg model for cross-modal learning.

    Includes separate encoders for audio and MIDI, plus VICReg loss.
    """

    def __init__(
        self,
        audio_input_dim: int = 128,
        midi_input_dim: int = 88,
        hidden_dim: int = 256,
        z_dim: int = 64,
        temporal: bool = True,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
    ):
        super().__init__()

        self.temporal = temporal
        self.z_dim = z_dim

        # Encoders
        if temporal:
            self.encoder_audio = VICRegTemporalEncoder(
                input_dim=audio_input_dim,
                hidden_dim=hidden_dim,
                z_dim=z_dim,
            )
            self.encoder_midi = VICRegTemporalEncoder(
                input_dim=midi_input_dim,
                hidden_dim=hidden_dim,
                z_dim=z_dim,
            )
        else:
            self.encoder_audio = VICRegEncoder(
                input_dim=audio_input_dim,
                hidden_dim=hidden_dim,
                z_dim=z_dim,
            )
            self.encoder_midi = VICRegEncoder(
                input_dim=midi_input_dim,
                hidden_dim=hidden_dim,
                z_dim=z_dim,
            )

        # Loss
        self.loss_fn = VICRegLoss(
            lambda_inv=lambda_inv,
            lambda_var=lambda_var,
            lambda_cov=lambda_cov,
        )

    def forward(
        self,
        audio: torch.Tensor,
        midi: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            audio: [B, T, D_audio] or [B, D_audio] audio features
            midi: [B, T, D_midi] or [B, D_midi] MIDI features
            lengths: [B] sequence lengths (if temporal)

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

        # Compute loss
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
        pool: str = 'mean',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings for retrieval.

        Args:
            pool: 'mean' to average over time, 'last' for last frame

        Returns:
            z_audio, z_midi: [B, z_dim] pooled embeddings
        """
        with torch.no_grad():
            if self.temporal and lengths is not None:
                z_audio = self.encoder_audio(audio, lengths)
                z_midi = self.encoder_midi(midi, lengths)
            else:
                z_audio = self.encoder_audio(audio)
                z_midi = self.encoder_midi(midi)

            # Pool
            if z_audio.dim() == 3:
                if pool == 'mean':
                    z_audio = z_audio.mean(dim=1)
                    z_midi = z_midi.mean(dim=1)
                else:  # last
                    z_audio = z_audio[:, -1]
                    z_midi = z_midi[:, -1]

        return z_audio, z_midi

    def check_collapse(self) -> Dict[str, float]:
        """Check if model has collapsed (all embeddings similar)."""
        # This should be called during validation with actual data
        # Here we just return placeholder
        return {'collapsed': False}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("VICReg Test")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test data
    B, T, D_audio, D_midi = 8, 50, 128, 88

    audio = torch.randn(B, T, D_audio).to(device)
    midi = torch.randn(B, T, D_midi).to(device)
    lengths = torch.tensor([T, T-5, T-10, T-15, T-20, T-25, T-30, T-35]).to(device)

    # Model
    model = VICRegCrossModal(
        audio_input_dim=D_audio,
        midi_input_dim=D_midi,
        hidden_dim=128,
        z_dim=32,
        temporal=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Forward
    outputs = model(audio, midi, lengths)

    print(f"\nz_audio shape: {outputs['z_audio'].shape}")
    print(f"z_midi shape: {outputs['z_midi'].shape}")
    print(f"Total loss: {outputs['total'].item():.4f}")
    print(f"  Invariance: {outputs['invariance'].item():.4f}")
    print(f"  Variance: {outputs['variance'].item():.4f}")
    print(f"  Covariance: {outputs['covariance'].item():.4f}")

    # Get embeddings for retrieval
    z_a, z_m = model.get_embeddings(audio, midi, lengths)
    print(f"\nPooled embeddings: {z_a.shape}")

    print("\n" + "=" * 50)
    print("VICReg test passed!")
