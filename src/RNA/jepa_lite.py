#!/usr/bin/env python3
"""
JEPA-Lite - Joint Embedding Predictive Architecture without Decoder (Fase 3A)
==============================================================================

JEPA-lite learns cross-modal correspondence by predicting one modality's
latent representation from the other, WITHOUT using a decoder.

This avoids the "reconstruction shortcut" problem where the VAE learns to
reconstruct from z_shared alone, ignoring cross-modal structure.

Architecture:
-------------
Audio  ──► Encoder ──► z_audio ──► Predictor ──► z_vib_pred
                                        ↓
                                   Loss: sim(z_vib_pred, z_vib)
                                        ↑
Vibr.  ──► Encoder ──► z_vib ───────────┘

Key Features:
- No decoder (avoids reconstruction shortcut)
- Predictor learns audio→vib and vib→audio mapping
- Contrastive loss on predicted vs actual embeddings
- Stop-gradient on target to prevent collapse

Configurations:
- C5: MLP+Attention encoder + JEPA predictor
- C6: Transformer encoder + JEPA predictor

Based on:
- JEPA (Joint Embedding Predictive Architecture) - LeCun et al.
- Audio-JEPA (2025)
- VL-JEPA for Vision-Language (2025)

Usage:
------
from src.RNA.jepa_lite import JEPALite

model = JEPALite(
    encoder_type='mlp',  # or 'transformer'
    token_dim=5,
    max_tokens=48,
    hidden_dim=128,
    z_dim=64,
)

outputs = model(audio_tokens, audio_mask, vib_tokens, vib_mask, lengths)
loss = model.compute_loss(outputs)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.RNA.constellation_vae import MLPConstellationEncoder, TransformerConstellationEncoder
except ImportError:
    from constellation_vae import MLPConstellationEncoder, TransformerConstellationEncoder


class JEPAPredictor(nn.Module):
    """
    Predictor network that maps z_source → z_target_pred.

    Simple MLP that learns the cross-modal mapping in latent space.
    """

    def __init__(
        self,
        z_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = z_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, z_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict target embedding from source embedding.

        Args:
            z: [B, T, z_dim] source embedding

        Returns:
            z_pred: [B, T, z_dim] predicted target embedding
        """
        return self.mlp(z)


class JEPALite(nn.Module):
    """
    JEPA-Lite model for cross-modal learning without decoder.

    Configurations:
    - C5: encoder_type='mlp' (MLP + Attention pooling + LSTM)
    - C6: encoder_type='transformer' (Full transformer)
    """

    def __init__(
        self,
        encoder_type: Literal['mlp', 'transformer'] = 'mlp',
        token_dim: int = 5,
        max_tokens: int = 48,
        hidden_dim: int = 128,
        z_dim: int = 64,
        predictor_hidden_dim: int = 128,
        predictor_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.07,
        use_ema_target: bool = False,
        ema_decay: float = 0.996,
    ):
        """
        Initialize JEPA-Lite model.

        Args:
            encoder_type: 'mlp' or 'transformer'
            token_dim: Dimension of each token (5 for constellation)
            max_tokens: Maximum tokens per frame (48)
            hidden_dim: Encoder hidden dimension
            z_dim: Latent dimension (no shared/private split in JEPA)
            predictor_hidden_dim: Predictor MLP hidden dimension
            predictor_layers: Number of predictor layers
            dropout: Dropout rate
            temperature: InfoNCE temperature
            use_ema_target: Use EMA encoder for target (like BYOL)
            ema_decay: EMA decay rate
        """
        super().__init__()

        self.encoder_type = encoder_type
        self.z_dim = z_dim
        self.temperature = temperature
        self.use_ema_target = use_ema_target
        self.ema_decay = ema_decay

        # Shared encoder for both modalities
        # Note: We use z_shared_dim=z_dim and z_private_dim=0 to get single z
        if encoder_type == 'mlp':
            self.encoder = MLPConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,  # No private component
                dropout=dropout,
            )
        else:
            self.encoder = TransformerConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )

        # Predictors: audio→vib and vib→audio
        self.predictor_a2v = JEPAPredictor(
            z_dim=z_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )
        self.predictor_v2a = JEPAPredictor(
            z_dim=z_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        # Optional: EMA encoder for target (prevents collapse)
        if use_ema_target:
            if encoder_type == 'mlp':
                self.target_encoder = MLPConstellationEncoder(
                    token_dim=token_dim,
                    max_tokens=max_tokens,
                    hidden_dim=hidden_dim,
                    z_shared_dim=z_dim,
                    z_private_dim=0,
                    dropout=dropout,
                )
            else:
                self.target_encoder = TransformerConstellationEncoder(
                    token_dim=token_dim,
                    max_tokens=max_tokens,
                    hidden_dim=hidden_dim,
                    z_shared_dim=z_dim,
                    z_private_dim=0,
                    dropout=dropout,
                )
            # Initialize with same weights
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            # Freeze target encoder (updated via EMA)
            for param in self.target_encoder.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with EMA of online encoder."""
        if not self.use_ema_target:
            return

        for online_param, target_param in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = (
                self.ema_decay * target_param.data +
                (1 - self.ema_decay) * online_param.data
            )

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        use_target_encoder: bool = False,
    ) -> torch.Tensor:
        """
        Encode tokens to latent space.

        Args:
            tokens: [B, T, K, D] constellation tokens
            mask: [B, T, K] validity mask
            lengths: [B] sequence lengths
            use_target_encoder: Use EMA target encoder (for BYOL-style)

        Returns:
            z: [B, T, z_dim] latent embedding
        """
        if use_target_encoder and self.use_ema_target:
            encoder = self.target_encoder
        else:
            encoder = self.encoder

        # Encoder returns (z_shared_mean, z_shared_logvar, z_private_mean, z_private_logvar)
        # With z_private_dim=0, we only use z_shared
        z_mean, z_logvar, _, _ = encoder(tokens, mask, lengths)

        # For JEPA, we use the mean directly (no sampling)
        return z_mean

    def forward(
        self,
        audio_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
        vib_tokens: torch.Tensor,
        vib_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA-Lite.

        Args:
            audio_tokens: [B, T, K, D] audio constellation tokens
            audio_mask: [B, T, K] audio validity mask
            vib_tokens: [B, T, K, D] vibration tokens
            vib_mask: [B, T, K] vibration validity mask
            lengths: [B] sequence lengths

        Returns:
            Dict with embeddings and predictions
        """
        # Encode both modalities (online encoder)
        z_audio = self.encode(audio_tokens, audio_mask, lengths)
        z_vib = self.encode(vib_tokens, vib_mask, lengths)

        # For target embeddings (with stop gradient or EMA encoder)
        if self.use_ema_target:
            with torch.no_grad():
                z_audio_target = self.encode(audio_tokens, audio_mask, lengths, use_target_encoder=True)
                z_vib_target = self.encode(vib_tokens, vib_mask, lengths, use_target_encoder=True)
        else:
            # Stop gradient on target
            z_audio_target = z_audio.detach()
            z_vib_target = z_vib.detach()

        # Predict cross-modal
        z_vib_pred = self.predictor_a2v(z_audio)  # audio → vib
        z_audio_pred = self.predictor_v2a(z_vib)  # vib → audio

        return {
            'z_audio': z_audio,
            'z_vib': z_vib,
            'z_audio_target': z_audio_target,
            'z_vib_target': z_vib_target,
            'z_vib_pred': z_vib_pred,
            'z_audio_pred': z_audio_pred,
        }

    def compute_prediction_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        loss_type: str = 'cosine',
    ) -> torch.Tensor:
        """
        Compute prediction loss between predicted and target embeddings.

        Args:
            z_pred: [B, T, D] predicted embedding
            z_target: [B, T, D] target embedding (should be detached)
            loss_type: 'cosine', 'mse', or 'smooth_l1'

        Returns:
            loss: scalar tensor
        """
        if loss_type == 'cosine':
            # Negative cosine similarity (we want to maximize similarity)
            z_pred_norm = F.normalize(z_pred, dim=-1)
            z_target_norm = F.normalize(z_target, dim=-1)
            loss = 2 - 2 * (z_pred_norm * z_target_norm).sum(dim=-1).mean()
        elif loss_type == 'mse':
            loss = F.mse_loss(z_pred, z_target)
        else:  # smooth_l1
            loss = F.smooth_l1_loss(z_pred, z_target)

        return loss

    def compute_infonce_loss(
        self,
        z_audio: torch.Tensor,
        z_vib: torch.Tensor,
        sample_frames: int = 10,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss for alignment.

        Similar to ConstellationVAE but on the direct embeddings.
        """
        B, T, D = z_audio.shape

        # Sample frames if needed
        if T > sample_frames:
            indices = torch.randperm(T, device=z_audio.device)[:sample_frames]
            z_a = z_audio[:, indices]
            z_v = z_vib[:, indices]
        else:
            z_a = z_audio
            z_v = z_vib

        # Flatten
        z_a = z_a.reshape(-1, D)
        z_v = z_v.reshape(-1, D)

        # Normalize
        z_a = F.normalize(z_a, dim=-1)
        z_v = F.normalize(z_v, dim=-1)

        # Cosine similarity
        logits = torch.matmul(z_a, z_v.T) / self.temperature
        labels = torch.arange(z_a.size(0), device=z_a.device)

        loss_a2v = F.cross_entropy(logits, labels)
        loss_v2a = F.cross_entropy(logits.T, labels)

        return (loss_a2v + loss_v2a) / 2

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        lambda_pred: float = 1.0,
        lambda_infonce: float = 1.0,
        pred_loss_type: str = 'cosine',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total JEPA-Lite loss.

        Args:
            outputs: Forward pass outputs
            lambda_pred: Weight for prediction loss
            lambda_infonce: Weight for InfoNCE loss
            pred_loss_type: Type of prediction loss

        Returns:
            Dict with loss components
        """
        losses = {}

        # Prediction losses (bidirectional)
        losses['pred_a2v'] = lambda_pred * self.compute_prediction_loss(
            outputs['z_vib_pred'],
            outputs['z_vib_target'],
            pred_loss_type,
        )
        losses['pred_v2a'] = lambda_pred * self.compute_prediction_loss(
            outputs['z_audio_pred'],
            outputs['z_audio_target'],
            pred_loss_type,
        )

        # InfoNCE contrastive loss
        losses['infonce'] = lambda_infonce * self.compute_infonce_loss(
            outputs['z_audio'],
            outputs['z_vib'],
        )

        # Total
        losses['total'] = losses['pred_a2v'] + losses['pred_v2a'] + losses['infonce']

        return losses

    def get_embeddings(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get embeddings for evaluation (no gradient).

        Args:
            tokens: [B, T, K, D] constellation tokens
            mask: [B, T, K] validity mask
            lengths: [B] sequence lengths

        Returns:
            z: [B, T, z_dim] embeddings
        """
        with torch.no_grad():
            return self.encode(tokens, mask, lengths)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Testing JEPA-Lite configurations...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test data
    B, T, K, D = 4, 50, 48, 5
    audio_tokens = torch.randn(B, T, K, D).to(device)
    audio_mask = (torch.rand(B, T, K) > 0.3).float().to(device)
    vib_tokens = torch.randn(B, T, K, D).to(device)
    vib_mask = (torch.rand(B, T, K) > 0.3).float().to(device)
    lengths = torch.tensor([T, T-10, T-20, T-30]).to(device)

    configs = [
        ('mlp', False),      # C5: MLP encoder, no EMA
        ('mlp', True),       # C5 variant with EMA
        ('transformer', False),  # C6: Transformer encoder
    ]

    for enc_type, use_ema in configs:
        ema_str = "+EMA" if use_ema else ""
        print(f"\nConfig: {enc_type}{ema_str}")

        model = JEPALite(
            encoder_type=enc_type,
            token_dim=D,
            max_tokens=K,
            hidden_dim=64,
            z_dim=32,
            predictor_hidden_dim=64,
            use_ema_target=use_ema,
        ).to(device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")

        # Forward pass
        outputs = model(audio_tokens, audio_mask, vib_tokens, vib_mask, lengths)
        print(f"  z_audio shape: {outputs['z_audio'].shape}")
        print(f"  z_vib_pred shape: {outputs['z_vib_pred'].shape}")

        # Compute loss
        losses = model.compute_loss(outputs)
        print(f"  Total loss: {losses['total'].item():.4f}")
        print(f"  Pred A→V: {losses['pred_a2v'].item():.4f}")
        print(f"  InfoNCE: {losses['infonce'].item():.4f}")

        # Test EMA update
        if use_ema:
            model.update_target_encoder()
            print("  ✓ EMA update successful")

    print("\n" + "=" * 60)
    print("✔ All JEPA-Lite configurations tested successfully!")
