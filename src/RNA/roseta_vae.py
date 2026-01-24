#!/usr/bin/env python3
"""
Roseta VAE - Dual-Domain Variational Autoencoder with InfoNCE Alignment
========================================================================

Multi-domain VAE for the Roseta Experiment that learns a shared latent
representation (z_shared) between Audio and Vibration domains.

Architecture:
-------------
Audio  ──► Encoder_A ──► [z_shared | z_private_A] ──► Decoder_A ──► Audio_recon
                              │
                         InfoNCE Loss (alignment)
                              │
Vibr.  ──► Encoder_B ──► [z_shared | z_private_B] ──► Decoder_B ──► Vibr_recon

Key Features:
- Factorized latent space: z = [z_shared | z_private]
- InfoNCE contrastive loss for cross-modal alignment
- Temporal processing with LSTM encoder/decoder
- Reconstruction loss for both domains
- KL divergence for regularization

Based on:
- Phideus VAE Temporal architecture
- Contrastive Multiview Coding (CMC)
- Cross-Modal Variational Autoencoders

Usage:
------
from src.RNA.roseta_vae import RosetaVAE, RosetaTrainer

model = RosetaVAE(
    input_bins=256,
    input_channels=3,
    z_shared_dim=32,
    z_private_dim=16,
    hidden_dim=128,
)

trainer = RosetaTrainer(model, device='cuda')
trainer.train(train_loader, val_loader, epochs=50)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────
# 1. InfoNCE Loss
# ───────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for cross-modal alignment.

    Given paired embeddings (z_a, z_b) from audio and vibration,
    encourages z_shared_a ≈ z_shared_b for the same timestamp.

    Loss = -log(exp(sim(z_a, z_b) / τ) / Σ_j exp(sim(z_a, z_j) / τ))

    Where sim(a, b) = a · b / (||a|| ||b||) (cosine similarity)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_audio: torch.Tensor,
        z_vibration: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z_audio: [B, D] audio embeddings (z_shared)
            z_vibration: [B, D] vibration embeddings (z_shared)

        Returns:
            loss: scalar tensor
        """
        # L2 normalize
        z_a = F.normalize(z_audio, dim=-1)
        z_v = F.normalize(z_vibration, dim=-1)

        # Cosine similarity matrix [B, B]
        logits = torch.matmul(z_a, z_v.T) / self.temperature

        # Positive pairs are on diagonal
        labels = torch.arange(z_a.size(0), device=z_a.device)

        # Cross entropy loss (both directions)
        loss_a_to_v = F.cross_entropy(logits, labels)
        loss_v_to_a = F.cross_entropy(logits.T, labels)

        return (loss_a_to_v + loss_v_to_a) / 2


class TemporalInfoNCELoss(nn.Module):
    """
    InfoNCE loss for temporal sequences.

    Handles sequences [B, T, D] by comparing frames at each timestep.
    """

    def __init__(self, temperature: float = 0.07, sample_frames: int = 10):
        super().__init__()
        self.temperature = temperature
        self.sample_frames = sample_frames
        self.infonce = InfoNCELoss(temperature)

    def forward(
        self,
        z_audio_seq: torch.Tensor,
        z_vibration_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute temporal InfoNCE loss.

        Args:
            z_audio_seq: [B, T, D] audio sequence
            z_vibration_seq: [B, T, D] vibration sequence
            lengths: [B] actual lengths (for padding)

        Returns:
            loss: scalar tensor
        """
        B, T, D = z_audio_seq.shape

        # Sample random frames to reduce computation
        if T > self.sample_frames:
            frame_indices = torch.randperm(T, device=z_audio_seq.device)[:self.sample_frames]
            z_a = z_audio_seq[:, frame_indices]      # [B, S, D]
            z_v = z_vibration_seq[:, frame_indices]  # [B, S, D]
            S = self.sample_frames
        else:
            z_a = z_audio_seq
            z_v = z_vibration_seq
            S = T

        # Flatten to [B*S, D]
        z_a_flat = z_a.reshape(-1, D)
        z_v_flat = z_v.reshape(-1, D)

        return self.infonce(z_a_flat, z_v_flat)


# ───────────────────────────────────
# 2. Encoder/Decoder Modules
# ───────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    LSTM-based temporal encoder for histogram sequences.

    Input: [B, T, bins, channels] → Output: [B, T, z_dim] (mean, logvar)
    """

    def __init__(
        self,
        input_bins: int = 256,
        input_channels: int = 3,
        hidden_dim: int = 128,
        z_shared_dim: int = 32,
        z_private_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_bins * input_channels
        self.hidden_dim = hidden_dim
        self.z_shared_dim = z_shared_dim
        self.z_private_dim = z_private_dim
        self.z_dim = z_shared_dim + z_private_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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

        # Output heads for z_shared and z_private
        lstm_out_dim = hidden_dim * 2  # bidirectional

        # z_shared (for cross-modal alignment)
        self.shared_mean = nn.Linear(lstm_out_dim, z_shared_dim)
        self.shared_logvar = nn.Linear(lstm_out_dim, z_shared_dim)

        # z_private (domain-specific)
        self.private_mean = nn.Linear(lstm_out_dim, z_private_dim)
        self.private_logvar = nn.Linear(lstm_out_dim, z_private_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode sequence to latent distributions.

        Args:
            x: [B, T, bins, channels]
            lengths: [B] actual lengths

        Returns:
            z_shared_mean: [B, T, z_shared_dim]
            z_shared_logvar: [B, T, z_shared_dim]
            z_private_mean: [B, T, z_private_dim]
            z_private_logvar: [B, T, z_private_dim]
        """
        B, T, bins, channels = x.shape

        # Flatten spatial dimensions
        x_flat = x.view(B, T, -1)  # [B, T, bins*channels]

        # Project
        h = self.input_proj(x_flat)  # [B, T, hidden_dim]

        # LSTM encoding
        if lengths is not None:
            # Pack for variable lengths
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.lstm(h)  # [B, T, hidden_dim*2]

        # Split into z_shared and z_private
        z_shared_mean = self.shared_mean(lstm_out)
        z_shared_logvar = self.shared_logvar(lstm_out)
        z_private_mean = self.private_mean(lstm_out)
        z_private_logvar = self.private_logvar(lstm_out)

        return z_shared_mean, z_shared_logvar, z_private_mean, z_private_logvar


class TemporalDecoder(nn.Module):
    """
    LSTM-based temporal decoder for histogram reconstruction.

    Input: [B, T, z_dim] → Output: [B, T, bins, channels]
    """

    def __init__(
        self,
        output_bins: int = 256,
        output_channels: int = 3,
        hidden_dim: int = 128,
        z_dim: int = 48,  # z_shared + z_private
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

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to histogram sequence.

        Args:
            z: [B, T, z_dim] concatenated [z_shared | z_private]

        Returns:
            recon: [B, T, bins, channels]
        """
        B, T, _ = z.shape

        # Project z
        h = self.z_proj(z)  # [B, T, hidden_dim]

        # LSTM decoding
        lstm_out, _ = self.lstm(h)  # [B, T, hidden_dim*2]

        # Output projection
        out = self.output_proj(lstm_out)  # [B, T, bins*channels]

        # Reshape to [B, T, bins, channels]
        out = out.view(B, T, self.output_bins, self.output_channels)

        # Softmax over bins for each channel (probability distribution)
        out = F.softmax(out, dim=2)

        return out


# ───────────────────────────────────
# 3. Roseta VAE Model
# ───────────────────────────────────

class RosetaVAE(nn.Module):
    """
    Dual-domain VAE for Roseta experiment.

    Learns aligned representations between audio and vibration
    using a factorized latent space with InfoNCE alignment.
    """

    def __init__(
        self,
        input_bins: int = 256,
        input_channels: int = 3,
        hidden_dim: int = 128,
        z_shared_dim: int = 32,
        z_private_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        infonce_temperature: float = 0.07,
    ):
        super().__init__()

        self.input_bins = input_bins
        self.input_channels = input_channels
        self.z_shared_dim = z_shared_dim
        self.z_private_dim = z_private_dim
        self.z_dim = z_shared_dim + z_private_dim

        # Audio encoder/decoder
        self.encoder_audio = TemporalEncoder(
            input_bins=input_bins,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            z_shared_dim=z_shared_dim,
            z_private_dim=z_private_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder_audio = TemporalDecoder(
            output_bins=input_bins,
            output_channels=input_channels,
            hidden_dim=hidden_dim,
            z_dim=self.z_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Vibration encoder/decoder
        self.encoder_vibration = TemporalEncoder(
            input_bins=input_bins,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            z_shared_dim=z_shared_dim,
            z_private_dim=z_private_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder_vibration = TemporalDecoder(
            output_bins=input_bins,
            output_channels=input_channels,
            hidden_dim=hidden_dim,
            z_dim=self.z_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # InfoNCE loss for z_shared alignment
        self.infonce_loss = TemporalInfoNCELoss(
            temperature=infonce_temperature,
            sample_frames=10,
        )

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def encode_audio(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode audio to latent distributions."""
        z_sh_mean, z_sh_logvar, z_pr_mean, z_pr_logvar = self.encoder_audio(x, lengths)

        z_shared = self.reparameterize(z_sh_mean, z_sh_logvar)
        z_private = self.reparameterize(z_pr_mean, z_pr_logvar)

        return {
            'z_shared': z_shared,
            'z_shared_mean': z_sh_mean,
            'z_shared_logvar': z_sh_logvar,
            'z_private': z_private,
            'z_private_mean': z_pr_mean,
            'z_private_logvar': z_pr_logvar,
        }

    def encode_vibration(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode vibration to latent distributions."""
        z_sh_mean, z_sh_logvar, z_pr_mean, z_pr_logvar = self.encoder_vibration(x, lengths)

        z_shared = self.reparameterize(z_sh_mean, z_sh_logvar)
        z_private = self.reparameterize(z_pr_mean, z_pr_logvar)

        return {
            'z_shared': z_shared,
            'z_shared_mean': z_sh_mean,
            'z_shared_logvar': z_sh_logvar,
            'z_private': z_private,
            'z_private_mean': z_pr_mean,
            'z_private_logvar': z_pr_logvar,
        }

    def decode_audio(
        self,
        z_shared: torch.Tensor,
        z_private: torch.Tensor,
        dropout_shared: float = 0.0,
    ) -> torch.Tensor:
        """
        Decode audio from latent.

        Args:
            z_shared: shared latent representation
            z_private: private (modality-specific) latent
            dropout_shared: dropout probability for z_shared during training
                           (forces decoder to rely more on z_private)
        """
        if self.training and dropout_shared > 0:
            z_shared = F.dropout(z_shared, p=dropout_shared, training=True)
        z = torch.cat([z_shared, z_private], dim=-1)
        return self.decoder_audio(z)

    def decode_vibration(
        self,
        z_shared: torch.Tensor,
        z_private: torch.Tensor,
        dropout_shared: float = 0.0,
    ) -> torch.Tensor:
        """
        Decode vibration from latent.

        Args:
            z_shared: shared latent representation
            z_private: private (modality-specific) latent
            dropout_shared: dropout probability for z_shared during training
        """
        if self.training and dropout_shared > 0:
            z_shared = F.dropout(z_shared, p=dropout_shared, training=True)
        z = torch.cat([z_shared, z_private], dim=-1)
        return self.decoder_vibration(z)

    def forward(
        self,
        audio: torch.Tensor,
        vibration: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        dropout_shared: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both domains.

        Args:
            audio: [B, T, bins, channels]
            vibration: [B, T, bins, channels]
            lengths: [B] actual sequence lengths
            dropout_shared: dropout on z_shared during decoding (forces z_private usage)

        Returns:
            dict with all outputs and latents
        """
        # Encode both domains
        audio_enc = self.encode_audio(audio, lengths)
        vibration_enc = self.encode_vibration(vibration, lengths)

        # Decode both domains (self-reconstruction)
        audio_recon = self.decode_audio(
            audio_enc['z_shared'],
            audio_enc['z_private'],
            dropout_shared=dropout_shared,
        )
        vibration_recon = self.decode_vibration(
            vibration_enc['z_shared'],
            vibration_enc['z_private'],
            dropout_shared=dropout_shared,
        )

        # Cross-reconstruction (for evaluation - no dropout)
        # Audio z_shared → Vibration decoder (using random z_private)
        cross_vibration = self.decode_vibration(
            audio_enc['z_shared'],
            vibration_enc['z_private'],  # Use target's private
            dropout_shared=0.0,  # No dropout for cross-recon
        )

        return {
            # Reconstructions
            'audio_recon': audio_recon,
            'vibration_recon': vibration_recon,
            'cross_vibration': cross_vibration,
            # Audio latents
            'audio_z_shared': audio_enc['z_shared'],
            'audio_z_shared_mean': audio_enc['z_shared_mean'],
            'audio_z_shared_logvar': audio_enc['z_shared_logvar'],
            'audio_z_private': audio_enc['z_private'],
            'audio_z_private_mean': audio_enc['z_private_mean'],
            'audio_z_private_logvar': audio_enc['z_private_logvar'],
            # Vibration latents
            'vibration_z_shared': vibration_enc['z_shared'],
            'vibration_z_shared_mean': vibration_enc['z_shared_mean'],
            'vibration_z_shared_logvar': vibration_enc['z_shared_logvar'],
            'vibration_z_private': vibration_enc['z_private'],
            'vibration_z_private_mean': vibration_enc['z_private_mean'],
            'vibration_z_private_logvar': vibration_enc['z_private_logvar'],
        }

    def compute_loss(
        self,
        audio: torch.Tensor,
        vibration: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        lengths: Optional[torch.Tensor] = None,
        beta_kl: float = 1.0,
        beta_kl_shared: Optional[float] = None,
        beta_kl_private: Optional[float] = None,
        lambda_infonce: float = 1.0,
        lambda_diff: float = 0.0,
        diff_margin: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Total loss = recon_audio + recon_vibration + β_shared*KL_shared + β_private*KL_private
                     + λ_infonce*InfoNCE + λ_diff*diff_loss

        Args:
            audio: [B, T, bins, channels] input
            vibration: [B, T, bins, channels] input
            outputs: forward() outputs
            lengths: sequence lengths
            beta_kl: KL divergence weight (legacy, used if shared/private not specified)
            beta_kl_shared: KL weight for z_shared (default: beta_kl)
            beta_kl_private: KL weight for z_private (use low value like 0.01 to prevent collapse)
            lambda_infonce: InfoNCE weight
            lambda_diff: Weight for z_private differentiation loss (pushes modalities apart)
            diff_margin: Margin for hinge loss on z_private difference

        Returns:
            dict with all loss components
        """
        # Handle legacy beta_kl parameter
        if beta_kl_shared is None:
            beta_kl_shared = beta_kl
        if beta_kl_private is None:
            beta_kl_private = beta_kl

        # Reconstruction loss (MSE)
        recon_audio = F.mse_loss(outputs['audio_recon'], audio)
        recon_vibration = F.mse_loss(outputs['vibration_recon'], vibration)

        # KL divergence loss (both domains)
        def kl_divergence(mean, logvar):
            return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        kl_audio_shared = kl_divergence(
            outputs['audio_z_shared_mean'],
            outputs['audio_z_shared_logvar']
        )
        kl_audio_private = kl_divergence(
            outputs['audio_z_private_mean'],
            outputs['audio_z_private_logvar']
        )
        kl_vibration_shared = kl_divergence(
            outputs['vibration_z_shared_mean'],
            outputs['vibration_z_shared_logvar']
        )
        kl_vibration_private = kl_divergence(
            outputs['vibration_z_private_mean'],
            outputs['vibration_z_private_logvar']
        )

        # Separate KL for shared vs private (WP3: fix z_private collapse)
        kl_shared = kl_audio_shared + kl_vibration_shared
        kl_private = kl_audio_private + kl_vibration_private
        kl_total = kl_shared + kl_private  # For logging

        # InfoNCE alignment loss
        infonce = self.infonce_loss(
            outputs['audio_z_shared'],
            outputs['vibration_z_shared'],
            lengths
        )

        # Differentiation loss: penalize if z_private_audio ≈ z_private_vib
        # This encourages modality-specific information in z_private
        if lambda_diff > 0:
            # Use mean representations for stable comparison
            z_priv_a = outputs['audio_z_private_mean']  # [B, T, D]
            z_priv_v = outputs['vibration_z_private_mean']

            # L2 distance between z_private of different modalities
            diff = torch.norm(z_priv_a - z_priv_v, dim=-1)  # [B, T]

            # Hinge loss: want diff > margin, so penalize if diff < margin
            diff_loss = F.relu(diff_margin - diff).mean()
        else:
            diff_loss = torch.tensor(0.0, device=audio.device)

        # Total loss with separate KL weights
        total_loss = (recon_audio + recon_vibration +
                      beta_kl_shared * kl_shared +
                      beta_kl_private * kl_private +
                      lambda_infonce * infonce +
                      lambda_diff * diff_loss)

        return {
            'total': total_loss,
            'recon_audio': recon_audio,
            'recon_vibration': recon_vibration,
            'kl_total': kl_total,
            'kl_shared': kl_shared,
            'kl_private': kl_private,
            'kl_audio_shared': kl_audio_shared,
            'kl_audio_private': kl_audio_private,
            'kl_vibration_shared': kl_vibration_shared,
            'kl_vibration_private': kl_vibration_private,
            'infonce': infonce,
            'diff_loss': diff_loss,
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ───────────────────────────────────
# 4. Evaluation Metrics
# ───────────────────────────────────

def compute_alignment_metrics(
    z_audio: torch.Tensor,
    z_vibration: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute alignment metrics between audio and vibration z_shared.

    Args:
        z_audio: [N, D] audio embeddings
        z_vibration: [N, D] vibration embeddings

    Returns:
        dict with metrics
    """
    # Flatten if needed
    if z_audio.dim() == 3:
        z_audio = z_audio.view(-1, z_audio.size(-1))
        z_vibration = z_vibration.view(-1, z_vibration.size(-1))

    # Cosine similarity (paired)
    z_a = F.normalize(z_audio, dim=-1)
    z_v = F.normalize(z_vibration, dim=-1)
    cos_sim = (z_a * z_v).sum(dim=-1).mean().item()

    # L2 distance
    l2_dist = torch.norm(z_audio - z_vibration, dim=-1).mean().item()

    # InfoNCE accuracy (retrieval)
    logits = torch.matmul(z_a, z_v.T)
    labels = torch.arange(z_a.size(0), device=z_a.device)
    preds = logits.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()

    return {
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'retrieval_accuracy': accuracy,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing RosetaVAE...")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RosetaVAE(
        input_bins=256,
        input_channels=3,
        hidden_dim=128,
        z_shared_dim=32,
        z_private_dim=16,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    B, T, bins, channels = 4, 100, 256, 3
    audio = torch.randn(B, T, bins, channels).to(device)
    vibration = torch.randn(B, T, bins, channels).to(device)
    lengths = torch.tensor([100, 80, 90, 70])

    outputs = model(audio, vibration, lengths)
    losses = model.compute_loss(audio, vibration, outputs, lengths)

    print(f"\nInput shapes: audio={audio.shape}, vibration={vibration.shape}")
    print(f"Recon shapes: audio={outputs['audio_recon'].shape}")
    print(f"z_shared shape: {outputs['audio_z_shared'].shape}")

    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Test alignment metrics
    metrics = compute_alignment_metrics(
        outputs['audio_z_shared'],
        outputs['vibration_z_shared']
    )
    print(f"\nAlignment metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n✔ RosetaVAE test passed!")
