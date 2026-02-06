"""
Cross-Modal Model for Audio-MIDI learning.

Combines:
- MERT encoder (frozen) for audio
- Transformer encoder for MIDI
- Projection heads for shared embedding space
- Optional DANN for domain adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from ..encoders.mert_encoder import MERTEncoder, MERTEncoderLite
from ..encoders.midi_encoder import MIDIEncoder
from ..encoders.projection import ProjectionHead
from ..losses.dann import DANNLoss

logger = logging.getLogger(__name__)


class CrossModalModel(nn.Module):
    """
    Cross-modal model for Audio-MIDI learning.

    Architecture:
        Audio → MERT → Projection → Audio Embedding
        MIDI  → Transformer → Projection → MIDI Embedding

    Training with VICReg loss to align embeddings.
    Optional DANN for domain-invariant representations.
    """

    def __init__(
        self,
        # Audio encoder config
        audio_encoder: str = "mert",  # "mert" or "lite"
        audio_encoder_frozen: bool = True,
        audio_hidden_dim: int = 1024,  # MERT output dim

        # MIDI encoder config
        midi_embed_dim: int = 512,
        midi_n_layers: int = 4,
        midi_n_heads: int = 8,
        midi_dropout: float = 0.1,

        # Projection config
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 256,
        proj_n_layers: int = 3,

        # DANN config (optional)
        use_dann: bool = False,
        dann_hidden_dim: int = 64,
        dann_lambda_schedule: str = "linear_0_to_1",
        dann_max_steps: int = 10000,
        dann_dropout: float = 0.1,
        dann_lambda_max: float = 1.0,
        dann_warmup_steps: int = 0,
        dann_ramp_steps: int = 0,

        # VICReg config
        vicreg_invariance_weight: float = 10.0,
        vicreg_variance_weight: float = 10.0,
        vicreg_covariance_weight: float = 1.0,

        # Device
        device: Optional[str] = None,
    ):
        super().__init__()

        self.device = device
        self.use_dann = use_dann
        self.proj_output_dim = proj_output_dim

        # Audio encoder
        if audio_encoder == "mert":
            self.audio_encoder = MERTEncoder(
                freeze=audio_encoder_frozen,
                device=device,
            )
        elif audio_encoder == "lite":
            self.audio_encoder = MERTEncoderLite(
                output_dim=audio_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown audio encoder: {audio_encoder}")

        audio_out_dim = self.audio_encoder.get_output_dim()

        # MIDI encoder
        self.midi_encoder = MIDIEncoder(
            embed_dim=midi_embed_dim,
            n_layers=midi_n_layers,
            n_heads=midi_n_heads,
            dropout=midi_dropout,
        )
        midi_out_dim = self.midi_encoder.get_output_dim()

        # Projection heads
        self.audio_projection = ProjectionHead(
            input_dim=audio_out_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            n_layers=proj_n_layers,
        )

        self.midi_projection = ProjectionHead(
            input_dim=midi_out_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            n_layers=proj_n_layers,
        )

        # DANN (optional)
        if use_dann:
            self.dann = DANNLoss(
                input_dim=proj_output_dim,
                hidden_dim=dann_hidden_dim,
                lambda_schedule=dann_lambda_schedule,
                max_steps=dann_max_steps,
                dropout=dann_dropout,
                lambda_max=dann_lambda_max,
                warmup_steps=dann_warmup_steps,
                ramp_steps=dann_ramp_steps,
            )
        else:
            self.dann = None

        # VICReg weights
        self.vicreg_invariance_weight = vicreg_invariance_weight
        self.vicreg_variance_weight = vicreg_variance_weight
        self.vicreg_covariance_weight = vicreg_covariance_weight

        logger.info(
            f"CrossModalModel: audio={audio_encoder}({audio_out_dim}), "
            f"midi={midi_out_dim}, proj={proj_output_dim}, dann={use_dann}"
        )

    def encode_audio(
        self,
        audio: torch.Tensor,
        return_projected: bool = True
    ) -> torch.Tensor:
        """
        Encode audio to embedding.

        Args:
            audio: [B, T] waveform
            return_projected: If True, return projected embedding

        Returns:
            [B, D] embedding
        """
        features = self.audio_encoder(audio)

        if return_projected:
            return self.audio_projection(features)
        return features

    def encode_midi(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        duration: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_projected: bool = True
    ) -> torch.Tensor:
        """
        Encode MIDI to embedding.

        Args:
            pitch: [B, T] pitch values
            velocity: [B, T] velocity values
            duration: [B, T] duration buckets
            padding_mask: [B, T] True for padding
            return_projected: If True, return projected embedding

        Returns:
            [B, D] embedding
        """
        features = self.midi_encoder(
            pitch=pitch,
            velocity=velocity,
            duration=duration,
            padding_mask=padding_mask,
        )

        if return_projected:
            return self.midi_projection(features)
        return features

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            audio: [B, T] waveform
            midi_pitch: [B, N] pitches
            midi_velocity: [B, N] velocities
            midi_duration: [B, N] duration buckets
            midi_mask: [B, N] padding mask

        Returns:
            audio_emb: [B, D]
            midi_emb: [B, D]
        """
        audio_emb = self.encode_audio(audio)
        midi_emb = self.encode_midi(
            pitch=midi_pitch,
            velocity=midi_velocity,
            duration=midi_duration,
            padding_mask=midi_mask,
        )
        return audio_emb, midi_emb

    def compute_vicreg_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VICReg loss.

        Args:
            z1: [B, D] first embedding
            z2: [B, D] second embedding

        Returns:
            loss: Scalar
            metrics: Dict with inv_loss, var_loss, cov_loss
        """
        # Invariance loss (MSE between pairs)
        inv_loss = F.mse_loss(z1, z2)

        # Variance loss (prevent collapse)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance loss (decorrelate dimensions)
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        B, D = z1.shape
        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)

        # Off-diagonal elements
        cov_loss = (
            self._off_diagonal(cov_z1).pow(2).sum() / D +
            self._off_diagonal(cov_z2).pow(2).sum() / D
        )

        # Total loss
        loss = (
            self.vicreg_invariance_weight * inv_loss +
            self.vicreg_variance_weight * var_loss +
            self.vicreg_covariance_weight * cov_loss
        )

        metrics = {
            'vicreg_loss': loss.item(),
            'inv_loss': inv_loss.item(),
            'var_loss': var_loss.item(),
            'cov_loss': cov_loss.item(),
            'std_z1': std_z1.mean().item(),
            'std_z2': std_z2.mean().item(),
        }

        return loss, metrics

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Extract off-diagonal elements."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        dann_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute full loss.

        Args:
            batch: Dict with audio, midi_pitch, midi_velocity, midi_duration, midi_mask
            dann_weight: Weight for DANN loss

        Returns:
            loss: Scalar
            metrics: Dict with all metrics
        """
        # Encode
        audio_emb, midi_emb = self(
            audio=batch['audio'],
            midi_pitch=batch['midi_pitch'],
            midi_velocity=batch['midi_velocity'],
            midi_duration=batch['midi_duration'],
            midi_mask=batch.get('midi_mask'),
        )

        # VICReg loss
        vicreg_loss, vicreg_metrics = self.compute_vicreg_loss(audio_emb, midi_emb)

        metrics = vicreg_metrics.copy()
        total_loss = vicreg_loss

        # DANN loss (optional)
        if self.use_dann and self.dann is not None:
            # Concatenate embeddings
            B = audio_emb.size(0)
            embeddings = torch.cat([audio_emb, midi_emb], dim=0)
            # L2-normalize before domain head so classifier cannot use
            # embedding magnitude as a trivial domain discriminator
            embeddings_norm = F.normalize(embeddings, dim=1)
            domain_labels = torch.cat([
                torch.zeros(B, dtype=torch.long, device=audio_emb.device),
                torch.ones(B, dtype=torch.long, device=audio_emb.device),
            ])

            dann_loss, dann_metrics = self.dann(embeddings_norm, domain_labels)
            total_loss = total_loss + dann_weight * dann_loss

            metrics.update({
                'dann_loss': dann_metrics['domain_loss'],
                'dann_accuracy': dann_metrics['domain_accuracy'],
                'dann_lambda': dann_metrics['lambda'],
            })

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def get_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings without computing loss.

        Returns:
            audio_emb: [B, D]
            midi_emb: [B, D]
        """
        with torch.no_grad():
            return self(
                audio=batch['audio'],
                midi_pitch=batch['midi_pitch'],
                midi_velocity=batch['midi_velocity'],
                midi_duration=batch['midi_duration'],
                midi_mask=batch.get('midi_mask'),
            )

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.device = str(device)
        return self


def compute_retrieval_metrics(
    audio_embeddings: torch.Tensor,
    midi_embeddings: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """
    Compute retrieval metrics.

    Args:
        audio_embeddings: [N, D] audio embeddings
        midi_embeddings: [N, D] MIDI embeddings
        k_values: K values for Recall@K

    Returns:
        Dict with Recall@K for both directions, MRR
    """
    N = audio_embeddings.size(0)

    # Normalize
    audio_norm = F.normalize(audio_embeddings, dim=1)
    midi_norm = F.normalize(midi_embeddings, dim=1)

    # Compute similarity matrix
    sim_matrix = audio_norm @ midi_norm.T  # [N, N]

    metrics = {}

    # Audio → MIDI retrieval
    for direction, sim in [("a2m", sim_matrix), ("m2a", sim_matrix.T)]:
        # Get rankings
        _, indices = sim.sort(dim=1, descending=True)

        # Ground truth: diagonal
        gt = torch.arange(N, device=sim.device)

        # Find rank of correct match
        ranks = (indices == gt.unsqueeze(1)).nonzero()[:, 1].float()

        # Recall@K
        for k in k_values:
            recall = (ranks < k).float().mean().item()
            metrics[f'{direction}_recall@{k}'] = recall

        # MRR
        mrr = (1.0 / (ranks + 1)).mean().item()
        metrics[f'{direction}_mrr'] = mrr

    # Gap between aligned and random
    aligned_sim = sim_matrix.diag().mean().item()

    # Random: mean of off-diagonal
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    random_sim = sim_matrix[mask].mean().item()

    metrics['aligned_similarity'] = aligned_sim
    metrics['random_similarity'] = random_sim
    metrics['gap'] = aligned_sim - random_sim

    return metrics
