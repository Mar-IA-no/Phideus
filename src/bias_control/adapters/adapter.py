"""
Adapter modules for Bloque A (Adapter/Unfreezing controlado).

AdapterBottleneck: Linear(d->bottleneck)->GELU->Linear(bottleneck->d)+residual.
BloqueAModel: Top-level wrapper around CrossModalModel that intercalates adapters
in the audio encoder's transformer layers without modifying the base model.

Design rationale (DEC-005 / Plan v1.1):
  - CrossModalModel is NOT modified — BloqueAModel wraps it entirely.
  - State dict keys stay clean: base_model.audio_encoder.* for original weights,
    audio_adapters.layer_N.* for new adapter parameters.
  - Forward parity with MERTEncoderLite including pos_embedding interpolation branch.
  - Compatible with extract_all_embeddings() signature: forward(audio, midi_pitch, ...).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class AdapterBottleneck(nn.Module):
    """
    Bottleneck adapter: Linear(d->bottleneck)->GELU->Linear(bottleneck->d) + residual.

    Near-identity init: up projection initialized to zeros so adapter starts
    as identity function and training can learn deviations.

    Params per adapter: d*bottleneck + bottleneck + bottleneck*d + d
    For d=1024, bottleneck=64: ~131K params.
    """

    def __init__(self, d_model: int = 1024, adapter_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, d_model)

        # Near-identity init: adapter output starts at zero
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, D] with residual."""
        return x + self.up(self.act(self.down(x)))


class BloqueAModel(nn.Module):
    """
    Top-level wrapper for Bloque A experiments.

    Wraps a CrossModalModel and optionally intercalates AdapterBottleneck modules
    after specified transformer layers in the audio encoder.

    State dict keys:
        base_model.audio_encoder.transformer.*    # original weights
        base_model.midi_encoder.*
        base_model.audio_projection.*
        base_model.midi_projection.*
        audio_adapters.layer_0.down.weight         # new adapter params
        audio_adapters.layer_0.up.weight
        ...

    Args:
        base_model: CrossModalModel instance (with MERTEncoderLite audio encoder).
        adapter_dim: Bottleneck dimension for adapters.
        adapter_layers: List of transformer layer indices to add adapters after.
            Empty list or None means no adapters (useful for run-b).
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_dim: int = 64,
        adapter_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_layers = adapter_layers or []

        if self.adapter_layers:
            d_model = base_model.audio_encoder.output_dim  # 1024
            self.audio_adapters = nn.ModuleDict({
                f"layer_{i}": AdapterBottleneck(d_model, adapter_dim)
                for i in self.adapter_layers
            })
        else:
            self.audio_adapters = None

    def _encode_audio_adapted(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward audio through CNN + transformer with adapters intercalated.

        Maintains full parity with MERTEncoderLite.forward(), including
        the pos_embedding interpolation branch for T > max_pos_len.
        """
        encoder = self.base_model.audio_encoder

        # CNN (identical to MERTEncoderLite.forward)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        features = encoder.feature_extractor(audio)  # [B, D, T']
        features = features.transpose(1, 2)           # [B, T', D]

        # Positional embedding (with interpolation branch)
        T = features.size(1)
        if T <= encoder.max_pos_len:
            features = features + encoder.pos_embedding[:, :T, :]
        else:
            pos = F.interpolate(
                encoder.pos_embedding.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
            features = features + pos

        # Transformer layers WITH adapters
        for i, layer in enumerate(encoder.transformer.layers):
            features = layer(features)
            if self.audio_adapters and f"layer_{i}" in self.audio_adapters:
                features = self.audio_adapters[f"layer_{i}"](features)

        # Mean pooling (identical to MERTEncoderLite)
        pooled = features.mean(dim=1)  # [B, D]

        # Project through audio projection
        return self.base_model.audio_projection(pooled)

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compatible with extract_all_embeddings() signature.

        Returns:
            (audio_emb, midi_emb): Both [B, D=256].
        """
        if self.audio_adapters:
            audio_emb = self._encode_audio_adapted(audio)
        else:
            audio_emb = self.base_model.encode_audio(audio)

        midi_emb = self.base_model.encode_midi(
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
    ) -> Tuple[torch.Tensor, dict]:
        """Delegate to base model's VICReg loss."""
        return self.base_model.compute_vicreg_loss(z1, z2)
