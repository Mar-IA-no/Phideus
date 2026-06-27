"""WavLM + injection module + classifier head para Fase 1.

Arquitectura compartida entre las 4 configs:

    WavLM frozen → [B, T, 1024]
      → Injection module {concat | FiLM | xattn | none}
      → [B, T, 1024]
      → Mean pool over T → [B, 1024]
      → Linear(1024, 5) → logits

Los 3 mecanismos operan frame-level post-WavLM, pre-pool, para que la comparación
entre mecanismos sea metodológicamente homogénea. Algoritmos heredados de E2
(reimplementación compatible, no import directo):

    - Concat: near-identity init [I | 0] de SpeechEGGEncoderAug
    - FiLM: zero-init de ConditionedProjectionHead, reformulado frame-level
    - xattn: MultiheadAttention(embed=1024, heads=4) + scale=0.01 de SpeechEGGEncoderXAttn
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatInjection(nn.Module):
    """Frame-level concat + linear projection con near-identity init."""

    def __init__(self, feature_dim: int = 1024, descriptor_dim: int = 12):
        super().__init__()
        self.proj = nn.Linear(feature_dim + descriptor_dim, feature_dim)

        # Near-identity init: W = [I | 0], bias = 0
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:feature_dim, :feature_dim] = torch.eye(feature_dim)
            self.proj.bias.zero_()

    def forward(self, features: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        # features: [B, T, 1024], descriptor: [B, T, 12]
        merged = torch.cat([features, descriptor], dim=-1)  # [B, T, 1036]
        return self.proj(merged)


class FiLMInjection(nn.Module):
    """Frame-level FiLM: per-frame gamma/beta from descriptor con zero-init."""

    def __init__(self, feature_dim: int = 1024, descriptor_dim: int = 12,
                 hidden_dim: int = 64):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim),  # gamma + beta
        )
        # Zero-init last layer → gamma=0, beta=0 → identity at init
        with torch.no_grad():
            self.gen[-1].weight.zero_()
            self.gen[-1].bias.zero_()

    def forward(self, features: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        film = self.gen(descriptor)                  # [B, T, 2*1024]
        gamma, beta = film.chunk(2, dim=-1)          # cada uno [B, T, 1024]
        return (1.0 + gamma) * features + beta


class XAttnInjection(nn.Module):
    """Cross-attention residual: Q=descriptor projected, K/V=features.

    Near-zero residual: scale=0.01 inicial, parametro entrenable.
    """

    def __init__(self, feature_dim: int = 1024, descriptor_dim: int = 12,
                 n_heads: int = 4):
        super().__init__()
        self.desc_proj = nn.Linear(descriptor_dim, feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, features: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        # features: [B, T, 1024], descriptor: [B, T, 12]
        desc_q = self.desc_proj(descriptor)          # [B, T, 1024]
        attn_out, _ = self.cross_attn(
            query=desc_q, key=features, value=features,
            need_weights=False,
        )
        return features + self.scale * self.norm(attn_out)


class WavLMInjectionClassifier(nn.Module):
    """Plantilla compartida: features → injection → pool → classifier.

    WavLM NO se incluye en el módulo — se asume que el caller pasa features
    ya extraídas (precomputadas para eficiencia). El injection module y el
    classifier head son las únicas piezas entrenables.

    Args:
        mechanism: 'none' (baseline), 'concat', 'film', 'xattn'.
        feature_dim: dimensión de WavLM (1024 para wavlm-large).
        descriptor_dim: dimensión del descriptor (12 para Familia A).
        n_classes: 5 emociones de ESD.
    """

    def __init__(
        self,
        mechanism: str = "none",
        feature_dim: int = 1024,
        descriptor_dim: int = 12,
        n_classes: int = 5,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.feature_dim = feature_dim

        if mechanism == "none":
            self.injection = None
        elif mechanism == "concat":
            self.injection = ConcatInjection(feature_dim, descriptor_dim)
        elif mechanism == "film":
            self.injection = FiLMInjection(feature_dim, descriptor_dim)
        elif mechanism == "xattn":
            self.injection = XAttnInjection(feature_dim, descriptor_dim)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        self.classifier = nn.Linear(feature_dim, n_classes)

    def get_embedding(
        self,
        features: torch.Tensor,
        descriptor: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return utterance-level embedding [B, 1024] post-injection, post-pool.

        Used both inside forward() (for classification) and for CKA computation.

        Args:
            features: [B, T, 1024] WavLM frame-level features.
            descriptor: [B, T, 12] aligned descriptor; None for mechanism='none'.
            mask: [B, T] bool, True for valid positions, False for padding.
                  If None, all positions used.
        """
        if self.injection is not None:
            assert descriptor is not None, f"Mechanism {self.mechanism} requires descriptor"
            features = self.injection(features, descriptor)

        # Mean pool over T, respecting mask if provided
        if mask is None:
            embedding = features.mean(dim=1)
        else:
            mask_f = mask.float().unsqueeze(-1)              # [B, T, 1]
            sum_features = (features * mask_f).sum(dim=1)    # [B, 1024]
            denom = mask_f.sum(dim=1).clamp(min=1.0)         # [B, 1]
            embedding = sum_features / denom
        return embedding

    def forward(
        self,
        features: torch.Tensor,
        descriptor: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward: features + descriptor → logits.

        Returns:
            [B, n_classes] logits.
        """
        embedding = self.get_embedding(features, descriptor, mask)
        return self.classifier(embedding)
