"""Escalón 3: Toroidal and Mixed projectors for P5/P6.

TorusProjector: Projects to T^N via pair-normalize + atan2. Output in (-π, π].
MixedProjector: Splits into euclidean (128d) + toroidal (64 angles) branches.

No modular wrap in forward(). Wrap-around handled by loss and evaluation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorusProjector(nn.Module):
    """Projects encoder output to T^N via pair-normalize + atan2.

    input_dim → MLP → 2*n_angles raw → pair normalize → atan2 → [B, n_angles] in (-π, π]
    No modular wrap in forward. Wrap handled by loss/eval.

    For P6: n_angles=128 (256 raw outputs → 128 angle pairs).
    """

    def __init__(self, input_dim: int = 256, n_angles: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.n_angles = n_angles
        self.output_dim = n_angles  # for compatibility with evaluation code

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * n_angles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, input_dim] → [B, n_angles] angles in (-π, π]."""
        h = self.mlp(x)                           # [B, 2*N]
        h = h.view(-1, self.n_angles, 2)          # [B, N, 2]
        h = F.normalize(h, dim=-1)                # pair normalize to unit circle
        theta = torch.atan2(h[..., 1], h[..., 0]) # [B, N] in (-π, π]
        return theta


class MixedProjector(nn.Module):
    """Splits projection into euclidean branch + toroidal branch.

    For P5: euc_dim=128, n_angles=64. Total comparable to proj_dim=256.
    The shared trunk processes encoder output, then splits into two heads.
    """

    def __init__(self, input_dim: int = 256, euc_dim: int = 128,
                 n_angles: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.euc_dim = euc_dim
        self.n_angles = n_angles

        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.euc_head = nn.Linear(hidden_dim, euc_dim)
        self.torus_head = nn.Linear(hidden_dim, 2 * n_angles)

    def forward(self, x: torch.Tensor) -> dict:
        """[B, input_dim] → {'euclidean': [B, euc_dim], 'torus': [B, n_angles]}."""
        h = self.shared_trunk(x)                    # [B, hidden_dim]

        z_euc = self.euc_head(h)                    # [B, euc_dim]

        h_torus = self.torus_head(h)                # [B, 2*n_angles]
        h_torus = h_torus.view(-1, self.n_angles, 2)
        h_torus = F.normalize(h_torus, dim=-1)      # pair normalize
        theta = torch.atan2(h_torus[..., 1], h_torus[..., 0])  # [B, n_angles]

        return {'euclidean': z_euc, 'torus': theta}
