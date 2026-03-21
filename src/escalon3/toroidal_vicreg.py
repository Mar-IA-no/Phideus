"""Escalón 3: Toroidal VICReg loss for P5/P6.

TVICRegLoss: Circular invariance + circular variance + circular covariance on T^N.
MixedVICRegLoss: Standard VICReg on euclidean branch + TVICReg on torus branch.

Training loss uses chordal/circular invariance (1 - cos(Δθ)).
Retrieval/evaluation uses geodesic distance (wrap-aware).
"""

import math
from typing import Dict

import torch
import torch.nn as nn

EPS = 1e-8


class TVICRegLoss(nn.Module):
    """T-VICReg: VICReg on T^N with circular statistics.

    Components:
    - Invariance: chordal (1 - cos(Δθ)), NOT geodesic. Smooth, differentiable.
    - Variance: mean resultant length R̄ per dimension. R̄→0 = uniform, R̄→1 = collapse.
    - Covariance: Jammalamadaka-Sarma circular correlation (off-diagonal).

    Input: angles in (-π, π] from TorusProjector. Wrap-aware by construction
    (cos/sin operations handle any range).
    """

    def __init__(self, lambda_inv: float = 10.0, lambda_var: float = 10.0,
                 lambda_cov: float = 1.0):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def chordal_invariance(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        """Chordal invariance: mean(1 - cos(Δθ)). Bounded [0, 2].
        Equivalent to squared chord length on the unit circle, averaged over dimensions.
        """
        return (1 - torch.cos(theta_a - theta_b)).mean()

    def circular_variance(self, theta: torch.Tensor) -> torch.Tensor:
        """Mean resultant length across batch, averaged over dimensions.
        R̄_d = |mean(exp(i·θ_d))|. Minimize R̄ = maximize angular spread.
        """
        cos_mean = torch.cos(theta).mean(dim=0)  # [N]
        sin_mean = torch.sin(theta).mean(dim=0)  # [N]
        R_bar = (cos_mean ** 2 + sin_mean ** 2 + EPS).sqrt()  # [N]
        return R_bar.mean()

    def circular_covariance(self, theta: torch.Tensor) -> torch.Tensor:
        """Off-diagonal Jammalamadaka-Sarma circular correlation.
        Penalizes correlation between angular dimensions.
        """
        B, N = theta.shape
        # Circular mean per dimension
        mu = torch.atan2(
            torch.sin(theta).mean(dim=0),
            torch.cos(theta).mean(dim=0),
        )  # [N]

        # Sin deviations from circular mean
        sin_dev = torch.sin(theta - mu.unsqueeze(0))  # [B, N]

        # Circular covariance matrix
        cov = (sin_dev.T @ sin_dev) / max(B - 1, 1)  # [N, N]

        # Off-diagonal penalty
        off_diag_sq = cov ** 2
        off_diag_sum = off_diag_sq.sum() - off_diag_sq.diagonal().sum()
        return off_diag_sum / N

    def forward(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute T-VICReg loss.

        Args:
            theta_a, theta_b: [B, N] angles in (-π, π] from TorusProjector.
        Returns:
            Dict with 'total', 'chordal_invariance', 'circular_variance', 'circular_covariance'.
        """
        inv = self.chordal_invariance(theta_a, theta_b)
        var_a = self.circular_variance(theta_a)
        var_b = self.circular_variance(theta_b)
        var = (var_a + var_b) / 2
        cov_a = self.circular_covariance(theta_a)
        cov_b = self.circular_covariance(theta_b)
        cov = (cov_a + cov_b) / 2

        total = self.lambda_inv * inv + self.lambda_var * var + self.lambda_cov * cov

        return {
            'total': total,
            'chordal_invariance': inv,
            'circular_variance': var,
            'circular_covariance': cov,
            'var_a': var_a,
            'var_b': var_b,
        }


class MixedVICRegLoss(nn.Module):
    """Combined loss: VICReg on euclidean branch + TVICReg on torus branch.

    L_total = L_vicreg_euclidean + L_tvicreg_torus
    No coupling between branches in first implementation.
    """

    def __init__(self,
                 lambda_inv: float = 25.0, lambda_var: float = 25.0, lambda_cov: float = 1.0,
                 lambda_t_inv: float = 10.0, lambda_t_var: float = 10.0, lambda_t_cov: float = 1.0):
        super().__init__()
        # Import here to avoid circular dependency
        from src.RNA.vicreg import VICRegLoss
        self.euc_loss = VICRegLoss(lambda_inv, lambda_var, lambda_cov)
        self.torus_loss = TVICRegLoss(lambda_t_inv, lambda_t_var, lambda_t_cov)

    def forward(self, z_euc_a: torch.Tensor, z_euc_b: torch.Tensor,
                theta_a: torch.Tensor, theta_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute mixed loss.

        Args:
            z_euc_a, z_euc_b: [B, D_euc] euclidean embeddings.
            theta_a, theta_b: [B, N_angles] toroidal angles in (-π, π].
        Returns:
            Dict with 'total', 'euc_total', 'torus_total', and all component losses.
        """
        euc = self.euc_loss(z_euc_a, z_euc_b)
        torus = self.torus_loss(theta_a, theta_b)

        total = euc['total'] + torus['total']

        return {
            'total': total,
            'euc_total': euc['total'],
            'torus_total': torus['total'],
            'euc_invariance': euc['invariance'],
            'euc_variance': euc['variance'],
            'euc_covariance': euc['covariance'],
            'torus_chordal_invariance': torus['chordal_invariance'],
            'torus_circular_variance': torus['circular_variance'],
            'torus_circular_covariance': torus['circular_covariance'],
        }
