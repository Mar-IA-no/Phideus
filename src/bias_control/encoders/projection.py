"""
Projection Head: MLP to map encoder outputs to shared embedding space.

Projects from encoder-specific dimensions to a shared embedding dimension
for cross-modal contrastive learning.
"""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Architecture: Linear → BN → ReLU → Linear → BN → ReLU → Linear

    This follows the SimCLR/VICReg pattern of using a deeper projection
    head for better representation learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
        use_layernorm: bool = False,
        final_activation: bool = False,
    ):
        """
        Args:
            input_dim: Input dimension from encoder
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            n_layers: Number of layers (2 or 3)
            dropout: Dropout rate
            use_batchnorm: Use batch normalization
            use_layernorm: Use layer normalization (mutually exclusive with batchnorm)
            final_activation: Whether to apply activation after final layer
        """
        super().__init__()

        assert not (use_batchnorm and use_layernorm), "Cannot use both batchnorm and layernorm"
        assert n_layers >= 2, "Need at least 2 layers"

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Middle layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        if final_activation:
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(output_dim))
            elif use_layernorm:
                layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project encoder output to shared embedding space.

        Args:
            x: [B, D_in] encoder output

        Returns:
            [B, D_out] projected embedding
        """
        return self.mlp(x)

    def get_output_dim(self) -> int:
        """Return output embedding dimension."""
        return self.output_dim


class DualProjectionHead(nn.Module):
    """
    Dual projection head with separate paths for different objectives.

    Used when you need separate embedding spaces for e.g. retrieval vs domain classification.
    """

    def __init__(
        self,
        input_dim: int,
        retrieval_dim: int = 256,
        auxiliary_dim: int = 64,
        hidden_dim: int = 512,
    ):
        """
        Args:
            input_dim: Input dimension from encoder
            retrieval_dim: Output dimension for retrieval embeddings
            auxiliary_dim: Output dimension for auxiliary task (e.g., domain classification)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Shared initial projection
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Retrieval head
        self.retrieval_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, retrieval_dim),
        )

        # Auxiliary head (e.g., for domain classification)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(hidden_dim, auxiliary_dim),
        )

        self.retrieval_dim = retrieval_dim
        self.auxiliary_dim = auxiliary_dim

    def forward(
        self,
        x: torch.Tensor,
        return_auxiliary: bool = False
    ) -> torch.Tensor:
        """
        Project encoder output.

        Args:
            x: [B, D_in] encoder output
            return_auxiliary: If True, also return auxiliary embedding

        Returns:
            [B, D_retrieval] or ([B, D_retrieval], [B, D_aux]) if return_auxiliary
        """
        shared = self.shared(x)
        retrieval = self.retrieval_head(shared)

        if return_auxiliary:
            auxiliary = self.auxiliary_head(shared)
            return retrieval, auxiliary

        return retrieval


class ExpanderProjection(nn.Module):
    """
    Expander projection from VICReg paper.

    Projects to a higher-dimensional space before VICReg loss,
    then projects back to target dimension for evaluation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        expander_dim: int = 8192,
        output_dim: int = 256,
    ):
        """
        Args:
            input_dim: Input dimension from encoder
            hidden_dim: Hidden dimension
            expander_dim: Expanded dimension for VICReg (large)
            output_dim: Final output dimension for retrieval
        """
        super().__init__()

        # Expander (for training with VICReg)
        self.expander = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, expander_dim),
        )

        # Projector (maps from expander to retrieval space)
        self.projector = nn.Sequential(
            nn.Linear(expander_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        self.expander_dim = expander_dim
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        return_expanded: bool = False
    ) -> torch.Tensor:
        """
        Project encoder output.

        Args:
            x: [B, D_in] encoder output
            return_expanded: If True, return expanded representation (for VICReg loss)

        Returns:
            [B, D_out] or [B, D_expanded] if return_expanded
        """
        expanded = self.expander(x)

        if return_expanded:
            return expanded

        return self.projector(expanded)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_expander_dim(self) -> int:
        return self.expander_dim
