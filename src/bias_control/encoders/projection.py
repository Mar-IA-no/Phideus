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


class ConditionedProjectionHead(nn.Module):
    """
    FiLM-conditioned projection head (Gate 5A).

    Same MLP structure as ProjectionHead but with FiLM modulation between
    hidden blocks. Zero-init on FiLM generators ensures identity at init.

    Architecture:
        Layer 0: Linear(D_in, 512) → BN → ReLU → FiLM(cond)
        Layer 1: Linear(512, 512)  → BN → ReLU → FiLM(cond)
        Layer 2: Linear(512, 256)  (output, no FiLM)

    FiLM: h' = (1 + gamma) * h + beta, with gamma/beta from zero-init MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        cond_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cond_dim = cond_dim

        # Hidden blocks: each is (Linear, BN, ReLU) — stored flat for state_dict compat
        self.hidden_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]),
            nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]),
        ])

        # Final linear (no FiLM)
        self.final_linear = nn.Linear(hidden_dim, output_dim)

        # FiLM generators — one per hidden layer
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cond_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * hidden_dim),  # gamma + beta
            )
            for _ in range(2)
        ])

        self._zero_init_film()

    def _zero_init_film(self):
        """Zero-init last layer of each FiLM generator -> identity at init."""
        for gen in self.film_generators:
            nn.init.zeros_(gen[-1].weight)
            nn.init.zeros_(gen[-1].bias)

    @classmethod
    def from_projection_head(cls, orig: 'ProjectionHead', cond_dim: int) -> 'ConditionedProjectionHead':
        """Create conditioned projection by copying ALL weights+buffers from existing ProjectionHead."""
        # Structural asserts: verify exact topology before copying
        assert len(orig.mlp) == 7, (
            f"Expected ProjectionHead with 7 modules (3-layer, BN, no dropout), got {len(orig.mlp)}. "
            f"Modules: {[type(m).__name__ for m in orig.mlp]}"
        )
        assert isinstance(orig.mlp[0], nn.Linear), f"mlp[0] should be Linear, got {type(orig.mlp[0])}"
        assert isinstance(orig.mlp[1], nn.BatchNorm1d), f"mlp[1] should be BN, got {type(orig.mlp[1])}"
        assert isinstance(orig.mlp[3], nn.Linear), f"mlp[3] should be Linear, got {type(orig.mlp[3])}"
        assert isinstance(orig.mlp[4], nn.BatchNorm1d), f"mlp[4] should be BN, got {type(orig.mlp[4])}"
        assert isinstance(orig.mlp[6], nn.Linear), f"mlp[6] should be Linear, got {type(orig.mlp[6])}"

        hidden_dim = orig.mlp[0].out_features

        new = cls(
            input_dim=orig.input_dim,
            output_dim=orig.output_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
        )

        # Copy weights+buffers: orig.mlp[idx] -> new structure
        # Layer 0: mlp[0]=Linear, mlp[1]=BN
        new.hidden_layers[0][0].load_state_dict(orig.mlp[0].state_dict())
        new.hidden_layers[0][1].load_state_dict(orig.mlp[1].state_dict())

        # Layer 1: mlp[3]=Linear, mlp[4]=BN
        new.hidden_layers[1][0].load_state_dict(orig.mlp[3].state_dict())
        new.hidden_layers[1][1].load_state_dict(orig.mlp[4].state_dict())

        # Final: mlp[6]=Linear
        new.final_linear.load_state_dict(orig.mlp[6].state_dict())

        # FiLM generators stay zero-init (untouched)
        return new

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project with optional FiLM conditioning.

        Args:
            x: [B, D_in] encoder output
            cond: [B, cond_dim] conditioning vector, or None to use cached

        Returns:
            [B, D_out] projected embedding
        """
        if cond is None:
            cond = getattr(self, '_cached_cond', None)

        for i, (linear, bn, relu) in enumerate(self.hidden_layers):
            x = relu(bn(linear(x)))
            if cond is not None and i < len(self.film_generators):
                film_params = self.film_generators[i](cond)
                gamma, beta = film_params.chunk(2, dim=-1)
                x = (1 + gamma) * x + beta

        return self.final_linear(x)

    def get_output_dim(self) -> int:
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
