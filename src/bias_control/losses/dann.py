"""
Domain Adversarial Neural Network (DANN) loss.

Implements gradient reversal for domain-invariant representation learning.
Used to make embeddings modal-agnostic (can't distinguish Audio vs MIDI).

Reference: Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Tuple
import math


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL).

    During forward pass, acts as identity.
    During backward pass, multiplies gradients by -lambda.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse gradients with scaling factor
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.

    Wraps GradientReversalFunction for use in nn.Sequential.
    """

    def __init__(self, lambda_: float = 1.0):
        """
        Args:
            lambda_: Gradient reversal scaling factor
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update lambda value."""
        self.lambda_ = lambda_


class DomainClassifier(nn.Module):
    """
    Domain classifier for DANN.

    Predicts which domain (Audio=0, MIDI=1) an embedding came from.
    With gradient reversal, trains encoder to produce domain-invariant embeddings.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        n_domains: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            n_domains: Number of domains (2 for Audio/MIDI)
            dropout: Dropout rate
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_domains),
        )

        self.n_domains = n_domains

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict domain.

        Args:
            x: [B, D] embeddings

        Returns:
            [B, n_domains] logits
        """
        return self.classifier(x)


class DANNLoss(nn.Module):
    """
    Domain Adversarial Neural Network loss module.

    Combines:
    1. Gradient Reversal Layer
    2. Domain Classifier
    3. Cross-entropy loss for domain prediction

    Training objective: Minimize domain classification accuracy → modal-agnostic embeddings
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        n_domains: int = 2,
        lambda_init: float = 0.0,
        lambda_schedule: str = "linear_0_to_1",
        max_steps: int = 10000,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension for classifier
            n_domains: Number of domains
            lambda_init: Initial gradient reversal strength
            lambda_schedule: Schedule for lambda ("constant", "linear_0_to_1", "cosine")
            max_steps: Maximum training steps (for scheduling)
            dropout: Dropout rate in classifier
        """
        super().__init__()

        self.grl = GradientReversalLayer(lambda_=lambda_init)
        self.classifier = DomainClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_domains=n_domains,
            dropout=dropout
        )

        self.lambda_init = lambda_init
        self.lambda_schedule = lambda_schedule
        self.max_steps = max_steps
        self.current_step = 0
        self.current_lambda = lambda_init

    def get_lambda(self, step: Optional[int] = None) -> float:
        """
        Compute lambda value for current step.

        Args:
            step: Training step (uses self.current_step if None)

        Returns:
            Lambda value
        """
        if step is None:
            step = self.current_step

        progress = min(1.0, step / self.max_steps)

        if self.lambda_schedule == "constant":
            return self.lambda_init
        elif self.lambda_schedule == "linear_0_to_1":
            return progress
        elif self.lambda_schedule == "cosine":
            return 0.5 * (1 - math.cos(math.pi * progress))
        elif self.lambda_schedule == "step":
            # Step schedule: 0 for first half, 1 for second half
            return 1.0 if progress >= 0.5 else 0.0
        else:
            raise ValueError(f"Unknown lambda schedule: {self.lambda_schedule}")

    def update_lambda(self, step: int):
        """
        Update lambda value for current training step.

        Args:
            step: Current training step
        """
        self.current_step = step
        self.current_lambda = self.get_lambda(step)
        self.grl.set_lambda(self.current_lambda)

    def forward(
        self,
        embeddings: torch.Tensor,
        domain_labels: torch.Tensor,
        update_step: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute DANN loss.

        Args:
            embeddings: [B, D] embeddings from both domains
            domain_labels: [B] domain labels (0=Audio, 1=MIDI)
            update_step: Whether to increment step counter

        Returns:
            loss: Scalar loss value
            metrics: Dict with domain_loss, domain_accuracy, lambda
        """
        if update_step:
            self.current_step += 1
            self.current_lambda = self.get_lambda()
            self.grl.set_lambda(self.current_lambda)

        # Apply gradient reversal
        reversed_embeddings = self.grl(embeddings)

        # Domain classification
        domain_logits = self.classifier(reversed_embeddings)

        # Cross-entropy loss
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        # Compute accuracy (for monitoring)
        with torch.no_grad():
            predictions = domain_logits.argmax(dim=1)
            accuracy = (predictions == domain_labels).float().mean()

        metrics = {
            "domain_loss": domain_loss.item(),
            "domain_accuracy": accuracy.item(),
            "lambda": self.current_lambda,
        }

        return domain_loss, metrics

    def reset(self):
        """Reset step counter."""
        self.current_step = 0
        self.current_lambda = self.lambda_init
        self.grl.set_lambda(self.lambda_init)


class ConditionalDANNLoss(nn.Module):
    """
    Conditional DANN loss.

    Domain classifier is conditioned on piece identity to avoid
    discouraging piece-specific information.

    This variant only penalizes domain-specific features that are NOT
    piece-specific, allowing the model to retain piece identity.
    """

    def __init__(
        self,
        input_dim: int = 256,
        piece_embed_dim: int = 64,
        hidden_dim: int = 64,
        n_domains: int = 2,
        n_pieces: int = 1000,
        lambda_init: float = 0.0,
        lambda_schedule: str = "linear_0_to_1",
        max_steps: int = 10000,
    ):
        """
        Args:
            input_dim: Input embedding dimension
            piece_embed_dim: Dimension for piece embedding
            hidden_dim: Hidden layer dimension
            n_domains: Number of domains
            n_pieces: Maximum number of unique pieces
            lambda_init: Initial lambda value
            lambda_schedule: Lambda schedule type
            max_steps: Maximum training steps
        """
        super().__init__()

        self.grl = GradientReversalLayer(lambda_=lambda_init)

        # Piece embedding
        self.piece_embedding = nn.Embedding(n_pieces, piece_embed_dim)

        # Conditional classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim + piece_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_domains),
        )

        self.lambda_schedule = lambda_schedule
        self.max_steps = max_steps
        self.current_step = 0

    def forward(
        self,
        embeddings: torch.Tensor,
        domain_labels: torch.Tensor,
        piece_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute conditional DANN loss.

        Args:
            embeddings: [B, D] embeddings
            domain_labels: [B] domain labels
            piece_ids: [B] piece identifiers

        Returns:
            loss: Scalar loss
            metrics: Dict with metrics
        """
        self.current_step += 1
        lambda_ = self._get_lambda()
        self.grl.set_lambda(lambda_)

        # Apply GRL
        reversed_embeddings = self.grl(embeddings)

        # Get piece embeddings
        piece_emb = self.piece_embedding(piece_ids)

        # Concatenate
        conditioned = torch.cat([reversed_embeddings, piece_emb], dim=-1)

        # Classify
        domain_logits = self.classifier(conditioned)
        loss = F.cross_entropy(domain_logits, domain_labels)

        with torch.no_grad():
            acc = (domain_logits.argmax(1) == domain_labels).float().mean()

        return loss, {
            "domain_loss": loss.item(),
            "domain_accuracy": acc.item(),
            "lambda": lambda_
        }

    def _get_lambda(self) -> float:
        progress = min(1.0, self.current_step / self.max_steps)
        if self.lambda_schedule == "linear_0_to_1":
            return progress
        return 1.0
