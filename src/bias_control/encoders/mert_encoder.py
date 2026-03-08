"""
MERT Encoder: Pre-trained audio encoder from HuggingFace.

MERT (Music Understanding Model with Large-Scale Self-supervised Training)
is a BERT-like model pre-trained on music audio.

Model: m-a-p/MERT-v1-330M
- Input: Raw audio waveform at 24kHz
- Output: [B, T, 1024] hidden states
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MERTEncoder(nn.Module):
    """
    MERT audio encoder wrapper.

    Loads pre-trained MERT model from HuggingFace and provides
    a clean interface for extracting audio embeddings.
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-330M",
        freeze: bool = True,
        aggregation: str = "mean",  # "mean", "cls", "last"
        layer: int = -1,  # Which layer to use (-1 = last)
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: If True, freeze all MERT parameters
            aggregation: How to aggregate temporal dimension
            layer: Which transformer layer to extract from
            device: Device to load model on
        """
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.aggregation = aggregation
        self.layer = layer
        self.device = device

        # Will be loaded lazily
        self._model = None
        self._processor = None
        self._loaded = False

        # MERT output dimension
        self.output_dim = 1024
        self.sample_rate = 24000

    def _load_model(self):
        """Lazy load MERT model from HuggingFace."""
        if self._loaded:
            return

        try:
            from transformers import Wav2Vec2FeatureExtractor, AutoModel

            logger.info(f"Loading MERT model: {self.model_name}")

            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            if self.freeze:
                for param in self._model.parameters():
                    param.requires_grad = False
                self._model.eval()
                logger.info("MERT model frozen")

            if self.device:
                self._model = self._model.to(self.device)

            self._loaded = True
            logger.info(f"MERT model loaded successfully. Output dim: {self.output_dim}")

        except ImportError:
            raise ImportError(
                "transformers is required for MERT. Install with: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load MERT model: {e}")
            raise

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio waveform for MERT.

        Args:
            waveform: [B, T] or [B, 1, T] audio at self.sample_rate

        Returns:
            Preprocessed input for MERT
        """
        self._load_model()

        # Ensure correct shape
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # Process each sample (MERT processor expects numpy)
        device = waveform.device
        processed = []

        for wav in waveform:
            wav_np = wav.cpu().numpy()
            inputs = self._processor(
                wav_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            processed.append(inputs.input_values)

        # Stack batch
        batch = torch.cat(processed, dim=0).to(device)
        return batch

    def forward(
        self,
        waveform: torch.Tensor,
        preprocess: bool = True,
        return_all_layers: bool = False,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Extract embeddings from audio.

        Args:
            waveform: [B, T] audio at 24kHz, or [B, T'] preprocessed
            preprocess: Whether to run preprocessing
            return_all_layers: Return all hidden states
            return_sequence: If True, return [B, T, 1024] pre-aggregation

        Returns:
            [B, D] aggregated embeddings where D=1024,
            or [B, T, 1024] if return_sequence=True,
            or tuple of all layers if return_all_layers=True
        """
        self._load_model()

        if preprocess:
            inputs = self.preprocess(waveform)
        else:
            inputs = waveform

        # Move to model device
        if self.device:
            inputs = inputs.to(self.device)
        elif next(self._model.parameters()).device != inputs.device:
            inputs = inputs.to(next(self._model.parameters()).device)

        # Forward pass
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self._model(
                inputs,
                output_hidden_states=True
            )

        # Extract hidden states from specified layer
        hidden_states = outputs.hidden_states[self.layer]  # [B, T, 1024]

        if return_all_layers:
            return outputs.hidden_states

        if return_sequence:
            return hidden_states  # [B, T, 1024]

        # Aggregate temporal dimension
        if self.aggregation == "mean":
            embeddings = hidden_states.mean(dim=1)  # [B, 1024]
        elif self.aggregation == "cls":
            embeddings = hidden_states[:, 0, :]  # [B, 1024]
        elif self.aggregation == "last":
            embeddings = hidden_states[:, -1, :]  # [B, 1024]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return embeddings

    def train(self, mode=True):
        """Override to keep frozen model in eval mode always.

        When freeze=True, the HuggingFace model must stay in eval mode
        to prevent dropout/batchnorm from reactivating when the parent
        module calls model.train().
        """
        super().train(mode)
        if self.freeze and self._loaded and self._model is not None:
            self._model.eval()
        return self

    def get_output_dim(self) -> int:
        """Return output embedding dimension."""
        return self.output_dim

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.device = str(device)
        if self._loaded and self._model is not None:
            self._model = self._model.to(device)
        return self


class MERTEncoderLite(nn.Module):
    """
    Lightweight MERT-like encoder for testing without HuggingFace.

    Uses a simple CNN + Transformer architecture to mimic MERT's
    behavior for quick prototyping.
    """

    def __init__(
        self,
        output_dim: int = 1024,
        n_layers: int = 4,
        n_heads: int = 8,
        sample_rate: int = 24000,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.sample_rate = sample_rate

        # CNN feature extractor (similar to wav2vec2)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv1d(512, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, output_dim),
            nn.GELU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learnable position embeddings (max 6000 tokens for 8s audio at 24kHz)
        # After CNN downsampling (~40x), 8s*24kHz/40 ≈ 4800 tokens
        self.max_pos_len = 6000
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_pos_len, output_dim) * 0.02)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] audio

        Returns:
            [B, D] embeddings
        """
        # Ensure [B, 1, T]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # CNN features: [B, D, T']
        features = self.feature_extractor(waveform)

        # Transpose to [B, T', D]
        features = features.transpose(1, 2)

        # Add position embeddings (truncate or extend if needed)
        T = features.size(1)
        if T <= self.max_pos_len:
            features = features + self.pos_embedding[:, :T, :]
        else:
            # For very long sequences, interpolate position embeddings
            pos = torch.nn.functional.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            features = features + pos

        # Transformer: [B, T', D]
        encoded = self.transformer(features)

        # Mean pooling: [B, D]
        embeddings = encoded.mean(dim=1)

        return embeddings

    def get_output_dim(self) -> int:
        return self.output_dim
