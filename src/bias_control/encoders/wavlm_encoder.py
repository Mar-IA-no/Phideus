"""
WavLM Encoder: Frozen pre-trained speech encoder from HuggingFace.

WavLM-Large (microsoft/wavlm-large) is a self-supervised speech model
pre-trained on 94K hours of speech data with gated relative position bias
and utterance mixing strategy.

Model: microsoft/wavlm-large
- Input: Raw audio waveform at 16kHz
- Output: [B, T, 1024] hidden states (T ≈ samples/320)
- Parameters: ~316M
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WavLMEncoder(nn.Module):
    """
    Frozen WavLM-Large encoder wrapper for S2-P3.

    Follows the same pattern as MERTEncoder: lazy loading, eval-mode
    enforcement, and anti-ghost protocol.

    Key difference from MERT: WavLM takes raw waveform directly at 16kHz
    (no separate feature extractor / processor needed).
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze: bool = True,
        aggregation: str = "mean",
        layer: int = -1,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.aggregation = aggregation
        self.layer = layer
        self.device = device

        self._model = None
        self._loaded = False

        self.output_dim = 1024
        self.sample_rate = 16000

    def _load_model(self):
        """Lazy load WavLM model from HuggingFace."""
        if self._loaded:
            return

        try:
            from transformers import WavLMModel

            logger.info(f"Loading WavLM model: {self.model_name}")

            self._model = WavLMModel.from_pretrained(self.model_name)

            actual_hidden = self._model.config.hidden_size
            if actual_hidden != self.output_dim:
                logger.warning(
                    f"WavLM hidden_size={actual_hidden}, expected {self.output_dim}. "
                    f"Updating output_dim."
                )
                self.output_dim = actual_hidden

            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"WavLM loaded: {n_params/1e6:.1f}M params, hidden={self.output_dim}")

            if self.freeze:
                for param in self._model.parameters():
                    param.requires_grad = False
                self._model.eval()
                logger.info("WavLM model frozen")

            if self.device:
                self._model = self._model.to(self.device)

            self._loaded = True

        except ImportError:
            raise ImportError(
                "transformers is required for WavLM. Install with: pip install transformers"
            )
        except OSError as e:
            if "not a local folder" in str(e) or "Connection" in str(e):
                raise RuntimeError(
                    f"WavLM model not in cache and download failed. "
                    f"Download first with internet access, or check HuggingFace cache."
                ) from e
            raise

    def forward(
        self,
        waveform: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Extract embeddings from speech audio.

        Args:
            waveform: [B, T] raw audio at 16kHz
            return_sequence: If True, return [B, T_frames, 1024] pre-aggregation

        Returns:
            [B, 1024] mean-pooled embeddings,
            or [B, T_frames, 1024] if return_sequence=True
        """
        self._load_model()

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        model_device = next(self._model.parameters()).device
        if waveform.device != model_device:
            waveform = waveform.to(model_device)

        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self._model(
                waveform,
                output_hidden_states=(self.layer != -1),
            )

        if self.layer != -1:
            hidden_states = outputs.hidden_states[self.layer]
        else:
            hidden_states = outputs.last_hidden_state

        if return_sequence:
            return hidden_states

        if self.aggregation == "mean":
            return hidden_states.mean(dim=1)
        elif self.aggregation == "cls":
            return hidden_states[:, 0, :]
        elif self.aggregation == "last":
            return hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def train(self, mode=True):
        """Override to keep frozen model in eval mode always."""
        super().train(mode)
        if self.freeze and self._loaded and self._model is not None:
            self._model.eval()
        return self

    def get_output_dim(self) -> int:
        return self.output_dim

    def to(self, device):
        super().to(device)
        self.device = str(device)
        if self._loaded and self._model is not None:
            self._model = self._model.to(device)
        return self

    def verify_determinism(self, waveform: torch.Tensor) -> bool:
        """Run same input twice and verify outputs match. For precomputation QA."""
        self._load_model()
        out1 = self.forward(waveform)
        out2 = self.forward(waveform)
        match = torch.allclose(out1, out2, atol=1e-6)
        if not match:
            max_diff = (out1 - out2).abs().max().item()
            logger.warning(f"WavLM determinism check FAILED: max_diff={max_diff:.2e}")
        return match
