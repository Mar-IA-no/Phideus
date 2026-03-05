#!/usr/bin/env python3
"""
Gate 7 — HuggingFace MERT feature extractor wrapper.

Handles MERT-v1-95M and MERT-v1-330M from HuggingFace.
Input:  waveform [B, T] at 24kHz (raw float32, already at correct SR)
Output: dict with pooled features, optional pre-pool sequence, optional per-layer

Hidden size is read from model config — NOT hardcoded.
Usage:
    extractor = MERTLargeExtractor('m-a-p/MERT-v1-330M', device='cuda')
    result = extractor.extract(waveform_batch)  # dict
    pooled = result['pooled']   # [B, 1024]
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Canonical short names → HuggingFace model IDs
HF_MODEL_IDS = {
    'MERT-v1-95M':  'm-a-p/MERT-v1-95M',
    'MERT-v1-330M': 'm-a-p/MERT-v1-330M',
}


class MERTLargeExtractor(nn.Module):
    """
    Wrapper over HuggingFace MERT for feature extraction in Gate 7.

    Input:  waveform [B, T] float32 @ 24kHz
    Output: dict with:
        'pooled':     [B, hidden_size]              — mean pool over T_mert
        'last_layer': [B, T_mert, hidden_size]      — pre-pool sequence
        'hidden_size': int
        'T_mert': int                               — number of MERT frames
        'all_layers': tuple of [B, T_mert, H]       — only if return_all_layers=True
    """

    def __init__(self, model_name: str = 'm-a-p/MERT-v1-330M', device: str = 'cuda'):
        super().__init__()
        self.model_name = model_name
        self.device_str = device

        from transformers import Wav2Vec2FeatureExtractor, AutoModel

        logger.info(f"Loading MERT: {model_name}")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
        ).to(device)
        self.model.eval()

        # Read from model config — not hardcoded
        self.hidden_size = self.model.config.hidden_size
        self.sample_rate = self.processor.sampling_rate  # 24000 for MERT-v1

        logger.info(
            f"  hidden_size={self.hidden_size}, "
            f"sr={self.sample_rate}, "
            f"n_layers={self.model.config.num_hidden_layers}"
        )

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor, return_all_layers: bool = False) -> dict:
        """
        Extract features from raw waveform batch.

        Args:
            waveform: [B, T] float32 at 24kHz (already normalized, no extra preprocessing)
            return_all_layers: also return per-layer hidden states (for Exp 7.0b)

        Returns:
            dict with 'pooled', 'last_layer', 'hidden_size', 'T_mert',
            optionally 'all_layers'
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        waveform = waveform.to(self.device_str)

        outputs = self.model(
            waveform,
            output_hidden_states=return_all_layers,
        )

        last = outputs.last_hidden_state  # [B, T_mert, hidden_size]
        T_mert = last.size(1)

        result = {
            'last_layer': last,
            'pooled': last.mean(dim=1),  # [B, hidden_size]
            'hidden_size': self.hidden_size,
            'T_mert': T_mert,
        }
        if return_all_layers:
            result['all_layers'] = outputs.hidden_states  # tuple of [B, T_mert, H]

        return result

    @staticmethod
    def align_a4_to_mert(a4: torch.Tensor, T_mert: int) -> torch.Tensor:
        """
        Downsample A4 descriptor sequence to MERT temporal resolution.
        Uses adaptive_avg_pool1d (correct for temporal aggregation — no interpolation).

        Args:
            a4: [B, T_a4, 8] — A4 descriptors at high temporal resolution
            T_mert: target temporal length (MERT frame count, typically ~300 for 4s)

        Returns:
            [B, T_mert, 8]
        """
        return F.adaptive_avg_pool1d(
            a4.permute(0, 2, 1),  # [B, 8, T_a4]
            output_size=T_mert,
        ).permute(0, 2, 1)  # [B, T_mert, 8]
