#!/usr/bin/env python3
"""
Gate 4.2 Training: Ratio Descriptor Screening (Post Bloque A Foundation).

Tests whether ratio-based descriptors add causal signal to cross-modal learning.
Each descriptor variant trains on the SAME foundation with the SAME freeze policy,
differing ONLY in how (or whether) ratio information is injected.

Descriptors:
  d0 — Control: VICReg only (no ratio signal). Replicates Run B freeze policy.
  d1 — Pitch ratio histogram (128 bins, 1 channel).
  d2 — Enriched ratio histogram (128 bins, 3 channels: velocity/duration/unweighted).
  d3 — Temporal-rhythmic: IOI ratios + duration ratios + pitch intervals (153 dims).
  d4 — Input augmentation: local interval features concatenated to MIDI embeddings.

Usage:
    # D0 control
    python experiments/bias_control/gate42_training.py \\
        --descriptor d0 \\
        --checkpoint data/bias_control_medium/training_outputs/foundation.pt \\
        --output data/bias_control_medium/training_outputs/gate42_d0 \\
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \\
        --epochs 5 --batch-size 16 --num-workers 8

    # D1 ratio auxiliary
    python experiments/bias_control/gate42_training.py \\
        --descriptor d1 --ratio-weight 0.1 \\
        --checkpoint data/bias_control_medium/training_outputs/foundation.pt \\
        --output data/bias_control_medium/training_outputs/gate42_d1 \\
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \\
        --epochs 5 --batch-size 16 --num-workers 8

    # Evaluate a checkpoint
    python experiments/bias_control/gate42_training.py \\
        --mode evaluate \\
        --checkpoint data/bias_control_medium/training_outputs/gate42_d1/checkpoint_epoch5.pt \\
        --output data/bias_control_medium/evaluations/gate42_d1_ep5.json \\
        --maestro-dir data/maestro_v3/maestro-v3.0.0
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.bias_control.architectures.cross_modal_model import CrossModalModel
from src.bias_control.training.preflight import (
    validate_training_setup,
    DriftSentinel,
    PARAM_RANGES,
)
from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
)
from src.bias_control.ratio_descriptors import (
    compute_descriptor_d3,
    compute_local_interval_features,
)

# Imports from sibling experiment scripts
from experiments.bias_control.gate4_ratio_auxiliary import (
    RatioEncoder,
    compute_batch_ratio_histograms,
    compute_batch_ratio_histograms_enriched,
)
from experiments.bias_control.evaluate_structured_pool import (
    build_segment_index,
    extract_all_embeddings,
    evaluate_with_precomputed_embeddings,
    analyze_hard_negatives_fast,
    PoolConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reused utilities (from bloqueA_training.py — duplicated to avoid fragile import)
# ---------------------------------------------------------------------------

class LinearWarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_last_lr(self) -> List[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {'step_count': self.step_count, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _dataloader_kwargs(num_workers: int) -> dict:
    if num_workers > 0:
        return {'pin_memory': True, 'prefetch_factor': 2}
    return {}


# ---------------------------------------------------------------------------
# Foundation loader (F2 fix — handles Run B / C / D checkpoints)
# ---------------------------------------------------------------------------

def load_foundation(checkpoint_path: str, device: torch.device) -> CrossModalModel:
    """
    Load foundation checkpoint for Gate 4.2.

    Accepts:
    - CrossModalModel pure state dict (Run B _base.pt or foundation.pt)
    - Full checkpoint with model_state_dict key
    - Wrapper checkpoints with 'base_model.' prefix (Run C) — strips prefix

    Always validates that the result is a valid CrossModalModel.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both wrapped checkpoints (dict with 'model_state_dict') and raw state dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and any(
        k.startswith(('audio_encoder.', 'midi_encoder.', 'base_model.'))
        for k in checkpoint.keys()
    ):
        # Raw state dict (from torch.save(model.state_dict(), path))
        state_dict = checkpoint
        checkpoint = {'model_state_dict': state_dict}  # normalize for epoch access later
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint format in {checkpoint_path}. "
            f"Expected dict with 'model_state_dict' or a raw state_dict."
        )

    # Detect wrapper prefix
    has_wrapper_prefix = any(k.startswith('base_model.') for k in state_dict)
    if has_wrapper_prefix:
        logger.info("Detected wrapper prefix in checkpoint — stripping 'base_model.'")
        state_dict = {
            k.replace('base_model.', '', 1): v
            for k, v in state_dict.items()
            if k.startswith('base_model.')
        }

    model = CrossModalModel(audio_encoder='lite', use_dann=False)
    result = model.load_state_dict(state_dict, strict=False)

    # Validate: no missing encoder/projection keys
    critical_missing = [
        k for k in result.missing_keys
        if 'audio_encoder' in k or 'midi_encoder' in k or 'projection' in k
    ]
    if critical_missing:
        raise RuntimeError(
            f"Foundation load missing critical keys: {critical_missing[:5]}"
        )

    # Allow DANN/adapter unexpected keys
    bad_unexpected = [
        k for k in result.unexpected_keys
        if not any(k.startswith(p) for p in ('dann.', 'audio_adapters.'))
    ]
    if bad_unexpected:
        raise RuntimeError(
            f"Foundation load has unexpected keys: {bad_unexpected[:5]}"
        )

    logger.info(
        f"Foundation loaded: {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch', '?')}, "
        f"missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)})"
    )
    return model


# ---------------------------------------------------------------------------
# Gate42Model: Wrapper for D0-D3 (auxiliary ratio branch)
# ---------------------------------------------------------------------------

class Gate42Model(nn.Module):
    """
    Gate 4.2 model with optional auxiliary ratio branch.

    D0: control — forward() + VICReg only.
    D1-D3: forward() + VICReg + auxiliary VICReg(audio, ratio) + VICReg(midi, ratio).

    forward() ALWAYS returns (audio_emb, midi_emb) for eval compatibility
    with extract_all_embeddings().
    """

    def __init__(
        self,
        base_model: CrossModalModel,
        descriptor_fn=None,
        ratio_encoder: Optional[RatioEncoder] = None,
        ratio_projection: Optional[nn.Module] = None,
        ratio_weight: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.descriptor_fn = descriptor_fn
        self.ratio_encoder = ratio_encoder
        self.ratio_projection = ratio_projection
        self.ratio_weight = ratio_weight

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        return self.base_model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        midi_onset: Optional[torch.Tensor] = None,
        midi_duration_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss: VICReg + optional auxiliary ratio VICReg.

        Returns (loss, metrics_dict).
        """
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )

        # Primary VICReg
        vicreg_loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        total_loss = vicreg_loss

        # Auxiliary ratio branch (D1-D3)
        if self.descriptor_fn is not None and self.ratio_encoder is not None:
            descriptor = self.descriptor_fn(
                midi_pitch=midi_pitch,
                midi_velocity=midi_velocity,
                midi_duration=midi_duration,
                midi_mask=midi_mask,
                midi_onset=midi_onset,
                midi_duration_sec=midi_duration_sec,
            )  # [B, descriptor_dim]

            ratio_emb = self.ratio_encoder(descriptor)  # [B, ratio_output_dim]
            ratio_proj = self.ratio_projection(ratio_emb)  # [B, 256]

            # Auxiliary VICReg: audio<->ratio + midi<->ratio
            loss_ar, _ = self.base_model.compute_vicreg_loss(audio_emb, ratio_proj)
            loss_mr, _ = self.base_model.compute_vicreg_loss(midi_emb, ratio_proj)
            aux_loss = (loss_ar + loss_mr) / 2.0

            total_loss = vicreg_loss + self.ratio_weight * aux_loss
            metrics['ratio_aux_loss'] = aux_loss.item()
            metrics['total_loss'] = total_loss.item()
        else:
            metrics['ratio_aux_loss'] = 0.0
            metrics['total_loss'] = vicreg_loss.item()

        return total_loss, metrics


# ---------------------------------------------------------------------------
# Gate42InputAugModel: Wrapper for D4 (input augmentation)
# ---------------------------------------------------------------------------

class Gate42InputAugModel(nn.Module):
    """
    Gate 4.2 model with input augmentation (D4).

    Concatenates local interval features to MIDI event embeddings,
    then projects back to embed_dim before the transformer.
    VICReg loss is standard (no auxiliary branch).
    """

    def __init__(self, base_model: CrossModalModel, interval_dim: int = 4):
        super().__init__()
        self.base_model = base_model
        self.interval_dim = interval_dim

        midi_embed_dim = base_model.midi_encoder.embed_dim  # 512
        self.interval_projection = nn.Sequential(
            nn.Linear(midi_embed_dim + interval_dim, midi_embed_dim),
            nn.LayerNorm(midi_embed_dim),
        )

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        audio_emb = self.base_model.encode_audio(audio)
        midi_emb = self._encode_midi_augmented(
            midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        return audio_emb, midi_emb

    def _encode_midi_augmented(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        duration: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Replicate MIDIEncoder pipeline with interval features injected
        after event_embedding and before positional encoding.

        Pipeline:
          event_emb → cat(interval_feats) → linear(516→512) → norm →
          [CLS if cls] → pos_encoding → transformer → output_norm → pooling → proj
        """
        enc = self.base_model.midi_encoder

        # 1. Event embedding: [B, T, 512]
        x = enc.event_embedding(pitch, velocity, duration)

        # 2. Compute interval features: [B, T, 4]
        interval_feats = compute_local_interval_features(pitch, mask)

        # 3. Concatenate and project: [B, T, 516] → [B, T, 512]
        x = torch.cat([x, interval_feats], dim=-1)
        x = self.interval_projection(x)

        # 4. CLS token (if using cls aggregation)
        B = pitch.shape[0]
        if enc.aggregation == "cls":
            cls_tokens = enc.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        # 5. Positional encoding
        x = enc.pos_encoding(x)

        # 6. Transformer
        if mask is not None:
            x = enc.transformer(x, src_key_padding_mask=mask)
        else:
            x = enc.transformer(x)

        # 7. Output norm
        x = enc.output_norm(x)

        # 8. Pooling
        if enc.aggregation == "mean":
            if mask is not None:
                m = ~mask.unsqueeze(-1)
                x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif enc.aggregation == "cls":
            x = x[:, 0, :]
        elif enc.aggregation == "attention":
            weights = enc.attention_pool(x)
            if mask is not None:
                weights = weights.masked_fill(mask.unsqueeze(-1), float("-inf"))
            weights = torch.softmax(weights, dim=1)
            x = (x * weights).sum(dim=1)

        # 9. MIDI projection (from CrossModalModel)
        x = self.base_model.midi_projection(x)

        return x  # [B, 256]

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        midi_onset: Optional[torch.Tensor] = None,
        midi_duration_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """VICReg only (no auxiliary branch for D4)."""
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# Audio-augmented models (Gate 4.3)
# ---------------------------------------------------------------------------

from src.bias_control.audio_descriptors import (
    compute_audio_descriptor_a4,
    compute_audio_descriptor_a7,
    compute_audio_descriptor_a8,
    compute_audio_descriptor_a9,
)


def _encode_audio_with_descriptor(
    base_model: CrossModalModel,
    audio: torch.Tensor,
    descriptor_type: str,
    descriptor_projection: nn.Module,
) -> torch.Tensor:
    """
    Audio encoder augmented with descriptor injection.
    Replicates MERTEncoderLite pipeline with descriptor injected
    after CNN and before positional encoding + Transformer.

    Used by Gate42AudioAugModel and Gate42DualAugModel.
    """
    enc = base_model.audio_encoder

    # 1. CNN: [B, 1024, T'] → transpose → [B, T', 1024]
    waveform = audio.unsqueeze(1) if audio.dim() == 2 else audio
    features = enc.feature_extractor(waveform).transpose(1, 2)
    T_prime = features.size(1)

    # 2. Compute audio descriptor (no grad — pure signal processing)
    with torch.no_grad():
        if descriptor_type == 'a4':
            desc = compute_audio_descriptor_a4(audio, target_length=T_prime)
        elif descriptor_type == 'a7':
            desc = compute_audio_descriptor_a7(audio, target_length=T_prime)
        else:
            raise ValueError(f"Unknown audio descriptor type: {descriptor_type}")

    # 3. Concat + project: [B, T', 1024+K] → [B, T', 1024]
    features = torch.cat([features, desc.detach()], dim=-1)
    features = descriptor_projection(features)

    # 4. Positional embeddings
    T = features.size(1)
    if T <= enc.max_pos_len:
        features = features + enc.pos_embedding[:, :T, :]
    else:
        pos = F.interpolate(
            enc.pos_embedding.transpose(1, 2), size=T,
            mode='linear', align_corners=False,
        ).transpose(1, 2)
        features = features + pos

    # 5. Transformer
    encoded = enc.transformer(features)

    # 6. Mean pooling → [B, 1024]
    embeddings = encoded.mean(dim=1)

    # 7. Audio projection → [B, 256]
    return base_model.audio_projection(embeddings)


def _encode_midi_with_intervals(
    base_model: CrossModalModel,
    pitch: torch.Tensor,
    velocity: torch.Tensor,
    duration: torch.Tensor,
    mask: Optional[torch.Tensor],
    interval_projection: nn.Module,
    interval_dim: int = 4,
) -> torch.Tensor:
    """
    MIDI encoder augmented with local interval features.
    Replicates MIDIEncoder pipeline with intervals injected
    after event_embedding and before positional encoding.

    Used by Gate42InputAugModel (refactor) and Gate42DualAugModel.
    """
    enc = base_model.midi_encoder

    # 1. Event embedding: [B, T, 512]
    x = enc.event_embedding(pitch, velocity, duration)

    # 2. Compute interval features: [B, T, 4]
    interval_feats = compute_local_interval_features(pitch, mask)

    # 3. Concatenate and project: [B, T, 516] → [B, T, 512]
    x = torch.cat([x, interval_feats], dim=-1)
    x = interval_projection(x)

    # 4. CLS token (if using cls aggregation)
    B = pitch.shape[0]
    if enc.aggregation == "cls":
        cls_tokens = enc.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

    # 5. Positional encoding
    x = enc.pos_encoding(x)

    # 6. Transformer
    if mask is not None:
        x = enc.transformer(x, src_key_padding_mask=mask)
    else:
        x = enc.transformer(x)

    # 7. Output norm
    x = enc.output_norm(x)

    # 8. Pooling
    if enc.aggregation == "mean":
        if mask is not None:
            m = ~mask.unsqueeze(-1)
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
    elif enc.aggregation == "cls":
        x = x[:, 0, :]
    elif enc.aggregation == "attention":
        weights = enc.attention_pool(x)
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(weights, dim=1)
        x = (x * weights).sum(dim=1)

    # 9. MIDI projection → [B, 256]
    return base_model.midi_projection(x)


class Gate42AudioAugModel(nn.Module):
    """
    Gate 4.2/4.3 model with audio-side descriptor injection (A4 or A7).

    Injects audio descriptor after CNN output, before audio Transformer.
    MIDI encoder passes through unchanged.
    """

    def __init__(self, base_model: CrossModalModel,
                 audio_descriptor_type: str = 'a4',
                 audio_descriptor_dim: int = 8):
        super().__init__()
        self.base_model = base_model
        self.audio_descriptor_type = audio_descriptor_type
        self.audio_descriptor_dim = audio_descriptor_dim

        audio_embed_dim = base_model.audio_encoder.output_dim  # 1024
        self.audio_descriptor_projection = nn.Sequential(
            nn.Linear(audio_embed_dim + audio_descriptor_dim, audio_embed_dim),
            nn.LayerNorm(audio_embed_dim),
        )

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        audio_emb = _encode_audio_with_descriptor(
            self.base_model, audio, self.audio_descriptor_type,
            self.audio_descriptor_projection,
        )
        midi_emb = self.base_model.encode_midi(
            pitch=midi_pitch, velocity=midi_velocity,
            duration=midi_duration, padding_mask=midi_mask,
        )
        return audio_emb, midi_emb

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        midi_onset: Optional[torch.Tensor] = None,
        midi_duration_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """VICReg only (no auxiliary branch)."""
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


class Gate42DualAugModel(nn.Module):
    """
    Gate 4.3 model with dual injection: D4 in MIDI + A4/A7 in audio.

    Combines interval injection (MIDI side) with audio descriptor injection.
    """

    def __init__(self, base_model: CrossModalModel,
                 audio_descriptor_type: str = 'a4',
                 audio_descriptor_dim: int = 8,
                 interval_dim: int = 4):
        super().__init__()
        self.base_model = base_model
        self.audio_descriptor_type = audio_descriptor_type
        self.audio_descriptor_dim = audio_descriptor_dim
        self.interval_dim = interval_dim

        # Audio side projection
        audio_embed_dim = base_model.audio_encoder.output_dim  # 1024
        self.audio_descriptor_projection = nn.Sequential(
            nn.Linear(audio_embed_dim + audio_descriptor_dim, audio_embed_dim),
            nn.LayerNorm(audio_embed_dim),
        )

        # MIDI side projection (same as Gate42InputAugModel)
        midi_embed_dim = base_model.midi_encoder.embed_dim  # 512
        self.interval_projection = nn.Sequential(
            nn.Linear(midi_embed_dim + interval_dim, midi_embed_dim),
            nn.LayerNorm(midi_embed_dim),
        )

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        audio_emb = _encode_audio_with_descriptor(
            self.base_model, audio, self.audio_descriptor_type,
            self.audio_descriptor_projection,
        )
        midi_emb = _encode_midi_with_intervals(
            self.base_model, midi_pitch, midi_velocity, midi_duration,
            midi_mask, self.interval_projection, self.interval_dim,
        )
        return audio_emb, midi_emb

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        midi_onset: Optional[torch.Tensor] = None,
        midi_duration_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """VICReg only (no auxiliary branch)."""
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# Cross-attention audio-augmented models (Gate 4.3 extension)
# ---------------------------------------------------------------------------

def _encode_audio_with_cross_attention(
    base_model: CrossModalModel,
    audio: torch.Tensor,
    descriptor_type: str,
    descriptor_kv_proj: nn.Module,
    cross_attention: nn.MultiheadAttention,
    cross_attn_norm: nn.Module,
) -> torch.Tensor:
    """
    Audio encoder with cross-attention descriptor injection.

    Features (Q) attend to descriptor (K,V). Audio frames "ask" the ratio
    descriptor which information is relevant. Descriptor stays at native STFT
    resolution (~188 frames); cross-attention handles Q/K length mismatch.

    Pipeline:
      CNN → pos_emb → cross_attention(Q=features, K/V=descriptor) → residual+LN → Transformer → pool → proj
    """
    enc = base_model.audio_encoder

    # 1. CNN features [B, T'=2400, 1024]
    waveform = audio.unsqueeze(1) if audio.dim() == 2 else audio
    features = enc.feature_extractor(waveform).transpose(1, 2)

    # 2. Add positional embeddings BEFORE cross-attention (temporal awareness)
    T = features.size(1)
    if T <= enc.max_pos_len:
        features = features + enc.pos_embedding[:, :T, :]
    else:
        pos = F.interpolate(
            enc.pos_embedding.transpose(1, 2), size=T,
            mode='linear', align_corners=False,
        ).transpose(1, 2)
        features = features + pos

    # 3. Descriptor at NATIVE STFT resolution [B, T_stft~188, K]
    with torch.no_grad():
        if descriptor_type == 'a4':
            desc = compute_audio_descriptor_a4(audio, target_length=None)
        elif descriptor_type == 'a7':
            desc = compute_audio_descriptor_a7(audio, target_length=None)
        else:
            raise ValueError(f"Unknown audio descriptor type: {descriptor_type}")

    # 4. Cross-attention: features (Q=2400) attend to descriptor (K/V=188)
    desc_proj = descriptor_kv_proj(desc.detach())  # [B, T_stft, 1024]
    attn_output, _ = cross_attention(
        query=features,       # [B, 2400, 1024]
        key=desc_proj,        # [B, T_stft, 1024]
        value=desc_proj,      # [B, T_stft, 1024]
        need_weights=False,
    )
    features = cross_attn_norm(features + attn_output)  # residual + norm

    # 5. Transformer + pool (pos_emb already added in step 2 — do NOT add again)
    encoded = enc.transformer(features)
    embeddings = encoded.mean(dim=1)  # [B, 1024]

    # 6. Audio projection → [B, 256]
    return base_model.audio_projection(embeddings)


class Gate42AudioCrossAttModel(nn.Module):
    """
    Gate 4.3 model with cross-attention audio descriptor injection (A4x or A7x).

    Features (Q) attend to descriptor (K,V) — audio learns to selectively
    consult ratio information. Descriptor at native STFT resolution.
    """

    def __init__(self, base_model: CrossModalModel,
                 audio_descriptor_type: str = 'a4',
                 audio_descriptor_dim: int = 8):
        super().__init__()
        self.base_model = base_model
        self.audio_descriptor_type = audio_descriptor_type
        self.audio_descriptor_dim = audio_descriptor_dim

        audio_embed_dim = base_model.audio_encoder.output_dim  # 1024

        # Project low-dim descriptor to model dim for K,V
        self.descriptor_kv_proj = nn.Linear(audio_descriptor_dim, audio_embed_dim)

        # Cross-attention: features query descriptor
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=audio_embed_dim, num_heads=8,
            batch_first=True, dropout=0.1,
        )
        self.cross_attn_norm = nn.LayerNorm(audio_embed_dim)

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        audio_emb = _encode_audio_with_cross_attention(
            self.base_model, audio, self.audio_descriptor_type,
            self.descriptor_kv_proj, self.cross_attention, self.cross_attn_norm,
        )
        midi_emb = self.base_model.encode_midi(
            pitch=midi_pitch, velocity=midi_velocity,
            duration=midi_duration, padding_mask=midi_mask,
        )
        return audio_emb, midi_emb

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        midi_onset: Optional[torch.Tensor] = None,
        midi_duration_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """VICReg only (no auxiliary branch)."""
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# MIDI Cross-Attention helpers + model (D4x)
# ---------------------------------------------------------------------------

def _encode_midi_with_cross_attention(
    base_model: CrossModalModel,
    pitch: torch.Tensor,
    velocity: torch.Tensor,
    duration: torch.Tensor,
    mask: Optional[torch.Tensor],
    interval_kv_proj: nn.Module,
    cross_attention: nn.MultiheadAttention,
    cross_attn_norm: nn.Module,
) -> torch.Tensor:
    """
    MIDI encoder with cross-attention interval injection.

    Embeddings (Q) attend to interval features (K,V). Each MIDI token can
    attend to intervals at ALL positions (non-local access), unlike concat
    where each token only sees its own interval.

    Pipeline:
      event_emb → pos_enc → cross_attn(Q=emb, K/V=intervals) → residual+LN → Transformer → pool → proj

    Note: Q and K/V have the SAME sequence length (N tokens). No temporal
    mismatch to resolve, unlike audio where Q=2400 and K/V=188.
    """
    enc = base_model.midi_encoder

    # 1. Event embedding: [B, N, 512]
    x = enc.event_embedding(pitch, velocity, duration)

    # 2. CLS token (if using cls aggregation) — BEFORE pos_encoding
    B = pitch.shape[0]
    cross_attn_mask = mask  # Keep original mask for cross-attention K/V
    if enc.aggregation == "cls":
        cls_tokens = enc.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

    # 3. Positional encoding BEFORE cross-attention (temporal awareness)
    x = enc.pos_encoding(x)

    # 4. Compute interval features at native resolution: [B, N, 4]
    with torch.no_grad():
        interval_feats = compute_local_interval_features(pitch, cross_attn_mask)

    # 5. Cross-attention: embeddings (Q=N) attend to intervals (K/V=N)
    interval_proj = interval_kv_proj(interval_feats.detach())  # [B, N, 512]

    # If CLS was prepended to Q, we need to handle K/V length matching
    # K/V stays at original N (without CLS) — cross-attn handles Lq != Lk
    attn_output, _ = cross_attention(
        query=x,               # [B, N(+1 if cls), 512]
        key=interval_proj,     # [B, N, 512]
        value=interval_proj,   # [B, N, 512]
        need_weights=False,
    )
    x = cross_attn_norm(x + attn_output)  # residual + norm

    # 6. Transformer (pos_enc already applied in step 3 — do NOT add again)
    if mask is not None:
        x = enc.transformer(x, src_key_padding_mask=mask)
    else:
        x = enc.transformer(x)

    # 7. Output norm
    x = enc.output_norm(x)

    # 8. Pooling
    if enc.aggregation == "mean":
        if mask is not None:
            m = ~mask.unsqueeze(-1)
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
    elif enc.aggregation == "cls":
        x = x[:, 0, :]
    elif enc.aggregation == "attention":
        weights = enc.attention_pool(x)
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(weights, dim=1)
        x = (x * weights).sum(dim=1)

    # 9. MIDI projection → [B, 256]
    return base_model.midi_projection(x)


class Gate42MidiCrossAttModel(nn.Module):
    """
    Gate 4.3 model with cross-attention MIDI interval injection (D4x).

    Embeddings (Q) attend to interval features (K,V) — each MIDI token can
    selectively consult interval information from all positions.
    Same pattern as Gate42AudioCrossAttModel but on MIDI side (d=512).
    """

    def __init__(self, base_model: CrossModalModel, interval_dim: int = 4):
        super().__init__()
        self.base_model = base_model
        self.interval_dim = interval_dim

        midi_embed_dim = base_model.midi_encoder.embed_dim  # 512

        # Project low-dim intervals to model dim for K,V
        self.interval_kv_proj = nn.Linear(interval_dim, midi_embed_dim)

        # Cross-attention: embeddings query intervals
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=midi_embed_dim, num_heads=8,
            batch_first=True, dropout=0.1,
        )
        self.cross_attn_norm = nn.LayerNorm(midi_embed_dim)

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        audio_emb = self.base_model.encode_audio(audio)
        midi_emb = _encode_midi_with_cross_attention(
            self.base_model, midi_pitch, midi_velocity, midi_duration, midi_mask,
            self.interval_kv_proj, self.cross_attention, self.cross_attn_norm,
        )
        return audio_emb, midi_emb

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        midi_onset: Optional[torch.Tensor] = None,
        midi_duration_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """VICReg only (no auxiliary branch)."""
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# Cross-modal dual injection (Gate 4.3 — d4a4cm)
# ---------------------------------------------------------------------------

def _encode_audio_with_cross_modal_intervals(
    base_model: CrossModalModel,
    audio: torch.Tensor,
    pitch: torch.Tensor,
    mask: Optional[torch.Tensor],
    cross_modal_audio_projection: nn.Module,
    interval_dim: int = 4,
) -> torch.Tensor:
    """
    Audio encoder augmented with MIDI interval features (cross-modal).

    Instead of same-modality audio descriptors, injects MIDI intervals into the
    audio encoder. Intervals are interpolated from MIDI sequence length (N) to
    CNN temporal resolution (T'=2400).

    Pipeline:
      CNN → concat(features, interp_intervals) → proj → pos_emb → Transformer → pool → proj
    """
    enc = base_model.audio_encoder

    # 1. CNN: [B, 1024, T'] → transpose → [B, T', 1024]
    waveform = audio.unsqueeze(1) if audio.dim() == 2 else audio
    features = enc.feature_extractor(waveform).transpose(1, 2)
    T_prime = features.size(1)

    # 2. Compute MIDI interval features: [B, N, 4]
    with torch.no_grad():
        interval_feats = compute_local_interval_features(pitch, mask)

    # 3. Interpolate from MIDI resolution (N) to audio resolution (T')
    # [B, N, 4] → [B, 4, N] → interpolate → [B, 4, T'] → [B, T', 4]
    interp = F.interpolate(
        interval_feats.detach().transpose(1, 2),
        size=T_prime, mode='linear', align_corners=False,
    ).transpose(1, 2)

    # 4. Concat + project: [B, T', 1024+4] → [B, T', 1024]
    features = torch.cat([features, interp], dim=-1)
    features = cross_modal_audio_projection(features)

    # 5. Positional embeddings
    T = features.size(1)
    if T <= enc.max_pos_len:
        features = features + enc.pos_embedding[:, :T, :]
    else:
        pos = F.interpolate(
            enc.pos_embedding.transpose(1, 2), size=T,
            mode='linear', align_corners=False,
        ).transpose(1, 2)
        features = features + pos

    # 6. Transformer + pool + projection
    encoded = enc.transformer(features)
    embeddings = encoded.mean(dim=1)
    return base_model.audio_projection(embeddings)


def _encode_midi_with_cross_modal_audio_desc(
    base_model: CrossModalModel,
    audio: torch.Tensor,
    pitch: torch.Tensor,
    velocity: torch.Tensor,
    duration: torch.Tensor,
    mask: Optional[torch.Tensor],
    cross_modal_midi_projection: nn.Module,
    audio_descriptor_type: str = 'a4',
) -> torch.Tensor:
    """
    MIDI encoder augmented with audio descriptor features (cross-modal).

    Instead of same-modality MIDI intervals, injects audio spectral descriptors
    into the MIDI encoder. Descriptors are interpolated from STFT resolution
    (~188 frames) to MIDI sequence length (N).

    Pipeline:
      event_emb → concat(emb, interp_audio_desc) → proj → CLS → pos_enc → Transformer → pool → proj
    """
    enc = base_model.midi_encoder

    # 1. Event embedding: [B, N, 512]
    x = enc.event_embedding(pitch, velocity, duration)
    N = x.size(1)

    # 2. Compute audio descriptor at native STFT resolution: [B, T_stft, K]
    with torch.no_grad():
        if audio_descriptor_type == 'a4':
            desc = compute_audio_descriptor_a4(audio, target_length=None)
        else:
            desc = compute_audio_descriptor_a7(audio, target_length=None)

    # 3. Interpolate from STFT resolution to MIDI sequence length
    # [B, T_stft, K] → [B, K, T_stft] → interpolate → [B, K, N] → [B, N, K]
    interp = F.interpolate(
        desc.detach().transpose(1, 2),
        size=N, mode='linear', align_corners=False,
    ).transpose(1, 2)

    # 4. Concat + project: [B, N, 512+K] → [B, N, 512]
    x = torch.cat([x, interp], dim=-1)
    x = cross_modal_midi_projection(x)

    # 5. CLS token (if using cls aggregation)
    B = pitch.shape[0]
    if enc.aggregation == "cls":
        cls_tokens = enc.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

    # 6. Positional encoding
    x = enc.pos_encoding(x)

    # 7. Transformer
    if mask is not None:
        x = enc.transformer(x, src_key_padding_mask=mask)
    else:
        x = enc.transformer(x)

    # 8. Output norm
    x = enc.output_norm(x)

    # 9. Pooling
    if enc.aggregation == "mean":
        if mask is not None:
            m = ~mask.unsqueeze(-1)
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
    elif enc.aggregation == "cls":
        x = x[:, 0, :]
    elif enc.aggregation == "attention":
        weights = enc.attention_pool(x)
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(weights, dim=1)
        x = (x * weights).sum(dim=1)

    # 10. MIDI projection → [B, 256]
    return base_model.midi_projection(x)


class Gate42DualCrossModalModel(nn.Module):
    """
    Gate 4.3 cross-modal dual injection: descriptors from one domain injected
    into the OTHER domain's encoder.

    - Audio encoder receives MIDI interval features (D4, interpolated N→T')
    - MIDI encoder receives audio spectral descriptors (A4, interpolated T_stft→N)

    Contrast with Gate42DualAugModel (same-modality): each descriptor goes into
    its OWN domain's encoder. Here they cross over.
    """

    def __init__(self, base_model: CrossModalModel,
                 audio_descriptor_type: str = 'a4',
                 audio_descriptor_dim: int = 8,
                 interval_dim: int = 4):
        super().__init__()
        self.base_model = base_model
        self.audio_descriptor_type = audio_descriptor_type
        self.audio_descriptor_dim = audio_descriptor_dim
        self.interval_dim = interval_dim

        audio_embed_dim = base_model.audio_encoder.output_dim  # 1024
        midi_embed_dim = base_model.midi_encoder.embed_dim  # 512

        # Audio encoder receives MIDI intervals (4d)
        self.cross_modal_audio_projection = nn.Sequential(
            nn.Linear(audio_embed_dim + interval_dim, audio_embed_dim),
            nn.LayerNorm(audio_embed_dim),
        )

        # MIDI encoder receives audio descriptors (8d or 12d)
        self.cross_modal_midi_projection = nn.Sequential(
            nn.Linear(midi_embed_dim + audio_descriptor_dim, midi_embed_dim),
            nn.LayerNorm(midi_embed_dim),
        )

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        # Audio encoder gets MIDI intervals (cross-modal)
        audio_emb = _encode_audio_with_cross_modal_intervals(
            self.base_model, audio, midi_pitch, midi_mask,
            self.cross_modal_audio_projection, self.interval_dim,
        )
        # MIDI encoder gets audio descriptors (cross-modal)
        midi_emb = _encode_midi_with_cross_modal_audio_desc(
            self.base_model, audio, midi_pitch, midi_velocity, midi_duration,
            midi_mask, self.cross_modal_midi_projection, self.audio_descriptor_type,
        )
        return audio_emb, midi_emb

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        midi_onset: Optional[torch.Tensor] = None,
        midi_duration_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """VICReg only (no auxiliary branch)."""
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# Reverse cross-attention model (Gate 4.3 Fase 5: A4r)
# ---------------------------------------------------------------------------

def _encode_audio_with_reverse_cross_attention(
    base_model: CrossModalModel,
    audio: torch.Tensor,
    descriptor_type: str,
    descriptor_q_proj: nn.Module,
    desc_pos_embedding: nn.Parameter,
    cross_attention: nn.MultiheadAttention,
    cross_attn_norm: nn.Module,
) -> torch.Tensor:
    """
    Reverse cross-attention: descriptors (Q) attend to features (K/V).

    Pipeline:
      CNN → +pos_emb → (K/V)
      STFT → descriptor → q_proj → +desc_pos_emb → (Q)
      cross_attn(Q=desc, K/V=features) → residual+LN → Transformer(188 tokens) → pool → proj

    Key difference: Transformer processes 188 tokens (vs 2400 in regular),
    so self-attention is 12.8x cheaper per layer.
    """
    enc = base_model.audio_encoder

    # 1. CNN features [B, T'=2400, 1024]
    waveform = audio.unsqueeze(1) if audio.dim() == 2 else audio
    features = enc.feature_extractor(waveform).transpose(1, 2)

    # 2. Add pos_emb to features (K/V needs temporal info)
    T = features.size(1)
    if T <= enc.max_pos_len:
        features = features + enc.pos_embedding[:, :T, :]
    else:
        pos = F.interpolate(
            enc.pos_embedding.transpose(1, 2), size=T,
            mode='linear', align_corners=False,
        ).transpose(1, 2)
        features = features + pos

    # 3. Descriptor at NATIVE STFT resolution [B, T_stft~188, K]
    with torch.no_grad():
        if descriptor_type == 'a4':
            desc = compute_audio_descriptor_a4(audio, target_length=None)
        elif descriptor_type == 'a7':
            desc = compute_audio_descriptor_a7(audio, target_length=None)
        elif descriptor_type == 'a8':
            desc = compute_audio_descriptor_a8(audio, target_length=None)
        elif descriptor_type == 'a9':
            desc = compute_audio_descriptor_a9(audio, target_length=None)
        else:
            raise ValueError(f"Unknown audio descriptor type: {descriptor_type}")

    # 4. Project descriptor to Q dimension + positional embedding
    desc_proj = descriptor_q_proj(desc.detach())  # [B, T_stft, 1024]
    T_desc = desc_proj.size(1)
    desc_proj = desc_proj + desc_pos_embedding[:, :T_desc, :]

    # 5. REVERSE cross-attention: descriptor (Q) attends to features (K/V)
    attn_output, _ = cross_attention(
        query=desc_proj,    # [B, 188, 1024]  — descriptors ASK
        key=features,       # [B, 2400, 1024] — features ANSWER
        value=features,     # [B, 2400, 1024]
        need_weights=False,
    )
    desc_proj = cross_attn_norm(desc_proj + attn_output)  # residual + norm

    # 6. Transformer (reuses enc.transformer — length-agnostic, 188 tokens)
    encoded = enc.transformer(desc_proj)  # [B, 188, 1024]
    embeddings = encoded.mean(dim=1)      # [B, 1024]

    # 7. Audio projection → [B, 256]
    return base_model.audio_projection(embeddings)


class Gate42AudioReverseCrossAttModel(nn.Module):
    """
    Gate 4.3 Fase 5: Reverse cross-attention audio descriptor injection (A4r).

    Descriptors (Q) attend to features (K/V) — ratios organize features.
    Transformer processes 188 tokens instead of 2400 (12.8x less self-attn compute).
    """

    def __init__(self, base_model: CrossModalModel,
                 audio_descriptor_type: str = 'a4',
                 audio_descriptor_dim: int = 8):
        super().__init__()
        self.base_model = base_model
        self.audio_descriptor_type = audio_descriptor_type
        self.audio_descriptor_dim = audio_descriptor_dim

        audio_embed_dim = base_model.audio_encoder.output_dim  # 1024

        self.descriptor_q_proj = nn.Linear(audio_descriptor_dim, audio_embed_dim)
        self.desc_pos_embedding = nn.Parameter(
            torch.randn(1, 200, audio_embed_dim) * 0.02
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=audio_embed_dim, num_heads=8,
            batch_first=True, dropout=0.1,
        )
        self.cross_attn_norm = nn.LayerNorm(audio_embed_dim)

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_emb = _encode_audio_with_reverse_cross_attention(
            self.base_model, audio, self.audio_descriptor_type,
            self.descriptor_q_proj, self.desc_pos_embedding,
            self.cross_attention, self.cross_attn_norm,
        )
        midi_emb = self.base_model.encode_midi(
            pitch=midi_pitch, velocity=midi_velocity,
            duration=midi_duration, padding_mask=midi_mask,
        )
        return audio_emb, midi_emb

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        midi_onset: Optional[torch.Tensor] = None,
        midi_duration_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# Descriptor function wrappers
# ---------------------------------------------------------------------------

def _make_descriptor_fn(descriptor: str):
    """
    Return a callable(**kwargs) → [B, dim] or None.

    D0/D4 return None (no auxiliary descriptor).
    D1/D2/D3 return the appropriate histogram/descriptor tensor.
    """
    if descriptor == 'd0':
        return None
    elif descriptor == 'd1':
        def fn(midi_pitch, midi_mask, **kwargs):
            return compute_batch_ratio_histograms(
                midi_pitch, midi_mask, n_bins=128, max_notes=128,
            )
        return fn
    elif descriptor == 'd2':
        def fn(midi_pitch, midi_velocity, midi_duration, midi_mask, **kwargs):
            hist = compute_batch_ratio_histograms_enriched(
                midi_pitch, midi_velocity, midi_duration, midi_mask,
                n_bins=128, max_notes=128,
            )  # [B, 128, 3]
            return hist.view(hist.size(0), -1)  # [B, 384]
        return fn
    elif descriptor == 'd3':
        def fn(midi_pitch, midi_onset, midi_duration_sec, midi_mask, **kwargs):
            return compute_descriptor_d3(
                midi_pitch, midi_onset, midi_duration_sec, midi_mask,
            )
        return fn
    elif descriptor == 'd4':
        return None  # D4 uses input augmentation, not auxiliary branch
    else:
        raise ValueError(f"Unknown descriptor: {descriptor}")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_gate42_model(
    descriptor: str,
    base_model: CrossModalModel,
    ratio_weight: float = 0.1,
) -> nn.Module:
    """Create Gate 4.2 model for the given descriptor."""
    if descriptor == 'd4':
        return Gate42InputAugModel(base_model, interval_dim=4)
    elif descriptor == 'a4':
        return Gate42AudioAugModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
    elif descriptor == 'a7':
        return Gate42AudioAugModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
    elif descriptor == 'd4a4':
        return Gate42DualAugModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
    elif descriptor == 'd4a7':
        return Gate42DualAugModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
    elif descriptor == 'a4x':
        return Gate42AudioCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
    elif descriptor == 'a7x':
        return Gate42AudioCrossAttModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
    elif descriptor == 'd4x':
        return Gate42MidiCrossAttModel(base_model, interval_dim=4)
    elif descriptor == 'd4a4cm':
        return Gate42DualCrossModalModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
    elif descriptor == 'a4r':
        return Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)

    descriptor_fn = _make_descriptor_fn(descriptor)

    if descriptor == 'd0':
        return Gate42Model(
            base_model, descriptor_fn=None,
            ratio_encoder=None, ratio_projection=None,
            ratio_weight=0.0,
        )
    elif descriptor == 'd1':
        ratio_encoder = RatioEncoder(n_bins=128, n_channels=1, hidden_dim=128, output_dim=64)
        ratio_projection = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
        )
        return Gate42Model(
            base_model, descriptor_fn=descriptor_fn,
            ratio_encoder=ratio_encoder, ratio_projection=ratio_projection,
            ratio_weight=ratio_weight,
        )
    elif descriptor == 'd2':
        ratio_encoder = RatioEncoder(n_bins=128, n_channels=3, hidden_dim=256, output_dim=128)
        ratio_projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
        )
        return Gate42Model(
            base_model, descriptor_fn=descriptor_fn,
            ratio_encoder=ratio_encoder, ratio_projection=ratio_projection,
            ratio_weight=ratio_weight,
        )
    elif descriptor == 'd3':
        ratio_encoder = RatioEncoder(n_bins=153, n_channels=1, hidden_dim=128, output_dim=64)
        ratio_projection = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
        )
        return Gate42Model(
            base_model, descriptor_fn=descriptor_fn,
            ratio_encoder=ratio_encoder, ratio_projection=ratio_projection,
            ratio_weight=ratio_weight,
        )
    else:
        raise ValueError(f"Unknown descriptor: {descriptor}")


# ---------------------------------------------------------------------------
# Freeze policy (Run B style: CNN + pos_embedding + layers 0-1 frozen)
# ---------------------------------------------------------------------------

def apply_freeze_policy(base_model: CrossModalModel, policy: str = 'run-b'):
    """Apply freeze policy to foundation model.

    Policies:
      run-b: CNN + PosEmb + layers 0-1 frozen, layers 2-3 trainable
      run-d: CNN + PosEmb frozen, all transformer layers trainable (split-LR)
    """
    # Freeze CNN (always)
    for p in base_model.audio_encoder.feature_extractor.parameters():
        p.requires_grad = False
    # Freeze PosEmb (always)
    base_model.audio_encoder.pos_embedding.requires_grad = False

    if policy == 'run-b':
        # Freeze layers 0-1
        for p in base_model.audio_encoder.transformer.layers[0].parameters():
            p.requires_grad = False
        for p in base_model.audio_encoder.transformer.layers[1].parameters():
            p.requires_grad = False
        # Layers 2-3 stay trainable (default)
    elif policy == 'run-d':
        # All transformer layers trainable (split-LR handled by optimizer)
        pass
    else:
        raise ValueError(f"Unknown freeze policy: {policy}")


# ---------------------------------------------------------------------------
# Optimizer creation
# ---------------------------------------------------------------------------

def create_gate42_optimizer(
    model: nn.Module,
    descriptor: str,
    freeze_policy: str = 'run-b',
    lr_audio: float = 1e-5,
    lr_audio_low: float = 5e-6,
    lr_midi: float = 5e-5,
    lr_proj: float = 1e-4,
    lr_ratio: float = 5e-4,
) -> AdamW:
    """Create optimizer with param groups for Gate 4.2.

    Args:
        freeze_policy: 'run-b' (layers 2-3 only) or 'run-d' (all layers, split-LR)
        lr_audio: LR for audio layers 2-3
        lr_audio_low: LR for audio layers 0-1 (only used with run-d)
    """
    base = model.base_model

    # Audio param groups depend on freeze policy
    if freeze_policy == 'run-d':
        audio_groups = [
            {
                'params': (
                    list(base.audio_encoder.transformer.layers[0].parameters()) +
                    list(base.audio_encoder.transformer.layers[1].parameters())
                ),
                'lr': lr_audio_low,
                'name': 'audio_layers_0_1',
            },
            {
                'params': (
                    list(base.audio_encoder.transformer.layers[2].parameters()) +
                    list(base.audio_encoder.transformer.layers[3].parameters())
                ),
                'lr': lr_audio,
                'name': 'audio_layers_2_3',
            },
        ]
    else:
        audio_groups = [
            {
                'params': (
                    list(base.audio_encoder.transformer.layers[2].parameters()) +
                    list(base.audio_encoder.transformer.layers[3].parameters())
                ),
                'lr': lr_audio,
                'name': 'audio_layers_2_3',
            },
        ]

    param_groups = audio_groups + [
        {
            'params': list(base.midi_encoder.parameters()),
            'lr': lr_midi,
            'name': 'midi_encoder',
        },
        {
            'params': (
                list(base.audio_projection.parameters()) +
                list(base.midi_projection.parameters())
            ),
            'lr': lr_proj,
            'name': 'projections',
        },
    ]

    # Add descriptor-specific params
    if descriptor == 'd4':
        param_groups.append({
            'params': list(model.interval_projection.parameters()),
            'lr': lr_ratio,
            'name': 'interval_projection',
        })
    elif descriptor in ('a4', 'a7'):
        param_groups.append({
            'params': list(model.audio_descriptor_projection.parameters()),
            'lr': lr_ratio,
            'name': 'audio_descriptor_projection',
        })
    elif descriptor in ('d4a4', 'd4a7'):
        param_groups.append({
            'params': list(model.audio_descriptor_projection.parameters()),
            'lr': lr_ratio,
            'name': 'audio_descriptor_projection',
        })
        param_groups.append({
            'params': list(model.interval_projection.parameters()),
            'lr': lr_ratio,
            'name': 'interval_projection',
        })
    elif descriptor in ('a4x', 'a7x'):
        param_groups.append({
            'params': list(model.descriptor_kv_proj.parameters()),
            'lr': lr_ratio,
            'name': 'descriptor_kv_proj',
        })
        param_groups.append({
            'params': list(model.cross_attention.parameters()),
            'lr': lr_ratio,
            'name': 'cross_attention',
        })
        param_groups.append({
            'params': list(model.cross_attn_norm.parameters()),
            'lr': lr_ratio,
            'name': 'cross_attn_norm',
        })
    elif descriptor == 'd4x':
        param_groups.append({
            'params': list(model.interval_kv_proj.parameters()),
            'lr': lr_ratio,
            'name': 'interval_kv_proj',
        })
        param_groups.append({
            'params': list(model.cross_attention.parameters()),
            'lr': lr_ratio,
            'name': 'cross_attention',
        })
        param_groups.append({
            'params': list(model.cross_attn_norm.parameters()),
            'lr': lr_ratio,
            'name': 'cross_attn_norm',
        })
    elif descriptor == 'd4a4cm':
        param_groups.append({
            'params': list(model.cross_modal_audio_projection.parameters()),
            'lr': lr_ratio,
            'name': 'cross_modal_audio_projection',
        })
        param_groups.append({
            'params': list(model.cross_modal_midi_projection.parameters()),
            'lr': lr_ratio,
            'name': 'cross_modal_midi_projection',
        })
    elif descriptor == 'a4r':
        param_groups.append({
            'params': list(model.descriptor_q_proj.parameters()),
            'lr': lr_ratio,
            'name': 'descriptor_q_proj',
        })
        param_groups.append({
            'params': [model.desc_pos_embedding],
            'lr': lr_ratio,
            'name': 'desc_pos_embedding',
        })
        param_groups.append({
            'params': list(model.cross_attention.parameters()),
            'lr': lr_ratio,
            'name': 'cross_attention',
        })
        param_groups.append({
            'params': list(model.cross_attn_norm.parameters()),
            'lr': lr_ratio,
            'name': 'cross_attn_norm',
        })
    elif hasattr(model, 'ratio_encoder') and model.ratio_encoder is not None:
        param_groups.append({
            'params': (
                list(model.ratio_encoder.parameters()) +
                list(model.ratio_projection.parameters())
            ),
            'lr': lr_ratio,
            'name': 'ratio_encoder',
        })

    return AdamW(param_groups, weight_decay=1e-4)


# ---------------------------------------------------------------------------
# Preflight contracts for Gate 4.2
# ---------------------------------------------------------------------------

# Approximate trainable param ranges per descriptor and freeze policy
# Run B (~39.7M base), Run D adds ~25M for layers 0-1
GATE42_PARAM_RANGES = {
    'run-b': {
        'd0': (39_000_000, 40_500_000),
        'd1': (39_050_000, 40_600_000),
        'd2': (39_100_000, 40_700_000),
        'd3': (39_050_000, 40_600_000),
        'd4': (39_000_000, 40_800_000),
        'a4': (39_000_000, 42_000_000),
        'a7': (39_000_000, 42_000_000),
        'd4a4': (39_000_000, 42_500_000),
        'd4a7': (39_000_000, 42_500_000),
        'a4x': (39_000_000, 46_000_000),   # +~4.2M (cross-attn + kv_proj + norm)
        'a7x': (39_000_000, 46_000_000),
        'd4x': (39_000_000, 42_000_000),   # +~1.05M (cross-attn d=512 + kv_proj + norm)
        'd4a4cm': (39_000_000, 42_500_000),  # +~1.3M (audio_proj(1028→1024) + midi_proj(520→512))
        'a4r': (39_000_000, 46_000_000),   # +~4.2M (q_proj(8→1024) + pos_emb + cross-attn + norm)
    },
    'run-d': {
        'd0': (64_000_000, 66_000_000),
        'd1': (64_050_000, 66_100_000),
        'd2': (64_100_000, 66_200_000),
        'd3': (64_050_000, 66_100_000),
        'd4': (64_000_000, 66_300_000),
        'a4': (64_000_000, 68_000_000),    # +~1M (Linear(1032,1024)+LN)
        'a7': (64_000_000, 68_000_000),    # +~1M (Linear(1036,1024)+LN)
        'd4a4': (64_000_000, 68_500_000),  # audio_proj + interval_proj
        'd4a7': (64_000_000, 68_500_000),
        'a4x': (64_000_000, 72_000_000),   # +~4.2M (cross-attn + kv_proj + norm)
        'a7x': (64_000_000, 72_000_000),
        'd4x': (64_000_000, 67_500_000),   # +~1.05M (cross-attn d=512 + kv_proj + norm)
        'd4a4cm': (64_000_000, 68_500_000),  # +~1.3M (audio_proj(1028→1024) + midi_proj(520→512))
        'a4r': (64_000_000, 72_000_000),   # +~4.2M (q_proj(8→1024) + pos_emb + cross-attn + norm)
    },
}


def get_gate42_preflight_contract(descriptor: str, freeze_policy: str = 'run-b') -> dict:
    """Return preflight contract for Gate 4.2 descriptor."""
    # Gate42Model uses base_model.* prefix for CrossModalModel params
    frozen_prefixes = [
        'base_model.audio_encoder.feature_extractor.',
        'base_model.audio_encoder.pos_embedding',
    ]

    trainable_prefixes = [
        'base_model.midi_encoder.',
        'base_model.audio_projection.',
        'base_model.midi_projection.',
    ]

    if freeze_policy == 'run-b':
        frozen_prefixes.extend([
            'base_model.audio_encoder.transformer.layers.0.',
            'base_model.audio_encoder.transformer.layers.1.',
        ])
        trainable_prefixes.extend([
            'base_model.audio_encoder.transformer.layers.2.',
            'base_model.audio_encoder.transformer.layers.3.',
        ])
    elif freeze_policy == 'run-d':
        # All transformer layers trainable
        trainable_prefixes.extend([
            'base_model.audio_encoder.transformer.layers.0.',
            'base_model.audio_encoder.transformer.layers.1.',
            'base_model.audio_encoder.transformer.layers.2.',
            'base_model.audio_encoder.transformer.layers.3.',
        ])

    if descriptor in ('d1', 'd2', 'd3'):
        trainable_prefixes.extend(['ratio_encoder.', 'ratio_projection.'])
    elif descriptor == 'd4':
        trainable_prefixes.append('interval_projection.')
    elif descriptor in ('a4', 'a7'):
        trainable_prefixes.append('audio_descriptor_projection.')
    elif descriptor in ('d4a4', 'd4a7'):
        trainable_prefixes.append('audio_descriptor_projection.')
        trainable_prefixes.append('interval_projection.')
    elif descriptor in ('a4x', 'a7x'):
        trainable_prefixes.extend([
            'descriptor_kv_proj.',
            'cross_attention.',
            'cross_attn_norm.',
        ])
    elif descriptor == 'd4x':
        trainable_prefixes.extend([
            'interval_kv_proj.',
            'cross_attention.',
            'cross_attn_norm.',
        ])
    elif descriptor == 'd4a4cm':
        trainable_prefixes.extend([
            'cross_modal_audio_projection.',
            'cross_modal_midi_projection.',
        ])
    elif descriptor == 'a4r':
        trainable_prefixes.extend([
            'descriptor_q_proj.',
            'desc_pos_embedding',
            'cross_attention.',
            'cross_attn_norm.',
        ])

    return {
        'frozen_prefixes': frozen_prefixes,
        'trainable_prefixes': trainable_prefixes,
    }


# ---------------------------------------------------------------------------
# Checkpoint saving + validation (F1 fix)
# ---------------------------------------------------------------------------

def _validate_base_checkpoint(path: Path):
    """Verify _base.pt is a clean CrossModalModel state dict (F1 fix).

    Checks:
    1. No wrapper prefixes (base_model.*)
    2. Has audio_encoder and midi_encoder keys
    3. Loads with strict=True into a fresh CrossModalModel
    """
    ckpt = torch.load(path, map_location='cpu')
    keys = list(ckpt['model_state_dict'].keys())

    # Must NOT have 'base_model.' prefix
    bad = [k for k in keys if k.startswith('base_model.')]
    if bad:
        raise RuntimeError(f"_base.pt has wrapper prefixes: {bad[:3]}...")

    # Must have audio_encoder and midi_encoder
    has_audio = any(k.startswith('audio_encoder.') for k in keys)
    has_midi = any(k.startswith('midi_encoder.') for k in keys)
    if not has_audio or not has_midi:
        raise RuntimeError(f"_base.pt missing expected encoder keys")

    # Trial strict load to catch incomplete state dicts
    test_model = CrossModalModel(audio_encoder='lite', use_dann=False)
    test_model.load_state_dict(ckpt['model_state_dict'], strict=True)

    logger.info(f"  _base.pt validated: {len(keys)} keys, strict load OK")


def save_gate42_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LinearWarmupCosineScheduler,
    epoch: int,
    best_S: float,
    descriptor: str,
    arch_config: dict,
    output_dir: Path,
    filename: str,
):
    """
    Save dual checkpoint:
    - Full: model + optimizer + scheduler (for resume)
    - Base: CrossModalModel pure state dict (for eval with evaluate_structured_pool.py)
    """
    path = output_dir / filename

    # Full checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'arch_config': {
            **arch_config,
            'checkpoint_type': 'full',
            'eval_compatible': descriptor not in ('d4', 'a4', 'a7', 'd4a4', 'd4a7', 'a4x', 'a7x', 'd4x', 'd4a4cm', 'a4r'),
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_S': best_S,
    }, path)

    # Base checkpoint: CrossModalModel pure state dict
    if descriptor in ('d4', 'a4', 'a7', 'd4a4', 'd4a7', 'a4x', 'a7x', 'd4x', 'd4a4cm', 'a4r'):
        # Augmented pipelines: not eval-compatible with evaluate_structured_pool.py
        # Save archive_base for reference only
        archive_path = path.with_name(path.stem + '_archive_base_not_for_eval.pt')
        torch.save({
            'model_state_dict': model.base_model.state_dict(),
            'arch_config': {
                **arch_config,
                'checkpoint_type': 'archive_base',
                'eval_compatible': False,
            },
            'epoch': epoch,
        }, archive_path)
    else:
        # D0-D3: base_model is a clean CrossModalModel
        base_path = path.with_name(path.stem + '_base.pt')
        torch.save({
            'model_state_dict': model.base_model.state_dict(),
            'arch_config': {
                **arch_config,
                'checkpoint_type': 'base',
                'eval_compatible': True,
            },
            'epoch': epoch,
        }, base_path)
        _validate_base_checkpoint(base_path)


# ---------------------------------------------------------------------------
# Structured pool evaluation (reused from bloqueA_training.py)
# ---------------------------------------------------------------------------

def run_structured_eval(
    model: nn.Module,
    maestro_dir: Path,
    device: torch.device,
    seed: int = 42,
    pool_size: int = 256,
    n_queries: int = 500,
    embed_batch_size: int = 16,
) -> dict:
    """Run structured pool evaluation."""
    val_dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir, segment_len=4.0, hop=1.0, split='validation',
    )
    index = build_segment_index(val_dataset)
    logger.info(f"Eval: {len(val_dataset)} segments, {len(index['by_piece'])} pieces")

    torch.cuda.empty_cache()
    model.eval()

    t0 = time.time()
    logger.info("  [eval] Extracting embeddings (batch_size=%d)...", embed_batch_size)
    with torch.no_grad():
        audio_embs, midi_embs = extract_all_embeddings(
            model, val_dataset, device, batch_size=embed_batch_size,
        )
    logger.info("  [eval] Embeddings extracted in %.1fs", time.time() - t0)

    config = PoolConfig(
        pool_size=pool_size, n_hard_negatives=64,
        n_semi_hard_negatives=32, n_queries=n_queries,
    )

    t0 = time.time()
    logger.info("  [eval] Running A2M retrieval...")
    a2m = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, val_dataset, index, config,
        direction='a2m', seed=seed,
    )
    logger.info("  [eval] A2M done in %.1fs — R@10=%.1f%%", time.time() - t0, a2m['mean_recall@10'] * 100)

    t0 = time.time()
    logger.info("  [eval] Running M2A retrieval...")
    m2a = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, val_dataset, index, config,
        direction='m2a', seed=seed,
    )
    logger.info("  [eval] M2A done in %.1fs — R@10=%.1f%%", time.time() - t0, m2a['mean_recall@10'] * 100)

    t0 = time.time()
    logger.info("  [eval] Running hard negative analysis...")
    hard_neg = analyze_hard_negatives_fast(
        audio_embs, midi_embs, val_dataset, index, n_samples=500, seed=seed,
    )
    logger.info("  [eval] Hard neg done in %.1fs — acc=%.1f%%", time.time() - t0, hard_neg['accuracy_vs_same_piece'] * 100)

    a2m_r10 = a2m['mean_recall@10']
    m2a_r10 = m2a['mean_recall@10']
    S = min(a2m_r10, m2a_r10)

    results = {
        'a2m': a2m, 'm2a': m2a,
        'hard_negative_analysis': hard_neg,
        'gate_metrics': {
            'S': S, 'hard_neg': hard_neg['accuracy_vs_same_piece'],
            'a2m_r10': a2m_r10, 'm2a_r10': m2a_r10,
        },
        'pool_config': {
            'pool_size': config.pool_size, 'n_queries': config.n_queries,
            'seed': seed, 'n_hard_negatives': config.n_hard_negatives,
            'n_semi_hard_negatives': config.n_semi_hard_negatives,
        },
    }

    logger.info("=" * 60)
    logger.info("STRUCTURED POOL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  A2M R@10: {a2m_r10:.1%}  |  M2A R@10: {m2a_r10:.1%}")
    logger.info(f"  S = min(A2M, M2A) = {S:.1%}")
    logger.info(f"  Hard neg acc: {hard_neg['accuracy_vs_same_piece']:.1%}")
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Quick inline eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_val_eval(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    """Fast validation eval using a subset of val data."""
    model.eval()
    audio_embs, midi_embs = [], []

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="  [quick_val]", total=max_batches)):
        if batch_idx >= max_batches:
            break
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        z_audio, z_midi = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
        audio_embs.append(z_audio.cpu())
        midi_embs.append(z_midi.cpu())

    audio_embs = torch.cat(audio_embs, dim=0)
    midi_embs = torch.cat(midi_embs, dim=0)

    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)
    sim = audio_norm @ midi_norm.T
    N = sim.size(0)

    _, a2m_indices = sim.sort(dim=1, descending=True)
    gt = torch.arange(N)
    a2m_ranks = (a2m_indices == gt.unsqueeze(1)).nonzero()[:, 1].float()
    a2m_r10 = (a2m_ranks < 10).float().mean().item()

    _, m2a_indices = sim.T.sort(dim=1, descending=True)
    m2a_ranks = (m2a_indices == gt.unsqueeze(1)).nonzero()[:, 1].float()
    m2a_r10 = (m2a_ranks < 10).float().mean().item()

    gap = sim.diag().mean().item() - sim[~torch.eye(N, dtype=bool)].mean().item()

    return {
        'val_a2m_r10': a2m_r10,
        'val_m2a_r10': m2a_r10,
        'val_S': min(a2m_r10, m2a_r10),
        'val_gap': gap,
        'val_n_segments': N,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop_gate42(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LinearWarmupCosineScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    descriptor: str,
    arch_config: dict,
    output_dir: Path,
    maestro_dir: Path,
    device: torch.device,
    epochs: int = 5,
    max_batches_per_epoch: int = 1000,
    max_val_batches: int = 846,
    seed: int = 42,
    embed_batch_size: int = 16,
    start_epoch: int = 1,
    initial_best_S: float = 0.0,
    initial_best_epoch: int = 0,
    skip_structured_eval: bool = False,
    structured_eval_epochs: Optional[List[int]] = None,
) -> dict:
    """Training loop for Gate 4.2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / 'eval_per_epoch'
    eval_dir.mkdir(exist_ok=True)

    sentinel = DriftSentinel(model)
    best_S = initial_best_S
    best_epoch = initial_best_epoch
    history = []

    if start_epoch > 1:
        logger.info(f"Resuming Gate 4.2 training: descriptor={descriptor}, "
                     f"epochs {start_epoch}-{epochs} (best so far: S={best_S:.1%} @ epoch {best_epoch})")
    else:
        logger.info(f"Starting Gate 4.2 training: descriptor={descriptor}, epochs={epochs}")
    logger.info(f"  Batches/epoch: {max_batches_per_epoch}, Val batches: {max_val_batches}")
    total_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        total_loss = 0.0
        total_aux = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", total=max_batches_per_epoch)
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches_per_epoch:
                break

            audio = batch['audio'].to(device, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
            midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
            midi_duration = batch['midi_duration'].to(device, non_blocking=True)
            midi_mask = batch['midi_mask'].to(device, non_blocking=True)

            # D3 needs onset and duration_sec
            midi_onset = None
            midi_duration_sec = None
            if batch.get('midi_onset') is not None:
                midi_onset = batch['midi_onset'].to(device, non_blocking=True)
            if batch.get('midi_duration_sec') is not None:
                midi_duration_sec = batch['midi_duration_sec'].to(device, non_blocking=True)

            optimizer.zero_grad()

            loss, metrics = model.compute_total_loss(
                audio, midi_pitch, midi_velocity, midi_duration, midi_mask,
                midi_onset=midi_onset, midi_duration_sec=midi_duration_sec,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += metrics['vicreg_loss']
            total_aux += metrics.get('ratio_aux_loss', 0.0)
            n_batches += 1

            # Update postfix every batch for live visibility
            lrs = scheduler.get_last_lr()
            postfix = {
                'loss': f"{metrics['vicreg_loss']:.4f}",
                'avg': f"{total_loss / n_batches:.4f}",
                'std_z1': f"{metrics['std_z1']:.3f}",
                'lr0': f"{lrs[0]:.1e}",
            }
            if metrics.get('ratio_aux_loss', 0) > 0:
                postfix['aux'] = f"{metrics['ratio_aux_loss']:.4f}"
            pbar.set_postfix(postfix)

        avg_loss = total_loss / max(n_batches, 1)
        avg_aux = total_aux / max(n_batches, 1)
        train_time = (time.time() - epoch_start) / 60

        # --- DriftSentinel after epoch 1 ---
        if epoch == 1:
            sentinel.check(model)

        # --- Save checkpoint FIRST (resilience) ---
        save_gate42_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_S=best_S, descriptor=descriptor,
            arch_config=arch_config, output_dir=output_dir,
            filename=f'checkpoint_epoch{epoch}.pt',
        )
        logger.info(f"  Checkpoint saved: checkpoint_epoch{epoch}.pt")

        # --- Quick val ---
        t_qv = time.time()
        logger.info(f"  [quick_val] Starting ({max_val_batches} batches)...")
        quick_metrics = quick_val_eval(model, val_loader, device, max_batches=max_val_batches)
        logger.info(f"  [quick_val] Done in {time.time() - t_qv:.1f}s")

        # --- Canonical structured pool eval (optional) ---
        do_structured = not skip_structured_eval
        if structured_eval_epochs is not None:
            do_structured = epoch in structured_eval_epochs
        if do_structured:
            logger.info(f"  [structured_eval] Starting canonical eval for epoch {epoch}...")
            structured_results = run_structured_eval(
                model=model, maestro_dir=maestro_dir, device=device,
                seed=seed, embed_batch_size=embed_batch_size,
            )

            structured_S = structured_results['gate_metrics']['S']

            # Save eval JSON
            epoch_eval = {'epoch': epoch, 'descriptor': descriptor, **structured_results}
            with open(eval_dir / f'eval_epoch{epoch}.json', 'w') as f:
                json.dump(epoch_eval, f, indent=2)

            # Track best by structured S
            if structured_S > best_S:
                best_S = structured_S
                best_epoch = epoch
                save_gate42_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, best_S=best_S, descriptor=descriptor,
                    arch_config=arch_config, output_dir=output_dir,
                    filename='best_model.pt',
                )
                logger.info(f"  >>> New best model: epoch {epoch}, structured S={structured_S:.1%}")
        else:
            structured_S = 0.0
            structured_results = None

        epoch_time = (time.time() - epoch_start) / 60

        epoch_record = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_aux_loss': avg_aux,
            'train_time_min': round(train_time, 1),
            'epoch_time_min': round(epoch_time, 1),
            **quick_metrics,
        }
        if structured_results:
            epoch_record.update({
                'structured_a2m_r10': structured_results['gate_metrics']['a2m_r10'],
                'structured_m2a_r10': structured_results['gate_metrics']['m2a_r10'],
                'structured_S': structured_S,
                'structured_hard_neg': structured_results['gate_metrics']['hard_neg'],
            })
        history.append(epoch_record)

        if structured_results:
            logger.info(
                f"Epoch {epoch}/{epochs} ({epoch_time:.1f}min): "
                f"loss={avg_loss:.4f}, aux={avg_aux:.4f}, "
                f"quick[A2M={quick_metrics['val_a2m_r10']:.1%} M2A={quick_metrics['val_m2a_r10']:.1%}], "
                f"CANONICAL[A2M={structured_results['gate_metrics']['a2m_r10']:.1%} "
                f"M2A={structured_results['gate_metrics']['m2a_r10']:.1%} "
                f"S={structured_S:.1%} hard_neg={structured_results['gate_metrics']['hard_neg']:.1%}]"
            )
        else:
            logger.info(
                f"Epoch {epoch}/{epochs} ({epoch_time:.1f}min): "
                f"loss={avg_loss:.4f}, aux={avg_aux:.4f}, "
                f"quick[A2M={quick_metrics['val_a2m_r10']:.1%} M2A={quick_metrics['val_m2a_r10']:.1%}]"
            )

    total_time = (time.time() - total_start) / 60

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    if skip_structured_eval:
        logger.info(
            f"Training complete: {total_time:.1f} minutes, {epochs} epochs. "
            f"Structured eval was SKIPPED — run eval on checkpoints post-hoc."
        )
    else:
        logger.info(
            f"Training complete: {total_time:.1f} minutes, "
            f"best structured S={best_S:.1%} (epoch {best_epoch})"
        )

    return {
        'history': history,
        'best_S': best_S,
        'best_epoch': best_epoch,
        'training_time_minutes': round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# Mode: train
# ---------------------------------------------------------------------------

def run_train(args):
    """Train a Gate 4.2 descriptor variant."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    start_epoch = 1
    initial_best_S = 0.0
    initial_best_epoch = 0

    if args.resume:
        # --- RESUME from Gate 4.2 checkpoint ---
        logger.info("=" * 60)
        logger.info("GATE 4.2 — RESUME TRAINING")
        logger.info("=" * 60)

        resume_ckpt = torch.load(args.resume, map_location=device)
        resume_arch = resume_ckpt.get('arch_config', {})

        if resume_arch.get('checkpoint_type') == 'base':
            raise RuntimeError(
                "Cannot resume from a _base.pt checkpoint (no optimizer state). "
                "Use the full checkpoint (e.g., checkpoint_epoch3.pt)."
            )
        if 'optimizer_state_dict' not in resume_ckpt:
            raise RuntimeError(
                f"Checkpoint {args.resume} has no optimizer_state_dict — "
                "cannot resume. Use a full checkpoint."
            )

        # Auto-detect descriptor from checkpoint
        ckpt_descriptor = resume_arch.get('descriptor')
        if args.descriptor and ckpt_descriptor and args.descriptor != ckpt_descriptor:
            raise ValueError(
                f"--descriptor={args.descriptor} conflicts with "
                f"checkpoint descriptor={ckpt_descriptor}"
            )
        descriptor = ckpt_descriptor or args.descriptor
        if not descriptor:
            raise ValueError("Cannot determine descriptor — provide --descriptor or use a Gate 4.2 checkpoint")

        freeze_policy = resume_arch.get('freeze_policy', args.freeze_policy)
        ratio_weight = resume_arch.get('ratio_weight', args.ratio_weight)
        start_epoch = resume_ckpt['epoch'] + 1
        initial_best_S = resume_ckpt.get('best_S', 0.0)
        initial_best_epoch = resume_ckpt.get('epoch', 0)

        logger.info(f"  Resuming descriptor={descriptor}, freeze_policy={freeze_policy}")
        logger.info(f"  From epoch {resume_ckpt['epoch']}, best_S={initial_best_S:.1%}")
        logger.info(f"  Will train epochs {start_epoch} to {args.epochs}")

        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint is at epoch {resume_ckpt['epoch']} but "
                f"--epochs={args.epochs}. Use --epochs > {resume_ckpt['epoch']}."
            )

        # Rebuild model architecture (same as fresh, then load weights)
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        apply_freeze_policy(base_model, policy=freeze_policy)
        model = create_gate42_model(descriptor, base_model, ratio_weight=ratio_weight)
        model.load_state_dict(resume_ckpt['model_state_dict'], strict=True)
        model = model.to(device)

        # Create optimizer (same structure) then restore state
        optimizer = create_gate42_optimizer(
            model, descriptor,
            freeze_policy=freeze_policy,
            lr_audio=args.lr_audio_unfreeze,
            lr_audio_low=args.lr_audio_low,
            lr_midi=args.lr_midi,
            lr_proj=args.lr_proj,
            lr_ratio=args.lr_ratio,
        )
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])

        arch_config = resume_arch.copy()
        arch_config['resumed_from'] = args.resume
        arch_config['resumed_epoch'] = resume_ckpt['epoch']

    else:
        # --- FRESH training (from foundation or from scratch) ---
        descriptor = args.descriptor or 'd0'

        logger.info("=" * 60)
        if args.from_scratch:
            logger.info(f"GATE 4.3 SCRATCH — DESCRIPTOR: {descriptor.upper()}")
            logger.info("Training FROM SCRATCH (MERT pretrained + random MIDI)")
        else:
            logger.info(f"GATE 4.2 — DESCRIPTOR: {descriptor.upper()}")
        logger.info("=" * 60)

        if args.from_scratch:
            # Create fresh model: MERT pretrained audio + random MIDI encoder
            base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
            logger.info("Model created from scratch (no foundation checkpoint)")
        else:
            # Load foundation
            base_model = load_foundation(args.checkpoint, device)

        # Apply freeze policy
        freeze_policy = args.freeze_policy
        apply_freeze_policy(base_model, policy=freeze_policy)

        # Create model
        model = create_gate42_model(descriptor, base_model, ratio_weight=args.ratio_weight)
        model = model.to(device)

        # Create optimizer
        optimizer = create_gate42_optimizer(
            model, descriptor,
            freeze_policy=freeze_policy,
            lr_audio=args.lr_audio_unfreeze,
            lr_audio_low=args.lr_audio_low,
            lr_midi=args.lr_midi,
            lr_proj=args.lr_proj,
            lr_ratio=args.lr_ratio,
        )

        arch_config = {
            'gate': '4.3-scratch' if args.from_scratch else '4.2',
            'descriptor': descriptor,
            'ratio_weight': args.ratio_weight,
            'foundation_checkpoint': None if args.from_scratch else args.checkpoint,
            'from_scratch': args.from_scratch,
            'freeze_policy': freeze_policy,
        }

    # Preflight
    mode_key = f'gate42_{descriptor}'
    PARAM_RANGES[mode_key] = GATE42_PARAM_RANGES[freeze_policy][descriptor]
    contract = get_gate42_preflight_contract(descriptor, freeze_policy=freeze_policy)
    validate_training_setup(model, optimizer, mode=mode_key, **contract)

    # DataLoaders
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = MaestroSegmentDataset(
        maestro_dir=args.maestro_dir, segment_len=4.0, hop=1.0, split='train',
    )
    val_dataset = MaestroSegmentDataset(
        maestro_dir=args.maestro_dir, segment_len=4.0, hop=1.0, split='validation',
    )

    dl_kwargs = _dataloader_kwargs(args.num_workers)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_segments,
        drop_last=True, generator=g, worker_init_fn=seed_worker,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_segments,
        **dl_kwargs,
    )

    # Scheduler
    total_steps = args.epochs * args.max_batches_per_epoch
    scheduler = LinearWarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps,
    )

    # Restore scheduler state if resuming
    if args.resume and 'scheduler_state_dict' in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        logger.info(f"  Scheduler restored: step_count={scheduler.step_count}, "
                     f"total_steps={scheduler.total_steps} (new)")

    # Save config
    config = {**vars(args), **arch_config}
    if args.resume:
        config['resumed_from'] = args.resume
        config['resumed_at_epoch'] = start_epoch - 1
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Train
    train_results = train_loop_gate42(
        model=model, optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        descriptor=descriptor, arch_config=arch_config,
        output_dir=output_dir, maestro_dir=Path(args.maestro_dir),
        device=device, epochs=args.epochs,
        max_batches_per_epoch=args.max_batches_per_epoch,
        max_val_batches=args.max_val_batches,
        seed=args.seed, embed_batch_size=args.embed_batch_size,
        start_epoch=start_epoch,
        initial_best_S=initial_best_S,
        initial_best_epoch=initial_best_epoch,
        skip_structured_eval=args.skip_structured_eval,
        structured_eval_epochs=getattr(args, 'structured_eval_epochs', None),
    )

    # Final results
    best_ep = train_results['best_epoch']
    best_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{best_ep}.json'
    last_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{args.epochs}.json'

    best_eval = json.loads(best_eval_path.read_text()) if best_eval_path.exists() else None
    last_eval = json.loads(last_eval_path.read_text()) if last_eval_path.exists() else None

    final = {
        'gate': '4.2',
        'descriptor': descriptor,
        'training': train_results,
        'evaluation_best': best_eval,
        'evaluation_final': last_eval,
    }
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final, f, indent=2)

    return final


# ---------------------------------------------------------------------------
# Mode: evaluate
# ---------------------------------------------------------------------------

def run_evaluate(args):
    """Evaluate a Gate 4.2 checkpoint."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 60)
    logger.info("GATE 4.2 — EVALUATE")
    logger.info("=" * 60)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch_config = checkpoint.get('arch_config')

    if arch_config is None:
        raise RuntimeError("Missing arch_config in checkpoint")

    # Reject archive_base checkpoints (stripped model without augmentation layers)
    if arch_config.get('checkpoint_type') == 'archive_base':
        raise ValueError(
            f"Checkpoint '{args.checkpoint}' is an archive_base checkpoint "
            f"(descriptor={arch_config.get('descriptor')}). "
            f"archive_base checkpoints cannot be evaluated — use the full checkpoint instead."
        )

    # Read descriptor from checkpoint (F6 fix)
    ckpt_descriptor = arch_config.get('descriptor')
    if args.descriptor and ckpt_descriptor and args.descriptor != ckpt_descriptor:
        raise ValueError(
            f"--descriptor={args.descriptor} conflicts with "
            f"checkpoint descriptor={ckpt_descriptor}"
        )
    descriptor = ckpt_descriptor or args.descriptor
    if not descriptor:
        raise ValueError("Cannot determine descriptor — provide --descriptor or use a Gate 4.2 checkpoint")

    logger.info(f"Descriptor: {descriptor}, arch_config: {arch_config}")

    if descriptor == 'd4':
        # Reconstruct Gate42InputAugModel
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42InputAugModel(base_model, interval_dim=4)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'a4':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioAugModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'a7':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioAugModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor in ('d4a4', 'd4a7'):
        ad_type = 'a4' if descriptor == 'd4a4' else 'a7'
        ad_dim = 8 if ad_type == 'a4' else 12
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42DualAugModel(base_model, audio_descriptor_type=ad_type, audio_descriptor_dim=ad_dim)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'a4x':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'a7x':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioCrossAttModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'd4x':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42MidiCrossAttModel(base_model, interval_dim=4)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'd4a4cm':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42DualCrossModalModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'a4r':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif arch_config.get('checkpoint_type') == 'base':
        # Base checkpoint: load as CrossModalModel directly
        model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        # Full checkpoint: reconstruct Gate42Model
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = create_gate42_model(descriptor, base_model, ratio_weight=0.0)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    model = model.to(device)
    model.eval()

    # Audio-aug models use more VRAM from intermediate STFT
    eval_bs = getattr(args, 'embed_batch_size', 64) or 64
    if descriptor in ('a4x', 'a7x', 'a4r') and eval_bs > 16:
        eval_bs = 16  # cross-attn matrix heavier than concat
    elif descriptor in ('a4', 'a7', 'd4a4', 'd4a7', 'd4a4cm') and eval_bs > 32:
        eval_bs = 32

    results = run_structured_eval(
        model=model, maestro_dir=Path(args.maestro_dir),
        device=device, seed=args.seed,
        embed_batch_size=eval_bs,
    )

    results['gate'] = '4.2'
    results['descriptor'] = descriptor
    results['checkpoint'] = args.checkpoint
    results['arch_config'] = arch_config
    results['epoch'] = checkpoint.get('epoch', '?')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gate 4.2 Training: Ratio Descriptor Screening"
    )
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'evaluate'],
        help='Execution mode',
    )
    parser.add_argument(
        '--descriptor', type=str, default=None,
        choices=['d0', 'd1', 'd2', 'd3', 'd4', 'a4', 'a7', 'd4a4', 'd4a7', 'a4x', 'a7x', 'd4x', 'd4a4cm', 'a4r'],
        help='Descriptor variant (required for train, auto-detected for evaluate)',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Foundation checkpoint (train) or Gate 4.2 checkpoint (evaluate)',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume training from a Gate 4.2 full checkpoint (not _base). '
             'When used, --checkpoint is ignored.',
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory (train) or output file (evaluate)',
    )
    parser.add_argument(
        '--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0',
    )

    # Training params
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-batches-per-epoch', type=int, default=1000)
    parser.add_argument('--max-val-batches', type=int, default=846)
    parser.add_argument('--embed-batch-size', type=int, default=16)
    parser.add_argument('--ratio-weight', type=float, default=0.1)

    # Freeze policy
    parser.add_argument(
        '--freeze-policy', type=str, default='run-b',
        choices=['run-b', 'run-d'],
        help='Freeze policy: run-b (layers 0-1 frozen) or run-d (all layers, split-LR)',
    )

    # Learning rates
    parser.add_argument('--lr-audio-unfreeze', type=float, default=1e-5,
                        help='LR for audio layers 2-3')
    parser.add_argument('--lr-audio-low', type=float, default=5e-6,
                        help='LR for audio layers 0-1 (only used with --freeze-policy run-d)')
    parser.add_argument('--lr-midi', type=float, default=5e-5)
    parser.add_argument('--lr-proj', type=float, default=1e-4)
    parser.add_argument('--lr-ratio', type=float, default=5e-4)
    parser.add_argument('--warmup-steps', type=int, default=200)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--from-scratch', action='store_true',
        help='Train from scratch (MERT pretrained + random MIDI) instead of foundation checkpoint. '
             'When set, --checkpoint is not required.',
    )
    parser.add_argument(
        '--skip-structured-eval', action='store_true',
        help='Skip structured pool eval (embedding extraction + retrieval) after each epoch. '
             'Only quick_val is run. Use for long runs; eval checkpoints post-hoc.',
    )
    parser.add_argument(
        '--structured-eval-epochs', type=int, nargs='+', default=None,
        help='Run structured eval ONLY at these epochs (e.g. --structured-eval-epochs 10 15 20 25 28 29 30). '
             'Overrides --skip-structured-eval for the specified epochs.',
    )

    args = parser.parse_args()

    if args.mode == 'train':
        if args.resume is None and args.checkpoint is None and not args.from_scratch:
            parser.error("--checkpoint is required for train mode (unless --resume or --from-scratch is used)")
        if args.descriptor is None and args.resume is None:
            args.descriptor = 'd0'
            logger.info("No --descriptor specified for train mode, defaulting to d0 (control)")
        run_train(args)
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            parser.error("--checkpoint is required for evaluate mode")
        run_evaluate(args)


if __name__ == '__main__':
    main()
