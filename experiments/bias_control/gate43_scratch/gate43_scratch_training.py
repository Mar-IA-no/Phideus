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
from src.bias_control.encoders.midi_encoder import SinusoidalPositionalEncoding
from src.bias_control.encoders.projection import ProjectionHead

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
    """Linear warmup, optional hold, then cosine annealing.

    Modes:
        Standard:    warmup → [hold] → cosine → 0
        Cosine-tail: warmup → cosine(ref_steps) → linear tail (floor → tail_end)

    Standard mode (hold_fraction=0.0 is default):
        1. Warmup:  steps [1, warmup_steps] — linear ramp 0 → 1
        2. Hold:    steps (warmup_steps, hold_end] — constant at 1.0
        3. Cosine:  steps (hold_end, total_steps] — cosine decay 1 → 0

    Cosine-tail mode (cosine_ref_steps > 0):
        Replicates the LR curve of a shorter run (e.g. 30ep) and adds a
        linear tail once the cosine reaches lr_floor.
        1. Warmup:  steps [1, warmup_steps] — linear ramp 0 → 1
        2. Cosine:  steps (warmup_steps, tail_start] — cosine with ref_steps
        3. Tail:    steps (tail_start, total_steps] — linear floor → tail_end
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 hold_fraction: float = 0.0,
                 cosine_ref_steps: int = 0,
                 lr_floor: float = 0.0,
                 lr_tail_end: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold_fraction = hold_fraction
        self.cosine_ref_steps = cosine_ref_steps
        self.lr_floor = lr_floor
        self.lr_tail_end = lr_tail_end

        # Hold mode (standard)
        self.hold_end = warmup_steps + int(
            hold_fraction * (total_steps - warmup_steps)
        )

        # Cosine-tail mode: find step where cosine reaches lr_floor
        self.tail_start = 0
        if cosine_ref_steps > 0 and lr_floor > 0:
            # 0.5*(1+cos(π*p)) = lr_floor  →  p = arccos(2*lr_floor - 1)/π
            progress_at_floor = math.acos(2 * lr_floor - 1) / math.pi
            self.tail_start = warmup_steps + int(
                progress_at_floor * (cosine_ref_steps - warmup_steps)
            )

        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        elif self.cosine_ref_steps > 0 and self.lr_floor > 0:
            # --- Cosine-tail mode ---
            if self.step_count <= self.tail_start:
                progress = (self.step_count - self.warmup_steps) / max(
                    1, self.cosine_ref_steps - self.warmup_steps
                )
                scale = 0.5 * (1 + math.cos(math.pi * progress))
            else:
                tail_progress = (self.step_count - self.tail_start) / max(
                    1, self.total_steps - self.tail_start
                )
                tail_progress = min(tail_progress, 1.0)
                scale = self.lr_floor + (self.lr_tail_end - self.lr_floor) * tail_progress
        elif self.step_count <= self.hold_end:
            scale = 1.0
        else:
            progress = (self.step_count - self.hold_end) / max(
                1, self.total_steps - self.hold_end
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_last_lr(self) -> List[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    @property
    def lr_mult(self) -> float:
        """Current LR multiplier (0..1)."""
        if not self.base_lrs or self.base_lrs[0] == 0:
            return 0.0
        return self.optimizer.param_groups[0]['lr'] / self.base_lrs[0]

    def state_dict(self):
        return {
            'step_count': self.step_count, 'base_lrs': self.base_lrs,
            'hold_fraction': self.hold_fraction, 'hold_end': self.hold_end,
            'cosine_ref_steps': self.cosine_ref_steps,
            'lr_floor': self.lr_floor, 'lr_tail_end': self.lr_tail_end,
            'tail_start': self.tail_start,
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']
        if 'hold_fraction' in state_dict:
            self.hold_fraction = state_dict['hold_fraction']
            self.hold_end = state_dict['hold_end']
        if 'cosine_ref_steps' in state_dict:
            self.cosine_ref_steps = state_dict['cosine_ref_steps']
            self.lr_floor = state_dict['lr_floor']
            self.lr_tail_end = state_dict['lr_tail_end']
            self.tail_start = state_dict['tail_start']


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
    compute_audio_descriptor_a10a,
    compute_audio_descriptor_a10b,
    compute_audio_descriptor_a10c,
    compute_audio_descriptor_a10d,
    compute_audio_descriptor_a10e,
)
from src.bias_control.encoders.projection import ConditionedProjectionHead
from src.bias_control.encoders.speech_egg_encoder_attn_bias import AttentionBiasComputer


_DESCRIPTOR_DISPATCH = {
    'a4': compute_audio_descriptor_a4,
    'a7': compute_audio_descriptor_a7,
    'a8': compute_audio_descriptor_a8,
    'a9': compute_audio_descriptor_a9,
    'a10a': compute_audio_descriptor_a10a,
    'a10b': compute_audio_descriptor_a10b,
    'a10c': compute_audio_descriptor_a10c,
    'a10d': compute_audio_descriptor_a10d,
    'a10e': compute_audio_descriptor_a10e,
}


def _compute_descriptor_any(audio, descriptor_type, target_length=None):
    """Compute any audio descriptor by type. Shared by all mechanisms."""
    fn = _DESCRIPTOR_DISPATCH.get(descriptor_type)
    if fn is None:
        raise ValueError(f"Unknown audio descriptor type: {descriptor_type}")
    return fn(audio, target_length=target_length)


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
        desc = _compute_descriptor_any(audio, descriptor_type, target_length=T_prime)

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
        desc = _compute_descriptor_any(audio, descriptor_type, target_length=None)

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
# Gate 10: PCA — FiLM-conditioned audio projection
# ---------------------------------------------------------------------------

class Gate42AudioPCAModel(nn.Module):
    """Audio-side PCA: standard encoding + FiLM-conditioned audio projection.

    Uses encode_audio(return_projected=False) for clean features [B, 1024],
    then applies its own ConditionedProjectionHead.

    base_model.audio_projection is UNTOUCHED (frozen, not used in forward).
    This preserves the base_model checkpoint contract: model.base_model.state_dict()
    matches CrossModalModel exactly.
    """
    def __init__(self, base_model, audio_descriptor_type, audio_descriptor_dim):
        super().__init__()
        self.base_model = base_model
        self.audio_descriptor_type = audio_descriptor_type
        self.audio_descriptor_dim = audio_descriptor_dim

        self.cond_audio_projection = ConditionedProjectionHead.from_projection_head(
            base_model.audio_projection, cond_dim=audio_descriptor_dim,
        )

    def forward(self, audio, midi_pitch, midi_velocity, midi_duration, midi_mask=None):
        # Segment-level descriptor for FiLM conditioning
        with torch.no_grad():
            desc = _compute_descriptor_any(audio, self.audio_descriptor_type)  # [B, T, K]
            cond = desc.mean(dim=1)  # [B, K]

        # Audio: standard encoding WITHOUT projection
        audio_features = self.base_model.encode_audio(audio, return_projected=False)  # [B, 1024]
        # Apply conditioned projection (NOT base_model.audio_projection)
        audio_emb = self.cond_audio_projection(audio_features, cond=cond)  # [B, 256]

        # MIDI: standard (includes midi_projection)
        midi_emb = self.base_model.encode_midi(
            pitch=midi_pitch, velocity=midi_velocity,
            duration=midi_duration, padding_mask=midi_mask,
        )
        return audio_emb, midi_emb

    def compute_total_loss(self, audio, midi_pitch, midi_velocity, midi_duration,
                           midi_mask=None, midi_onset=None, midi_duration_sec=None):
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# Gate 10: Attention Bias — descriptor modulates Transformer self-attention
# ---------------------------------------------------------------------------

def _transformer_forward_with_bias(encoder, features, mask=None):
    """Manual transformer forward with attn_mask.
    Avoids PyTorch 2.10 fused kernel NaN bug in eval mode.
    Identical to SpeechEGGEncoderAttnBias._transformer_forward.
    """
    if mask is None:
        return encoder(features)
    x = features
    for layer in encoder.layers:
        sa_out = layer.self_attn(
            x, x, x, attn_mask=mask, need_weights=False,
        )[0]
        x = layer.norm1(x + layer.dropout1(sa_out))
        ff_out = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(x)))
        )
        x = layer.norm2(x + layer.dropout2(ff_out))
    if encoder.norm is not None:
        x = encoder.norm(x)
    return x


class Gate42AudioAttnBiasModel(nn.Module):
    """Audio-side attention bias: descriptor modulates Transformer self-attention
    via factored bilinear bias. Encoding path is standard CNN -> Transformer,
    but self-attention receives additive bias from descriptor content.

    bias[h,i,j] = scale * phi(d_i)^T W_h psi(d_j), zero-init -> identity at ep0.
    """
    def __init__(self, base_model, audio_descriptor_type, audio_descriptor_dim):
        super().__init__()
        self.base_model = base_model
        self.audio_descriptor_type = audio_descriptor_type

        self.bias_computer = AttentionBiasComputer(
            desc_dim=audio_descriptor_dim,
            n_heads=8,
            d_bias=16,
        )

    def forward(self, audio, midi_pitch, midi_velocity, midi_duration, midi_mask=None):
        enc = self.base_model.audio_encoder

        # CNN features
        waveform = audio.unsqueeze(1) if audio.dim() == 2 else audio
        features = enc.feature_extractor(waveform).transpose(1, 2)  # [B, T', 1024]
        T = features.size(1)
        if T <= enc.max_pos_len:
            features = features + enc.pos_embedding[:, :T, :]
        else:
            pos = F.interpolate(
                enc.pos_embedding.transpose(1, 2), size=T,
                mode='linear', align_corners=False,
            ).transpose(1, 2)
            features = features + pos

        # Descriptor interpolated to CNN output resolution
        with torch.no_grad():
            desc = _compute_descriptor_any(audio, self.audio_descriptor_type, target_length=T)

        # Compute bias and run Transformer with manual forward
        bias = self.bias_computer(desc)  # [B*8, T, T]
        encoded = _transformer_forward_with_bias(enc.transformer, features, mask=bias)

        embeddings = encoded.mean(dim=1)  # [B, 1024]
        audio_emb = self.base_model.audio_projection(embeddings)  # [B, 256]

        # MIDI standard
        midi_emb = self.base_model.encode_midi(
            pitch=midi_pitch, velocity=midi_velocity,
            duration=midi_duration, padding_mask=midi_mask,
        )
        return audio_emb, midi_emb

    def compute_total_loss(self, audio, midi_pitch, midi_velocity, midi_duration,
                           midi_mask=None, midi_onset=None, midi_duration_sec=None):
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# Reverse cross-attention model (Gate 4.3: D4r — MIDI reverse)
# ---------------------------------------------------------------------------

def _encode_midi_with_reverse_cross_attention(
    base_model: CrossModalModel,
    pitch: torch.Tensor,
    velocity: torch.Tensor,
    duration: torch.Tensor,
    mask: Optional[torch.Tensor],
    interval_q_proj: nn.Module,
    cross_attention: nn.MultiheadAttention,
    cross_attn_norm: nn.Module,
) -> torch.Tensor:
    """
    Reverse cross-attention for MIDI: intervals (Q) attend to embeddings (K/V).

    Same sequence length for Q and K/V (both N tokens). The semantic difference
    is that intervals organize the representation instead of being consulted.

    Pipeline:
      event_emb → +CLS → +pos_enc → (K/V)
      intervals → q_proj → +CLS → +pos_enc → (Q)
      cross_attn(Q=intervals, K/V=embeddings) → residual+LN → Transformer → pool → proj
    """
    enc = base_model.midi_encoder

    # 1. Event embedding: [B, N, 512]
    x = enc.event_embedding(pitch, velocity, duration)

    # 2. CLS token + pos_encoding for embeddings (K/V)
    B = pitch.shape[0]
    cross_attn_mask = mask
    emb_mask = mask
    if enc.aggregation == "cls":
        cls_tokens = enc.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            emb_mask = torch.cat([cls_mask, mask], dim=1)
    x = enc.pos_encoding(x)  # embeddings with pos info = K/V

    # 3. Compute interval features at native resolution
    with torch.no_grad():
        interval_feats = compute_local_interval_features(pitch, cross_attn_mask)

    # 4. Project intervals to Q dimension
    interval_proj = interval_q_proj(interval_feats.detach())  # [B, N, 512]

    # 5. CLS + pos_encoding for interval Q
    if enc.aggregation == "cls":
        interval_cls = enc.cls_token.expand(B, -1, -1)
        interval_proj = torch.cat([interval_cls, interval_proj], dim=1)
    interval_proj = enc.pos_encoding(interval_proj)

    # 6. REVERSE cross-attention: intervals (Q) attend to embeddings (K/V)
    attn_output, _ = cross_attention(
        query=interval_proj,  # [B, N(+1), 512]
        key=x,                # [B, N(+1), 512]
        value=x,              # [B, N(+1), 512]
        need_weights=False,
    )
    interval_proj = cross_attn_norm(interval_proj + attn_output)

    # 7. Transformer
    if emb_mask is not None:
        interval_proj = enc.transformer(interval_proj, src_key_padding_mask=emb_mask)
    else:
        interval_proj = enc.transformer(interval_proj)

    # 8. Output norm + pool + projection
    interval_proj = enc.output_norm(interval_proj)

    if enc.aggregation == "mean":
        if emb_mask is not None:
            m = ~emb_mask.unsqueeze(-1)
            x_out = (interval_proj * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x_out = interval_proj.mean(dim=1)
    elif enc.aggregation == "cls":
        x_out = interval_proj[:, 0, :]
    elif enc.aggregation == "attention":
        weights = enc.attention_pool(interval_proj)
        if emb_mask is not None:
            weights = weights.masked_fill(emb_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(weights, dim=1)
        x_out = (interval_proj * weights).sum(dim=1)

    return base_model.midi_projection(x_out)


# ---------------------------------------------------------------------------
# Dual reverse cross-attention model (Gate 4.3: d4a4r — A4r + D4r)
# ---------------------------------------------------------------------------

class Gate42DualReverseCrossAttModel(nn.Module):
    """
    Dual reverse cross-attention: both audio (A4r) and MIDI (D4r) use reverse
    cross-attention. Descriptors/intervals organize features in BOTH encoders.

    Audio: Q=descriptor(188), K/V=CNN_features(2400) → Transformer(188 tokens)
    MIDI:  Q=intervals(N), K/V=event_emb(N) → Transformer(N tokens)
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

        # Audio reverse cross-att components (A4r)
        self.descriptor_q_proj = nn.Linear(audio_descriptor_dim, audio_embed_dim)
        self.desc_pos_embedding = nn.Parameter(
            torch.randn(1, 200, audio_embed_dim) * 0.02
        )
        self.audio_cross_attention = nn.MultiheadAttention(
            embed_dim=audio_embed_dim, num_heads=8,
            batch_first=True, dropout=0.1,
        )
        self.audio_cross_attn_norm = nn.LayerNorm(audio_embed_dim)

        # MIDI reverse cross-att components (D4r)
        self.interval_q_proj = nn.Linear(interval_dim, midi_embed_dim)
        self.midi_cross_attention = nn.MultiheadAttention(
            embed_dim=midi_embed_dim, num_heads=8,
            batch_first=True, dropout=0.1,
        )
        self.midi_cross_attn_norm = nn.LayerNorm(midi_embed_dim)

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
            self.audio_cross_attention, self.audio_cross_attn_norm,
        )
        midi_emb = _encode_midi_with_reverse_cross_attention(
            self.base_model, midi_pitch, midi_velocity, midi_duration, midi_mask,
            self.interval_q_proj, self.midi_cross_attention, self.midi_cross_attn_norm,
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
# Gate 4.3-ext: Dual Mixed Injection (D4 concat MIDI + A4r reverse audio)
# ---------------------------------------------------------------------------

class Gate42DualMixedModel(nn.Module):
    """
    Gate 4.3 extension: D4 concat (MIDI) + A4r reverse cross-att (audio).
    Combines best mechanism per modality from Gate 4.3 results:
    - MIDI: D4 concat (simple Linear(516,512)) — sufficient for d=512 encoder
    - Audio: A4r reverse cross-att (Q=A4 188 tokens, K/V=features 2400) — needed for d=1024
    """

    def __init__(self, base_model: CrossModalModel,
                 audio_descriptor_type: str = 'a4',
                 audio_descriptor_dim: int = 8,
                 interval_dim: int = 4):
        super().__init__()
        self.base_model = base_model

        # MIDI side: D4 concat (from Gate42InputAugModel)
        midi_embed_dim = base_model.midi_encoder.embed_dim  # 512
        self.interval_dim = interval_dim
        self.interval_projection = nn.Sequential(
            nn.Linear(midi_embed_dim + interval_dim, midi_embed_dim),
            nn.LayerNorm(midi_embed_dim),
        )

        # Audio side: A4r reverse cross-att (from Gate42AudioReverseCrossAttModel)
        audio_embed_dim = base_model.audio_encoder.output_dim  # 1024
        self.audio_descriptor_type = audio_descriptor_type
        self.audio_descriptor_dim = audio_descriptor_dim
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
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        # Audio: reverse cross-att (reuses existing helper)
        audio_emb = _encode_audio_with_reverse_cross_attention(
            self.base_model, audio, self.audio_descriptor_type,
            self.descriptor_q_proj, self.desc_pos_embedding,
            self.cross_attention, self.cross_attn_norm,
        )
        # MIDI: D4 concat (reuses existing helper)
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
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['ratio_aux_loss'] = 0.0
        metrics['total_loss'] = loss.item()
        return loss, metrics


# ---------------------------------------------------------------------------
# Gate 4.4 — Helper: D4 mask-aware interpolation
# ---------------------------------------------------------------------------

def interpolate_d4_masked(d4: torch.Tensor, mask: Optional[torch.Tensor],
                          target_len: int = 188) -> torch.Tensor:
    """Per-sample valid-length interpolation. No zero-smearing.

    Args:
        d4: [B, N, K] local interval features
        mask: [B, N] True=padding, or None
        target_len: target temporal length (188 = STFT frames for A4)

    Returns:
        [B, target_len, K] interpolated features
    """
    B, N, K = d4.shape

    # Guard: if no mask, simple batch interpolation
    if mask is None:
        return F.interpolate(
            d4.transpose(1, 2), size=target_len, mode='linear', align_corners=False
        ).transpose(1, 2)

    result = torch.zeros(B, target_len, K, device=d4.device)
    for b in range(B):
        valid_len = (~mask[b]).sum().item()
        if valid_len == 0:
            continue
        valid_d4 = d4[b, :valid_len, :]
        interp = F.interpolate(
            valid_d4.T.unsqueeze(0),  # [1, K, valid_len]
            size=target_len, mode='linear', align_corners=False
        )
        result[b] = interp.squeeze(0).T  # [target_len, K]
    return result


# ---------------------------------------------------------------------------
# Gate 4.4 — Helper: FiLMGenerator
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """Generates (gamma, beta) per layer from a descriptor vector."""

    def __init__(self, descriptor_dim: int, d_model: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_layers * 2 * d_model),
        )
        # Init last layer to ~0 for identity at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            descriptor: [B, descriptor_dim] (already pooled)
        Returns:
            [B, n_layers, 2, d_model] FiLM parameters
        """
        B = descriptor.size(0)
        params = self.mlp(descriptor)
        return params.view(B, self.n_layers, 2, self.d_model)


# ---------------------------------------------------------------------------
# Gate 4.4 — Helper: MoEAdapter
# ---------------------------------------------------------------------------

class MoEAdapter(nn.Module):
    """MoE-gated adapter: multiple expert FFNs with descriptor-conditioned routing."""

    def __init__(self, d_model: int, descriptor_dim: int, n_experts: int = 2,
                 bottleneck_ratio: int = 4, expert_init_std: float = 0.0,
                 router_noise_std: float = 0.0, use_top1: bool = False,
                 entropy_weight: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.router_noise_std = router_noise_std
        self.use_top1 = use_top1
        self.entropy_weight = entropy_weight
        bottleneck = d_model // bottleneck_ratio

        # Router: conditioned on [features, descriptor_summary]
        self.router = nn.Sequential(
            nn.Linear(d_model + descriptor_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_experts),
        )

        # Experts: small bottleneck FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, d_model),
            )
            for _ in range(n_experts)
        ])

        # Initialize expert outputs: non-zero breaks symmetry (v2+)
        if expert_init_std > 0:
            for expert in self.experts:
                nn.init.normal_(expert[-1].weight, mean=0.0, std=expert_init_std)
                nn.init.zeros_(expert[-1].bias)
        else:
            for expert in self.experts:
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)

    def forward(self, x: torch.Tensor, descriptor_summary: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, T, D] transformer output
            descriptor_summary: [B, desc_dim] pooled descriptor
            padding_mask: [B, T] True=padding (optional, for MIDI)

        Returns:
            (moe_out [B,T,D], load_balance_loss scalar, segment_pref_var float,
             routing_entropy scalar)
        """
        B, T, D = x.shape
        desc_expanded = descriptor_summary.unsqueeze(1).expand(B, T, -1)
        router_input = torch.cat([x, desc_expanded], dim=-1)

        logits = self.router(router_input)  # [B, T, n_experts]

        # Router noise: exploration pressure with decay (v2+)
        if self.router_noise_std > 0 and self.training:
            logits = logits + torch.randn_like(logits) * self.router_noise_std

        weights = torch.softmax(logits, dim=-1)

        # Top-1 hard gating with straight-through gradient (v4)
        if self.use_top1:
            top_idx = weights.argmax(dim=-1)  # [B, T]
            hard_weights = F.one_hot(top_idx, self.n_experts).float()
            weights = hard_weights - weights.detach() + weights  # straight-through

        expert_outs = torch.stack([e(x) for e in self.experts], dim=-1)  # [B,T,D,n_experts]
        moe_out = (expert_outs * weights.unsqueeze(2)).sum(dim=-1)  # [B,T,D]

        # Load balance — exclude padded tokens
        if padding_mask is not None:
            valid_mask = ~padding_mask  # [B, T] True=valid
            if valid_mask.any():
                valid_weights = weights[valid_mask]  # [N_valid, n_experts]
                mean_load = valid_weights.mean(dim=0)
                load_balance_loss = self.n_experts * mean_load.var()
            else:
                load_balance_loss = torch.tensor(0.0, device=x.device)
        else:
            mean_load = weights.mean(dim=(0, 1))
            load_balance_loss = self.n_experts * mean_load.var()

        # Segment-level expert preference variance
        assert self.n_experts == 2, f"segment_pref_var assumes 2 experts, got {self.n_experts}"
        expert_pref = weights.argmax(dim=-1)  # [B, T]
        if padding_mask is not None and padding_mask.any():
            valid_mask_2d = ~padding_mask
            per_sample = []
            for b in range(B):
                if valid_mask_2d[b].any():
                    per_sample.append(expert_pref[b][valid_mask_2d[b]].float().mean())
            if len(per_sample) >= 2:
                seg_pref_var = torch.stack(per_sample).var().item()
            else:
                seg_pref_var = 0.0
        else:
            per_sample_pref = expert_pref.float().mean(dim=1)  # [B]
            seg_pref_var = per_sample_pref.var().item() if B >= 2 else 0.0

        # Routing entropy per token (for entropy penalty, v3)
        if self.entropy_weight > 0:
            token_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)  # [B, T]
            if padding_mask is not None:
                valid = ~padding_mask
                if valid.any():
                    routing_entropy = token_entropy[valid].mean()
                else:
                    routing_entropy = torch.tensor(0.0, device=x.device)
            else:
                routing_entropy = token_entropy.mean()
        else:
            routing_entropy = torch.tensor(0.0, device=x.device)

        return moe_out, load_balance_loss, seg_pref_var, routing_entropy


# ---------------------------------------------------------------------------
# Gate 4.4 — Helper: FiLM encode functions
# ---------------------------------------------------------------------------

def _encode_audio_with_film(audio: torch.Tensor, base_model: CrossModalModel,
                            film_gen: FiLMGenerator) -> torch.Tensor:
    """Audio encoder with FiLM modulation applied post-layer (norm-agnostic)."""
    enc = base_model.audio_encoder

    # CNN features
    waveform = audio.unsqueeze(1) if audio.dim() == 2 else audio
    features = enc.feature_extractor(waveform).transpose(1, 2)
    T = features.size(1)
    if T <= enc.max_pos_len:
        features = features + enc.pos_embedding[:, :T, :]
    else:
        pos = F.interpolate(enc.pos_embedding.transpose(1, 2), size=T,
                            mode='linear', align_corners=False).transpose(1, 2)
        features = features + pos

    # A4 descriptor → pool → FiLM params
    with torch.no_grad():
        a4_desc = compute_audio_descriptor_a4(audio)  # [B, 188, 8]
    desc_pooled = a4_desc.mean(dim=1)  # [B, 8]
    film_params = film_gen(desc_pooled)  # [B, 4, 2, 1024]

    # Layer-by-layer with FiLM (norm-agnostic — post-layer modulation)
    for i, layer in enumerate(enc.transformer.layers):
        features = layer(features)
        gamma = film_params[:, i, 0, :].unsqueeze(1)  # [B, 1, 1024]
        beta = film_params[:, i, 1, :].unsqueeze(1)   # [B, 1, 1024]
        features = (1 + gamma) * features + beta

    # MERTEncoderLite has NO output_norm (post-norm already in each layer)
    embeddings = features.mean(dim=1)
    return base_model.audio_projection(embeddings)


def _encode_midi_with_film(pitch: torch.Tensor, vel: torch.Tensor,
                           dur: torch.Tensor, mask: Optional[torch.Tensor],
                           base_model: CrossModalModel,
                           film_gen: FiLMGenerator) -> torch.Tensor:
    """MIDI encoder with FiLM modulation applied post-layer."""
    enc = base_model.midi_encoder
    B = pitch.shape[0]

    # Compute D4 with ORIGINAL note_mask BEFORE CLS insertion
    note_mask = mask
    with torch.no_grad():
        d4 = compute_local_interval_features(pitch, note_mask)  # [B, N, 4]
    # Masked mean for descriptor
    if note_mask is not None:
        valid = ~note_mask.unsqueeze(-1)  # [B, N, 1]
        desc_pooled = (d4 * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # [B, 4]
    else:
        desc_pooled = d4.mean(dim=1)
    film_params = film_gen(desc_pooled)  # [B, 4, 2, 512]

    x = enc.event_embedding(pitch, vel, dur)

    # CLS token insertion — replicate midi_encoder.py L204-210
    transformer_mask = mask
    if enc.aggregation == "cls":
        cls_tokens = enc.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if transformer_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)

    x = enc.pos_encoding(x)

    # Layer-by-layer with FiLM + PADDING MASK
    for i, layer in enumerate(enc.transformer.layers):
        if transformer_mask is not None:
            x = layer(x, src_key_padding_mask=transformer_mask)
        else:
            x = layer(x)
        gamma = film_params[:, i, 0, :].unsqueeze(1)
        beta = film_params[:, i, 1, :].unsqueeze(1)
        x = (1 + gamma) * x + beta

    # output_norm (MIDIEncoder has separate output_norm)
    x = enc.output_norm(x)

    # Pooling — replicate MIDIEncoder aggregation logic
    if enc.aggregation == "mean":
        if transformer_mask is not None:
            m = ~transformer_mask.unsqueeze(-1)
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
    elif enc.aggregation == "cls":
        x = x[:, 0, :]
    elif enc.aggregation == "attention":
        weights = enc.attention_pool(x)
        if transformer_mask is not None:
            weights = weights.masked_fill(transformer_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(weights, dim=1)
        x = (x * weights).sum(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {enc.aggregation}")

    return base_model.midi_projection(x)


# ---------------------------------------------------------------------------
# Gate 4.4 — Third Tower Model (t3-tri, t3-anc, t3-wt)
# ---------------------------------------------------------------------------

class Gate44ThirdTowerModel(nn.Module):
    """Third Tower: ratio descriptors as independent modality.

    Uses a lightweight Transformer to encode concatenated A4+D4 descriptors
    into a ratio embedding, then applies VICReg between all three pairs.

    Args:
        base_model: CrossModalModel (audio + MIDI encoders)
        loss_mode: 'triangular' | 'anchor' | 'weighted'
        alpha_ratio: weight for ratio terms in 'weighted' mode
        use_d4a4_injection: if True, use d4a4-style concat in base encoders;
                            if False, use vanilla encoding (for t3-anc)
    """

    def __init__(self, base_model: CrossModalModel, loss_mode: str = 'triangular',
                 alpha_ratio: float = 0.3, use_d4a4_injection: bool = True):
        super().__init__()
        self.base_model = base_model
        self.loss_mode = loss_mode
        self.alpha = alpha_ratio
        self.use_d4a4_injection = use_d4a4_injection

        # Ratio tower components
        self.ratio_input_proj = nn.Sequential(
            nn.Linear(12, 256),
            nn.LayerNorm(256),
        )
        self.ratio_pos_enc = SinusoidalPositionalEncoding(d_model=256, max_len=1000, dropout=0.1)
        ratio_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.ratio_transformer = nn.TransformerEncoder(
            ratio_layer, num_layers=2, enable_nested_tensor=False,
        )
        self.ratio_norm = nn.LayerNorm(256)
        self.ratio_projection = ProjectionHead(
            input_dim=256, hidden_dim=256, output_dim=256,
            n_layers=2, use_batchnorm=True,
        )

        # d4a4 injection projections (only if using injection)
        if use_d4a4_injection:
            audio_embed_dim = base_model.audio_encoder.output_dim  # 1024
            midi_embed_dim = base_model.midi_encoder.embed_dim  # 512
            self.audio_descriptor_projection = nn.Sequential(
                nn.Linear(audio_embed_dim + 8, audio_embed_dim),  # 1032 → 1024
                nn.LayerNorm(audio_embed_dim),
            )
            self.interval_projection = nn.Sequential(
                nn.Linear(midi_embed_dim + 4, midi_embed_dim),  # 516 → 512
                nn.LayerNorm(midi_embed_dim),
            )

    def encode_ratios(self, audio: torch.Tensor, midi_pitch: torch.Tensor,
                      midi_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode ratio descriptors into embedding space.

        Returns:
            ratio_emb: [B, 256]
        """
        with torch.no_grad():
            a4 = compute_audio_descriptor_a4(audio)  # [B, 188, 8]
            d4 = compute_local_interval_features(midi_pitch, midi_mask)  # [B, N, 4]
            d4_interp = interpolate_d4_masked(d4, midi_mask, target_len=a4.size(1))  # [B, 188, 4]

        combined = torch.cat([a4.detach(), d4_interp.detach()], dim=-1)  # [B, 188, 12]
        x = self.ratio_input_proj(combined)  # [B, 188, 256]
        x = self.ratio_pos_enc(x)
        x = self.ratio_transformer(x)
        x = self.ratio_norm(x)
        x = x.mean(dim=1)  # [B, 256]
        return self.ratio_projection(x)

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (audio_emb [B,256], midi_emb [B,256])."""
        if self.use_d4a4_injection:
            audio_emb = _encode_audio_with_descriptor(
                self.base_model, audio, 'a4', self.audio_descriptor_projection,
            )
            midi_emb = _encode_midi_with_intervals(
                self.base_model, midi_pitch, midi_velocity, midi_duration,
                midi_mask, self.interval_projection, 4,
            )
        else:
            # Vanilla encoding (t3-anc): no descriptor injection
            audio_emb, midi_emb = self.base_model(
                audio, midi_pitch, midi_velocity, midi_duration, midi_mask,
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
        ratio_emb = self.encode_ratios(audio, midi_pitch, midi_mask)

        vicreg_am, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        vicreg_ar, _ = self.base_model.compute_vicreg_loss(audio_emb, ratio_emb)
        vicreg_mr, _ = self.base_model.compute_vicreg_loss(midi_emb, ratio_emb)

        if self.loss_mode == 'triangular':
            total = (vicreg_am + vicreg_ar + vicreg_mr) / 3
        elif self.loss_mode == 'anchor':
            total = (vicreg_ar + vicreg_mr) / 2
        elif self.loss_mode == 'weighted':
            total = vicreg_am + self.alpha * (vicreg_ar + vicreg_mr) / 2
        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

        # vicreg_loss = VICReg(a,m) ALWAYS for comparability
        metrics['vicreg_loss'] = vicreg_am.item()
        metrics['vicreg_ar'] = vicreg_ar.item()
        metrics['vicreg_mr'] = vicreg_mr.item()
        metrics['ratio_aux_loss'] = (vicreg_ar.item() + vicreg_mr.item()) / 2
        metrics['total_loss'] = total.item()
        return total, metrics


# ---------------------------------------------------------------------------
# Gate 4.4 — FiLM Model (film-a4, film-d4, film-dual)
# ---------------------------------------------------------------------------

class Gate44FiLMModel(nn.Module):
    """FiLM modulation: descriptors modulate Transformer layers.

    Non-FiLM side uses VANILLA encoding (no d4a4 concat) to isolate the
    FiLM mechanism as the only variable.

    Args:
        base_model: CrossModalModel
        film_mode: 'audio' (A4 modulates audio), 'midi' (D4 modulates MIDI),
                   'dual' (both)
    """

    def __init__(self, base_model: CrossModalModel, film_mode: str = 'audio'):
        super().__init__()
        self.base_model = base_model
        self.film_mode = film_mode

        if film_mode in ('audio', 'dual'):
            self.audio_film_gen = FiLMGenerator(8, 1024, 4)   # A4(8d) → audio layers(4)
        if film_mode in ('midi', 'dual'):
            self.midi_film_gen = FiLMGenerator(4, 512, 4)     # D4(4d) → MIDI layers(4)

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.film_mode in ('audio', 'dual'):
            audio_emb = _encode_audio_with_film(audio, self.base_model, self.audio_film_gen)
        else:
            audio_emb = self.base_model.encode_audio(audio)  # VANILLA

        if self.film_mode in ('midi', 'dual'):
            midi_emb = _encode_midi_with_film(
                midi_pitch, midi_velocity, midi_duration, midi_mask,
                self.base_model, self.midi_film_gen,
            )
        else:
            midi_emb = self.base_model.encode_midi(
                midi_pitch, midi_velocity, midi_duration, midi_mask,
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
# Gate 4.4 — MoE Model (moe-a4, moe-dual)
# ---------------------------------------------------------------------------

class Gate44MoEModel(nn.Module):
    """MoE adapters: descriptor-conditioned expert routing post-layer.

    Args:
        base_model: CrossModalModel
        moe_mode: 'audio' (MoE on audio) or 'dual' (MoE on both)
        n_experts: number of experts per adapter (default 2)
        lb_weight: load balance loss weight (default 0.01)
        expert_init_std: non-zero init for expert outputs (v2+, default 0.0=zero-init)
        router_noise_start: initial router noise std (v2+, default 0.0)
        router_noise_end: final router noise std after decay (v2+, default 0.0)
        use_top1: top-1 hard gating with straight-through (v4, default False)
        entropy_weight: penalty for uniform routing per-token (v3, default 0.0)
    """

    def __init__(self, base_model: CrossModalModel, moe_mode: str = 'audio',
                 n_experts: int = 2, lb_weight: float = 0.01,
                 expert_init_std: float = 0.0, router_noise_start: float = 0.0,
                 router_noise_end: float = 0.0, use_top1: bool = False,
                 entropy_weight: float = 0.0):
        super().__init__()
        self.base_model = base_model
        self.moe_mode = moe_mode
        self.lb_weight = lb_weight
        self.expert_init_std = expert_init_std
        self.router_noise_start = router_noise_start
        self.router_noise_end = router_noise_end
        self.use_top1 = use_top1
        self.entropy_weight = entropy_weight

        moe_kwargs = dict(expert_init_std=expert_init_std,
                          router_noise_std=router_noise_start,
                          use_top1=use_top1, entropy_weight=entropy_weight)

        if moe_mode in ('audio', 'dual'):
            self.audio_moe = nn.ModuleList([
                MoEAdapter(1024, 8, n_experts, **moe_kwargs)  # A4 desc_dim=8, audio d=1024
                for _ in range(4)
            ])
        if moe_mode == 'dual':
            self.midi_moe = nn.ModuleList([
                MoEAdapter(512, 4, n_experts, **moe_kwargs)  # D4 desc_dim=4, midi d=512
                for _ in range(4)
            ])

    def update_schedule(self, epoch: int, total_epochs: int):
        """Called by training loop at start of each epoch. Decays router noise."""
        if self.router_noise_start > 0 and total_epochs > 1:
            frac = (epoch - 1) / (total_epochs - 1)
            noise = self.router_noise_start + (self.router_noise_end - self.router_noise_start) * frac
            for adapter in self.audio_moe:
                adapter.router_noise_std = noise
            if hasattr(self, 'midi_moe'):
                for adapter in self.midi_moe:
                    adapter.router_noise_std = noise

    def _encode_audio_with_moe(self, audio: torch.Tensor):
        """Returns (pooled_features [B,1024], avg_load_balance, avg_seg_pref_var, avg_entropy)."""
        enc = self.base_model.audio_encoder

        waveform = audio.unsqueeze(1) if audio.dim() == 2 else audio
        features = enc.feature_extractor(waveform).transpose(1, 2)
        T = features.size(1)
        if T <= enc.max_pos_len:
            features = features + enc.pos_embedding[:, :T, :]
        else:
            pos = F.interpolate(enc.pos_embedding.transpose(1, 2), size=T,
                                mode='linear', align_corners=False).transpose(1, 2)
            features = features + pos

        with torch.no_grad():
            a4_desc = compute_audio_descriptor_a4(audio)
        desc_summary = a4_desc.mean(dim=1)  # [B, 8]

        total_lb = 0.0
        total_spv = 0.0
        total_entropy = 0.0
        for i, layer in enumerate(enc.transformer.layers):
            features = layer(features)
            moe_delta, lb, spv, ent = self.audio_moe[i](features, desc_summary)
            features = features + moe_delta
            total_lb = total_lb + lb
            total_spv += spv
            total_entropy = total_entropy + ent

        # MERTEncoderLite has no output_norm
        return features.mean(dim=1), total_lb / 4, total_spv / 4, total_entropy / 4

    def _encode_midi_with_moe(self, pitch: torch.Tensor, vel: torch.Tensor,
                              dur: torch.Tensor, mask: Optional[torch.Tensor]):
        """Returns (pooled_features [B,512], avg_load_balance, avg_seg_pref_var, avg_entropy)."""
        enc = self.base_model.midi_encoder
        B = pitch.shape[0]

        # Compute D4 with ORIGINAL note_mask BEFORE CLS insertion
        note_mask = mask
        with torch.no_grad():
            d4 = compute_local_interval_features(pitch, note_mask)
        if note_mask is not None:
            valid = ~note_mask.unsqueeze(-1)
            desc_summary = (d4 * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            desc_summary = d4.mean(dim=1)  # [B, 4]

        x = enc.event_embedding(pitch, vel, dur)

        # CLS token insertion
        transformer_mask = mask
        if enc.aggregation == "cls":
            cls_tokens = enc.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if transformer_mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
                transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)

        x = enc.pos_encoding(x)

        # Transformer with MoE adapters + PADDING MASK
        total_lb = 0.0
        total_spv = 0.0
        total_entropy = 0.0
        for i, layer in enumerate(enc.transformer.layers):
            if transformer_mask is not None:
                x = layer(x, src_key_padding_mask=transformer_mask)
            else:
                x = layer(x)
            moe_delta, lb, spv, ent = self.midi_moe[i](x, desc_summary, padding_mask=transformer_mask)
            x = x + moe_delta
            total_lb = total_lb + lb
            total_spv += spv
            total_entropy = total_entropy + ent

        x = enc.output_norm(x)

        # Pooling — replicate MIDIEncoder aggregation logic
        if enc.aggregation == "mean":
            if transformer_mask is not None:
                m = ~transformer_mask.unsqueeze(-1)
                x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif enc.aggregation == "cls":
            x = x[:, 0, :]
        elif enc.aggregation == "attention":
            weights = enc.attention_pool(x)
            if transformer_mask is not None:
                weights = weights.masked_fill(transformer_mask.unsqueeze(-1), float("-inf"))
            weights = torch.softmax(weights, dim=1)
            x = (x * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {enc.aggregation}")

        return x, total_lb / 4, total_spv / 4, total_entropy / 4

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_emb_raw, _, _, _ = self._encode_audio_with_moe(audio)
        audio_emb = self.base_model.audio_projection(audio_emb_raw)

        if self.moe_mode == 'dual':
            midi_emb_raw, _, _, _ = self._encode_midi_with_moe(
                midi_pitch, midi_velocity, midi_duration, midi_mask,
            )
            midi_emb = self.base_model.midi_projection(midi_emb_raw)
        else:
            midi_emb = self.base_model.encode_midi(
                midi_pitch, midi_velocity, midi_duration, midi_mask,
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
        audio_emb_raw, audio_lb, audio_spv, audio_ent = self._encode_audio_with_moe(audio)
        audio_emb = self.base_model.audio_projection(audio_emb_raw)

        if self.moe_mode == 'dual':
            midi_emb_raw, midi_lb, midi_spv, midi_ent = self._encode_midi_with_moe(
                midi_pitch, midi_velocity, midi_duration, midi_mask,
            )
            midi_emb = self.base_model.midi_projection(midi_emb_raw)
            total_lb = (audio_lb + midi_lb) / 2
            total_spv = (audio_spv + midi_spv) / 2
            total_entropy = (audio_ent + midi_ent) / 2
        else:
            midi_emb = self.base_model.encode_midi(
                midi_pitch, midi_velocity, midi_duration, midi_mask,
            )
            total_lb = audio_lb
            total_spv = audio_spv
            total_entropy = audio_ent

        vicreg_loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        total_loss = vicreg_loss + self.lb_weight * total_lb

        # Entropy penalty: penalizes uniform routing per-token (v3)
        if self.entropy_weight > 0:
            total_loss = total_loss + self.entropy_weight * total_entropy

        metrics['ratio_aux_loss'] = total_lb.item() if isinstance(total_lb, torch.Tensor) else total_lb
        metrics['total_loss'] = total_loss.item()
        metrics['load_balance'] = total_lb.item() if isinstance(total_lb, torch.Tensor) else total_lb
        metrics['moe_segment_pref_var'] = total_spv
        # Separate audio/midi for dual diagnostics
        if self.moe_mode == 'dual':
            metrics['audio_lb'] = audio_lb.item() if isinstance(audio_lb, torch.Tensor) else audio_lb
            metrics['midi_lb'] = midi_lb.item() if isinstance(midi_lb, torch.Tensor) else midi_lb
            metrics['audio_spv'] = audio_spv
            metrics['midi_spv'] = midi_spv
        if self.entropy_weight > 0:
            metrics['routing_entropy'] = total_entropy.item() if isinstance(total_entropy, torch.Tensor) else total_entropy
        return total_loss, metrics


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
    elif descriptor == 'a10a':
        return Gate42AudioAugModel(base_model, audio_descriptor_type='a10a', audio_descriptor_dim=12)
    elif descriptor == 'a10d':
        return Gate42AudioAugModel(base_model, audio_descriptor_type='a10d', audio_descriptor_dim=32)
    # Gate 10: PCA — FiLM-conditioned audio projection
    elif descriptor == 'a7-pca':
        return Gate42AudioPCAModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
    elif descriptor == 'a10a-pca':
        return Gate42AudioPCAModel(base_model, audio_descriptor_type='a10a', audio_descriptor_dim=12)
    elif descriptor == 'a10d-pca':
        return Gate42AudioPCAModel(base_model, audio_descriptor_type='a10d', audio_descriptor_dim=32)
    # Gate 10: Attention Bias
    elif descriptor == 'a7-ab':
        return Gate42AudioAttnBiasModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
    elif descriptor == 'a10a-ab':
        return Gate42AudioAttnBiasModel(base_model, audio_descriptor_type='a10a', audio_descriptor_dim=12)
    elif descriptor == 'a10d-ab':
        return Gate42AudioAttnBiasModel(base_model, audio_descriptor_type='a10d', audio_descriptor_dim=32)
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
    elif descriptor == 'a7r':
        return Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
    elif descriptor == 'a9r':
        return Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a9', audio_descriptor_dim=12)
    elif descriptor == 'a10ar':
        return Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a10a', audio_descriptor_dim=12)
    elif descriptor == 'a10br':
        return Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a10b', audio_descriptor_dim=12)
    elif descriptor == 'a10cr':
        return Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a10c', audio_descriptor_dim=6)
    elif descriptor == 'a10dr':
        return Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a10d', audio_descriptor_dim=32)
    elif descriptor == 'a10er':
        return Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a10e', audio_descriptor_dim=32)
    elif descriptor == 'd4a4r':
        return Gate42DualReverseCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8, interval_dim=4)
    # Gate 4.3-ext — Dual Mixed
    elif descriptor == 'd4-a4r':
        return Gate42DualMixedModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8, interval_dim=4)
    # Gate 4.4 — Third Tower
    elif descriptor == 't3-tri':
        return Gate44ThirdTowerModel(base_model, loss_mode='triangular', use_d4a4_injection=True)
    elif descriptor == 't3-anc':
        return Gate44ThirdTowerModel(base_model, loss_mode='anchor', use_d4a4_injection=False)
    elif descriptor == 't3-wt':
        return Gate44ThirdTowerModel(base_model, loss_mode='weighted', alpha_ratio=0.3, use_d4a4_injection=True)
    # Gate 4.4 — FiLM
    elif descriptor == 'film-a4':
        return Gate44FiLMModel(base_model, film_mode='audio')
    elif descriptor == 'film-d4':
        return Gate44FiLMModel(base_model, film_mode='midi')
    elif descriptor == 'film-dual':
        return Gate44FiLMModel(base_model, film_mode='dual')
    # Gate 4.4 — MoE
    elif descriptor == 'moe-a4':
        return Gate44MoEModel(base_model, moe_mode='audio')
    elif descriptor == 'moe-dual':
        return Gate44MoEModel(base_model, moe_mode='dual')
    # Gate 4.4 — MoE variants (symmetry-breaking)
    elif descriptor == 'moe-a4-v2':
        return Gate44MoEModel(base_model, moe_mode='audio',
                              expert_init_std=0.02,
                              router_noise_start=0.5, router_noise_end=0.05)
    elif descriptor == 'moe-a4-v3':
        return Gate44MoEModel(base_model, moe_mode='audio',
                              expert_init_std=0.02,
                              router_noise_start=0.5, router_noise_end=0.05,
                              entropy_weight=0.01)
    elif descriptor == 'moe-a4-v4':
        return Gate44MoEModel(base_model, moe_mode='audio',
                              expert_init_std=0.02,
                              router_noise_start=0.5, router_noise_end=0.05,
                              use_top1=True)

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

    # For PCA descriptors, audio_projection is frozen (not used in forward),
    # so exclude it from the projections group.
    if descriptor.endswith('-pca'):
        for p in model.base_model.audio_projection.parameters():
            p.requires_grad = False
        proj_params = list(base.midi_projection.parameters())
    else:
        proj_params = (
            list(base.audio_projection.parameters()) +
            list(base.midi_projection.parameters())
        )

    param_groups = audio_groups + [
        {
            'params': list(base.midi_encoder.parameters()),
            'lr': lr_midi,
            'name': 'midi_encoder',
        },
        {
            'params': proj_params,
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
    elif descriptor in ('a4', 'a7', 'a10a', 'a10d'):
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
    elif descriptor in ('a4r', 'a7r', 'a9r', 'a10ar', 'a10br', 'a10cr', 'a10dr', 'a10er'):
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
    elif descriptor == 'd4a4r':
        # Audio reverse cross-att components
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
            'params': list(model.audio_cross_attention.parameters()),
            'lr': lr_ratio,
            'name': 'audio_cross_attention',
        })
        param_groups.append({
            'params': list(model.audio_cross_attn_norm.parameters()),
            'lr': lr_ratio,
            'name': 'audio_cross_attn_norm',
        })
        # MIDI reverse cross-att components
        param_groups.append({
            'params': list(model.interval_q_proj.parameters()),
            'lr': lr_ratio,
            'name': 'interval_q_proj',
        })
        param_groups.append({
            'params': list(model.midi_cross_attention.parameters()),
            'lr': lr_ratio,
            'name': 'midi_cross_attention',
        })
        param_groups.append({
            'params': list(model.midi_cross_attn_norm.parameters()),
            'lr': lr_ratio,
            'name': 'midi_cross_attn_norm',
        })
    # Gate 4.3-ext — Dual Mixed (A4r audio + D4 concat MIDI)
    elif descriptor == 'd4-a4r':
        # Audio reverse cross-att components (same as a4r)
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
        # MIDI D4 concat component
        param_groups.append({
            'params': list(model.interval_projection.parameters()),
            'lr': lr_ratio,
            'name': 'interval_projection',
        })
    # Gate 4.4 — Third Tower
    elif descriptor in ('t3-tri', 't3-anc', 't3-wt'):
        param_groups.append({
            'params': list(model.ratio_input_proj.parameters()),
            'lr': lr_ratio,
            'name': 'ratio_input_proj',
        })
        param_groups.append({
            'params': list(model.ratio_transformer.parameters()),
            'lr': lr_ratio,
            'name': 'ratio_transformer',
        })
        param_groups.append({
            'params': list(model.ratio_norm.parameters()),
            'lr': lr_ratio,
            'name': 'ratio_norm',
        })
        param_groups.append({
            'params': list(model.ratio_projection.parameters()),
            'lr': lr_ratio,
            'name': 'ratio_projection',
        })
        # Pos encoding has no trainable params (sinusoidal), skip
        if model.use_d4a4_injection:
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
    # Gate 4.4 — FiLM
    elif descriptor in ('film-a4', 'film-d4', 'film-dual'):
        if hasattr(model, 'audio_film_gen'):
            param_groups.append({
                'params': list(model.audio_film_gen.parameters()),
                'lr': lr_ratio,
                'name': 'audio_film_gen',
            })
        if hasattr(model, 'midi_film_gen'):
            param_groups.append({
                'params': list(model.midi_film_gen.parameters()),
                'lr': lr_ratio,
                'name': 'midi_film_gen',
            })
    # Gate 4.4 — MoE
    elif descriptor in ('moe-a4', 'moe-dual', 'moe-a4-v2', 'moe-a4-v3', 'moe-a4-v4'):
        if hasattr(model, 'audio_moe'):
            param_groups.append({
                'params': list(model.audio_moe.parameters()),
                'lr': lr_ratio,
                'name': 'audio_moe',
            })
        if hasattr(model, 'midi_moe'):
            param_groups.append({
                'params': list(model.midi_moe.parameters()),
                'lr': lr_ratio,
                'name': 'midi_moe',
            })
    # Gate 10 — PCA (FiLM-conditioned audio projection)
    elif descriptor.endswith('-pca'):
        # Standard projection weights at lr_proj
        standard_params = (
            list(model.cond_audio_projection.hidden_layers.parameters()) +
            list(model.cond_audio_projection.final_linear.parameters())
        )
        param_groups.append({
            'params': standard_params,
            'lr': lr_proj,
            'name': 'cond_audio_projection',
        })
        # FiLM generators at lr_ratio (these are the NEW learnable params)
        param_groups.append({
            'params': list(model.cond_audio_projection.film_generators.parameters()),
            'lr': lr_ratio,
            'name': 'film_generators',
        })
    # Gate 10 — Attention Bias
    elif descriptor.endswith('-ab'):
        param_groups.append({
            'params': list(model.bias_computer.parameters()),
            'lr': lr_ratio,
            'name': 'bias_computer',
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
        'a7r': (39_000_000, 46_000_000),   # same arch as a4r but q_proj(12→1024)
        'a9r': (39_000_000, 46_000_000),   # same arch as a4r but q_proj(12→1024)
        'a10ar': (39_000_000, 46_000_000),  # same arch as a7r: q_proj(12→1024)
        'a10br': (39_000_000, 46_000_000),  # same arch as a7r: q_proj(12→1024)
        'a10cr': (39_000_000, 46_000_000),  # q_proj(6→1024), slightly fewer params
        'a10dr': (39_000_000, 46_000_000),
        'a10er': (39_000_000, 46_000_000),
        'd4a4r': (39_000_000, 48_000_000),  # +~5.3M (A4r ~4.4M + D4r ~1.05M)
        'd4-a4r': (43_400_000, 45_400_000),  # actual run-b: ~44.4M (A4r ~4.2M + D4 concat ~0.26M)
        # Gate 4.4 — Third Tower (~3.4M ratio tower + optional d4a4 ~1.3M injection)
        't3-tri': (41_500_000, 44_000_000),   # actual: 42,740,992
        't3-anc': (40_000_000, 42_500_000),   # actual: 41,415,424 (NO d4a4 injection)
        't3-wt':  (41_500_000, 44_000_000),   # actual: 42,740,992
        # Gate 4.4 — FiLM (~1M audio, ~0.5M midi)
        'film-a4':   (39_500_000, 42_000_000),   # actual: 40,757,376
        'film-d4':   (39_000_000, 41_500_000),   # actual: 40,228,480
        'film-dual': (40_000_000, 42_500_000),   # actual: 41,286,400
        # Gate 4.4 — MoE (~4.4M audio, ~1.2M midi)
        'moe-a4':   (43_000_000, 45_500_000),   # actual: 44,168,968
        'moe-dual': (44_000_000, 46_500_000),   # actual: 45,355,536
        'moe-a4-v2': (43_000_000, 45_500_000),  # same arch as moe-a4
        'moe-a4-v3': (43_000_000, 45_500_000),  # same arch as moe-a4
        'moe-a4-v4': (43_000_000, 45_500_000),  # same arch as moe-a4
        # Gate 10 — concat (same arch as a7)
        'a10a': (39_000_000, 42_000_000),   # Linear(1036,1024)+LN, same as a7
        'a10d': (39_000_000, 42_000_000),   # Linear(1056,1024)+LN, slightly larger
        # Gate 10 — PCA (FiLM cond projection, audio_projection frozen)
        'a7-pca':   (38_000_000, 41_500_000),   # calibrate from smoke test
        'a10a-pca': (38_000_000, 41_500_000),
        'a10d-pca': (38_000_000, 41_500_000),
        # Gate 10 — Attention Bias (+~2.2K, essentially same as d0)
        'a7-ab':   (39_000_000, 41_000_000),
        'a10a-ab': (39_000_000, 41_000_000),
        'a10d-ab': (39_000_000, 41_000_000),
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
        'a7r': (64_000_000, 72_000_000),   # same arch as a4r but q_proj(12→1024)
        'a9r': (64_000_000, 72_000_000),   # same arch as a4r but q_proj(12→1024)
        'a10ar': (64_000_000, 72_000_000),  # same arch as a7r: q_proj(12→1024)
        'a10br': (64_000_000, 72_000_000),  # same arch as a7r: q_proj(12→1024)
        'a10cr': (64_000_000, 72_000_000),  # q_proj(6→1024), slightly fewer params
        'a10dr': (64_000_000, 72_000_000),
        'a10er': (64_000_000, 72_000_000),
        'd4a4r': (64_000_000, 74_000_000),  # +~5.3M (A4r ~4.4M + D4r ~1.05M)
        'd4-a4r': (68_600_000, 70_600_000),  # actual run-d: 69,572,096 (A4r ~4.2M + D4 concat ~0.26M)
        # Gate 4.4 — Third Tower
        't3-tri': (66_500_000, 69_000_000),   # actual: 67,933,440
        't3-anc': (65_500_000, 68_000_000),   # actual: 66,607,872
        't3-wt':  (66_500_000, 69_000_000),   # actual: 67,933,440
        # Gate 4.4 — FiLM
        'film-a4':   (65_000_000, 67_000_000),   # actual: 65,949,824
        'film-d4':   (64_500_000, 66_500_000),   # actual: 65,420,928
        'film-dual': (65_500_000, 67_500_000),   # actual: 66,478,848
        # Gate 4.4 — MoE
        'moe-a4':   (68_000_000, 71_000_000),   # actual: 69,361,416
        'moe-dual': (69_000_000, 72_000_000),   # actual: 70,547,984
        'moe-a4-v2': (68_000_000, 71_000_000),  # same arch as moe-a4
        'moe-a4-v3': (68_000_000, 71_000_000),  # same arch as moe-a4
        'moe-a4-v4': (68_000_000, 71_000_000),  # same arch as moe-a4
        # Gate 10 — concat (same arch as a7)
        'a10a': (64_000_000, 68_000_000),   # Linear(1036,1024)+LN, same as a7
        'a10d': (64_000_000, 68_000_000),   # Linear(1056,1024)+LN, slightly larger
        # Gate 10 — PCA (FiLM cond projection, audio_projection frozen)
        'a7-pca':   (64_000_000, 66_300_000),   # calibrate from smoke test
        'a10a-pca': (64_000_000, 66_300_000),
        'a10d-pca': (64_000_000, 66_500_000),   # cond_dim=32, FiLM slightly larger
        # Gate 10 — Attention Bias (+~2.2K, essentially same as d0)
        'a7-ab':   (64_000_000, 66_200_000),
        'a10a-ab': (64_000_000, 66_200_000),
        'a10d-ab': (64_000_000, 66_200_000),
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
    elif descriptor in ('a4', 'a7', 'a10a', 'a10d'):
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
    elif descriptor in ('a4r', 'a7r', 'a9r', 'a10ar', 'a10br', 'a10cr', 'a10dr', 'a10er'):
        trainable_prefixes.extend([
            'descriptor_q_proj.',
            'desc_pos_embedding',
            'cross_attention.',
            'cross_attn_norm.',
        ])
    elif descriptor == 'd4a4r':
        trainable_prefixes.extend([
            'descriptor_q_proj.',
            'desc_pos_embedding',
            'audio_cross_attention.',
            'audio_cross_attn_norm.',
            'interval_q_proj.',
            'midi_cross_attention.',
            'midi_cross_attn_norm.',
        ])
    # Gate 4.3-ext — Dual Mixed (A4r audio + D4 concat MIDI)
    elif descriptor == 'd4-a4r':
        trainable_prefixes.extend([
            'descriptor_q_proj.',
            'desc_pos_embedding',
            'cross_attention.',
            'cross_attn_norm.',
            'interval_projection.',
        ])
    # Gate 4.4 — Third Tower
    elif descriptor in ('t3-tri', 't3-wt'):
        trainable_prefixes.extend([
            'ratio_input_proj.', 'ratio_transformer.', 'ratio_norm.',
            'ratio_projection.',
            'audio_descriptor_projection.', 'interval_projection.',
        ])
    elif descriptor == 't3-anc':
        trainable_prefixes.extend([
            'ratio_input_proj.', 'ratio_transformer.', 'ratio_norm.',
            'ratio_projection.',
        ])
    # Gate 4.4 — FiLM
    elif descriptor == 'film-a4':
        trainable_prefixes.append('audio_film_gen.')
    elif descriptor == 'film-d4':
        trainable_prefixes.append('midi_film_gen.')
    elif descriptor == 'film-dual':
        trainable_prefixes.extend(['audio_film_gen.', 'midi_film_gen.'])
    # Gate 4.4 — MoE
    elif descriptor in ('moe-a4', 'moe-a4-v2', 'moe-a4-v3', 'moe-a4-v4'):
        trainable_prefixes.append('audio_moe.')
    elif descriptor == 'moe-dual':
        trainable_prefixes.extend(['audio_moe.', 'midi_moe.'])
    # Gate 10 — PCA (FiLM-conditioned projection replaces audio_projection)
    elif descriptor.endswith('-pca'):
        # Remove audio_projection from default trainable (it's frozen for PCA)
        trainable_prefixes = [p for p in trainable_prefixes
                              if not p.startswith('base_model.audio_projection.')]
        frozen_prefixes.append('base_model.audio_projection.')
        trainable_prefixes.append('cond_audio_projection.')
    # Gate 10 — Attention Bias
    elif descriptor.endswith('-ab'):
        trainable_prefixes.append('bias_computer.')

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
            'eval_compatible': descriptor not in ('d4', 'a4', 'a7', 'a10a', 'a10d', 'd4a4', 'd4a7', 'a4x', 'a7x', 'd4x', 'd4a4cm', 'a4r', 'a7r', 'a9r', 'a10ar', 'a10br', 'a10cr', 'a10dr', 'a10er', 'd4a4r', 'd4-a4r', 't3-tri', 't3-anc', 't3-wt', 'film-a4', 'film-d4', 'film-dual', 'moe-a4', 'moe-dual', 'moe-a4-v2', 'moe-a4-v3', 'moe-a4-v4') and not descriptor.endswith(('-pca', '-ab')),
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_S': best_S,
    }, path)

    # Base checkpoint: CrossModalModel pure state dict
    if descriptor in ('d4', 'a4', 'a7', 'a10a', 'a10d', 'd4a4', 'd4a7', 'a4x', 'a7x', 'd4x', 'd4a4cm', 'a4r', 'a7r', 'a9r', 'a10ar', 'a10br', 'a10cr', 'a10dr', 'a10er', 'd4a4r', 'd4-a4r', 't3-tri', 't3-anc', 't3-wt', 'film-a4', 'film-d4', 'film-dual', 'moe-a4', 'moe-dual', 'moe-a4-v2', 'moe-a4-v3', 'moe-a4-v4') or descriptor.endswith(('-pca', '-ab')):
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

        # MoE schedule: decay router noise per epoch
        if hasattr(model, 'update_schedule'):
            model.update_schedule(epoch, epochs)

        # --- Train ---
        model.train()
        total_loss = 0.0
        total_aux = 0.0
        total_lb = 0.0   # MoE load_balance accumulator
        total_spv = 0.0  # MoE segment_pref_var accumulator
        total_rent = 0.0      # routing entropy
        total_audio_lb = 0.0  # separate audio lb (dual)
        total_midi_lb = 0.0   # separate midi lb (dual)
        total_audio_spv = 0.0
        total_midi_spv = 0.0
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
            total_lb += metrics.get('load_balance', 0.0)
            total_spv += metrics.get('moe_segment_pref_var', 0.0)
            total_rent += metrics.get('routing_entropy', 0.0)
            total_audio_lb += metrics.get('audio_lb', 0.0)
            total_midi_lb += metrics.get('midi_lb', 0.0)
            total_audio_spv += metrics.get('audio_spv', 0.0)
            total_midi_spv += metrics.get('midi_spv', 0.0)
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
            if metrics.get('load_balance') is not None and descriptor.startswith('moe-'):
                postfix['lb'] = f"{metrics['load_balance']:.4f}"
            if metrics.get('routing_entropy') is not None and metrics.get('routing_entropy', 0) > 0:
                postfix['rent'] = f"{metrics['routing_entropy']:.4f}"
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
            'lr_mult': round(scheduler.lr_mult, 6),
            'train_time_min': round(train_time, 1),
            'epoch_time_min': round(epoch_time, 1),
            **quick_metrics,
        }
        # MoE-specific metrics
        if descriptor.startswith('moe-'):
            epoch_record['load_balance'] = total_lb / max(n_batches, 1)
            epoch_record['moe_segment_pref_var'] = total_spv / max(n_batches, 1)
            epoch_record['routing_entropy'] = total_rent / max(n_batches, 1)
            # Separate audio/midi for dual (non-zero only if moe_mode='dual')
            if total_audio_lb > 0 or total_midi_lb > 0:
                epoch_record['audio_lb'] = total_audio_lb / max(n_batches, 1)
                epoch_record['midi_lb'] = total_midi_lb / max(n_batches, 1)
                epoch_record['audio_spv'] = total_audio_spv / max(n_batches, 1)
                epoch_record['midi_spv'] = total_midi_spv / max(n_batches, 1)
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

        _is_gate44 = descriptor.startswith(('t3-', 'film-', 'moe-'))
        _is_gate43_ext = descriptor == 'd4-a4r'
        if getattr(args, 'gate', None):
            _gate_label = args.gate
        elif _is_gate44:
            _gate_label = '4.4'
        elif _is_gate43_ext:
            _gate_label = '4.3-ext'
        elif args.from_scratch:
            _gate_label = '4.3-scratch'
        else:
            _gate_label = '4.2'

        logger.info("=" * 60)
        if args.from_scratch:
            logger.info(f"GATE {_gate_label} — DESCRIPTOR: {descriptor.upper()}")
            logger.info("Training FROM SCRATCH (MERT pretrained + random MIDI)")
        else:
            logger.info(f"GATE {_gate_label} — DESCRIPTOR: {descriptor.upper()}")
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

        # Gate 10 guard: must use run-d
        if getattr(args, 'gate', None) == '10' and freeze_policy != 'run-d':
            raise ValueError("Gate 10 requires --freeze-policy run-d")

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
            'gate': _gate_label,
            'descriptor': descriptor,
            'family': 'dual-mixed' if _is_gate43_ext else None,
            'ratio_weight': args.ratio_weight,
            'foundation_checkpoint': None if args.from_scratch else args.checkpoint,
            'from_scratch': args.from_scratch,
            'freeze_policy': freeze_policy,
            'use_d4a4_injection': getattr(model, 'use_d4a4_injection', None),
        }
        # MoE variant config — saved in checkpoint for eval reconstruction
        if descriptor.startswith('moe-'):
            arch_config['moe_config'] = {
                'lb_weight': getattr(model, 'lb_weight', 0.01),
                'expert_init_std': getattr(model, 'expert_init_std', 0.0),
                'router_noise_start': getattr(model, 'router_noise_start', 0.0),
                'router_noise_end': getattr(model, 'router_noise_end', 0.0),
                'use_top1': getattr(model, 'use_top1', False),
                'entropy_weight': getattr(model, 'entropy_weight', 0.0),
            }

    # Preflight
    mode_key = f'gate42_{descriptor}'
    PARAM_RANGES[mode_key] = GATE42_PARAM_RANGES[freeze_policy][descriptor]

    # Guard: d4-a4r param range must be calibrated (width ≤ 2M)
    if descriptor == 'd4-a4r':
        _lo, _hi = GATE42_PARAM_RANGES[freeze_policy][descriptor]
        if _hi - _lo > 2_000_000:
            raise ValueError(
                f"d4-a4r param range too wide: ({_lo:,}, {_hi:,}), "
                f"width={_hi-_lo:,} > 2M. Calibrate with real pilot count."
            )

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
    cosine_ref_steps = (args.lr_cosine_ref_epochs * args.max_batches_per_epoch
                        if args.lr_cosine_ref_epochs > 0 else 0)
    scheduler = LinearWarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps,
        hold_fraction=args.lr_hold_fraction,
        cosine_ref_steps=cosine_ref_steps,
        lr_floor=args.lr_floor,
        lr_tail_end=args.lr_tail_end,
    )

    if cosine_ref_steps > 0 and args.lr_floor > 0:
        tail_ep = scheduler.tail_start / args.max_batches_per_epoch
        tail_steps = total_steps - scheduler.tail_start
        logger.info(f"  LR schedule: warmup {args.warmup_steps} steps → "
                     f"cosine(ref={args.lr_cosine_ref_epochs}ep) → "
                     f"tail {args.lr_floor}→{args.lr_tail_end} "
                     f"(from e{tail_ep:.1f}, {tail_steps} steps, "
                     f"{tail_steps/args.max_batches_per_epoch:.1f} ep)")
    elif args.lr_hold_fraction > 0:
        hold_ep = scheduler.hold_end / args.max_batches_per_epoch
        decay_steps = total_steps - scheduler.hold_end
        logger.info(f"  LR schedule: warmup {args.warmup_steps} steps → "
                     f"HOLD {args.lr_hold_fraction:.0%} ({hold_ep:.1f} ep) → "
                     f"cosine decay ({decay_steps} steps, "
                     f"{decay_steps/args.max_batches_per_epoch:.1f} ep)")

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
        'gate': arch_config.get('gate', '4.2'),
        'descriptor': descriptor,
        'training': train_results,
        'evaluation_best': best_eval,
        'evaluation_final': last_eval,
    }
    # MoE metrics from BEST epoch (same epoch as best_S)
    if descriptor.startswith('moe-'):
        history = train_results.get('history', [])
        best_ep = train_results.get('best_epoch', 0)
        best_ep_record = next((r for r in history if r.get('epoch') == best_ep), history[-1] if history else {})
        final['moe_metrics'] = {
            'load_balance': best_ep_record.get('load_balance', None),
            'moe_segment_pref_var': best_ep_record.get('moe_segment_pref_var', None),
            'epoch': best_ep_record.get('epoch', None),
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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch_config = checkpoint.get('arch_config')

    if arch_config is None:
        raise RuntimeError("Missing arch_config in checkpoint")

    logger.info("=" * 60)
    logger.info(f"GATE {arch_config.get('gate', '4.2')} — EVALUATE")
    logger.info("=" * 60)

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
    elif descriptor == 'a10a':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioAugModel(base_model, audio_descriptor_type='a10a', audio_descriptor_dim=12)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'a10d':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioAugModel(base_model, audio_descriptor_type='a10d', audio_descriptor_dim=32)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # Gate 10 — PCA
    elif descriptor.endswith('-pca'):
        desc_type = descriptor.replace('-pca', '')
        dim_map = {'a7': 12, 'a10a': 12, 'a10d': 32}
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioPCAModel(base_model, desc_type, dim_map[desc_type])
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # Gate 10 — Attention Bias
    elif descriptor.endswith('-ab'):
        desc_type = descriptor.replace('-ab', '')
        dim_map = {'a7': 12, 'a10a': 12, 'a10d': 32}
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioAttnBiasModel(base_model, desc_type, dim_map[desc_type])
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
    elif descriptor in ('a7r', 'a9r'):
        ad_type = 'a7' if descriptor == 'a7r' else 'a9'
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type=ad_type, audio_descriptor_dim=12)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor in ('a10ar', 'a10br', 'a10cr', 'a10dr', 'a10er'):
        ad_type = {'a10ar': 'a10a', 'a10br': 'a10b', 'a10cr': 'a10c',
                   'a10dr': 'a10d', 'a10er': 'a10e'}[descriptor]
        ad_dim = {'a10ar': 12, 'a10br': 12, 'a10cr': 6,
                  'a10dr': 32, 'a10er': 32}[descriptor]
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type=ad_type, audio_descriptor_dim=ad_dim)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'd4a4r':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42DualReverseCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8, interval_dim=4)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # Gate 4.3-ext — Dual Mixed
    elif descriptor == 'd4-a4r':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate42DualMixedModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8, interval_dim=4)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # Gate 4.4 — Third Tower
    elif descriptor == 't3-anc':
        use_injection = arch_config.get('use_d4a4_injection', False)
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate44ThirdTowerModel(base_model, loss_mode='anchor', use_d4a4_injection=use_injection)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 't3-tri':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate44ThirdTowerModel(base_model, loss_mode='triangular', use_d4a4_injection=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 't3-wt':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate44ThirdTowerModel(base_model, loss_mode='weighted', alpha_ratio=0.3, use_d4a4_injection=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # Gate 4.4 — FiLM
    elif descriptor == 'film-a4':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate44FiLMModel(base_model, film_mode='audio')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'film-d4':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate44FiLMModel(base_model, film_mode='midi')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif descriptor == 'film-dual':
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = Gate44FiLMModel(base_model, film_mode='dual')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # Gate 4.4 — MoE (all variants)
    elif descriptor in ('moe-a4', 'moe-dual', 'moe-a4-v2', 'moe-a4-v3', 'moe-a4-v4'):
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        moe_mode = 'dual' if descriptor == 'moe-dual' else 'audio'
        moe_kwargs = arch_config.get('moe_config', {})
        model = Gate44MoEModel(base_model, moe_mode=moe_mode, **moe_kwargs)
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
    # Attention bias: bias [B*8, T, T] is heavy — clamp to 8
    if descriptor.endswith('-ab') and eval_bs > 8:
        eval_bs = 8
    elif descriptor in ('a4x', 'a7x', 'a4r', 'a7r', 'a9r', 'a10ar', 'a10br', 'a10cr', 'a10dr', 'a10er', 'd4a4r', 'd4-a4r') and eval_bs > 16:
        eval_bs = 16  # cross-attn matrix heavier than concat
    elif descriptor in ('a4', 'a7', 'a10a', 'a10d', 'd4a4', 'd4a7', 'd4a4cm') and eval_bs > 32:
        eval_bs = 32
    elif descriptor.endswith('-pca') and eval_bs > 32:
        eval_bs = 32
    # Gate 4.4: layer iteration ~ same memory
    elif descriptor in ('t3-tri', 't3-anc', 't3-wt', 'film-a4', 'film-d4', 'film-dual', 'moe-a4', 'moe-dual', 'moe-a4-v2', 'moe-a4-v3', 'moe-a4-v4') and eval_bs > 32:
        eval_bs = 32

    results = run_structured_eval(
        model=model, maestro_dir=Path(args.maestro_dir),
        device=device, seed=args.seed,
        embed_batch_size=eval_bs,
    )

    results['gate'] = arch_config.get('gate', '4.2')
    results['family'] = arch_config.get('family')
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
        choices=['d0', 'd1', 'd2', 'd3', 'd4', 'a4', 'a7', 'a10a', 'a10d', 'd4a4', 'd4a7', 'a4x', 'a7x', 'd4x', 'd4a4cm', 'a4r', 'a7r', 'a9r', 'a10ar', 'a10br', 'a10cr', 'a10dr', 'a10er', 'd4a4r', 'd4-a4r', 't3-tri', 't3-anc', 't3-wt', 'film-a4', 'film-d4', 'film-dual', 'moe-a4', 'moe-dual', 'moe-a4-v2', 'moe-a4-v3', 'moe-a4-v4', 'a7-pca', 'a10a-pca', 'a10d-pca', 'a7-ab', 'a10a-ab', 'a10d-ab'],
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
    parser.add_argument(
        '--gate', type=str, default=None,
        help='Override gate label (e.g., "10"). Used for trazabilidad.',
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
    parser.add_argument('--lr-hold-fraction', type=float, default=0.0,
                        help='Fraction of post-warmup steps to hold LR at max before cosine decay (0=pure cosine)')
    parser.add_argument('--lr-cosine-ref-epochs', type=int, default=0,
                        help='Reference epoch count for cosine phase (cosine-tail mode). '
                             'E.g., 30 replicates the 30ep LR curve in a longer run.')
    parser.add_argument('--lr-floor', type=float, default=0.0,
                        help='LR mult where cosine stops and linear tail begins (cosine-tail mode)')
    parser.add_argument('--lr-tail-end', type=float, default=0.0,
                        help='Final LR mult at end of training (cosine-tail mode)')

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
