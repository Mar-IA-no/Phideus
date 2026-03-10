#!/usr/bin/env python3
"""
Universal Checkpoint Loader for Gate 4.x models.

Extracts the model-reconstruction logic from gate43_scratch_training.py::run_evaluate()
into a reusable module. Handles all descriptor variants including augmented models
that require wrapper classes beyond the base CrossModalModel.

CRITICAL: All imports of model classes from gate43_scratch_training are done LAZILY
inside load_model_from_checkpoint() to avoid circular import with evaluate_structured_pool.

Usage:
    from experiments.bias_control.gate5b.checkpoint_loader import load_model_from_checkpoint
    model, meta = load_model_from_checkpoint('models/gate5b/d4a4/best_model.pt')
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)


# Batch size by descriptor complexity (RTX 3090 24GB)
_EVAL_BATCH_SIZES = {
    # Cross-attention / reverse cross-att: heavier intermediate buffers
    'a4x': 16, 'a7x': 16, 'a4r': 16, 'a7r': 16, 'a9r': 16, 'd4a4r': 16, 'd4-a4r': 16,
    # Concat-based augmentations: moderate
    'a4': 32, 'a7': 32, 'd4a4': 32, 'd4a7': 32, 'd4a4cm': 32,
    # Gate 4.4 models: moderate
    't3-anc': 32, 't3-tri': 32, 't3-wt': 32,
    'film-a4': 32, 'film-d4': 32, 'film-dual': 32,
    'moe-a4': 32, 'moe-dual': 32,
    'moe-a4-v2': 32, 'moe-a4-v3': 32, 'moe-a4-v4': 32,
}


def get_eval_batch_size(descriptor: str) -> int:
    """Optimal batch size for evaluation given descriptor type and VRAM budget."""
    return _EVAL_BATCH_SIZES.get(descriptor, 64)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a Gate 4.x checkpoint and reconstruct the complete model.

    Handles all descriptor variants: D0 (base), d4/a4/a7/d4a4 (concat),
    a4r/d4a4r/d4-a4r (cross-att), t3-*/film-*/moe-* (Gate 4.4).

    Returns:
        (model, metadata)
            model: nn.Module in eval mode on device
            metadata: dict with keys: descriptor, epoch, best_S, arch_config,
                      eval_compatible, checkpoint_path

    Raises:
        RuntimeError: if checkpoint_type == 'archive_base' (stripped checkpoint)
        ValueError: if descriptor is unknown
    """
    if device is None:
        device = torch.device('cpu')

    # --- Lazy imports to avoid circular dependency ---
    from src.bias_control.architectures.cross_modal_model import CrossModalModel
    from experiments.bias_control.gate43_scratch.gate43_scratch_training import (
        Gate42Model,
        Gate42InputAugModel,
        Gate42AudioAugModel,
        Gate42DualAugModel,
        Gate42AudioCrossAttModel,
        Gate42MidiCrossAttModel,
        Gate42DualCrossModalModel,
        Gate42AudioReverseCrossAttModel,
        Gate42DualReverseCrossAttModel,
        Gate42DualMixedModel,
        Gate44ThirdTowerModel,
        Gate44FiLMModel,
        Gate44MoEModel,
        create_gate42_model,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch_config = checkpoint.get('arch_config')

    if arch_config is None:
        raise RuntimeError(f"Missing arch_config in checkpoint: {checkpoint_path}")

    if arch_config.get('checkpoint_type') == 'archive_base':
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is archive_base "
            f"(descriptor={arch_config.get('descriptor')}). "
            f"Cannot evaluate — use the full checkpoint."
        )

    descriptor = arch_config.get('descriptor')
    if not descriptor:
        raise ValueError(f"No descriptor found in arch_config: {arch_config}")

    logger.info(f"Loading checkpoint: descriptor={descriptor}, "
                f"eval_compatible={arch_config.get('eval_compatible')}")

    # --- Determine audio encoder type from arch_config ---
    audio_enc = arch_config.get('audio_encoder', 'lite')  # backward compatible

    # Gate 7.1+ checkpoints use MERT-330M
    if audio_enc == 'mert':
        from experiments.bias_control.gate71.train_gate71 import Gate71Model
        base_model = CrossModalModel(
            audio_encoder='mert', audio_encoder_frozen=True, use_dann=False,
        )
        # Force load MERT model for weight loading
        base_model.audio_encoder._load_model()
        model = Gate71Model(base_model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model = model.to(device)
        model.eval()

        metadata = {
            'descriptor': descriptor,
            'epoch': checkpoint.get('epoch', -1),
            'best_S': checkpoint.get('best_S', None),
            'arch_config': arch_config,
            'eval_compatible': arch_config.get('eval_compatible', True),
            'checkpoint_path': str(checkpoint_path),
        }
        logger.info(f"Loaded MERT-330M checkpoint: descriptor={descriptor}, "
                     f"epoch={metadata['epoch']}, best_S={metadata['best_S']}")
        return model, metadata

    # --- Standard Gate 4.x models (MERTEncoderLite) ---
    # Mirrors gate43_scratch_training.py::run_evaluate() lines 3932-4022
    base_model = CrossModalModel(audio_encoder='lite', use_dann=False)

    if descriptor == 'd4':
        model = Gate42InputAugModel(base_model, interval_dim=4)
    elif descriptor == 'a4':
        model = Gate42AudioAugModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
    elif descriptor == 'a7':
        model = Gate42AudioAugModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
    elif descriptor in ('d4a4', 'd4a7'):
        ad_type = 'a4' if descriptor == 'd4a4' else 'a7'
        ad_dim = 8 if ad_type == 'a4' else 12
        model = Gate42DualAugModel(base_model, audio_descriptor_type=ad_type, audio_descriptor_dim=ad_dim)
    elif descriptor == 'a4x':
        model = Gate42AudioCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
    elif descriptor == 'a7x':
        model = Gate42AudioCrossAttModel(base_model, audio_descriptor_type='a7', audio_descriptor_dim=12)
    elif descriptor == 'd4x':
        model = Gate42MidiCrossAttModel(base_model, interval_dim=4)
    elif descriptor == 'd4a4cm':
        model = Gate42DualCrossModalModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
    elif descriptor == 'a4r':
        model = Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8)
        # Gate 8 conditioned projections (promoted from Gate 5A C1)
        if arch_config.get('proj_cond_audio') or arch_config.get('proj_cond_midi'):
            from experiments.bias_control.gate5a_proj_cond import setup_conditioned_projections
            setup_conditioned_projections(
                model,
                proj_cond_audio=arch_config.get('proj_cond_audio', False),
                proj_cond_midi=arch_config.get('proj_cond_midi', False),
                zero_cond=arch_config.get('zero_cond', False),
            )
    elif descriptor in ('a7r', 'a9r'):
        ad_type = 'a7' if descriptor == 'a7r' else 'a9'
        model = Gate42AudioReverseCrossAttModel(base_model, audio_descriptor_type=ad_type, audio_descriptor_dim=12)
    elif descriptor == 'd4a4r':
        model = Gate42DualReverseCrossAttModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8, interval_dim=4)
    elif descriptor == 'd4-a4r':
        model = Gate42DualMixedModel(base_model, audio_descriptor_type='a4', audio_descriptor_dim=8, interval_dim=4)
    elif descriptor == 't3-anc':
        use_injection = arch_config.get('use_d4a4_injection', False)
        model = Gate44ThirdTowerModel(base_model, loss_mode='anchor', use_d4a4_injection=use_injection)
    elif descriptor == 't3-tri':
        model = Gate44ThirdTowerModel(base_model, loss_mode='triangular', use_d4a4_injection=True)
    elif descriptor == 't3-wt':
        model = Gate44ThirdTowerModel(base_model, loss_mode='weighted', alpha_ratio=0.3, use_d4a4_injection=True)
    elif descriptor == 'film-a4':
        model = Gate44FiLMModel(base_model, film_mode='audio')
    elif descriptor == 'film-d4':
        model = Gate44FiLMModel(base_model, film_mode='midi')
    elif descriptor == 'film-dual':
        model = Gate44FiLMModel(base_model, film_mode='dual')
    elif descriptor in ('moe-a4', 'moe-dual', 'moe-a4-v2', 'moe-a4-v3', 'moe-a4-v4'):
        moe_mode = 'dual' if descriptor == 'moe-dual' else 'audio'
        moe_kwargs = arch_config.get('moe_config', {})
        model = Gate44MoEModel(base_model, moe_mode=moe_mode, **moe_kwargs)
    elif arch_config.get('checkpoint_type') == 'base':
        # Base checkpoint without wrapper
        model = CrossModalModel(audio_encoder='lite', use_dann=False)
    elif descriptor == 'd0':
        # D0 uses Gate42Model wrapper with no ratio components
        model = Gate42Model(
            base_model, descriptor_fn=None,
            ratio_encoder=None, ratio_projection=None,
            ratio_weight=0.0,
        )
    else:
        # Fallback: try create_gate42_model (handles d1, d2, d3, etc.)
        model = create_gate42_model(descriptor, base_model, ratio_weight=0.0)

    # Load state dict with strict=True to catch mismatches
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    model = model.to(device)
    model.eval()

    metadata = {
        'descriptor': descriptor,
        'epoch': checkpoint.get('epoch', -1),
        'best_S': checkpoint.get('best_S', None),
        'arch_config': arch_config,
        'eval_compatible': arch_config.get('eval_compatible', True),
        'checkpoint_path': str(checkpoint_path),
    }

    logger.info(f"Loaded: descriptor={descriptor}, epoch={metadata['epoch']}, "
                f"best_S={metadata['best_S']}, "
                f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    return model, metadata
