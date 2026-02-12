#!/usr/bin/env python3
"""
Bloque A Training: Adapter / Unfreezing controlado (Plan v1.1 Post-DEC-005).

Modes:
  s0     — Eval-only: reproduce Gate 2 baseline with structured pool.
  run-a  — Adapter bottleneck: 4 adapters on frozen audio encoder.
  run-b  — Partial unfreeze: audio transformer layers 2-3 trainable.
  run-c  — Hybrid: adapters on layers 0-1 + unfreeze layers 2-3.
  evaluate — Standalone evaluation of any Bloque A checkpoint.

Usage:
    # S0: Eval-only control
    python experiments/bias_control/bloqueA_training.py \
        --mode s0 \
        --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
        --output data/bias_control_medium/training_outputs/bloqueA_S0 \
        --maestro-dir data/maestro_v3/maestro-v3.0.0

    # Run A: Adapter bottleneck
    python experiments/bias_control/bloqueA_training.py \
        --mode run-a \
        --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
        --output data/bias_control_medium/training_outputs/bloqueA_runA \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --epochs 5 --batch-size 16 --num-workers 8

    # Run B: Partial unfreeze
    python experiments/bias_control/bloqueA_training.py \
        --mode run-b \
        --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
        --output data/bias_control_medium/training_outputs/bloqueA_runB \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --epochs 5 --batch-size 16 --num-workers 8

    # Run C: Hybrid
    python experiments/bias_control/bloqueA_training.py \
        --mode run-c \
        --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
        --output data/bias_control_medium/training_outputs/bloqueA_runC \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --epochs 5 --batch-size 16 --num-workers 8

    # Evaluate a checkpoint
    python experiments/bias_control/bloqueA_training.py \
        --mode evaluate \
        --checkpoint data/bias_control_medium/training_outputs/bloqueA_runA/checkpoint_epoch5.pt \
        --output data/bias_control_medium/evaluations/bloqueA/runA_ep5.json \
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bias_control.architectures.cross_modal_model import CrossModalModel
from src.bias_control.adapters.adapter import AdapterBottleneck, BloqueAModel
from src.bias_control.training.preflight import (
    validate_training_setup,
    DriftSentinel,
    validate_checkpoint_load,
)
from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
)

# Import evaluation functions from evaluate_structured_pool.py
from evaluate_structured_pool import (
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
# LinearWarmupCosine Scheduler (from gate4_ratio_auxiliary.py)
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


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    """Strong determinism for A/B/C comparability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader worker seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _dataloader_kwargs(num_workers: int) -> dict:
    """Extra DataLoader kwargs that depend on num_workers > 0."""
    if num_workers > 0:
        return {'pin_memory': True, 'prefetch_factor': 2}
    return {}


# ---------------------------------------------------------------------------
# Model creation helpers
# ---------------------------------------------------------------------------

def load_base_model(checkpoint_path: str, device: torch.device) -> CrossModalModel:
    """Load CrossModalModel from Gate 2 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CrossModalModel(audio_encoder='lite', use_dann=False)

    result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    validate_checkpoint_load(
        missing_keys=result.missing_keys,
        unexpected_keys=result.unexpected_keys,
        expected_missing=[],
        expected_unexpected=['dann\\..*'],  # Gate 2 may have DANN keys
    )

    logger.info(
        f"Loaded base model from {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch', '?')})"
    )
    return model


def create_run_a_model(
    base_model: CrossModalModel,
    adapter_dim: int = 64,
) -> Tuple[BloqueAModel, dict]:
    """
    Run A: Adapter bottleneck on all 4 audio transformer layers.
    Audio encoder stays FROZEN. Only adapters + MIDI + projections train.
    """
    # Freeze entire audio encoder
    for p in base_model.audio_encoder.parameters():
        p.requires_grad = False

    model = BloqueAModel(
        base_model=base_model,
        adapter_dim=adapter_dim,
        adapter_layers=[0, 1, 2, 3],
    )

    arch_config = {
        'mode': 'run-a',
        'adapter_dim': adapter_dim,
        'adapter_layers': [0, 1, 2, 3],
        'unfrozen_layers': [],
    }
    return model, arch_config


def create_run_b_model(base_model: CrossModalModel) -> Tuple[CrossModalModel, dict]:
    """
    Run B: Partial unfreeze — layers 2-3 of audio encoder.
    CNN, PosEmb, layers 0-1 stay frozen. NO adapters.
    Returns CrossModalModel directly (no wrapper).
    """
    # Freeze CNN
    for p in base_model.audio_encoder.feature_extractor.parameters():
        p.requires_grad = False
    # Freeze PosEmb
    base_model.audio_encoder.pos_embedding.requires_grad = False
    # Freeze layers 0-1
    for p in base_model.audio_encoder.transformer.layers[0].parameters():
        p.requires_grad = False
    for p in base_model.audio_encoder.transformer.layers[1].parameters():
        p.requires_grad = False
    # Layers 2-3 stay requires_grad=True (default)

    arch_config = {
        'mode': 'run-b',
        'adapter_dim': 0,
        'adapter_layers': [],
        'unfrozen_layers': [2, 3],
    }
    return base_model, arch_config


def create_run_c_model(
    base_model: CrossModalModel,
    adapter_dim: int = 64,
) -> Tuple[BloqueAModel, dict]:
    """
    Run C: Hybrid — adapters on layers 0-1 (frozen) + unfreeze layers 2-3.
    """
    # Freeze CNN
    for p in base_model.audio_encoder.feature_extractor.parameters():
        p.requires_grad = False
    # Freeze PosEmb
    base_model.audio_encoder.pos_embedding.requires_grad = False
    # Freeze layers 0-1
    for p in base_model.audio_encoder.transformer.layers[0].parameters():
        p.requires_grad = False
    for p in base_model.audio_encoder.transformer.layers[1].parameters():
        p.requires_grad = False
    # Layers 2-3 stay requires_grad=True

    model = BloqueAModel(
        base_model=base_model,
        adapter_dim=adapter_dim,
        adapter_layers=[0, 1],  # adapters only on frozen layers
    )

    arch_config = {
        'mode': 'run-c',
        'adapter_dim': adapter_dim,
        'adapter_layers': [0, 1],
        'unfrozen_layers': [2, 3],
    }
    return model, arch_config


# ---------------------------------------------------------------------------
# Optimizer creation
# ---------------------------------------------------------------------------

def create_optimizer_run_a(
    model: BloqueAModel,
    lr_adapter: float,
    lr_midi: float,
    lr_proj: float,
) -> AdamW:
    """Optimizer for Run A: 3 param groups."""
    return AdamW([
        {
            'params': list(model.audio_adapters.parameters()),
            'lr': lr_adapter,
            'name': 'adapters',
        },
        {
            'params': list(model.base_model.midi_encoder.parameters()),
            'lr': lr_midi,
            'name': 'midi_encoder',
        },
        {
            'params': (
                list(model.base_model.audio_projection.parameters()) +
                list(model.base_model.midi_projection.parameters())
            ),
            'lr': lr_proj,
            'name': 'projections',
        },
    ], weight_decay=1e-4)


def create_optimizer_run_b(
    model: CrossModalModel,
    lr_audio_unfreeze: float,
    lr_midi: float,
    lr_proj: float,
) -> AdamW:
    """Optimizer for Run B: 3 param groups (audio layers 2-3, midi, projections)."""
    audio_layers_23_params = (
        list(model.audio_encoder.transformer.layers[2].parameters()) +
        list(model.audio_encoder.transformer.layers[3].parameters())
    )
    return AdamW([
        {
            'params': audio_layers_23_params,
            'lr': lr_audio_unfreeze,
            'name': 'audio_layers_2_3',
        },
        {
            'params': list(model.midi_encoder.parameters()),
            'lr': lr_midi,
            'name': 'midi_encoder',
        },
        {
            'params': (
                list(model.audio_projection.parameters()) +
                list(model.midi_projection.parameters())
            ),
            'lr': lr_proj,
            'name': 'projections',
        },
    ], weight_decay=1e-4)


def create_optimizer_run_c(
    model: BloqueAModel,
    lr_adapter: float,
    lr_audio_unfreeze: float,
    lr_midi: float,
    lr_proj: float,
) -> AdamW:
    """Optimizer for Run C: 4 param groups."""
    audio_layers_23_params = (
        list(model.base_model.audio_encoder.transformer.layers[2].parameters()) +
        list(model.base_model.audio_encoder.transformer.layers[3].parameters())
    )
    return AdamW([
        {
            'params': list(model.audio_adapters.parameters()),
            'lr': lr_adapter,
            'name': 'adapters',
        },
        {
            'params': audio_layers_23_params,
            'lr': lr_audio_unfreeze,
            'name': 'audio_layers_2_3',
        },
        {
            'params': list(model.base_model.midi_encoder.parameters()),
            'lr': lr_midi,
            'name': 'midi_encoder',
        },
        {
            'params': (
                list(model.base_model.audio_projection.parameters()) +
                list(model.base_model.midi_projection.parameters())
            ),
            'lr': lr_proj,
            'name': 'projections',
        },
    ], weight_decay=1e-4)


# ---------------------------------------------------------------------------
# Preflight contracts per mode
# ---------------------------------------------------------------------------

def get_preflight_contract(mode: str, uses_wrapper: bool):
    """
    Return (frozen_prefixes, trainable_prefixes) for each mode.

    Prefixes depend on whether we use BloqueAModel wrapper or CrossModalModel directly.
    """
    if mode == 'run-a':
        # BloqueAModel wrapper: base_model.* prefix
        return {
            'frozen_prefixes': [
                'base_model.audio_encoder.',
            ],
            'trainable_prefixes': [
                'audio_adapters.',
                'base_model.midi_encoder.',
                'base_model.audio_projection.',
                'base_model.midi_projection.',
            ],
        }
    elif mode == 'run-b':
        # CrossModalModel directly: no base_model prefix
        return {
            'frozen_prefixes': [
                'audio_encoder.feature_extractor.',
                'audio_encoder.pos_embedding',
                'audio_encoder.transformer.layers.0.',
                'audio_encoder.transformer.layers.1.',
            ],
            'trainable_prefixes': [
                'audio_encoder.transformer.layers.2.',
                'audio_encoder.transformer.layers.3.',
                'midi_encoder.',
                'audio_projection.',
                'midi_projection.',
            ],
        }
    elif mode == 'run-c':
        # BloqueAModel wrapper
        return {
            'frozen_prefixes': [
                'base_model.audio_encoder.feature_extractor.',
                'base_model.audio_encoder.pos_embedding',
                'base_model.audio_encoder.transformer.layers.0.',
                'base_model.audio_encoder.transformer.layers.1.',
            ],
            'trainable_prefixes': [
                'audio_adapters.',
                'base_model.audio_encoder.transformer.layers.2.',
                'base_model.audio_encoder.transformer.layers.3.',
                'base_model.midi_encoder.',
                'base_model.audio_projection.',
                'base_model.midi_projection.',
            ],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LinearWarmupCosineScheduler,
    epoch: int,
    best_recall: float,
    arch_config: dict,
    output_dir: Path,
    filename: str,
    mode: str,
):
    """
    Save checkpoint with arch_config metadata.

    For Run A/C (BloqueAModel): saves full checkpoint + archive_base (NOT for eval).
    For Run B (CrossModalModel): saves full checkpoint + _base.pt (for direct eval).
    """
    path = output_dir / filename

    # Full checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'arch_config': {
            **arch_config,
            'checkpoint_type': 'full',
            'eval_compatible': True,
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_recall': best_recall,
    }, path)

    # Secondary checkpoint
    if mode in ('run-a', 'run-c'):
        # Archive base: NOT for evaluation
        archive_path = path.with_name(path.stem + '_archive_base.pt')
        torch.save({
            'model_state_dict': model.base_model.state_dict(),
            'arch_config': {
                **arch_config,
                'checkpoint_type': 'archive_base',
                'eval_compatible': False,
            },
            'epoch': epoch,
        }, archive_path)
    elif mode == 'run-b':
        # Base checkpoint: compatible with evaluate_structured_pool.py
        base_path = path.with_name(path.stem + '_base.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'arch_config': {
                **arch_config,
                'checkpoint_type': 'base',
                'eval_compatible': True,
            },
            'epoch': epoch,
        }, base_path)


# ---------------------------------------------------------------------------
# Structured pool evaluation
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
    """
    Run structured pool evaluation using functions from evaluate_structured_pool.py.

    Returns dict with a2m, m2a, hard_negative_analysis, gate_metrics, pool_config.
    """
    # Load validation dataset
    val_dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=4.0,
        hop=1.0,
        split='validation',
    )

    index = build_segment_index(val_dataset)
    logger.info(f"Eval: {len(val_dataset)} segments, {len(index['by_piece'])} pieces")

    # Extract embeddings (no_grad to avoid unnecessary graph overhead)
    # Empty cache first — during training, optimizer states fragment GPU memory
    torch.cuda.empty_cache()
    model.eval()
    t0 = time.time()
    logger.info("  [eval] Extracting embeddings (batch_size=%d, %d segments)...", embed_batch_size, len(val_dataset))
    with torch.no_grad():
        audio_embs, midi_embs = extract_all_embeddings(
            model, val_dataset, device, batch_size=embed_batch_size,
        )
    logger.info("  [eval] Embeddings extracted in %.1fs", time.time() - t0)

    config = PoolConfig(
        pool_size=pool_size,
        n_hard_negatives=64,
        n_semi_hard_negatives=32,
        n_queries=n_queries,
    )

    # A2M
    t0 = time.time()
    logger.info("  [eval] Running A2M retrieval (pool=%d, queries=%d)...", pool_size, n_queries)
    a2m = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, val_dataset, index, config,
        direction='a2m', seed=seed,
    )
    logger.info("  [eval] A2M done in %.1fs — R@10=%.1f%%", time.time() - t0, a2m['mean_recall@10'] * 100)

    # M2A
    t0 = time.time()
    logger.info("  [eval] Running M2A retrieval...")
    m2a = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, val_dataset, index, config,
        direction='m2a', seed=seed,
    )
    logger.info("  [eval] M2A done in %.1fs — R@10=%.1f%%", time.time() - t0, m2a['mean_recall@10'] * 100)

    # Hard negatives
    t0 = time.time()
    logger.info("  [eval] Running hard negative analysis (n=%d)...", 500)
    hard_neg = analyze_hard_negatives_fast(
        audio_embs, midi_embs, val_dataset, index,
        n_samples=500, seed=seed,
    )
    logger.info("  [eval] Hard neg done in %.1fs — acc=%.1f%%", time.time() - t0, hard_neg['accuracy_vs_same_piece'] * 100)

    # Gate metrics
    a2m_r10 = a2m['mean_recall@10']
    m2a_r10 = m2a['mean_recall@10']
    S = min(a2m_r10, m2a_r10)
    hard_neg_acc = hard_neg['accuracy_vs_same_piece']

    results = {
        'a2m': a2m,
        'm2a': m2a,
        'hard_negative_analysis': hard_neg,
        'gate_metrics': {
            'S': S,
            'hard_neg': hard_neg_acc,
            'a2m_r10': a2m_r10,
            'm2a_r10': m2a_r10,
        },
        'pool_config': {
            'pool_size': config.pool_size,
            'n_queries': config.n_queries,
            'seed': seed,
            'n_hard_negatives': config.n_hard_negatives,
            'n_semi_hard_negatives': config.n_semi_hard_negatives,
        },
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("STRUCTURED POOL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  A2M R@10: {a2m_r10:.1%}  |  M2A R@10: {m2a_r10:.1%}")
    logger.info(f"  S = min(A2M, M2A) = {S:.1%}")
    logger.info(f"  Hard neg acc (vs same piece): {hard_neg_acc:.1%}")
    logger.info("=" * 60)

    return results


def eval_best_model(
    output_dir: Path,
    maestro_dir: Path,
    device: torch.device,
    seed: int = 42,
) -> Optional[dict]:
    """
    Load best_model.pt from output_dir and run structured pool evaluation.
    Returns None if best_model.pt doesn't exist.
    """
    best_path = output_dir / 'best_model.pt'
    if not best_path.exists():
        logger.warning(f"best_model.pt not found at {best_path}, skipping best eval")
        return None

    logger.info("Evaluating best_model.pt (best S during training)...")
    checkpoint = torch.load(best_path, map_location=device)
    arch_config = checkpoint['arch_config']
    mode = arch_config['mode']

    if mode in ('run-a', 'run-c'):
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        best_model = BloqueAModel(
            base_model=base_model,
            adapter_dim=arch_config.get('adapter_dim', 64),
            adapter_layers=arch_config.get('adapter_layers', []),
        )
        best_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif mode == 'run-b':
        best_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        best_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        logger.warning(f"Unknown mode '{mode}' in best_model.pt, skipping best eval")
        return None

    best_model = best_model.to(device)
    best_model.eval()

    results = run_structured_eval(
        model=best_model,
        maestro_dir=maestro_dir,
        device=device,
        seed=seed,
    )
    results['epoch'] = checkpoint.get('epoch', '?')
    results['source'] = 'best_model'
    return results


# ---------------------------------------------------------------------------
# Quick inline eval (during training, fast, small batch)
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

    audio_embs = []
    midi_embs = []

    for batch_idx, batch in enumerate(val_loader):
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

    # Simple retrieval metrics
    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)
    sim = audio_norm @ midi_norm.T
    N = sim.size(0)

    # A2M
    _, a2m_indices = sim.sort(dim=1, descending=True)
    gt = torch.arange(N)
    a2m_ranks = (a2m_indices == gt.unsqueeze(1)).nonzero()[:, 1].float()
    a2m_r10 = (a2m_ranks < 10).float().mean().item()

    # M2A
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

def train_loop(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LinearWarmupCosineScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    arch_config: dict,
    output_dir: Path,
    maestro_dir: Path,
    device: torch.device,
    mode: str,
    epochs: int = 5,
    max_batches_per_epoch: int = 1000,
    max_val_batches: int = 846,
    seed: int = 42,
    embed_batch_size: int = 16,
    start_epoch: int = 1,
) -> dict:
    """Main training loop for runs A/B/C.

    Runs canonical structured pool evaluation every epoch for decision metrics.
    Best model selected by structured S = min(A2M, M2A), not quick_val.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / 'eval_per_epoch'
    eval_dir.mkdir(exist_ok=True)

    # DriftSentinel
    sentinel = DriftSentinel(model)

    best_S_structured = 0.0
    best_epoch = 0
    history = []

    # Load best_S from existing evals (for resume)
    if start_epoch > 1:
        for ep in range(1, start_epoch):
            eval_path = eval_dir / f'eval_epoch{ep}.json'
            if eval_path.exists():
                prev = json.loads(eval_path.read_text())
                prev_S = prev.get('gate_metrics', {}).get('S', 0.0)
                if prev_S > best_S_structured:
                    best_S_structured = prev_S
                    best_epoch = ep
        logger.info(f"Resuming from epoch {start_epoch}, best so far: epoch {best_epoch} S={best_S_structured:.1%}")

    logger.info(f"Starting Bloque A training: mode={mode}, epochs={epochs}")
    logger.info(f"  Batches/epoch: {max_batches_per_epoch}, Val batches: {max_val_batches}")
    logger.info(f"  Canonical structured pool eval EVERY epoch (best by structured S)")
    total_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        total_loss = 0.0
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

            optimizer.zero_grad()

            audio_emb, midi_emb = model(
                audio, midi_pitch, midi_velocity, midi_duration, midi_mask
            )

            # VICReg loss
            if hasattr(model, 'compute_vicreg_loss'):
                loss, metrics = model.compute_vicreg_loss(audio_emb, midi_emb)
            else:
                loss, metrics = model.base_model.compute_vicreg_loss(audio_emb, midi_emb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += metrics['vicreg_loss']
            n_batches += 1

            if batch_idx % 100 == 0:
                lrs = scheduler.get_last_lr()
                pbar.set_postfix({
                    'loss': f"{metrics['vicreg_loss']:.4f}",
                    'std_z1': f"{metrics['std_z1']:.3f}",
                    'lr0': f"{lrs[0]:.1e}",
                })

        avg_loss = total_loss / max(n_batches, 1)
        train_time = (time.time() - epoch_start) / 60

        # --- DriftSentinel after epoch 1 ---
        if epoch == 1:
            sentinel.check(model)

        # --- Save checkpoint FIRST (resilience: never lose training to eval crash) ---
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_recall=best_S_structured,
            arch_config=arch_config,
            output_dir=output_dir,
            filename=f'checkpoint_epoch{epoch}.pt',
            mode=mode,
        )
        logger.info(f"  Checkpoint saved: checkpoint_epoch{epoch}.pt")

        # --- Quick val eval (fast monitoring for log) ---
        t_qv = time.time()
        logger.info(f"  [quick_val] Starting ({max_val_batches} batches)...")
        quick_metrics = quick_val_eval(model, val_loader, device, max_batches=max_val_batches)
        logger.info(f"  [quick_val] Done in {time.time() - t_qv:.1f}s")

        # --- Canonical structured pool eval (decision metric) ---
        logger.info(f"  [structured_eval] Starting canonical eval for epoch {epoch}...")
        structured_results = run_structured_eval(
            model=model,
            maestro_dir=maestro_dir,
            device=device,
            seed=seed,
            embed_batch_size=embed_batch_size,
        )

        epoch_time = (time.time() - epoch_start) / 60
        structured_S = structured_results['gate_metrics']['S']

        # --- Save per-epoch eval JSON ---
        epoch_eval = {
            'epoch': epoch,
            'mode': mode,
            **structured_results,
        }
        with open(eval_dir / f'eval_epoch{epoch}.json', 'w') as f:
            json.dump(epoch_eval, f, indent=2)

        # --- Track best by structured S ---
        if structured_S > best_S_structured:
            best_S_structured = structured_S
            best_epoch = epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_recall=best_S_structured,
                arch_config=arch_config,
                output_dir=output_dir,
                filename='best_model.pt',
                mode=mode,
            )
            logger.info(f"  >>> New best model: epoch {epoch}, structured S={structured_S:.1%}")

        epoch_record = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_time_min': round(train_time, 1),
            'epoch_time_min': round(epoch_time, 1),
            # Quick val (monitoring)
            **quick_metrics,
            # Canonical structured pool (decision)
            'structured_a2m_r10': structured_results['gate_metrics']['a2m_r10'],
            'structured_m2a_r10': structured_results['gate_metrics']['m2a_r10'],
            'structured_S': structured_S,
            'structured_hard_neg': structured_results['gate_metrics']['hard_neg'],
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch {epoch}/{epochs} ({epoch_time:.1f}min): "
            f"loss={avg_loss:.4f}, "
            f"quick[A2M={quick_metrics['val_a2m_r10']:.1%} M2A={quick_metrics['val_m2a_r10']:.1%}], "
            f"CANONICAL[A2M={structured_results['gate_metrics']['a2m_r10']:.1%} "
            f"M2A={structured_results['gate_metrics']['m2a_r10']:.1%} "
            f"S={structured_S:.1%} hard_neg={structured_results['gate_metrics']['hard_neg']:.1%}]"
        )

    total_time = (time.time() - total_start) / 60

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(
        f"Training complete: {total_time:.1f} minutes, "
        f"best structured S={best_S_structured:.1%} (epoch {best_epoch})"
    )

    return {
        'history': history,
        'best_recall': best_S_structured,
        'best_epoch': best_epoch,
        'training_time_minutes': round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# Mode: s0 (eval-only)
# ---------------------------------------------------------------------------

def run_s0(args):
    """Eval-only: reproduce Gate 2 baseline with structured pool."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODE: S0 (eval-only control)")
    logger.info("=" * 60)

    model = load_base_model(args.checkpoint, device)
    model = model.to(device)
    model.eval()

    results = run_structured_eval(
        model=model,
        maestro_dir=Path(args.maestro_dir),
        device=device,
        seed=args.seed,
        embed_batch_size=64,  # No optimizer in memory for eval-only modes
    )

    results['mode'] = 's0'
    results['checkpoint'] = args.checkpoint

    output_path = output_dir / 's0_eval.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"S0 results saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Mode: run-a
# ---------------------------------------------------------------------------

def run_a(args):
    """Adapter bottleneck on all 4 audio transformer layers."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    logger.info("=" * 60)
    logger.info("MODE: Run A (adapter bottleneck)")
    logger.info("=" * 60)

    base_model = load_base_model(args.checkpoint, device)
    model, arch_config = create_run_a_model(base_model, adapter_dim=args.adapter_dim)
    model = model.to(device)

    optimizer = create_optimizer_run_a(
        model, lr_adapter=args.lr_adapter, lr_midi=args.lr_midi, lr_proj=args.lr_proj,
    )

    # Resume from Bloque A checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        resume_ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(resume_ckpt['model_state_dict'], strict=True)
        logger.info(f"  Model state loaded (epoch {resume_ckpt.get('epoch', '?')})")

    # Preflight
    contract = get_preflight_contract('run-a', uses_wrapper=True)
    validate_training_setup(
        model, optimizer, mode='run-a', **contract,
    )

    # DataLoaders with deterministic seeding
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

    # Load optimizer/scheduler state for resume
    if args.resume_from:
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            logger.info("  Optimizer state restored")
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
            logger.info("  Scheduler state restored")

    # Save config
    config = {**vars(args), **arch_config}
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Train
    train_results = train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        arch_config=arch_config,
        output_dir=output_dir,
        maestro_dir=Path(args.maestro_dir),
        device=device,
        mode='run-a',
        epochs=args.epochs,
        max_batches_per_epoch=args.max_batches_per_epoch,
        max_val_batches=args.max_val_batches,
        seed=args.seed,
        embed_batch_size=args.embed_batch_size,
        start_epoch=args.start_epoch,
    )

    # Per-epoch canonical eval already done inside train_loop.
    # Final results reference the structured eval from best and last epoch.
    best_ep = train_results['best_epoch']
    best_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{best_ep}.json'
    last_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{args.epochs}.json'

    best_eval = json.loads(best_eval_path.read_text()) if best_eval_path.exists() else None
    last_eval = json.loads(last_eval_path.read_text()) if last_eval_path.exists() else None

    final = {
        'mode': 'run-a',
        'training': train_results,
        'evaluation_best': best_eval,
        'evaluation_final': last_eval,
    }
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final, f, indent=2)

    return final


# ---------------------------------------------------------------------------
# Mode: run-b
# ---------------------------------------------------------------------------

def run_b(args):
    """Partial unfreeze: audio transformer layers 2-3 trainable."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    logger.info("=" * 60)
    logger.info("MODE: Run B (partial unfreeze layers 2-3)")
    logger.info("=" * 60)

    base_model = load_base_model(args.checkpoint, device)
    model, arch_config = create_run_b_model(base_model)
    model = model.to(device)

    optimizer = create_optimizer_run_b(
        model, lr_audio_unfreeze=args.lr_audio_unfreeze,
        lr_midi=args.lr_midi, lr_proj=args.lr_proj,
    )

    # Preflight
    contract = get_preflight_contract('run-b', uses_wrapper=False)
    validate_training_setup(
        model, optimizer, mode='run-b', **contract,
    )

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

    total_steps = args.epochs * args.max_batches_per_epoch
    scheduler = LinearWarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps,
    )

    config = {**vars(args), **arch_config}
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    train_results = train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        arch_config=arch_config,
        output_dir=output_dir,
        maestro_dir=Path(args.maestro_dir),
        device=device,
        mode='run-b',
        epochs=args.epochs,
        max_batches_per_epoch=args.max_batches_per_epoch,
        max_val_batches=args.max_val_batches,
        seed=args.seed,
        embed_batch_size=args.embed_batch_size,
    )

    best_ep = train_results['best_epoch']
    best_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{best_ep}.json'
    last_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{args.epochs}.json'

    best_eval = json.loads(best_eval_path.read_text()) if best_eval_path.exists() else None
    last_eval = json.loads(last_eval_path.read_text()) if last_eval_path.exists() else None

    final = {
        'mode': 'run-b',
        'training': train_results,
        'evaluation_best': best_eval,
        'evaluation_final': last_eval,
    }
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final, f, indent=2)

    return final


# ---------------------------------------------------------------------------
# Mode: run-c
# ---------------------------------------------------------------------------

def run_c(args):
    """Hybrid: adapters on layers 0-1 + unfreeze layers 2-3."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    logger.info("=" * 60)
    logger.info("MODE: Run C (hybrid: adapters 0-1 + unfreeze 2-3)")
    logger.info("=" * 60)

    base_model = load_base_model(args.checkpoint, device)
    model, arch_config = create_run_c_model(base_model, adapter_dim=args.adapter_dim)
    model = model.to(device)

    optimizer = create_optimizer_run_c(
        model, lr_adapter=args.lr_adapter, lr_audio_unfreeze=args.lr_audio_unfreeze,
        lr_midi=args.lr_midi, lr_proj=args.lr_proj,
    )

    contract = get_preflight_contract('run-c', uses_wrapper=True)
    validate_training_setup(
        model, optimizer, mode='run-c', **contract,
    )

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

    total_steps = args.epochs * args.max_batches_per_epoch
    scheduler = LinearWarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps,
    )

    config = {**vars(args), **arch_config}
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    train_results = train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        arch_config=arch_config,
        output_dir=output_dir,
        maestro_dir=Path(args.maestro_dir),
        device=device,
        mode='run-c',
        epochs=args.epochs,
        max_batches_per_epoch=args.max_batches_per_epoch,
        max_val_batches=args.max_val_batches,
        seed=args.seed,
        embed_batch_size=args.embed_batch_size,
    )

    best_ep = train_results['best_epoch']
    best_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{best_ep}.json'
    last_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{args.epochs}.json'

    best_eval = json.loads(best_eval_path.read_text()) if best_eval_path.exists() else None
    last_eval = json.loads(last_eval_path.read_text()) if last_eval_path.exists() else None

    final = {
        'mode': 'run-c',
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
    """Standalone evaluation of a Bloque A checkpoint."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 60)
    logger.info("MODE: Evaluate")
    logger.info("=" * 60)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Safety gates
    arch_config = checkpoint.get('arch_config')
    if arch_config is None:
        raise RuntimeError(
            "Missing arch_config in checkpoint — cannot reconstruct architecture. "
            "This may be a Gate 2 or Gate 4 checkpoint, not a Bloque A checkpoint."
        )

    if not arch_config.get('eval_compatible', True):
        raise RuntimeError(
            "This is an archive checkpoint (eval_compatible=False), not for evaluation. "
            "Use the full checkpoint instead."
        )

    mode = arch_config['mode']
    logger.info(f"Checkpoint mode: {mode}, arch_config: {arch_config}")

    # Reconstruct model
    if mode in ('run-a', 'run-c'):
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = BloqueAModel(
            base_model=base_model,
            adapter_dim=arch_config.get('adapter_dim', 64),
            adapter_layers=arch_config.get('adapter_layers', []),
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif mode == 'run-b':
        model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        raise RuntimeError(f"Unknown mode in arch_config: {mode}")

    model = model.to(device)
    model.eval()

    results = run_structured_eval(
        model=model,
        maestro_dir=Path(args.maestro_dir),
        device=device,
        seed=args.seed,
        embed_batch_size=64,  # No optimizer in memory for eval-only modes
    )

    results['mode'] = mode
    results['checkpoint'] = args.checkpoint
    results['arch_config'] = arch_config
    results['epoch'] = checkpoint.get('epoch', '?')

    # Save
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
        description="Bloque A Training: Adapter / Unfreezing controlado (Plan v1.1)"
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['s0', 'run-a', 'run-b', 'run-c', 'evaluate'],
        help='Execution mode',
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to checkpoint (Gate 2 for training modes, Bloque A for evaluate)',
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory (training modes) or output file (evaluate mode)',
    )
    parser.add_argument(
        '--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0',
        help='Path to MAESTRO dataset',
    )

    # Training params
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-batches-per-epoch', type=int, default=1000)
    parser.add_argument('--max-val-batches', type=int, default=846)
    parser.add_argument('--embed-batch-size', type=int, default=16,
                        help='Batch size for embedding extraction during eval (16 for training, 64 for standalone)')
    parser.add_argument('--adapter-dim', type=int, default=64)

    # Learning rates
    parser.add_argument('--lr-adapter', type=float, default=5e-4)
    parser.add_argument('--lr-audio-unfreeze', type=float, default=1e-5)
    parser.add_argument('--lr-midi', type=float, default=5e-5)
    parser.add_argument('--lr-proj', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=200)

    # Misc
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to Bloque A checkpoint to resume training (loads model+optimizer+scheduler)')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='Epoch to start from (use with --resume-from to skip completed epochs)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.mode == 's0':
        run_s0(args)
    elif args.mode == 'run-a':
        run_a(args)
    elif args.mode == 'run-b':
        run_b(args)
    elif args.mode == 'run-c':
        run_c(args)
    elif args.mode == 'evaluate':
        run_evaluate(args)


if __name__ == '__main__':
    main()
