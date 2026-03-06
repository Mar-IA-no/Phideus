#!/usr/bin/env python3
"""
Gate 7.1a — MERT-330M Frozen Cross-Modal Probe (D0 Pilot)

Question: Does S(D0) rise with MERT-330M frozen vs MERTLite (75.2%)?

Architecture:
    Audio  -> MERT-330M (frozen) -> Projection -> z_audio [B, 256]
    MIDI   -> Transformer (from scratch) -> Projection -> z_midi [B, 256]
    Loss   = VICReg(z_audio, z_midi)

Key differences from gate43_scratch:
    - Audio encoder = MERTEncoder (HuggingFace MERT-330M), NOT MERTEncoderLite
    - Audio encoder fully frozen (no fine-tuning at all)
    - MIDI encoder + projections trained from scratch
    - Anti-ghost checks specific to frozen encoder
    - Throughput benchmark in first epoch

Usage:
    python experiments/bias_control/gate71/train_gate71.py \
        --descriptor d0 \
        --epochs 30 --batch-size 8 --seed 42 \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/gate71_results/d0_mert330m_seed42 \
        --structured-eval-epochs 5 10 15 20 25 28 29 30
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.bias_control.architectures.cross_modal_model import CrossModalModel
from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
)
from src.bias_control.encoders.mert_encoder import MERTEncoder
from src.bias_control.training.preflight import DriftSentinel

# Eval imports (same as gate43)
from experiments.bias_control.evaluate_structured_pool import (
    extract_all_embeddings,
    evaluate_with_precomputed_embeddings,
    build_segment_index,
    PoolConfig,
    analyze_hard_negatives_fast,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


class LinearWarmupCosineScheduler:
    """Linear warmup then cosine annealing."""

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

    @property
    def lr_mult(self) -> float:
        if not self.base_lrs or self.base_lrs[0] == 0:
            return 0.0
        return self.optimizer.param_groups[0]['lr'] / self.base_lrs[0]

    def state_dict(self):
        return {'step_count': self.step_count, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']


# ---------------------------------------------------------------------------
# Anti-ghost checks for frozen MERT encoder
# ---------------------------------------------------------------------------

def run_antighost_checks(model: nn.Module, base_model: CrossModalModel):
    """Validate parameter counts and frozen state after model construction."""
    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen_audio = sum(p.numel() for p in base_model.audio_encoder.parameters())

    logger.info("=" * 60)
    logger.info("ANTI-GHOST CHECKS")
    logger.info("=" * 60)
    logger.info(f"  Total params:     {total:>12,}")
    logger.info(f"  Trainable params: {trainable:>12,}")
    logger.info(f"  Frozen audio:     {frozen_audio:>12,}")
    logger.info(f"  Trainable ratio:  {trainable/total:.1%}")

    # Check 1: Audio encoder should have ~330M params
    assert frozen_audio > 300_000_000, (
        f"MERT-330M should have >300M params, got {frozen_audio:,}"
    )

    # Check 2: All audio encoder params should be frozen
    audio_trainable = sum(
        p.numel() for p in base_model.audio_encoder.parameters() if p.requires_grad
    )
    assert audio_trainable == 0, (
        f"Audio encoder has {audio_trainable:,} trainable params, expected 0"
    )

    # Check 3: Trainable should be ~15M (MIDI + projections)
    assert 10_000_000 < trainable < 25_000_000, (
        f"Trainable params {trainable:,} outside expected range [10M, 25M]"
    )

    # Check 4: MERT HF model should be in eval mode
    assert not base_model.audio_encoder._model.training, (
        "MERT HF model should be in eval mode after construction"
    )

    logger.info("  All anti-ghost checks PASSED")
    logger.info("=" * 60)

    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_audio_params': frozen_audio,
    }


def check_train_mode_leak(base_model: CrossModalModel):
    """Verify MERT stays in eval after model.train() call."""
    assert not base_model.audio_encoder._model.training, (
        "LEAK DETECTED: MERT HF model switched to training mode after model.train()"
    )


def check_weight_drift(
    snapshot: Dict[str, torch.Tensor],
    model: nn.Module,
    prefix_filter: str = 'base_model.audio_encoder',
):
    """Verify frozen weights haven't changed after training."""
    drifted = []
    for name, initial in snapshot.items():
        for n, p in model.named_parameters():
            if n == name:
                if not torch.equal(initial, p.detach().cpu()):
                    drifted.append(name)
                break

    if drifted:
        raise RuntimeError(
            f"FROZEN WEIGHT DRIFT: {len(drifted)} audio encoder params changed: "
            f"{drifted[:5]}..."
        )
    logger.info(f"  Weight drift check PASSED ({len(snapshot)} params unchanged)")


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def benchmark_throughput(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 4,
    measure_batches: int = 20,
) -> Dict:
    """Measure throughput after warmup (lazy load, JIT, cache)."""
    model.train()
    times = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= warmup_batches + measure_batches:
            break

        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        torch.cuda.synchronize()
        t0 = time.time()

        z_audio, z_midi = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
        loss, _ = model.base_model.compute_vicreg_loss(z_audio, z_midi)
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.time()

        if batch_idx >= warmup_batches:
            times.append(t1 - t0)

    if not times:
        return {'error': 'No measurement batches completed'}

    avg_time = np.mean(times)
    std_time = np.std(times)
    batches_per_min = 60.0 / avg_time
    gpu_mem_peak = torch.cuda.max_memory_allocated() / 1e9

    result = {
        'avg_batch_time_s': round(avg_time, 3),
        'std_batch_time_s': round(std_time, 3),
        'batches_per_min': round(batches_per_min, 1),
        'gpu_mem_peak_gb': round(gpu_mem_peak, 2),
        'measured_batches': len(times),
    }

    # Extrapolate: 1000 batches/epoch * 30 epochs
    time_per_epoch_min = (1000 * avg_time) / 60
    time_30ep_h = (time_per_epoch_min * 30) / 60

    result['est_time_per_epoch_min'] = round(time_per_epoch_min, 1)
    result['est_time_30ep_hours'] = round(time_30ep_h, 1)

    logger.info("=" * 60)
    logger.info("THROUGHPUT BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"  Avg batch time: {avg_time:.3f}s ± {std_time:.3f}s")
    logger.info(f"  Batches/min:    {batches_per_min:.1f}")
    logger.info(f"  GPU mem peak:   {gpu_mem_peak:.2f} GB")
    logger.info(f"  Est. per epoch: {time_per_epoch_min:.1f} min (1000 batches)")
    logger.info(f"  Est. 30 epochs: {time_30ep_h:.1f} hours")
    logger.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# Quick val eval (same pattern as gate43)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_val_eval(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    """Fast validation eval on a subset of val data."""
    model.eval()
    audio_embs, midi_embs = [], []

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
# Structured eval (canonical, same as gate43)
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
    """Run structured pool evaluation (canonical S metric)."""
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
    a2m = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, val_dataset, index, config,
        direction='a2m', seed=seed,
    )
    logger.info("  [eval] A2M done in %.1fs — R@10=%.1f%%",
                time.time() - t0, a2m['mean_recall@10'] * 100)

    t0 = time.time()
    m2a = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, val_dataset, index, config,
        direction='m2a', seed=seed,
    )
    logger.info("  [eval] M2A done in %.1fs — R@10=%.1f%%",
                time.time() - t0, m2a['mean_recall@10'] * 100)

    t0 = time.time()
    hard_neg = analyze_hard_negatives_fast(
        audio_embs, midi_embs, val_dataset, index, n_samples=500, seed=seed,
    )
    logger.info("  [eval] Hard neg done in %.1fs — acc=%.1f%%",
                time.time() - t0, hard_neg['accuracy_vs_same_piece'] * 100)

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
            'seed': seed,
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
# Checkpoint save (adapted from gate43)
# ---------------------------------------------------------------------------

def save_checkpoint(
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
    """Save checkpoint with full state for resume."""
    path = output_dir / filename
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
        'best_S': best_S,
    }, path)
    logger.info(f"  Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# Gate71Model — thin wrapper for D0 with MERT-330M
# ---------------------------------------------------------------------------

class Gate71Model(nn.Module):
    """D0 model with MERT-330M frozen encoder.

    Wraps CrossModalModel so that forward() returns (audio_emb, midi_emb)
    compatible with the eval harness.
    """

    def __init__(self, base_model: CrossModalModel):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)

    def compute_total_loss(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        audio_emb, midi_emb = self.forward(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask,
        )
        loss, metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)
        metrics['total_loss'] = loss.item()
        metrics['ratio_aux_loss'] = 0.0
        return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(
    model: Gate71Model,
    optimizer: AdamW,
    scheduler: LinearWarmupCosineScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    descriptor: str,
    arch_config: dict,
    output_dir: Path,
    maestro_dir: Path,
    device: torch.device,
    epochs: int = 30,
    max_batches_per_epoch: int = 1000,
    seed: int = 42,
    embed_batch_size: int = 16,
    structured_eval_epochs: Optional[List[int]] = None,
) -> dict:
    """Training loop for Gate 7.1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / 'eval_per_epoch'
    eval_dir.mkdir(exist_ok=True)

    # Snapshot frozen weights for drift check
    frozen_snapshot = {}
    for name, param in model.named_parameters():
        if 'audio_encoder' in name and not param.requires_grad:
            frozen_snapshot[name] = param.detach().cpu().clone()
    logger.info(f"Frozen weight snapshot: {len(frozen_snapshot)} params")

    sentinel = DriftSentinel(model)
    best_S = 0.0
    best_epoch = 0
    history = []
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()

        # Verify MERT stays in eval after model.train()
        check_train_mode_leak(model.base_model)

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}",
                     total=max_batches_per_epoch)
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches_per_epoch:
                break

            audio = batch['audio'].to(device, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
            midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
            midi_duration = batch['midi_duration'].to(device, non_blocking=True)
            midi_mask = batch['midi_mask'].to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, metrics = model.compute_total_loss(
                audio, midi_pitch, midi_velocity, midi_duration, midi_mask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += metrics['vicreg_loss']
            n_batches += 1

            lrs = scheduler.get_last_lr()
            pbar.set_postfix({
                'loss': f"{metrics['vicreg_loss']:.4f}",
                'avg': f"{total_loss / n_batches:.4f}",
                'std_z1': f"{metrics['std_z1']:.3f}",
                'lr': f"{lrs[0]:.1e}",
            })

        avg_loss = total_loss / max(n_batches, 1)
        train_time = (time.time() - epoch_start) / 60

        # --- DriftSentinel after epoch 1 ---
        if epoch == 1:
            sentinel.check(model)
            # Also check frozen weights didn't change
            check_weight_drift(frozen_snapshot, model)

        # --- Save checkpoint ---
        save_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_S=best_S, descriptor=descriptor,
            arch_config=arch_config, output_dir=output_dir,
            filename=f'checkpoint_epoch{epoch}.pt',
        )

        # --- Quick val ---
        t_qv = time.time()
        quick_metrics = quick_val_eval(model, val_loader, device, max_batches=50)
        logger.info(f"  [quick_val] S={quick_metrics['val_S']:.1%}, "
                     f"gap={quick_metrics['val_gap']:.4f} "
                     f"({time.time() - t_qv:.1f}s)")

        # --- Structured eval (selective) ---
        structured_S = 0.0
        structured_results = None
        do_structured = (structured_eval_epochs is not None and
                         epoch in structured_eval_epochs)

        if do_structured:
            logger.info(f"  [structured_eval] Running canonical eval for epoch {epoch}...")
            structured_results = run_structured_eval(
                model=model, maestro_dir=maestro_dir, device=device,
                seed=seed, embed_batch_size=embed_batch_size,
            )
            structured_S = structured_results['gate_metrics']['S']

            # Save eval JSON
            epoch_eval = {'epoch': epoch, 'descriptor': descriptor, **structured_results}
            with open(eval_dir / f'eval_epoch{epoch}.json', 'w') as f:
                json.dump(epoch_eval, f, indent=2)

            if structured_S > best_S:
                best_S = structured_S
                best_epoch = epoch
                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, best_S=best_S, descriptor=descriptor,
                    arch_config=arch_config, output_dir=output_dir,
                    filename='best_model.pt',
                )
                logger.info(f"  >>> New best model: epoch {epoch}, S={structured_S:.1%}")

        epoch_time = (time.time() - epoch_start) / 60

        epoch_record = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'lr_mult': round(scheduler.lr_mult, 6),
            'train_time_min': round(train_time, 1),
            'epoch_time_min': round(epoch_time, 1),
            **quick_metrics,
        }
        if structured_results:
            epoch_record.update({
                'structured_S': structured_S,
                'structured_a2m_r10': structured_results['gate_metrics']['a2m_r10'],
                'structured_m2a_r10': structured_results['gate_metrics']['m2a_r10'],
                'structured_hard_neg': structured_results['gate_metrics']['hard_neg'],
            })
        history.append(epoch_record)

        logger.info(
            f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, "
            f"val_S={quick_metrics['val_S']:.1%}, "
            f"time={epoch_time:.1f}min"
            + (f", structured_S={structured_S:.1%}" if structured_results else "")
        )

    total_time = (time.time() - total_start) / 60

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final frozen weight check
    check_weight_drift(frozen_snapshot, model)

    return {
        'history': history,
        'best_S': best_S,
        'best_epoch': best_epoch,
        'training_time_minutes': round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Gate 7.1a: MERT-330M Frozen Cross-Modal Probe')

    # Required
    parser.add_argument('--maestro-dir', type=str, required=True,
                        help='Path to MAESTRO v3.0.0 directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for checkpoints and results')

    # Model
    parser.add_argument('--descriptor', type=str, default='d0',
                        choices=['d0'],
                        help='Descriptor type (d0 for Phase 7.1a)')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-batches-per-epoch', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)

    # LR
    parser.add_argument('--lr-midi', type=float, default=5e-5)
    parser.add_argument('--lr-proj', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=500)

    # Eval
    parser.add_argument('--structured-eval-epochs', type=int, nargs='+',
                        default=[5, 10, 15, 20, 25, 28, 29, 30])
    parser.add_argument('--embed-batch-size', type=int, default=16)

    # Benchmark
    parser.add_argument('--skip-benchmark', action='store_true',
                        help='Skip throughput benchmark')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Run only throughput benchmark then exit')

    args = parser.parse_args()

    # Setup
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    maestro_dir = Path(args.maestro_dir)

    logger.info("=" * 70)
    logger.info("GATE 7.1a — MERT-330M FROZEN CROSS-MODAL PROBE (D0)")
    logger.info("=" * 70)
    logger.info(f"  Device: {device}")
    logger.info(f"  MAESTRO: {maestro_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Descriptor: {args.descriptor}")
    logger.info(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    logger.info(f"  Max batches/epoch: {args.max_batches_per_epoch}")
    logger.info(f"  LR: midi={args.lr_midi}, proj={args.lr_proj}")
    logger.info(f"  Structured eval epochs: {args.structured_eval_epochs}")
    logger.info("=" * 70)

    # --- Build model ---
    logger.info("Building CrossModalModel with MERT-330M frozen encoder...")
    base_model = CrossModalModel(
        audio_encoder='mert',
        audio_encoder_frozen=True,
        use_dann=False,
    )

    # Force load MERT model (lazy load workaround)
    logger.info("Force-loading MERT-330M from HuggingFace...")
    base_model.audio_encoder._load_model()
    assert base_model.audio_encoder._loaded, "MERT model failed to load"

    model = Gate71Model(base_model)
    model = model.to(device)

    # Anti-ghost checks
    param_info = run_antighost_checks(model, base_model)

    # Save args + param info
    run_config = {
        'args': vars(args),
        'param_info': param_info,
        'device': str(device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    # --- Build arch_config for checkpoints ---
    arch_config = {
        'gate': '7.1',
        'descriptor': args.descriptor,
        'audio_encoder': 'mert',
        'audio_encoder_frozen': True,
        'from_scratch': True,
        'eval_compatible': True,
    }

    # --- Optimizer (exclude audio encoder entirely) ---
    param_groups = [
        {
            'params': list(base_model.midi_encoder.parameters()),
            'lr': args.lr_midi,
            'name': 'midi_encoder',
        },
        {
            'params': (
                list(base_model.audio_projection.parameters()) +
                list(base_model.midi_projection.parameters())
            ),
            'lr': args.lr_proj,
            'name': 'projections',
        },
    ]
    optimizer = AdamW(param_groups, weight_decay=0.01)

    # Verify optimizer doesn't include frozen params
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer_param_ids.add(id(p))

    for name, param in model.named_parameters():
        if not param.requires_grad and id(param) in optimizer_param_ids:
            raise RuntimeError(f"Frozen param '{name}' found in optimizer!")
        if param.requires_grad and id(param) not in optimizer_param_ids:
            raise RuntimeError(f"Trainable param '{name}' NOT in optimizer!")

    # Log optimizer groups
    for gi, group in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in group['params'])
        logger.info(f"  Optimizer [{gi}] {group['name']}: {n_params:,} params, lr={group['lr']:.1e}")

    # --- Scheduler ---
    total_steps = args.epochs * args.max_batches_per_epoch
    scheduler = LinearWarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps,
    )

    # --- Data ---
    logger.info("Loading MAESTRO datasets...")
    train_dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir, segment_len=8.0, hop=2.0,
        sample_rate=24000, split='train',
    )
    val_dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir, segment_len=8.0, hop=2.0,
        sample_rate=24000, split='validation',
    )

    dl_kwargs = {'pin_memory': True, 'prefetch_factor': 2} if args.num_workers > 0 else {}
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_segments,
        drop_last=True, worker_init_fn=seed_worker, **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_segments,
        **dl_kwargs,
    )

    logger.info(f"  Train: {len(train_dataset)} segments, {len(train_loader)} batches/epoch")
    logger.info(f"  Val: {len(val_dataset)} segments")

    # --- Throughput benchmark ---
    if not args.skip_benchmark:
        logger.info("Running throughput benchmark...")
        throughput = benchmark_throughput(model, train_loader, device)
        run_config['throughput'] = throughput
        with open(output_dir / 'run_config.json', 'w') as f:
            json.dump(run_config, f, indent=2)

        if args.benchmark_only:
            logger.info("Benchmark complete. Exiting (--benchmark-only).")
            return

        # Reset optimizer state after benchmark
        optimizer.zero_grad(set_to_none=True)
        scheduler.step_count = 0
        for pg, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            pg['lr'] = base_lr * 0  # Will be set by first scheduler.step()

    # --- Train ---
    result = train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        descriptor=args.descriptor,
        arch_config=arch_config,
        output_dir=output_dir,
        maestro_dir=maestro_dir,
        device=device,
        epochs=args.epochs,
        max_batches_per_epoch=args.max_batches_per_epoch,
        seed=args.seed,
        embed_batch_size=args.embed_batch_size,
        structured_eval_epochs=args.structured_eval_epochs,
    )

    logger.info("=" * 70)
    logger.info("GATE 7.1a TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Best S: {result['best_S']:.1%} @ epoch {result['best_epoch']}")
    logger.info(f"  Total time: {result['training_time_minutes']:.1f} min")
    logger.info("=" * 70)

    # Save final summary
    run_config['result'] = result
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)


if __name__ == '__main__':
    main()
