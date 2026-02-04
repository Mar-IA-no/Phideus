#!/usr/bin/env python3
"""
Gate 2: Cross-Modal Foundation Baseline

Objectives:
1. Train cross-modal model with VICReg (no DANN yet)
2. Establish baseline cross-modal retrieval
3. Verify no collapse (variance check)

Criteria GO:
- Audio→MIDI Recall@10: > 20%
- MIDI→Audio Recall@10: > 20%
- vs Random: > 5x

Usage:
    python experiments/bias_control/gate2_foundation.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/training_outputs/bias_control/gate2 \
        --epochs 100 --batch-size 64
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
    create_dataloaders,
)
from src.bias_control.architectures.cross_modal_model import (
    CrossModalModel,
    compute_retrieval_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for cross-modal model."""

    def __init__(
        self,
        model: CrossModalModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        lr_projection: float = 1e-3,
        lr_midi_encoder: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        max_epochs: int = 100,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps

        # Separate parameter groups
        midi_params = list(model.midi_encoder.parameters())
        proj_params = (
            list(model.audio_projection.parameters()) +
            list(model.midi_projection.parameters())
        )

        self.optimizer = AdamW([
            {'params': midi_params, 'lr': lr_midi_encoder},
            {'params': proj_params, 'lr': lr_projection},
        ], weight_decay=weight_decay)

        # Scheduler
        total_steps = max_epochs * len(train_loader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
        )

        self.global_step = 0
        self.best_recall = 0.0
        self.history = []

    def warmup_lr(self, step: int) -> float:
        """Linear warmup."""
        if step < self.warmup_steps:
            return step / self.warmup_steps
        return 1.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        n_batches = 0
        metrics_sum = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")

        for batch in pbar:
            # Move to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Warmup learning rate
            if self.global_step < self.warmup_steps:
                lr_scale = self.warmup_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * lr_scale

            # Forward
            self.optimizer.zero_grad()
            loss, metrics = self.model.compute_loss(batch)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.global_step >= self.warmup_steps:
                self.scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            n_batches += 1
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'std': f"{metrics.get('std_z1', 0):.3f}",
            })

        # Average metrics
        avg_metrics = {
            'train_loss': total_loss / n_batches,
            **{f'train_{k}': v / n_batches for k, v in metrics_sum.items()}
        }

        return avg_metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        audio_embeddings = []
        midi_embeddings = []
        piece_indices = []

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            audio_emb, midi_emb = self.model.get_embeddings(batch)
            audio_embeddings.append(audio_emb.cpu())
            midi_embeddings.append(midi_emb.cpu())
            piece_indices.append(batch['piece_idx'])

        audio_embeddings = torch.cat(audio_embeddings, dim=0)
        midi_embeddings = torch.cat(midi_embeddings, dim=0)
        piece_indices = torch.cat(piece_indices, dim=0)

        # Compute retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(
            audio_embeddings, midi_embeddings,
            k_values=(1, 5, 10, 20),
        )

        # Check for collapse
        std_audio = audio_embeddings.std(dim=0).mean().item()
        std_midi = midi_embeddings.std(dim=0).mean().item()

        metrics = {
            **{f'val_{k}': v for k, v in retrieval_metrics.items()},
            'val_std_audio': std_audio,
            'val_std_midi': std_midi,
            'val_no_collapse': std_audio > 0.1 and std_midi > 0.1,
        }

        return metrics

    def train(self) -> Dict:
        """Full training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        # Store initial LRs
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        start_time = time.time()

        for epoch in range(self.max_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            val_metrics = self.evaluate()

            # Combine metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_metrics)

            # Log
            logger.info(
                f"Epoch {epoch+1}: "
                f"loss={train_metrics['train_loss']:.4f}, "
                f"a2m_r@10={val_metrics['val_a2m_recall@10']:.3f}, "
                f"m2a_r@10={val_metrics['val_m2a_recall@10']:.3f}, "
                f"gap={val_metrics['val_gap']:.3f}"
            )

            # Save best model
            recall_avg = (
                val_metrics['val_a2m_recall@10'] +
                val_metrics['val_m2a_recall@10']
            ) / 2

            if recall_avg > self.best_recall:
                self.best_recall = recall_avg
                self.save_checkpoint('best_model.pt')
                logger.info(f"New best model: recall@10={recall_avg:.3f}")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pt')

        # Save final model
        self.save_checkpoint('final_model.pt')

        # Save history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")

        return {
            'best_recall': self.best_recall,
            'final_metrics': self.history[-1],
            'training_time_minutes': total_time / 60,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_recall': self.best_recall,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_recall = checkpoint.get('best_recall', 0.0)


def run_gate2(
    maestro_dir: Path,
    output_dir: Path,
    segment_len: float = 8.0,
    hop: float = 2.0,
    batch_size: int = 64,
    num_workers: int = 8,
    epochs: int = 100,
    lr_projection: float = 1e-3,
    lr_midi_encoder: float = 1e-4,
    use_mert_lite: bool = True,
    device: Optional[str] = None,
) -> Dict:
    """
    Run Gate 2: Cross-Modal Foundation Baseline.

    Returns:
        Results dict with GO/NO-GO decision
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'gate': 2,
        'name': 'Cross-Modal Foundation Baseline',
        'config': {
            'segment_len': segment_len,
            'hop': hop,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr_projection': lr_projection,
            'lr_midi_encoder': lr_midi_encoder,
            'use_mert_lite': use_mert_lite,
            'device': device,
        },
    }

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = CrossModalModel(
        audio_encoder='lite' if use_mert_lite else 'mert',
        audio_encoder_frozen=True,
        use_dann=False,  # No DANN in Gate 2
        device=device,
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        lr_projection=lr_projection,
        lr_midi_encoder=lr_midi_encoder,
        max_epochs=epochs,
        device=device,
    )

    training_results = trainer.train()
    results['training'] = training_results

    # Final evaluation
    logger.info("Final evaluation...")
    final_metrics = trainer.evaluate()

    # Compute vs random
    a2m_recall = final_metrics['val_a2m_recall@10']
    m2a_recall = final_metrics['val_m2a_recall@10']
    n_samples = len(val_loader.dataset)
    random_recall = 10 / n_samples  # Recall@10 for random

    vs_random_a2m = a2m_recall / random_recall if random_recall > 0 else 0
    vs_random_m2a = m2a_recall / random_recall if random_recall > 0 else 0

    results['retrieval'] = {
        'a2m_recall@10': a2m_recall,
        'm2a_recall@10': m2a_recall,
        'random_baseline': random_recall,
        'vs_random_a2m': vs_random_a2m,
        'vs_random_m2a': vs_random_m2a,
        'gap': final_metrics['val_gap'],
        'threshold_recall': 0.20,
        'threshold_vs_random': 5.0,
        'pass_recall': a2m_recall >= 0.20 and m2a_recall >= 0.20,
        'pass_vs_random': vs_random_a2m >= 5.0 and vs_random_m2a >= 5.0,
    }

    # Collapse check
    results['collapse_check'] = {
        'std_audio': final_metrics['val_std_audio'],
        'std_midi': final_metrics['val_std_midi'],
        'threshold': 0.1,
        'pass': final_metrics['val_no_collapse'],
    }

    # Overall decision
    all_pass = (
        results['retrieval']['pass_recall'] and
        results['retrieval']['pass_vs_random'] and
        results['collapse_check']['pass']
    )

    # Strong GO check (for skipping Gate 3)
    strong_go = (
        a2m_recall >= 0.40 and
        m2a_recall >= 0.40 and
        vs_random_a2m >= 10.0 and
        vs_random_m2a >= 10.0
    )

    results['decision'] = 'STRONG GO' if strong_go else ('GO' if all_pass else 'NO-GO')
    results['status'] = 'PASS' if all_pass else 'FAIL'
    results['skip_gate3'] = strong_go

    return results


def main():
    parser = argparse.ArgumentParser(description="Gate 2: Cross-Modal Foundation")
    parser.add_argument(
        '--maestro-dir',
        type=str,
        required=True,
        help='Path to MAESTRO dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/training_outputs/bias_control/gate2',
        help='Output directory'
    )
    parser.add_argument(
        '--segment-len',
        type=float,
        default=8.0,
        help='Segment length in seconds'
    )
    parser.add_argument(
        '--hop',
        type=float,
        default=2.0,
        help='Hop between segments'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='DataLoader workers'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs'
    )
    parser.add_argument(
        '--lr-projection',
        type=float,
        default=1e-3,
        help='Learning rate for projection heads'
    )
    parser.add_argument(
        '--lr-midi',
        type=float,
        default=1e-4,
        help='Learning rate for MIDI encoder'
    )
    parser.add_argument(
        '--use-full-mert',
        action='store_true',
        help='Use full MERT model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu)'
    )

    args = parser.parse_args()

    maestro_dir = Path(args.maestro_dir)
    output_dir = Path(args.output)

    if not maestro_dir.exists():
        logger.error(f"MAESTRO directory not found: {maestro_dir}")
        sys.exit(1)

    # Run Gate 2
    results = run_gate2(
        maestro_dir=maestro_dir,
        output_dir=output_dir,
        segment_len=args.segment_len,
        hop=args.hop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr_projection=args.lr_projection,
        lr_midi_encoder=args.lr_midi,
        use_mert_lite=not args.use_full_mert,
        device=args.device,
    )

    # Save results
    results_path = output_dir / "gate2_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GATE 2: CROSS-MODAL FOUNDATION - RESULTS")
    print("=" * 60)

    print(f"\n1. Cross-Modal Retrieval:")
    print(f"   Audio→MIDI Recall@10: {results['retrieval']['a2m_recall@10']:.1%} (threshold: 20%)")
    print(f"   MIDI→Audio Recall@10: {results['retrieval']['m2a_recall@10']:.1%} (threshold: 20%)")
    print(f"   vs Random (A→M): {results['retrieval']['vs_random_a2m']:.1f}x (threshold: 5x)")
    print(f"   vs Random (M→A): {results['retrieval']['vs_random_m2a']:.1f}x (threshold: 5x)")
    print(f"   Gap: {results['retrieval']['gap']:.3f}")
    print(f"   Status: {'PASS' if results['retrieval']['pass_recall'] and results['retrieval']['pass_vs_random'] else 'FAIL'}")

    print(f"\n2. Collapse Check:")
    print(f"   Std Audio: {results['collapse_check']['std_audio']:.3f}")
    print(f"   Std MIDI: {results['collapse_check']['std_midi']:.3f}")
    print(f"   Status: {'PASS' if results['collapse_check']['pass'] else 'FAIL (COLLAPSED!)'}")

    print(f"\n3. Training:")
    print(f"   Best Recall: {results['training']['best_recall']:.3f}")
    print(f"   Time: {results['training']['training_time_minutes']:.1f} minutes")

    print(f"\n" + "=" * 60)
    print(f"DECISION: {results['decision']}")
    if results.get('skip_gate3'):
        print("(STRONG GO - Can skip Gate 3)")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")

    return 0 if results['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
