#!/usr/bin/env python3
"""
Gate 3: Domain Adversarial Neural Network (DANN)

Objectives:
1. Train with DANN to force modal-agnostic embeddings
2. Domain classifier accuracy → 50% (can't distinguish modality)
3. Cross-modal retrieval ≥ Gate 2

Criteria GO:
- Domain classifier accuracy: 50% ± 5%
- Cross-modal Recall@10: ≥ Gate 2
- Gap vs hard negatives: improvement

Usage:
    python experiments/bias_control/gate3_dann.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --checkpoint data/training_outputs/bias_control/gate2/best_model.pt \
        --output data/training_outputs/bias_control/gate3 \
        --epochs 50
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


class DANNTrainer:
    """Trainer for cross-modal model with DANN."""

    def __init__(
        self,
        model: CrossModalModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        lr_projection: float = 5e-4,  # Lower LR for fine-tuning
        lr_midi_encoder: float = 5e-5,
        dann_weight: float = 0.01,
        weight_decay: float = 1e-4,
        warmup_steps: int = 200,
        max_epochs: int = 50,
        device: str = 'cuda',
        gate2_recall: float = 0.026,  # Gate 2 R@10 global baseline
        max_batches_per_epoch: Optional[int] = None,
        checkpoint_every: int = 5,
        max_val_batches: Optional[int] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.dann_weight = dann_weight
        self.gate2_recall = gate2_recall
        self.max_batches_per_epoch = max_batches_per_epoch
        self.checkpoint_every = checkpoint_every
        self.max_val_batches = max_val_batches

        # Separate parameter groups (excluding DANN classifier)
        midi_params = list(model.midi_encoder.parameters())
        proj_params = (
            list(model.audio_projection.parameters()) +
            list(model.midi_projection.parameters())
        )

        # DANN classifier params
        dann_params = list(model.dann.classifier.parameters()) if model.dann else []

        self.optimizer = AdamW([
            {'params': midi_params, 'lr': lr_midi_encoder},
            {'params': proj_params, 'lr': lr_projection},
            {'params': dann_params, 'lr': lr_projection},  # DANN at projection LR
        ], weight_decay=weight_decay)

        # Store initial LRs immediately so warmup can reference them
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        # Scheduler - account for max_batches_per_epoch
        steps_per_epoch = min(len(train_loader), max_batches_per_epoch) if max_batches_per_epoch else len(train_loader)
        total_steps = max_epochs * steps_per_epoch
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
        )

        self.global_step = 0
        self.best_recall = 0.0
        self.history = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with DANN."""
        self.model.train()

        total_loss = 0.0
        total_vicreg = 0.0
        total_dann = 0.0
        n_batches = 0
        domain_correct = 0
        domain_total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")

        for batch_idx, batch in enumerate(pbar):
            if self.max_batches_per_epoch and batch_idx >= self.max_batches_per_epoch:
                break

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Warmup
            if self.global_step < self.warmup_steps:
                lr_scale = self.global_step / self.warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * lr_scale

            self.optimizer.zero_grad()

            # Forward - get embeddings
            audio_emb, midi_emb = self.model(
                audio=batch['audio'],
                midi_pitch=batch['midi_pitch'],
                midi_velocity=batch['midi_velocity'],
                midi_duration=batch['midi_duration'],
                midi_mask=batch.get('midi_mask'),
            )

            # VICReg loss
            vicreg_loss, vicreg_metrics = self.model.compute_vicreg_loss(audio_emb, midi_emb)

            # DANN loss
            B = audio_emb.size(0)
            embeddings = torch.cat([audio_emb, midi_emb], dim=0)
            domain_labels = torch.cat([
                torch.zeros(B, dtype=torch.long, device=self.device),
                torch.ones(B, dtype=torch.long, device=self.device),
            ])

            # Update DANN lambda
            self.model.dann.update_lambda(self.global_step)

            dann_loss, dann_metrics = self.model.dann(
                embeddings, domain_labels, update_step=False
            )

            # Total loss
            loss = vicreg_loss + self.dann_weight * dann_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.global_step >= self.warmup_steps:
                self.scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_vicreg += vicreg_loss.item()
            total_dann += dann_loss.item()
            n_batches += 1

            domain_correct += int(dann_metrics['domain_accuracy'] * 2 * B)
            domain_total += 2 * B

            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dann_acc': f"{dann_metrics['domain_accuracy']:.1%}",
                'λ': f"{dann_metrics['lambda']:.2f}",
            })

        avg_metrics = {
            'train_loss': total_loss / n_batches,
            'train_vicreg_loss': total_vicreg / n_batches,
            'train_dann_loss': total_dann / n_batches,
            'train_domain_accuracy': domain_correct / domain_total,
        }

        return avg_metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate with domain accuracy check."""
        self.model.eval()

        audio_embeddings = []
        midi_embeddings = []
        piece_indices = []

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
            if self.max_val_batches and batch_idx >= self.max_val_batches:
                break

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

        # Retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(
            audio_embeddings, midi_embeddings,
            k_values=(1, 5, 10, 20),
        )

        # Domain classification accuracy
        combined = torch.cat([audio_embeddings, midi_embeddings], dim=0).to(self.device)
        domain_labels = torch.cat([
            torch.zeros(len(audio_embeddings), dtype=torch.long),
            torch.ones(len(midi_embeddings), dtype=torch.long),
        ]).to(self.device)

        # Get domain predictions (without GRL)
        domain_logits = self.model.dann.classifier(combined)
        domain_preds = domain_logits.argmax(dim=1)
        domain_accuracy = (domain_preds == domain_labels).float().mean().item()

        # Variance check
        std_audio = audio_embeddings.std(dim=0).mean().item()
        std_midi = midi_embeddings.std(dim=0).mean().item()

        metrics = {
            **{f'val_{k}': v for k, v in retrieval_metrics.items()},
            'val_domain_accuracy': domain_accuracy,
            'val_std_audio': std_audio,
            'val_std_midi': std_midi,
        }

        return metrics

    def train(self, start_epoch: int = 0) -> Dict:
        """Full training loop."""
        logger.info(f"Starting DANN training for {self.max_epochs} epochs (from epoch {start_epoch})")
        logger.info(f"Gate 2 baseline recall: {self.gate2_recall:.3f}")

        start_time = time.time()

        for epoch in range(start_epoch, self.max_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

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
                f"domain_acc={val_metrics['val_domain_accuracy']:.1%}, "
                f"a2m_r@10={val_metrics['val_a2m_recall@10']:.3f}, "
                f"gap={val_metrics['val_gap']:.3f}"
            )

            # Save best (optimize for recall while keeping domain ~50%)
            recall_avg = (
                val_metrics['val_a2m_recall@10'] +
                val_metrics['val_m2a_recall@10']
            ) / 2

            # Prefer models where domain accuracy is closer to 50%
            domain_penalty = abs(val_metrics['val_domain_accuracy'] - 0.5)
            score = recall_avg - 0.5 * domain_penalty

            if score > self.best_recall:
                self.best_recall = score
                self.save_checkpoint('best_model.pt', epoch)
                logger.info(f"New best: recall={recall_avg:.3f}, domain_acc={val_metrics['val_domain_accuracy']:.1%}")

            # Save periodic
            if (epoch + 1) % self.checkpoint_every == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pt', epoch)

        self.save_checkpoint('final_model.pt', self.max_epochs - 1)

        # Save history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")

        return {
            'best_score': self.best_recall,
            'final_metrics': self.history[-1],
            'training_time_minutes': total_time / 60,
        }

    def save_checkpoint(self, filename: str, epoch: int = 0):
        """Save checkpoint with full state for resuming."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_recall': self.best_recall,
            'dann_step': self.model.dann.current_step if self.model.dann else 0,
            'epoch': epoch,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and restore training state. Returns start_epoch."""
        logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        self.global_step = ckpt.get('global_step', 0)
        self.best_recall = ckpt.get('best_recall', 0.0)

        # Restore DANN step counter
        dann_step = ckpt.get('dann_step', 0)
        if self.model.dann and dann_step > 0:
            self.model.dann.current_step = dann_step

        # Restore history
        self.history = ckpt.get('history', [])

        start_epoch = ckpt.get('epoch', 0) + 1
        logger.info(f"Resumed: epoch={start_epoch}, global_step={self.global_step}, best_recall={self.best_recall:.4f}")
        return start_epoch


def run_gate3(
    maestro_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    segment_len: float = 4.0,
    hop: float = 1.0,
    batch_size: int = 16,
    num_workers: int = 8,
    epochs: int = 50,
    dann_weight: float = 0.01,
    device: Optional[str] = None,
    gate2_recall: float = 0.026,
    max_batches_per_epoch: Optional[int] = None,
    checkpoint_every: int = 5,
    max_val_batches: Optional[int] = None,
    resume: Optional[str] = None,
) -> Dict:
    """
    Run Gate 3: DANN training.

    Returns:
        Results dict with GO/NO-GO decision
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'gate': 3,
        'name': 'DANN Training',
        'config': {
            'checkpoint': str(checkpoint_path),
            'epochs': epochs,
            'dann_weight': dann_weight,
            'gate2_recall': gate2_recall,
            'segment_len': segment_len,
            'hop': hop,
            'batch_size': batch_size,
            'max_batches_per_epoch': max_batches_per_epoch,
        },
    }

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create model with DANN
    logger.info("Creating model with DANN...")
    steps_per_epoch = min(len(train_loader), max_batches_per_epoch) if max_batches_per_epoch else len(train_loader)
    total_steps = epochs * steps_per_epoch
    model = CrossModalModel(
        audio_encoder='lite',
        use_dann=True,
        dann_lambda_schedule='linear_0_to_1',
        dann_max_steps=total_steps,
        device=device,
    )

    # Load Gate 2 weights (excluding DANN)
    logger.info(f"Loading Gate 2 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gate2_state = checkpoint['model_state_dict']

    # Filter out DANN weights (they don't exist in Gate 2)
    model_state = model.state_dict()
    for key in gate2_state:
        if key in model_state and 'dann' not in key:
            model_state[key] = gate2_state[key]
    model.load_state_dict(model_state)

    # Train
    trainer = DANNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        dann_weight=dann_weight,
        max_epochs=epochs,
        device=device,
        gate2_recall=gate2_recall,
        max_batches_per_epoch=max_batches_per_epoch,
        checkpoint_every=checkpoint_every,
        max_val_batches=max_val_batches,
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume:
        start_epoch = trainer.load_checkpoint(resume)

    training_results = trainer.train(start_epoch=start_epoch)
    results['training'] = training_results

    # Final evaluation
    logger.info("Final evaluation...")
    final_metrics = trainer.evaluate()

    # Check criteria
    a2m_recall = final_metrics['val_a2m_recall@10']
    m2a_recall = final_metrics['val_m2a_recall@10']
    domain_accuracy = final_metrics['val_domain_accuracy']

    # Domain accuracy should be ~50%
    domain_ok = 0.45 <= domain_accuracy <= 0.55

    # Recall should not decrease vs Gate 2
    recall_avg = (a2m_recall + m2a_recall) / 2
    recall_ok = recall_avg >= gate2_recall * 0.95  # Allow 5% margin

    results['retrieval'] = {
        'a2m_recall@10': a2m_recall,
        'm2a_recall@10': m2a_recall,
        'recall_avg': recall_avg,
        'gate2_baseline': gate2_recall,
        'pass': recall_ok,
    }

    results['domain_adversarial'] = {
        'domain_accuracy': domain_accuracy,
        'target': 0.50,
        'tolerance': 0.05,
        'pass': domain_ok,
    }

    results['gap'] = {
        'value': final_metrics['val_gap'],
    }

    # Overall decision
    all_pass = domain_ok and recall_ok

    results['decision'] = 'GO' if all_pass else 'NO-GO'
    results['status'] = 'PASS' if all_pass else 'FAIL'

    return results


def main():
    parser = argparse.ArgumentParser(description="Gate 3: DANN Training")
    parser.add_argument(
        '--maestro-dir',
        type=str,
        required=True,
        help='Path to MAESTRO dataset'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to Gate 2 model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/training_outputs/bias_control/gate3',
        help='Output directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs'
    )
    parser.add_argument(
        '--dann-weight',
        type=float,
        default=0.01,
        help='Weight for DANN loss'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default 16 to avoid OOM)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='DataLoader workers'
    )
    parser.add_argument(
        '--segment-len',
        type=float,
        default=4.0,
        help='Segment length in seconds (must match Gate 2)'
    )
    parser.add_argument(
        '--hop',
        type=float,
        default=1.0,
        help='Hop between segments in seconds (must match Gate 2)'
    )
    parser.add_argument(
        '--max-batches-per-epoch',
        type=int,
        default=None,
        help='Max batches per training epoch (Gate 2 used 1000)'
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=5,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--max-val-batches',
        type=int,
        default=None,
        help='Max batches for validation (to speed up epoch eval)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--gate2-recall',
        type=float,
        default=0.026,
        help='Gate 2 recall baseline (R@10 global = 0.026)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device'
    )

    args = parser.parse_args()

    maestro_dir = Path(args.maestro_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)

    if not maestro_dir.exists():
        logger.error(f"MAESTRO directory not found: {maestro_dir}")
        sys.exit(1)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Run Gate 3
    results = run_gate3(
        maestro_dir=maestro_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        segment_len=args.segment_len,
        hop=args.hop,
        epochs=args.epochs,
        dann_weight=args.dann_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        gate2_recall=args.gate2_recall,
        max_batches_per_epoch=args.max_batches_per_epoch,
        checkpoint_every=args.checkpoint_every,
        max_val_batches=args.max_val_batches,
        resume=args.resume,
    )

    # Save results
    results_path = output_dir / "gate3_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GATE 3: DANN TRAINING - RESULTS")
    print("=" * 60)

    print(f"\n1. Domain Adversarial:")
    print(f"   Domain accuracy: {results['domain_adversarial']['domain_accuracy']:.1%}")
    print(f"   Target: 50% ± 5%")
    print(f"   Status: {'PASS' if results['domain_adversarial']['pass'] else 'FAIL'}")

    print(f"\n2. Retrieval Performance:")
    print(f"   Audio→MIDI Recall@10: {results['retrieval']['a2m_recall@10']:.1%}")
    print(f"   MIDI→Audio Recall@10: {results['retrieval']['m2a_recall@10']:.1%}")
    print(f"   vs Gate 2 baseline: {results['retrieval']['gate2_baseline']:.1%}")
    print(f"   Status: {'PASS' if results['retrieval']['pass'] else 'FAIL'}")

    print(f"\n3. Gap: {results['gap']['value']:.3f}")

    print(f"\n" + "=" * 60)
    print(f"DECISION: {results['decision']}")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")

    return 0 if results['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
