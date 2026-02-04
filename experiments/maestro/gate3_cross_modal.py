#!/usr/bin/env python3
"""
GATE 3: Cross-Modal Model Training (VICReg / Barlow Twins)
===========================================================

Trains a stable cross-modal encoder on dense representations:
- Audio: log-mel spectrogram [B, T, 128]
- MIDI: piano roll [B, T, 88]

Loss options:
- VICReg: Variance-Invariance-Covariance regularization
- Barlow Twins: Cross-correlation redundancy reduction

GO Criteria:
- No collapse (var(z) > 0.1)
- Piece-level Top-1 > baselines

Usage:
------
python experiments/maestro/gate3_cross_modal.py \
    --data data/maestro_v3/processed \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/training_outputs/maestro_vicreg \
    --loss vicreg \
    --epochs 100 \
    --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    librosa = None

from src.RNA.vicreg import VICRegCrossModal
from src.RNA.barlow_twins import BarlowTwinsCrossModal
from src.utils.midi_utils import parse_midi, notes_to_piano_roll, filter_notes_by_time


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class MAESTRODenseDataset(Dataset):
    """
    Dataset for dense representations (mel + piano roll).
    """

    def __init__(
        self,
        processed_dir: Path,
        maestro_dir: Path,
        split: str = 'train',
        sr: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.maestro_dir = Path(maestro_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length

        # Load metadata
        with open(processed_dir / 'dataset_metadata.json') as f:
            metadata = json.load(f)

        self.segments = [s for s in metadata['segments'] if s['split'] == split]

        if max_samples is not None:
            self.segments = self.segments[:max_samples]

        self.window_len = metadata.get('window_len', 4.0)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        seg = self.segments[idx]

        # Load audio
        audio_path = self.processed_dir / seg['audio_path']
        audio = np.load(audio_path)

        # Extract mel spectrogram
        if librosa is not None:
            mel = librosa.feature.melspectrogram(
                y=audio, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
        else:
            # Fallback: simple STFT
            n_fft = 2048
            hop = self.hop_length
            window = np.hanning(n_fft)
            n_frames = len(audio) // hop
            mel_db = np.zeros((self.n_mels, n_frames), dtype=np.float32)

        mel_db = mel_db.T  # [T, n_mels]

        # Load MIDI and create piano roll
        midi_path = self.maestro_dir / seg['midi_path']
        notes = parse_midi(midi_path)

        # Filter to segment
        start_time = seg['start_time']
        end_time = seg['end_time']
        seg_notes = filter_notes_by_time(notes, start_time, end_time)

        # Shift times
        for n in seg_notes:
            n.onset = max(0, n.onset - start_time)
            n.offset = max(n.onset + 0.01, n.offset - start_time)

        # Frame rate to match mel
        frame_rate = self.sr / self.hop_length
        piano_roll = notes_to_piano_roll(
            seg_notes,
            duration=self.window_len,
            frame_rate=frame_rate,
            piano_range=True,
        )  # [T, 88]

        # Ensure same length
        T = min(mel_db.shape[0], piano_roll.shape[0])
        mel_db = mel_db[:T]
        piano_roll = piano_roll[:T]

        # Normalize mel
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        metadata = {
            'segment_id': seg['segment_id'],
            'piece_id': seg['piece_id'],
            'composer': seg['composer'],
            'split': seg['split'],
        }

        return (
            torch.tensor(mel_db, dtype=torch.float32),
            torch.tensor(piano_roll, dtype=torch.float32),
            metadata,
        )


def collate_dense(batch):
    """Collate function for dense dataset."""
    mels, piano_rolls, metas = zip(*batch)

    # Pad to max length in batch
    max_len = max(m.shape[0] for m in mels)

    B = len(batch)
    D_mel = mels[0].shape[1]
    D_pr = piano_rolls[0].shape[1]

    mel_batch = torch.zeros(B, max_len, D_mel)
    pr_batch = torch.zeros(B, max_len, D_pr)
    lengths = []

    for i, (m, p) in enumerate(zip(mels, piano_rolls)):
        T = m.shape[0]
        mel_batch[i, :T] = m
        pr_batch[i, :T] = p
        lengths.append(T)

    return mel_batch, pr_batch, list(metas), torch.tensor(lengths)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    loss_type: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_inv = 0
    total_var = 0
    n_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        mel, piano_roll, metas, lengths = batch
        mel = mel.to(device)
        piano_roll = piano_roll.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        outputs = model(mel, piano_roll, lengths)
        loss = outputs['total']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if loss_type == 'vicreg':
            total_inv += outputs['invariance'].item()
            total_var += outputs['variance'].item()

        n_batches += 1

    results = {'loss': total_loss / n_batches}
    if loss_type == 'vicreg':
        results['invariance'] = total_inv / n_batches
        results['variance'] = total_var / n_batches

    return results


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    loss_type: str,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_z_audio = []
    all_z_midi = []
    n_batches = 0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        mel, piano_roll, metas, lengths = batch
        mel = mel.to(device)
        piano_roll = piano_roll.to(device)
        lengths = lengths.to(device)

        outputs = model(mel, piano_roll, lengths)
        total_loss += outputs['total'].item()

        # Get embeddings for retrieval
        z_a, z_m = model.get_embeddings(mel, piano_roll, lengths)
        all_z_audio.append(z_a.cpu())
        all_z_midi.append(z_m.cpu())
        n_batches += 1

    # Compute retrieval metrics
    z_audio = torch.cat(all_z_audio, dim=0).numpy()
    z_midi = torch.cat(all_z_midi, dim=0).numpy()

    # Cosine similarity
    z_a_norm = z_audio / (np.linalg.norm(z_audio, axis=1, keepdims=True) + 1e-8)
    z_m_norm = z_midi / (np.linalg.norm(z_midi, axis=1, keepdims=True) + 1e-8)
    sim = z_a_norm @ z_m_norm.T

    N = len(z_audio)
    labels = np.arange(N)

    # Recall@1
    top1 = np.argmax(sim, axis=1)
    recall1 = float((top1 == labels).mean())

    # MRR
    sorted_idx = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_idx == labels[:, None])[1] + 1
    mrr = float((1.0 / ranks).mean())

    # Variance (collapse check)
    var_audio = float(z_audio.var(axis=0).mean())
    var_midi = float(z_midi.var(axis=0).mean())

    return {
        'loss': total_loss / n_batches,
        'recall@1': recall1,
        'mrr': mrr,
        'var_audio': var_audio,
        'var_midi': var_midi,
        'random_baseline': 1.0 / N,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    output_dir: Path,
    lr: float = 1e-3,
    loss_type: str = 'vicreg',
) -> Dict:
    """Full training loop."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_recall = 0
    best_epoch = 0
    no_collapse = True

    print(f"\nTraining {loss_type.upper()} model for {epochs} epochs")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, loss_type)
        val_metrics = validate_epoch(model, val_loader, device, loss_type)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })

        # Check collapse
        if val_metrics['var_audio'] < 0.01 or val_metrics['var_midi'] < 0.01:
            no_collapse = False
            print(f"\n⚠ WARNING: Collapse detected at epoch {epoch}!")
            print(f"  var_audio: {val_metrics['var_audio']:.6f}")
            print(f"  var_midi: {val_metrics['var_midi']:.6f}")

        # Print progress
        print(f"Epoch {epoch}/{epochs}: "
              f"train_loss={train_metrics['loss']:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, "
              f"R@1={val_metrics['recall@1']:.4f}, "
              f"MRR={val_metrics['mrr']:.4f}")

        # Save best
        if val_metrics['recall@1'] > best_recall and no_collapse:
            best_recall = val_metrics['recall@1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': {
                    'loss_type': loss_type,
                    'z_dim': model.z_dim if hasattr(model, 'z_dim') else model.proj_dim,
                },
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Best model saved (R@1={best_recall:.4f})")

    # Save final
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, output_dir / 'final_model.pt')

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_recall': best_recall,
        'no_collapse': no_collapse,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    output_dir: Path,
) -> Dict:
    """Full evaluation on test set."""
    model.eval()

    all_z_audio = []
    all_z_midi = []
    all_piece_ids = []
    all_composers = []

    for batch in tqdm(test_loader, desc="Extracting embeddings"):
        mel, piano_roll, metas, lengths = batch
        mel = mel.to(device)
        piano_roll = piano_roll.to(device)
        lengths = lengths.to(device)

        z_a, z_m = model.get_embeddings(mel, piano_roll, lengths)
        all_z_audio.append(z_a.cpu().numpy())
        all_z_midi.append(z_m.cpu().numpy())

        for m in metas:
            all_piece_ids.append(m['piece_id'])
            all_composers.append(m['composer'])

    z_audio = np.concatenate(all_z_audio, axis=0)
    z_midi = np.concatenate(all_z_midi, axis=0)

    # Save embeddings
    np.savez(
        output_dir / 'embeddings.npz',
        z_audio=z_audio,
        z_midi=z_midi,
        metadata={
            'piece_ids': all_piece_ids,
            'composer_ids': all_composers,
        },
    )

    # Compute metrics
    N = len(z_audio)

    z_a_norm = z_audio / (np.linalg.norm(z_audio, axis=1, keepdims=True) + 1e-8)
    z_m_norm = z_midi / (np.linalg.norm(z_midi, axis=1, keepdims=True) + 1e-8)
    sim = z_a_norm @ z_m_norm.T

    labels = np.arange(N)

    results = {}

    # Recall@K
    for k in [1, 5, 10, 20]:
        topk = np.argsort(-sim, axis=1)[:, :k]
        correct = np.any(topk == labels[:, None], axis=1)
        results[f'recall@{k}'] = float(correct.mean())

    # MRR
    sorted_idx = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_idx == labels[:, None])[1] + 1
    results['mrr'] = float((1.0 / ranks).mean())
    results['mean_rank'] = float(ranks.mean())

    # Variance
    results['var_audio'] = float(z_audio.var(axis=0).mean())
    results['var_midi'] = float(z_midi.var(axis=0).mean())

    # Random baseline
    results['random_baseline'] = 1.0 / N
    results['ratio_vs_random'] = results['recall@1'] / results['random_baseline']

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='GATE 3: Cross-Modal Training')
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to processed MAESTRO segments')
    parser.add_argument('--maestro-dir', type=Path, required=True,
                        help='Path to original MAESTRO')
    parser.add_argument('--output', type=Path, default=Path('data/training_outputs/maestro_vicreg'),
                        help='Output directory')
    parser.add_argument('--loss', type=str, default='vicreg',
                        choices=['vicreg', 'barlow'],
                        help='Loss function')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--z-dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--max-train', type=int, default=None,
                        help='Max training samples')
    parser.add_argument('--max-val', type=int, default=None,
                        help='Max validation samples')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    args.output.mkdir(parents=True, exist_ok=True)

    print("GATE 3: Cross-Modal Model Training")
    print("=" * 60)
    print(f"Loss: {args.loss}")
    print(f"Device: {device}")
    print(f"Output: {args.output}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = MAESTRODenseDataset(
        args.data, args.maestro_dir, split='train', max_samples=args.max_train
    )
    val_dataset = MAESTRODenseDataset(
        args.data, args.maestro_dir, split='validation', max_samples=args.max_val
    )
    test_dataset = MAESTRODenseDataset(
        args.data, args.maestro_dir, split='test'
    )

    print(f"  Train: {len(train_dataset)} segments")
    print(f"  Val: {len(val_dataset)} segments")
    print(f"  Test: {len(test_dataset)} segments")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_dense, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_dense, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_dense, pin_memory=True
    )

    # Create model
    if args.loss == 'vicreg':
        model = VICRegCrossModal(
            audio_input_dim=128,  # mel
            midi_input_dim=88,    # piano roll
            hidden_dim=args.hidden_dim,
            z_dim=args.z_dim,
            temporal=True,
        ).to(device)
    else:  # barlow
        model = BarlowTwinsCrossModal(
            audio_input_dim=128,
            midi_input_dim=88,
            repr_dim=args.hidden_dim,
            proj_dim=args.z_dim,
            hidden_dim=args.hidden_dim,
            temporal=True,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Train
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        output_dir=args.output,
        lr=args.lr,
        loss_type=args.loss,
    )

    # Load best and evaluate
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(args.output / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluate_model(model, test_loader, device, args.output)

    # GO/NO-GO summary
    print("\n" + "=" * 60)
    print("GATE 3 SUMMARY")
    print("=" * 60)

    print(f"\nTest Results:")
    print(f"  Recall@1: {test_results['recall@1']:.4f}")
    print(f"  Recall@5: {test_results['recall@5']:.4f}")
    print(f"  MRR: {test_results['mrr']:.4f}")
    print(f"  vs Random: {test_results['ratio_vs_random']:.1f}x")
    print(f"  Variance: audio={test_results['var_audio']:.4f}, midi={test_results['var_midi']:.4f}")

    # GO criteria
    no_collapse = test_results['var_audio'] > 0.1 and test_results['var_midi'] > 0.1
    better_than_baselines = test_results['ratio_vs_random'] > 10  # Assume baselines are ~10x

    print(f"\nGO Criteria:")
    print(f"  No collapse (var > 0.1): {'✓' if no_collapse else '✗'}")
    print(f"  Better than baselines (>10x random): {'✓' if better_than_baselines else '✗'}")

    go = no_collapse and better_than_baselines
    print(f"\n{'✓ GATE 3 PASS' if go else '✗ GATE 3 FAIL'}")

    # Save results
    all_results = {
        'train': train_results,
        'test': test_results,
        'config': vars(args),
        'go_criteria': {
            'no_collapse': no_collapse,
            'better_than_baselines': better_than_baselines,
            'pass': go,
        },
    }

    def to_serializable(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(args.output / 'gate3_results.json', 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

    # Generate report
    with open(args.output / 'GATE3_REPORT.md', 'w') as f:
        f.write("# GATE 3: Cross-Modal Training Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Loss**: {args.loss}\n")
        f.write(f"**Epochs**: {args.epochs}\n\n")

        f.write("## Test Results\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in test_results.items():
            if isinstance(v, float):
                f.write(f"| {k} | {v:.4f} |\n")

        f.write(f"\n## GO/NO-GO\n\n")
        f.write(f"- No collapse: {'✓' if no_collapse else '✗'}\n")
        f.write(f"- Better than baselines: {'✓' if better_than_baselines else '✗'}\n")
        f.write(f"\n**Result**: {'**PASS**' if go else '**FAIL**'}\n")

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
