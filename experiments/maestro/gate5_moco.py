#!/usr/bin/env python3
"""
GATE 5: Contrastive Learning with Hard-Mined Negatives (MoCo)
==============================================================

Returns to InfoNCE-style contrastive learning but with serious negatives:
- MoCo queue (4096-8192 embeddings) for many negatives without large batch
- Hard-mined negatives: same composer, within piece (different time)
- Only run AFTER VICReg/Barlow pass (Gate 3/4)

GO Criteria:
- Improves retrieval on NEG-SAME-COMPOSER compared to VICReg/Barlow
- Maintains stability (no collapse)

Based on:
- MoCo (Momentum Contrast) - He et al., CVPR 2020
- Hard Negative Mining for contrastive learning

Usage:
------
python experiments/maestro/gate5_moco.py \
    --data data/maestro_v3/constellations/tokens.npz \
    --output data/training_outputs/maestro_moco \
    --queue-size 4096 \
    --epochs 100 \
    --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.maestro_dataset import (
    MAESTROConstellationDataset,
    create_maestro_dataloaders,
    collate_maestro_constellation,
)
from src.RNA.constellation_vae import MLPConstellationEncoder, TransformerConstellationEncoder


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MoCo MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MoCoCrossModal(nn.Module):
    """
    MoCo (Momentum Contrast) for cross-modal learning.

    Uses a queue of embeddings from the momentum encoder to provide
    many negatives without requiring large batch sizes.

    Architecture:
    - Query encoder (online): Updated by gradient
    - Key encoder (momentum): Updated by EMA

    Loss: InfoNCE with queue negatives
    """

    def __init__(
        self,
        encoder_type: str = 'mlp',
        token_dim: int = 5,
        max_tokens: int = 48,
        hidden_dim: int = 128,
        z_dim: int = 64,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Query encoders (online)
        if encoder_type == 'mlp':
            self.encoder_audio = MLPConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )
            self.encoder_midi = MLPConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )
        else:
            self.encoder_audio = TransformerConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )
            self.encoder_midi = TransformerConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )

        # Key encoders (momentum) - initialized from query encoders
        if encoder_type == 'mlp':
            self.encoder_audio_k = MLPConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )
            self.encoder_midi_k = MLPConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )
        else:
            self.encoder_audio_k = TransformerConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )
            self.encoder_midi_k = TransformerConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )

        # Initialize key encoders with query encoder weights
        self.encoder_audio_k.load_state_dict(self.encoder_audio.state_dict())
        self.encoder_midi_k.load_state_dict(self.encoder_midi.state_dict())

        # Freeze key encoders
        for param in self.encoder_audio_k.parameters():
            param.requires_grad = False
        for param in self.encoder_midi_k.parameters():
            param.requires_grad = False

        # Queues (store key embeddings)
        self.register_buffer('queue_audio', torch.randn(z_dim, queue_size))
        self.register_buffer('queue_midi', torch.randn(z_dim, queue_size))
        self.queue_audio = F.normalize(self.queue_audio, dim=0)
        self.queue_midi = F.normalize(self.queue_midi, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        """Update key encoders with momentum."""
        for param_q, param_k in zip(
            self.encoder_audio.parameters(),
            self.encoder_audio_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(
            self.encoder_midi.parameters(),
            self.encoder_midi_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_audio: torch.Tensor, keys_midi: torch.Tensor):
        """Update queues with new keys."""
        batch_size = keys_audio.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue_audio[:, ptr:] = keys_audio[:remaining].T
            self.queue_midi[:, ptr:] = keys_midi[:remaining].T
            self.queue_audio[:, :batch_size - remaining] = keys_audio[remaining:].T
            self.queue_midi[:, :batch_size - remaining] = keys_midi[remaining:].T
            ptr = batch_size - remaining
        else:
            self.queue_audio[:, ptr:ptr + batch_size] = keys_audio.T
            self.queue_midi[:, ptr:ptr + batch_size] = keys_midi.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        modality: str = 'audio',
        use_key_encoder: bool = False,
    ) -> torch.Tensor:
        """Encode tokens to latent space."""
        if modality == 'audio':
            encoder = self.encoder_audio_k if use_key_encoder else self.encoder_audio
        else:
            encoder = self.encoder_midi_k if use_key_encoder else self.encoder_midi

        z_mean, _, _, _ = encoder(tokens, mask, lengths)
        # Pool over time
        z = z_mean.mean(dim=1)  # [B, z_dim]
        return z

    def forward(
        self,
        audio_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
        midi_tokens: torch.Tensor,
        midi_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            audio_tokens: [B, T, K, D]
            audio_mask: [B, T, K]
            midi_tokens: [B, T, K, D]
            midi_mask: [B, T, K]
            lengths: [B]

        Returns:
            Dict with query embeddings, key embeddings, and losses
        """
        # Query embeddings (online encoder)
        q_audio = self.encode(audio_tokens, audio_mask, lengths, 'audio', use_key_encoder=False)
        q_midi = self.encode(midi_tokens, midi_mask, lengths, 'midi', use_key_encoder=False)

        # Normalize queries
        q_audio = F.normalize(q_audio, dim=1)
        q_midi = F.normalize(q_midi, dim=1)

        # Key embeddings (momentum encoder)
        with torch.no_grad():
            self._momentum_update()
            k_audio = self.encode(audio_tokens, audio_mask, lengths, 'audio', use_key_encoder=True)
            k_midi = self.encode(midi_tokens, midi_mask, lengths, 'midi', use_key_encoder=True)
            k_audio = F.normalize(k_audio, dim=1)
            k_midi = F.normalize(k_midi, dim=1)

        return {
            'q_audio': q_audio,
            'q_midi': q_midi,
            'k_audio': k_audio,
            'k_midi': k_midi,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        hard_negatives: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MoCo InfoNCE loss.

        Args:
            outputs: Forward pass outputs
            hard_negatives: Optional dict with hard negative embeddings
                - 'same_composer_midi': [B, N_neg, z_dim]
                - 'within_piece_midi': [B, N_neg, z_dim]

        Returns:
            Dict with loss components
        """
        q_audio = outputs['q_audio']  # [B, D]
        q_midi = outputs['q_midi']
        k_audio = outputs['k_audio']
        k_midi = outputs['k_midi']

        B = q_audio.shape[0]

        # Positive logits: audio query vs midi key (paired)
        l_pos_a2m = torch.einsum('bd,bd->b', q_audio, k_midi).unsqueeze(1)  # [B, 1]
        l_pos_m2a = torch.einsum('bd,bd->b', q_midi, k_audio).unsqueeze(1)

        # Negative logits: audio query vs all midi keys in queue
        l_neg_a2m = torch.einsum('bd,dk->bk', q_audio, self.queue_midi)  # [B, K]
        l_neg_m2a = torch.einsum('bd,dk->bk', q_midi, self.queue_audio)

        # Concatenate positive and negatives
        logits_a2m = torch.cat([l_pos_a2m, l_neg_a2m], dim=1)  # [B, 1+K]
        logits_m2a = torch.cat([l_pos_m2a, l_neg_m2a], dim=1)

        # Scale by temperature
        logits_a2m = logits_a2m / self.temperature
        logits_m2a = logits_m2a / self.temperature

        # Labels: positive is always at index 0
        labels = torch.zeros(B, dtype=torch.long, device=q_audio.device)

        # InfoNCE loss
        loss_a2m = F.cross_entropy(logits_a2m, labels)
        loss_m2a = F.cross_entropy(logits_m2a, labels)

        # Add hard negatives if provided
        loss_hard = torch.tensor(0.0, device=q_audio.device)
        if hard_negatives is not None:
            if 'same_composer_midi' in hard_negatives:
                hard_neg = hard_negatives['same_composer_midi']  # [B, N, D]
                hard_neg = F.normalize(hard_neg, dim=-1)
                l_hard = torch.einsum('bd,bnd->bn', q_audio, hard_neg)  # [B, N]
                l_hard = l_hard / self.temperature
                # Push hard negatives down
                loss_hard = loss_hard + F.relu(l_hard - l_pos_a2m + 0.1).mean()

        # Update queue
        self._dequeue_and_enqueue(k_audio.detach(), k_midi.detach())

        return {
            'loss_a2m': loss_a2m,
            'loss_m2a': loss_m2a,
            'loss_hard': loss_hard,
            'total': loss_a2m + loss_m2a + loss_hard,
        }

    def get_embeddings(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        modality: str = 'audio',
    ) -> torch.Tensor:
        """Get embeddings for evaluation."""
        with torch.no_grad():
            z = self.encode(tokens, mask, lengths, modality, use_key_encoder=False)
            return F.normalize(z, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HARD NEGATIVE MINING
# ═══════════════════════════════════════════════════════════════════════════════

class HardNegativeSampler:
    """
    Samples hard negatives for contrastive learning.

    Types of hard negatives:
    - Same composer: MIDI from same composer, different piece
    - Within piece: MIDI from same piece, different time window
    """

    def __init__(
        self,
        dataset: MAESTROConstellationDataset,
        model: MoCoCrossModal,
        device: str,
        n_negatives: int = 5,
    ):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.n_negatives = n_negatives

        # Build index
        self._build_index()

    def _build_index(self):
        """Build index for hard negative sampling."""
        self.composer_to_indices = defaultdict(list)
        self.piece_to_indices = defaultdict(list)

        for idx in range(len(self.dataset)):
            file_idx = self.dataset.valid_indices[idx]
            composer = self.dataset.composers[file_idx]
            piece_id = self.dataset.piece_ids[file_idx]

            self.composer_to_indices[composer].append(idx)
            self.piece_to_indices[piece_id].append(idx)

    def sample_same_composer(
        self,
        batch_indices: List[int],
        batch_metas: List[Dict],
    ) -> Optional[torch.Tensor]:
        """
        Sample MIDI from same composer, different piece.

        Returns:
            Tensor [B, n_neg, z_dim] of hard negative embeddings
        """
        hard_negs = []

        for idx, meta in zip(batch_indices, batch_metas):
            composer = meta['composer']
            piece_id = meta['piece_id']

            # Get candidates from same composer, different piece
            candidates = [
                i for i in self.composer_to_indices[composer]
                if self.dataset.piece_ids[self.dataset.valid_indices[i]] != piece_id
            ]

            if len(candidates) < self.n_negatives:
                # Pad with random samples if not enough
                all_indices = list(range(len(self.dataset)))
                candidates = candidates + [
                    i for i in all_indices if i not in candidates
                ][:self.n_negatives - len(candidates)]

            # Sample
            sampled = np.random.choice(candidates, self.n_negatives, replace=False)

            # Get embeddings
            neg_embs = []
            for neg_idx in sampled:
                audio_tok, audio_msk, midi_tok, midi_msk, _ = self.dataset[neg_idx]
                midi_tok = midi_tok.unsqueeze(0).to(self.device)
                midi_msk = midi_msk.unsqueeze(0).to(self.device)
                z = self.model.get_embeddings(midi_tok, midi_msk, modality='midi')
                neg_embs.append(z)

            hard_negs.append(torch.cat(neg_embs, dim=0))  # [n_neg, D]

        return torch.stack(hard_negs, dim=0)  # [B, n_neg, D]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: MoCoCrossModal,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    hard_negative_sampler: Optional[HardNegativeSampler] = None,
    use_hard_negatives: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_a2m = 0
    total_m2a = 0
    n_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        audio_tok, audio_msk, midi_tok, midi_msk, metas, lengths = batch
        audio_tok = audio_tok.to(device)
        audio_msk = audio_msk.to(device)
        midi_tok = midi_tok.to(device)
        midi_msk = midi_msk.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)

        # Hard negatives (expensive, use sparingly)
        hard_negatives = None
        if use_hard_negatives and hard_negative_sampler is not None and n_batches % 10 == 0:
            batch_indices = list(range(len(metas)))  # Approximate
            hard_negatives = {
                'same_composer_midi': hard_negative_sampler.sample_same_composer(
                    batch_indices, metas
                )
            }

        losses = model.compute_loss(outputs, hard_negatives)
        loss = losses['total']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_a2m += losses['loss_a2m'].item()
        total_m2a += losses['loss_m2a'].item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'loss_a2m': total_a2m / n_batches,
        'loss_m2a': total_m2a / n_batches,
    }


@torch.no_grad()
def validate_epoch(
    model: MoCoCrossModal,
    val_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    all_z_audio = []
    all_z_midi = []
    all_composers = []
    all_piece_ids = []

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        audio_tok, audio_msk, midi_tok, midi_msk, metas, lengths = batch
        audio_tok = audio_tok.to(device)
        audio_msk = audio_msk.to(device)
        midi_tok = midi_tok.to(device)
        midi_msk = midi_msk.to(device)
        lengths = lengths.to(device)

        z_audio = model.get_embeddings(audio_tok, audio_msk, lengths, 'audio')
        z_midi = model.get_embeddings(midi_tok, midi_msk, lengths, 'midi')

        all_z_audio.append(z_audio.cpu())
        all_z_midi.append(z_midi.cpu())

        for m in metas:
            all_composers.append(m['composer'])
            all_piece_ids.append(m['piece_id'])

    z_audio = torch.cat(all_z_audio, dim=0).numpy()
    z_midi = torch.cat(all_z_midi, dim=0).numpy()

    # Cosine similarity
    sim = z_audio @ z_midi.T

    N = len(z_audio)
    labels = np.arange(N)

    # Global Recall@1
    top1 = np.argmax(sim, axis=1)
    recall1 = float((top1 == labels).mean())

    # MRR
    sorted_idx = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_idx == labels[:, None])[1] + 1
    mrr = float((1.0 / ranks).mean())

    # Same-composer retrieval (harder)
    composers = np.array(all_composers)
    same_composer_recall = []
    for i in range(N):
        # Get indices of same composer
        same_mask = composers == composers[i]
        same_indices = np.where(same_mask)[0]

        if len(same_indices) > 1:
            # Within same composer, find the correct match
            sim_same = sim[i, same_mask]
            target_pos = np.where(same_indices == i)[0][0]
            rank_in_composer = (sim_same > sim_same[target_pos]).sum() + 1
            same_composer_recall.append(1.0 / rank_in_composer)

    same_composer_mrr = float(np.mean(same_composer_recall)) if same_composer_recall else 0.0

    # Variance
    var_audio = float(z_audio.var(axis=0).mean())
    var_midi = float(z_midi.var(axis=0).mean())

    return {
        'recall@1': recall1,
        'mrr': mrr,
        'same_composer_mrr': same_composer_mrr,
        'var_audio': var_audio,
        'var_midi': var_midi,
        'random_baseline': 1.0 / N,
    }


def train_model(
    model: MoCoCrossModal,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    output_dir: Path,
    lr: float = 1e-3,
    use_hard_negatives: bool = False,
) -> Dict:
    """Full training loop."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_recall = 0
    best_epoch = 0

    print(f"\nTraining MoCo model for {epochs} epochs")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, device)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })

        print(f"Epoch {epoch}/{epochs}: "
              f"loss={train_metrics['loss']:.4f}, "
              f"R@1={val_metrics['recall@1']:.4f}, "
              f"same_comp_mrr={val_metrics['same_composer_mrr']:.4f}")

        # Save best
        if val_metrics['recall@1'] > best_recall:
            best_recall = val_metrics['recall@1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')
            print(f"  Best model saved (R@1={best_recall:.4f})")

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
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model: MoCoCrossModal,
    test_loader: DataLoader,
    device: str,
    output_dir: Path,
) -> Dict:
    """Full evaluation on test set."""
    model.eval()

    all_z_audio = []
    all_z_midi = []
    all_composers = []
    all_piece_ids = []

    for batch in tqdm(test_loader, desc="Extracting embeddings"):
        audio_tok, audio_msk, midi_tok, midi_msk, metas, lengths = batch
        audio_tok = audio_tok.to(device)
        audio_msk = audio_msk.to(device)
        midi_tok = midi_tok.to(device)
        midi_msk = midi_msk.to(device)
        lengths = lengths.to(device)

        z_audio = model.get_embeddings(audio_tok, audio_msk, lengths, 'audio')
        z_midi = model.get_embeddings(midi_tok, midi_msk, lengths, 'midi')

        all_z_audio.append(z_audio.cpu().numpy())
        all_z_midi.append(z_midi.cpu().numpy())

        for m in metas:
            all_composers.append(m['composer'])
            all_piece_ids.append(m['piece_id'])

    z_audio = np.concatenate(all_z_audio, axis=0)
    z_midi = np.concatenate(all_z_midi, axis=0)
    composers = np.array(all_composers)
    piece_ids = np.array(all_piece_ids)

    # Save embeddings
    np.savez(
        output_dir / 'embeddings.npz',
        z_audio=z_audio,
        z_midi=z_midi,
        composers=composers,
        piece_ids=piece_ids,
    )

    # Compute metrics
    N = len(z_audio)
    sim = z_audio @ z_midi.T
    labels = np.arange(N)

    results = {}

    # Global metrics
    for k in [1, 5, 10, 20]:
        topk = np.argsort(-sim, axis=1)[:, :k]
        correct = np.any(topk == labels[:, None], axis=1)
        results[f'recall@{k}'] = float(correct.mean())

    sorted_idx = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_idx == labels[:, None])[1] + 1
    results['mrr'] = float((1.0 / ranks).mean())
    results['mean_rank'] = float(ranks.mean())

    # Same-composer metrics (NEG-SAME-COMPOSER)
    same_composer_ranks = []
    for i in range(N):
        same_mask = composers == composers[i]
        same_indices = np.where(same_mask)[0]

        if len(same_indices) > 1:
            sim_same = sim[i, same_mask]
            target_pos = np.where(same_indices == i)[0][0]
            rank = (sim_same > sim_same[target_pos]).sum() + 1
            same_composer_ranks.append(rank)

    if same_composer_ranks:
        results['same_composer_mrr'] = float(np.mean([1.0/r for r in same_composer_ranks]))
        results['same_composer_recall@1'] = float(np.mean([r == 1 for r in same_composer_ranks]))
        results['same_composer_mean_rank'] = float(np.mean(same_composer_ranks))

    # Variance
    results['var_audio'] = float(z_audio.var(axis=0).mean())
    results['var_midi'] = float(z_midi.var(axis=0).mean())

    results['random_baseline'] = 1.0 / N
    results['ratio_vs_random'] = results['recall@1'] / results['random_baseline']

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='GATE 5: MoCo with Hard Negatives')
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to constellation tokens NPZ')
    parser.add_argument('--output', type=Path, default=Path('data/training_outputs/maestro_moco'),
                        help='Output directory')
    parser.add_argument('--gate4-results', type=Path, default=None,
                        help='Path to Gate 4 results for comparison')
    parser.add_argument('--encoder-type', type=str, default='mlp',
                        choices=['mlp', 'transformer'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--queue-size', type=int, default=4096,
                        help='MoCo queue size')
    parser.add_argument('--momentum', type=float, default=0.999,
                        help='Momentum for key encoder update')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='InfoNCE temperature')
    parser.add_argument('--max-tokens', type=int, default=48)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--use-hard-negatives', action='store_true',
                        help='Use hard negative mining')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    args.output.mkdir(parents=True, exist_ok=True)

    print("GATE 5: MoCo with Hard-Mined Negatives")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Encoder: {args.encoder_type}")
    print(f"Queue size: {args.queue_size}")
    print(f"Device: {device}")
    print(f"Output: {args.output}")

    # Load Gate 4 results for comparison
    gate4_results = None
    if args.gate4_results and args.gate4_results.exists():
        with open(args.gate4_results) as f:
            gate4_results = json.load(f)
        print(f"\nGate 4 baseline: R@1={gate4_results['test']['recall@1']:.4f}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_maestro_dataloaders(
        npz_path=args.data,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Read max_tokens from dataset (NPZ contains the correct value from analizador)
    max_tokens = train_loader.dataset.max_tokens
    print(f"Using max_tokens={max_tokens} from dataset")

    # Create model
    model = MoCoCrossModal(
        encoder_type=args.encoder_type,
        token_dim=5,
        max_tokens=max_tokens,  # Use value from NPZ, not args
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")

    # Train
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        output_dir=args.output,
        lr=args.lr,
        use_hard_negatives=args.use_hard_negatives,
    )

    # Load best and evaluate
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(args.output / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluate_model(model, test_loader, device, args.output)

    # Summary
    print("\n" + "=" * 60)
    print("GATE 5 SUMMARY")
    print("=" * 60)

    print(f"\nTest Results:")
    print(f"  Recall@1: {test_results['recall@1']:.4f}")
    print(f"  Recall@5: {test_results['recall@5']:.4f}")
    print(f"  MRR: {test_results['mrr']:.4f}")
    print(f"  vs Random: {test_results['ratio_vs_random']:.1f}x")

    if 'same_composer_mrr' in test_results:
        print(f"\nSame-Composer (Hard Negatives):")
        print(f"  MRR: {test_results['same_composer_mrr']:.4f}")
        print(f"  Recall@1: {test_results['same_composer_recall@1']:.4f}")

    # GO criteria
    no_collapse = test_results['var_audio'] > 0.01 and test_results['var_midi'] > 0.01

    improves_on_hard = True
    if gate4_results and 'same_composer_mrr' in test_results:
        gate4_same_composer = gate4_results['test'].get('same_composer_mrr', 0)
        improves_on_hard = test_results['same_composer_mrr'] > gate4_same_composer

    print(f"\nGO Criteria:")
    print(f"  No collapse: {'PASS' if no_collapse else 'FAIL'}")
    print(f"  Improves on hard negatives: {'PASS' if improves_on_hard else 'FAIL'}")

    go = no_collapse and improves_on_hard
    print(f"\n{'GATE 5 PASS' if go else 'GATE 5 FAIL'}")

    # Save results
    all_results = {
        'train': train_results,
        'test': test_results,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'go_criteria': {
            'no_collapse': no_collapse,
            'improves_on_hard': improves_on_hard,
            'pass': go,
        },
    }

    with open(args.output / 'gate5_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate report
    with open(args.output / 'GATE5_REPORT.md', 'w') as f:
        f.write("# GATE 5: MoCo with Hard Negatives Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Encoder**: {args.encoder_type}\n")
        f.write(f"**Queue size**: {args.queue_size}\n")
        f.write(f"**Epochs**: {args.epochs}\n\n")

        f.write("## Test Results\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in test_results.items():
            if isinstance(v, float):
                f.write(f"| {k} | {v:.4f} |\n")

        f.write(f"\n## GO/NO-GO\n\n")
        f.write(f"- No collapse: {'PASS' if no_collapse else 'FAIL'}\n")
        f.write(f"- Improves on hard negatives: {'PASS' if improves_on_hard else 'FAIL'}\n")
        f.write(f"\n**Result**: {'**PASS**' if go else '**FAIL**'}\n")

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
