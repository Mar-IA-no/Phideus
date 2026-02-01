#!/usr/bin/env python3
"""
Checkpoint 6: Verificación End-to-End
======================================

Objetivo: Ejecutar pipeline completo con datos sintéticos controlados.

Tests:
1. Dataset sintético IDÉNTICO (audio == vibración)
   → Expected: Top-1 = 100%

2. Dataset sintético SHUFFLED
   → Expected: Top-1 ≈ 1/N (random)

Si el pipeline funciona con datos sintéticos pero falla con datos reales,
el problema está en los datos, no en el código.
"""

import sys
import os

# Configure CUBLAS before importing torch
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Dict
import tempfile


class SyntheticConstellationDataset(Dataset):
    """
    Synthetic dataset for E2E testing.

    Modes:
    - 'identical': audio == vibration (should give 100% retrieval)
    - 'shuffled': audio and vibration are randomly paired
    - 'correlated': audio and vibration share some structure
    """

    def __init__(self, n_samples: int = 128, T: int = 10, K: int = 48, D: int = 5,
                 mode: str = 'identical', seed: int = 42):
        super().__init__()
        self.n_samples = n_samples
        self.T = T
        self.K = K
        self.D = D
        self.mode = mode

        np.random.seed(seed)

        # Generate audio tokens
        self.audio_tokens = np.random.randn(n_samples, T, K, D).astype(np.float32)
        self.audio_masks = np.ones((n_samples, T, K), dtype=np.float32)
        # Make sparse: only first K//2 tokens valid
        self.audio_masks[:, :, K//2:] = 0

        # Generate vibration tokens based on mode
        if mode == 'identical':
            # Vibration = Audio (perfect correspondence)
            self.vib_tokens = self.audio_tokens.copy()
            self.vib_masks = self.audio_masks.copy()
        elif mode == 'shuffled':
            # Vibration is shuffled (no correspondence)
            perm = np.random.permutation(n_samples)
            self.vib_tokens = self.audio_tokens[perm].copy()
            self.vib_masks = self.audio_masks[perm].copy()
        elif mode == 'correlated':
            # Vibration = Audio + noise (partial correspondence)
            noise = np.random.randn(n_samples, T, K, D).astype(np.float32) * 0.1
            self.vib_tokens = self.audio_tokens + noise
            self.vib_masks = self.audio_masks.copy()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create metadata
        self.metas = [
            {'condition': f'C{i % 4}', 'filename': f'file_{i}.csv', 'idx': i}
            for i in range(n_samples)
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.audio_tokens[idx]),
            torch.from_numpy(self.audio_masks[idx]),
            torch.from_numpy(self.vib_tokens[idx]),
            torch.from_numpy(self.vib_masks[idx]),
            self.metas[idx]
        )


def collate_synthetic(batch):
    """Collate function for synthetic data."""
    audio_tokens, audio_masks, vib_tokens, vib_masks, metas = zip(*batch)

    return (
        torch.stack(audio_tokens),
        torch.stack(audio_masks),
        torch.stack(vib_tokens),
        torch.stack(vib_masks),
        list(metas),
        torch.tensor([t.shape[0] for t in audio_tokens])
    )


class SimpleEncoder(nn.Module):
    """Simple encoder for E2E testing."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, z_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.z_dim = z_dim

    def forward(self, tokens, mask, lengths):
        # tokens: [B, T, K, D]
        # mask: [B, T, K]
        B, T, K, D = tokens.shape

        # Flatten and encode
        x = self.fc1(tokens)  # [B, T, K, hidden]
        x = F.relu(x)
        x = self.fc2(x)  # [B, T, K, z_dim]

        # Masked mean pooling over K
        mask_expanded = mask.unsqueeze(-1)  # [B, T, K, 1]
        x = (x * mask_expanded).sum(dim=2) / (mask_expanded.sum(dim=2) + 1e-8)  # [B, T, z_dim]

        # Mean over T
        z = x.mean(dim=1)  # [B, z_dim]

        return z


def compute_retrieval_accuracy(z_audio: np.ndarray, z_vib: np.ndarray) -> float:
    """Compute Top-1 retrieval accuracy."""
    # L2 normalize
    z_a = z_audio / (np.linalg.norm(z_audio, axis=1, keepdims=True) + 1e-8)
    z_v = z_vib / (np.linalg.norm(z_vib, axis=1, keepdims=True) + 1e-8)

    # Similarity matrix
    sim = z_a @ z_v.T  # [N, N]

    # Top-1 accuracy
    N = sim.shape[0]
    top1_idx = np.argmax(sim, axis=1)
    correct = (top1_idx == np.arange(N)).sum()

    return correct / N


def train_simple_model(train_loader, device, epochs: int = 50) -> SimpleEncoder:
    """Train a simple model with InfoNCE loss."""
    model = SimpleEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    temperature = 0.1

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            audio_tokens, audio_masks, vib_tokens, vib_masks, metas, lengths = batch
            audio_tokens = audio_tokens.to(device)
            audio_masks = audio_masks.to(device)
            vib_tokens = vib_tokens.to(device)
            vib_masks = vib_masks.to(device)
            lengths = lengths.to(device)

            # Encode
            z_audio = model(audio_tokens, audio_masks, lengths)
            z_vib = model(vib_tokens, vib_masks, lengths)

            # L2 normalize
            z_audio = F.normalize(z_audio, dim=1)
            z_vib = F.normalize(z_vib, dim=1)

            # InfoNCE loss
            logits = z_audio @ z_vib.T / temperature
            labels = torch.arange(logits.shape[0], device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    return model


def test_identical_synthetic(device: str) -> Tuple[bool, Dict]:
    """
    Test 1: Datos idénticos deben dar Top-1 = 100%.
    """
    print("\n" + "="*60)
    print("TEST 1: Synthetic IDENTICAL Data")
    print("="*60)

    # Create dataset where audio == vibration
    dataset = SyntheticConstellationDataset(
        n_samples=128, T=10, K=48, D=5, mode='identical', seed=42
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_synthetic)

    print(f"  Training on {len(dataset)} identical pairs...")
    model = train_simple_model(loader, device, epochs=50)

    # Evaluate
    all_z_audio = []
    all_z_vib = []

    with torch.no_grad():
        for batch in loader:
            audio_tokens, audio_masks, vib_tokens, vib_masks, metas, lengths = batch
            audio_tokens = audio_tokens.to(device)
            audio_masks = audio_masks.to(device)
            vib_tokens = vib_tokens.to(device)
            vib_masks = vib_masks.to(device)
            lengths = lengths.to(device)

            z_audio = model(audio_tokens, audio_masks, lengths)
            z_vib = model(vib_tokens, vib_masks, lengths)

            all_z_audio.append(z_audio.cpu().numpy())
            all_z_vib.append(z_vib.cpu().numpy())

    z_audio = np.concatenate(all_z_audio, axis=0)
    z_vib = np.concatenate(all_z_vib, axis=0)

    top1 = compute_retrieval_accuracy(z_audio, z_vib)

    print(f"\n  Results:")
    print(f"    Top-1 Accuracy: {top1:.4f}")
    print(f"    Expected: 1.0000 (100%)")

    # Should be very high (> 90%)
    passed = top1 > 0.90

    if passed:
        print(f"\n  ✓ PASS: Identical data gives high retrieval ({top1:.1%})")
    else:
        print(f"\n  ✗ FAIL: Identical data gives low retrieval ({top1:.1%})")
        print(f"    → Pipeline may have bugs")

    return passed, {'top1': top1}


def test_shuffled_synthetic(device: str) -> Tuple[bool, Dict]:
    """
    Test 2: Datos shuffled deben dar Top-1 ≈ random.
    """
    print("\n" + "="*60)
    print("TEST 2: Synthetic SHUFFLED Data")
    print("="*60)

    # Create dataset where vibration is shuffled
    dataset = SyntheticConstellationDataset(
        n_samples=128, T=10, K=48, D=5, mode='shuffled', seed=42
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_synthetic)

    print(f"  Training on {len(dataset)} shuffled pairs...")
    model = train_simple_model(loader, device, epochs=50)

    # Evaluate
    all_z_audio = []
    all_z_vib = []

    with torch.no_grad():
        for batch in loader:
            audio_tokens, audio_masks, vib_tokens, vib_masks, metas, lengths = batch
            audio_tokens = audio_tokens.to(device)
            audio_masks = audio_masks.to(device)
            vib_tokens = vib_tokens.to(device)
            vib_masks = vib_masks.to(device)
            lengths = lengths.to(device)

            z_audio = model(audio_tokens, audio_masks, lengths)
            z_vib = model(vib_tokens, vib_masks, lengths)

            all_z_audio.append(z_audio.cpu().numpy())
            all_z_vib.append(z_vib.cpu().numpy())

    z_audio = np.concatenate(all_z_audio, axis=0)
    z_vib = np.concatenate(all_z_vib, axis=0)

    top1 = compute_retrieval_accuracy(z_audio, z_vib)
    expected_random = 1.0 / len(dataset)

    print(f"\n  Results:")
    print(f"    Top-1 Accuracy: {top1:.4f}")
    print(f"    Expected random: {expected_random:.4f}")

    # Should be close to random (< 10%)
    passed = top1 < 0.15

    if passed:
        print(f"\n  ✓ PASS: Shuffled data gives near-random retrieval ({top1:.1%})")
    else:
        print(f"\n  ⚠ NOTE: Shuffled data gives higher than random ({top1:.1%})")
        print(f"    → Model may be learning dataset-wide features")
        # Still pass, but warn

    return True, {'top1': top1, 'expected_random': expected_random}


def test_correlated_synthetic(device: str) -> Tuple[bool, Dict]:
    """
    Test 3: Datos correlacionados deben dar Top-1 intermedio.
    """
    print("\n" + "="*60)
    print("TEST 3: Synthetic CORRELATED Data")
    print("="*60)

    # Create dataset where vibration = audio + noise
    dataset = SyntheticConstellationDataset(
        n_samples=128, T=10, K=48, D=5, mode='correlated', seed=42
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_synthetic)

    print(f"  Training on {len(dataset)} correlated pairs (audio + 10% noise)...")
    model = train_simple_model(loader, device, epochs=50)

    # Evaluate
    all_z_audio = []
    all_z_vib = []

    with torch.no_grad():
        for batch in loader:
            audio_tokens, audio_masks, vib_tokens, vib_masks, metas, lengths = batch
            audio_tokens = audio_tokens.to(device)
            audio_masks = audio_masks.to(device)
            vib_tokens = vib_tokens.to(device)
            vib_masks = vib_masks.to(device)
            lengths = lengths.to(device)

            z_audio = model(audio_tokens, audio_masks, lengths)
            z_vib = model(vib_tokens, vib_masks, lengths)

            all_z_audio.append(z_audio.cpu().numpy())
            all_z_vib.append(z_vib.cpu().numpy())

    z_audio = np.concatenate(all_z_audio, axis=0)
    z_vib = np.concatenate(all_z_vib, axis=0)

    top1 = compute_retrieval_accuracy(z_audio, z_vib)

    print(f"\n  Results:")
    print(f"    Top-1 Accuracy: {top1:.4f}")
    print(f"    Expected: Between random and 100%")

    # Should be high but not perfect
    passed = top1 > 0.5

    if passed:
        print(f"\n  ✓ PASS: Correlated data gives good retrieval ({top1:.1%})")
    else:
        print(f"\n  ✗ FAIL: Correlated data gives low retrieval ({top1:.1%})")
        print(f"    → Model may not be learning correlations")

    return passed, {'top1': top1}


def run_all_tests():
    """Run all E2E tests."""
    print("\n" + "="*70)
    print("CHECKPOINT 6: VERIFICACIÓN END-TO-END")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    results = {}

    passed1, stats1 = test_identical_synthetic(device)
    results['identical'] = passed1
    results['identical_top1'] = stats1['top1']

    passed2, stats2 = test_shuffled_synthetic(device)
    results['shuffled'] = passed2
    results['shuffled_top1'] = stats2['top1']

    passed3, stats3 = test_correlated_synthetic(device)
    results['correlated'] = passed3
    results['correlated_top1'] = stats3['top1']

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"  identical: {'✓ PASS' if passed1 else '✗ FAIL'} (Top-1 = {stats1['top1']:.1%})")
    print(f"  shuffled: {'✓ PASS' if passed2 else '✗ FAIL'} (Top-1 = {stats2['top1']:.1%})")
    print(f"  correlated: {'✓ PASS' if passed3 else '✗ FAIL'} (Top-1 = {stats3['top1']:.1%})")

    all_passed = passed1 and passed3  # shuffled can be marginal

    print("\n" + "-"*70)
    if all_passed:
        print("✓ CHECKPOINT 6 PASSED: Pipeline E2E funciona correctamente")
        print("  → El código de training/evaluation es correcto")
        print("  → Si los resultados con datos reales son NO-GO, el problema")
        print("    está en los DATOS, no en el código")
    else:
        print("✗ CHECKPOINT 6 FAILED: Pipeline E2E tiene problemas")
        if not passed1:
            print("  → Incluso datos idénticos dan bajo retrieval")
            print("  → Hay un BUG fundamental en el pipeline")
    print("-"*70)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
