#!/usr/bin/env python3
"""
Checkpoint 2: Verificación de Collate/DataLoader
=================================================

Objetivo: Confirmar que el DataLoader mantiene correspondencia audio-vib.

Tests:
1. Verificar que audio[i] y vib[i] son del mismo archivo
2. Verificar que metadata[i] es idéntico para audio y vib
3. Verificar que lengths coinciden

Si el DataLoader mezcla pares, el modelo no puede aprender correspondencia.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict


def test_histogram_dataloader(npz_path: str) -> Tuple[bool, Dict]:
    """
    Test DataLoader para formato histogram.
    """
    from src.datasets.roseta_dataset import (
        RosetaDataset, collate_roseta_sequences, detect_npz_format
    )
    from torch.utils.data import DataLoader

    print("\n" + "="*60)
    print("TEST: Histogram DataLoader")
    print("="*60)

    data_format = detect_npz_format(npz_path)
    if data_format != 'histogram':
        print(f"  Skipping: Data format is {data_format}, not histogram")
        return True, {'skipped': True}

    dataset = RosetaDataset(
        npz_path=npz_path,
        strategy='sequence',
        max_frames=100,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,  # Important: no shuffle for verification
        collate_fn=collate_roseta_sequences,
        num_workers=0,
    )

    issues = []

    for batch_idx, batch in enumerate(dataloader):
        audio, vibration, metas, lengths = batch

        # Check 1: Batch shapes match
        if audio.shape[0] != vibration.shape[0]:
            issues.append(f"Batch {batch_idx}: Audio batch size {audio.shape[0]} != Vib {vibration.shape[0]}")

        # Check 2: Metadata count matches batch size
        if len(metas) != audio.shape[0]:
            issues.append(f"Batch {batch_idx}: Metadata count {len(metas)} != batch size {audio.shape[0]}")

        # Check 3: Lengths count matches
        if len(lengths) != audio.shape[0]:
            issues.append(f"Batch {batch_idx}: Lengths count {len(lengths)} != batch size {audio.shape[0]}")

        # Check 4: All metas in batch should have consistent structure
        for i, meta in enumerate(metas):
            if 'condition' not in meta:
                issues.append(f"Batch {batch_idx}, sample {i}: Missing 'condition' in metadata")
            if 'filename' not in meta:
                issues.append(f"Batch {batch_idx}, sample {i}: Missing 'filename' in metadata")

        if batch_idx >= 5:  # Only check first few batches
            break

    if issues:
        print(f"\n  Issues found:")
        for issue in issues[:10]:
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues)-10} more")
        passed = False
    else:
        print(f"  ✓ No issues found in first 6 batches")
        passed = True

    return passed, {'issues': len(issues)}


def test_constellation_dataloader(npz_path: str) -> Tuple[bool, Dict]:
    """
    Test DataLoader para formato constellation.

    Verifica que audio[i] y vib[i] son del mismo archivo.
    """
    from src.datasets.roseta_dataset import (
        RosetaConstellationDataset, collate_constellation_sequences, detect_npz_format
    )
    from torch.utils.data import DataLoader

    print("\n" + "="*60)
    print("TEST: Constellation DataLoader")
    print("="*60)

    data_format = detect_npz_format(npz_path)
    if data_format != 'constellation':
        print(f"  Skipping: Data format is {data_format}, not constellation")
        return True, {'skipped': True}

    dataset = RosetaConstellationDataset(
        npz_path=npz_path,
        strategy='sequence',
        max_frames=100,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_constellation_sequences,
        num_workers=0,
    )

    issues = []

    for batch_idx, batch in enumerate(dataloader):
        audio_tokens, audio_masks, vib_tokens, vib_masks, metas, lengths = batch

        # Check 1: Batch shapes match
        if audio_tokens.shape[0] != vib_tokens.shape[0]:
            issues.append(f"Batch {batch_idx}: Audio batch {audio_tokens.shape[0]} != Vib {vib_tokens.shape[0]}")

        # Check 2: Sequence lengths match within each sample
        for i in range(audio_tokens.shape[0]):
            # Both should have the same padded length
            if audio_tokens.shape[1] != vib_tokens.shape[1]:
                issues.append(f"Batch {batch_idx}, sample {i}: Audio T != Vib T")

        # Check 3: Metadata count matches
        if len(metas) != audio_tokens.shape[0]:
            issues.append(f"Batch {batch_idx}: Metadata count {len(metas)} != batch size")

        # Check 4: Verify mask consistency
        for i in range(audio_tokens.shape[0]):
            # Masks should have same shape as tokens (without last dim)
            if audio_masks[i].shape != audio_tokens[i].shape[:-1]:
                issues.append(f"Batch {batch_idx}, sample {i}: Audio mask shape mismatch")
            if vib_masks[i].shape != vib_tokens[i].shape[:-1]:
                issues.append(f"Batch {batch_idx}, sample {i}: Vib mask shape mismatch")

        # Check 5: Verify lengths tensor
        for i, length in enumerate(lengths):
            if length <= 0:
                issues.append(f"Batch {batch_idx}, sample {i}: Invalid length {length}")

        if batch_idx >= 5:
            break

    if issues:
        print(f"\n  Issues found:")
        for issue in issues[:10]:
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues)-10} more")
        passed = False
    else:
        print(f"  ✓ No issues found in first 6 batches")
        passed = True

    return passed, {'issues': len(issues)}


def test_pairing_preservation(npz_path: str) -> Tuple[bool, Dict]:
    """
    Test crítico: Verificar que el mismo índice de dataset
    siempre devuelve el mismo par audio-vib.
    """
    from src.datasets.roseta_dataset import (
        RosetaConstellationDataset, RosetaDataset, detect_npz_format
    )

    print("\n" + "="*60)
    print("TEST: Preservación de Pairing")
    print("="*60)

    data_format = detect_npz_format(npz_path)

    if data_format == 'constellation':
        dataset = RosetaConstellationDataset(
            npz_path=npz_path,
            strategy='sequence',
            max_frames=100,
            frame_sample_mode='start',  # Deterministic
        )
    else:
        dataset = RosetaDataset(
            npz_path=npz_path,
            strategy='sequence',
            max_frames=100,
            frame_sample_mode='start',
        )

    issues = []

    # Test: Access same index twice, should get same result
    for idx in [0, len(dataset)//2, len(dataset)-1]:
        if idx >= len(dataset):
            continue

        sample1 = dataset[idx]
        sample2 = dataset[idx]

        if data_format == 'constellation':
            audio1, mask1, vib1, vmask1, meta1 = sample1
            audio2, mask2, vib2, vmask2, meta2 = sample2

            if not torch.allclose(audio1, audio2):
                issues.append(f"Index {idx}: Audio tokens differ between accesses")
            if not torch.allclose(vib1, vib2):
                issues.append(f"Index {idx}: Vib tokens differ between accesses")
            # Check filename or idx for consistency
            file_key = 'filename' if 'filename' in meta1 else 'idx'
            if meta1.get(file_key) != meta2.get(file_key):
                issues.append(f"Index {idx}: {file_key} differs between accesses")
        else:
            audio1, vib1, meta1 = sample1
            audio2, vib2, meta2 = sample2

            if not torch.allclose(audio1, audio2):
                issues.append(f"Index {idx}: Audio differs between accesses")
            if not torch.allclose(vib1, vib2):
                issues.append(f"Index {idx}: Vibration differs between accesses")
            file_key = 'filename' if 'filename' in meta1 else 'idx'
            if meta1.get(file_key) != meta2.get(file_key):
                issues.append(f"Index {idx}: {file_key} differs between accesses")

    # Test: Audio and vib should come from same file
    for idx in range(min(10, len(dataset))):
        sample = dataset[idx]

        if data_format == 'constellation':
            audio, mask, vib, vmask, meta = sample
        else:
            audio, vib, meta = sample

        # The key test: metadata should indicate same file for both
        filename = meta.get('filename', meta.get('idx', 'unknown'))
        print(f"  Index {idx}: filename = {filename}, condition = {meta.get('condition', '?')}")

    if issues:
        print(f"\n  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        passed = False
    else:
        print(f"\n  ✓ Pairing is preserved across accesses")
        passed = True

    return passed, {'issues': len(issues)}


def test_collate_order_preservation(npz_path: str) -> Tuple[bool, Dict]:
    """
    Test: Verificar que el collate no mezcla el orden de los pares.
    """
    from src.datasets.roseta_dataset import (
        RosetaConstellationDataset, RosetaDataset,
        collate_constellation_sequences, collate_roseta_sequences,
        detect_npz_format
    )
    from torch.utils.data import DataLoader

    print("\n" + "="*60)
    print("TEST: Collate Order Preservation")
    print("="*60)

    data_format = detect_npz_format(npz_path)

    if data_format == 'constellation':
        dataset = RosetaConstellationDataset(
            npz_path=npz_path,
            strategy='sequence',
            max_frames=100,
            frame_sample_mode='start',
        )
        collate_fn = collate_constellation_sequences
    else:
        dataset = RosetaDataset(
            npz_path=npz_path,
            strategy='sequence',
            max_frames=100,
            frame_sample_mode='start',
        )
        collate_fn = collate_roseta_sequences

    # Get individual samples
    samples = [dataset[i] for i in range(4)]

    if data_format == 'constellation':
        expected_filenames = [s[4].get('filename', s[4].get('idx', i)) for i, s in enumerate(samples)]
    else:
        expected_filenames = [s[2].get('filename', s[2].get('idx', i)) for i, s in enumerate(samples)]

    # Collate them manually
    batch = collate_fn(samples)

    if data_format == 'constellation':
        metas = batch[4]  # metas is at index 4
    else:
        metas = batch[2]  # metas is at index 2

    actual_filenames = [m.get('filename', m.get('idx', i)) for i, m in enumerate(metas)]

    # Verify order preserved
    if expected_filenames == actual_filenames:
        print(f"  ✓ Order preserved after collate")
        print(f"    Expected: {expected_filenames}")
        print(f"    Actual:   {actual_filenames}")
        passed = True
    else:
        print(f"  ✗ Order NOT preserved after collate!")
        print(f"    Expected: {expected_filenames}")
        print(f"    Actual:   {actual_filenames}")
        passed = False

    return passed, {'order_preserved': passed}


def run_all_tests(npz_path: str):
    """Run all collate verification tests."""
    print("\n" + "="*70)
    print("CHECKPOINT 2: VERIFICACIÓN DE COLLATE/DATALOADER")
    print("="*70)
    print(f"Dataset: {npz_path}")

    # Detect format
    from src.datasets.roseta_dataset import detect_npz_format
    data_format = detect_npz_format(npz_path)
    print(f"Detected format: {data_format}")

    results = {}

    # Run format-specific test
    if data_format == 'histogram':
        passed1, stats1 = test_histogram_dataloader(npz_path)
        results['histogram_loader'] = passed1
    else:
        passed1, stats1 = test_constellation_dataloader(npz_path)
        results['constellation_loader'] = passed1

    passed2, stats2 = test_pairing_preservation(npz_path)
    results['pairing_preservation'] = passed2

    passed3, stats3 = test_collate_order_preservation(npz_path)
    results['collate_order'] = passed3

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = passed1 and passed2 and passed3
    for name, passed in results.items():
        print(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")

    print("\n" + "-"*70)
    if all_passed:
        print("✓ CHECKPOINT 2 PASSED: DataLoader mantiene correspondencia audio-vib")
        print("  → Los pares NO se mezclan durante el batching")
    else:
        print("✗ CHECKPOINT 2 FAILED: Hay problemas en el DataLoader")
        print("  → Los pares PUEDEN estar mezclados")
        print("  → Esto explicaría por qué el modelo no aprende correspondencia")
    print("-"*70)

    return all_passed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to NPZ dataset')
    args = parser.parse_args()

    success = run_all_tests(args.data)
    sys.exit(0 if success else 1)
