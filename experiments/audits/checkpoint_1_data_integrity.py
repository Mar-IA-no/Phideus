#!/usr/bin/env python3
"""
Checkpoint 1: Verificación de Datos (Dataset Integrity)
========================================================

Objetivo: Confirmar que los tokens constellation son válidos y discriminativos.

Tests:
1. Verificar shapes y rangos de valores
2. Verificar correspondencia audio-vibración
3. Comparar tokens intra-condición vs inter-condición

Si los datos no son discriminativos pre-red, el modelo no puede aprender.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple


def load_constellation_data(npz_path: str) -> Dict:
    """Load constellation NPZ and return structured data."""
    data = np.load(npz_path, allow_pickle=True)

    return {
        'data': data,
        'n_files': int(data['n_files']),
        'filenames': list(data['filenames']),
        'conditions': list(data['conditions']),
        'metadata': data['metadata'].item(),
        'output_format': str(data.get('output_format', 'unknown')),
        'max_tokens': int(data.get('max_tokens_per_frame', 48)),
        'token_dim': int(data.get('token_dim', 5)),
    }


def test_shapes_and_ranges(info: Dict) -> Tuple[bool, Dict]:
    """
    Test 1: Verificar shapes y rangos de tokens.

    Token format: [log_ratio, delta_t, weight, anchor_band, target_band]
    Expected:
    - log_ratio: [-inf, inf] (typical: [-4, 4])
    - delta_t: [0, hop_size] (typical: [0, 1024])
    - weight: [0, inf] (sqrt of amplitude product)
    - anchor_band: [0, n_bands-1]
    - target_band: [0, n_bands-1]
    """
    print("\n" + "="*60)
    print("TEST 1: Shapes y rangos de tokens")
    print("="*60)

    data = info['data']
    n_files = info['n_files']
    max_tokens = info['max_tokens']
    token_dim = info['token_dim']

    stats = {
        'shapes_ok': True,
        'total_frames': 0,
        'sparse_frames': 0,  # Frames with < 10 tokens
        'empty_frames': 0,   # Frames with 0 tokens
        'log_ratio': {'min': np.inf, 'max': -np.inf},
        'delta_t': {'min': np.inf, 'max': -np.inf},
        'weight': {'min': np.inf, 'max': -np.inf},
        'anchor_band': {'min': np.inf, 'max': -np.inf},
        'target_band': {'min': np.inf, 'max': -np.inf},
    }

    for idx in range(n_files):
        # Load tokens and masks
        audio_tokens = data[f'audio_tokens_{idx}']
        audio_mask = data[f'audio_mask_{idx}']
        vib_tokens = data[f'vibration_tokens_{idx}']
        vib_mask = data[f'vibration_mask_{idx}']

        # Check shapes
        T_audio = audio_tokens.shape[0]
        T_vib = vib_tokens.shape[0]

        if T_audio != T_vib:
            print(f"  ⚠ File {idx}: Audio T={T_audio} != Vib T={T_vib}")
            stats['shapes_ok'] = False

        if audio_tokens.shape[1] != max_tokens or audio_tokens.shape[2] != token_dim:
            print(f"  ⚠ File {idx}: Audio shape {audio_tokens.shape} != expected (T, {max_tokens}, {token_dim})")
            stats['shapes_ok'] = False

        # Update statistics
        stats['total_frames'] += T_audio

        # Check sparsity
        for t in range(T_audio):
            n_valid_audio = audio_mask[t].sum()
            n_valid_vib = vib_mask[t].sum()

            if n_valid_audio < 10 or n_valid_vib < 10:
                stats['sparse_frames'] += 1
            if n_valid_audio == 0 or n_valid_vib == 0:
                stats['empty_frames'] += 1

        # Collect value ranges (only for valid tokens)
        for tokens, mask in [(audio_tokens, audio_mask), (vib_tokens, vib_mask)]:
            valid = mask > 0
            if valid.any():
                valid_tokens = tokens[valid]

                # log_ratio (col 0)
                stats['log_ratio']['min'] = min(stats['log_ratio']['min'], valid_tokens[:, 0].min())
                stats['log_ratio']['max'] = max(stats['log_ratio']['max'], valid_tokens[:, 0].max())

                # delta_t (col 1)
                stats['delta_t']['min'] = min(stats['delta_t']['min'], valid_tokens[:, 1].min())
                stats['delta_t']['max'] = max(stats['delta_t']['max'], valid_tokens[:, 1].max())

                # weight (col 2)
                stats['weight']['min'] = min(stats['weight']['min'], valid_tokens[:, 2].min())
                stats['weight']['max'] = max(stats['weight']['max'], valid_tokens[:, 2].max())

                # anchor_band (col 3)
                stats['anchor_band']['min'] = min(stats['anchor_band']['min'], valid_tokens[:, 3].min())
                stats['anchor_band']['max'] = max(stats['anchor_band']['max'], valid_tokens[:, 3].max())

                # target_band (col 4)
                stats['target_band']['min'] = min(stats['target_band']['min'], valid_tokens[:, 4].min())
                stats['target_band']['max'] = max(stats['target_band']['max'], valid_tokens[:, 4].max())

    # Print results
    print(f"\n  Files: {n_files}")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Sparse frames (<10 tokens): {stats['sparse_frames']} ({100*stats['sparse_frames']/max(1,stats['total_frames']):.1f}%)")
    print(f"  Empty frames (0 tokens): {stats['empty_frames']} ({100*stats['empty_frames']/max(1,stats['total_frames']):.1f}%)")

    print(f"\n  Value ranges:")
    print(f"    log_ratio: [{stats['log_ratio']['min']:.3f}, {stats['log_ratio']['max']:.3f}]")
    print(f"    delta_t:   [{stats['delta_t']['min']:.1f}, {stats['delta_t']['max']:.1f}]")
    print(f"    weight:    [{stats['weight']['min']:.4f}, {stats['weight']['max']:.4f}]")
    print(f"    anchor_band: [{stats['anchor_band']['min']:.0f}, {stats['anchor_band']['max']:.0f}]")
    print(f"    target_band: [{stats['target_band']['min']:.0f}, {stats['target_band']['max']:.0f}]")

    # Validation
    passed = stats['shapes_ok'] and stats['empty_frames'] < stats['total_frames'] * 0.1
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: Shapes y rangos de datos")

    return passed, stats


def test_audio_vibration_correspondence(info: Dict) -> Tuple[bool, Dict]:
    """
    Test 2: Verificar correspondencia audio-vibración.

    - audio_tokens_{idx} y vibration_tokens_{idx} deben tener el mismo T
    - Metadata debe coincidir
    """
    print("\n" + "="*60)
    print("TEST 2: Correspondencia Audio-Vibración")
    print("="*60)

    data = info['data']
    n_files = info['n_files']
    metadata = info['metadata']
    filenames = info['filenames']

    mismatches = []

    for idx in range(n_files):
        audio_tokens = data[f'audio_tokens_{idx}']
        vib_tokens = data[f'vibration_tokens_{idx}']

        T_audio = audio_tokens.shape[0]
        T_vib = vib_tokens.shape[0]

        if T_audio != T_vib:
            mismatches.append({
                'idx': idx,
                'filename': filenames[idx],
                'T_audio': T_audio,
                'T_vib': T_vib,
            })

    if mismatches:
        print(f"\n  ⚠ Found {len(mismatches)} files with T mismatch:")
        for m in mismatches[:5]:
            print(f"    File {m['idx']} ({m['filename']}): T_audio={m['T_audio']}, T_vib={m['T_vib']}")
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches)-5} more")
    else:
        print(f"\n  ✓ All {n_files} files have matching T between audio and vibration")

    # Check metadata structure
    print(f"\n  Checking metadata structure...")
    sample_meta = metadata[filenames[0]]
    print(f"    Sample metadata keys: {list(sample_meta.keys())}")

    required_keys = ['condition', 'is_healthy']
    missing_keys = [k for k in required_keys if k not in sample_meta]
    if missing_keys:
        print(f"    ⚠ Missing required keys: {missing_keys}")

    passed = len(mismatches) == 0 and len(missing_keys) == 0
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: Correspondencia audio-vibración")

    return passed, {'mismatches': len(mismatches)}


def flatten_tokens_to_histogram(tokens: np.ndarray, mask: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """
    Convert sparse tokens to a histogram representation for comparison.

    Uses log_ratio values binned into n_bins.
    """
    # Only use valid tokens
    valid = mask > 0
    if not valid.any():
        return np.zeros(n_bins)

    log_ratios = tokens[valid][:, 0]  # log_ratio is column 0
    weights = tokens[valid][:, 2]     # weight is column 2

    # Bin log_ratios into [-4, 4] range
    bins = np.linspace(-4, 4, n_bins + 1)
    hist, _ = np.histogram(log_ratios, bins=bins, weights=weights)

    # Normalize
    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist


def test_intra_vs_inter_condition(info: Dict) -> Tuple[bool, Dict]:
    """
    Test 3: Comparar similaridad intra-condición vs inter-condición.

    Tokens de la misma condición deberían ser más similares que tokens
    de diferente condición.

    Si no hay diferencia → problema en extractor, no en modelo
    """
    print("\n" + "="*60)
    print("TEST 3: Discriminabilidad por condición (pre-red)")
    print("="*60)

    data = info['data']
    n_files = info['n_files']
    conditions = info['conditions']

    # Group files by condition
    condition_files = defaultdict(list)
    for idx in range(n_files):
        condition_files[conditions[idx]].append(idx)

    print(f"\n  Condiciones encontradas: {list(condition_files.keys())}")
    for cond, files in condition_files.items():
        print(f"    {cond}: {len(files)} files")

    # Compute average histogram per file (averaging over time)
    file_histograms = {}
    for idx in range(n_files):
        audio_tokens = data[f'audio_tokens_{idx}']
        audio_mask = data[f'audio_mask_{idx}']

        # Average histogram over all frames
        T = audio_tokens.shape[0]
        frame_hists = []
        for t in range(T):
            h = flatten_tokens_to_histogram(audio_tokens[t], audio_mask[t])
            frame_hists.append(h)

        file_histograms[idx] = np.mean(frame_hists, axis=0)

    # Compute intra-condition and inter-condition similarities
    intra_sims = []
    inter_sims = []

    unique_conditions = list(condition_files.keys())

    for cond in unique_conditions:
        files_in_cond = condition_files[cond]

        # Intra-condition: pairs within same condition
        for i, f1 in enumerate(files_in_cond):
            for f2 in files_in_cond[i+1:]:
                h1 = file_histograms[f1]
                h2 = file_histograms[f2]
                if h1.sum() > 0 and h2.sum() > 0:
                    sim = 1 - cosine(h1, h2)
                    intra_sims.append(sim)

        # Inter-condition: pairs across different conditions
        for other_cond in unique_conditions:
            if other_cond == cond:
                continue
            for f1 in files_in_cond:
                for f2 in condition_files[other_cond]:
                    h1 = file_histograms[f1]
                    h2 = file_histograms[f2]
                    if h1.sum() > 0 and h2.sum() > 0:
                        sim = 1 - cosine(h1, h2)
                        inter_sims.append(sim)

    # Statistics
    if intra_sims and inter_sims:
        mean_intra = np.mean(intra_sims)
        std_intra = np.std(intra_sims)
        mean_inter = np.mean(inter_sims)
        std_inter = np.std(inter_sims)
        gap = mean_intra - mean_inter

        print(f"\n  Similaridad coseno de histogramas:")
        print(f"    Intra-condición: {mean_intra:.4f} ± {std_intra:.4f} (N={len(intra_sims)})")
        print(f"    Inter-condición: {mean_inter:.4f} ± {std_inter:.4f} (N={len(inter_sims)})")
        print(f"    Gap (intra - inter): {gap:.4f}")

        # Expected: intra > inter (positive gap)
        # Threshold from plan: gap > 0.1 is significant
        if gap > 0.1:
            print(f"\n  ✓ PASS: Gap significativo ({gap:.4f} > 0.1)")
            print(f"    → Los datos SON discriminativos pre-red")
            passed = True
        elif gap > 0.05:
            print(f"\n  ⚠ MARGINAL: Gap pequeño ({gap:.4f})")
            print(f"    → Los datos tienen poca discriminabilidad")
            passed = True  # Still pass, but warn
        else:
            print(f"\n  ✗ FAIL: Gap insuficiente ({gap:.4f} ≤ 0.05)")
            print(f"    → Los datos NO son discriminativos pre-red")
            print(f"    → El problema está en el extractor, no en el modelo")
            passed = False
    else:
        print("\n  ⚠ No se pudieron calcular similaridades")
        passed = False
        gap = 0.0

    return passed, {'gap': gap if 'gap' in dir() else 0.0}


def test_audio_vs_vibration_similarity(info: Dict) -> Tuple[bool, Dict]:
    """
    Test 4: Verificar que audio y vibración del mismo archivo son similares.

    Si los pares audio-vib no están correlacionados pre-red, el modelo
    no puede aprender la correspondencia.
    """
    print("\n" + "="*60)
    print("TEST 4: Correlación Audio-Vibración por archivo")
    print("="*60)

    data = info['data']
    n_files = info['n_files']

    paired_sims = []
    unpaired_sims = []

    # Compute histogram per file (for both audio and vibration)
    audio_hists = {}
    vib_hists = {}

    for idx in range(n_files):
        audio_tokens = data[f'audio_tokens_{idx}']
        audio_mask = data[f'audio_mask_{idx}']
        vib_tokens = data[f'vibration_tokens_{idx}']
        vib_mask = data[f'vibration_mask_{idx}']

        # Average histogram over all frames
        T = audio_tokens.shape[0]
        audio_frame_hists = []
        vib_frame_hists = []

        for t in range(T):
            audio_frame_hists.append(flatten_tokens_to_histogram(audio_tokens[t], audio_mask[t]))
            vib_frame_hists.append(flatten_tokens_to_histogram(vib_tokens[t], vib_mask[t]))

        audio_hists[idx] = np.mean(audio_frame_hists, axis=0)
        vib_hists[idx] = np.mean(vib_frame_hists, axis=0)

    # Paired similarities (audio[i] vs vib[i])
    for idx in range(n_files):
        h_a = audio_hists[idx]
        h_v = vib_hists[idx]
        if h_a.sum() > 0 and h_v.sum() > 0:
            sim = 1 - cosine(h_a, h_v)
            paired_sims.append(sim)

    # Unpaired similarities (audio[i] vs vib[j] where i != j)
    for i in range(n_files):
        for j in range(n_files):
            if i == j:
                continue
            h_a = audio_hists[i]
            h_v = vib_hists[j]
            if h_a.sum() > 0 and h_v.sum() > 0:
                sim = 1 - cosine(h_a, h_v)
                unpaired_sims.append(sim)

    # Statistics
    mean_paired = np.mean(paired_sims)
    std_paired = np.std(paired_sims)
    mean_unpaired = np.mean(unpaired_sims)
    std_unpaired = np.std(unpaired_sims)
    gap = mean_paired - mean_unpaired

    print(f"\n  Similaridad coseno Audio-Vibración:")
    print(f"    Pares aligned: {mean_paired:.4f} ± {std_paired:.4f} (N={len(paired_sims)})")
    print(f"    Pares shuffled: {mean_unpaired:.4f} ± {std_unpaired:.4f} (N={len(unpaired_sims)})")
    print(f"    Gap (aligned - shuffled): {gap:.4f}")

    # This is the critical metric for cross-modal pairing
    if gap > 0.1:
        print(f"\n  ✓ PASS: Gap significativo ({gap:.4f} > 0.1)")
        print(f"    → Los pares audio-vib SON discriminativos pre-red")
        passed = True
    elif gap > 0.05:
        print(f"\n  ⚠ MARGINAL: Gap pequeño ({gap:.4f})")
        print(f"    → Los pares tienen poca discriminabilidad")
        passed = True
    else:
        print(f"\n  ✗ FAIL: Gap insuficiente ({gap:.4f} ≤ 0.05)")
        print(f"    → Los pares audio-vib NO son discriminativos")
        print(f"    → El modelo NO puede aprender correspondencia")
        passed = False

    return passed, {'gap': gap}


def run_all_tests(npz_path: str):
    """Run all data integrity tests."""
    print("\n" + "="*70)
    print("CHECKPOINT 1: VERIFICACIÓN DE DATOS (Dataset Integrity)")
    print("="*70)
    print(f"Dataset: {npz_path}")

    # Load data
    info = load_constellation_data(npz_path)
    print(f"\nFormat detected: {info['output_format']}")
    print(f"Files: {info['n_files']}, Max tokens: {info['max_tokens']}, Token dim: {info['token_dim']}")

    # Run tests
    results = {}

    passed1, stats1 = test_shapes_and_ranges(info)
    results['shapes'] = passed1

    passed2, stats2 = test_audio_vibration_correspondence(info)
    results['correspondence'] = passed2

    passed3, stats3 = test_intra_vs_inter_condition(info)
    results['condition_discriminability'] = passed3
    results['condition_gap'] = stats3['gap']

    passed4, stats4 = test_audio_vs_vibration_similarity(info)
    results['audio_vib_discriminability'] = passed4
    results['audio_vib_gap'] = stats4['gap']

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = all([passed1, passed2, passed3, passed4])
    print(f"  shapes: {'✓ PASS' if passed1 else '✗ FAIL'}")
    print(f"  correspondence: {'✓ PASS' if passed2 else '✗ FAIL'}")
    print(f"  condition_discriminability: {'✓ PASS' if passed3 else '✗ FAIL'} (gap={stats3['gap']:.4f})")
    print(f"  audio_vib_discriminability: {'✓ PASS' if passed4 else '✗ FAIL'} (gap={stats4['gap']:.4f})")

    print("\n" + "-"*70)
    if all_passed:
        print("✓ CHECKPOINT 1 PASSED: Datos de entrada son válidos y discriminativos")
        print("  → El problema NO está en el extractor")
    else:
        print("✗ CHECKPOINT 1 FAILED: Hay problemas en los datos de entrada")
        print("  → El problema PUEDE estar en el extractor")
        if not passed3 or not passed4:
            print("  → Considerar volver a Fase 3A-1 con extractor mejorado")
    print("-"*70)

    return all_passed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='data/datasets/roseta_constellation.npz',
                        help='Path to constellation NPZ')
    args = parser.parse_args()

    success = run_all_tests(args.data)
    sys.exit(0 if success else 1)
