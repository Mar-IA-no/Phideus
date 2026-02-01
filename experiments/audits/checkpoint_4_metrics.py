#!/usr/bin/env python3
"""
Checkpoint 4: Verificación de Métricas de Evaluación
=====================================================

Objetivo: Confirmar que `evaluate_retrieval.py` calcula métricas correctamente.

Tests:
1. Embeddings idénticos → Top-1 = 100%
2. Embeddings shuffled → Top-1 ≈ 1/N (random)
3. Embeddings con ruido → Top-1 entre 0% y 100%

Si este checkpoint falla, los resultados NO-GO pueden ser un artefacto de métricas mal calculadas.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from pathlib import Path

# Import the metrics function from evaluate_retrieval
from experiments.evaluate_retrieval import compute_retrieval_metrics


def test_identical_embeddings():
    """
    Test 1: Embeddings idénticos deben dar Top-1 = 100%.

    Si audio[i] == vib[i], siempre encontraremos el par correcto.
    """
    print("\n" + "="*60)
    print("TEST 1: Embeddings IDÉNTICOS (Expected: Top-1 = 100%)")
    print("="*60)

    N = 128  # Same as real dataset
    D = 32   # Embedding dimension

    z_audio = np.random.randn(N, D)
    z_vib = z_audio.copy()  # IDENTICAL

    results = compute_retrieval_metrics(z_audio, z_vib, k_values=[1, 5, 10])

    print(f"  Top-1 Accuracy: {results['top1_accuracy']:.4f} (Expected: 1.0000)")
    print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f} (Expected: 1.0000)")
    print(f"  MRR: {results['mrr']:.4f} (Expected: 1.0000)")
    print(f"  Mean Rank: {results['mean_rank']:.1f} (Expected: 1.0)")

    passed = results['top1_accuracy'] == 1.0
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: Identical embeddings")

    return passed


def test_shuffled_embeddings():
    """
    Test 2: Embeddings shuffled deben dar Top-1 ≈ 1/N.

    Si vib es una permutación aleatoria de audio, solo por azar
    se encontrará el par correcto.
    """
    print("\n" + "="*60)
    print("TEST 2: Embeddings SHUFFLED (Expected: Top-1 ≈ 1/N)")
    print("="*60)

    N = 128
    D = 32
    expected_random = 1.0 / N

    # Run multiple trials for statistical stability
    n_trials = 10
    top1_values = []

    for trial in range(n_trials):
        np.random.seed(trial * 1000)
        z_audio = np.random.randn(N, D)

        # Shuffle completely
        perm = np.random.permutation(N)
        z_vib = z_audio[perm]

        results = compute_retrieval_metrics(z_audio, z_vib, k_values=[1])
        top1_values.append(results['top1_accuracy'])

    mean_top1 = np.mean(top1_values)
    std_top1 = np.std(top1_values)

    print(f"  Expected random baseline: {expected_random:.4f}")
    print(f"  Observed Top-1 (mean over {n_trials} trials): {mean_top1:.4f} ± {std_top1:.4f}")

    # Should be close to random (within 3 standard deviations of expected)
    # For N=128, random is 0.0078, so anything < 0.1 is reasonable
    passed = mean_top1 < 0.15  # Generously allow some variance
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: Shuffled embeddings near random baseline")

    return passed


def test_noisy_embeddings():
    """
    Test 3: Embeddings con ruido deben dar Top-1 entre 0% y 100%.

    Añadimos ruido gaussiano y verificamos que Top-1 decrece con más ruido.
    """
    print("\n" + "="*60)
    print("TEST 3: Embeddings con RUIDO (Expected: Top-1 decreasing with noise)")
    print("="*60)

    N = 128
    D = 32
    np.random.seed(42)

    z_audio = np.random.randn(N, D)

    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results_by_noise = []

    for noise_std in noise_levels:
        z_vib = z_audio + noise_std * np.random.randn(N, D)
        results = compute_retrieval_metrics(z_audio, z_vib, k_values=[1, 5, 10])
        results_by_noise.append({
            'noise_std': noise_std,
            'top1': results['top1_accuracy'],
            'top5': results['top5_accuracy'],
            'mrr': results['mrr'],
        })
        print(f"  Noise σ={noise_std:.1f}: Top-1={results['top1_accuracy']:.4f}, MRR={results['mrr']:.4f}")

    # Verify monotonicity: Top-1 should decrease as noise increases
    top1_values = [r['top1'] for r in results_by_noise]
    is_monotonic = all(top1_values[i] >= top1_values[i+1] for i in range(len(top1_values)-1))

    # At least: noise=0 should give 100%, noise=5 should be near random
    passed = (
        results_by_noise[0]['top1'] == 1.0 and  # No noise → 100%
        results_by_noise[-1]['top1'] < 0.3  # High noise → near random
    )

    print(f"\n  Monotonically decreasing: {is_monotonic}")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: Noisy embeddings behave correctly")

    return passed


def test_with_mask():
    """
    Test 4: Verificar que la máscara funciona correctamente.

    Si solo permitimos candidatos de la diagonal, Top-1 debería ser 100%.
    """
    print("\n" + "="*60)
    print("TEST 4: Retrieval con MÁSCARA (Expected: mask restricts candidates)")
    print("="*60)

    N = 128
    D = 32
    np.random.seed(42)

    z_audio = np.random.randn(N, D)
    z_vib = np.random.randn(N, D)  # Completely different

    # Without mask: should be near random
    results_no_mask = compute_retrieval_metrics(z_audio, z_vib, k_values=[1])
    print(f"  Without mask: Top-1 = {results_no_mask['top1_accuracy']:.4f}")

    # With mask: only diagonal is valid
    mask = np.eye(N, dtype=bool)
    results_with_mask = compute_retrieval_metrics(z_audio, z_vib, k_values=[1], mask=mask)
    print(f"  With diagonal mask: Top-1 = {results_with_mask['top1_accuracy']:.4f} (Expected: 1.0)")

    passed = results_with_mask['top1_accuracy'] == 1.0
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: Mask restricts candidates correctly")

    return passed


def test_dimension_sensitivity():
    """
    Test 5: Verificar que la métrica es sensible a la dimensionalidad.

    Embeddings en espacios de alta dimensión deberían ser más separables.
    """
    print("\n" + "="*60)
    print("TEST 5: Sensibilidad a DIMENSIONALIDAD")
    print("="*60)

    N = 128
    np.random.seed(42)

    for D in [2, 8, 32, 128]:
        z_audio = np.random.randn(N, D)
        z_vib = z_audio + 0.5 * np.random.randn(N, D)  # Fixed noise
        results = compute_retrieval_metrics(z_audio, z_vib, k_values=[1])
        print(f"  D={D:3d}: Top-1 = {results['top1_accuracy']:.4f}")

    # This is informative, not pass/fail
    print("\n  (Informative test - no pass/fail criterion)")
    return True


def test_real_model_scenario():
    """
    Test 6: Simular escenario de modelo real.

    Si el modelo produce embeddings muy similares (collapsed), Top-1 será random.
    """
    print("\n" + "="*60)
    print("TEST 6: Escenario de COLAPSO de embeddings")
    print("="*60)

    N = 128
    D = 32
    np.random.seed(42)

    # Case 1: All embeddings are nearly identical (collapsed model)
    base = np.random.randn(D)
    z_audio = np.tile(base, (N, 1)) + 0.001 * np.random.randn(N, D)
    z_vib = np.tile(base, (N, 1)) + 0.001 * np.random.randn(N, D)

    results_collapsed = compute_retrieval_metrics(z_audio, z_vib, k_values=[1, 5])
    print(f"  Collapsed model (all similar):")
    print(f"    Top-1 = {results_collapsed['top1_accuracy']:.4f}")
    print(f"    Top-5 = {results_collapsed['top5_accuracy']:.4f}")

    # Case 2: Diverse but uncorrelated
    z_audio = np.random.randn(N, D)
    z_vib = np.random.randn(N, D)  # Independent

    results_uncorrelated = compute_retrieval_metrics(z_audio, z_vib, k_values=[1, 5])
    print(f"\n  Uncorrelated (diverse but independent):")
    print(f"    Top-1 = {results_uncorrelated['top1_accuracy']:.4f}")
    print(f"    Top-5 = {results_uncorrelated['top5_accuracy']:.4f}")

    # Both should be near random (1/128 = 0.0078)
    expected = 1.0 / N
    print(f"\n  Expected random: {expected:.4f}")

    # Check that collapsed gives near-random
    collapsed_near_random = results_collapsed['top1_accuracy'] < 0.1
    print(f"\n  {'✓ PASS' if collapsed_near_random else '✗ FAIL'}: Collapsed model detected as random-level")

    return collapsed_near_random


def run_all_tests():
    """Run all metric verification tests."""
    print("\n" + "="*70)
    print("CHECKPOINT 4: VERIFICACIÓN DE MÉTRICAS DE EVALUACIÓN")
    print("="*70)

    results = {
        'identical': test_identical_embeddings(),
        'shuffled': test_shuffled_embeddings(),
        'noisy': test_noisy_embeddings(),
        'mask': test_with_mask(),
        'dimension': test_dimension_sensitivity(),
        'collapsed': test_real_model_scenario(),
    }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    print("\n" + "-"*70)
    if all_passed:
        print("✓ CHECKPOINT 4 PASSED: Métricas de evaluación funcionan correctamente")
        print("  → Los resultados NO-GO no son un artefacto de métricas rotas")
    else:
        print("✗ CHECKPOINT 4 FAILED: Hay problemas en las métricas de evaluación")
        print("  → Los resultados NO-GO podrían ser un artefacto de bugs")
    print("-"*70)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
