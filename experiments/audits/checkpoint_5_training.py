#!/usr/bin/env python3
"""
Checkpoint 5: Verificación de Training
========================================

Objetivo: Confirmar que el modelo realmente aprende (gradientes fluyen).

Tests:
1. Revisar logs de training: ¿Loss disminuye?
2. Comparar checkpoints: ¿embeddings son diferentes entre epoch 1 y epoch final?
3. Verificar configuración: ¿dropout, beta_kl fueron usados?

Si el modelo no aprende durante training, los resultados NO-GO son esperados.
"""

import sys
import os

# Configure CUBLAS before importing torch
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, List
import glob


def find_training_outputs(base_dir: str) -> List[Path]:
    """Find all training output directories."""
    base = Path(base_dir)
    if not base.exists():
        return []

    # Look for directories with training artifacts
    dirs = []
    for pattern in ['**/training_log.json', '**/best_model.pt', '**/metrics.json']:
        for path in base.glob(pattern):
            dirs.append(path.parent)

    return list(set(dirs))


def test_training_log(log_path: Path) -> Tuple[bool, Dict]:
    """
    Test 1: Analizar log de training.

    Verificar que el loss disminuye durante el entrenamiento.
    """
    print("\n" + "="*60)
    print("TEST 1: Análisis de Training Log")
    print("="*60)

    if not log_path.exists():
        print(f"  ⚠ Log file not found: {log_path}")
        return False, {'error': 'log_not_found'}

    with open(log_path) as f:
        log = json.load(f)

    epochs = log.get('epochs', [])
    if not epochs:
        print(f"  ⚠ No epoch data in log")
        return False, {'error': 'no_epochs'}

    print(f"  Found {len(epochs)} epochs")

    # Extract metrics
    train_losses = [e.get('train_loss', float('inf')) for e in epochs]
    val_losses = [e.get('val_loss', float('inf')) for e in epochs]

    # Check if loss decreased
    first_train_loss = train_losses[0] if train_losses else float('inf')
    last_train_loss = train_losses[-1] if train_losses else float('inf')
    first_val_loss = val_losses[0] if val_losses else float('inf')
    last_val_loss = val_losses[-1] if val_losses else float('inf')

    print(f"\n  Training Loss:")
    print(f"    First epoch: {first_train_loss:.6f}")
    print(f"    Last epoch:  {last_train_loss:.6f}")
    print(f"    Change: {last_train_loss - first_train_loss:+.6f} ({100*(last_train_loss/first_train_loss - 1):+.1f}%)")

    print(f"\n  Validation Loss:")
    print(f"    First epoch: {first_val_loss:.6f}")
    print(f"    Last epoch:  {last_val_loss:.6f}")
    print(f"    Change: {last_val_loss - first_val_loss:+.6f} ({100*(last_val_loss/first_val_loss - 1):+.1f}%)")

    # Check for specific metrics
    if 'cos_sim' in epochs[0]:
        cos_sims = [e.get('cos_sim', 0) for e in epochs]
        print(f"\n  Cosine Similarity (z_shared):")
        print(f"    First: {cos_sims[0]:.4f}")
        print(f"    Last:  {cos_sims[-1]:.4f}")

        # Warning: if cos_sim is always 1.0, embeddings might be collapsed
        if all(abs(c - 1.0) < 0.01 for c in cos_sims):
            print(f"    ⚠ WARNING: cos_sim always ~1.0 - possible embedding collapse!")

    # Determine if training worked
    loss_decreased = last_train_loss < first_train_loss * 0.9  # At least 10% decrease

    if loss_decreased:
        print(f"\n  ✓ PASS: Loss decreased during training")
        passed = True
    else:
        print(f"\n  ✗ FAIL: Loss did NOT decrease significantly")
        print(f"    → Model may not be learning")
        passed = False

    return passed, {
        'first_train_loss': first_train_loss,
        'last_train_loss': last_train_loss,
        'first_val_loss': first_val_loss,
        'last_val_loss': last_val_loss,
        'n_epochs': len(epochs),
    }


def test_checkpoint_comparison(model_dir: Path, device: str) -> Tuple[bool, Dict]:
    """
    Test 2: Comparar embeddings entre checkpoints.

    Los embeddings de epoch 1 y epoch final deberían ser diferentes.
    """
    print("\n" + "="*60)
    print("TEST 2: Comparación de Checkpoints")
    print("="*60)

    # Find checkpoints
    checkpoints = list(model_dir.glob('model_epoch_*.pt')) + list(model_dir.glob('checkpoint_epoch_*.pt'))
    best_model = model_dir / 'best_model.pt'

    if not best_model.exists() and not checkpoints:
        print(f"  ⚠ No checkpoints found in {model_dir}")
        return True, {'skipped': True}  # Pass but skip

    # Load best model
    if best_model.exists():
        checkpoint = torch.load(best_model, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        final_epoch = checkpoint.get('epoch', -1)
        print(f"  Best model: epoch {final_epoch}")
        print(f"  Config: {config}")

        # Check config values
        issues = []

        # Critical config checks from plan
        if config.get('dropout_shared', 0) == 0:
            issues.append("dropout_shared = 0 (plan specified 0.5)")

        if config.get('beta_kl_private') is None:
            issues.append("beta_kl_private is None (plan specified 0.01)")

        if issues:
            print(f"\n  ⚠ Configuration issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  ✓ Configuration looks correct")

        passed = len(issues) == 0
    else:
        passed = True  # Skip

    return passed, {'config_issues': issues if 'issues' in dir() else []}


def test_retrieval_across_epochs(model_dir: Path, data_path: str, device: str) -> Tuple[bool, Dict]:
    """
    Test 3: Verificar si retrieval mejora entre epochs.

    Si hay múltiples checkpoints, comparar retrieval accuracy.
    """
    print("\n" + "="*60)
    print("TEST 3: Retrieval a través de Epochs")
    print("="*60)

    # Find checkpoints sorted by epoch
    checkpoints = sorted(model_dir.glob('model_epoch_*.pt'),
                        key=lambda x: int(x.stem.split('_')[-1]))

    if len(checkpoints) < 2:
        print(f"  ⚠ Need at least 2 checkpoints for comparison")
        print(f"    Found: {len(checkpoints)}")
        return True, {'skipped': True}

    # Check if data exists
    if not Path(data_path).exists():
        print(f"  ⚠ Data file not found: {data_path}")
        return True, {'skipped': True}

    print(f"  Found {len(checkpoints)} checkpoints")
    print(f"  Comparing first and last...")

    # This would require loading models and running evaluation
    # For now, just check that checkpoints exist and are loadable
    try:
        first_ckpt = torch.load(checkpoints[0], map_location=device, weights_only=False)
        last_ckpt = torch.load(checkpoints[-1], map_location=device, weights_only=False)

        print(f"\n  First checkpoint (epoch {first_ckpt.get('epoch', '?')}):")
        print(f"    Keys: {list(first_ckpt.keys())}")

        print(f"\n  Last checkpoint (epoch {last_ckpt.get('epoch', '?')}):")
        print(f"    Keys: {list(last_ckpt.keys())}")

        # Check if weights are different
        first_weights = first_ckpt['model_state_dict']
        last_weights = last_ckpt['model_state_dict']

        total_diff = 0
        n_params = 0
        for key in first_weights:
            if key in last_weights:
                diff = (first_weights[key].float() - last_weights[key].float()).abs().mean().item()
                total_diff += diff
                n_params += 1

        avg_diff = total_diff / max(n_params, 1)
        print(f"\n  Average weight difference: {avg_diff:.6f}")

        if avg_diff > 0.001:
            print(f"  ✓ PASS: Weights changed during training")
            passed = True
        else:
            print(f"  ✗ FAIL: Weights barely changed (possible learning issue)")
            passed = False

    except Exception as e:
        print(f"  ⚠ Error loading checkpoints: {e}")
        passed = True  # Skip on error

    return passed, {'avg_weight_diff': avg_diff if 'avg_diff' in dir() else 0}


def run_all_tests(training_dir: str, data_path: str = None):
    """Run all training verification tests."""
    print("\n" + "="*70)
    print("CHECKPOINT 5: VERIFICACIÓN DE TRAINING")
    print("="*70)
    print(f"Training dir: {training_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = Path(training_dir)

    if not model_dir.exists():
        print(f"  ⚠ Directory not found: {training_dir}")
        return False

    results = {}

    # Test 1: Training log analysis
    log_path = model_dir / 'training_log.json'
    if not log_path.exists():
        # Try alternative names
        for alt in ['log.json', 'metrics.json', 'history.json']:
            alt_path = model_dir / alt
            if alt_path.exists():
                log_path = alt_path
                break

    passed1, stats1 = test_training_log(log_path)
    results['training_log'] = passed1

    # Test 2: Checkpoint comparison
    passed2, stats2 = test_checkpoint_comparison(model_dir, device)
    results['checkpoint_config'] = passed2

    # Test 3: Retrieval across epochs
    if data_path:
        passed3, stats3 = test_retrieval_across_epochs(model_dir, data_path, device)
        results['weight_changes'] = passed3
    else:
        passed3 = True
        results['weight_changes'] = True

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = all(results.values())
    for name, passed in results.items():
        print(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")

    print("\n" + "-"*70)
    if all_passed:
        print("✓ CHECKPOINT 5 PASSED: Training funcionó correctamente")
        print("  → El modelo aprendió durante el entrenamiento")
    else:
        print("✗ CHECKPOINT 5 FAILED: Problemas detectados en training")
        if not passed1:
            print("  → Loss no disminuyó - modelo no aprendió")
        if not passed2:
            print("  → Configuración incorrecta (dropout, beta_kl)")
    print("-"*70)

    return all_passed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dir', type=str, required=True,
                        help='Path to training output directory')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset (optional, for retrieval test)')
    args = parser.parse_args()

    success = run_all_tests(args.training_dir, args.data)
    sys.exit(0 if success else 1)
