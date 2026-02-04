#!/usr/bin/env python3
"""
GATE 0: Test Harness and Evaluation Framework for MAESTRO Experiment
=====================================================================

Anti-autoengano evaluation framework with:
- Metrics: Recall@K (K=1,5,10,20), MRR, Gap aligned-shuffled
- Bootstrap CI for confidence intervals
- Negative controls: NEG_RANDOM, NEG_WITHIN_PIECE, NEG_SAME_COMPOSER
- Positive control: POS_ORACLE (synthesized audio from MIDI)

GO Criterion: Oracle > 90%, random ~ 1/N

Usage:
------
python experiments/maestro/gate0_harness.py \
    --embeddings data/evaluations/maestro_embeddings.npz \
    --output data/evaluations/maestro_gate0

Or use as library:
    from gate0_harness import compute_retrieval_metrics, bootstrap_ci, MAESTROEvaluator
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# 1. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_similarity_matrix(
    z_query: np.ndarray,
    z_gallery: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between query and gallery embeddings.

    Args:
        z_query: [N_q, D] query embeddings
        z_gallery: [N_g, D] gallery embeddings
        normalize: If True, L2 normalize before computing similarity

    Returns:
        similarity: [N_q, N_g] cosine similarity matrix
    """
    if normalize:
        z_query = z_query / (np.linalg.norm(z_query, axis=1, keepdims=True) + 1e-8)
        z_gallery = z_gallery / (np.linalg.norm(z_gallery, axis=1, keepdims=True) + 1e-8)

    return z_query @ z_gallery.T


def compute_retrieval_metrics(
    z_audio: np.ndarray,
    z_midi: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute retrieval metrics for cross-modal matching.

    Args:
        z_audio: [N, D] audio embeddings
        z_midi: [N, D] MIDI embeddings (paired with audio)
        k_values: List of k for Recall@K
        mask: [N, N] boolean mask where True = valid candidate

    Returns:
        Dict with:
        - recall@k for each k
        - mrr (Mean Reciprocal Rank)
        - mean_rank
        - gap (difference from random baseline)
    """
    N = len(z_audio)
    sim = compute_similarity_matrix(z_audio, z_midi)

    if mask is not None:
        sim = np.where(mask, sim, -np.inf)

    # Ground truth: diagonal (i-th audio matches i-th MIDI)
    labels = np.arange(N)

    results = {}

    # Recall@K
    for k in k_values:
        k_actual = min(k, N)
        topk_indices = np.argsort(-sim, axis=1)[:, :k_actual]
        correct = np.any(topk_indices == labels[:, None], axis=1)
        recall = float(correct.mean())
        results[f'recall@{k}'] = recall

    # Mean Reciprocal Rank
    sorted_indices = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_indices == labels[:, None])[1] + 1
    mrr = float((1.0 / ranks).mean())
    results['mrr'] = mrr

    # Mean rank
    results['mean_rank'] = float(ranks.mean())

    # Random baseline expectation
    random_recall1 = 1.0 / N
    results['random_baseline'] = random_recall1
    results['gap_vs_random'] = results['recall@1'] - random_recall1

    return results


def bootstrap_ci(
    metric_fn,
    z_audio: np.ndarray,
    z_midi: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap confidence intervals for metrics.

    Args:
        metric_fn: Function that takes (z_audio, z_midi) and returns dict
        z_audio, z_midi: Embeddings
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Dict mapping metric name to (mean, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    N = len(z_audio)

    # Collect bootstrap samples
    bootstrap_results = []
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        indices = rng.choice(N, size=N, replace=True)
        metrics = metric_fn(z_audio[indices], z_midi[indices])
        bootstrap_results.append(metrics)

    # Aggregate
    all_metrics = {}
    for key in bootstrap_results[0].keys():
        values = [r[key] for r in bootstrap_results]
        all_metrics[key] = values

    # Compute CIs
    alpha = (1 - ci) / 2
    results = {}
    for key, values in all_metrics.items():
        values = np.array(values)
        mean = float(values.mean())
        ci_low = float(np.percentile(values, 100 * alpha))
        ci_high = float(np.percentile(values, 100 * (1 - alpha)))
        results[key] = (mean, ci_low, ci_high)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NEGATIVE CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NegativeControl:
    """Defines a negative control configuration."""
    name: str
    description: str
    create_mask: callable  # Function to create [N, N] validity mask


def create_random_mask(
    n_samples: int,
    metadata: Dict,
) -> np.ndarray:
    """NEG_RANDOM: All pairs are valid (global random matching)."""
    return np.ones((n_samples, n_samples), dtype=bool)


def create_within_piece_mask(
    n_samples: int,
    metadata: Dict,
) -> np.ndarray:
    """
    NEG_WITHIN_PIECE: Only match within same piece (different segment).

    For each audio segment, only MIDI segments from the same piece
    are valid candidates (but not the exactly paired segment).
    """
    piece_ids = metadata.get('piece_ids', np.arange(n_samples))
    mask = np.zeros((n_samples, n_samples), dtype=bool)

    for i in range(n_samples):
        for j in range(n_samples):
            # Same piece, different segment
            if piece_ids[i] == piece_ids[j] and i != j:
                mask[i, j] = True

    # Always include diagonal (true match)
    np.fill_diagonal(mask, True)

    return mask


def create_same_composer_mask(
    n_samples: int,
    metadata: Dict,
) -> np.ndarray:
    """
    NEG_SAME_COMPOSER: Only match within same composer.

    Harder negative: MIDI from same composer but different piece.
    """
    composer_ids = metadata.get('composer_ids', np.arange(n_samples))
    mask = np.zeros((n_samples, n_samples), dtype=bool)

    for i in range(n_samples):
        for j in range(n_samples):
            if composer_ids[i] == composer_ids[j]:
                mask[i, j] = True

    return mask


def create_cross_piece_mask(
    n_samples: int,
    metadata: Dict,
) -> np.ndarray:
    """
    NEG_CROSS_PIECE: Only match across different pieces.

    For testing: can we still find the match among truly different pieces?
    """
    piece_ids = metadata.get('piece_ids', np.arange(n_samples))
    mask = np.zeros((n_samples, n_samples), dtype=bool)

    # Diagonal (true pairs) + different pieces
    np.fill_diagonal(mask, True)

    for i in range(n_samples):
        for j in range(n_samples):
            if piece_ids[i] != piece_ids[j]:
                mask[i, j] = True

    return mask


NEGATIVE_CONTROLS = {
    'random': NegativeControl(
        name='NEG_RANDOM',
        description='Global random matching (all pairs valid)',
        create_mask=create_random_mask,
    ),
    'within_piece': NegativeControl(
        name='NEG_WITHIN_PIECE',
        description='Match only within same piece (temporal negative)',
        create_mask=create_within_piece_mask,
    ),
    'same_composer': NegativeControl(
        name='NEG_SAME_COMPOSER',
        description='Match only within same composer (style negative)',
        create_mask=create_same_composer_mask,
    ),
    'cross_piece': NegativeControl(
        name='NEG_CROSS_PIECE',
        description='Match only across different pieces',
        create_mask=create_cross_piece_mask,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POSITIVE CONTROL (ORACLE)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_oracle(
    z_audio_synth: np.ndarray,
    z_midi: np.ndarray,
) -> Dict[str, float]:
    """
    POS_ORACLE: Audio synthesized from MIDI should match perfectly.

    This tests if the embedding space preserves perfect correspondence.
    GO criterion: Oracle Recall@1 > 90%
    """
    metrics = compute_retrieval_metrics(z_audio_synth, z_midi)
    return {f'oracle_{k}': v for k, v in metrics.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN EVALUATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MAESTROEvaluator:
    """
    Comprehensive evaluator for MAESTRO cross-modal experiment.

    Runs all negative controls and computes confidence intervals.
    """
    z_audio: np.ndarray
    z_midi: np.ndarray
    metadata: Dict = field(default_factory=dict)
    z_audio_synth: Optional[np.ndarray] = None  # For oracle test

    def run_all_evaluations(
        self,
        k_values: List[int] = [1, 5, 10, 20],
        n_bootstrap: int = 100,
        ci: float = 0.95,
    ) -> Dict[str, Dict]:
        """
        Run complete evaluation suite.

        Returns:
            Dict with results for each control type
        """
        results = {}
        n_samples = len(self.z_audio)

        print(f"Running MAESTRO evaluation on {n_samples} samples")
        print("=" * 60)

        # 1. Global metrics (no mask)
        print("\n[1] Global retrieval (audio -> MIDI)...")
        global_metrics = compute_retrieval_metrics(
            self.z_audio, self.z_midi, k_values
        )
        results['global'] = global_metrics
        print(f"  Recall@1: {global_metrics['recall@1']:.4f}")
        print(f"  MRR: {global_metrics['mrr']:.4f}")
        print(f"  Random baseline: {global_metrics['random_baseline']:.6f}")

        # 2. Negative controls
        print("\n[2] Negative controls...")
        for ctrl_name, ctrl in NEGATIVE_CONTROLS.items():
            print(f"\n  {ctrl.name}: {ctrl.description}")
            mask = ctrl.create_mask(n_samples, self.metadata)
            n_valid = mask.sum() - n_samples  # Exclude diagonal

            if n_valid > 0:
                metrics = compute_retrieval_metrics(
                    self.z_audio, self.z_midi, k_values, mask
                )
                results[ctrl_name] = metrics
                print(f"    Valid candidates: {n_valid / n_samples:.1f} per sample")
                print(f"    Recall@1: {metrics['recall@1']:.4f}")
            else:
                print(f"    Skipped (no valid candidates)")

        # 3. Oracle test (if synthesized audio available)
        if self.z_audio_synth is not None:
            print("\n[3] Oracle test (synthesized audio)...")
            oracle_metrics = evaluate_oracle(self.z_audio_synth, self.z_midi)
            results['oracle'] = oracle_metrics
            print(f"  Oracle Recall@1: {oracle_metrics['oracle_recall@1']:.4f}")

            if oracle_metrics['oracle_recall@1'] > 0.9:
                print("  ✓ PASS: Oracle > 90%")
            else:
                print("  ✗ FAIL: Oracle < 90%")

        # 4. Bootstrap CIs for main metrics
        if n_bootstrap > 0:
            print(f"\n[4] Bootstrap confidence intervals (n={n_bootstrap})...")

            def metric_fn(za, zm):
                return compute_retrieval_metrics(za, zm, [1, 5, 10])

            ci_results = bootstrap_ci(
                metric_fn, self.z_audio, self.z_midi,
                n_bootstrap=n_bootstrap, ci=ci
            )
            results['bootstrap_ci'] = {
                k: {'mean': v[0], 'ci_low': v[1], 'ci_high': v[2]}
                for k, v in ci_results.items()
            }

            print(f"  Recall@1: {ci_results['recall@1'][0]:.4f} "
                  f"[{ci_results['recall@1'][1]:.4f}, {ci_results['recall@1'][2]:.4f}]")

        # 5. GO/NO-GO summary
        print("\n" + "=" * 60)
        print("GO/NO-GO CRITERIA")
        print("=" * 60)

        go_criteria = []

        # Criterion 1: Better than random
        recall1 = global_metrics['recall@1']
        random_baseline = global_metrics['random_baseline']
        ratio_vs_random = recall1 / random_baseline if random_baseline > 0 else 0

        if ratio_vs_random > 10:
            print(f"✓ Recall@1 > 10x random: {ratio_vs_random:.1f}x")
            go_criteria.append(True)
        else:
            print(f"✗ Recall@1 not > 10x random: {ratio_vs_random:.1f}x")
            go_criteria.append(False)

        # Criterion 2: Oracle (if available)
        if 'oracle' in results:
            oracle_recall = results['oracle']['oracle_recall@1']
            if oracle_recall > 0.9:
                print(f"✓ Oracle > 90%: {oracle_recall:.1%}")
                go_criteria.append(True)
            else:
                print(f"✗ Oracle < 90%: {oracle_recall:.1%}")
                go_criteria.append(False)

        # Final verdict
        results['go_criteria'] = {
            'vs_random': ratio_vs_random > 10,
            'oracle_pass': results.get('oracle', {}).get('oracle_recall@1', 0) > 0.9,
            'all_pass': all(go_criteria),
        }

        if all(go_criteria):
            print("\n✓ GATE 0 PASS: Evaluation framework validated")
        else:
            print("\n✗ GATE 0 FAIL: Criteria not met")

        return results

    def generate_report(
        self,
        results: Dict,
        output_path: Path,
    ) -> None:
        """Generate markdown report."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / 'GATE0_REPORT.md'

        with open(report_path, 'w') as f:
            f.write("# GATE 0: Evaluation Framework Report\n\n")
            f.write(f"**Samples**: {len(self.z_audio)}\n")
            f.write(f"**Embedding dim**: {self.z_audio.shape[1]}\n\n")

            f.write("---\n\n")

            # Global metrics
            f.write("## 1. Global Retrieval (Audio → MIDI)\n\n")
            global_m = results.get('global', {})
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in global_m.items():
                f.write(f"| {k} | {v:.4f} |\n")
            f.write("\n")

            # Negative controls
            f.write("## 2. Negative Controls\n\n")
            f.write("| Control | Recall@1 | MRR | Description |\n")
            f.write("|---------|----------|-----|-------------|\n")
            for ctrl_name, ctrl in NEGATIVE_CONTROLS.items():
                if ctrl_name in results:
                    m = results[ctrl_name]
                    f.write(f"| {ctrl.name} | {m['recall@1']:.4f} | {m['mrr']:.4f} | {ctrl.description} |\n")
            f.write("\n")

            # Oracle
            if 'oracle' in results:
                f.write("## 3. Oracle Test\n\n")
                oracle = results['oracle']
                f.write(f"- Recall@1: {oracle['oracle_recall@1']:.4f}\n")
                f.write(f"- MRR: {oracle['oracle_mrr']:.4f}\n\n")

            # Bootstrap CI
            if 'bootstrap_ci' in results:
                f.write("## 4. Bootstrap Confidence Intervals (95%)\n\n")
                f.write("| Metric | Mean | CI Low | CI High |\n")
                f.write("|--------|------|--------|--------|\n")
                for k, v in results['bootstrap_ci'].items():
                    f.write(f"| {k} | {v['mean']:.4f} | {v['ci_low']:.4f} | {v['ci_high']:.4f} |\n")
                f.write("\n")

            # GO/NO-GO
            f.write("## GO/NO-GO Summary\n\n")
            criteria = results.get('go_criteria', {})
            f.write(f"- Better than 10x random: {'✓' if criteria.get('vs_random') else '✗'}\n")
            f.write(f"- Oracle > 90%: {'✓' if criteria.get('oracle_pass') else '✗ (or N/A)'}\n")
            f.write(f"\n**VERDICT**: {'**GO**' if criteria.get('all_pass') else '**NO-GO**'}\n")

        print(f"\nReport saved to {report_path}")

        # Also save JSON
        json_path = output_path / 'gate0_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='GATE 0: Evaluation Framework')
    parser.add_argument('--embeddings', type=Path, required=True,
                        help='Path to embeddings NPZ (z_audio, z_midi, metadata)')
    parser.add_argument('--output', type=Path, default=Path('data/evaluations/maestro_gate0'),
                        help='Output directory')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                        help='Number of bootstrap samples (0 to skip)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print("GATE 0: Evaluation Framework for MAESTRO")
    print("=" * 60)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    data = np.load(args.embeddings, allow_pickle=True)

    z_audio = data['z_audio']
    z_midi = data['z_midi']
    metadata = data['metadata'].item() if 'metadata' in data else {}

    # Optional: synthesized audio embeddings for oracle
    z_audio_synth = data.get('z_audio_synth', None)

    print(f"  Audio embeddings: {z_audio.shape}")
    print(f"  MIDI embeddings: {z_midi.shape}")
    if z_audio_synth is not None:
        print(f"  Oracle embeddings: {z_audio_synth.shape}")

    # Run evaluation
    np.random.seed(args.seed)

    evaluator = MAESTROEvaluator(
        z_audio=z_audio,
        z_midi=z_midi,
        metadata=metadata,
        z_audio_synth=z_audio_synth,
    )

    results = evaluator.run_all_evaluations(
        n_bootstrap=args.n_bootstrap,
    )

    # Generate report
    evaluator.generate_report(results, args.output)

    print("\n✓ GATE 0 evaluation complete!")


if __name__ == '__main__':
    main()
