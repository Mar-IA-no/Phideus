#!/usr/bin/env python3
"""
Comparative evaluation of Gate 3 DANN checkpoints.

Runs evaluate_structured_pool.py on all candidate checkpoints with identical
parameters to produce a fair apples-to-apples comparison.

Usage:
    python experiments/bias_control/compare_gate3_checkpoints.py [--device cuda]
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All checkpoints to evaluate
CHECKPOINTS = {
    "gate2_ep45": {
        "path": "data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt",
        "description": "Gate 2 baseline (no DANN)",
    },
    "runA_best_ep7": {
        "path": "data/bias_control_medium/training_outputs/gate3/best_model.pt",
        "description": "Run A (sin norm), best recall, epoch 7",
    },
    "runB_ep5": {
        "path": "data/bias_control_medium/training_outputs/gate3_norm/checkpoint_epoch5.pt",
        "description": "Run B (F.normalize), epoch 5, lambda~0.17",
    },
    "runB_ep10": {
        "path": "data/bias_control_medium/training_outputs/gate3_norm/checkpoint_epoch10.pt",
        "description": "Run B (F.normalize), epoch 10, lambda~0.33",
    },
    "runC_best_ep4": {
        "path": "data/bias_control_medium/training_outputs/gate3_c/best_model.pt",
        "description": "Run C (optimized), best recall, epoch 4",
    },
    "runC_ep13": {
        "path": "data/bias_control_medium/training_outputs/gate3_c/checkpoint_epoch13.pt",
        "description": "Run C (optimized), epoch 13, lambda=0.80",
    },
}

# Fixed evaluation parameters (identical for all)
EVAL_PARAMS = {
    "pool_size": 256,
    "n_hard": 64,
    "n_semi_hard": 32,
    "n_queries": 500,
    "seed": 42,
    "batch_size": 64,
    "num_workers": 8,
}


def run_evaluation(name: str, info: dict, output_dir: Path, device: str, maestro_dir: str) -> dict:
    """Run evaluate_structured_pool.py on a single checkpoint."""
    checkpoint_path = info["path"]
    output_json = output_dir / f"{name}.json"

    if not Path(checkpoint_path).exists():
        logger.warning(f"SKIP {name}: checkpoint not found at {checkpoint_path}")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {name}")
    logger.info(f"  {info['description']}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"{'='*60}")

    cmd = [
        sys.executable,
        "experiments/bias_control/evaluate_structured_pool.py",
        "--model", checkpoint_path,
        "--maestro-dir", maestro_dir,
        "--output", str(output_json),
        "--pool-size", str(EVAL_PARAMS["pool_size"]),
        "--n-hard", str(EVAL_PARAMS["n_hard"]),
        "--n-semi-hard", str(EVAL_PARAMS["n_semi_hard"]),
        "--n-queries", str(EVAL_PARAMS["n_queries"]),
        "--seed", str(EVAL_PARAMS["seed"]),
        "--batch-size", str(EVAL_PARAMS["batch_size"]),
        "--num-workers", str(EVAL_PARAMS["num_workers"]),
        "--device", device,
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error(f"FAILED: {name} (return code {result.returncode})")
        return None

    if output_json.exists():
        with open(output_json) as f:
            return json.load(f)
    return None


def print_comparison_table(all_results: dict):
    """Print a formatted comparison table."""

    print("\n" + "=" * 120)
    print("COMPARATIVE EVALUATION - Gate 3 DANN Checkpoints")
    print(f"Pool: {EVAL_PARAMS['pool_size']} candidates (64 hard + 32 semi-hard + 159 random + 1 positive)")
    print(f"Queries: {EVAL_PARAMS['n_queries']}, Seed: {EVAL_PARAMS['seed']}")
    print("=" * 120)

    # Header
    header = f"{'Checkpoint':<20} | {'R@1 a2m':>8} {'R@5':>8} {'R@10':>8} | {'R@1 m2a':>8} {'R@5':>8} {'R@10':>8} | {'MRR a2m':>8} {'MRR m2a':>8} | {'vs Hard':>8} {'vs Rand':>8} | {'Decision':>8}"
    print(header)
    print("-" * 120)

    for name, results in all_results.items():
        if results is None:
            print(f"{name:<20} | {'FAILED':>8}")
            continue

        a2m = results.get('a2m', {})
        m2a = results.get('m2a', {})
        hn = results.get('hard_negative_analysis', {})

        row = (
            f"{name:<20} | "
            f"{a2m.get('mean_recall@1', 0):>7.1%} {a2m.get('mean_recall@5', 0):>7.1%} {a2m.get('mean_recall@10', 0):>7.1%} | "
            f"{m2a.get('mean_recall@1', 0):>7.1%} {m2a.get('mean_recall@5', 0):>7.1%} {m2a.get('mean_recall@10', 0):>7.1%} | "
            f"{a2m.get('mean_reciprocal_rank', 0):>8.3f} {m2a.get('mean_reciprocal_rank', 0):>8.3f} | "
            f"{hn.get('accuracy_vs_same_piece', 0):>7.1%} {hn.get('accuracy_vs_random', 0):>7.1%} | "
            f"{results.get('decision', '?'):>8}"
        )
        print(row)

    print("-" * 120)

    # vs_random multipliers
    print(f"\n{'Improvement over random (x):'}")
    header2 = f"{'Checkpoint':<20} | {'R@1 a2m':>8} {'R@5 a2m':>8} {'R@10 a2m':>8} | {'R@1 m2a':>8} {'R@5 m2a':>8} {'R@10 m2a':>8}"
    print(header2)
    print("-" * 90)

    for name, results in all_results.items():
        if results is None:
            continue
        a2m = results.get('a2m', {})
        m2a = results.get('m2a', {})
        row = (
            f"{name:<20} | "
            f"{a2m.get('vs_random@1', 0):>7.1f}x {a2m.get('vs_random@5', 0):>7.1f}x {a2m.get('vs_random@10', 0):>7.1f}x | "
            f"{m2a.get('vs_random@1', 0):>7.1f}x {m2a.get('vs_random@5', 0):>7.1f}x {m2a.get('vs_random@10', 0):>7.1f}x"
        )
        print(row)

    print("-" * 90)

    # Mean rank comparison
    print(f"\n{'Mean / Median Rank (lower is better):'}")
    header3 = f"{'Checkpoint':<20} | {'Mean a2m':>10} {'Med a2m':>10} | {'Mean m2a':>10} {'Med m2a':>10}"
    print(header3)
    print("-" * 70)

    for name, results in all_results.items():
        if results is None:
            continue
        a2m = results.get('a2m', {})
        m2a = results.get('m2a', {})
        row = (
            f"{name:<20} | "
            f"{a2m.get('mean_rank', 0):>9.1f} {a2m.get('median_rank', 0):>9.1f} | "
            f"{m2a.get('mean_rank', 0):>9.1f} {m2a.get('median_rank', 0):>9.1f}"
        )
        print(row)

    print("-" * 70)


def save_summary(all_results: dict, output_dir: Path):
    """Save complete results and markdown summary."""
    # Save raw JSON
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "eval_params": EVAL_PARAMS,
            "checkpoints": {k: v for k, v in CHECKPOINTS.items()},
            "results": all_results,
        }, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Generate markdown table
    md_path = output_dir / "COMPARISON_GATE3.md"
    with open(md_path, 'w') as f:
        f.write("# Gate 3 DANN - Comparative Evaluation\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Pool**: {EVAL_PARAMS['pool_size']} candidates (64 hard + 32 semi-hard + 159 random + 1 positive)\n")
        f.write(f"**Queries**: {EVAL_PARAMS['n_queries']}, **Seed**: {EVAL_PARAMS['seed']}\n\n")

        f.write("## Retrieval Metrics (Structured Pool)\n\n")
        f.write("| Checkpoint | Description | R@1 a2m | R@5 a2m | R@10 a2m | R@1 m2a | R@5 m2a | R@10 m2a | MRR a2m | MRR m2a |\n")
        f.write("|-----------|-------------|---------|---------|----------|---------|---------|----------|---------|----------|\n")
        for name, results in all_results.items():
            if results is None:
                continue
            a2m = results.get('a2m', {})
            m2a = results.get('m2a', {})
            desc = CHECKPOINTS[name]["description"]
            f.write(
                f"| {name} | {desc} | "
                f"{a2m.get('mean_recall@1', 0):.1%} | {a2m.get('mean_recall@5', 0):.1%} | {a2m.get('mean_recall@10', 0):.1%} | "
                f"{m2a.get('mean_recall@1', 0):.1%} | {m2a.get('mean_recall@5', 0):.1%} | {m2a.get('mean_recall@10', 0):.1%} | "
                f"{a2m.get('mean_reciprocal_rank', 0):.3f} | {m2a.get('mean_reciprocal_rank', 0):.3f} |\n"
            )

        f.write("\n## Improvement over Random\n\n")
        f.write("| Checkpoint | R@1 a2m (x) | R@5 a2m (x) | R@10 a2m (x) | R@1 m2a (x) | R@5 m2a (x) | R@10 m2a (x) |\n")
        f.write("|-----------|-------------|-------------|-------------|-------------|-------------|-------------|\n")
        for name, results in all_results.items():
            if results is None:
                continue
            a2m = results.get('a2m', {})
            m2a = results.get('m2a', {})
            f.write(
                f"| {name} | "
                f"{a2m.get('vs_random@1', 0):.1f}x | {a2m.get('vs_random@5', 0):.1f}x | {a2m.get('vs_random@10', 0):.1f}x | "
                f"{m2a.get('vs_random@1', 0):.1f}x | {m2a.get('vs_random@5', 0):.1f}x | {m2a.get('vs_random@10', 0):.1f}x |\n"
            )

        f.write("\n## Hard Negative Analysis\n\n")
        f.write("| Checkpoint | vs Same-Piece-Diff-Time | vs Random | Decision |\n")
        f.write("|-----------|------------------------|-----------|----------|\n")
        for name, results in all_results.items():
            if results is None:
                continue
            hn = results.get('hard_negative_analysis', {})
            f.write(
                f"| {name} | "
                f"{hn.get('accuracy_vs_same_piece', 0):.1%} | "
                f"{hn.get('accuracy_vs_random', 0):.1%} | "
                f"{results.get('decision', '?')} |\n"
            )

        f.write("\n## Mean / Median Rank (lower is better, out of 256)\n\n")
        f.write("| Checkpoint | Mean a2m | Median a2m | Mean m2a | Median m2a |\n")
        f.write("|-----------|----------|-----------|----------|------------|\n")
        for name, results in all_results.items():
            if results is None:
                continue
            a2m = results.get('a2m', {})
            m2a = results.get('m2a', {})
            f.write(
                f"| {name} | "
                f"{a2m.get('mean_rank', 0):.1f} | {a2m.get('median_rank', 0):.1f} | "
                f"{m2a.get('mean_rank', 0):.1f} | {m2a.get('median_rank', 0):.1f} |\n"
            )

        f.write("\n## Configuration Notes\n\n")
        f.write("- **Run A**: No F.normalize before domain head. Linear lambda 0→1.\n")
        f.write("- **Run B**: F.normalize before domain head. Linear lambda 0→1.\n")
        f.write("- **Run C**: F.normalize + warmup_ramp_cap schedule (warmup=2000, ramp=6000, lambda_max=0.8). 3 LR groups.\n")
        f.write("- **All runs** start from Gate 2 checkpoint (epoch 45).\n")
        f.write("- **Evaluation**: Identical structured pool (256 candidates, 500 queries, seed 42) for all checkpoints.\n")

    logger.info(f"Markdown report saved to {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Gate 3 DANN checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--output-dir', type=str,
                        default='data/bias_control_medium/evaluations/gate3_comparison')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, info in CHECKPOINTS.items():
        results = run_evaluation(name, info, output_dir, args.device, args.maestro_dir)
        all_results[name] = results

    print_comparison_table(all_results)
    save_summary(all_results, output_dir)

    logger.info(f"\nAll results saved to {output_dir}/")


if __name__ == '__main__':
    main()
