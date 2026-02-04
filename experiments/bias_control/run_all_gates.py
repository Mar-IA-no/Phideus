#!/usr/bin/env python3
"""
Run All Gates: Complete Cross-Modal Learning Pipeline

Orchestrates execution of all gates (0-4) with proper sequencing
and result aggregation.

Usage:
    python experiments/bias_control/run_all_gates.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/bias_control

    # Run specific gates
    python experiments/bias_control/run_all_gates.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/bias_control \
        --gates 0 1 2

    # Skip to gate 3 with existing checkpoint
    python experiments/bias_control/run_all_gates.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/bias_control \
        --gates 3 4 \
        --gate2-checkpoint data/training_outputs/bias_control/gate2/best_model.pt
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_gate0(maestro_dir: Path, output_dir: Path, config: Dict) -> Dict:
    """Run Gate 0: Data Integrity."""
    from gate0_data_integrity import run_gate0 as _run_gate0

    logger.info("=" * 60)
    logger.info("GATE 0: DATA INTEGRITY")
    logger.info("=" * 60)

    return _run_gate0(
        maestro_dir=maestro_dir,
        output_dir=output_dir / "segments",
        segment_len=config.get('segment_len', 8.0),
        hop=config.get('hop', 2.0),
        max_alignment_check=config.get('max_alignment_check', 100),
    )


def run_gate1(maestro_dir: Path, output_dir: Path, config: Dict) -> Dict:
    """Run Gate 1: Intra-Modal Baselines."""
    from gate1_intra_modal import run_gate1 as _run_gate1

    logger.info("=" * 60)
    logger.info("GATE 1: INTRA-MODAL BASELINES")
    logger.info("=" * 60)

    return _run_gate1(
        maestro_dir=maestro_dir,
        output_dir=output_dir / "evaluations" / "gate1",
        segment_len=config.get('segment_len', 8.0),
        hop=config.get('hop', 2.0),
        n_samples=config.get('n_samples', 500),
        use_mert_lite=config.get('use_mert_lite', True),
        device=config.get('device'),
    )


def run_gate2(maestro_dir: Path, output_dir: Path, config: Dict) -> Dict:
    """Run Gate 2: Cross-Modal Foundation."""
    from gate2_foundation import run_gate2 as _run_gate2

    logger.info("=" * 60)
    logger.info("GATE 2: CROSS-MODAL FOUNDATION")
    logger.info("=" * 60)

    return _run_gate2(
        maestro_dir=maestro_dir,
        output_dir=output_dir / "training_outputs" / "gate2",
        segment_len=config.get('segment_len', 8.0),
        hop=config.get('hop', 2.0),
        batch_size=config.get('batch_size', 64),
        num_workers=config.get('num_workers', 8),
        epochs=config.get('epochs_gate2', 100),
        lr_projection=config.get('lr_projection', 1e-3),
        lr_midi_encoder=config.get('lr_midi_encoder', 1e-4),
        use_mert_lite=config.get('use_mert_lite', True),
        device=config.get('device'),
    )


def run_gate2_5(maestro_dir: Path, output_dir: Path, checkpoint: Path, config: Dict) -> Dict:
    """Run Gate 2.5: Embedding Analysis."""
    from gate2_5_embedding_analysis import run_analysis

    logger.info("=" * 60)
    logger.info("GATE 2.5: EMBEDDING ANALYSIS")
    logger.info("=" * 60)

    return run_analysis(
        model_path=checkpoint,
        maestro_dir=maestro_dir,
        output_dir=output_dir / "evaluations" / "gate2_5",
        segment_len=config.get('segment_len', 8.0),
        hop=config.get('hop', 2.0),
        max_samples=config.get('max_samples_analysis', 1000),
        device=config.get('device'),
    )


def run_gate3(
    maestro_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    gate2_recall: float,
    config: Dict
) -> Dict:
    """Run Gate 3: DANN."""
    from gate3_dann import run_gate3 as _run_gate3

    logger.info("=" * 60)
    logger.info("GATE 3: DANN TRAINING")
    logger.info("=" * 60)

    return _run_gate3(
        maestro_dir=maestro_dir,
        checkpoint_path=checkpoint,
        output_dir=output_dir / "training_outputs" / "gate3",
        segment_len=config.get('segment_len', 8.0),
        hop=config.get('hop', 2.0),
        batch_size=config.get('batch_size', 64),
        num_workers=config.get('num_workers', 8),
        epochs=config.get('epochs_gate3', 50),
        dann_weight=config.get('dann_weight', 0.01),
        device=config.get('device'),
        gate2_recall=gate2_recall,
    )


def run_gate4(
    maestro_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    gate3_recall: float,
    config: Dict
) -> Dict:
    """Run Gate 4: Ratio Auxiliary."""
    from gate4_ratio_auxiliary import run_gate4 as _run_gate4

    logger.info("=" * 60)
    logger.info("GATE 4: RATIO AUXILIARY")
    logger.info("=" * 60)

    return _run_gate4(
        maestro_dir=maestro_dir,
        checkpoint_path=checkpoint,
        output_dir=output_dir / "training_outputs" / "gate4",
        segment_len=config.get('segment_len', 8.0),
        hop=config.get('hop', 2.0),
        batch_size=config.get('batch_size', 64),
        num_workers=config.get('num_workers', 8),
        epochs=config.get('epochs_gate4', 30),
        ratio_weight=config.get('ratio_weight', 0.05),
        device=config.get('device'),
        gate3_recall=gate3_recall,
    )


def run_pipeline(
    maestro_dir: Path,
    output_dir: Path,
    gates: List[int],
    config: Dict,
    gate2_checkpoint: Optional[Path] = None,
    gate3_checkpoint: Optional[Path] = None,
) -> Dict:
    """
    Run the complete pipeline.

    Args:
        maestro_dir: MAESTRO dataset directory
        output_dir: Output directory for all results
        gates: List of gates to run [0, 1, 2, 3, 4]
        config: Configuration dict
        gate2_checkpoint: Pre-existing Gate 2 checkpoint (to skip Gate 2)
        gate3_checkpoint: Pre-existing Gate 3 checkpoint (to skip Gate 3)

    Returns:
        Complete results dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'maestro_dir': str(maestro_dir),
        'output_dir': str(output_dir),
        'config': config,
        'gates': {},
    }

    gate2_recall = 0.0
    gate3_recall = 0.0

    # Gate 0
    if 0 in gates:
        results['gates'][0] = run_gate0(maestro_dir, output_dir, config)
        if results['gates'][0]['decision'] == 'NO-GO':
            logger.error("Gate 0 FAILED - stopping pipeline")
            results['final_status'] = 'FAILED_GATE_0'
            return results

    # Gate 1
    if 1 in gates:
        results['gates'][1] = run_gate1(maestro_dir, output_dir, config)
        if results['gates'][1]['decision'] == 'NO-GO':
            logger.warning("Gate 1 FAILED - continuing but results may be unreliable")

    # Gate 2
    if 2 in gates:
        results['gates'][2] = run_gate2(maestro_dir, output_dir, config)
        gate2_checkpoint = output_dir / "training_outputs" / "gate2" / "best_model.pt"

        if results['gates'][2]['decision'] == 'NO-GO':
            logger.error("Gate 2 FAILED - stopping pipeline")
            results['final_status'] = 'FAILED_GATE_2'
            return results

        gate2_recall = (
            results['gates'][2]['retrieval']['a2m_recall@10'] +
            results['gates'][2]['retrieval']['m2a_recall@10']
        ) / 2

        # Check for STRONG GO (can skip Gate 3)
        if results['gates'][2].get('skip_gate3', False):
            logger.info("STRONG GO from Gate 2 - can skip Gate 3")
            if 3 in gates:
                gates = [g for g in gates if g != 3]
                gate3_checkpoint = gate2_checkpoint
                gate3_recall = gate2_recall

    # Gate 2.5 (diagnostic)
    if '2.5' in [str(g) for g in gates] or (2 in gates and gate2_checkpoint):
        checkpoint = gate2_checkpoint or Path(config.get('gate2_checkpoint', ''))
        if checkpoint and checkpoint.exists():
            results['gates']['2.5'] = run_gate2_5(maestro_dir, output_dir, checkpoint, config)

    # Gate 3
    if 3 in gates and gate2_checkpoint:
        results['gates'][3] = run_gate3(
            maestro_dir, output_dir, gate2_checkpoint, gate2_recall, config
        )
        gate3_checkpoint = output_dir / "training_outputs" / "gate3" / "best_model.pt"

        if results['gates'][3]['decision'] == 'NO-GO':
            logger.warning("Gate 3 FAILED - continuing to Gate 4 anyway")

        gate3_recall = (
            results['gates'][3]['retrieval']['a2m_recall@10'] +
            results['gates'][3]['retrieval']['m2a_recall@10']
        ) / 2

    # Gate 4
    if 4 in gates:
        checkpoint = gate3_checkpoint or gate2_checkpoint
        if checkpoint and checkpoint.exists():
            results['gates'][4] = run_gate4(
                maestro_dir, output_dir, checkpoint,
                gate3_recall or gate2_recall, config
            )

    # Final status
    failed_gates = [
        g for g in results['gates'].keys()
        if isinstance(g, int) and results['gates'][g].get('decision') == 'NO-GO'
    ]

    if failed_gates:
        results['final_status'] = f'FAILED_GATES_{failed_gates}'
    else:
        results['final_status'] = 'SUCCESS'

    return results


def print_summary(results: Dict):
    """Print pipeline summary."""
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    for gate_id, gate_results in results['gates'].items():
        decision = gate_results.get('decision', 'N/A')
        name = gate_results.get('name', f'Gate {gate_id}')

        status_icon = '✓' if decision in ['GO', 'STRONG GO'] else '✗' if decision == 'NO-GO' else '?'
        print(f"\n{status_icon} Gate {gate_id}: {name}")
        print(f"  Decision: {decision}")

        # Print key metrics
        if 'retrieval' in gate_results:
            r = gate_results['retrieval']
            print(f"  Recall@10: A→M={r.get('a2m_recall@10', 0):.1%}, M→A={r.get('m2a_recall@10', 0):.1%}")

        if 'domain_adversarial' in gate_results:
            d = gate_results['domain_adversarial']
            print(f"  Domain Accuracy: {d.get('domain_accuracy', 0):.1%}")

    print(f"\n{'=' * 70}")
    print(f"FINAL STATUS: {results['final_status']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run Cross-Modal Learning Pipeline")
    parser.add_argument(
        '--maestro-dir',
        type=str,
        required=True,
        help='Path to MAESTRO dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/bias_control',
        help='Output directory'
    )
    parser.add_argument(
        '--gates',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='Gates to run (default: all)'
    )
    parser.add_argument(
        '--gate2-checkpoint',
        type=str,
        default=None,
        help='Existing Gate 2 checkpoint (skip Gate 2)'
    )
    parser.add_argument(
        '--gate3-checkpoint',
        type=str,
        default=None,
        help='Existing Gate 3 checkpoint (skip Gate 3)'
    )

    # Config options
    parser.add_argument('--segment-len', type=float, default=8.0)
    parser.add_argument('--hop', type=float, default=2.0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs-gate2', type=int, default=100)
    parser.add_argument('--epochs-gate3', type=int, default=50)
    parser.add_argument('--epochs-gate4', type=int, default=30)
    parser.add_argument('--use-full-mert', action='store_true')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    maestro_dir = Path(args.maestro_dir)
    output_dir = Path(args.output)

    if not maestro_dir.exists():
        logger.error(f"MAESTRO directory not found: {maestro_dir}")
        sys.exit(1)

    config = {
        'segment_len': args.segment_len,
        'hop': args.hop,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epochs_gate2': args.epochs_gate2,
        'epochs_gate3': args.epochs_gate3,
        'epochs_gate4': args.epochs_gate4,
        'use_mert_lite': not args.use_full_mert,
        'device': args.device,
    }

    # Run pipeline
    results = run_pipeline(
        maestro_dir=maestro_dir,
        output_dir=output_dir,
        gates=args.gates,
        config=config,
        gate2_checkpoint=Path(args.gate2_checkpoint) if args.gate2_checkpoint else None,
        gate3_checkpoint=Path(args.gate3_checkpoint) if args.gate3_checkpoint else None,
    )

    # Save results
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print_summary(results)

    print(f"\nResults saved to: {results_path}")

    return 0 if results['final_status'] == 'SUCCESS' else 1


if __name__ == '__main__':
    sys.exit(main())
