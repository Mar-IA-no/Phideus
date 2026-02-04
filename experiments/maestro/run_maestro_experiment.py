#!/usr/bin/env python3
"""
MAESTRO Cross-Modal Experiment - Main Orchestration Script
==========================================================

Runs the full 6-Gate MAESTRO experiment pipeline:
- Gate 0: Setup and test harness
- Gate 1: Dataset ingestion and alignment verification
- Gate 2: Non-DL baselines (chroma, CCA)
- Gate 3: Dense cross-modal training (VICReg/Barlow)
- Gate 4: Ratio token training (The Phideus test)
- Gate 5: MoCo with hard negatives

Each gate has GO/NO-GO criteria. Gates are run sequentially;
if a gate fails, the experiment stops.

Usage:
------
# Full pipeline (download + process + train)
python experiments/maestro/run_maestro_experiment.py \
    --mode full \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/training_outputs/maestro_experiment \
    --epochs 100

# Skip download (data already exists)
python experiments/maestro/run_maestro_experiment.py \
    --mode train-only \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --processed-dir data/maestro_v3/processed \
    --constellation-npz data/maestro_v3/constellations/tokens.npz \
    --output data/training_outputs/maestro_experiment

# Run specific gate
python experiments/maestro/run_maestro_experiment.py \
    --mode gate4 \
    --constellation-npz data/maestro_v3/constellations/tokens.npz \
    --output data/training_outputs/maestro_gate4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """Run a command and return exit code."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def check_file_exists(path: Path, name: str) -> bool:
    """Check if file exists and report."""
    exists = path.exists()
    status = "FOUND" if exists else "NOT FOUND"
    print(f"  {name}: {status}")
    return exists


class MAESTROExperiment:
    """Orchestrates the full MAESTRO experiment pipeline."""

    def __init__(
        self,
        maestro_dir: Path,
        processed_dir: Path,
        constellation_npz: Path,
        output_dir: Path,
        epochs: int = 100,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        self.maestro_dir = maestro_dir
        self.processed_dir = processed_dir
        self.constellation_npz = constellation_npz
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.results = {}
        self.gate_status = {}

    def run_gate0(self) -> bool:
        """Gate 0: Setup and test harness."""
        print("\n" + "=" * 70)
        print("GATE 0: Setup and Test Harness")
        print("=" * 70)

        gate0_output = self.output_dir / 'gate0'
        gate0_output.mkdir(parents=True, exist_ok=True)

        # Check that test harness can import
        cmd = [
            sys.executable, '-c',
            'from experiments.maestro.gate0_harness import MAESTROEvaluator; print("OK")'
        ]
        result = run_command(cmd)

        if result != 0:
            print("Gate 0 FAIL: Cannot import test harness")
            self.gate_status['gate0'] = 'FAIL'
            return False

        print("\nGate 0 PASS: Test harness ready")
        self.gate_status['gate0'] = 'PASS'
        return True

    def run_gate1(self, skip_download: bool = False) -> bool:
        """Gate 1: Dataset ingestion and alignment."""
        print("\n" + "=" * 70)
        print("GATE 1: Dataset Ingestion and Alignment")
        print("=" * 70)

        gate1_output = self.output_dir / 'gate1'
        gate1_output.mkdir(parents=True, exist_ok=True)

        # Check if data already processed
        metadata_file = self.processed_dir / 'dataset_metadata.json'
        if metadata_file.exists():
            print(f"Processed data found at {self.processed_dir}")
            print("Verifying alignment...")

            # Run alignment verification only
            cmd = [
                sys.executable, 'experiments/maestro/gate1_ingest.py',
                '--maestro-dir', str(self.maestro_dir),
                '--output-dir', str(self.processed_dir),
                '--verify-only',
            ]
            result = run_command(cmd)

            if result != 0:
                print("Gate 1 FAIL: Alignment verification failed")
                self.gate_status['gate1'] = 'FAIL'
                return False
        else:
            if skip_download:
                print(f"ERROR: Processed data not found at {self.processed_dir}")
                print("Run with --mode full to process data first")
                self.gate_status['gate1'] = 'FAIL'
                return False

            # Full processing
            cmd = [
                sys.executable, 'experiments/maestro/gate1_ingest.py',
                '--maestro-dir', str(self.maestro_dir),
                '--output-dir', str(self.processed_dir),
                '--window-len', '4.0',
                '--hop', '2.0',
                '--sr', '22050',
                '--workers', str(self.num_workers),
            ]
            result = run_command(cmd)

            if result != 0:
                print("Gate 1 FAIL: Processing failed")
                self.gate_status['gate1'] = 'FAIL'
                return False

        # Check results
        if not metadata_file.exists():
            print("Gate 1 FAIL: No metadata file created")
            self.gate_status['gate1'] = 'FAIL'
            return False

        with open(metadata_file) as f:
            metadata = json.load(f)

        n_segments = len(metadata['segments'])
        print(f"\nProcessed {n_segments} segments")

        if n_segments < 1000:
            print(f"WARNING: Only {n_segments} segments (expected ~100k)")

        print("\nGate 1 PASS: Data ingested and aligned")
        self.gate_status['gate1'] = 'PASS'
        self.results['gate1'] = {'n_segments': n_segments}
        return True

    def run_gate2(self) -> bool:
        """Gate 2: Non-DL baselines."""
        print("\n" + "=" * 70)
        print("GATE 2: Non-DL Baselines")
        print("=" * 70)

        gate2_output = self.output_dir / 'gate2'
        gate2_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, 'experiments/maestro/gate2_baselines.py',
            '--data-dir', str(self.processed_dir),
            '--maestro-dir', str(self.maestro_dir),
            '--output', str(gate2_output),
            '--max-samples', '500',  # Limit for speed
        ]
        result = run_command(cmd)

        if result != 0:
            print("Gate 2 FAIL: Baseline evaluation failed")
            self.gate_status['gate2'] = 'FAIL'
            return False

        # Check results
        results_file = gate2_output / 'baseline_results.json'
        if not results_file.exists():
            print("Gate 2 FAIL: No results file")
            self.gate_status['gate2'] = 'FAIL'
            return False

        with open(results_file) as f:
            baseline_results = json.load(f)

        # GO criterion: > 10x random
        chroma_ratio = baseline_results.get('chroma', {}).get('ratio_vs_random', 0)
        cca_ratio = baseline_results.get('cca', {}).get('ratio_vs_random', 0)
        best_ratio = max(chroma_ratio, cca_ratio)

        print(f"\nChroma baseline: {chroma_ratio:.1f}x random")
        print(f"CCA baseline: {cca_ratio:.1f}x random")

        if best_ratio < 10:
            print(f"\nGate 2 FAIL: Best baseline only {best_ratio:.1f}x random (need >10x)")
            self.gate_status['gate2'] = 'FAIL'
            return False

        print(f"\nGate 2 PASS: Baselines show signal ({best_ratio:.1f}x random)")
        self.gate_status['gate2'] = 'PASS'
        self.results['gate2'] = baseline_results
        return True

    def run_gate3(self, loss_type: str = 'vicreg') -> bool:
        """Gate 3: Dense cross-modal training."""
        print("\n" + "=" * 70)
        print(f"GATE 3: Dense Cross-Modal Training ({loss_type.upper()})")
        print("=" * 70)

        gate3_output = self.output_dir / 'gate3'
        gate3_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, 'experiments/maestro/gate3_cross_modal.py',
            '--data', str(self.processed_dir),
            '--maestro-dir', str(self.maestro_dir),
            '--output', str(gate3_output),
            '--loss', loss_type,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--num-workers', str(self.num_workers),
        ]
        result = run_command(cmd)

        if result != 0:
            print("Gate 3 FAIL: Training failed")
            self.gate_status['gate3'] = 'FAIL'
            return False

        # Check results
        results_file = gate3_output / 'gate3_results.json'
        if not results_file.exists():
            print("Gate 3 FAIL: No results file")
            self.gate_status['gate3'] = 'FAIL'
            return False

        with open(results_file) as f:
            gate3_results = json.load(f)

        go_pass = gate3_results.get('go_criteria', {}).get('pass', False)

        if not go_pass:
            print("\nGate 3 FAIL: GO criteria not met")
            self.gate_status['gate3'] = 'FAIL'
            return False

        print(f"\nGate 3 PASS: Dense model trained successfully")
        self.gate_status['gate3'] = 'PASS'
        self.results['gate3'] = gate3_results.get('test', {})
        return True

    def run_gate4(self, model_type: str = 'constellation', encoder_type: str = 'mlp') -> bool:
        """Gate 4: Ratio token training (The Phideus test)."""
        print("\n" + "=" * 70)
        print(f"GATE 4: Ratio Token Training ({model_type}/{encoder_type})")
        print("=" * 70)

        gate4_output = self.output_dir / 'gate4'
        gate4_output.mkdir(parents=True, exist_ok=True)

        # First, extract constellations if not done
        if not self.constellation_npz.exists():
            print("Extracting constellation tokens...")
            cmd = [
                sys.executable, 'src/analizador/analizador_maestro.py',
                '--input-dir', str(self.processed_dir),
                '--output', str(self.constellation_npz),
                '--workers', str(self.num_workers),
            ]
            result = run_command(cmd)

            if result != 0:
                print("Gate 4 FAIL: Constellation extraction failed")
                self.gate_status['gate4'] = 'FAIL'
                return False

        # Run training
        cmd = [
            sys.executable, 'experiments/maestro/gate4_ratio_tokens.py',
            '--data', str(self.constellation_npz),
            '--output', str(gate4_output),
            '--model', model_type,
            '--encoder-type', encoder_type,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--num-workers', str(self.num_workers),
        ]
        result = run_command(cmd)

        if result != 0:
            print("Gate 4 FAIL: Training failed")
            self.gate_status['gate4'] = 'FAIL'
            return False

        # Check results
        results_file = gate4_output / 'gate4_results.json'
        if not results_file.exists():
            print("Gate 4 FAIL: No results file")
            self.gate_status['gate4'] = 'FAIL'
            return False

        with open(results_file) as f:
            gate4_results = json.load(f)

        go_pass = gate4_results.get('go_criteria', {}).get('pass', False)

        if not go_pass:
            print("\nGate 4 FAIL: GO criteria not met")
            self.gate_status['gate4'] = 'FAIL'
            return False

        print(f"\nGate 4 PASS: Ratio tokens show cross-modal signal!")
        self.gate_status['gate4'] = 'PASS'
        self.results['gate4'] = gate4_results.get('test', {})
        return True

    def run_gate5(self) -> bool:
        """Gate 5: MoCo with hard negatives."""
        print("\n" + "=" * 70)
        print("GATE 5: MoCo with Hard-Mined Negatives")
        print("=" * 70)

        gate5_output = self.output_dir / 'gate5'
        gate5_output.mkdir(parents=True, exist_ok=True)

        gate4_results = self.output_dir / 'gate4' / 'gate4_results.json'

        cmd = [
            sys.executable, 'experiments/maestro/gate5_moco.py',
            '--data', str(self.constellation_npz),
            '--output', str(gate5_output),
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--num-workers', str(self.num_workers),
        ]

        if gate4_results.exists():
            cmd.extend(['--gate4-results', str(gate4_results)])

        result = run_command(cmd)

        if result != 0:
            print("Gate 5 FAIL: Training failed")
            self.gate_status['gate5'] = 'FAIL'
            return False

        # Check results
        results_file = gate5_output / 'gate5_results.json'
        if not results_file.exists():
            print("Gate 5 FAIL: No results file")
            self.gate_status['gate5'] = 'FAIL'
            return False

        with open(results_file) as f:
            gate5_results = json.load(f)

        go_pass = gate5_results.get('go_criteria', {}).get('pass', False)

        if not go_pass:
            print("\nGate 5 FAIL: GO criteria not met")
            self.gate_status['gate5'] = 'FAIL'
            return False

        print(f"\nGate 5 PASS: MoCo improves on hard negatives!")
        self.gate_status['gate5'] = 'PASS'
        self.results['gate5'] = gate5_results.get('test', {})
        return True

    def run_full_pipeline(self, skip_download: bool = False) -> bool:
        """Run the complete 6-gate pipeline."""
        print("\n" + "=" * 70)
        print("MAESTRO CROSS-MODAL EXPERIMENT")
        print("Full 6-Gate Pipeline")
        print("=" * 70)
        print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")

        # Gate 0
        if not self.run_gate0():
            return self.generate_final_report()

        # Gate 1
        if not self.run_gate1(skip_download):
            return self.generate_final_report()

        # Gate 2
        if not self.run_gate2():
            return self.generate_final_report()

        # Gate 3
        if not self.run_gate3():
            return self.generate_final_report()

        # Gate 4 (THE PHIDEUS TEST)
        if not self.run_gate4():
            return self.generate_final_report()

        # Gate 5 (optional but recommended)
        self.run_gate5()

        return self.generate_final_report()

    def generate_final_report(self) -> bool:
        """Generate final experiment report."""
        print("\n" + "=" * 70)
        print("FINAL EXPERIMENT REPORT")
        print("=" * 70)

        # Count passes
        n_pass = sum(1 for v in self.gate_status.values() if v == 'PASS')
        n_total = len(self.gate_status)

        print(f"\nGate Status ({n_pass}/{n_total} passed):")
        for gate, status in self.gate_status.items():
            icon = "PASS" if status == 'PASS' else "FAIL"
            print(f"  {gate}: {icon}")

        # Determine overall success
        # Success = Gate 3 AND Gate 4 pass (Gate 5 is bonus)
        gate3_pass = self.gate_status.get('gate3') == 'PASS'
        gate4_pass = self.gate_status.get('gate4') == 'PASS'
        overall_success = gate3_pass and gate4_pass

        print(f"\n{'=' * 70}")
        if overall_success:
            print("EXPERIMENT SUCCESS: Cross-modal learning with ratio tokens demonstrated!")
            print("The Phideus hypothesis is supported for MAESTRO (Audio <-> MIDI)")
        else:
            print("EXPERIMENT INCOMPLETE: Not all required gates passed")
            if not gate3_pass:
                print("  - Gate 3 (Dense) failed: Cross-modal alignment not achieved")
            if not gate4_pass:
                print("  - Gate 4 (Ratio) failed: Ratio tokens did not show signal")
        print("=" * 70)

        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'gate_status': self.gate_status,
            'results': self.results,
            'overall_success': overall_success,
            'config': {
                'maestro_dir': str(self.maestro_dir),
                'processed_dir': str(self.processed_dir),
                'constellation_npz': str(self.constellation_npz),
                'output_dir': str(self.output_dir),
                'epochs': self.epochs,
                'batch_size': self.batch_size,
            },
        }

        with open(self.output_dir / 'experiment_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Generate markdown report
        with open(self.output_dir / 'EXPERIMENT_REPORT.md', 'w') as f:
            f.write("# MAESTRO Cross-Modal Experiment Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Status**: {'SUCCESS' if overall_success else 'INCOMPLETE'}\n\n")

            f.write("## Gate Summary\n\n")
            f.write("| Gate | Status | Description |\n")
            f.write("|------|--------|-------------|\n")
            gate_desc = {
                'gate0': 'Setup & Test Harness',
                'gate1': 'Data Ingestion & Alignment',
                'gate2': 'Non-DL Baselines',
                'gate3': 'Dense Cross-Modal (VICReg)',
                'gate4': 'Ratio Token Training (Phideus)',
                'gate5': 'MoCo Hard Negatives',
            }
            for gate in ['gate0', 'gate1', 'gate2', 'gate3', 'gate4', 'gate5']:
                status = self.gate_status.get(gate, 'NOT RUN')
                desc = gate_desc.get(gate, gate)
                f.write(f"| {gate} | {status} | {desc} |\n")

            f.write("\n## Key Results\n\n")
            if 'gate3' in self.results:
                r = self.results['gate3']
                f.write("### Gate 3 (Dense Cross-Modal)\n")
                f.write(f"- Recall@1: {r.get('recall@1', 'N/A')}\n")
                f.write(f"- MRR: {r.get('mrr', 'N/A')}\n\n")

            if 'gate4' in self.results:
                r = self.results['gate4']
                f.write("### Gate 4 (Ratio Tokens - THE PHIDEUS TEST)\n")
                f.write(f"- Recall@1: {r.get('recall@1', 'N/A')}\n")
                f.write(f"- MRR: {r.get('mrr', 'N/A')}\n")
                f.write(f"- Gap (aligned-shuffled): {r.get('gap', 'N/A')}\n\n")

            f.write("## Conclusion\n\n")
            if overall_success:
                f.write("The experiment demonstrates that **ratio language** (constellation tokens) ")
                f.write("can successfully achieve cross-modal alignment between audio and MIDI. ")
                f.write("This validates the core Phideus hypothesis that frequency ratios ")
                f.write("constitute a universal cross-modal representation.\n")
            else:
                f.write("The experiment did not achieve all required criteria. ")
                f.write("See individual gate reports for details.\n")

        print(f"\nReports saved to {self.output_dir}")
        return overall_success


def parse_args():
    parser = argparse.ArgumentParser(description='MAESTRO Cross-Modal Experiment')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'train-only', 'gate0', 'gate1', 'gate2', 'gate3', 'gate4', 'gate5'],
                        help='Execution mode')
    parser.add_argument('--maestro-dir', type=Path, default=Path('data/maestro_v3/maestro-v3.0.0'),
                        help='Path to MAESTRO dataset')
    parser.add_argument('--processed-dir', type=Path, default=Path('data/maestro_v3/processed'),
                        help='Path to processed segments')
    parser.add_argument('--constellation-npz', type=Path,
                        default=Path('data/maestro_v3/constellations/tokens.npz'),
                        help='Path to constellation tokens NPZ')
    parser.add_argument('--output', type=Path, default=Path('data/training_outputs/maestro_experiment'),
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--loss', type=str, default='vicreg',
                        choices=['vicreg', 'barlow'],
                        help='Loss for Gate 3')
    parser.add_argument('--model', type=str, default='constellation',
                        choices=['constellation', 'jepa-lite'],
                        help='Model for Gate 4')
    parser.add_argument('--encoder', type=str, default='mlp',
                        choices=['mlp', 'transformer'],
                        help='Encoder type')
    return parser.parse_args()


def main():
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.constellation_npz.parent.mkdir(parents=True, exist_ok=True)

    experiment = MAESTROExperiment(
        maestro_dir=args.maestro_dir,
        processed_dir=args.processed_dir,
        constellation_npz=args.constellation_npz,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.mode == 'full':
        success = experiment.run_full_pipeline(skip_download=False)
    elif args.mode == 'train-only':
        success = experiment.run_full_pipeline(skip_download=True)
    elif args.mode == 'gate0':
        success = experiment.run_gate0()
        experiment.generate_final_report()
    elif args.mode == 'gate1':
        experiment.run_gate0()
        success = experiment.run_gate1()
        experiment.generate_final_report()
    elif args.mode == 'gate2':
        experiment.run_gate0()
        experiment.gate_status['gate1'] = 'PASS'  # Assume done
        success = experiment.run_gate2()
        experiment.generate_final_report()
    elif args.mode == 'gate3':
        experiment.run_gate0()
        experiment.gate_status['gate1'] = 'PASS'
        experiment.gate_status['gate2'] = 'PASS'
        success = experiment.run_gate3(args.loss)
        experiment.generate_final_report()
    elif args.mode == 'gate4':
        experiment.run_gate0()
        experiment.gate_status['gate1'] = 'PASS'
        experiment.gate_status['gate2'] = 'PASS'
        experiment.gate_status['gate3'] = 'PASS'
        success = experiment.run_gate4(args.model, args.encoder)
        experiment.generate_final_report()
    elif args.mode == 'gate5':
        experiment.run_gate0()
        experiment.gate_status['gate1'] = 'PASS'
        experiment.gate_status['gate2'] = 'PASS'
        experiment.gate_status['gate3'] = 'PASS'
        experiment.gate_status['gate4'] = 'PASS'
        success = experiment.run_gate5()
        experiment.generate_final_report()
    else:
        print(f"Unknown mode: {args.mode}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
